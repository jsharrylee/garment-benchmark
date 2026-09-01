"""Runtime tracing for the local GarmentCode drafting recipe.

The serialized GarmentCode specification is the *result* of drafting and has
already lost most construction intent.  :class:`RuntimeTraceRecorder` instead
wraps the four recipe helpers used by the basic shirt/T-shirt lane while the
recipe is running:

* ``pygarment.ops.cut_corner``;
* ``assets.garment_programs.sleeves.ArmholeCurve``;
* ``assets.garment_programs.collars.CircleNeckHalf``; and
* ``pygarment.Panel.add_dart``.

Two lower-level factories are also wrapped when available:
``EdgeSeqFactory.from_verts`` records point/straight-edge construction and
``CurveEdgeFactory.curve_from_tangents`` records tangent-derived Beziers.
``EdgeSequence.close_loop`` observes newly appended closure lines, while
``ops.even_armhole_openings`` observes replacement sleeve-head geometry when
those revision-dependent helpers are available.  ``Edge.subdivide_param`` and
``Edge.subdivide_len`` expose the exact creator of cut/subdivided curve pieces;
when nested, their primitives remain visible on the subdivision event but
creation ownership is transparently propagated to the enclosing semantic
helper so enrichment is applied once at the recipe-operation boundary.
Returned geometry from every helper is classified into explicit line,
quadratic/cubic Bezier, and circular-arc primitives.

The recorder deliberately makes no claim that these hooks cover every
GarmentCode operation.  In particular, the global ``Edge`` constructors are
not monkeypatched: doing so duplicates every factory event, affects unrelated
GarmentCode internals, and is more likely to change constructor semantics.
Direct curves used by the basic T-shirt are nevertheless captured at their
enclosing ``ArmholeCurve``/``CircleNeckHalf`` creation boundary.  Each event
contains arguments, input points, created primitives, before/after geometry, a
recipe call-site, stable per-trace object identity tokens, and dependencies
inferred from objects produced or mutated by earlier events.

Monkeypatching is process-global.  Use one recorder around one synchronous
recipe construction; do not generate recipes concurrently in other threads
while the context is active.  No GarmentCode source file is edited.  All
patched attributes and a temporarily inserted checkout path are restored on
every context exit, including partial ``__enter__`` failure and recipe errors.
"""

from __future__ import annotations

import copy
import functools
import importlib
import inspect
import json
import math
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable


TRACE_SCHEMA_VERSION = "garmentcode-runtime-trace-1.0"


class RuntimeTraceError(RuntimeError):
    """Raised when the requested GarmentCode interception surface is absent."""


class RuntimeTraceRecorder:
    """Record construction-time GarmentCode geometry without editing upstream.

    Parameters
    ----------
    garmentcode_root:
        Root of a local GarmentCode checkout.  When omitted, the repository's
        ``external/GarmentCode`` checkout is used if present; otherwise the
        modules must already be importable.  A path inserted into ``sys.path``
        is removed at context exit.
    pyg, sleeves, collars, panel_cls:
        Optional dependency injection points.  They are primarily useful for
        isolated verification and also avoid imposing one import layout on a
        vendored GarmentCode revision.  ``pyg`` must expose ``ops.cut_corner``.
        The module helpers and ``Panel`` class are resolved lazily on entry.
    modules:
        Optional mapping aliases for the same injection points.  Accepted keys
        are ``pyg``, ``sleeves``, ``collars``, and ``panel_cls``.  Explicit
        keyword arguments win over mapping values.
    metadata:
        Caller-owned run metadata (sample id, body-measurement id, split, and
        so on).  It is normalized to JSON values but otherwise uninterpreted.
    event_enricher:
        Optional callback invoked as ``event_enricher(event)`` after an event's
        post-call geometry is complete and while the trace context remains
        active.  It may attach JSON-compatible semantic names, formulas, or
        adapter cross-links.  Callback failures are recorded on that event,
        partial callback mutations are rolled back, and recipe behavior is
        never changed.

    Notes
    -----
    Event dependencies express observed runtime object flow, not a static
    whole-program analysis.  A dependency is added when an input object was an
    output of, or was mutated by, an earlier intercepted operation.  Object
    tokens are deterministic within one trace only; they intentionally do not
    expose CPython memory addresses and must not be compared across runs.
    """

    def __init__(
        self,
        garmentcode_root: str | Path | None = None,
        *,
        pyg: Any | None = None,
        sleeves: Any | None = None,
        collars: Any | None = None,
        panel_cls: type[Any] | None = None,
        modules: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        event_enricher: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        injected = dict(modules or {})
        self._pyg = pyg if pyg is not None else injected.get("pyg")
        self._sleeves = sleeves if sleeves is not None else injected.get("sleeves")
        self._collars = collars if collars is not None else injected.get("collars")
        self._panel_cls = panel_cls if panel_cls is not None else injected.get("panel_cls")

        if garmentcode_root is None:
            candidate = Path(__file__).resolve().parents[2] / "external" / "GarmentCode"
            self.garmentcode_root: Path | None = candidate if candidate.is_dir() else None
        else:
            self.garmentcode_root = Path(garmentcode_root).expanduser().resolve()

        self.metadata = _plain_json(metadata or {})
        self.event_enricher = event_enricher
        self.events: list[dict[str, Any]] = []
        self._object_catalog: dict[str, dict[str, Any]] = {}
        self._tokens_by_identity: dict[int, str] = {}
        # Strong references prevent CPython from reusing an id during a trace.
        self._objects_by_identity: dict[int, Any] = {}
        self._kind_counts: dict[str, int] = {}
        self._last_event_for_token: dict[str, str] = {}
        self._creator_event_for_token: dict[str, str] = {}
        self._parent_event: dict[str, str] = {}
        self._patches: list[tuple[Any, str, Any]] = []
        self._inserted_sys_path: str | None = None
        self._active = False
        self._closed = False
        self._lock = threading.RLock()
        self._thread_state = threading.local()
        self._coverage: dict[str, Any] = {
            "patched_helpers": [],
            "output_geometry_classification": [
                "point",
                "line",
                "quadratic_bezier",
                "cubic_bezier",
                "circular_arc",
            ],
            "gaps": [
                {
                    "targets": ["Edge.__init__", "CurveEdge.__init__", "CircleEdge.__init__"],
                    "reason": (
                        "global constructor wrapping would duplicate factory events and affect "
                        "unrelated GarmentCode internals; direct basic-T-shirt curves are "
                        "classified from ArmholeCurve/CircleNeckHalf outputs"
                    ),
                }
            ],
        }

    def __enter__(self) -> RuntimeTraceRecorder:
        """Resolve GarmentCode modules and install the tracing surface atomically."""

        if self._active:
            raise RuntimeTraceError("a RuntimeTraceRecorder instance cannot be re-entered")
        if self._closed:
            raise RuntimeTraceError("a closed RuntimeTraceRecorder cannot be reused")

        try:
            self._prepare_import_path()
            self._resolve_targets()
            assert self._pyg is not None
            assert self._sleeves is not None
            assert self._collars is not None
            assert self._panel_cls is not None

            self._install(
                self._pyg.ops,
                "cut_corner",
                operation="cut_corner",
                helper="pyg.ops.cut_corner",
                mutated_parameters=("target_interface",),
            )
            self._install(
                self._sleeves,
                "ArmholeCurve",
                operation="ArmholeCurve",
                helper="sleeves.ArmholeCurve",
            )
            self._install(
                self._collars,
                "CircleNeckHalf",
                operation="CircleNeckHalf",
                helper="collars.CircleNeckHalf",
            )
            self._install(
                self._panel_cls,
                "add_dart",
                operation="Panel.add_dart",
                helper="Panel.add_dart",
                mutated_parameters=("self", "edge", "edge_seq", "int_edge_seq"),
            )
            self._install_optional(
                getattr(self._pyg, "EdgeSeqFactory", None),
                "from_verts",
                operation="EdgeSeqFactory.from_verts",
                helper="pyg.EdgeSeqFactory.from_verts",
                point_parameters=("verts",),
            )
            self._install_optional(
                getattr(self._pyg, "CurveEdgeFactory", None),
                "curve_from_tangents",
                operation="CurveEdgeFactory.curve_from_tangents",
                helper="pyg.CurveEdgeFactory.curve_from_tangents",
                point_parameters=("start", "end"),
            )
            self._install_optional(
                getattr(self._pyg, "EdgeSequence", None),
                "close_loop",
                operation="EdgeSequence.close_loop",
                helper="pyg.EdgeSequence.close_loop",
                mutated_parameters=("self",),
            )
            self._install_optional(
                getattr(self._pyg, "ops", None),
                "even_armhole_openings",
                operation="even_armhole_openings",
                helper="pyg.ops.even_armhole_openings",
                mutated_parameters=("front_opening", "back_opening"),
            )
            self._install_optional(
                getattr(self._pyg, "Edge", None),
                "subdivide_param",
                operation="Edge.subdivide_param",
                helper="pyg.Edge.subdivide_param",
                transparent_creation=True,
            )
            self._install_optional(
                getattr(self._pyg, "Edge", None),
                "subdivide_len",
                operation="Edge.subdivide_len",
                helper="pyg.Edge.subdivide_len",
                transparent_creation=True,
            )
            self._active = True
            return self
        except BaseException:
            # __exit__ is not called when __enter__ fails, so rollback here.
            try:
                self._restore_patches()
            finally:
                self._restore_import_path()
            raise

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        """Always restore upstream functions and never suppress recipe errors."""

        restore_error: BaseException | None = None
        try:
            self._restore_patches()
        except BaseException as error:  # pragma: no cover - exotic proxy objects
            restore_error = error
        finally:
            try:
                self._restore_import_path()
            except BaseException as error:  # pragma: no cover - sys.path proxying
                if restore_error is None:
                    restore_error = error
            self._active = False
            self._closed = True

        if restore_error is not None:
            if exc is not None and hasattr(exc, "add_note"):
                exc.add_note(f"RuntimeTraceRecorder restoration also failed: {restore_error}")
            elif exc is None:
                raise RuntimeTraceError("failed to restore a GarmentCode trace patch") from restore_error
        return False

    @property
    def active(self) -> bool:
        """Whether the global wrappers are currently installed."""

        return self._active

    @property
    def objects(self) -> dict[str, dict[str, Any]]:
        """Return a detached token-to-object descriptor catalog."""

        return copy.deepcopy(self._object_catalog)

    @property
    def coverage(self) -> dict[str, Any]:
        """Return exact installed-hook coverage and intentional trace gaps."""

        return copy.deepcopy(self._coverage)

    def object_token(self, value: Any, kind: str | None = None) -> str:
        """Return a stable per-trace identity token for adapter cross-linking.

        The recorder retains a strong reference until the trace object is
        released, preventing Python identity reuse.  ``kind`` only affects a
        newly seen object and is normalized to a JSON/DAG-friendly token
        prefix; an existing object's token never changes.
        """

        if value is None:
            raise TypeError("cannot assign an object token to None")
        token_kind = _normalize_token_kind(kind) if kind is not None else None
        with self._lock:
            return self._token(value, token_kind)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, strict-JSON-compatible trace document."""

        document = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "state": "active" if self._active else ("closed" if self._closed else "new"),
            "metadata": copy.deepcopy(self.metadata),
            "events": copy.deepcopy(self.events),
            "objects": self.objects,
            "coverage": self.coverage,
        }
        # This is both a defensive assertion for future snapshot extensions and
        # a convenient way to detach any unusual Mapping implementations.
        return json.loads(json.dumps(document, allow_nan=False, sort_keys=True))

    def write_json(self, path: str | Path) -> None:
        """Write :meth:`to_dict` as deterministic UTF-8 JSON."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    # -- context setup -------------------------------------------------

    def _prepare_import_path(self) -> None:
        if self.garmentcode_root is None:
            return
        if not self.garmentcode_root.is_dir():
            raise RuntimeTraceError(f"GarmentCode checkout does not exist: {self.garmentcode_root}")
        root = str(self.garmentcode_root)
        if root not in sys.path:
            sys.path.insert(0, root)
            self._inserted_sys_path = root

    def _restore_import_path(self) -> None:
        if self._inserted_sys_path is None:
            return
        try:
            sys.path.remove(self._inserted_sys_path)
        except ValueError:
            pass
        finally:
            self._inserted_sys_path = None

    def _resolve_targets(self) -> None:
        try:
            if self._pyg is None:
                self._pyg = importlib.import_module("pygarment")
            if self._sleeves is None:
                self._sleeves = importlib.import_module("assets.garment_programs.sleeves")
            if self._collars is None:
                self._collars = importlib.import_module("assets.garment_programs.collars")
            if self._panel_cls is None:
                self._panel_cls = getattr(self._pyg, "Panel", None)
                if self._panel_cls is None:
                    panel_module = importlib.import_module("pygarment.garmentcode.panel")
                    self._panel_cls = panel_module.Panel
        except (ImportError, AttributeError) as error:
            raise RuntimeTraceError(
                "could not import the local GarmentCode tracing targets; pass "
                "garmentcode_root or inject pyg/sleeves/collars/panel_cls"
            ) from error

        missing = []
        if not hasattr(self._pyg, "ops") or not hasattr(self._pyg.ops, "cut_corner"):
            missing.append("pyg.ops.cut_corner")
        if not hasattr(self._sleeves, "ArmholeCurve"):
            missing.append("sleeves.ArmholeCurve")
        if not hasattr(self._collars, "CircleNeckHalf"):
            missing.append("collars.CircleNeckHalf")
        if self._panel_cls is None or not hasattr(self._panel_cls, "add_dart"):
            missing.append("Panel.add_dart")
        if missing:
            raise RuntimeTraceError(f"GarmentCode tracing targets are missing: {', '.join(missing)}")

    def _install(
        self,
        owner: Any,
        attribute: str,
        *,
        operation: str,
        helper: str,
        mutated_parameters: tuple[str, ...] = (),
        point_parameters: tuple[str, ...] = (),
        transparent_creation: bool = False,
    ) -> None:
        raw_original = inspect.getattr_static(owner, attribute)
        if isinstance(raw_original, staticmethod):
            original = raw_original.__func__
            descriptor: Callable[[Callable[..., Any]], Any] = staticmethod
        elif isinstance(raw_original, classmethod):
            original = raw_original.__func__
            descriptor = classmethod
        else:
            original = getattr(owner, attribute)
            descriptor = lambda function: function
        implementation = inspect.unwrap(original)
        implementation_reference = self._implementation_reference(implementation)

        @functools.wraps(original)
        def traced(*args: Any, **kwargs: Any) -> Any:
            # A caller can retain a reference to a temporary wrapper.  Once the
            # context has closed it behaves exactly like the original and does
            # not append events to a completed document.
            if not self._active:
                return original(*args, **kwargs)
            return self._trace_call(
                original,
                implementation,
                args,
                kwargs,
                operation=operation,
                helper=helper,
                implementation_reference=implementation_reference,
                mutated_parameters=mutated_parameters,
                point_parameters=point_parameters,
                transparent_creation=transparent_creation,
            )

        self._patches.append((owner, attribute, raw_original))
        setattr(owner, attribute, descriptor(traced))
        self._coverage["patched_helpers"].append(helper)

    def _install_optional(
        self,
        owner: Any | None,
        attribute: str,
        *,
        operation: str,
        helper: str,
        mutated_parameters: tuple[str, ...] = (),
        point_parameters: tuple[str, ...] = (),
        transparent_creation: bool = False,
    ) -> None:
        """Install a revision-dependent hook, documenting absence instead of failing."""

        if owner is None or not hasattr(owner, attribute):
            self._coverage["gaps"].append(
                {
                    "targets": [helper],
                    "reason": "helper is absent from this GarmentCode revision",
                }
            )
            return
        self._install(
            owner,
            attribute,
            operation=operation,
            helper=helper,
            mutated_parameters=mutated_parameters,
            point_parameters=point_parameters,
            transparent_creation=transparent_creation,
        )

    def _restore_patches(self) -> None:
        first_error: BaseException | None = None
        while self._patches:
            owner, attribute, original = self._patches.pop()
            try:
                setattr(owner, attribute, original)
            except BaseException as error:  # restore every remaining target
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    # -- call recording ------------------------------------------------

    def _trace_call(
        self,
        original: Callable[..., Any],
        implementation: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        operation: str,
        helper: str,
        implementation_reference: str,
        mutated_parameters: tuple[str, ...],
        point_parameters: tuple[str, ...],
        transparent_creation: bool,
    ) -> Any:
        try:
            bound = _bind_arguments(implementation, args, kwargs)
            event = self._start_event(
                operation=operation,
                helper=helper,
                implementation_reference=implementation_reference,
                bound=bound,
                mutated_parameters=mutated_parameters,
                point_parameters=point_parameters,
                transparent_creation=transparent_creation,
            )
        except Exception:
            # Observability must not make an otherwise valid recipe fail.
            event = None

        if event is not None:
            self._push_event(event["id"])
        try:
            result = original(*args, **kwargs)
        except BaseException as error:
            if event is not None:
                try:
                    self._finish_event(event, bound, mutated_parameters, result=None, error=error)
                finally:
                    self._pop_event(event["id"])
            raise
        else:
            if event is not None:
                try:
                    self._finish_event(event, bound, mutated_parameters, result=result, error=None)
                finally:
                    self._pop_event(event["id"])
            return result

    def _start_event(
        self,
        *,
        operation: str,
        helper: str,
        implementation_reference: str,
        bound: dict[str, Any],
        mutated_parameters: tuple[str, ...],
        point_parameters: tuple[str, ...],
        transparent_creation: bool,
    ) -> dict[str, Any]:
        with self._lock:
            order = len(self.events)
            event_id = f"op_{order + 1:06d}"
            parent_id = self._active_event_id()
            if parent_id is not None:
                self._parent_event[event_id] = parent_id
            point_inputs = self._point_records_for_named(bound, point_parameters)
            input_tokens = _unique(
                (*self._tokens_in(bound), *(point["object_token"] for point in point_inputs))
            )
            dependencies = _unique(
                (
                    *((parent_id,) if parent_id is not None else ()),
                    *(
                        self._last_event_for_token[token]
                        for token in input_tokens
                        if token in self._last_event_for_token
                    ),
                )
            )
            panel_id = self._panel_id(bound)
            pre_geometry = {
                name: self._safe_snapshot(value)
                for name, value in bound.items()
                if self._has_geometry(value)
            }
            event: dict[str, Any] = {
                "id": event_id,
                "order": order,
                "operation": operation,
                "parent_event_id": parent_id,
                "dependencies": dependencies,
                "inputs": input_tokens,
                "outputs": [],
                "parameters": {
                    name: self._parameter_value(value)
                    for name, value in bound.items()
                    if name != "self"
                },
                "source_reference": self._callsite_reference(),
                "implementation_reference": implementation_reference,
                "pre_geometry": pre_geometry,
                "post_geometry": {},
                "helper": helper,
                "is_helper": True,
                "status": "running",
                "object_tokens": {
                    "inputs": input_tokens,
                    "mutated": self._tokens_for_named(bound, mutated_parameters),
                    "outputs": [],
                },
                "panel_id": panel_id,
                "object_type": self._primary_object_type(bound),
                "point_inputs": point_inputs,
                "created_points": [],
                "created_primitives": [],
                "transparent_creation": transparent_creation,
                "error": None,
            }
            # Reserve the slot at function entry so nested traced calls retain
            # actual invocation order even though this event finishes later.
            self.events.append(event)
            return event

    def _finish_event(
        self,
        event: dict[str, Any],
        bound: dict[str, Any],
        mutated_parameters: tuple[str, ...],
        *,
        result: Any,
        error: BaseException | None,
    ) -> None:
        try:
            with self._lock:
                post = {
                    name: self._safe_snapshot(value)
                    for name, value in bound.items()
                    if self._has_geometry(value)
                }
                if error is None or result is not None:
                    post["result"] = self._safe_snapshot(result)

                output_tokens = self._tokens_in(result) if error is None else []
                mutated_tokens = self._tokens_for_named(bound, mutated_parameters)
                mutated_values = {
                    name: bound[name]
                    for name in mutated_parameters
                    if name in bound and bound[name] is not None
                }
                pre_call_tokens = set(event["object_tokens"]["inputs"])
                # A nested traced helper owns primitives it created.  Exclude
                # them from its parent's completion event to avoid duplicate
                # creation labels while retaining the parent->child DAG edge.
                creation_exclusions = pre_call_tokens | set(self._creator_event_for_token)
                creation_sources = [
                    ("result", result if error is None else None),
                    *(
                        (f"mutated_parameter:{name}", value)
                        for name, value in mutated_values.items()
                    ),
                ]
                created_primitives = self._created_primitives_from_sources(
                    creation_sources,
                    event_id=event["id"],
                    exclude_tokens=creation_exclusions,
                )
                event["post_geometry"] = post
                event["outputs"] = output_tokens
                event["object_tokens"]["mutated"] = mutated_tokens
                event["object_tokens"]["outputs"] = output_tokens
                event["created_primitives"] = created_primitives
                event["created_points"] = self._created_points(
                    created_primitives,
                    exclude_tokens=pre_call_tokens,
                )
                event["status"] = "ok" if error is None else "error"
                if error is not None:
                    event["error"] = {
                        "type": f"{type(error).__module__}.{type(error).__qualname__}",
                        "message": str(error),
                    }

                deferred_creation_claim = bool(
                    event.get("transparent_creation") and event.get("parent_event_id") is not None
                )
                event["creation_attribution"] = (
                    "pending_semantic_enrichment" if deferred_creation_claim else "this_event"
                )
                if not deferred_creation_claim:
                    self._register_created_tokens(event, pre_call_tokens)

                if error is None:
                    for token in _unique((*mutated_tokens, *output_tokens)):
                        previous = self._last_event_for_token.get(token)
                        if previous is not None and self._is_descendant_event(previous, event["id"]):
                            continue
                        self._last_event_for_token[token] = event["id"]
        except Exception as capture_error:
            # Preserve the recipe's return value/exception even if an unusual
            # third-party object defeats a best-effort geometry snapshot.
            event["status"] = "error" if error is not None else "ok"
            event["capture_error"] = {
                "type": f"{type(capture_error).__module__}.{type(capture_error).__qualname__}",
                "message": str(capture_error),
            }
        finally:
            self._run_event_enricher(event)
            self._finalize_deferred_creation_claim(event)

    def _register_created_tokens(self, event: Mapping[str, Any], pre_call_tokens: set[str]) -> None:
        with self._lock:
            for primitive in event.get("created_primitives", ()):
                if not isinstance(primitive, Mapping):
                    continue
                edge_token = primitive.get("edge_token")
                if isinstance(edge_token, str):
                    self._creator_event_for_token.setdefault(edge_token, str(event["id"]))
                for point_name in ("start_point", "end_point"):
                    point = primitive.get(point_name)
                    token = point.get("object_token") if isinstance(point, Mapping) else None
                    if isinstance(token, str) and token not in pre_call_tokens:
                        self._creator_event_for_token.setdefault(token, str(event["id"]))
                for point in primitive.get("control_points", ()):
                    token = point.get("object_token") if isinstance(point, Mapping) else None
                    if isinstance(token, str):
                        self._creator_event_for_token.setdefault(token, str(event["id"]))

    def _finalize_deferred_creation_claim(self, event: dict[str, Any]) -> None:
        if event.get("creation_attribution") != "pending_semantic_enrichment":
            return
        semantic_primitives = event.get("semantic_primitives", ())
        claimed = isinstance(semantic_primitives, Sequence) and any(
            isinstance(item, Mapping) and item.get("edge_token")
            for item in semantic_primitives
        )
        if claimed:
            self._register_created_tokens(event, set(event.get("inputs", ())))
            event["creation_attribution"] = "this_event_after_semantic_enrichment"
        else:
            event["creation_attribution"] = "enclosing_parent_event"

    def _run_event_enricher(self, event: dict[str, Any]) -> None:
        callback = self.event_enricher
        if callback is None:
            return
        before = copy.deepcopy(event)
        try:
            callback(event)
            normalized = _plain_json(event)
            if not isinstance(normalized, dict):
                raise TypeError("event_enricher must leave the event as a mapping")
            event.clear()
            event.update(normalized)
        except BaseException as enrichment_error:
            event.clear()
            event.update(before)
            event["enrichment_error"] = {
                "type": (
                    f"{type(enrichment_error).__module__}."
                    f"{type(enrichment_error).__qualname__}"
                ),
                "message": _safe_error_message(enrichment_error),
            }

    def _event_stack(self) -> list[str]:
        stack = getattr(self._thread_state, "event_stack", None)
        if stack is None:
            stack = []
            self._thread_state.event_stack = stack
        return stack

    def _active_event_id(self) -> str | None:
        stack = self._event_stack()
        return stack[-1] if stack else None

    def _push_event(self, event_id: str) -> None:
        self._event_stack().append(event_id)

    def _pop_event(self, event_id: str) -> None:
        stack = self._event_stack()
        if stack and stack[-1] == event_id:
            stack.pop()
            return
        # Defensive recovery for an enricher that made a re-entrant traced
        # call and failed unusually.  Never let bookkeeping alter the recipe.
        for index in range(len(stack) - 1, -1, -1):
            if stack[index] == event_id:
                del stack[index]
                break

    def _is_descendant_event(self, candidate: str, ancestor: str) -> bool:
        current = candidate
        seen: set[str] = set()
        while current in self._parent_event and current not in seen:
            seen.add(current)
            current = self._parent_event[current]
            if current == ancestor:
                return True
        return False

    # -- identity and dependency support ------------------------------

    def _token(self, value: Any, kind: str | None = None) -> str:
        identity = id(value)
        existing = self._tokens_by_identity.get(identity)
        if existing is not None:
            return existing

        token_kind = _token_kind(value) if kind is None else kind
        count = self._kind_counts.get(token_kind, 0) + 1
        self._kind_counts[token_kind] = count
        token = f"{token_kind}_{count:06d}"
        self._tokens_by_identity[identity] = token
        self._objects_by_identity[identity] = value

        descriptor: dict[str, Any] = {
            "object_type": f"{type(value).__module__}.{type(value).__qualname__}",
        }
        name = getattr(value, "name", None)
        if isinstance(name, (str, int, float, bool)):
            descriptor["name"] = _plain_json(name)
        label = getattr(value, "label", None)
        if isinstance(label, (str, int, float, bool)) and label != "":
            descriptor["label"] = _plain_json(label)
        self._object_catalog[token] = descriptor
        return token

    def _tokens_in(self, value: Any) -> list[str]:
        tokens: list[str] = []
        seen: set[int] = set()

        def visit(item: Any) -> None:
            if item is None or isinstance(item, (str, bytes, int, float, bool, Path)):
                return
            identity = id(item)
            if identity in seen:
                return
            seen.add(identity)

            if self._is_domain_object(item):
                if self._is_panel(item):
                    tokens.append(self._token(item, "panel"))
                elif self._is_interface(item):
                    tokens.append(self._token(item, "interface"))
                elif self._is_edge_sequence(item):
                    tokens.append(self._token(item, "edge_sequence"))
                else:
                    tokens.append(self._token(item, "edge"))
                if self._is_edge(item):
                    tokens.append(self._token(item.start, "vertex"))
                    tokens.append(self._token(item.end, "vertex"))
                    for control in getattr(item, "control_points", ()) or ():
                        tokens.append(self._token(control, "control_point"))
                if self._is_interface(item):
                    visit(getattr(item, "edges", None))
                    visit(getattr(item, "panel", None))
                elif self._is_panel(item) or self._is_edge_sequence(item):
                    visit(getattr(item, "edges", None))
                return
            if isinstance(item, Mapping):
                for child in item.values():
                    visit(child)
            elif isinstance(item, Sequence):
                for child in item:
                    visit(child)

        visit(value)
        return _unique(tokens)

    def _point_records_for_named(
        self, bound: Mapping[str, Any], names: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        """Capture point objects at selected factory inputs without guessing semantics."""

        records: list[dict[str, Any]] = []
        seen: set[int] = set()

        def visit(value: Any, parameter: str) -> None:
            if _is_coordinate(value):
                if id(value) not in seen:
                    seen.add(id(value))
                    records.append(
                        {
                            "parameter": parameter,
                            "object_token": self._token(value, "vertex"),
                            "xy": _coordinate(value),
                            "coordinate_space": "local_2d_cm",
                            "role": "factory_input",
                        }
                    )
                return
            if isinstance(value, Mapping):
                for key, child in value.items():
                    visit(child, f"{parameter}.{key}")
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                for index, child in enumerate(value):
                    visit(child, f"{parameter}[{index}]")

        for name in names:
            if name in bound:
                visit(bound[name], name)
        return records

    def _created_primitives(
        self,
        value: Any,
        *,
        event_id: str,
        exclude_tokens: set[str],
    ) -> list[dict[str, Any]]:
        """Describe newly returned line/Bezier/arc edges once per identity."""

        edges: list[Any] = []
        seen_objects: set[int] = set()

        def visit(item: Any) -> None:
            if item is None or isinstance(item, (str, bytes, int, float, bool, Path)):
                return
            identity = id(item)
            if identity in seen_objects:
                return
            seen_objects.add(identity)
            if self._is_edge(item):
                edges.append(item)
                return
            if self._is_interface(item) or self._is_panel(item) or self._is_edge_sequence(item):
                visit(getattr(item, "edges", None))
                return
            if isinstance(item, Mapping):
                for child in item.values():
                    visit(child)
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                for child in item:
                    visit(child)

        visit(value)
        primitives: list[dict[str, Any]] = []
        for edge in edges:
            edge_token = self._token(edge, "edge")
            if edge_token in exclude_tokens:
                continue
            controls = list(getattr(edge, "control_points", ()) or ())
            if controls:
                kind = "quadratic_bezier" if len(controls) == 1 else "cubic_bezier"
                formula = (
                    "B(t)=(1-t)^2*P0+2(1-t)t*P1+t^2*P2"
                    if len(controls) == 1
                    else "B(t)=(1-t)^3*P0+3(1-t)^2t*P1+3(1-t)t^2*P2+t^3*P3"
                )
            elif hasattr(edge, "control_y"):
                kind = "circular_arc"
                formula = "GarmentCode CircleEdge(start,end,relative_control_y)"
            else:
                kind = "line"
                formula = "P(t)=(1-t)*start+t*end"

            primitive: dict[str, Any] = {
                "id": f"{event_id}.primitive_{len(primitives) + 1:04d}",
                "kind": kind,
                "edge_token": edge_token,
                "start_point": {
                    "object_token": self._token(edge.start, "vertex"),
                    "xy": _coordinate(edge.start),
                    "coordinate_space": "local_2d_cm",
                },
                "end_point": {
                    "object_token": self._token(edge.end, "vertex"),
                    "xy": _coordinate(edge.end),
                    "coordinate_space": "local_2d_cm",
                },
                "construction_formula": formula,
                "length": _safe_length(edge),
            }
            if controls:
                primitive["control_points"] = [
                    {
                        "object_token": self._token(control, "control_point"),
                        "xy": _coordinate(control),
                        "coordinate_space": "edge_relative",
                    }
                    for control in controls
                ]
            elif hasattr(edge, "control_y"):
                primitive["relative_control_y"] = _plain_json(edge.control_y)
            primitives.append(primitive)
        return primitives

    def _created_primitives_from_sources(
        self,
        sources: Sequence[tuple[str, Any]],
        *,
        event_id: str,
        exclude_tokens: set[str],
    ) -> list[dict[str, Any]]:
        """Merge result- and mutation-observed primitives without duplicates."""

        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for source, value in sources:
            for primitive in self._created_primitives(
                value,
                event_id=event_id,
                exclude_tokens=exclude_tokens,
            ):
                token = str(primitive["edge_token"])
                if token not in merged:
                    primitive["observed_via"] = [source]
                    merged[token] = primitive
                    order.append(token)
                elif source not in merged[token]["observed_via"]:
                    merged[token]["observed_via"].append(source)
        output = [merged[token] for token in order]
        for index, primitive in enumerate(output, start=1):
            primitive["id"] = f"{event_id}.primitive_{index:04d}"
        return output

    @staticmethod
    def _created_points(
        primitives: Sequence[Mapping[str, Any]], *, exclude_tokens: set[str]
    ) -> list[dict[str, Any]]:
        """Flatten unique output endpoint/control points for point supervision."""

        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for primitive in primitives:
            candidates = [primitive.get("start_point"), primitive.get("end_point")]
            candidates.extend(primitive.get("control_points", ()))
            for point in candidates:
                if not isinstance(point, Mapping):
                    continue
                token = point.get("object_token")
                if not isinstance(token, str) or token in exclude_tokens or token in seen:
                    continue
                seen.add(token)
                record = dict(point)
                record["role"] = "created_control_point" if token.startswith("control_point_") else "created_endpoint"
                output.append(record)
        return output

    def _tokens_for_named(self, bound: Mapping[str, Any], names: tuple[str, ...]) -> list[str]:
        tokens: list[str] = []
        for name in names:
            if name in bound and bound[name] is not None:
                tokens.extend(self._tokens_in(bound[name]))
                if name == "target_interface":
                    tokens.extend(self._tokens_in(getattr(bound[name], "panel", None)))
        return _unique(tokens)

    # -- JSON and geometry snapshots ----------------------------------

    def _safe_snapshot(self, value: Any) -> Any:
        try:
            return self._snapshot(value, set())
        except Exception as error:
            return {
                "capture_error": {
                    "type": f"{type(error).__module__}.{type(error).__qualname__}",
                    "message": str(error),
                }
            }

    def _snapshot(self, value: Any, ancestors: set[int]) -> Any:
        if value is None or isinstance(value, (str, int, bool)):
            return value
        if isinstance(value, float):
            return _finite_float(value)
        if isinstance(value, Path):
            return str(value)

        identity = id(value)
        if identity in ancestors:
            return {"object_token": self._token(value), "recursive": True}
        nested = set(ancestors)
        nested.add(identity)

        if self._is_panel(value):
            edges = getattr(value, "edges", None)
            snapshot: dict[str, Any] = {
                "object_token": self._token(value, "panel"),
                "object_type": type(value).__name__,
                "name": _plain_json(getattr(value, "name", None)),
                "label": _plain_json(getattr(value, "label", None)),
                "edges": self._snapshot(edges, nested),
            }
            translation = getattr(value, "translation", None)
            if translation is not None:
                snapshot["translation"] = _plain_json(translation)
            rotation = getattr(value, "rotation", None)
            if rotation is not None:
                try:
                    snapshot["rotation_xyz_degrees"] = _plain_json(
                        rotation.as_euler("XYZ", degrees=True)
                    )
                except Exception:
                    snapshot["rotation"] = self._parameter_value(rotation)
            interfaces = getattr(value, "interfaces", None)
            if isinstance(interfaces, Mapping):
                snapshot["interfaces"] = {
                    str(key): self._interface_reference(interface)
                    for key, interface in interfaces.items()
                }
            stitching = getattr(value, "stitching_rules", None)
            if stitching is not None:
                snapshot["stitching_rule_count"] = _safe_len(stitching)
            return snapshot

        if self._is_interface(value):
            panels = getattr(value, "panel", ())
            if not isinstance(panels, Sequence):
                panels = (panels,)
            return {
                "object_token": self._token(value, "interface"),
                "object_type": type(value).__name__,
                "panel_tokens": [
                    self._token(panel, "panel") for panel in panels if panel is not None
                ],
                "panel_ids": [
                    _plain_json(getattr(panel, "name", None)) for panel in panels if panel is not None
                ],
                "edges": self._snapshot(getattr(value, "edges", None), nested),
                "ruffle": _plain_json(getattr(value, "ruffle", None)),
                "edges_flipping": _plain_json(getattr(value, "edges_flipping", None)),
                "right_wrong": _plain_json(getattr(value, "right_wrong", None)),
            }

        if self._is_edge_sequence(value):
            edges = list(getattr(value, "edges", ()))
            vertices: list[dict[str, Any]] = []
            seen_vertices: set[int] = set()
            for edge in edges:
                for vertex in (getattr(edge, "start", None), getattr(edge, "end", None)):
                    if vertex is None or id(vertex) in seen_vertices:
                        continue
                    seen_vertices.add(id(vertex))
                    vertices.append(
                        {
                            "object_token": self._token(vertex, "vertex"),
                            "xy": _coordinate(vertex),
                        }
                    )
            return {
                "object_token": self._token(value, "edge_sequence"),
                "object_type": type(value).__name__,
                "edges": [self._snapshot(edge, nested) for edge in edges],
                "vertices": vertices,
                "length": _safe_length(value),
                "is_chained": _safe_predicate(value, "isChained"),
                "is_loop": _safe_predicate(value, "isLoop"),
            }

        if self._is_edge(value):
            snapshot = {
                "object_token": self._token(value, "edge"),
                "object_type": type(value).__name__,
                "start": _coordinate(value.start),
                "end": _coordinate(value.end),
                "start_token": self._token(value.start, "vertex"),
                "end_token": self._token(value.end, "vertex"),
                "length": _safe_length(value),
                "label": _plain_json(getattr(value, "label", None)),
            }
            controls = getattr(value, "control_points", None)
            if controls is not None:
                snapshot["curvature"] = {
                    "type": "quadratic" if _safe_len(controls) == 1 else "cubic",
                    "relative_control_points": _plain_json(controls),
                    "control_points": [
                        {
                            "object_token": self._token(control, "control_point"),
                            "xy": _coordinate(control),
                            "coordinate_space": "edge_relative",
                        }
                        for control in controls
                    ],
                }
            elif hasattr(value, "control_y"):
                snapshot["curvature"] = {
                    "type": "circle",
                    "relative_control_y": _plain_json(getattr(value, "control_y")),
                }
            else:
                snapshot["curvature"] = {"type": "line"}
            return snapshot

        if isinstance(value, Mapping):
            return {str(key): self._snapshot(child, nested) for key, child in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._snapshot(child, nested) for child in value]
        if hasattr(value, "tolist"):
            try:
                return _plain_json(value.tolist())
            except Exception:
                pass
        if self._is_domain_object(value):
            return {
                "object_token": self._token(value),
                "object_type": f"{type(value).__module__}.{type(value).__qualname__}",
            }
        return _plain_json(value)

    def _parameter_value(self, value: Any) -> Any:
        if self._is_domain_object(value):
            return {
                "object_token": self._token(value),
                "object_type": type(value).__name__,
            }
        if isinstance(value, Mapping):
            return {str(key): self._parameter_value(child) for key, child in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._parameter_value(child) for child in value]
        return _plain_json(value)

    def _interface_reference(self, value: Any) -> Any:
        if not self._is_interface(value):
            return self._parameter_value(value)
        return {
            "object_token": self._token(value, "interface"),
            "edge_tokens": [
                self._token(edge, "edge") for edge in getattr(getattr(value, "edges", None), "edges", ())
            ],
        }

    # -- structural predicates and source references ------------------

    @staticmethod
    def _is_edge(value: Any) -> bool:
        return (
            value is not None
            and hasattr(value, "start")
            and hasattr(value, "end")
            and callable(getattr(value, "length", None))
            and not hasattr(value, "edges")
        )

    @classmethod
    def _is_interface(cls, value: Any) -> bool:
        return (
            value is not None
            and hasattr(value, "panel")
            and hasattr(value, "edges")
            and not hasattr(value, "translation")
        )

    @classmethod
    def _is_panel(cls, value: Any) -> bool:
        return (
            value is not None
            and hasattr(value, "edges")
            and hasattr(value, "name")
            and (hasattr(value, "translation") or hasattr(value, "interfaces"))
        )

    @classmethod
    def _is_edge_sequence(cls, value: Any) -> bool:
        return (
            value is not None
            and hasattr(value, "edges")
            and not cls._is_panel(value)
            and not cls._is_interface(value)
        )

    @classmethod
    def _is_domain_object(cls, value: Any) -> bool:
        return (
            cls._is_edge(value)
            or cls._is_interface(value)
            or cls._is_panel(value)
            or cls._is_edge_sequence(value)
        )

    @classmethod
    def _has_geometry(cls, value: Any) -> bool:
        if cls._is_domain_object(value):
            return True
        if isinstance(value, Mapping):
            return any(cls._has_geometry(child) for child in value.values())
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return any(cls._has_geometry(child) for child in value)
        return False

    @classmethod
    def _panel_id(cls, bound: Mapping[str, Any]) -> str | None:
        panel = bound.get("self")
        if cls._is_panel(panel):
            name = getattr(panel, "name", None)
            return str(name) if name is not None else None
        interface = bound.get("target_interface")
        if cls._is_interface(interface):
            panels = getattr(interface, "panel", ())
            if not isinstance(panels, Sequence):
                panels = (panels,)
            names = [str(getattr(item, "name")) for item in panels if getattr(item, "name", None) is not None]
            return names[0] if names and len(set(names)) == 1 else ("|".join(_unique(names)) or None)
        return None

    @classmethod
    def _primary_object_type(cls, bound: Mapping[str, Any]) -> str | None:
        for name in ("self", "target_interface", "target_shape", "dart_shape", "edge"):
            value = bound.get(name)
            if cls._is_domain_object(value):
                return type(value).__name__
        return None

    def _implementation_reference(self, function: Callable[..., Any]) -> str:
        module_name = getattr(function, "__module__", type(function).__module__)
        qualname = getattr(function, "__qualname__", getattr(function, "__name__", "<callable>"))
        try:
            file_name = inspect.getsourcefile(function) or inspect.getfile(function)
            line = inspect.getsourcelines(function)[1]
            return self._path_reference(Path(file_name), line)
        except (OSError, TypeError):
            return f"{module_name}:{qualname}"

    def _callsite_reference(self) -> str | None:
        own_file = Path(__file__).resolve()
        try:
            frame = sys._getframe(1)
        except ValueError:  # pragma: no cover
            return None
        while frame is not None:
            path = Path(frame.f_code.co_filename)
            try:
                is_own_file = path.resolve() == own_file
            except OSError:
                is_own_file = path == own_file
            if not is_own_file:
                return self._path_reference(path, frame.f_lineno)
            frame = frame.f_back
        return None

    def _path_reference(self, path: Path, line: int) -> str:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if self.garmentcode_root is not None:
            try:
                relative = resolved.relative_to(self.garmentcode_root.resolve())
                return f"external/GarmentCode/{relative.as_posix()}:{line}"
            except (OSError, ValueError):
                pass
        try:
            repository_root = Path(__file__).resolve().parents[2]
            relative = resolved.relative_to(repository_root)
            return f"{relative.as_posix()}:{line}"
        except (OSError, ValueError):
            return f"{resolved.name}:{line}"


def _bind_arguments(
    function: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind call arguments while remaining usable with C/fake callables."""

    try:
        signature = inspect.signature(function)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (TypeError, ValueError):
        output = {f"arg_{index}": value for index, value in enumerate(args)}
        output.update({str(key): value for key, value in kwargs.items()})
        return output


def _plain_json(value: Any, ancestors: set[int] | None = None) -> Any:
    """Convert arbitrary metadata/scalars to strict JSON without ``repr`` ids."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return _finite_float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}

    if ancestors is None:
        ancestors = set()
    identity = id(value)
    if identity in ancestors:
        return {"recursive": True, "object_type": type(value).__name__}
    nested = set(ancestors)
    nested.add(identity)

    if isinstance(value, Mapping):
        return {str(key): _plain_json(child, nested) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_json(child, nested) for child in value]
    if hasattr(value, "tolist"):
        try:
            return _plain_json(value.tolist(), nested)
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _plain_json(value.item(), nested)
        except Exception:
            pass
    if isinstance(value, type):
        return {"type": f"{value.__module__}.{value.__qualname__}"}
    return {"object_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _finite_float(value: float) -> float | str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _coordinate(value: Any) -> list[Any] | None:
    try:
        return [_plain_json(value[0]), _plain_json(value[1])]
    except (IndexError, KeyError, TypeError):
        return None


def _is_coordinate(value: Any) -> bool:
    """Whether *value* is a mutable/array-like two-dimensional coordinate."""

    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return False
    try:
        if len(value) != 2:
            return False
        float(value[0])
        float(value[1])
        return True
    except (TypeError, ValueError, IndexError, KeyError):
        return False


def _safe_length(value: Any) -> float | str | None:
    try:
        raw = value.length()
        return _finite_float(float(raw))
    except Exception:
        return None


def _safe_len(value: Any) -> int | None:
    try:
        return len(value)
    except (TypeError, AttributeError):
        return None


def _safe_predicate(value: Any, method: str) -> bool | None:
    predicate = getattr(value, method, None)
    if not callable(predicate):
        return None
    try:
        return bool(predicate())
    except Exception:
        return None


def _token_kind(value: Any) -> str:
    return _normalize_token_kind(type(value).__name__)


def _normalize_token_kind(value: str) -> str:
    name = str(value).lower()
    cleaned = "".join(character if character.isalnum() else "_" for character in name).strip("_")
    return cleaned or "object"


def _safe_error_message(error: BaseException) -> str:
    try:
        return str(error)
    except BaseException:  # pragma: no cover - hostile callback exception
        return f"<{type(error).__module__}.{type(error).__qualname__}>"


def _unique(values: Any) -> list[Any]:
    output: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


__all__ = ["TRACE_SCHEMA_VERSION", "RuntimeTraceError", "RuntimeTraceRecorder"]
