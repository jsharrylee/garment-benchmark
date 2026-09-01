"""Shared 2D-pattern teacher and four-view semantic student contract.

The models in this module deliberately stop before CAD generation.  They
produce the *same* category-conditioned element representation so that a
vector-pattern encoder can supervise an image-only encoder without making the
pattern available at inference time.

The representation is intentionally explicit and auditable:

* one fixed query inventory for basic T-shirts, trousers, and skirts;
* a presence logit for every applicable semantic element;
* normalized element coordinates (point, path drafting geometry, or panel box); and
* a latent element token used for cross-modal distillation.

Coordinates use a canonical garment/panel frame.  Landmark queries use the
first two channels, panel and non-boundary reference-line queries use four,
and path queries add explicit arc-length, signed-depth, and endpoint-tangent
channels.  This
module defines the neural contract only; extraction of exact targets from
GarmentCode/FreeSewing records belongs in a separate data-integration step.

PyTorch is imported lazily so schema and inventory inspection remains usable
in lightweight preprocessing environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


CATEGORY_NAMES = ("tshirt", "pants", "skirt")
"""Stable category order used by ``category_ids`` tensors."""

ELEMENT_KINDS = ("panel", "path", "landmark", "reference_line")
MAX_COORDINATE_DIM = 8
PANEL_COORDINATES = ("center_u", "center_v", "width", "height")
PATH_COORDINATES = (
    "start_u",
    "start_v",
    "end_u",
    "end_v",
    "arc_length_norm",
    "signed_depth_norm",
    "start_tangent_angle_norm",
    "end_tangent_angle_norm",
)
LANDMARK_COORDINATES = ("u", "v")
REFERENCE_LINE_COORDINATES = ("start_u", "start_v", "end_u", "end_v")


@dataclass(frozen=True)
class SemanticQuery:
    """One category-specific item in the shared semantic query inventory."""

    category: str
    kind: str
    name: str
    coordinate_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.category not in CATEGORY_NAMES:
            raise ValueError(f"unsupported garment category: {self.category!r}")
        if self.kind not in ELEMENT_KINDS:
            raise ValueError(f"unsupported semantic element kind: {self.kind!r}")
        if not 1 <= len(self.coordinate_names) <= MAX_COORDINATE_DIM:
            raise ValueError(
                f"semantic query coordinates must use one to {MAX_COORDINATE_DIM} channels"
            )

    @property
    def key(self) -> str:
        return f"{self.category}:{self.kind}:{self.name}"


def _queries(
    category: str,
    kind: str,
    names: Sequence[str],
) -> tuple[SemanticQuery, ...]:
    coordinates = {
        "panel": PANEL_COORDINATES,
        "path": PATH_COORDINATES,
        "landmark": LANDMARK_COORDINATES,
        "reference_line": REFERENCE_LINE_COORDINATES,
    }[kind]
    return tuple(SemanticQuery(category, kind, name, coordinates) for name in names)


# The skirt inventory intentionally uses an exchangeable ``skirt_panel`` role.
# Front/back are properties only when a visible construction cue (dart, slit,
# closure, or attachment) makes that distinction observable.
SEMANTIC_QUERY_INVENTORY: tuple[SemanticQuery, ...] = (
    *_queries(
        "tshirt",
        "panel",
        ("front_bodice", "back_bodice", "sleeve", "neckband"),
    ),
    *_queries(
        "tshirt",
        "path",
        (
            "front_neckline",
            "back_neckline",
            "front_shoulder",
            "back_shoulder",
            "front_armhole",
            "back_armhole",
            "front_side_seam",
            "back_side_seam",
            "front_hemline",
            "back_hemline",
            "sleeve_head",
            "sleeve_underarm",
            "sleeve_hem",
            "neckband_attachment",
        ),
    ),
    *_queries(
        "tshirt",
        "landmark",
        (
            "FNP",
            "BNP",
            "SNP_front",
            "SNP_back",
            "SP_front",
            "SP_back",
            "front_underarm",
            "back_underarm",
            "sleeve_cap_apex",
        ),
    ),
    *_queries(
        "tshirt",
        "reference_line",
        ("front_BL", "back_BL", "front_WL", "back_WL", "front_HL", "back_HL"),
    ),
    *_queries("pants", "panel", ("front_pants", "back_pants", "waistband")),
    *_queries(
        "pants",
        "path",
        (
            "front_waistline",
            "back_waistline",
            "side_seam",
            "inseam",
            "front_crotch_curve",
            "back_crotch_curve",
            "hemline",
            "front_dart_leg",
            "back_dart_leg",
            # Deprecated compatibility query.  Older provisional-block
            # adapters emitted one combined dart path; target integration
            # keeps it UNKNOWN and writes the two role-specific queries above.
            "dart_leg",
            "waistband_attachment",
        ),
    ),
    *_queries(
        "pants",
        "landmark",
        (
            "CF_waist",
            "CB_waist",
            "front_side_waist",
            "back_side_waist",
            "front_side_hip",
            "back_side_hip",
            "front_center_hip",
            "back_center_hip",
            "front_crotch_point",
            "back_crotch_point",
            "front_knee_in",
            "front_knee_out",
            "back_knee_in",
            "back_knee_out",
            "front_hem_in",
            "front_hem_out",
            "back_hem_in",
            "back_hem_out",
            "front_dart_apex",
            "back_dart_apex",
            "front_dart_leg_left",
            "front_dart_leg_right",
            "back_dart_leg_left",
            "back_dart_leg_right",
            # Deprecated compatibility query; see the path note above.
            "dart_apex",
        ),
    ),
    *_queries(
        "pants",
        "reference_line",
        (
            "front_WL",
            "back_WL",
            "front_HL",
            "back_HL",
            "front_KL",
            "back_KL",
            "front_CL",
            "back_CL",
            "front_GRAIN",
            "back_GRAIN",
        ),
    ),
    *_queries("skirt", "panel", ("skirt_panel", "waistband")),
    *_queries(
        "skirt",
        "path",
        (
            "waistline",
            "side_seam",
            "center_seam",
            "hemline",
            "front_dart_leg",
            "back_dart_leg",
            # Deprecated compatibility query.
            "dart_leg",
            "slit",
            "closure",
            "waistband_attachment",
        ),
    ),
    *_queries(
        "skirt",
        "landmark",
        (
            "front_center_waist",
            "back_center_waist",
            "front_side_waist",
            "back_side_waist",
            "front_side_hip",
            "back_side_hip",
            "front_center_hip",
            "back_center_hip",
            "front_hem_center",
            "back_hem_center",
            "front_hem_side",
            "back_hem_side",
            # Deprecated exchangeable compatibility queries.  They remain in
            # the schema so older pattern adapters can be specialized without
            # editing their source, but training targets keep them UNKNOWN.
            "center_waist",
            "side_waist",
            "side_hip",
            "hem_center",
            "hem_side",
            "front_dart_apex",
            "back_dart_apex",
            "front_dart_leg_left",
            "front_dart_leg_right",
            "back_dart_leg_left",
            "back_dart_leg_right",
            # Deprecated compatibility queries.
            "dart_apex",
            "dart_leg_left",
            "dart_leg_right",
            "slit_end",
            "closure_end",
        ),
    ),
    *_queries(
        "skirt",
        "reference_line",
        ("front_WL", "back_WL", "front_HL", "back_HL", "front_GRAIN", "back_GRAIN"),
    ),
)

SEMANTIC_QUERY_KEYS = tuple(query.key for query in SEMANTIC_QUERY_INVENTORY)
SEMANTIC_QUERY_SCHEMA_VERSION = "basic-semantic-query/v3-reference-construction-lines"
SEMANTIC_QUERY_INDEX = {key: index for index, key in enumerate(SEMANTIC_QUERY_KEYS)}
CATEGORY_QUERY_INDICES = {
    category: tuple(
        index
        for index, query in enumerate(SEMANTIC_QUERY_INVENTORY)
        if query.category == category
    )
    for category in CATEGORY_NAMES
}


def category_query_mask(category: str) -> tuple[bool, ...]:
    """Return the fixed applicability mask for one category."""

    if category not in CATEGORY_NAMES:
        raise ValueError(f"unsupported garment category: {category!r}")
    return tuple(query.category == category for query in SEMANTIC_QUERY_INVENTORY)


def query_coordinate_mask() -> tuple[tuple[bool, ...], ...]:
    """Return coordinate-channel applicability for every semantic query."""

    return tuple(
        tuple(channel < len(query.coordinate_names) for channel in range(MAX_COORDINATE_DIM))
        for query in SEMANTIC_QUERY_INVENTORY
    )


DEFAULT_SEMANTIC_TEACHER_STUDENT_CONFIG: dict[str, Any] = {
    "edge_feature_dim": 25,
    "spatial_feature_dim": 256,
    "global_feature_dim": 2048,
    "width": 128,
    "token_dim": 64,
    "heads": 4,
    "encoder_layers": 2,
    "decoder_layers": 2,
    "feedforward_multiplier": 3,
    "dropout": 0.1,
    "max_views": 4,
    # Set by the 2D training integration when primitive panel/edge role
    # supervision is available.  Zero keeps the shared query-only contract.
    "panel_role_count": 0,
    "edge_role_count": 0,
}


def _resolved_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    output = dict(DEFAULT_SEMANTIC_TEACHER_STUDENT_CONFIG)
    if config:
        output.update(config)
    integer_keys = (
        "edge_feature_dim",
        "spatial_feature_dim",
        "global_feature_dim",
        "width",
        "token_dim",
        "heads",
        "encoder_layers",
        "decoder_layers",
        "feedforward_multiplier",
        "max_views",
    )
    for key in integer_keys:
        output[key] = int(output[key])
        if output[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("panel_role_count", "edge_role_count"):
        output[key] = int(output.get(key, 0))
        if output[key] < 0:
            raise ValueError(f"{key} must be non-negative")
    output["dropout"] = float(output["dropout"])
    if not 0.0 <= output["dropout"] < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if output["width"] % output["heads"]:
        raise ValueError("width must be divisible by heads")
    return output


class ModalityContractError(ValueError):
    """Raised when vector-pattern information is passed to the visual student."""


def _install_query_decoder(module: Any, torch: Any, config: Mapping[str, Any]) -> None:
    width = int(config["width"])
    feedforward = width * int(config["feedforward_multiplier"])
    module.category_embedding = torch.nn.Embedding(len(CATEGORY_NAMES), width)
    module.semantic_queries = torch.nn.Embedding(len(SEMANTIC_QUERY_INVENTORY), width)
    decoder_layer = torch.nn.TransformerDecoderLayer(
        d_model=width,
        nhead=int(config["heads"]),
        dim_feedforward=feedforward,
        dropout=float(config["dropout"]),
        activation="gelu",
        batch_first=True,
    )
    module.semantic_decoder = torch.nn.TransformerDecoder(
        decoder_layer,
        num_layers=int(config["decoder_layers"]),
        norm=torch.nn.LayerNorm(width),
    )
    module.element_token_head = torch.nn.Sequential(
        torch.nn.LayerNorm(width),
        torch.nn.Linear(width, int(config["token_dim"])),
    )
    module.presence_head = torch.nn.Sequential(
        torch.nn.LayerNorm(int(config["token_dim"])),
        torch.nn.Linear(int(config["token_dim"]), 1),
    )
    module.coordinate_head = torch.nn.Sequential(
        torch.nn.LayerNorm(int(config["token_dim"])),
        torch.nn.Linear(int(config["token_dim"]), width),
        torch.nn.GELU(),
        torch.nn.Linear(width, MAX_COORDINATE_DIM),
    )
    torch.nn.init.normal_(module.semantic_queries.weight, std=0.02)


def _validate_category_ids(torch: Any, category_ids: Any, batch_size: int, device: Any) -> Any:
    if category_ids is None:
        raise ValueError("category_ids are required for category-specific semantic queries")
    if tuple(category_ids.shape) != (batch_size,):
        raise ValueError(f"category_ids must have shape ({batch_size},)")
    category_ids = category_ids.to(device=device, dtype=torch.long)
    if bool(torch.any(category_ids < 0)) or bool(torch.any(category_ids >= len(CATEGORY_NAMES))):
        raise ValueError(f"category_ids must be in [0, {len(CATEGORY_NAMES) - 1}]")
    return category_ids


def _decode_semantics(
    module: Any,
    torch: Any,
    memory: Any,
    memory_padding_mask: Any,
    category_ids: Any,
) -> dict[str, Any]:
    batch_size = int(memory.shape[0])
    category_ids = _validate_category_ids(torch, category_ids, batch_size, memory.device)
    category_context = module.category_embedding(category_ids)
    queries = module.semantic_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
    queries = queries + category_context.unsqueeze(1)
    decoded = module.semantic_decoder(
        queries,
        memory,
        memory_key_padding_mask=memory_padding_mask,
    )
    category_table = torch.tensor(
        tuple(category_query_mask(name) for name in CATEGORY_NAMES),
        dtype=torch.bool,
        device=memory.device,
    )
    coordinate_table = torch.tensor(
        query_coordinate_mask(), dtype=torch.bool, device=memory.device
    )
    query_mask = category_table.index_select(0, category_ids)
    coordinate_mask = coordinate_table.unsqueeze(0) & query_mask.unsqueeze(-1)
    # Presence and coordinate reconstruction are deliberately decoded *from*
    # the compact element token.  This makes the representation distilled to
    # the four-view student an explicitly supervised semantic bottleneck,
    # rather than an otherwise-random side projection of decoder state.
    element_tokens = module.element_token_head(decoded)
    return {
        "element_tokens": element_tokens,
        "presence_logits": module.presence_head(element_tokens).squeeze(-1),
        "coordinates": module.coordinate_head(element_tokens),
        "query_mask": query_mask,
        "coordinate_mask": coordinate_mask,
    }


def build_vector_graph_teacher(config: Mapping[str, Any] | None = None):
    """Build the 2D vector-pattern semantic teacher.

    ``edge_features`` has shape ``[batch, panels, edges, features]``.  The
    panel dimension is not merely flattened: masked edge pooling creates a
    panel context token and that context is added back to its member edges.
    Thus the input is a panelized vector graph rather than an unstructured
    image or one global feature vector.
    """

    import torch

    resolved = _resolved_config(config)

    class VectorGraphSemanticTeacher(torch.nn.Module):
        modality_contract = "vector_pattern_teacher"

        def __init__(self) -> None:
            super().__init__()
            width = int(resolved["width"])
            feedforward = width * int(resolved["feedforward_multiplier"])
            self.edge_projection = torch.nn.Linear(int(resolved["edge_feature_dim"]), width)
            self.panel_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(width), torch.nn.Linear(width, width)
            )
            self.memory_type_embedding = torch.nn.Parameter(torch.zeros(3, width))
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=width,
                nhead=int(resolved["heads"]),
                dim_feedforward=feedforward,
                dropout=float(resolved["dropout"]),
                activation="gelu",
                batch_first=True,
            )
            self.graph_encoder = torch.nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(resolved["encoder_layers"]),
                norm=torch.nn.LayerNorm(width),
            )
            _install_query_decoder(self, torch, resolved)
            self.panel_role_head = (
                torch.nn.Sequential(
                    torch.nn.LayerNorm(width),
                    torch.nn.Linear(width, int(resolved["panel_role_count"])),
                )
                if int(resolved["panel_role_count"])
                else None
            )
            self.edge_role_head = (
                torch.nn.Sequential(
                    torch.nn.LayerNorm(width),
                    torch.nn.Linear(width, int(resolved["edge_role_count"])),
                )
                if int(resolved["edge_role_count"])
                else None
            )
            torch.nn.init.normal_(self.memory_type_embedding, std=0.02)

        def forward(
            self,
            edge_features,
            *,
            edge_valid=None,
            panel_valid=None,
            category_ids=None,
        ):
            if edge_features.ndim != 4:
                raise ValueError(
                    "edge_features must have shape [batch, panels, edges, features]"
                )
            batch, panels, edges, feature_dim = edge_features.shape
            if feature_dim != int(resolved["edge_feature_dim"]):
                raise ValueError(
                    f"edge feature dimension {feature_dim} does not match configured "
                    f"{resolved['edge_feature_dim']}"
                )
            device = edge_features.device
            if edge_valid is None:
                edge_valid = torch.ones(
                    (batch, panels, edges), dtype=torch.bool, device=device
                )
            elif tuple(edge_valid.shape) != (batch, panels, edges):
                raise ValueError(
                    f"edge_valid must have shape {(batch, panels, edges)}"
                )
            else:
                edge_valid = edge_valid.to(device=device, dtype=torch.bool)
            if panel_valid is None:
                panel_valid = edge_valid.any(dim=-1)
            elif tuple(panel_valid.shape) != (batch, panels):
                raise ValueError(f"panel_valid must have shape {(batch, panels)}")
            else:
                panel_valid = panel_valid.to(device=device, dtype=torch.bool)
            edge_valid = edge_valid & panel_valid.unsqueeze(-1)

            projected_edges = self.edge_projection(edge_features)
            edge_weight = edge_valid.unsqueeze(-1).to(projected_edges.dtype)
            panel_sum = (projected_edges * edge_weight).sum(dim=2)
            panel_count = edge_weight.sum(dim=2).clamp_min(1.0)
            panel_tokens = self.panel_projection(panel_sum / panel_count)
            projected_edges = projected_edges + panel_tokens.unsqueeze(2)

            category_ids = _validate_category_ids(torch, category_ids, batch, device)
            category_token = self.category_embedding(category_ids).unsqueeze(1)
            category_token = category_token + self.memory_type_embedding[0]
            panel_tokens = panel_tokens + self.memory_type_embedding[1]
            projected_edges = projected_edges + self.memory_type_embedding[2]
            memory = torch.cat(
                (
                    category_token,
                    panel_tokens,
                    projected_edges.reshape(batch, panels * edges, -1),
                ),
                dim=1,
            )
            memory_valid = torch.cat(
                (
                    torch.ones((batch, 1), dtype=torch.bool, device=device),
                    panel_valid,
                    edge_valid.reshape(batch, panels * edges),
                ),
                dim=1,
            )
            memory = self.graph_encoder(memory, src_key_padding_mask=~memory_valid)
            result = _decode_semantics(self, torch, memory, ~memory_valid, category_ids)
            if self.panel_role_head is not None:
                result["panel_role_logits"] = self.panel_role_head(
                    memory[:, 1 : 1 + panels]
                )
            if self.edge_role_head is not None:
                edge_memory = memory[:, 1 + panels :].reshape(batch, panels, edges, -1)
                result["edge_role_logits"] = self.edge_role_head(edge_memory)
            return result

    return VectorGraphSemanticTeacher()


def build_four_view_semantic_student(config: Mapping[str, Any] | None = None):
    """Build the image-only student that predicts the teacher representation.

    Spatial features have shape ``[batch, views, patches, channels]`` and
    global features have shape ``[batch, views, channels]``.  Either stream may
    be omitted, but vector-pattern fields are explicitly rejected.
    """

    import torch

    resolved = _resolved_config(config)

    class FourViewSemanticStudent(torch.nn.Module):
        modality_contract = "four_view_only_no_pattern_input"

        def __init__(self) -> None:
            super().__init__()
            width = int(resolved["width"])
            feedforward = width * int(resolved["feedforward_multiplier"])
            self.spatial_projection = torch.nn.Linear(
                int(resolved["spatial_feature_dim"]), width
            )
            self.global_projection = torch.nn.Linear(
                int(resolved["global_feature_dim"]), width
            )
            self.view_embedding = torch.nn.Parameter(
                torch.zeros(int(resolved["max_views"]), width)
            )
            self.memory_type_embedding = torch.nn.Parameter(torch.zeros(3, width))
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=width,
                nhead=int(resolved["heads"]),
                dim_feedforward=feedforward,
                dropout=float(resolved["dropout"]),
                activation="gelu",
                batch_first=True,
            )
            self.visual_encoder = torch.nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(resolved["encoder_layers"]),
                norm=torch.nn.LayerNorm(width),
            )
            _install_query_decoder(self, torch, resolved)
            torch.nn.init.normal_(self.view_embedding, std=0.02)
            torch.nn.init.normal_(self.memory_type_embedding, std=0.02)

        def forward(
            self,
            *,
            category_ids=None,
            spatial_features=None,
            global_features=None,
            view_valid=None,
            pattern_graph=None,
            edge_features=None,
            panel_features=None,
        ):
            if any(
                value is not None
                for value in (pattern_graph, edge_features, panel_features)
            ):
                raise ModalityContractError(
                    "the four-view student is image-only; vector-pattern input is "
                    "permitted for the teacher during training, never for student inference"
                )
            if spatial_features is None and global_features is None:
                raise ValueError("at least one visual feature stream is required")

            reference = spatial_features if spatial_features is not None else global_features
            device = reference.device
            if spatial_features is not None:
                if spatial_features.ndim != 4:
                    raise ValueError(
                        "spatial_features must have shape [batch, views, patches, features]"
                    )
                batch, views, patches, spatial_dim = spatial_features.shape
                if spatial_dim != int(resolved["spatial_feature_dim"]):
                    raise ValueError(
                        f"spatial feature dimension {spatial_dim} does not match configured "
                        f"{resolved['spatial_feature_dim']}"
                    )
            else:
                if global_features.ndim != 3:
                    raise ValueError(
                        "global_features must have shape [batch, views, features]"
                    )
                batch, views, _ = global_features.shape
                patches = 0
            if views > int(resolved["max_views"]):
                raise ValueError(
                    f"received {views} views but max_views={resolved['max_views']}"
                )
            if global_features is not None:
                if global_features.ndim != 3:
                    raise ValueError(
                        "global_features must have shape [batch, views, features]"
                    )
                if tuple(global_features.shape[:2]) != (batch, views):
                    raise ValueError("spatial and global features must share batch/views")
                if global_features.shape[-1] != int(resolved["global_feature_dim"]):
                    raise ValueError(
                        f"global feature dimension {global_features.shape[-1]} does not match "
                        f"configured {resolved['global_feature_dim']}"
                    )
            if view_valid is None:
                view_valid = torch.ones((batch, views), dtype=torch.bool, device=device)
            elif tuple(view_valid.shape) != (batch, views):
                raise ValueError(f"view_valid must have shape {(batch, views)}")
            else:
                view_valid = view_valid.to(device=device, dtype=torch.bool)
            category_ids = _validate_category_ids(torch, category_ids, batch, device)

            memory_parts = []
            valid_parts = []
            category_token = self.category_embedding(category_ids).unsqueeze(1)
            memory_parts.append(category_token + self.memory_type_embedding[0])
            valid_parts.append(torch.ones((batch, 1), dtype=torch.bool, device=device))
            view_context = self.view_embedding[:views]
            if spatial_features is not None:
                spatial = self.spatial_projection(spatial_features)
                spatial = spatial + view_context[None, :, None, :]
                spatial = spatial + self.memory_type_embedding[1]
                memory_parts.append(spatial.reshape(batch, views * patches, -1))
                valid_parts.append(
                    view_valid.unsqueeze(-1).expand(-1, -1, patches).reshape(batch, -1)
                )
            if global_features is not None:
                global_tokens = self.global_projection(global_features)
                global_tokens = global_tokens + view_context[None, :, :]
                global_tokens = global_tokens + self.memory_type_embedding[2]
                memory_parts.append(global_tokens)
                valid_parts.append(view_valid)
            memory = torch.cat(memory_parts, dim=1)
            memory_valid = torch.cat(valid_parts, dim=1)
            memory = self.visual_encoder(memory, src_key_padding_mask=~memory_valid)
            return _decode_semantics(self, torch, memory, ~memory_valid, category_ids)

    return FourViewSemanticStudent()


def freeze_semantic_teacher(teacher: Any) -> Any:
    """Freeze a teacher explicitly and put dropout/norm layers in eval mode."""

    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def detached_teacher_forward(teacher: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run an already-frozen teacher and return graph-detached targets.

    Requiring an explicit prior ``freeze_semantic_teacher`` call prevents a
    silent failure mode where the teacher accidentally receives optimizer
    gradients during joint training.
    """

    import torch

    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError(
            "teacher parameters are trainable; call freeze_semantic_teacher first"
        )
    teacher.eval()
    with torch.no_grad():
        output = teacher(*args, **kwargs)
    return {
        key: value.detach() if torch.is_tensor(value) else value
        for key, value in output.items()
    }


def _masked_mean(torch: Any, values: Any, mask: Any) -> Any:
    mask = mask.to(device=values.device, dtype=values.dtype)
    denominator = mask.sum()
    if int(denominator.detach().item()) == 0:
        return values.sum() * 0.0
    return (values * mask).sum() / denominator


def semantic_token_reconstruction_loss(
    output: Mapping[str, Any],
    *,
    presence_targets: Any,
    coordinate_targets: Any,
    coordinate_mask: Any,
    query_mask: Any | None = None,
) -> dict[str, Any]:
    """Supervise the semantic token through its presence/geometry decoders.

    Both prediction heads consume ``element_tokens`` directly, so this loss
    provides an auditable training signal to ``element_token_head`` before its
    frozen outputs become distillation targets.  It intentionally does not
    add a free latent target whose meaning could drift away from the declared
    semantic-query schema.
    """

    import torch
    import torch.nn.functional as functional

    tokens = output["element_tokens"]
    presence_logits = output["presence_logits"]
    coordinates = output["coordinates"]
    if tokens.ndim != 3 or presence_logits.ndim != 2:
        raise ValueError("semantic outputs must have [B,Q,D] tokens and [B,Q] presence")
    batch, queries = presence_logits.shape
    expected_coordinates = (batch, queries, MAX_COORDINATE_DIM)
    if tuple(coordinates.shape) != expected_coordinates:
        raise ValueError(f"coordinates must have shape {expected_coordinates}")
    if tuple(presence_targets.shape) != (batch, queries):
        raise ValueError(f"presence_targets must have shape {(batch, queries)}")
    if tuple(coordinate_targets.shape) != expected_coordinates:
        raise ValueError(f"coordinate_targets must have shape {expected_coordinates}")
    if tuple(coordinate_mask.shape) != expected_coordinates:
        raise ValueError(f"coordinate_mask must have shape {expected_coordinates}")

    device = presence_logits.device
    presence_target = presence_targets.to(device=device, dtype=presence_logits.dtype)
    applicable = output.get("query_mask")
    if applicable is None:
        applicable = torch.ones_like(presence_logits, dtype=torch.bool)
    else:
        applicable = applicable.to(device=device, dtype=torch.bool)
    if query_mask is not None:
        if tuple(query_mask.shape) != (batch, queries):
            raise ValueError(f"query_mask must have shape {(batch, queries)}")
        applicable = applicable & query_mask.to(device=device, dtype=torch.bool)

    presence_error = functional.binary_cross_entropy_with_logits(
        presence_logits, presence_target, reduction="none"
    )
    presence_loss = _masked_mean(torch, presence_error, applicable)
    coordinate_target = coordinate_targets.to(device=device, dtype=coordinates.dtype)
    active_coordinates = (
        applicable.unsqueeze(-1)
        & (presence_target > 0.5).unsqueeze(-1)
        & coordinate_mask.to(device=device, dtype=torch.bool)
        & torch.isfinite(coordinate_target)
    )
    coordinate_error = functional.smooth_l1_loss(
        coordinates, torch.nan_to_num(coordinate_target), reduction="none"
    )
    coordinate_loss = _masked_mean(torch, coordinate_error, active_coordinates)
    return {
        "loss": presence_loss + coordinate_loss,
        "presence_loss": presence_loss,
        "coordinate_loss": coordinate_loss,
        "active_query_count": applicable.sum().detach(),
        "active_coordinate_count": active_coordinates.sum().detach(),
    }


def semantic_distillation_loss(
    student_output: Mapping[str, Any],
    teacher_output: Mapping[str, Any],
    *,
    presence_targets: Any | None = None,
    coordinate_targets: Any | None = None,
    coordinate_mask: Any | None = None,
    query_mask: Any | None = None,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compute masked token distillation, presence, and coordinate losses.

    Ground-truth presence/coordinates take precedence when supplied.  Without
    them, the detached teacher prediction becomes a soft presence target and a
    coordinate pseudo-target.  Token and coordinate losses are active only for
    applicable, present elements.  Absent or inapplicable coordinates therefore
    cannot pull the student toward arbitrary padded zeros.
    """

    import torch
    import torch.nn.functional as functional

    student_tokens = student_output["element_tokens"]
    teacher_tokens = teacher_output["element_tokens"].detach()
    student_presence = student_output["presence_logits"]
    teacher_presence = teacher_output["presence_logits"].detach()
    student_coordinates = student_output["coordinates"]
    teacher_coordinates = teacher_output["coordinates"].detach()
    if student_tokens.shape != teacher_tokens.shape:
        raise ValueError("student and teacher element token shapes must match")
    if student_presence.shape != teacher_presence.shape:
        raise ValueError("student and teacher presence shapes must match")
    if student_coordinates.shape != teacher_coordinates.shape:
        raise ValueError("student and teacher coordinate shapes must match")
    batch, queries = student_presence.shape
    expected_coordinates = (batch, queries, MAX_COORDINATE_DIM)
    if tuple(student_coordinates.shape) != expected_coordinates:
        raise ValueError(f"coordinates must have shape {expected_coordinates}")

    applicable = torch.ones_like(student_presence, dtype=torch.bool)
    for source in (student_output.get("query_mask"), teacher_output.get("query_mask"), query_mask):
        if source is None:
            continue
        if tuple(source.shape) != (batch, queries):
            raise ValueError(f"query masks must have shape {(batch, queries)}")
        applicable = applicable & source.to(device=student_presence.device, dtype=torch.bool)

    if presence_targets is None:
        presence_target = torch.sigmoid(teacher_presence)
    else:
        if tuple(presence_targets.shape) != (batch, queries):
            raise ValueError(f"presence_targets must have shape {(batch, queries)}")
        presence_target = presence_targets.to(
            device=student_presence.device, dtype=student_presence.dtype
        ).detach()
    presence_target = presence_target.clamp(0.0, 1.0)
    present = presence_target > 0.5

    token_error = (student_tokens - teacher_tokens).pow(2).mean(dim=-1)
    distillation = _masked_mean(torch, token_error, applicable & present)
    presence_error = functional.binary_cross_entropy_with_logits(
        student_presence, presence_target, reduction="none"
    )
    presence = _masked_mean(torch, presence_error, applicable)

    if coordinate_targets is None:
        coordinate_target = teacher_coordinates
    else:
        if tuple(coordinate_targets.shape) != expected_coordinates:
            raise ValueError(f"coordinate_targets must have shape {expected_coordinates}")
        coordinate_target = coordinate_targets.to(
            device=student_coordinates.device, dtype=student_coordinates.dtype
        ).detach()
    active_coordinates = applicable.unsqueeze(-1) & present.unsqueeze(-1)
    for source in (
        student_output.get("coordinate_mask"),
        teacher_output.get("coordinate_mask"),
        coordinate_mask,
    ):
        if source is None:
            continue
        if source.ndim == 2:
            if tuple(source.shape) != (batch, queries):
                raise ValueError(f"coordinate masks must have shape {(batch, queries)} or {expected_coordinates}")
            source = source.unsqueeze(-1).expand(-1, -1, MAX_COORDINATE_DIM)
        elif tuple(source.shape) != expected_coordinates:
            raise ValueError(f"coordinate masks must have shape {(batch, queries)} or {expected_coordinates}")
        active_coordinates = active_coordinates & source.to(
            device=student_coordinates.device, dtype=torch.bool
        )
    active_coordinates = active_coordinates & torch.isfinite(coordinate_target)
    coordinate_error = functional.smooth_l1_loss(
        student_coordinates,
        torch.nan_to_num(coordinate_target),
        reduction="none",
    )
    coordinate = _masked_mean(torch, coordinate_error, active_coordinates)

    loss_weights = {"distillation": 1.0, "presence": 1.0, "coordinate": 1.0}
    if weights:
        unknown = set(weights) - set(loss_weights)
        if unknown:
            raise ValueError(f"unknown semantic loss weights: {sorted(unknown)}")
        loss_weights.update({key: float(value) for key, value in weights.items()})
    total = (
        loss_weights["distillation"] * distillation
        + loss_weights["presence"] * presence
        + loss_weights["coordinate"] * coordinate
    )
    return {
        "loss": total,
        "distillation_loss": distillation,
        "presence_loss": presence,
        "coordinate_loss": coordinate,
        "active_query_count": applicable.sum().detach(),
        "active_element_count": (applicable & present).sum().detach(),
        "active_coordinate_count": active_coordinates.sum().detach(),
    }


def infer_four_view_semantics(
    student: Any,
    *,
    category_ids: Any,
    spatial_features: Any | None = None,
    global_features: Any | None = None,
    view_valid: Any | None = None,
    pattern_graph: Any | None = None,
) -> dict[str, Any]:
    """Image-only inference entry point with an explicit no-pattern contract."""

    import torch

    if pattern_graph is not None:
        raise ModalityContractError(
            "pattern_graph is forbidden at student inference; use only unseen four-view features"
        )
    if getattr(student, "modality_contract", None) != "four_view_only_no_pattern_input":
        raise ModalityContractError("the supplied model is not a four-view semantic student")
    student.eval()
    with torch.no_grad():
        return student(
            category_ids=category_ids,
            spatial_features=spatial_features,
            global_features=global_features,
            view_valid=view_valid,
        )


__all__ = [
    "CATEGORY_NAMES",
    "CATEGORY_QUERY_INDICES",
    "DEFAULT_SEMANTIC_TEACHER_STUDENT_CONFIG",
    "ELEMENT_KINDS",
    "LANDMARK_COORDINATES",
    "MAX_COORDINATE_DIM",
    "ModalityContractError",
    "PANEL_COORDINATES",
    "PATH_COORDINATES",
    "REFERENCE_LINE_COORDINATES",
    "SEMANTIC_QUERY_INDEX",
    "SEMANTIC_QUERY_INVENTORY",
    "SEMANTIC_QUERY_KEYS",
    "SEMANTIC_QUERY_SCHEMA_VERSION",
    "SemanticQuery",
    "build_four_view_semantic_student",
    "build_vector_graph_teacher",
    "category_query_mask",
    "detached_teacher_forward",
    "freeze_semantic_teacher",
    "infer_four_view_semantics",
    "query_coordinate_mask",
    "semantic_distillation_loss",
    "semantic_token_reconstruction_loss",
]
