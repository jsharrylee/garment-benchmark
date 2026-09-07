"""Public-safe utilities for cyclic L/Q/C/A panel-boundary signatures."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


PRIMITIVES = frozenset("LQCA")


def _validated_sequence(primitives: Iterable[str]) -> tuple[str, ...]:
    sequence = tuple(primitives)
    if not sequence:
        raise ValueError("a panel boundary must contain at least one primitive")
    invalid = sorted(set(sequence) - PRIMITIVES)
    if invalid:
        raise ValueError(f"unsupported primitive types: {invalid}")
    return sequence


def canonical_cycle(primitives: Iterable[str]) -> tuple[str, ...]:
    """Return a cycle invariant to its start edge and traversal reversal.

    Only primitive types are represented. Lengths, control points, arc
    parameters, seams, and semantic roles are intentionally outside this
    signature.
    """

    sequence = _validated_sequence(primitives)
    n = len(sequence)
    forward = [sequence[i:] + sequence[:i] for i in range(n)]
    reversed_sequence = tuple(reversed(sequence))
    backward = [
        reversed_sequence[i:] + reversed_sequence[:i] for i in range(n)
    ]
    return min(forward + backward)


def make_signature(primitives: Iterable[str]) -> str:
    cycle = canonical_cycle(primitives)
    return f"{len(cycle)}:{''.join(cycle)}"


def parse_signature(signature: str) -> tuple[str, ...]:
    try:
        count_text, sequence_text = signature.split(":", 1)
        count = int(count_text)
    except (ValueError, AttributeError) as error:
        raise ValueError(f"invalid signature: {signature!r}") from error
    sequence = _validated_sequence(sequence_text)
    if count != len(sequence):
        raise ValueError(
            f"declared edge count {count} does not match {len(sequence)}"
        )
    canonical = canonical_cycle(sequence)
    if sequence != canonical:
        raise ValueError("signature sequence is not in canonical cycle form")
    return sequence


def edit_distance(left: Sequence[str] | str, right: Sequence[str] | str) -> int:
    """Levenshtein distance used for descriptive near-confusion analysis."""

    a = tuple(left)
    b = tuple(right)
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (token_a != token_b),
                )
            )
        previous = current
    return previous[-1]
