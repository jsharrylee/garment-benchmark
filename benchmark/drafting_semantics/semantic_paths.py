"""Label-leakage-free reconstruction of semantic paths from edge predictions.

Vector CAD files commonly serialize one visually continuous seam as several
line/Bezier/arc primitives.  The model must not use the target role to merge
those primitives before classification.  This module therefore keeps every
primitive as an input token and joins *predicted* adjacent roles afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PredictedSemanticPath:
    """One cyclically contiguous run of predicted primitive-edge roles."""

    role: str
    edge_indices: tuple[int, ...]
    edge_ids: tuple[str, ...]
    primitive_count: int
    length_cm: float


def merge_predicted_semantic_paths(
    predicted_roles: Sequence[str],
    *,
    same_path_links: Sequence[bool] | None = None,
    edge_ids: Sequence[str] | None = None,
    edge_lengths_cm: Sequence[float] | None = None,
    closed_boundary: bool = True,
) -> tuple[PredictedSemanticPath, ...]:
    """Merge adjacent equal *predictions* without altering source geometry.

    The return value is a semantic view of the boundary.  Child primitive ids,
    order, and lengths are retained, so a two-Bezier armhole is exposed as one
    ``armhole`` path while rendering and exact length calculations still use
    both original curves.
    """

    roles = tuple(str(value) for value in predicted_roles)
    count = len(roles)
    if edge_ids is None:
        ids = tuple(f"edge_{index}" for index in range(count))
    else:
        ids = tuple(str(value) for value in edge_ids)
    if edge_lengths_cm is None:
        lengths = (0.0,) * count
    else:
        lengths = tuple(float(value) for value in edge_lengths_cm)
    if same_path_links is None:
        links = tuple(
            roles[index] == roles[(index + 1) % count]
            for index in range(count)
        )
    else:
        links = tuple(bool(value) for value in same_path_links)
    if len(ids) != count or len(lengths) != count or len(links) != count:
        raise ValueError(
            "roles, same_path_links, edge_ids, and edge_lengths_cm must have equal lengths"
        )
    if not count:
        return ()

    runs: list[tuple[str, list[int]]] = []
    for index, role in enumerate(roles):
        # Link i-1 says whether primitive i-1 and primitive i belong to one
        # semantic path.  Requiring the predicted role to agree prevents a
        # spurious positive link from joining two different seam types.
        if runs and runs[-1][0] == role and links[index - 1]:
            runs[-1][1].append(index)
        else:
            runs.append((role, [index]))

    # A closed panel may start in the middle of a semantic path.  Join the
    # first/last runs after classification so the arbitrary serialization start
    # does not split that path.
    if closed_boundary and len(runs) > 1 and runs[0][0] == runs[-1][0] and links[-1]:
        role = runs[0][0]
        wrapped = runs[-1][1] + runs[0][1]
        runs = [(role, wrapped), *runs[1:-1]]

    return tuple(
        PredictedSemanticPath(
            role=role,
            edge_indices=tuple(indices),
            edge_ids=tuple(ids[index] for index in indices),
            primitive_count=len(indices),
            length_cm=sum(lengths[index] for index in indices),
        )
        for role, indices in runs
    )


__all__ = ["PredictedSemanticPath", "merge_predicted_semantic_paths"]
