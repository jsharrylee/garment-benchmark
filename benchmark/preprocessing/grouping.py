from __future__ import annotations

from benchmark.adapters.synbody import SynBodyBundle


def rank_candidate_bundles(bundles: list[SynBodyBundle]) -> list[SynBodyBundle]:
    """Prefer one synchronized frame per sequence, spread across scenes."""
    seen: set[tuple[str, str]] = set()
    result: list[SynBodyBundle] = []
    for bundle in bundles:
        key = (bundle.scene, bundle.sequence)
        if key not in seen:
            seen.add(key)
            result.append(bundle)
    return result
