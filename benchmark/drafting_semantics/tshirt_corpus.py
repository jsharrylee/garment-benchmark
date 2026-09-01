"""Deterministic body/design/split plan for the basic T-shirt trace corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .tshirt_garmentcode import default_garmentcode_root, load_body_yaml


@dataclass(frozen=True)
class BodyCase:
    id: str
    family: str
    values: dict[str, float]
    anchor_ids: tuple[str, ...]


@dataclass(frozen=True)
class DesignCase:
    id: str
    family: str
    values: dict[str, float]


@dataclass(frozen=True)
class TShirtSampleCase:
    sample_id: str
    split: str
    body: BodyCase
    design: DesignCase


def _radical_inverse(index: int, base: int) -> float:
    value = 0.0
    factor = 1.0 / base
    while index:
        index, remainder = divmod(index, base)
        value += remainder * factor
        factor /= base
    return value


def _lerp(first: float, second: float, amount: float) -> float:
    return (1.0 - amount) * first + amount * second


def _scaled_body(
    first: Mapping[str, float],
    second: Mapping[str, float],
    amount: float,
    *,
    global_scale: float,
    girth_scale: float = 1.0,
    shoulder_scale: float = 1.0,
) -> dict[str, float]:
    shared = sorted(set(first) & set(second))
    output = {name: _lerp(float(first[name]), float(second[name]), amount) for name in shared}
    # Every source field is a length except the two angles.  Scaling all
    # lengths together keeps the official profiles internally correlated.
    for name in output:
        if name not in {"arm_pose_angle", "hip_inclination", "shoulder_incl"}:
            output[name] *= global_scale
    for name in (
        "bust",
        "underbust",
        "waist",
        "hips",
        "leg_circ",
        "back_width",
        "waist_back_width",
        "hip_back_width",
        "bust_points",
        "bum_points",
    ):
        if name in output:
            output[name] *= girth_scale
    for name in ("shoulder_w", "back_width"):
        if name in output:
            output[name] *= shoulder_scale
    # Guard only the physical constraints used by GarmentCode; these are not
    # claimed as population-statistical samples.
    output["waist"] = min(output["waist"], output["bust"] * 1.08)
    output["back_width"] = min(output["back_width"], output["bust"] * 0.58)
    output["waist_back_width"] = min(output["waist_back_width"], output["waist"] * 0.58)
    output["shoulder_w"] = min(output["shoulder_w"], output["back_width"] * 0.94)
    output["head_l"] = min(output["head_l"], output["height"] * 0.19)
    return {name: round(value, 6) for name, value in output.items()}


def build_body_cases(garmentcode_root: Path | None = None) -> tuple[BodyCase, ...]:
    root = (garmentcode_root or default_garmentcode_root()).resolve()
    paths = sorted((root / "assets/bodies").glob("*.yaml"))
    anchors = [(path.stem, load_body_yaml(path)) for path in paths]
    if len(anchors) < 6:
        raise ValueError("the body split plan requires the six official GarmentCode presets")

    output: list[BodyCase] = []
    families = (("central", 64), ("validation_body", 8), ("test_body", 16))
    seed_offset = 0
    for family, count in families:
        for local_index in range(count):
            seed = seed_offset + local_index + 1
            first_index = (seed * 5 + 1) % len(anchors)
            second_index = (seed * 7 + 3) % len(anchors)
            if second_index == first_index:
                second_index = (second_index + 1) % len(anchors)
            amount = 0.1 + 0.8 * _radical_inverse(seed, 2)
            scale = 0.95 + 0.10 * _radical_inverse(seed, 3)
            first_id, first = anchors[first_index]
            second_id, second = anchors[second_index]
            output.append(
                BodyCase(
                    id=f"{family}_{local_index:03d}",
                    family=family,
                    values=_scaled_body(first, second, amount, global_scale=scale),
                    anchor_ids=(first_id, second_id),
                )
            )
        seed_offset += count

    # Parametric extrapolation is isolated from all fitting and threshold
    # tuning.  It tests range generalization, not real population coverage.
    for local_index in range(16):
        first_id, first = anchors[local_index % len(anchors)]
        second_id, second = anchors[(local_index * 3 + 2) % len(anchors)]
        direction = -1.0 if local_index % 2 == 0 else 1.0
        # Keep every OOD case outside the central global-size band while adding
        # deterministic, bounded low-discrepancy variation.  The former fixed
        # factors repeated every 12 cases because both the anchor pairing and
        # all scale switches had returned to the same state.
        blend_jitter = _radical_inverse(local_index + 1, 5) - 0.5
        global_jitter = _radical_inverse(local_index + 1, 23) - 0.5
        girth_jitter = _radical_inverse(local_index + 1, 13) - 0.5
        shoulder_jitter = _radical_inverse(local_index + 1, 17) - 0.5
        blend_center = 0.25 if direction < 0 else 0.75
        global_base = 0.86 if direction < 0 else 1.14
        girth_base = 0.93 if (local_index // 2) % 2 == 0 else 1.08
        shoulder_base = 0.94 if local_index % 4 < 2 else 1.06
        output.append(
            BodyCase(
                id=f"ood_body_{local_index:03d}",
                family="ood_body",
                values=_scaled_body(
                    first,
                    second,
                    blend_center + 0.08 * blend_jitter,
                    global_scale=global_base * (1.0 + 0.012 * global_jitter),
                    girth_scale=girth_base * (1.0 + 0.01 * girth_jitter),
                    shoulder_scale=shoulder_base * (1.0 + 0.01 * shoulder_jitter),
                ),
                anchor_ids=(first_id, second_id),
            )
        )
    numeric_bodies = {tuple(sorted(case.values.items())) for case in output}
    if len(numeric_bodies) != len(output):
        raise AssertionError("body split plan contains numerically duplicate cases")
    return tuple(output)


def _design_from_index(index: int, *, family: str) -> dict[str, float]:
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
    unit = [_radical_inverse(index + 1, base) for base in primes]
    bounds = {
        "shirt_width": (1.00, 1.24),
        "shirt_flare": (0.86, 1.25),
        "shirt_length": (0.95, 1.72),
        "neck_width": (-0.15, 0.72),
        "front_neck_depth": (0.25, 1.25),
        "back_neck_depth": (0.00, 0.55),
        "sleeve_length": (0.15, 0.95),
        "armhole_depth": (0.05, 0.78),
        "sleeve_end_width": (0.72, 1.35),
        "sleeve_angle": (14.0, 40.0),
        "armhole_smoothing": (0.15, 0.35),
    }
    if family == "ood_design":
        # Controlled block extrapolations, rather than arbitrary corners of
        # the 11-dimensional design box.  Parameters outside the training
        # envelope are changed only within one interpretable block while all
        # other values stay at the same neutral, in-envelope reference.  The
        # values also remain inside the ranges declared by GarmentCode's
        # official t-shirt.yaml.  This makes an unseen-design failure easier
        # to attribute than the former all-at-once extremes.
        #
        #   0: compact/narrow silhouette below the fitted design envelope
        #   1: relaxed/long silhouette above the fitted design envelope
        #   2: wider/deeper neckline above the fitted design envelope
        #   3: longer/deeper/wider sleeve above the fitted design envelope
        neutral = {
            "shirt_width": 1.12,
            "shirt_flare": 1.04,
            "shirt_length": 1.30,
            "neck_width": 0.24,
            "front_neck_depth": 0.70,
            "back_neck_depth": 0.25,
            "sleeve_length": 0.55,
            "armhole_depth": 0.38,
            "sleeve_end_width": 0.95,
            "sleeve_angle": 22.0,
            "armhole_smoothing": 0.21,
        }
        extremes = (
            {
                **neutral,
                "shirt_width": 1.00,
                "shirt_flare": 0.82,
                "shirt_length": 0.90,
            },
            {
                **neutral,
                "shirt_width": 1.27,
                "shirt_flare": 1.30,
                "shirt_length": 1.78,
            },
            {
                **neutral,
                "neck_width": 0.80,
                "front_neck_depth": 1.35,
                "back_neck_depth": 0.65,
            },
            {
                **neutral,
                "sleeve_length": 1.02,
                "armhole_depth": 0.85,
                "sleeve_end_width": 1.30,
                "sleeve_angle": 34.0,
                "armhole_smoothing": 0.30,
            },
        )
        return dict(extremes[index % len(extremes)])
    return {
        name: round(_lerp(low, high, fraction), 6)
        for (name, (low, high)), fraction in zip(bounds.items(), unit)
    }


def build_design_cases() -> tuple[DesignCase, ...]:
    output: list[DesignCase] = []
    offset = 0
    for family, count in (
        ("train_design", 16),
        ("validation_design", 4),
        ("test_design", 4),
        ("ood_design", 4),
    ):
        for local_index in range(count):
            output.append(
                DesignCase(
                    id=f"{family}_{local_index:02d}",
                    family=family,
                    values=_design_from_index(offset + local_index if family != "ood_design" else local_index, family=family),
                )
            )
        offset += count
    return tuple(output)


def build_sample_plan(garmentcode_root: Path | None = None) -> tuple[TShirtSampleCase, ...]:
    bodies = build_body_cases(garmentcode_root)
    designs = build_design_cases()
    by_body = {family: [case for case in bodies if case.family == family] for family in {case.family for case in bodies}}
    by_design = {
        family: [case for case in designs if case.family == family] for family in {case.family for case in designs}
    }
    central = by_body["central"]
    train_design = by_design["train_design"]
    output: list[TShirtSampleCase] = []

    def add(body_cases: Iterable[BodyCase], design_cases: Iterable[DesignCase], split: str) -> None:
        for body in body_cases:
            for design in design_cases:
                output.append(
                    TShirtSampleCase(
                        sample_id=f"gc_tshirt__{body.id}__{design.id}", split=split, body=body, design=design
                    )
                )

    for body_index, body in enumerate(central):
        for design_index, design in enumerate(train_design):
            fold = (5 * body_index + 3 * design_index) % 8
            split = "iid_test" if fold == 0 else "iid_validation" if fold == 1 else "train"
            output.append(
                TShirtSampleCase(
                    sample_id=f"gc_tshirt__{body.id}__{design.id}", split=split, body=body, design=design
                )
            )
    add(by_body["validation_body"], train_design, "validation_body")
    add(central, by_design["validation_design"], "validation_design")
    add(by_body["validation_body"], by_design["validation_design"], "validation_double")
    add(by_body["test_body"], train_design, "test_body")
    add(central, by_design["test_design"], "test_design")
    add(by_body["test_body"], by_design["test_design"], "test_double")
    add(by_body["ood_body"], train_design, "unseen_body")
    add(central, by_design["ood_design"], "unseen_design")
    add(by_body["ood_body"], by_design["ood_design"], "unseen_double")
    if len(output) != 2592:
        raise AssertionError(f"unexpected corpus size: {len(output)}")
    return tuple(output)


def split_counts(plan: Iterable[TShirtSampleCase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in plan:
        counts[sample.split] = counts.get(sample.split, 0) + 1
    return dict(sorted(counts.items()))


def plan_digest(plan: Iterable[TShirtSampleCase]) -> str:
    rows = [
        {
            "sample_id": sample.sample_id,
            "split": sample.split,
            "body": sample.body.values,
            "design": sample.design.values,
        }
        for sample in plan
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "BodyCase",
    "DesignCase",
    "TShirtSampleCase",
    "build_body_cases",
    "build_design_cases",
    "build_sample_plan",
    "split_counts",
    "plan_digest",
]
