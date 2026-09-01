"""Leakage-resistant body/design splits for the FreeSewing Teagan corpus."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Iterable

from .tshirt_schema import TShirtTraceRecord


TRAIN_BODY_MODELS = tuple(
    [f"cisFemaleAdult{size}" for size in range(28, 40, 2)]
    + [f"cisMaleAdult{size}" for size in range(32, 44, 2)]
)
VALIDATION_BODY_MODELS = ("cisFemaleAdult40", "cisFemaleAdult42", "cisMaleAdult44", "cisMaleAdult46")
TEST_BODY_MODELS = ("cisFemaleAdult44", "cisFemaleAdult46", "cisMaleAdult48", "cisMaleAdult50")

TRAIN_DESIGNS = ("default", "fitted_short")
VALIDATION_DESIGNS = ("loose_long",)
TEST_DESIGNS = ("wide_deep_neck",)

EXPECTED_SPLIT_COUNTS = {
    "train": 24,
    "validation_body": 8,
    "validation_design": 12,
    "validation_double": 4,
    "test_body": 8,
    "test_design": 12,
    "test_double": 4,
    "test_design_on_validation_body": 4,
    "test_body_on_validation_design": 4,
}


def parse_teagan_sample_id(sample_id: str) -> tuple[str, str]:
    parts = sample_id.split("__")
    if len(parts) != 3 or parts[0] != "freesewing_teagan":
        raise ValueError(f"not a FreeSewing Teagan sample id: {sample_id}")
    return parts[1], parts[2]


def teagan_training_split(sample_id: str) -> str:
    body_model, design = parse_teagan_sample_id(sample_id)
    if body_model in TRAIN_BODY_MODELS:
        body_group = "train"
    elif body_model in VALIDATION_BODY_MODELS:
        body_group = "validation"
    elif body_model in TEST_BODY_MODELS:
        body_group = "test"
    else:
        raise ValueError(f"unassigned Teagan body model: {body_model}")

    if design in TRAIN_DESIGNS:
        design_group = "train"
    elif design in VALIDATION_DESIGNS:
        design_group = "validation"
    elif design in TEST_DESIGNS:
        design_group = "test"
    else:
        raise ValueError(f"unassigned Teagan design variant: {design}")

    if body_group == "train" and design_group == "train":
        return "train"
    if body_group == "validation" and design_group == "train":
        return "validation_body"
    if body_group == "train" and design_group == "validation":
        return "validation_design"
    if body_group == "validation" and design_group == "validation":
        return "validation_double"
    if body_group == "test" and design_group == "train":
        return "test_body"
    if body_group == "train" and design_group == "test":
        return "test_design"
    if body_group == "test" and design_group == "test":
        return "test_double"
    if body_group == "validation" and design_group == "test":
        return "test_design_on_validation_body"
    if body_group == "test" and design_group == "validation":
        return "test_body_on_validation_design"
    raise ValueError(f"unsupported Teagan body/design cross: {body_model}/{design}")


def repartition_teagan_records(records: Iterable[TShirtTraceRecord]) -> tuple[TShirtTraceRecord, ...]:
    output = tuple(replace(record, split=teagan_training_split(record.sample_id)) for record in records)
    observed = Counter(record.split for record in output)
    if dict(observed) != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"unexpected Teagan split cardinalities: {dict(observed)}")
    if len({record.sample_id for record in output}) != len(output):
        raise ValueError("duplicate Teagan sample id")
    return output


__all__ = [
    "EXPECTED_SPLIT_COUNTS",
    "TEST_BODY_MODELS",
    "TEST_DESIGNS",
    "TRAIN_BODY_MODELS",
    "TRAIN_DESIGNS",
    "VALIDATION_BODY_MODELS",
    "VALIDATION_DESIGNS",
    "parse_teagan_sample_id",
    "repartition_teagan_records",
    "teagan_training_split",
]
