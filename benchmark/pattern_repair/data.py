from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RepairPair:
    corrupted: np.ndarray
    clean: np.ndarray
    corruption: str


def normalize_loop(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(points, dtype=np.float32)
    center = values.mean(axis=0)
    span = np.ptp(values, axis=0)
    scale = float(max(span.max(), 1e-6))
    return (values - center) / scale, center, scale


def loop_features(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    previous = np.roll(values, 1, axis=0)
    following = np.roll(values, -1, axis=0)
    prev_delta = values - previous
    next_delta = following - values
    radial = np.linalg.norm(values, axis=1, keepdims=True)
    positions = np.arange(len(values), dtype=np.float32) / max(len(values), 1)
    phase = 2.0 * np.pi * positions
    turn = (prev_delta[:, 0] * next_delta[:, 1] - prev_delta[:, 1] * next_delta[:, 0])[:, None]
    return np.concatenate(
        (
            values,
            prev_delta,
            next_delta,
            radial,
            np.sin(phase)[:, None],
            np.cos(phase)[:, None],
            turn,
        ),
        axis=1,
    ).astype(np.float32)


def generate_clean_loop(rng: np.random.Generator, minimum_nodes: int = 16, maximum_nodes: int = 96) -> np.ndarray:
    count = int(rng.integers(minimum_nodes, maximum_nodes + 1))
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False, dtype=np.float32)
    radius = np.ones(count, dtype=np.float32)
    for harmonic in range(1, 5):
        amplitude = float(rng.uniform(-0.13, 0.13)) / harmonic**0.5
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        radius += amplitude * np.cos(harmonic * theta + phase)
    radius = np.maximum(radius, 0.35)
    points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    points *= np.array([rng.uniform(0.45, 1.5), rng.uniform(0.45, 1.5)], dtype=np.float32)
    shear = float(rng.uniform(-0.25, 0.25))
    points[:, 0] += shear * points[:, 1]
    rotation = float(rng.uniform(-np.pi, np.pi))
    matrix = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]], dtype=np.float32)
    points = points @ matrix.T
    points += rng.normal(0.0, 0.015, points.shape).astype(np.float32)
    normalized, _, _ = normalize_loop(points)
    return normalized


def corrupt_loop(clean: np.ndarray, rng: np.random.Generator) -> RepairPair:
    values = np.asarray(clean, dtype=np.float32).copy()
    count = len(values)
    corruption = str(rng.choice(("segment_reverse", "cross_swap", "inward_spike", "mixed_noise")))
    if corruption == "segment_reverse":
        length = int(rng.integers(max(3, count // 8), max(4, count // 3)))
        start = int(rng.integers(1, count - length))
        values[start : start + length] = values[start : start + length][::-1]
    elif corruption == "cross_swap":
        first = int(rng.integers(1, max(2, count // 3)))
        second = int(rng.integers(max(first + 2, count // 2), count - 1))
        values[[first, second]] = values[[second, first]]
    elif corruption == "inward_spike":
        index = int(rng.integers(1, count - 1))
        width = int(rng.integers(1, max(2, count // 12)))
        indices = np.arange(index - width, index + width + 1) % count
        values[indices] *= rng.uniform(-0.45, 0.15)
    else:
        values += rng.normal(0.0, 0.09, values.shape).astype(np.float32)
        index = int(rng.integers(0, count))
        values[index] += rng.normal(0.0, 0.65, 2).astype(np.float32)
    values += rng.normal(0.0, 0.018, values.shape).astype(np.float32)
    return RepairPair(values, np.asarray(clean, dtype=np.float32), corruption)


def synthetic_batch(
    rng: np.random.Generator,
    batch_size: int,
    *,
    maximum_nodes: int,
    minimum_nodes: int = 16,
    clean_pool: tuple[np.ndarray, ...] = (),
    clean_pool_probability: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.zeros((batch_size, maximum_nodes, 10), dtype=np.float32)
    targets = np.zeros((batch_size, maximum_nodes, 2), dtype=np.float32)
    mask = np.zeros((batch_size, maximum_nodes), dtype=bool)
    for batch_index in range(batch_size):
        if clean_pool and rng.random() < clean_pool_probability:
            clean = np.asarray(clean_pool[int(rng.integers(0, len(clean_pool)))], dtype=np.float32)
        else:
            clean = generate_clean_loop(rng, minimum_nodes, maximum_nodes)
        pair = corrupt_loop(clean, rng)
        count = len(clean)
        features[batch_index, :count] = loop_features(pair.corrupted)
        targets[batch_index, :count] = pair.clean
        mask[batch_index, :count] = True
    return features, targets, mask


def strict_self_intersections(points: np.ndarray, epsilon: float = 1e-8) -> int:
    values = np.asarray(points, dtype=float)
    segments = list(zip(values, np.roll(values, -1, axis=0), strict=True))

    def orientation(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    total = 0
    for first_index, (a, b) in enumerate(segments):
        for second_index in range(first_index + 1, len(segments)):
            if abs(first_index - second_index) <= 1 or {first_index, second_index} == {0, len(segments) - 1}:
                continue
            c, d = segments[second_index]
            total += int(orientation(a, b, c) * orientation(a, b, d) < -epsilon and orientation(c, d, a) * orientation(c, d, b) < -epsilon)
    return total
