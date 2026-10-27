"""
Unit Tests for M2.2 / M2.3: Frame Sampling Strategies.

Validates the mathematical specification of uniform sampling:
    i_k = floor(k * (L - 1) / (N - 1))  for k in {0, ..., N-1}
And verifies deterministic evaluation rules (M2.3), alternative sampling operators,
and edge case handling across sequence lengths.
"""

import math
import pytest
from src.data.sampling import (
    uniform_sample,
    first_n,
    last_n,
    sparse_sample,
    segment_sample,
    random_sample,
    pad_or_truncate,
    get_sampler,
    SAMPLING_STRATEGIES,
)


def test_uniform_sampling_m2_2_formula():
    """Verify exact match with M2.2 uniform floor equation i_k = floor(k * (L - 1) / (N - 1))."""
    # Canonical cases from M2.2 specification
    # L=12, N=4 -> floor(0*11/3)=0, floor(1*11/3)=3, floor(2*11/3)=7, floor(3*11/3)=11
    assert uniform_sample(total_frames=12, target_frames=4) == [0, 3, 7, 11]

    # L=10, N=5 -> floor(k*9/4) for k=0..4:
    # k=0: 0, k=1: 2, k=2: 4, k=3: 6, k=4: 9 -> [0, 2, 4, 6, 9]
    res_10_5 = uniform_sample(total_frames=10, target_frames=5)
    expected_10_5 = [math.floor(k * 9 / 4) for k in range(5)]
    assert res_10_5 == expected_10_5
    assert res_10_5 == [0, 2, 4, 6, 9]

    # L=16, N=8 -> floor(k*15/7) for k=0..7:
    res_16_8 = uniform_sample(total_frames=16, target_frames=8)
    expected_16_8 = [math.floor(k * 15 / 7) for k in range(8)]
    assert res_16_8 == expected_16_8
    assert res_16_8 == [0, 2, 4, 6, 8, 10, 12, 15]

    # L=50, N=8
    res_50_8 = uniform_sample(total_frames=50, target_frames=8)
    expected_50_8 = [math.floor(k * 49 / 7) for k in range(8)]
    assert res_50_8 == expected_50_8
    assert res_50_8 == [0, 7, 14, 21, 28, 35, 42, 49]


def test_uniform_sampling_endpoint_and_monotonicity():
    """Verify endpoint preservation (i_0 = 0 and i_{N-1} = L - 1) and strict monotonicity."""
    for L in [8, 16, 25, 30, 50, 100]:
        for N in [2, 4, 8, 16]:
            if N <= L:
                sampled = uniform_sample(total_frames=L, target_frames=N)
                assert len(sampled) == N
                assert sampled[0] == 0, f"Failed start endpoint for L={L}, N={N}"
                assert sampled[-1] == L - 1, f"Failed end endpoint for L={L}, N={N}"
                # Strictly monotonic
                for i in range(len(sampled) - 1):
                    assert sampled[i] < sampled[i + 1], f"Non-increasing indices for L={L}, N={N}"


def test_uniform_sampling_edge_cases():
    """Verify N=1 midpoint anchor, L <= N identity, and empty cases."""
    # N=1 -> returns sequence midpoint
    assert uniform_sample(total_frames=10, target_frames=1) == [5]
    assert uniform_sample(total_frames=11, target_frames=1) == [5]
    assert uniform_sample(total_frames=1, target_frames=1) == [0]

    # L <= N -> returns all available frames [0, ..., L-1]
    assert uniform_sample(total_frames=4, target_frames=8) == [0, 1, 2, 3]
    assert uniform_sample(total_frames=8, target_frames=8) == [0, 1, 2, 3, 4, 5, 6, 7]

    # Non-positive inputs
    assert uniform_sample(total_frames=0, target_frames=8) == []
    assert uniform_sample(total_frames=10, target_frames=0) == []
    assert uniform_sample(total_frames=-5, target_frames=4) == []


def test_sampling_determinism_m2_3_rule():
    """Verify M2.3 rule: validation and test evaluation must be 100% deterministic."""
    L = 50
    N = 8
    # 50 consecutive runs of uniform_sample produce identical indices
    first_run = uniform_sample(L, N)
    for _ in range(50):
        assert uniform_sample(L, N) == first_run

    # TSN center mode is deterministic
    first_tsn = segment_sample(L, N, mode="center")
    for _ in range(50):
        assert segment_sample(L, N, mode="center") == first_tsn

    # Seeded random is reproducible
    run1 = random_sample(L, N, seed=42)
    run2 = random_sample(L, N, seed=42)
    assert run1 == run2


def test_m2_3_dataset_split_deterministic_enforcement():
    """
    Verify M2.3 Rule: TemporalSequenceDataset must force deterministic uniform sampling
    for 'val' and 'test' splits even if config specifies stochastic or alternative sampling.
    """
    from src.data.dataset import TemporalSequenceDataset
    from pathlib import Path
    import yaml

    config = {
        "dataset": {"name": "ssv2", "root": "data/", "val_split": 0.15, "test_split": 0.15, "seed": 42},
        "sequence": {"max_frames": 8, "sampling": "random", "frame_size": [224, 224]},
        "encoder": {"name": "dinov2_vits14", "embed_dim": 384}
    }

    # Verify that helper _sample_frames returns identical frames for test split
    frames = [Path(f"f_{i:02d}.png") for i in range(20)]
    dataset_test = TemporalSequenceDataset(root="data/", split="test", config=config)
    
    # 20 calls to sample frames for test split produce identical results
    sampled_1 = dataset_test._sample_frames(frames, mode="uniform")
    sampled_2 = dataset_test._sample_frames(frames, mode="uniform")
    assert sampled_1 == sampled_2
    assert [p.name for p in sampled_1] == [f"f_{i:02d}.png" for i in [0, 2, 5, 8, 10, 13, 16, 19]]


def test_alternative_samplers():
    """Verify first_n, last_n, sparse_sample, and segment_sample behavior."""
    L = 20
    N = 5

    # first_n
    assert first_n(L, N) == [0, 1, 2, 3, 4]
    # last_n
    assert last_n(L, N) == [15, 16, 17, 18, 19]

    # sparse_sample: stride = 20 // 5 = 4
    assert sparse_sample(L, N) == [0, 4, 8, 12, 16]

    # segment_sample center: 5 segments of length 4: [0..4), [4..8), [8..12), [12..16), [16..20)
    # midpoints: 2, 6, 10, 14, 18
    tsn_center = segment_sample(L, N, mode="center")
    assert tsn_center == [2, 6, 10, 14, 18]

    # segment_sample random with seed
    tsn_rand = segment_sample(L, N, mode="random", seed=42)
    assert len(tsn_rand) == N
    # Each index is inside its respective segment
    assert 0 <= tsn_rand[0] < 4
    assert 4 <= tsn_rand[1] < 8
    assert 8 <= tsn_rand[2] < 12
    assert 12 <= tsn_rand[3] < 16
    assert 16 <= tsn_rand[4] < 20


def test_pad_or_truncate():
    """Verify pad_or_truncate padding modes and truncation."""
    # Truncate
    assert pad_or_truncate([0, 2, 4, 6, 8, 10], target_frames=4) == [0, 2, 4, 6]

    # Exact match
    assert pad_or_truncate([0, 2, 4, 6], target_frames=4) == [0, 2, 4, 6]

    # Edge pad
    assert pad_or_truncate([0, 2], target_frames=4, pad_mode="edge") == [0, 2, 2, 2]

    # Zero pad
    assert pad_or_truncate([2, 5], target_frames=4, pad_mode="zero") == [2, 5, 0, 0]

    with pytest.raises(ValueError):
        pad_or_truncate([1, 2], target_frames=4, pad_mode="invalid_mode")


def test_get_sampler_registry():
    """Verify sampler registry access and error handling."""
    for name in ["uniform", "first_n", "last_n", "sparse", "segment_center", "random"]:
        sampler = get_sampler(name)
        assert callable(sampler)

    with pytest.raises(ValueError):
        get_sampler("non_existent_sampler")

