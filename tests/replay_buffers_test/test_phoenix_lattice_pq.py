import math
import random

from pfrl.replay_buffers.phoenix_lattice_pq import (
    PHOENIX_LATTICE_ANCHORS_Q32,
    CustodianDeduplicatorPq,
    MarineSalienceGatePq,
    PhoenixInterferenceState,
    SumTreePqQ32,
    classify_phoenix_interference_q32,
    dequantize_obs_bfp64,
    quantize_obs_bfp64,
    restore_observation_dropouts_catmull_rom_pq,
)


def test_sum_tree_pq_q32_zero_drift():
    tree = SumTreePqQ32(capacity=16)
    rng = random.Random(42)

    expected_q32 = []
    for _ in range(100):
        val_q32 = rng.randint(1000, 1 << 32)
        if len(expected_q32) == 16:
            expected_q32.pop(0)
        expected_q32.append(val_q32)
        tree.append_q32(val_q32)
        assert tree.sum_q32() == sum(expected_q32)
        assert tree.min_q32() == min(expected_q32)

    # Perform 200 random updates and verify bit-exact 0 drift
    for _ in range(200):
        idx = rng.randint(0, len(expected_q32) - 1)
        new_val = rng.randint(1000, 1 << 32)
        expected_q32[idx] = new_val
        tree.set_q32(idx, new_val)
        assert tree.sum_q32() == sum(expected_q32)
        assert tree.min_q32() == min(expected_q32)

    ixs, vals = tree.prioritized_sample(8, remove=False)
    assert len(set(ixs)) == 8
    assert len(vals) == 8
    assert tree.sum_q32() == sum(expected_q32)


def test_phoenix_interference_and_marine_salience_gate():
    states = [classify_phoenix_interference_q32(a) for a in PHOENIX_LATTICE_ANCHORS_Q32]
    assert states == [
        PhoenixInterferenceState.BOUND,
        PhoenixInterferenceState.RELATED,
        PhoenixInterferenceState.INDEPENDENT,
        PhoenixInterferenceState.CONTRASTING,
        PhoenixInterferenceState.CONFLICTING,
    ]

    # Coherent high-surprise ramp triggers Golden Ratio phi = 4181/2584 boost
    gate_coherent = MarineSalienceGatePq(jitter_ref_q16=16384)
    base_q32 = 1 << 30
    for step in range(6):
        mod_q32, coh_q8, _ = gate_coherent.step_priority_q32(0.1 * (step + 1), base_q32)
    assert coh_q8 == 256
    assert mod_q32 == (base_q32 * 4181) // 2584

    # High-frequency alternating TD-error chatter triggers Marine jitter gate (< 90/256)
    gate_noisy = MarineSalienceGatePq(jitter_ref_q16=16384)
    for step in range(10):
        td = 1.5 if (step % 2 == 0) else 0.05
        mod_noisy_q32, coh_noisy_q8, _ = gate_noisy.step_priority_q32(td, base_q32)
    assert coh_noisy_q8 < 90
    assert mod_noisy_q32 == (base_q32 >> 1)


def test_custodian_deduplicator_trit64():
    custodian = CustodianDeduplicatorPq(similarity_threshold_q16=60000)
    rng = random.Random(7)
    obs1 = [rng.gauss(0.0, 1.0) for _ in range(64)]
    obs1_dup = [x + 0.001 * rng.gauss(0.0, 1.0) for x in obs1]
    obs2_novel = [-x for x in obs1]

    assert custodian.should_accept(obs1) is True
    assert custodian.should_accept(obs1_dup) is False
    assert custodian.should_accept(obs2_novel) is True


def test_catmull_rom_observation_dropout_and_bfp64():
    gt = []
    for step in range(16):
        t = 2.0 * step / 15.0
        gt.append([0.25 * (t ** 2) - 0.3 * t + 0.1 * c for c in range(8)])

    corrupted = [list(row) for row in gt]
    for step in range(5, 9):
        corrupted[step] = [float("nan")] * 8

    restored = restore_observation_dropouts_catmull_rom_pq(corrupted)
    for step in range(16):
        for c in range(8):
            assert not math.isnan(restored[step][c])
            assert abs(restored[step][c] - gt[step][c]) < 5e-4

    mantissas, scales, orig_len = quantize_obs_bfp64(gt)
    recon = dequantize_obs_bfp64(mantissas, scales, orig_len)
    flat_gt = [x for row in gt for x in row]
    assert len(recon) == len(flat_gt)
    for a, b in zip(recon, flat_gt):
        assert abs(a - b) < 0.015


if __name__ == "__main__":
    test_sum_tree_pq_q32_zero_drift()
    print("[PASS] test_sum_tree_pq_q32_zero_drift")
    test_phoenix_interference_and_marine_salience_gate()
    print("[PASS] test_phoenix_interference_and_marine_salience_gate")
    test_custodian_deduplicator_trit64()
    print("[PASS] test_custodian_deduplicator_trit64")
    test_catmull_rom_observation_dropout_and_bfp64()
    print("[PASS] test_catmull_rom_observation_dropout_and_bfp64")
