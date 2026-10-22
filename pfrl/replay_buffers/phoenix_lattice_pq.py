"""Zero-Float p/q SumTree & Laser Phoenix Lattice Salience Replay Buffer for PFRL.

Author: 8b-is (https://8b.is)
License: MIT
"""

from enum import IntEnum
import math
import random
from typing import Any, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore

# Canonical Golden Ratio rational convergent p/q = F_19 / F_18 = 4181 / 2584
PHI_P: int = 4181
PHI_Q: int = 2584

# Q32 fixed-point unit for exact integer priority trees
Q32_ONE: int = 1 << 32
Q16_ONE: int = 1 << 16


class PhoenixInterferenceState(IntEnum):
    """5-state Laser Phoenix Interference Lattice on Z_{2^32}."""

    BOUND = 0        # 0 deg   (|dtheta| < 22.5 deg)
    RELATED = 1      # 45 deg  (22.5 deg <= |dtheta| < 67.5 deg)
    INDEPENDENT = 2  # 90 deg  (67.5 deg <= |dtheta| < 112.5 deg)
    CONTRASTING = 3  # 135 deg (112.5 deg <= |dtheta| < 157.5 deg)
    CONFLICTING = 4  # 180 deg (|dtheta| >= 157.5 deg)


PHOENIX_LATTICE_ANCHORS_Q32: Tuple[int, int, int, int, int] = (
    0x00000000,  # 0.0 deg
    0x20000000,  # 45.0 deg
    0x40000000,  # 90.0 deg
    0x60000000,  # 135.0 deg
    0x80000000,  # 180.0 deg
)


def classify_phoenix_interference_q32(signed_delta_q32: int) -> PhoenixInterferenceState:
    """Classifies a signed modular phase offset on Z_{2^32} into the 5-state lattice."""
    abs_d = abs(int(signed_delta_q32)) & 0xFFFFFFFF
    if abs_d > 0x80000000:
        abs_d = 0x100000000 - abs_d
    if abs_d < 0x10000000:
        return PhoenixInterferenceState.BOUND
    if abs_d < 0x30000000:
        return PhoenixInterferenceState.RELATED
    if abs_d < 0x50000000:
        return PhoenixInterferenceState.INDEPENDENT
    if abs_d < 0x70000000:
        return PhoenixInterferenceState.CONTRASTING
    return PhoenixInterferenceState.CONFLICTING


def isqrt_u128(n: int) -> int:
    """Exact integer floor(sqrt(n)) for arbitrary-precision non-negative integers."""
    if n <= 0:
        return 0
    x = 1 << ((n.bit_length() + 1) >> 1)
    while True:
        y = (x + n // x) >> 1
        if y >= x:
            return x
        x = y


def _flatten_floats(data: Any) -> List[float]:
    if np is not None and isinstance(data, np.ndarray):
        return [float(x) for x in data.ravel().tolist()]
    if isinstance(data, (list, tuple)):
        out: List[float] = []
        for item in data:
            out.extend(_flatten_floats(item))
        return out
    return [float(data)]


class SumTreePqQ32:
    """Zero-float 64-bit integer Q32 flat segment Sum-Tree & Min-Tree.

    Eliminates cumulative IEEE-754 floating-point priority drift across millions of
    Prioritized Experience Replay (PER) updates by storing priorities as exact
    non-negative integers in Q32 (1.0 == 2^32).
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._tree_cap = 1
        while self._tree_cap < self.capacity:
            self._tree_cap <<= 1
        self._sum_tree: List[int] = [0] * (2 * self._tree_cap)
        self._min_tree: List[int] = [0] * (2 * self._tree_cap)
        self._present: List[bool] = [False] * self._tree_cap
        self.length: int = 0
        self._head: int = 0

    @staticmethod
    def float_to_q32(priority: float) -> int:
        p = max(1e-12, float(priority))
        return max(1, int(round(p * Q32_ONE)))

    @staticmethod
    def q32_to_float(priority_q32: int) -> float:
        return float(priority_q32) / float(Q32_ONE)

    def _set_physical_leaf(self, phys_idx: int, val_q32: int, present: bool) -> None:
        node = self._tree_cap + phys_idx
        self._sum_tree[node] = val_q32 if present else 0
        self._min_tree[node] = val_q32 if (present and val_q32 > 0) else 0
        self._present[phys_idx] = present
        node >>= 1
        while node >= 1:
            left = node << 1
            right = left | 1
            self._sum_tree[node] = self._sum_tree[left] + self._sum_tree[right]
            m_left = self._min_tree[left]
            m_right = self._min_tree[right]
            if m_left > 0 and m_right > 0:
                self._min_tree[node] = m_left if m_left < m_right else m_right
            else:
                self._min_tree[node] = m_left if m_left > 0 else m_right
            node >>= 1

    def _logical_to_physical(self, logical_idx: int) -> int:
        if not (0 <= logical_idx < self.length):
            raise IndexError(f"Index {logical_idx} out of range [0, {self.length})")
        return (self._head + logical_idx) % self.capacity

    def _physical_to_logical(self, phys_idx: int) -> int:
        return (phys_idx - self._head) % self.capacity

    def append_q32(self, val_q32: int) -> None:
        val_q32 = max(1, int(val_q32))
        if self.length == self.capacity:
            self.popleft_q32()
        phys_idx = (self._head + self.length) % self.capacity
        self.length += 1
        self._set_physical_leaf(phys_idx, val_q32, True)

    def append(self, priority: float) -> None:
        self.append_q32(self.float_to_q32(priority))

    def popleft_q32(self) -> int:
        if self.length == 0:
            raise IndexError("popleft from empty SumTreePqQ32")
        phys_idx = self._head
        old_val = self._sum_tree[self._tree_cap + phys_idx]
        self._set_physical_leaf(phys_idx, 0, False)
        self._head = (self._head + 1) % self.capacity
        self.length -= 1
        return old_val

    def __setitem__(self, logical_idx: int, priority: float) -> None:
        phys_idx = self._logical_to_physical(logical_idx)
        self._set_physical_leaf(phys_idx, self.float_to_q32(priority), True)

    def set_q32(self, logical_idx: int, val_q32: int) -> None:
        phys_idx = self._logical_to_physical(logical_idx)
        self._set_physical_leaf(phys_idx, max(1, int(val_q32)), True)

    def sum_q32(self) -> int:
        return self._sum_tree[1]

    def sum(self) -> float:
        return self.q32_to_float(self._sum_tree[1])

    def min_q32(self) -> int:
        return self._min_tree[1]

    def min(self) -> float:
        if self.length == 0 or self._min_tree[1] == 0:
            return float("inf")
        return self.q32_to_float(self._min_tree[1])

    def find_prefix_q32(self, target_q32: int) -> int:
        """Exact integer prefix-sum walk in [0, sum_q32). Never branches into empty leaves."""
        if self.length == 0 or self._sum_tree[1] <= 0:
            raise ValueError("Cannot sample from empty or zero-sum tree")
        pos = max(0, min(int(target_q32), self._sum_tree[1] - 1))
        node = 1
        while node < self._tree_cap:
            left = node << 1
            right = left | 1
            left_sum = self._sum_tree[left]
            right_sum = self._sum_tree[right]
            if right_sum == 0:
                node = left
            elif left_sum == 0:
                node = right
            elif pos < left_sum:
                node = left
            else:
                pos -= left_sum
                node = right
        phys_idx = node - self._tree_cap
        return self._physical_to_logical(phys_idx)

    def prioritized_sample(
        self, n: int, remove: bool = True
    ) -> Tuple[List[int], List[float]]:
        if n < 0 or n > self.length:
            raise ValueError(f"Invalid sample size {n} for length {self.length}")
        logical_indices: List[int] = []
        phys_indices: List[int] = []
        saved_q32: List[int] = []
        vals_float: List[float] = []

        for _ in range(n):
            total = self._sum_tree[1]
            target = random.randrange(0, max(1, total))
            log_ix = self.find_prefix_q32(target)
            phys_ix = self._logical_to_physical(log_ix)
            val_q32 = self._sum_tree[self._tree_cap + phys_ix]
            self._set_physical_leaf(phys_ix, 0, True)
            logical_indices.append(log_ix)
            phys_indices.append(phys_ix)
            saved_q32.append(val_q32)
            vals_float.append(self.q32_to_float(val_q32))

        if not remove:
            for phys_ix, val_q32 in zip(phys_indices, saved_q32):
                self._set_physical_leaf(phys_ix, val_q32, True)

        return logical_indices, vals_float


class MarineSalienceGatePq:
    """Marine second-order TD-error phase-velocity jitter gate with phi-harmonic boost.

    Tracks TD-error velocity differences in an 8-step sliding window:
      - When target-network chatter causes high second-order jitter (coherence < 0.35,
        i.e., < 90 / 256), priority gain is halved to suppress noisy oscillation.
      - When TD-error trajectory is coherent (coherence >= 180 / 256) and surprise
        exceeds threshold, priority is boosted by Golden Ratio phi = 4181 / 2584.
    """

    def __init__(self, jitter_ref_q16: int = 32768) -> None:
        self.jitter_ref_q16 = max(1, int(jitter_ref_q16))
        self._prev_td_q16: Optional[int] = None
        self._prev_vel_q16: Optional[int] = None
        self._jitter_ring: List[int] = [0] * 8
        self._idx: int = 0
        self._count: int = 0

    def step_priority_q32(
        self, td_error: float, base_priority_q32: int
    ) -> Tuple[int, int, PhoenixInterferenceState]:
        td_q16 = int(round(abs(float(td_error)) * Q16_ONE))
        if self._prev_td_q16 is None:
            self._prev_td_q16 = td_q16
            return base_priority_q32, 256, PhoenixInterferenceState.BOUND

        vel_q16 = td_q16 - self._prev_td_q16
        self._prev_td_q16 = td_q16

        phase_delta_q32 = (vel_q16 * 0x20000000) // self.jitter_ref_q16
        state = classify_phoenix_interference_q32(phase_delta_q32)

        if self._prev_vel_q16 is None:
            self._prev_vel_q16 = vel_q16
            return base_priority_q32, 256, state

        accel_q16 = abs(vel_q16 - self._prev_vel_q16)
        self._prev_vel_q16 = vel_q16

        self._jitter_ring[self._idx] = accel_q16
        self._idx = (self._idx + 1) & 7
        if self._count < 8:
            self._count += 1

        mean_jitter = sum(self._jitter_ring[: self._count]) // self._count
        penalty = (mean_jitter * 256) // self.jitter_ref_q16
        coherence_q8 = max(0, 256 - penalty)

        if coherence_q8 < 90:
            mod_q32 = max(1, base_priority_q32 >> 1)
        elif coherence_q8 >= 180 and td_q16 >= self.jitter_ref_q16:
            mod_q32 = (base_priority_q32 * PHI_P) // PHI_Q
        else:
            mod_q32 = base_priority_q32

        return mod_q32, coherence_q8, state


class CustodianDeduplicatorPq:
    """64-trit ternary observation hash + POPCNT cosine similarity deduplicator.

    Prevents stationary/redundant transitions from flooding the replay buffer.
    """

    def __init__(self, similarity_threshold_q16: int = 62259) -> None:
        self.similarity_threshold_q16 = int(similarity_threshold_q16)
        self._recent_signatures: List[Tuple[int, int]] = []
        self._max_history: int = 32

    @staticmethod
    def pack_obs_trit64(obs: Any) -> Tuple[int, int]:
        flat = _flatten_floats(obs)
        n = len(flat)
        if n == 0:
            return 0, 0
        if n < 64:
            flat = flat + [0.0] * (64 - n)
        elif n > 64:
            pooled: List[float] = []
            for b in range(64):
                start = (b * n) // 64
                end = max(start + 1, ((b + 1) * n) // 64)
                chunk = flat[start:end]
                pooled.append(sum(chunk) / len(chunk))
            flat = pooled

        tau = 0.25 * (sum(abs(x) for x in flat) / 64.0)
        pos_mask = 0
        neg_mask = 0
        for bit_idx in range(64):
            v = flat[bit_idx]
            if v > tau:
                pos_mask |= 1 << bit_idx
            elif v < -tau:
                neg_mask |= 1 << bit_idx
        return pos_mask, neg_mask

    @staticmethod
    def cosine_similarity_q16(sig_a: Tuple[int, int], sig_b: Tuple[int, int]) -> int:
        pos_a, neg_a = sig_a
        pos_b, neg_b = sig_b
        nz_a = (pos_a | neg_a).bit_count()
        nz_b = (pos_b | neg_b).bit_count()
        if nz_a == 0 or nz_b == 0:
            return Q16_ONE if (nz_a == nz_b) else 0
        agree = ((pos_a & pos_b) | (neg_a & neg_b)).bit_count()
        disagree = ((pos_a & neg_b) | (neg_a & pos_b)).bit_count()
        dot = agree - disagree
        if dot <= 0:
            return 0
        denom = isqrt_u128(nz_a * nz_b * (Q16_ONE * Q16_ONE))
        return min(Q16_ONE, (dot * (Q16_ONE * Q16_ONE)) // max(1, denom))

    def should_accept(self, obs: Any) -> bool:
        sig = self.pack_obs_trit64(obs)
        for prev in self._recent_signatures:
            if self.cosine_similarity_q16(sig, prev) >= self.similarity_threshold_q16:
                return False
        self._recent_signatures.append(sig)
        if len(self._recent_signatures) > self._max_history:
            self._recent_signatures.pop(0)
        return True


def _eval_catmull_rom_scalar_pq(
    p0: int, p1: int, p2: int, p3: int, k: int, span: int
) -> int:
    """Exact integer p/q Catmull-Rom cubic Hermite evaluation at t = k / span."""
    if span <= 0:
        return p1
    s2 = span * span
    s3 = s2 * span
    k2 = k * k
    k3 = k2 * k

    a0 = 2 * p1
    a1 = -p0 + p2
    a2 = 2 * p0 - 5 * p1 + 4 * p2 - p3
    a3 = -p0 + 3 * p1 - 3 * p2 + p3

    num = a0 * s3 + a1 * k * s2 + a2 * k2 * span + a3 * k3
    den = 2 * s3
    if num >= 0:
        return (num + den // 2) // den
    return -((-num + den // 2) // den)


def restore_observation_dropouts_catmull_rom_pq(
    trajectory_obs: Sequence[Sequence[float]],
    valid_mask: Optional[Sequence[bool]] = None,
) -> List[List[float]]:
    """Restores dropped or NaN observation frames along time axis 0 using non-uniform
    rational p/q Bessel-Overhauser Catmull-Rom cubic Hermite interpolation in Q16.
    """
    rows = [[float(v) for v in row] for row in trajectory_obs]
    t_steps = len(rows)
    if t_steps <= 1:
        return rows

    dim = len(rows[0])
    if valid_mask is None:
        flat_valid = [not any(math.isnan(v) for v in rows[t]) for t in range(t_steps)]
    else:
        flat_valid = [bool(v) for v in valid_mask]

    valid_indices = [t for t, ok in enumerate(flat_valid) if ok]
    if not valid_indices or len(valid_indices) == t_steps:
        return rows

    q16_anchors = {
        t: [int(round(v * Q16_ONE)) for v in rows[t]] for t in valid_indices
    }
    out = [list(r) for r in rows]

    for t in range(t_steps):
        if flat_valid[t]:
            continue
        if t < valid_indices[0]:
            out[t] = list(rows[valid_indices[0]])
            continue
        if t > valid_indices[-1]:
            out[t] = list(rows[valid_indices[-1]])
            continue

        pos = 0
        while pos + 1 < len(valid_indices) and valid_indices[pos + 1] < t:
            pos += 1

        k1 = pos
        k2 = min(len(valid_indices) - 1, pos + 1)
        k0 = max(0, k1 - 1)
        k3 = min(len(valid_indices) - 1, k2 + 1)

        idx0 = valid_indices[k0]
        idx1 = valid_indices[k1]
        idx2 = valid_indices[k2]
        idx3 = valid_indices[k3]

        span = idx2 - idx1
        step_k = t - idx1

        y0 = q16_anchors[idx0]
        p1 = q16_anchors[idx1]
        p2 = q16_anchors[idx2]
        y3 = q16_anchors[idx3]

        restored_row: List[float] = []
        h0 = idx1 - idx0
        h1 = span
        h2 = idx3 - idx2
        for d in range(dim):
            if h0 > 0 and h1 > 0:
                # Exact Bessel-Overhauser non-uniform parabolic tangent scaled by h1
                num_m1 = (h1 * h1 * (p1[d] - y0[d])) + (h0 * h0 * (p2[d] - p1[d]))
                m1 = num_m1 // (h0 * (h0 + h1))
            else:
                m1 = p2[d] - p1[d]

            if h2 > 0 and h1 > 0:
                num_m2 = (h2 * h2 * (p2[d] - p1[d])) + (h1 * h1 * (y3[d] - p2[d]))
                m2 = (num_m2 * h1) // (h2 * h1 * (h1 + h2))
            else:
                m2 = p2[d] - p1[d]

            p0_d = p2[d] - 2 * m1
            p3_d = p1[d] + 2 * m2
            interp_q16 = _eval_catmull_rom_scalar_pq(
                p0_d, p1[d], p2[d], p3_d, step_k, span
            )
            restored_row.append(float(interp_q16) / float(Q16_ONE))
        out[t] = restored_row

    return out


def quantize_obs_bfp64(obs: Any) -> Tuple[List[List[int]], List[float], int]:
    """Compresses a flat float observation sequence into Omni BFP-64 blocks (~3.9x smaller)."""
    flat = _flatten_floats(obs)
    orig_len = len(flat)
    pad = (64 - (orig_len % 64)) % 64
    if pad > 0:
        flat = flat + [0.0] * pad
    n_blocks = len(flat) // 64
    mantissas: List[List[int]] = []
    scales: List[float] = []
    for b in range(n_blocks):
        blk = flat[b * 64 : (b + 1) * 64]
        amax = max((abs(x) for x in blk), default=0.0)
        scale = (amax / 127.0) if amax > 0.0 else 1.0
        scales.append(scale)
        mantissas.append([max(-127, min(127, int(round(x / scale)))) for x in blk])
    return mantissas, scales, orig_len


def dequantize_obs_bfp64(
    mantissas: Sequence[Sequence[int]], scales: Sequence[float], orig_len: int
) -> List[float]:
    """Reconstructs a float observation sequence from Omni BFP-64 blocks."""
    out: List[float] = []
    for blk, sc in zip(mantissas, scales):
        for m in blk:
            out.append(float(m) * float(sc))
    return out[:orig_len]
