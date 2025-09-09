from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class ScoreNormalizer:
    """
    Running normalizer for final scores using Welford's algorithm.

    normalize(x) returns tanh((x - mean) / max(std, eps_or_initial)).
    """

    initial_scale: float = 50.0
    eps: float = 1e-6
    fixed_min: float | None = None
    fixed_max: float | None = None

    # Running stats
    count: int = 0
    mean: float = 0.0
    M2: float = 0.0  # sum of squares of differences from the current mean

    def update(self, x: float) -> None:
        self.count += 1
        if self.count == 1:
            self.mean = float(x)
            self.M2 = 0.0
            return
        delta = float(x) - self.mean
        self.mean += delta / float(self.count)
        delta2 = float(x) - self.mean
        self.M2 += delta * delta2

    @property
    def std(self) -> float:
        if self.count < 2:
            return max(self.initial_scale, self.eps)
        var = self.M2 / float(self.count - 1)
        return max(math.sqrt(max(var, 0.0)), self.eps)

    def normalize(self, x: float) -> float:
        # If fixed range is provided, map linearly to [-1, 1] and clip
        if self.fixed_min is not None and self.fixed_max is not None and self.fixed_max > self.fixed_min:
            t = (float(x) - float(self.fixed_min)) / (float(self.fixed_max) - float(self.fixed_min))
            return float(max(-1.0, min(1.0, 2.0 * t - 1.0)))
        z = (float(x) - self.mean) / self.std
        # tanh squashing into [-1, 1]
        return math.tanh(z)


