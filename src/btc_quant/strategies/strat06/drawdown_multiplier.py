"""DrawdownMultiplier: DCA scaling factor based on drawdown depth.

Implements the tiered multiplier table from STRAT-06 §3 Componente 2.
This is a pure function (stateless): given a drawdown, it returns a multiplier.

Default schedule:
    0%  to -10%:  1.0×  (baseline, no amplification)
   -10% to -20%:  1.25×
   -20% to -35%:  1.5×
   -35% to -50%:  2.0×
   -50% or worse: 2.5×

Thresholds are inclusive on the boundary (drawdown == -0.10 → 1.25×).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DrawdownThreshold:
    """A drawdown boundary and its corresponding DCA multiplier.

    Attributes:
        drawdown_pct: threshold as a non-positive fraction (e.g. -0.10 = -10%).
        multiplier: scale factor applied to the baseline DCA amount (>= 1.0).
    """

    drawdown_pct: float
    multiplier: float

    def __post_init__(self) -> None:
        if self.drawdown_pct > 0:
            raise ValueError(
                f"drawdown_pct must be <= 0 (got {self.drawdown_pct}). "
                "Drawdowns are expressed as non-positive fractions."
            )
        if self.multiplier < 1.0:
            raise ValueError(
                f"multiplier must be >= 1.0 (got {self.multiplier}). "
                "The DCA amount cannot be reduced below baseline."
            )


class DrawdownMultiplier:
    """Maps current drawdown depth to a DCA scaling factor.

    Stateless: every call to get_multiplier() is independent.
    Thresholds must be ordered from least negative to most negative
    (e.g. [-0.10, -0.20, -0.35, -0.50]).

    The matching rule is: apply the multiplier of the deepest threshold
    that the current drawdown has reached or exceeded.
    """

    DEFAULT_THRESHOLDS: list[DrawdownThreshold] = [
        DrawdownThreshold(drawdown_pct=-0.10, multiplier=1.25),
        DrawdownThreshold(drawdown_pct=-0.20, multiplier=1.50),
        DrawdownThreshold(drawdown_pct=-0.35, multiplier=2.00),
        DrawdownThreshold(drawdown_pct=-0.50, multiplier=2.50),
    ]

    def __init__(self, thresholds: list[DrawdownThreshold] | None = None) -> None:
        """
        Args:
            thresholds: ordered list of thresholds from least to most negative
                drawdown_pct. Defaults to DEFAULT_THRESHOLDS if None.

        Raises:
            ValueError: if thresholds are not strictly ordered least-to-most negative.
        """
        self._thresholds = thresholds if thresholds is not None else self.DEFAULT_THRESHOLDS

        for i in range(1, len(self._thresholds)):
            if self._thresholds[i].drawdown_pct >= self._thresholds[i - 1].drawdown_pct:
                raise ValueError(
                    "Thresholds must be ordered from least negative to most negative "
                    f"(got {self._thresholds[i-1].drawdown_pct} then "
                    f"{self._thresholds[i].drawdown_pct})"
                )

    def get_multiplier(self, drawdown: float) -> float:
        """Return the DCA multiplier for the given drawdown.

        Iterates thresholds from least to most negative and applies each
        whose drawdown_pct the current drawdown meets or exceeds, stopping
        early once the drawdown no longer qualifies.

        Args:
            drawdown: current drawdown as a non-positive fraction (e.g. -0.25).

        Returns:
            Multiplier >= 1.0.

        Raises:
            ValueError: if drawdown > 0.
        """
        if drawdown > 0:
            raise ValueError(
                f"drawdown must be <= 0 (got {drawdown}). "
                "Pass the output of ATHTracker.get_drawdown()."
            )

        applicable = 1.0
        for threshold in self._thresholds:
            if drawdown <= threshold.drawdown_pct:
                applicable = threshold.multiplier
            else:
                break  # remaining thresholds are more negative; drawdown can't qualify
        return applicable
