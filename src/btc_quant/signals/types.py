from dataclasses import dataclass


@dataclass(frozen=True)
class SignalResult:
    """Result of evaluating a single signal at a single point in time.

    Attributes:
        active: True if the signal fires (contributes to confluence score).
        value: The underlying numeric metric. Useful for debug/visualization.
               Sign convention: positive = bullish bias where applicable.
               NaN if insufficient data to compute.
    """

    active: bool
    value: float
