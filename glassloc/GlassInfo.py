from dataclasses import dataclass

from airo_typing import HomogeneousMatrixType


@dataclass
class GlassInfo:
    X_Platform_Glass: HomogeneousMatrixType
    radius: float
    height: float
    fluid_level: float | None


@dataclass
class BottleInfo:
    X_Platform_Bottle: HomogeneousMatrixType
    radius: float
    height: float
    color: tuple[float, float, float, float] | None = None
