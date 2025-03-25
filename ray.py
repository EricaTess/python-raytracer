from dataclasses import dataclass
from vec3 import Point3, Vec3


@dataclass(frozen=True)
class Ray:
    origin: Point3
    direction: Vec3

    def at(self, t: float) -> Point3:
        return self.origin + (self.direction * t)
