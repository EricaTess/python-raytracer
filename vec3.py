# Stdlib
import numpy as np
from dataclasses import dataclass, field
from math import sqrt
import typing
from random import random, uniform


@dataclass
class Vec3:
    coordinates: tuple[float, float, float] = field(default_factory=lambda: (0, 0, 0))

    @classmethod
    def from_xyz(cls, x: float, y: float, z: float) -> "Vec3":
        return cls((x, y, z))

    def __getitem__(self, key):
        return self.coordinates[key]

    def __setitem__(self, key, value):
        self.coordinates[key] = value

    def __repr__(self):
        return f"Vec3({self.x},{self.y},{self.z})"

    def __add__(self, other: "Vec3") -> "Vec3":
        return self.__class__.from_xyz(
            self[0] + other[0], self[1] + other[1], self[2] + other[2]
        )

    def __sub__(self, other: "Vec3") -> "Vec3":
        return self.__class__.from_xyz(
            self[0] - other[0], self[1] - other[1], self[2] - other[2]
        )

    def __mul__(self, other: typing.Union["Vec3", float]) -> "Vec3":
        if isinstance(other, Vec3):
            return self.__class__.from_xyz(
                self[0] * other[0], self[1] * other[1], self[2] * other[2]
            )
        else:
            return self.__class__.from_xyz(
                self[0] * other, self[1] * other, self[2] * other
            )

    def __truediv__(self, other: float) -> "Vec3":
        return self * (1 / other)

    def __neg__(self) -> "Vec3":
        return self.inverse()

    @property
    def x(self):
        return self.coordinates[0]

    @property
    def y(self):
        return self.coordinates[1]

    @property
    def z(self):
        return self.coordinates[2]

    def inverse(self) -> "Vec3":
        return Vec3.from_xyz(-self.x, -self.y, -self.z)

    def add(self, other: "Vec3") -> None:
        self[0] += other[0]
        self[1] += other[1]
        self[2] += other[2]

    def multiply(self, t: float) -> None:
        self[0] *= t
        self[1] *= t
        self[2] *= t

    def divide(self, t: float) -> None:
        self.multiply(1 / t)

    @property
    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    @property
    def length(self) -> float:
        return sqrt(self.length_squared)

    @classmethod
    def random(cls) -> "Vec3":
        return cls.from_xyz(random(), random(), random())

    @classmethod
    def uniform(cls, min: float, max: float) -> "Vec3":
        return cls.from_xyz(uniform(min, max), uniform(min, max), uniform(min, max))

    def near_zero(self) -> bool:
        s = 1e-8
        return (abs(self.x) < s) and (abs(self.y) < s) and (abs(self.z) < s)


class Point3(Vec3):
    pass


def dot(u: Vec3, v: Vec3) -> float:
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def cross(u: Vec3, v: Vec3) -> Vec3:
    return Vec3.from_xyz(u[1] * v[2] - u[2] * v[1],
                         u[2] * v[0] - u[0] * v[2],
                         u[0] * v[1] - u[1] * v[0]
                         )


def unit_vector(v: Vec3) -> Vec3:
    """Scale current vector of length 1"""
    return v / v.length


def random_unit_vector() -> Vec3:
    while True:
        p = Vec3.uniform(-1, 1)
        lensq = p.length_squared
        if 1e-160 < lensq and lensq <= 1:
            return p / sqrt(lensq)


def random_in_unit_disk() -> Vec3:
    while True:
        p = Vec3.from_xyz(uniform(-1, 1), uniform(-1, 1), 0)
        if p.length_squared < 1:
            return p


def random_on_hemisphere(normal: Vec3) -> Vec3:
    on_unit_sphere = random_unit_vector()
    if dot(on_unit_sphere, normal) > 0.0:
        return on_unit_sphere
    else:
        return -on_unit_sphere


def reflect(v: Vec3, normal: Vec3) -> Vec3:
    """Vector v is reflected across the normal vector of some mirrored plane"""
    return v - (normal * dot(v, normal) * 2)


def refract(uv: Vec3, n: Vec3, relative_refractive_index: float) -> Vec3:
    """Function to compute a new refracted vector after passing through a refractive object
    Arguments:
        uv: Incoming vector
        n: Normal vector of the refractive surface
        relative_refractive_index: refractive index of the object over the refactive index of the surrounding material (air)
    """
    cos_theta = min(dot(-uv, n), 1.0)
    r_out_perp = (uv + (n * cos_theta)) * relative_refractive_index
    r_out_parallel = n * -sqrt(abs(1.0 - r_out_perp.length_squared))
    return r_out_perp + r_out_parallel
