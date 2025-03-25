from dataclasses import dataclass
from abc import ABC, abstractmethod
import typing

from color import Color
from vec3 import Vec3, Point3, dot
from ray import Ray
from interval import Interval


@dataclass
class ScatterRecord:
    attenuation: Color
    scattered: Ray


@dataclass
class HitRecord:
    p: Point3
    normal: Vec3
    t: float
    material: "Material"
    front_face: bool


class Hittable(ABC):
    @abstractmethod
    def hit(self, r: Ray, t_range: Interval) -> typing.Optional[HitRecord]:
        raise NotImplementedError()


class Material(ABC):
    @abstractmethod
    def scatter(self, r: Ray, hit_record: HitRecord) -> typing.Optional[ScatterRecord]:
        raise NotImplementedError()
