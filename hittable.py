import copy
import typing
from dataclasses import dataclass, field
from math import sqrt

from vec3 import Point3, dot
from ray import Ray
from interval import Interval
import core


@dataclass
class World(core.Hittable):
    objects: list[core.Hittable] = field(default_factory=list)

    def clear(self) -> None:
        self.objects = []

    def add(self, object: core.Hittable) -> None:
        self.objects.append(object)

    def hit(self, r: Ray, t_range: Interval) -> typing.Optional[core.HitRecord]:
        closest_record = None
        curr_range = copy.copy(t_range)
        for object in self.objects:
            object_record = object.hit(r, curr_range)
            if object_record:
                closest_record = object_record
                curr_range.max = closest_record.t

        return closest_record


@dataclass
class Sphere(core.Hittable):
    center: Point3
    radius: float
    material: core.Material

    # Radius should only be a positive number
    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError("Radius can't be 0 or less")

    def hit(self, r: Ray, t_range: Interval) -> typing.Optional[core.HitRecord]:
        oc = self.center - r.origin
        a = r.direction.length_squared
        h = dot(r.direction, oc)
        c = oc.length_squared - (self.radius * self.radius)
        discriminant = (h * h) - (a * c)
        # Check if real-valued solutions exist
        if discriminant < 0:
            return None

        sqrtd = sqrt(discriminant)

        # Find nearest root that lies in acceptable range.
        root = (h - sqrtd) / a
        if not t_range.surrounds(root):
            root = (h + sqrtd) / a
            if not t_range.surrounds(root):
                return None

        point = r.at(root)
        outward_normal = (point - self.center) / self.radius
        front_face = dot(r.direction, outward_normal) < 0
        normal = outward_normal if front_face else -outward_normal
        return core.HitRecord(
            p=point,
            normal=normal,
            t=root,
            material=self.material,
            front_face=front_face,
        )
