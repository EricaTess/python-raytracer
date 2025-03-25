from dataclasses import dataclass
from typing import Optional
from math import sqrt
import random

from ray import Ray
from core import HitRecord
from color import Color
import vec3
import core


@dataclass
class Lambertian(core.Material):
    albedo: Color

    def scatter(self, r: Ray, hit_record: HitRecord) -> Optional[core.ScatterRecord]:
        scatter_direction = hit_record.normal + vec3.random_unit_vector()
        if scatter_direction.near_zero():
            scatter_direction = hit_record.normal
        scattered = Ray(hit_record.p, scatter_direction)
        return core.ScatterRecord(attenuation=self.albedo, scattered=scattered)


@dataclass
class Metal(core.Material):
    albedo: Color
    fuzz: float = 0

    def __post_init__(self):
        if self.fuzz < 0 or self.fuzz > 1:
            raise ValueError("Fuzz is out of range")

    def scatter(self, r: Ray, hit_record: HitRecord) -> Optional[core.ScatterRecord]:
        reflected = vec3.reflect(r.direction, hit_record.normal)
        reflected = vec3.unit_vector(reflected) + (
            vec3.random_unit_vector() * self.fuzz
        )
        scattered = Ray(hit_record.p, reflected)
        return core.ScatterRecord(attenuation=self.albedo, scattered=scattered)


@dataclass
class Dielectric(core.Material):
    refraction_index: float

    def scatter(self, r_in: Ray, hit_record: HitRecord) -> core.ScatterRecord:
        attenuation = Color.from_xyz(1.0, 1.0, 1.0)

        # We are hitting the front of the object if the dot product of
        # the input ray and normal are negative (opposite directions)
        ri = (
            (1.0 / self.refraction_index)
            if hit_record.front_face
            else self.refraction_index
        )
        unit_direction = vec3.unit_vector(r_in.direction)

        cos_theta = min(vec3.dot(-unit_direction, hit_record.normal), 1.0)
        sin_theta = sqrt(1.0 - (cos_theta * cos_theta))

        cannot_refract = ri * sin_theta > 1.0

        if cannot_refract or self.reflectance(cos_theta, ri) > random.random():
            direction = vec3.reflect(unit_direction, hit_record.normal)
        else:
            direction = vec3.refract(unit_direction, hit_record.normal, ri)

        scattered = Ray(hit_record.p, direction)
        return core.ScatterRecord(attenuation=attenuation, scattered=scattered)

    def reflectance(self, cosine: float, refraction_index: float) -> float:
        """Probability between 0 and 1 that a ray and the angle will reflect instead of refract"""
        r0 = (1 - refraction_index) / (1 + refraction_index)
        r0 = r0 * r0
        return r0 + (1 - r0) * pow((1 - cosine), 5)
