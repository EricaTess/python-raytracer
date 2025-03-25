from math import sqrt
from typing import TextIO
from vec3 import Vec3
from interval import Interval


def linear_to_gamma(linear_component: float) -> float:
    if linear_component > 0:
        return sqrt(linear_component)
    return 0


class Color(Vec3):
    @property
    def red(self) -> float:
        return self.x

    @property
    def green(self) -> float:
        return self.y

    @property
    def blue(self) -> float:
        return self.z


def write_color(color: Color, f: TextIO) -> None:
    r = linear_to_gamma(color.red)
    g = linear_to_gamma(color.green)
    b = linear_to_gamma(color.blue)

    intensity = Interval(0.00, 0.999)
    ir = int(256 * intensity.clamp(r))
    ig = int(256 * intensity.clamp(g))
    ib = int(256 * intensity.clamp(b))

    f.write(f"{ir} {ig} {ib}\n")


def make_color_array(color: Color) -> list[Color]:
    r = linear_to_gamma(color.red)
    g = linear_to_gamma(color.green)
    b = linear_to_gamma(color.blue)

    intensity = Interval(0.00, 0.999)
    ir = int(256 * intensity.clamp(r))
    ig = int(256 * intensity.clamp(g))
    ib = int(256 * intensity.clamp(b))

    return [ir, ig, ib]
