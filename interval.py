from dataclasses import dataclass
from utils import infinity


@dataclass
class Interval:
    min: float = -infinity
    max: float = infinity

    @property
    def size(self) -> float:
        return self.max - self.min

    def contains(self, x: float) -> bool:
        return self.min <= x and x <= self.max

    def surrounds(self, x: float) -> bool:
        return self.min < x and x < self.max

    def clamp(self, x: float) -> float:
        if x < self.min:
            return self.min
        if x > self.max:
            return self.max
        return x
