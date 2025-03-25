import math
from random import random
import tqdm
import typing
import multiprocessing
from functools import partial
from PIL import Image
import numpy as np
from color import Color, write_color, make_color_array
from ray import Ray
from core import Hittable
import utils
from interval import Interval
import vec3


class Camera:
    def __init__(
        self,
        aspect_ratio: float = (16.0 / 9.0),
        image_width: int = 400,
        samples_per_pixel: int = 10,
        max_depth: int = 10,
        vfov: float = 90,
        lookfrom: typing.Optional[vec3.Point3] = None,
        lookat: typing.Optional[vec3.Point3] = None,
        vup: typing.Optional[vec3.Vec3] = None,
        defocus_angle: float = 0,
        focus_dist: float = 10,
    ):
        self.image_width = image_width
        self.samples_per_pixel = samples_per_pixel
        self.image_height = max(1, int(self.image_width / aspect_ratio))
        self.max_depth = max_depth
        self.defocus_angle = defocus_angle
        lookfrom = lookfrom or vec3.Point3.from_xyz(0, 0, 0)
        lookat = lookat or vec3.Point3.from_xyz(0, 0, -1)
        vup = vup or vec3.Vec3.from_xyz(0, 1, 0)

        # Camera Parameters
        theta = utils.degrees_to_radians(vfov)
        h = math.tan(theta / 2)
        viewport_height = 2 * h * focus_dist
        viewport_width = viewport_height * (float(image_width) / self.image_height)
        self.camera_center = lookfrom

        # Calculate u, v, w unit basis vectors for camera coordinate frame
        w = vec3.unit_vector(lookfrom - lookat)
        u = vec3.unit_vector(vec3.cross(vup, w))
        v = vec3.cross(w, u)

        # Viewport positioning vectors
        viewport_u = u * viewport_width  # vector across viewpoint horizontal edge
        viewport_v = -v * viewport_height  # vector across viewpoirt vertical edge

        self.pixel_delta_u = viewport_u / image_width
        self.pixel_delta_v = viewport_v / self.image_height

        viewport_upper_left = (
            self.camera_center - (w * focus_dist) - viewport_u / 2 - viewport_v / 2
        )
        self.pixel00_loc = viewport_upper_left + (
            (self.pixel_delta_u + self.pixel_delta_v) * 0.5
        )

        defocus_radius = (
            math.tan(utils.degrees_to_radians(defocus_angle / 2)) * focus_dist
        )
        self.defocus_disk_u = u * defocus_radius
        self.defocus_disk_v = v * defocus_radius

    def render(self, world: Hittable, file_name: str) -> None:
        """Render image to PPM"""
        with open(file_name, "w") as f:
            f.write(f"P3\n{self.image_width} {self.image_height}\n255\n")
            for j in tqdm.tqdm(range(self.image_height)):
                for i in range(self.image_width):
                    pixel = Color.from_xyz(0, 0, 0)
                    for _ in range(self.samples_per_pixel):
                        r = self.get_ray(i, j)
                        pixel += self.ray_color(r, world, self.max_depth)

                    write_color(pixel / self.samples_per_pixel, f)

    def process_chunk(self, chunk, world):
        """Compute a chunk of pixels."""
        chunk_pixels = []
        print(f"Processing chunk with {len(chunk)} pixels")
        for j, i in chunk:
            pixel = Color.from_xyz(0, 0, 0)
            for _ in range(self.samples_per_pixel):
                r = self.get_ray(i, j)
                pixel += self.ray_color(r, world, self.max_depth)
            rgb = make_color_array(pixel / self.samples_per_pixel)
            chunk_pixels.append(((j, i), np.clip(rgb, 0, 255).astype(np.uint8)))
        print(f"Finished chunk with {len(chunk)} pixels!")
        return chunk_pixels

    def save_image_to_png(self, world: Hittable, filename: str) -> None:
        """Render the image to PNG using multiprocessing with chunked pixel processing."""
        num_cores = multiprocessing.cpu_count()

        color_array = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

        # Create chunks of pixels to distribute among processes
        pixels = [
            (j, i) for j in range(self.image_height) for i in range(self.image_width)
        ]
        num_chunks = multiprocessing.cpu_count() * 8
        chunk_size = len(pixels) // num_chunks
        chunks = [pixels[i : i + chunk_size] for i in range(0, len(pixels), chunk_size)]

        # Use multiprocessing to process pixel chunks in parallel
        print("Starting multiprocessing...")
        with multiprocessing.Pool(processes=num_cores) as pool:
            process_chunk_with_world = partial(self.process_chunk, world=world)
            results = list(
                tqdm.tqdm(
                    pool.imap_unordered(process_chunk_with_world, chunks),
                    total=len(chunks),
                )
            )
        print("Multiprocessing complete!")
        # Merge results into the color array
        for chunk in results:
            for (j, i), color in chunk:
                color_array[j, i] = color

        image = Image.fromarray(color_array, "RGB")
        image.save(filename, format="PNG")

    def _save_image_to_png(self, world: Hittable, filename: str) -> None:
        """Process pixels one at a time and save to PNG"""
        color_array = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        for j in tqdm.tqdm(range(self.image_height)):
            for i in range(self.image_width):
                pixel = Color.from_xyz(0, 0, 0)
                for _ in range(self.samples_per_pixel):
                    r = self.get_ray(i, j)
                    pixel += self.ray_color(r, world, self.max_depth)
                rgb = make_color_array(pixel / self.samples_per_pixel)
                color_array[j, i] = np.clip(rgb, 0, 255).astype(np.uint8)

        image = Image.fromarray(color_array, "RGB")
        image.save(filename, format="PNG")

    def ray_color(self, ray: Ray, world: Hittable, depth: int) -> Color:
        if depth == 0:
            return Color.from_xyz(0, 0, 0)
        t_range = Interval(0.001, utils.infinity)
        record = world.hit(ray, t_range)

        if record:
            scatter_record = record.material.scatter(ray, record)
            if scatter_record:
                return scatter_record.attenuation * self.ray_color(
                    ray=scatter_record.scattered, world=world, depth=depth - 1
                )
            return Color.from_xyz(0, 0, 0)

        unit_direction = vec3.unit_vector(ray.direction)
        a = (unit_direction.y + 1.0) * 0.5
        return (Color.from_xyz(1.0, 1.0, 1.0) * (1.0 - a)) + (
            Color.from_xyz(0.5, 0.7, 1.0) * a
        )

    def get_ray(self, i: int, j: int) -> Ray:
        offset = self.sample_square()
        pixel_sample = (
            self.pixel00_loc
            + (self.pixel_delta_u * (i + offset.x))
            + (self.pixel_delta_v * (j + offset.y))
        )
        ray_origin = (
            self.camera_center
            if self.defocus_angle <= 0
            else self.defocus_disk_sample()
        )
        ray_direction = pixel_sample - ray_origin
        return Ray(ray_origin, ray_direction)

    def defocus_disk_sample(self) -> vec3.Point3:
        """Returns a random point in the camera defocus disk"""
        p = vec3.random_in_unit_disk()
        return (
            self.camera_center
            + (self.defocus_disk_u * p[0])
            + (self.defocus_disk_v * p[1])
        )

    def sample_square(self) -> vec3.Vec3:
        return vec3.Vec3.from_xyz(random() - 0.5, random() - 0.5, 0)
