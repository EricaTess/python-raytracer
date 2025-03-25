import math

import multiprocessing
import random
import vec3
import hittable
from camera import Camera
import material
from color import Color


def write_image():
    # Image Parameters
    R = math.cos(math.pi / 4)

    camera = Camera(
        samples_per_pixel=40,
        max_depth=50,
        image_width=1200,
        vfov=20,
        lookfrom=vec3.Point3.from_xyz(13, 2, 3),
        lookat=vec3.Point3.from_xyz(0, 0, 0),
        vup=vec3.Vec3.from_xyz(0, 1, 0),
        defocus_angle=0.6,
        focus_dist=10.0,
    )

    # World
    world = hittable.World()
    ground_material = material.Lambertian(Color.from_xyz(0.5, 0.5, 0.5))
    world.add(hittable.Sphere(vec3.Point3.from_xyz(0, -1000, 0), 1000, ground_material))

    # Create different spheres with different materials
    for a in range(-8, 8):
        for b in range(-8, 8):
            radius = random.uniform(0.05, 0.5)
            choose_mat = random.random()
            center = vec3.Point3.from_xyz(
                a + 0.9 * random.random(), radius, b + 0.9 * random.random()
            )

            if choose_mat < 0.5:
                albedo = Color.random() * Color.random()
                sphere_material = material.Lambertian(albedo)
            elif choose_mat < 0.8:
                albedo = Color.uniform(0.5, 1)
                fuzz = random.uniform(0, 0.5)
                sphere_material = material.Metal(albedo, fuzz)
            else:
                sphere_material = material.Dielectric(1.5)

            world.add(
                hittable.Sphere(center=center, radius=radius, material=sphere_material)
            )

    camera.save_image_to_png(world, "mutliprocessed_orbs2.png")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    write_image()
