import copy
import random
import time

import numpy as np
import pandas as pd
from fire import Fire

import vamp
import vamp.pointcloud

# Starting configuration
a = [0., -0.785, 0., -2.356, 0., 1.571, 0.785]

# Goal configuration
b = [2.35, 1., 0., -0.8, 0, 2.5, 0.785]

# Problem specification: a list of sphere centers
problem = [
    [0.55, 0, 0.25],
    [0.35, 0.35, 0.25],
    [0, 0.55, 0.25],
    [-0.55, 0, 0.25],
    [-0.35, -0.35, 0.25],
    [0, -0.55, 0.25],
    [0.35, -0.35, 0.25],
    [0.35, 0.35, 0.8],
    [0, 0.55, 0.8],
    [-0.35, 0.35, 0.8],
    [-0.55, 0, 0.8],
    [-0.35, -0.35, 0.8],
    [0, -0.55, 0.8],
    [0.35, -0.35, 0.8],
    ]


def sphere_distances(spheres, radius, origin, spacing, shape):
    """Exact signed distances to a union of spheres at the points of a lattice."""
    axes = [origin[i] + spacing * np.arange(shape[i]) for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing = "ij"), -1)
    distances = np.full(shape, np.inf, dtype = np.float32)
    for center in spheres:
        np.minimum(distances, np.linalg.norm(points - center, axis = -1) - radius, out = distances)

    return distances


def main(
    variation: float = 0.01,               # Maximum variation applied to each sphere
    n_trials: int = 100,                   # Number of problems to plan for
    radius: float = 0.2,                   # Radius of the spheres in the cage
    points_per_sphere: int = 1000,         # Surface samples per sphere for the pointcloud
    point_radius: float = 0.0025,          # Radius of each pointcloud point
    spacing: float = 0.02,                 # Lattice spacing of the distance grid (m)
    lower: tuple = (-1.2, -1.2, -0.6),     # Lower corner of the distance grid's box (m)
    upper: tuple = (1.2, 1.2, 1.6),        # Upper corner of the distance grid's box (m)
    n_validate: int = 10000,               # Random configurations compared with the sphere geometry
    planner: str = "rrtc",
    sampler_name: str = "halton",          # Sampler to use.
    skip_rng_iterations: int = 0,          # Skip a number of RNG iterations
    **kwargs,
    ):
    """Plan on the sphere cage with three descriptions of the same scene: the spheres
    themselves, a CAPT pointcloud of their surfaces, and a distance grid of their exact signed
    distance. Reports the time to build each environment, planning and simplification times,
    and how often each description agrees with the spheres on random configurations."""

    (vamp_module, planner_func, plan_settings,
     simp_settings) = vamp.configure_robot_and_planner_with_kwargs("panda", planner, **kwargs)

    sampler = getattr(vamp_module, sampler_name)()
    sampler.skip(skip_rng_iterations)
    r_min, r_max = vamp_module.min_max_radii()

    origin = np.asarray(lower, dtype = np.float32)
    shape = tuple(int(n) for n in np.floor((np.asarray(upper) - origin) / spacing).astype(int) + 1)

    random.seed(0)
    np.random.seed(0)
    rng = np.random.default_rng(0)

    results = []
    spheres = [np.array(sphere) for sphere in problem]
    for _ in range(n_trials):
        random.shuffle(spheres)
        spheres_copy = copy.deepcopy(spheres)
        for sphere in spheres_copy:
            sphere += np.random.uniform(low = -variation, high = variation, size = (3, ))

        environments = {}

        e = vamp.Environment()
        for sphere in spheres_copy:
            e.add_sphere(vamp.Sphere(sphere, radius))
        environments["spheres"] = (e, 0.)

        cloud = np.vstack(
            [
                vamp.pointcloud.sphere_sample_surface(center, radius, points_per_sphere, 0.0)
                for center in spheres_copy
                ]
            ).astype(np.float32)
        e = vamp.Environment()
        build_ns = e.add_pointcloud(cloud.tolist(), r_min, r_max, point_radius)
        environments["pointcloud"] = (e, build_ns / 1e6)

        distances = sphere_distances(spheres_copy, radius, origin, spacing, shape)
        start = time.perf_counter()
        grid = vamp.DistanceGrid(distances, origin, [spacing] * 3)
        e = vamp.Environment()
        e.add_distance_grid(grid)
        environments["distance_grid"] = (e, (time.perf_counter() - start) * 1e3)

        configurations = rng.uniform(-np.pi, np.pi, size = (n_validate, len(a)))
        truth = np.array([vamp_module.validate(q, environments["spheres"][0]) for q in configurations])
        for name, (e, build_ms) in environments.items():
            if not (vamp_module.validate(a, e) and vamp_module.validate(b, e)):
                continue

            result = planner_func(a, b, e, plan_settings, sampler)
            simple = vamp_module.simplify(result.path, e, simp_settings, sampler)
            row = vamp.results_to_dict(result, simple)
            row["environment"] = name
            row["build_time"] = build_ms
            valid = np.array([vamp_module.validate(q, e) for q in configurations])
            row["agreement"] = float((valid == truth).mean())
            results.append(row)

    df = pd.DataFrame.from_dict(results)

    # Convert to microseconds
    df["planning_time"] = df["planning_time"].dt.microseconds
    df["simplification_time"] = df["simplification_time"].dt.microseconds

    print(f"Distance grid: {shape} points at {spacing} m")
    for name, group in df.groupby("environment", sort = False):
        print(f"\n{name}: build time in ms, planning and simplification times in us")
        print(
            group[[
                "build_time",
                "planning_time",
                "simplification_time",
                "planning_iterations",
                "agreement",
                ]].describe().to_string()
            )


if __name__ == "__main__":
    Fire(main)
