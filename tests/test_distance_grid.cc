#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <vamp/vector.hh>
#include <vamp/collision/environment.hh>
#include <vamp/collision/factory.hh>
#include <vamp/collision/sphere_distance_grid.hh>
#include <vamp/collision/validity.hh>

using Catch::Approx;
using vamp::FloatVector;
using vamp::FloatVectorWidth;
namespace vc = vamp::collision;
namespace vf = vamp::collision::factory;

namespace
{
    using V = FloatVector<FloatVectorWidth>;
    using Lanes = std::array<float, V::num_scalars_rounded>;

    // An anisotropic lattice with a different size along each axis, so that a mixed-up stride shows.
    constexpr std::array<float, 3> origin = {-1.F, 0.5F, -0.25F};
    constexpr std::array<float, 3> spacing = {0.1F, 0.2F, 0.05F};
    constexpr std::array<std::size_t, 3> size = {7, 5, 11};

    auto lattice_point(std::size_t i, std::size_t j, std::size_t k) -> std::array<float, 3>
    {
        return {
            origin[0] + spacing[0] * static_cast<float>(i),
            origin[1] + spacing[1] * static_cast<float>(j),
            origin[2] + spacing[2] * static_cast<float>(k)};
    }

    // Values of a field at the lattice points, row-major with the z index contiguous.
    template <typename F>
    auto sample_field(F &&f) -> std::vector<float>
    {
        std::vector<float> values;
        for (auto i = 0U; i < size[0]; ++i)
        {
            for (auto j = 0U; j < size[1]; ++j)
            {
                for (auto k = 0U; k < size[2]; ++k)
                {
                    const auto p = lattice_point(i, j, k);
                    values.emplace_back(f(p[0], p[1], p[2]));
                }
            }
        }

        return values;
    }

    auto upper(std::size_t axis) -> float
    {
        return origin[axis] + spacing[axis] * static_cast<float>(size[axis] - 1);
    }

    // Reference trilinear distance in double precision, read at the nearest lattice point outside
    // the lattice.
    auto reference(const std::vector<float> &values, double x, double y, double z) -> double
    {
        const std::array<double, 3> p = {x, y, z};
        std::array<std::size_t, 3> lower{};
        std::array<double, 3> fraction{};
        for (auto axis = 0U; axis < 3; ++axis)
        {
            const double last = static_cast<double>(size[axis] - 1);
            const double c = (p[axis] - origin[axis]) / spacing[axis];
            const double k = std::clamp(c, 0.0, last);
            const double f = std::min(std::floor(k), last - 1.0);
            lower[axis] = static_cast<std::size_t>(f);
            fraction[axis] = k - f;
        }

        double value = 0.0;
        for (auto corner = 0U; corner < 8; ++corner)
        {
            const std::array<std::size_t, 3> bit = {(corner >> 2) & 1U, (corner >> 1) & 1U, corner & 1U};
            double weight = 1.0;
            for (auto axis = 0U; axis < 3; ++axis)
            {
                weight *= bit[axis] ? fraction[axis] : 1.0 - fraction[axis];
            }

            const auto index =
                ((lower[0] + bit[0]) * size[1] + (lower[1] + bit[1])) * size[2] + lower[2] + bit[2];
            value += weight * values[index];
        }

        return value;
    }

    // Reference clearance: the distance minus the radius, at most the room inside the lattice's box.
    auto reference_clearance(const std::vector<float> &values, double x, double y, double z, double r)
        -> double
    {
        const std::array<double, 3> p = {x, y, z};
        double clearance = reference(values, x, y, z) - r;
        for (auto axis = 0U; axis < 3; ++axis)
        {
            clearance = std::min({clearance, p[axis] - origin[axis] - r, upper(axis) - p[axis] - r});
        }

        return clearance;
    }

    using Workspace = vc::DistanceGrid<float>::Workspace;
    constexpr auto inf = std::numeric_limits<float>::infinity();
    constexpr Workspace unbounded = {-inf, -inf, -inf, inf, inf, inf};

    auto make_grid(std::vector<float> values, std::vector<Workspace> workspaces = {})
        -> vc::DistanceGrid<float>
    {
        return vf::distance_grid::array(origin, spacing, size, std::move(values), std::move(workspaces));
    }

    auto clearances(
        const vc::DistanceGrid<V> &grid,
        std::size_t field,
        const Lanes &xs,
        const Lanes &ys,
        const Lanes &zs,
        const Lanes &rs) -> Lanes
    {
        return vc::sphere_distance_grid(grid, field, V(xs.data()), V(ys.data()), V(zs.data()), V(rs.data()))
            .to_array();
    }
}  // namespace

TEST_CASE("Distance grid interpolation is exact for linear fields", "[collision][distance_grid]")
{
    const auto linear = [](float x, float y, float z) { return 0.3F * x - 0.7F * y + 1.1F * z + 0.2F; };
    const auto grid = make_grid(sample_field(linear), {unbounded});
    const vc::DistanceGrid<V> simd_grid(grid);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> ux(origin[0], upper(0));
    std::uniform_real_distribution<float> uy(origin[1], upper(1));
    std::uniform_real_distribution<float> uz(origin[2], upper(2));
    std::uniform_real_distribution<float> ur(0.F, 0.3F);
    for (auto i = 0U; i < 1000; ++i)
    {
        Lanes xs, ys, zs, rs;
        for (auto lane = 0U; lane < xs.size(); ++lane)
        {
            xs[lane] = ux(rng);
            ys[lane] = uy(rng);
            zs[lane] = uz(rng);
            rs[lane] = ur(rng);
        }

        const auto actual = clearances(simd_grid, 0, xs, ys, zs, rs);
        for (auto lane = 0U; lane < xs.size(); ++lane)
        {
            REQUIRE(actual[lane] == Approx(linear(xs[lane], ys[lane], zs[lane]) - rs[lane]).margin(1e-5));
        }
    }
}

TEST_CASE("Distance grid SIMD lanes match a scalar trilinear reference", "[collision][distance_grid]")
{
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> value(-0.5F, 1.F);
    const auto values = sample_field([&](float, float, float) { return value(rng); });
    const auto grid = make_grid(values);
    const vc::DistanceGrid<V> simd_grid(grid);

    // Queries reach past every face of the lattice, and some land exactly on lattice points.
    std::uniform_real_distribution<float> qx(origin[0] - 0.3F, upper(0) + 0.3F);
    std::uniform_real_distribution<float> qy(origin[1] - 0.3F, upper(1) + 0.3F);
    std::uniform_real_distribution<float> qz(origin[2] - 0.3F, upper(2) + 0.3F);
    std::uniform_real_distribution<float> ur(0.F, 0.3F);
    std::uniform_int_distribution<std::size_t> pick(0, 3);
    std::size_t num_outside = 0;
    std::size_t num_colliding = 0;
    std::size_t num_free = 0;
    for (auto i = 0U; i < 2000; ++i)
    {
        Lanes xs, ys, zs, rs;
        for (auto lane = 0U; lane < xs.size(); ++lane)
        {
            xs[lane] = qx(rng);
            ys[lane] = qy(rng);
            zs[lane] = qz(rng);
            rs[lane] = ur(rng);
            if (pick(rng) == 0)
            {
                const auto p = lattice_point(pick(rng) % size[0], pick(rng) % size[1], pick(rng) % size[2]);
                xs[lane] = p[0];
                ys[lane] = p[1];
                zs[lane] = p[2];
            }
        }

        const auto actual = clearances(simd_grid, 0, xs, ys, zs, rs);
        for (auto lane = 0U; lane < xs.size(); ++lane)
        {
            const auto expected = reference_clearance(values, xs[lane], ys[lane], zs[lane], rs[lane]);
            CAPTURE(xs[lane], ys[lane], zs[lane], rs[lane], actual[lane], expected);
            REQUIRE(actual[lane] == Approx(expected).margin(1e-5));
            if (std::abs(expected) > 1e-5)
            {
                REQUIRE((actual[lane] < 0.F) == (expected < 0.0));
            }

            num_outside += xs[lane] < origin[0] or xs[lane] > upper(0) or zs[lane] > upper(2);
            (expected < 0.0 ? num_colliding : num_free) += 1;
        }
    }

    // Make sure the fuzzing covers the faces and both outcomes
    REQUIRE(num_outside > 100);
    REQUIRE(num_colliding > 100);
    REQUIRE(num_free > 100);
}

TEST_CASE("Spheres collide unless they lie wholly inside the workspace", "[collision][distance_grid]")
{
    vc::Environment<float> env;
    env.distance_grids.emplace_back(make_grid(sample_field([](float, float, float) { return 10.F; })));
    const vc::Environment<V> simd_env(env);

    const auto collides = [&](const vc::Environment<V> &e, float x, float y, float z, float r)
    { return vamp::sphere_environment_in_collision(e, V::fill(x), V::fill(y), V::fill(z), r); };

    // Far from every obstacle: free while the whole sphere stays in the lattice's box.
    const float r = 0.04F;
    REQUIRE_FALSE(collides(simd_env, origin[0] + r + 1e-5F, origin[1] + r + 1e-5F, origin[2] + r + 1e-5F, r));
    REQUIRE_FALSE(collides(simd_env, upper(0) - r - 1e-5F, upper(1) - r - 1e-5F, upper(2) - r - 1e-5F, r));
    REQUIRE(collides(simd_env, origin[0] + r - 1e-3F, 0.8F, 0.F, r));
    REQUIRE(collides(simd_env, upper(0) - r + 1e-3F, 0.8F, 0.F, r));
    REQUIRE(collides(simd_env, -0.5F, origin[1] + r - 1e-3F, 0.F, r));
    REQUIRE(collides(simd_env, -0.5F, upper(1) - r + 1e-3F, 0.F, r));
    REQUIRE(collides(simd_env, -0.5F, 0.8F, origin[2] + r - 1e-3F, r));
    REQUIRE(collides(simd_env, -0.5F, 0.8F, upper(2) - r + 1e-3F, r));

    // A workspace open below and at the sides: a sphere below the lattice reads its bottom layer,
    // and the top face still holds.
    vc::Environment<float> open;
    const Workspace below = {-inf, -inf, -inf, inf, inf, upper(2)};
    open.distance_grids.emplace_back(
        make_grid(sample_field([](float, float, float z) { return 1.F + z; }), {below}));
    const vc::Environment<V> simd_open(open);
    REQUIRE_FALSE(collides(simd_open, -0.5F, 0.8F, -3.F, 0.7F));
    REQUIRE(collides(simd_open, -0.5F, 0.8F, -3.F, 0.8F));
    REQUIRE(collides(simd_open, -0.5F, 0.8F, upper(2) - 0.05F, 0.06F));
}

TEST_CASE("Spheres read the distance grid field that the robot selects", "[collision][distance_grid]")
{
    // Field 0: a plane at z = 0 below free space; field 1: the same scene without the plane.
    auto values = sample_field([](float, float, float z) { return z; });
    const auto without_plane = sample_field([](float, float, float) { return 10.F; });
    values.insert(values.end(), without_plane.begin(), without_plane.end());

    vc::Environment<float> env;
    env.distance_grids.emplace_back(make_grid(values));
    REQUIRE(env.distance_grids.front().num_fields == 2);
    const vc::Environment<V> simd_env(env);

    const auto x = V::fill(-0.6F);
    const auto y = V::fill(0.8F);
    const auto z = V::fill(0.05F);
    REQUIRE(vamp::sphere_environment_in_collision(simd_env, x, y, z, 0.1F));
    REQUIRE(vamp::sphere_environment_in_collision<0>(simd_env, x, y, z, 0.1F));
    REQUIRE_FALSE(vamp::sphere_environment_in_collision<1>(simd_env, x, y, z, 0.1F));
    REQUIRE_FALSE(vamp::sphere_environment_in_collision(simd_env, x, y, z, 0.04F));

    // A grid without the selected field gives field 0.
    vc::Environment<float> single;
    single.distance_grids.emplace_back(make_grid(sample_field([](float, float, float z) { return z; })));
    const vc::Environment<V> simd_single(single);
    REQUIRE(vamp::sphere_environment_in_collision<1>(simd_single, x, y, z, 0.1F));

    // Debug queries name the grid that a sphere collides with.
    env.distance_grids.front().name = "scene";
    const vc::Environment<V> named(env);
    const std::vector<std::string> scene = {"scene"};
    REQUIRE(vamp::sphere_environment_get_collisions<V>(named, x, y, z, 0.1F) == scene);
}

TEST_CASE("Vector environments share the distance grid values", "[collision][distance_grid]")
{
    vc::Environment<float> env;
    env.distance_grids.emplace_back(make_grid(sample_field([](float x, float, float) { return x; })));
    const vc::Environment<V> simd_env(env);
    REQUIRE(simd_env.distance_grids.front().data.get() == env.distance_grids.front().data.get());
    REQUIRE(simd_env.distance_grids.front().field(0) == env.distance_grids.front().field(0));
}

TEST_CASE("Distance grid factory rejects malformed grids", "[collision][distance_grid]")
{
    const auto values = sample_field([](float, float, float) { return 1.F; });
    REQUIRE_NOTHROW(make_grid(values));

    auto too_few = values;
    too_few.pop_back();
    REQUIRE_THROWS_AS(make_grid(too_few), std::invalid_argument);
    REQUIRE_THROWS_AS(make_grid({}), std::invalid_argument);

    auto infinite = values;
    infinite[3] = std::numeric_limits<float>::infinity();
    REQUIRE_THROWS_AS(make_grid(infinite), std::invalid_argument);

    REQUIRE_THROWS_AS(
        vf::distance_grid::array(
            origin, spacing, {1, size[1], size[2]}, std::vector<float>(size[1] * size[2])),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        vf::distance_grid::array(origin, {0.1F, 0.F, 0.1F}, size, values), std::invalid_argument);
    REQUIRE_THROWS_AS(
        vf::distance_grid::array({std::numeric_limits<float>::quiet_NaN(), 0.F, 0.F}, spacing, size, values),
        std::invalid_argument);

    // One workspace for each field, each with lower <= upper and no NaN.
    REQUIRE_NOTHROW(make_grid(values, {unbounded}));
    REQUIRE_THROWS_AS(make_grid(values, {unbounded, unbounded}), std::invalid_argument);
    REQUIRE_THROWS_AS(make_grid(values, {{0.F, 0.F, 1.F, 1.F, 1.F, 0.F}}), std::invalid_argument);
    REQUIRE_THROWS_AS(
        make_grid(values, {{std::numeric_limits<float>::quiet_NaN(), 0.F, 0.F, 1.F, 1.F, 1.F}}),
        std::invalid_argument);
    REQUIRE(
        make_grid(values).workspaces ==
        std::vector<Workspace>{{origin[0], origin[1], origin[2], upper(0), upper(1), upper(2)}});
}
