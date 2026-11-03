#pragma once

#include <cstdint>
#include <limits>

#include <vamp/collision/shapes.hh>
#include <vamp/collision/math.hh>

namespace vamp::collision
{
    // The signed distance from a sphere to the obstacles of one field of a distance grid: the
    // trilinear distance at the sphere's centre minus its radius, negative in collision. Outside
    // the grid's box, the distance at the centre is minus the distance to the box, so a sphere
    // whose centre leaves the box always collides.
    template <typename DataT>
    inline constexpr auto sphere_distance_grid(
        const DistanceGrid<DataT> &g,
        const float *field,
        const DataT &x,
        const DataT &y,
        const DataT &z,
        const DataT &r) noexcept -> DataT
    {
        using IndexT = IntVector<DataT::num_scalars_per_row, DataT::num_rows>;

        // Lattice coordinates, the cell's lower lattice point and the fractions in the cell.
        const auto lx = static_cast<float>(g.nx - 1);
        const auto ly = static_cast<float>(g.ny - 1);
        const auto lz = static_cast<float>(g.nz - 1);
        const auto cx = (x - g.x) / g.xs;
        const auto cy = (y - g.y) / g.ys;
        const auto cz = (z - g.z) / g.zs;
        const auto kx = cx.clamp(0.F, lx);
        const auto ky = cy.clamp(0.F, ly);
        const auto kz = cz.clamp(0.F, lz);
        const auto fx = kx.floor().clamp(0.F, lx - 1.F);
        const auto fy = ky.floor().clamp(0.F, ly - 1.F);
        const auto fz = kz.floor().clamp(0.F, lz - 1.F);
        const auto ax = kx - fx;
        const auto ay = ky - fy;
        const auto az = kz - fz;

        // Integer indices: float indices lose exactness above 2^24 points.
        const std::int32_t step_y = g.nz;
        const std::int32_t step_x = g.ny * g.nz;
        const IndexT base =
            (fx.template to<IndexT>() * g.ny + fy.template to<IndexT>()) * g.nz + fz.template to<IndexT>();

        const auto bx = 1.F - ax;
        const auto by = 1.F - ay;
        const auto bz = 1.F - az;
        const auto w00 = bx * by;
        const auto w01 = bx * ay;
        const auto w10 = ax * by;
        const auto w11 = ax * ay;
        const auto value = w00 * bz * DataT::gather(field, base) + w00 * az * DataT::gather(field, base + 1) +
                           w01 * bz * DataT::gather(field, base + step_y) +
                           w01 * az * DataT::gather(field, base + (step_y + 1)) +
                           w10 * bz * DataT::gather(field, base + step_x) +
                           w10 * az * DataT::gather(field, base + (step_x + 1)) +
                           w11 * bz * DataT::gather(field, base + (step_x + step_y)) +
                           w11 * az * DataT::gather(field, base + (step_x + step_y + 1));

        // Outside the box: minus the distance to it.
        const auto inside = (cx >= 0.F) & (cx <= lx) & (cy >= 0.F) & (cy <= ly) & (cz >= 0.F) & (cz <= lz);
        const auto ox = (g.x - x).max(x - g.xu).max(0.F);
        const auto oy = (g.y - y).max(y - g.yu).max(0.F);
        const auto oz = (g.z - z).max(z - g.zu).max(0.F);
        const auto outside = -(ox * ox + oy * oy + oz * oz).max(std::numeric_limits<float>::min()).sqrt();

        return outside.blend(value, inside) - r;
    }
}  // namespace vamp::collision
