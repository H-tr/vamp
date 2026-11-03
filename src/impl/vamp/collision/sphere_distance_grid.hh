#pragma once

#include <cstddef>
#include <cstdint>

#include <vamp/collision/shapes.hh>
#include <vamp/collision/math.hh>

namespace vamp::collision
{
    // The clearance of a sphere in field ``index`` of a distance grid: the trilinear distance at
    // its centre minus its radius, and at most the sphere's room inside the field's workspace, so
    // it is negative in collision. The workspace takes the whole sphere, so a free bounding ball
    // keeps the spheres it holds inside the workspace (robots check a link's bounding ball first).
    template <typename DataT>
    inline constexpr auto sphere_distance_grid(
        const DistanceGrid<DataT> &g,
        std::size_t index,
        const DataT &x,
        const DataT &y,
        const DataT &z,
        const DataT &r) noexcept -> DataT
    {
        using IndexT = IntVector<DataT::num_scalars_per_row, DataT::num_rows>;
        const float *field = g.field(index);
        const auto &box = g.workspace(index);

        // Lattice coordinates clamped to the lattice, the cell's lower lattice point and the
        // fractions in the cell.
        const auto lx = static_cast<float>(g.nx - 1);
        const auto ly = static_cast<float>(g.ny - 1);
        const auto lz = static_cast<float>(g.nz - 1);
        const auto kx = ((x - g.x) / g.xs).clamp(0.F, lx);
        const auto ky = ((y - g.y) / g.ys).clamp(0.F, ly);
        const auto kz = ((z - g.z) / g.zs).clamp(0.F, lz);
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

        // The smallest of the clearances, with a blend so that a face at infinity stays inert.
        auto clearance = value - r;
        for (const auto &room :
             {x - box[0] - r, y - box[1] - r, z - box[2] - r, box[3] - x - r, box[4] - y - r, box[5] - z - r})
        {
            clearance = clearance.blend(room, room < clearance);
        }

        return clearance;
    }
}  // namespace vamp::collision
