
#include "amr-wind/turbulence/RANS/KOmegaSSTTerrainBase.H"
#include <fstream>
#include <AMReX_ParmParse.H>

#include "amr-wind/utilities/linear_interpolation.H"

namespace amr_wind::pde {

KwSSTTerrainBase::KwSSTTerrainBase(const CFDSim& sim)
    : m_sim(sim)
    , m_terrain_height(sim.repo().get_field("terrain_height"))
    , m_terrain_z0(sim.repo().get_field("terrain_z0"))
{
    parse_coeffs();
}

void KwSSTTerrainBase::parse_coeffs()
{
    amrex::ParmParse pp("ABL");
    pp.query("Cmu", m_Cmu);
    pp.query("kappa", m_kappa);
    pp.query("surface_roughness_z0", m_z0);
    pp.query("surface_temp_flux", m_heat_flux);
    pp.query("meso_sponge_start", m_meso_start);
    pp.query("rans_1dprofile_file", m_1d_rans);
    pp.query("horizontal_sponge_tke", m_horizontal_sponge);
    pp.query("wall_het_model", m_wall_het_model);
    pp.query("monin_obukhov_length", m_monin_obukhov_length);
    pp.query("mo_gamma_m", m_gamma_m);
    pp.query("mo_beta_m", m_beta_m);

    amrex::ParmParse pp_incflow("incflo");
    pp_incflow.queryarr("gravity", m_gravity);

    amrex::ParmParse pp_drag("DragForcing");
    pp_drag.query("sponge_strength", m_sponge_strength);
    pp_drag.query("sponge_distance_west", m_sponge_distance_west);
    pp_drag.query("sponge_distance_east", m_sponge_distance_east);
    pp_drag.query("sponge_distance_south", m_sponge_distance_south);
    pp_drag.query("sponge_distance_north", m_sponge_distance_north);
    pp_drag.query("sponge_west", m_sponge_west);
    pp_drag.query("sponge_east", m_sponge_east);
    pp_drag.query("sponge_south", m_sponge_south);
    pp_drag.query("sponge_north", m_sponge_north);

    if (!m_1d_rans.empty()) {
        load_1d_rans_profile();
    }
}

void KwSSTTerrainBase::load_1d_rans_profile()
{
    std::ifstream ransfile(m_1d_rans, std::ios::in);
    if (!ransfile.good()) {
        amrex::Abort("Cannot find 1-D RANS profile file " + m_1d_rans);
    }
    amrex::Real value1, value2, value3, value4, value5, value6;
    while (ransfile >> value1 >> value2 >> value3 >> value4 >> value5 >>
           value6) {
        m_wind_heights.push_back(value1);
        m_tke_values.push_back(value5);
        m_sdr_values.push_back(value6);
    }
    int num_wind_values = static_cast<int>(m_wind_heights.size());
    m_wind_heights_d.resize(num_wind_values);
    m_tke_values_d.resize(num_wind_values);
    m_sdr_values_d.resize(num_wind_values);
    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice, m_wind_heights.begin(), m_wind_heights.end(),
        m_wind_heights_d.begin());
    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice, m_tke_values.begin(), m_tke_values.end(),
        m_tke_values_d.begin());
    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice, m_sdr_values.begin(), m_sdr_values.end(),
        m_sdr_values_d.begin());
}

void KwSSTTerrainBase::apply_horizontal_sponge(
    const amrex::Box& bx,
    const amrex::Real* problo,
    const amrex::Real* probhi,
    const amrex::Real* dx,
    amrex::Real dt,
    const amrex::Gpu::DeviceVector<amrex::Real>& wind_heights_d,
    const amrex::Gpu::DeviceVector<amrex::Real>& values_d,
    const amrex::Array4<const amrex::Real>& field_arr,
    const amrex::Array4<amrex::Real>& src_term) const
{
    const auto vsize = m_wind_heights_d.size();
    const amrex::Real sponge_strength = m_sponge_strength;
    const amrex::Real start_east = probhi[0] - m_sponge_distance_east;
    const amrex::Real start_west = problo[0] - m_sponge_distance_west;
    const amrex::Real start_north = probhi[1] - m_sponge_distance_north;
    const amrex::Real start_south = problo[1] - m_sponge_distance_south;
    const int sponge_east = m_sponge_east;
    const int sponge_west = m_sponge_west;
    const int sponge_south = m_sponge_south;
    const int sponge_north = m_sponge_north;
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
        const amrex::Real y = problo[1] + (j + 0.5) * dx[1];
        const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
        amrex::Real xstart_damping = 0;
        amrex::Real ystart_damping = 0;
        amrex::Real xend_damping = 0;
        amrex::Real yend_damping = 0;
        amrex::Real xi_end = (x - start_east) / (probhi[0] - start_east);
        amrex::Real xi_start = (start_west - x) / (start_west - problo[0]);
        xi_start = sponge_west * std::max(xi_start, 0.0);
        xi_end = sponge_east * std::max(xi_end, 0.0);
        xi_start /= (xi_start + amr_wind::constants::EPS);
        xi_end /= (xi_end + amr_wind::constants::EPS);
        xstart_damping = sponge_west * sponge_strength * xi_start * xi_start;
        xend_damping = sponge_east * sponge_strength * xi_end * xi_end;
        amrex::Real yi_end = (y - start_north) / (probhi[1] - start_north);
        amrex::Real yi_start = (start_south - y) / (start_south - problo[1]);
        yi_start = sponge_south * std::max(yi_start, 0.0);
        yi_end = sponge_north * std::max(yi_end, 0.0);
        yi_start /= (yi_start + amr_wind::constants::EPS);
        yi_end /= (yi_end + amr_wind::constants::EPS);
        ystart_damping = sponge_strength * yi_start * yi_start;
        yend_damping = sponge_strength * yi_end * yi_end;
        const amrex::Real ref_value =
            (vsize > 0) ? interp::linear(
                              wind_heights_d.data(),
                              wind_heights_d.data() + vsize, values_d.data(), z)
                        : field_arr(i, j, k, 0);
        const amrex::Real damping_sum =
            (xstart_damping + xend_damping + ystart_damping + yend_damping +
             amr_wind::constants::EPS);
        const amrex::Real sponge_forcing =
            (xstart_damping + xend_damping + ystart_damping + yend_damping) /
            (damping_sum * dt) * (field_arr(i, j, k) - ref_value);
        src_term(i, j, k, 0) -= sponge_forcing;
    });
}

} // namespace amr_wind::pde