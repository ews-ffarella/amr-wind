#ifndef KRANSTERRAINBASE_H
#define KRANSTERRAINBASE_H

#include "amr-wind/CFDSim.H"
#include "amr-wind/wind_energy/MOData.H"
#include "amr-wind/utilities/constants.H"
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <string>

namespace amr_wind::pde {

class KransTerrainBase
{

protected:
    const CFDSim& m_sim;
    bool m_is_sdr{true}; // true for SDR, false for epsilon

    amrex::Real m_heat_flux{0.0};
    amrex::Real m_z0{0.1};
    amrex::Real m_kappa{0.41};
    amrex::Vector<amrex::Real> m_gravity{0.0, 0.0, -9.81};
    std::string m_wall_het_model{"none"};
    amrex::Real m_monin_obukhov_length{amr_wind::constants::LARGE_NUM};
    amrex::Real m_gamma_m{5.0};
    amrex::Real m_beta_m{16.0};

    amrex::Real m_meso_start{600};
    std::string m_1d_rans;
    amrex::Vector<amrex::Real> m_wind_heights;
    amrex::Vector<amrex::Real> m_tke_values;
    amrex::Vector<amrex::Real> m_sdr_values;
    amrex::Vector<amrex::Real> m_eps_values;
    amrex::Gpu::DeviceVector<amrex::Real> m_wind_heights_d;
    amrex::Gpu::DeviceVector<amrex::Real> m_tke_values_d;
    amrex::Gpu::DeviceVector<amrex::Real> m_sdr_values_d;
    amrex::Gpu::DeviceVector<amrex::Real> m_eps_values_d;
    bool m_horizontal_sponge{false};
    amrex::Real m_sponge_strength{1.0};
    amrex::Real m_sponge_distance_west{-1000};
    amrex::Real m_sponge_distance_east{1000};
    amrex::Real m_sponge_distance_south{-1000};
    amrex::Real m_sponge_distance_north{1000};
    int m_sponge_west{0};
    int m_sponge_east{1};
    int m_sponge_south{0};
    int m_sponge_north{1};

    KransTerrainBase(const CFDSim& sim, bool is_sdr = true);

    void parse_coeffs();

    // Utility: load 1D RANS profile (can be called by derived classes)
    void load_1d_rans_profile();

    void apply_horizontal_sponge(
        const amrex::Box& bx,
        const amrex::Real* problo,
        const amrex::Real* probhi,
        const amrex::Real* dx,
        amrex::Real dt,
        const amrex::Gpu::DeviceVector<amrex::Real>& wind_heights_d,
        const amrex::Gpu::DeviceVector<amrex::Real>& values_d, // tke or sdr
        const amrex::Array4<const amrex::Real>& field_arr, // tke_arr or sdr_arr
        const amrex::Array4<amrex::Real>& src_term) const;
};

} // namespace amr_wind::pde

#endif /* KRANSTERRAINBASE_H */