#include <AMReX_Orientation.H>

#include "amr-wind/equation_systems/sdr/source_terms/KOmegaSSTTerrainSDRSrc.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/turbulence/TurbulenceModel.H"
#include "amr-wind/wind_energy/MOData.H"
#include "amr-wind/utilities/constants.H"
#include "amr-wind/utilities/linear_interpolation.H"

namespace amr_wind::pde::tke {

KOmegaSSTTerrainSDRSrc::KOmegaSSTTerrainSDRSrc(const CFDSim& sim)
    : KransTerrainBase(sim)
    , m_sdr_src(sim.repo().get_field("omega_src"))
    , m_sdr_diss(sim.repo().get_field("sdr_dissipation"))
    , m_sdr(sim.repo().get_field("sdr"))
{}

KOmegaSSTTerrainSDRSrc::~KOmegaSSTTerrainSDRSrc() = default;

void KOmegaSSTTerrainSDRSrc::parse_coeffs()
{
    amrex::ParmParse pp("ABL");
    pp.query("Cmu", m_Cmu);
}

void KOmegaSSTTerrainSDRSrc::operator()(
    const int lev,
    const amrex::MFIter& mfi,
    const amrex::Box& bx,
    const FieldState fstate,
    const amrex::Array4<amrex::Real>& src_term) const
{

    const amrex::Real Cmu = m_Cmu;
    const amrex::Real kappa = m_kappa;
    amrex::Real z0 = m_z0;
    const amrex::Real z0_min = 1e-4;
    const auto& dt = this->m_sim.time().delta_t();

    const auto& geom = this->m_sim.mesh().Geom(lev);
    const auto& dx = geom.CellSizeArray();
    const auto& prob_lo = geom.ProbLoArray();
    const auto& prob_hi = geom.ProbHiArray();

    const auto& sdr_arr = (this->m_sdr)(lev).array(mfi);
    const auto& sdr_src_arr = (this->m_sdr_src)(lev).array(mfi);
    const auto& sdr_diss_arr = (this->m_sdr_diss)(lev).array(mfi);

    const auto& vel = this->m_sim.repo()
                          .get_field("velocity")
                          .state(field_impl::dof_state(fstate))(lev)
                          .const_array(mfi);

    const bool has_terrain =
        this->m_sim.repo().int_field_exists("terrain_blank");

    const bool has_roughness = this->m_sim.repo().field_exists("terrainz0");

    const auto* m_terrain_blank =
        has_terrain ? &this->m_sim.repo().get_int_field("terrain_blank")
                    : nullptr;

    const auto* m_terrain_drag =
        has_terrain ? &this->m_sim.repo().get_int_field("terrain_drag")
                    : nullptr;

    const auto* m_terrain_height =
        has_terrain ? &this->m_sim.repo().get_field("terrain_height") : nullptr;

    const auto* m_terrain_z0 =
        has_roughness ? &this->m_sim.repo().get_field("terrainz0") : nullptr;

    const auto& blank_arr = has_terrain
                                ? (*m_terrain_blank)(lev).const_array(mfi)
                                : amrex::Array4<const int>();

    const auto& drag_arr = has_terrain ? (*m_terrain_drag)(lev).const_array(mfi)
                                       : amrex::Array4<const int>();

    const auto& terrain_height_arr =
        has_terrain ? (*m_terrain_height)(lev).const_array(mfi)
                    : amrex::Array4<const double>();

    const auto& terrainz0_arr = has_roughness
                                    ? (*m_terrain_z0)(lev).const_array(mfi)
                                    : amrex::Array4<const double>();

    const amrex::Real sponge_start = m_meso_start;
    const auto vsize = m_wind_heights_d.size();
    const auto* wind_heights_d = m_wind_heights_d.data();
    const auto* sdr_values_d = m_sdr_values_d.data();

    const amrex::Real psi_m_non_neutral_neighbour =
        (m_wall_het_model == "mol")
            ? MOData::calc_psi_m(
                  1.5 * dx[2] / m_monin_obukhov_length, m_beta_m, m_gamma_m)
            : 0.0;
    const amrex::Real psi_m_non_neutral_cell =
        (m_wall_het_model == "mol")
            ? MOData::calc_psi_m(
                  0.5 * dx[2] / m_monin_obukhov_length, m_beta_m, m_gamma_m)
            : 0.0;

    const amrex::Real phi_eps_non_neutral_cell =
        (m_wall_het_model == "mol")
            ? MOData::calc_phi_eps(
                  0.5 * dx[2] / m_monin_obukhov_length, m_beta_m, m_gamma_m)
            : 0.0;

    // TODO: Not sure why we do this in k-omega sst
    const amrex::Real factor = (fstate == FieldState::NPH) ? 0.5 : 1.0;

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        amrex::Real bc_forcing = 0;
        const amrex::Real cell_z0 =
            has_roughness ? std::max(terrainz0_arr(i, j, k), z0_min) : z0;
        const amrex::Real terrain_height =
            has_terrain ? terrain_height_arr(i, j, k) : prob_lo[2];
        if (has_terrain && drag_arr(i, j, k) == 1) {
            const amrex::Real ux2r = vel(i, j, k + 1, 0);
            const amrex::Real uy2r = vel(i, j, k + 1, 1);
            const amrex::Real m2 = std::sqrt(ux2r * ux2r + uy2r * uy2r);
            const amrex::Real ustar =
                m2 * kappa /
                (std::log(1.5 * dx[2] / cell_z0) - psi_m_non_neutral_neighbour);

            const amrex::Real phi_m_non_neutral_cell =
                std::log(0.5 * dx[2] / cell_z0) - psi_m_non_neutral_cell;

            const amrex::Real eps = std::pow(ustar, 3) *
                                    phi_eps_non_neutral_cell /
                                    (kappa * 0.5 * dx[2]);
            const amrex::Real tke = std::sqrt(
                (kappa * ustar * 0.5 * dx[2] * eps) /
                (Cmu * phi_m_non_neutral_cell));
            const amrex::Real sdrTarget =
                eps / Cmu / amrex::max<amrex::Real>(tke, 1e-10);

            bc_forcing = -(sdrTarget - sdr_arr(i, j, k)) / dt;
        }
        // Target sdr intended for within terrain
        const amrex::Real target_sdr = sdr_arr(i, j, k);

        const amrex::Real z = std::max(
            prob_lo[2] + (k + 0.5) * dx[2] - terrain_height, 0.5 * dx[2]);

        const amrex::Real zi =
            std::max((z - sponge_start) / (prob_hi[2] - sponge_start), 0.0);

        amrex::Real ref_sdr = sdr_arr(i, j, k);
        if (zi > 0) {
            ref_sdr = (vsize > 0) ? interp::linear(
                                        wind_heights_d, wind_heights_d + vsize,
                                        sdr_values_d, z)
                                  : ref_sdr;
        }
        src_term(i, j, k) -= 1.0 / dt * (sdr_arr(i, j, k) - ref_sdr);

        src_term(i, j, k) -=
            (sdr_arr(i, j, k) - target_sdr) * blank_arr(i, j, k) +
            bc_forcing * drag_arr(i, j, k);

        src_term(i, j, k) +=
            (1 - blank_arr(i, j, k)) *
            (factor * sdr_diss_arr(i, j, k) + sdr_src_arr(i, j, k));
    });

    // if (m_horizontal_sponge) {
    //     apply_horizontal_sponge(
    //         bx, prob_lo.data(), prob_hi.data(), dx.data(), dt,
    //         m_wind_heights_d, m_sdr_values_d, sdr_arr, src_term);
    // }
}

} // namespace amr_wind::pde::tke
