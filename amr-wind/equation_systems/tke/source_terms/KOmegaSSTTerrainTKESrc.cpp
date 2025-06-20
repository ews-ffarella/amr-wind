#include <AMReX_Orientation.H>

#include "amr-wind/equation_systems/tke/source_terms/KOmegaSSTTerrainTKESrc.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/turbulence/TurbulenceModel.H"
#include "amr-wind/wind_energy/MOData.H"
#include "amr-wind/utilities/constants.H"
#include "amr-wind/utilities/linear_interpolation.H"

namespace amr_wind::pde::tke {

KOmegaSSTTerrainTKESrc::KOmegaSSTTerrainTKESrc(const CFDSim& sim)
    : KransTerrainBase(sim)
    , m_shear_prod(sim.repo().get_field("shear_prod"))
    , m_diss(sim.repo().get_field("dissipation"))
    , m_buoy_term(sim.repo().get_field("buoyancy_term"))
    , m_tke(sim.repo().get_field("tke"))
{}

KOmegaSSTTerrainTKESrc::~KOmegaSSTTerrainTKESrc() = default;

void KOmegaSSTTerrainTKESrc::parse_coeffs()
{
    amrex::ParmParse pp("ABL");
    pp.query("Cmu", m_Cmu);
}

void KOmegaSSTTerrainTKESrc::operator()(
    const int lev,
    const amrex::MFIter& mfi,
    const amrex::Box& bx,
    const FieldState fstate,
    const amrex::Array4<amrex::Real>& src_term) const
{

    const amrex::Real z0_min = 1e-4;

    const auto& geom = this->m_sim.mesh().Geom(lev);
    const auto& problo = geom.ProbLoArray();
    const auto& probhi = geom.ProbHiArray();
    const auto& dx = geom.CellSizeArray();
    const auto& dt = this->m_sim.time().delta_t();
    const amrex::Real Cmu = this->m_Cmu;
    const amrex::Real kappa = this->m_kappa;
    amrex::Real z0 = std::max(this->m_z0, z0_min);

    const auto& tke_arr = (this->m_tke)(lev).array(mfi);
    const auto& shear_prod_arr = (this->m_shear_prod)(lev).array(mfi);
    const auto& dissip_arr = (this->m_diss)(lev).array(mfi);
    const auto& buoy_prod_arr = (this->m_buoy_term)(lev).array(mfi);

    const auto& vel = this->m_sim.repo()
                          .get_field("velocity")
                          .state(field_impl::dof_state(fstate))(lev)
                          .const_array(mfi);

    const bool has_terrain =
        this->m_sim.repo().int_field_exists("terrain_blank");

    const amrex::Real sponge_start = this->m_meso_start;
    const auto vsize = this->m_wind_heights_d.size();
    const auto* wind_heights_d = this->m_wind_heights_d.data();
    const auto* tke_values_d = this->m_tke_values_d.data();

    amrex::Real psi_m = 0.0;
    amrex::Real phi_e = 0.0;
    if (this->m_wall_het_model == "mol") {
        psi_m = MOData::calc_psi_m(
            1.5 * dx[2] / this->m_monin_obukhov_length, this->m_beta_m,
            this->m_gamma_m);
        phi_e = MOData::calc_phi_eps(
            1.5 * dx[2] / this->m_monin_obukhov_length, this->m_beta_m,
            this->m_gamma_m);
    }

    // TODO: Not sure why we do this in k-omega sst
    const amrex::Real factor = (fstate == FieldState::NPH) ? 0.5 : 1.0;

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        amrex::Real bcforcing = 0;
        const amrex::Real z = problo[2] + (k + 0.5) * dx[2];
        if (k == 0) {
            const amrex::Real ux = vel(i, j, k + 1, 0);
            const amrex::Real uy = vel(i, j, k + 1, 1);
            const amrex::Real m = std::sqrt(ux * ux + uy * uy);
            const amrex::Real ustar =
                m * kappa / (std::log(3 * z / z0) - psi_m);

            const amrex::Real phi_m = std::log(3 * z / z0) - psi_m;
            const amrex::Real eps =
                std::pow(ustar, 3) * phi_e / (kappa * 0.5 * dx[2]);
            const amrex::Real tke =
                std::sqrt((kappa * ustar * 0.5 * dx[2] * eps) / (Cmu * phi_m));
            bcforcing = (tke - tke_arr(i, j, k)) / dt;
            // amrex::AllPrint()
            //     << "bcforcing: " << bcforcing << " tke: " << tke_arr(i, j, k)
            //     << " eps: " << eps << " ustar: " << ustar << " ref_tke: " <<
            //     tke
            //     << " phi_m: " << phi_m << " z: " << z << " z0: " << z0
            //     << " psi_m: " << psi_m << " phi_e: " << phi_e << " ux: " <<
            //     ux
            //     << " uy: " << uy << " m: " << m << std::endl;
            dissip_arr(i, j, k) = eps;
        }
        amrex::Real ref_tke = tke_arr(i, j, k);
        amrex::Real zi =
            std::max((z - sponge_start) / (probhi[2] - sponge_start), 0.0);
        zi = -1;
        if (zi > 0) {
            ref_tke = (vsize > 0) ? interp::linear(
                                        wind_heights_d, wind_heights_d + vsize,
                                        tke_values_d, z)
                                  : tke_arr(i, j, k, 0);
        }
        const amrex::Real sponge_forcing =
            1.0 / dt * (tke_arr(i, j, k) - ref_tke);

        // dissip_arr(i, j, k) = std::pow(Cmu, 3) *
        //                       std::pow(tke_arr(i, j, k), 1.5) /
        //                       (tlscale_arr(i, j, k) +
        //                       amr_wind::constants::EPS);
        src_term(i, j, k) +=
            shear_prod_arr(i, j, k) + buoy_prod_arr(i, j, k) -
            factor * dissip_arr(i, j, k) -
            (1 - static_cast<int>(has_terrain)) * (sponge_forcing - bcforcing);
    });
    if (has_terrain) {
        const auto* const m_terrain_blank =
            &this->m_sim.repo().get_int_field("terrain_blank");
        const auto* const m_terrain_drag =
            &this->m_sim.repo().get_int_field("terrain_drag");
        auto* const m_terrain_height =
            &this->m_sim.repo().get_field("terrain_height");
        auto* const m_terrainz0 = &this->m_sim.repo().get_field("terrainz0");
        const auto& blank_arr = (*m_terrain_blank)(lev).const_array(mfi);
        const auto& drag_arr = (*m_terrain_drag)(lev).const_array(mfi);
        const auto& terrain_height = (*m_terrain_height)(lev).const_array(mfi);
        const auto& terrainz0 = (*m_terrainz0)(lev).const_array(mfi);
        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const amrex::Real cell_z0 =
                    drag_arr(i, j, k) * std::max(terrainz0(i, j, k), z0_min) +
                    (1 - drag_arr(i, j, k)) * z0;
                amrex::Real terrainforcing = 0;
                amrex::Real dragforcing = 0;
                amrex::Real ux = vel(i, j, k + 1, 0);
                amrex::Real uy = vel(i, j, k + 1, 1);
                amrex::Real z = 0.5 * dx[2];
                amrex::Real m = std::sqrt(ux * ux + uy * uy);
                const amrex::Real phi_m = std::log(3 * z / cell_z0) - psi_m;
                const amrex::Real ustar =
                    m * kappa / (std::log(3 * z / cell_z0) - psi_m);
                const amrex::Real eps =
                    std::pow(ustar, 3) * phi_e / (kappa * 0.5 * dx[2]);
                const amrex::Real tke = std::sqrt(
                    (kappa * ustar * 0.5 * dx[2] * eps) / (Cmu * phi_m));

                // const amrex::Real T0 = ref_theta_arr(i, j, k);
                // const amrex::Real hf = std::abs(gravity[2]) / T0 * heat_flux;
                // const amrex::Real rans_b = std::pow(
                //     std::max(hf, 0.0) * kappa * z / std::pow(Cmu, 3),
                //     (2.0 / 3.0));
                terrainforcing = (tke - tke_arr(i, j, k)) / dt;
                amrex::Real bcforcing = 0;
                if (k == 0) {
                    bcforcing = (1 - blank_arr(i, j, k)) * terrainforcing;
                }
                ux = vel(i, j, k, 0);
                uy = vel(i, j, k, 1);
                const amrex::Real uz = vel(i, j, k, 2);
                m = std::sqrt(ux * ux + uy * uy + uz * uz);
                const amrex::Real Cd = std::min(
                    10 / (dx[2] * m + amr_wind::constants::EPS), 100 / dx[2]);
                dragforcing = -Cd * m * tke_arr(i, j, k, 0);
                z = std::max(
                    problo[2] + (k + 0.5) * dx[2] - terrain_height(i, j, k),
                    0.5 * dx[2]);

                amrex::Real zi = std::max(
                    (z - sponge_start) / (probhi[2] - sponge_start), 0.0);
                zi = -1;
                amrex::Real ref_tke = tke_arr(i, j, k);
                if (zi > 0) {
                    ref_tke = (vsize > 0)
                                  ? interp::linear(
                                        wind_heights_d, wind_heights_d + vsize,
                                        tke_values_d, z)
                                  : tke_arr(i, j, k, 0);
                }
                const amrex::Real sponge_forcing =
                    1.0 / dt * (tke_arr(i, j, k) - ref_tke);
                src_term(i, j, k) =
                    (1 - blank_arr(i, j, k)) * src_term(i, j, k) +
                    drag_arr(i, j, k) * terrainforcing +
                    blank_arr(i, j, k) * dragforcing -
                    static_cast<int>(has_terrain) *
                        (sponge_forcing - bcforcing);

                if (blank_arr(i, j, k) == 2) {
                    amrex::AllPrint()
                        << "src_term: " << src_term(i, j, k)
                        << " terrainforcing: "
                        << drag_arr(i, j, k) * terrainforcing
                        << " bcforcing: " << bcforcing
                        << " dragforcing: " << blank_arr(i, j, k) * dragforcing
                        << " tke: " << tke_arr(i, j, k) << " eps: " << eps
                        << " ustar: " << ustar << " ref_tke: " << tke
                        << " phi_m: " << phi_m << " z: " << z << " z0: " << z0
                        << " psi_m: " << psi_m << " phi_e: " << phi_e
                        << " ux: " << ux << " uy: " << uy << " m: " << m
                        << " x: " << (problo[0] + (i + 0.5) * dx[0])
                        << " y: " << (problo[1] + (j + 0.5) * dx[1])
                        << " z: " << (problo[2] + (k + 0.5) * dx[2])
                        << std::endl;
                }
            });
    }

    // if (m_horizontal_sponge) {
    //     apply_horizontal_sponge(
    //         bx, prob_lo.data(), prob_hi.data(), dx.data(), dt,
    //         m_wind_heights_d, m_tke_values_d, tke_arr, src_term);
    // }
}

} // namespace amr_wind::pde::tke
