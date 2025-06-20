#include "amr-wind/turbulence/RANS/KOmegaSSTTerrain.H"
#include "amr-wind/equation_systems/PDEBase.H"
#include "amr-wind/turbulence/TurbModelDefs.H"
#include "amr-wind/fvm/gradient.H"
#include "amr-wind/fvm/strainrate.H"
#include "amr-wind/turbulence/turb_utils.H"
#include "amr-wind/equation_systems/tke/TKE.H"
#include "amr-wind/equation_systems/sdr/SDR.H"

#include "AMReX_ParmParse.H"

namespace amr_wind {
namespace turbulence {

template <typename Transport>
KOmegaSSTTerrain<Transport>::KOmegaSSTTerrain(CFDSim& sim)
    : TurbModelBase<Transport>(sim)
    , m_vel(sim.repo().get_field("velocity"))
    , m_f1(sim.repo().declare_field("f1", 1))
    , m_shear_prod(sim.repo().declare_field("shear_prod", 1))
    , m_buoy_term(sim.repo().declare_field("buoyancy_term", 1))
    , m_diss(sim.repo().declare_field("dissipation", 1))
    , m_sdr_src(sim.repo().declare_field("omega_src", 1))
    , m_sdr_diss(sim.repo().declare_field("sdr_dissipation", 1))
    , m_rho(sim.repo().get_field("density"))
    , m_walldist(sim.repo().declare_field("wall_dist", 1, 1, 1))
    , m_temperature(sim.repo().get_field("temperature"))
{

    m_walldist.set_default_fillpatch_bc(sim.time());
    m_walldist.fillpatch_on_regrid() = true;

    auto& tke_eqn =
        sim.pde_manager().register_transport_pde(pde::TKE::pde_name());
    m_tke = &(tke_eqn.fields().field);

    auto& sdr_eqn =
        sim.pde_manager().register_transport_pde(pde::SDR::pde_name());
    m_sdr = &(sdr_eqn.fields().field);

    // TKE source term to be added to PDE
    turb_utils::inject_turbulence_src_terms(
        pde::TKE::pde_name(), {"KOmegaSSTTerrainTKESrc"});

    auto& phy_mgr = this->m_sim.physics_manager();
    if (!phy_mgr.contains("ABL")) {
        amrex::Abort("KOmegaSSTTerrain model only works with ABL physics");
    }
    {
        amrex::ParmParse pp("ABL");
        pp.get("surface_temp_flux", m_surf_flux);
    }
    {
        amrex::ParmParse pp("incflo");
        pp.queryarr("gravity", m_gravity);
    }
}

template <typename Transport>
void KOmegaSSTTerrain<Transport>::parse_model_coeffs()
{
    const std::string coeffs_dict = this->model_name() + "_coeffs";
    amrex::ParmParse pp(coeffs_dict);
    pp.query("beta_star", this->m_beta_star);
    pp.query("alpha1", this->m_alpha1);
    pp.query("alpha2", this->m_alpha2);
    pp.query("beta1", this->m_beta1);
    pp.query("beta2", this->m_beta2);
    pp.query("sigma_k1", this->m_sigma_k1);
    pp.query("sigma_k2", this->m_sigma_k2);
    pp.query("sigma_omega1", this->m_sigma_omega1);
    pp.query("sigma_omega2", this->m_sigma_omega2);
    pp.query("tke_amb", this->m_tke_amb);
    pp.query("sdr_amb", this->m_sdr_amb);
    pp.query("sigma_t", this->m_sigma_t);

    amrex::Print() << "KOmegaSSTTerrain model coefficients:\n"
                   << "  beta_star: " << this->m_beta_star << "\n"
                   << "  alpha1: " << this->m_alpha1 << "\n"
                   << "  alpha2: " << this->m_alpha2 << "\n"
                   << "  beta1: " << this->m_beta1 << "\n"
                   << "  beta2: " << this->m_beta2 << "\n"
                   << "  sigma_k1: " << this->m_sigma_k1 << "\n"
                   << "  sigma_k2: " << this->m_sigma_k2 << "\n"
                   << "  sigma_omega1: " << this->m_sigma_omega1 << "\n"
                   << "  sigma_omega2: " << this->m_sigma_omega2 << "\n"
                   << "  tke_amb: " << this->m_tke_amb << "\n"
                   << "  sdr_amb: " << this->m_sdr_amb << "\n"
                   << "  sigma_t: " << this->m_sigma_t << "\n";
    ;
}

template <typename Transport>
TurbulenceModel::CoeffsDictType
KOmegaSSTTerrain<Transport>::model_coeffs() const
{
    return TurbulenceModel::CoeffsDictType{
        {"beta_star", this->m_beta_star},
        {"alpha1", this->m_alpha1},
        {"alpha2", this->m_alpha2},
        {"beta1", this->m_beta1},
        {"beta2", this->m_beta2},
        {"sigma_k1", this->m_sigma_k1},
        {"sigma_k2", this->m_sigma_k2},
        {"sigma_omega1", this->m_sigma_omega1},
        {"sigma_omega2", this->m_sigma_omega2},
        {"tke_amb", this->m_tke_amb},
        {"sdr_amb", this->m_sdr_amb},
        {"a1", this->m_a1}};
}

template <typename Transport>
void KOmegaSSTTerrain<Transport>::update_turbulent_viscosity(
    const FieldState fstate, const DiffusionType diff_type)
{
    BL_PROFILE(
        "amr-wind::" + this->identifier() + "::update_turbulent_viscosity");

    const bool has_terrain =
        this->m_sim.repo().int_field_exists("terrain_blank");

    auto* const m_terrain_height =
        (has_terrain) ? &this->m_sim.repo().get_field("terrain_height")
                      : nullptr;

    auto* const m_terrain_blank =
        (has_terrain) ? &this->m_sim.repo().get_int_field("terrain_blank")
                      : nullptr;

    // auto gradT = (this->m_sim.repo()).create_scratch_field(3, 0);
    // fvm::gradient(*gradT, m_temperature.state(fstate));

    // Model constants
    const amrex::Real beta_star = this->m_beta_star;
    const amrex::Real alpha1 = this->m_alpha1;
    const amrex::Real alpha2 = this->m_alpha2;
    const amrex::Real beta1 = this->m_beta1;
    const amrex::Real beta2 = this->m_beta2;
    const amrex::Real sigma_omega2 = this->m_sigma_omega2;
    const amrex::Real a1 = this->m_a1;
    const amrex::Real tke_amb = this->m_tke_amb;
    const amrex::Real sdr_amb = this->m_sdr_amb;

    auto& mu_turb = this->mu_turb();
    auto lam_mu = (this->m_transport).mu();
    const auto& den = this->m_rho.state(fstate);
    const auto& tke = (*this->m_tke).state(fstate);
    const auto& sdr = (*this->m_sdr).state(fstate);
    const auto& repo = mu_turb.repo();
    auto& tke_lhs = (this->m_sim).repo().get_field("tke_lhs_src_term");
    tke_lhs.setVal(0.0);
    auto& sdr_lhs = (this->m_sim).repo().get_field("sdr_lhs_src_term");

    auto gradK = (this->m_sim.repo()).create_scratch_field(3, 0);
    fvm::gradient(*gradK, tke);

    auto gradOmega = (this->m_sim.repo()).create_scratch_field(3, 0);
    fvm::gradient(*gradOmega, sdr);

    // This is used for the buoyancy-modified version of the model
    auto gradden = (this->m_sim.repo()).create_scratch_field(3, 0);
    fvm::gradient(*gradden, den);

    const auto& vel = this->m_vel.state(fstate);
    // Compute strain rate into shear production term
    fvm::strainrate(this->m_shear_prod, vel);

    const amrex::Real delta_t = (this->m_sim).time().delta_t();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> gravity{
        m_gravity[0], m_gravity[1], m_gravity[2]};
    const amrex::Real Bfac = this->m_buoyancy_factor;
    const amrex::Real sigmat = this->m_sigma_t;

    const int nlevels = repo.num_active_levels();
    for (int lev = 0; lev < nlevels; ++lev) {

        const auto& lam_mu_arrs = (*lam_mu)(lev).const_arrays();
        const auto& mu_arrs = mu_turb(lev).arrays();
        const auto& gradrho_arrs = (*gradden)(lev).const_arrays();
        const auto& rho_arrs = den(lev).const_arrays();
        const auto& gradK_arrs = (*gradK)(lev).const_arrays();
        const auto& gradOmega_arrs = (*gradOmega)(lev).const_arrays();
        const auto& tke_arrs = tke(lev).const_arrays();
        const auto& sdr_arrs = sdr(lev).const_arrays();
        const auto& wd_arrs = (has_terrain)
                                  ? (*m_terrain_height)(lev).const_arrays()
                                  : (this->m_walldist)(lev).const_arrays();
        const auto& shear_prod_arrs = (this->m_shear_prod)(lev).arrays();
        const auto& diss_arrs = (this->m_diss)(lev).arrays();
        const auto& sdr_src_arrs = (this->m_sdr_src)(lev).arrays();
        const auto& sdr_diss_arrs = (this->m_sdr_diss)(lev).arrays();
        const auto& f1_arrs = (this->m_f1)(lev).arrays();
        const auto& tke_lhs_arrs = tke_lhs(lev).arrays();
        const auto& sdr_lhs_arrs = sdr_lhs(lev).arrays();
        const auto& buoy_arrs = (this->m_buoy_term(lev)).arrays();

        const auto& blank_arrs = has_terrain
                                     ? (*m_terrain_blank)(lev).const_arrays()
                                     : amrex::MultiArray4<const int>();

        amrex::ParallelFor(
            mu_turb(lev),
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
                const amrex::Real blankTerrain =
                    (has_terrain) ? 1 - blank_arrs[nbx](i, j, k, 0) : 1.0;

                amrex::Real gko =
                    (gradK_arrs[nbx](i, j, k, 0) *
                         gradOmega_arrs[nbx](i, j, k, 0) +
                     gradK_arrs[nbx](i, j, k, 1) *
                         gradOmega_arrs[nbx](i, j, k, 1) +
                     gradK_arrs[nbx](i, j, k, 2) *
                         gradOmega_arrs[nbx](i, j, k, 2));

                amrex::Real cdkomega = amrex::max<amrex::Real>(
                    1e-10, 2.0 * rho_arrs[nbx](i, j, k) * sigma_omega2 * gko /
                               (sdr_arrs[nbx](i, j, k) + 1e-15));

                amrex::Real tmp1 =
                    4.0 * rho_arrs[nbx](i, j, k) * sigma_omega2 *
                    tke_arrs[nbx](i, j, k) /
                    (cdkomega * wd_arrs[nbx](i, j, k) * wd_arrs[nbx](i, j, k));
                amrex::Real tmp2 = std::sqrt(tke_arrs[nbx](i, j, k)) /
                                   (beta_star * sdr_arrs[nbx](i, j, k) *
                                        wd_arrs[nbx](i, j, k) +
                                    1e-15);
                amrex::Real tmp3 =
                    500.0 * lam_mu_arrs[nbx](i, j, k) /
                    (wd_arrs[nbx](i, j, k) * wd_arrs[nbx](i, j, k) *
                         sdr_arrs[nbx](i, j, k) * rho_arrs[nbx](i, j, k) +
                     1e-15);
                amrex::Real tmp4 = shear_prod_arrs[nbx](i, j, k);

                amrex::Real arg1 = amrex::min<amrex::Real>(
                    amrex::max<amrex::Real>(tmp2, tmp3), tmp1);
                amrex::Real tmp_f1 = std::tanh(arg1 * arg1 * arg1 * arg1);

                amrex::Real alpha = tmp_f1 * (alpha1 - alpha2) + alpha2;
                amrex::Real beta = tmp_f1 * (beta1 - beta2) + beta2;

                amrex::Real arg2 = amrex::max<amrex::Real>(2.0 * tmp2, tmp3);
                amrex::Real f2 = std::tanh(arg2 * arg2);

                mu_arrs[nbx](i, j, k) =
                    blankTerrain * rho_arrs[nbx](i, j, k) * a1 *
                    tke_arrs[nbx](i, j, k) /
                    amrex::max<amrex::Real>(
                        a1 * sdr_arrs[nbx](i, j, k), tmp4 * f2);

                // Buoyancy term
                amrex::Real tmpB =
                    -(gravity[0] * gradrho_arrs[nbx](i, j, k, 0) +
                      gravity[1] * gradrho_arrs[nbx](i, j, k, 1) +
                      gravity[2] * gradrho_arrs[nbx](i, j, k, 2));

                buoy_arrs[nbx](i, j, k) =
                    blankTerrain * Bfac * tmpB *
                    (mu_arrs[nbx](i, j, k) / rho_arrs[nbx](i, j, k)) / sigmat;

                f1_arrs[nbx](i, j, k) = tmp_f1;

                // For TKE equation:
                shear_prod_arrs[nbx](i, j, k) =
                    blankTerrain *
                    amrex::min<amrex::Real>(
                        amrex::max<amrex::Real>(
                            mu_arrs[nbx](i, j, k) * tmp4 * tmp4, 0.0),
                        10.0 * beta_star * rho_arrs[nbx](i, j, k) *
                            tke_arrs[nbx](i, j, k) * sdr_arrs[nbx](i, j, k));

                const amrex::Real diss_amb = blankTerrain * beta_star *
                                             rho_arrs[nbx](i, j, k) * sdr_amb *
                                             tke_amb;

                diss_arrs[nbx](i, j, k) = -beta_star * rho_arrs[nbx](i, j, k) *
                                              tke_arrs[nbx](i, j, k) *
                                              sdr_arrs[nbx](i, j, k) +
                                          diss_amb;

                tke_lhs_arrs[nbx](i, j, k) = blankTerrain * beta_star *
                                             rho_arrs[nbx](i, j, k) *
                                             sdr_arrs[nbx](i, j, k) * delta_t;

                // For SDR equation:
                amrex::Real production_omega =
                    blankTerrain * rho_arrs[nbx](i, j, k) * alpha *
                    amrex::min<amrex::Real>(
                        tmp4 * tmp4, 10.0 * beta_star * sdr_arrs[nbx](i, j, k) *
                                         sdr_arrs[nbx](i, j, k));

                amrex::Real cross_diffusion = blankTerrain * (1.0 - tmp_f1) *
                                              2.0 * rho_arrs[nbx](i, j, k) *
                                              sigma_omega2 * gko /
                                              (sdr_arrs[nbx](i, j, k) + 1e-15);

                const amrex::Real sdr_diss_amb = blankTerrain * beta *
                                                 rho_arrs[nbx](i, j, k) *
                                                 sdr_amb * sdr_amb;

                if (diff_type == DiffusionType::Crank_Nicolson) {

                    tke_lhs_arrs[nbx](i, j, k) =
                        0.5 * tke_lhs_arrs[nbx](i, j, k);

                    sdr_src_arrs[nbx](i, j, k) = production_omega;

                    sdr_diss_arrs[nbx](i, j, k) = cross_diffusion;

                    sdr_lhs_arrs[nbx](i, j, k) =
                        blankTerrain *
                        (rho_arrs[nbx](i, j, k) * beta *
                             sdr_arrs[nbx](i, j, k) +
                         0.5 * std::abs(cross_diffusion) /
                             (sdr_arrs[nbx](i, j, k) + 1e-15)) *
                        delta_t;
                } else if (diff_type == DiffusionType::Implicit) {
                    /* Source term linearization is based on Florian
                       Menter's (1993) AIAA paper */
                    diss_arrs[nbx](i, j, k) = 0.0;

                    sdr_src_arrs[nbx](i, j, k) = production_omega;

                    sdr_diss_arrs[nbx](i, j, k) = 0.0;

                    sdr_lhs_arrs[nbx](i, j, k) =
                        blankTerrain *
                        (2.0 * rho_arrs[nbx](i, j, k) * beta *
                             sdr_arrs[nbx](i, j, k) +
                         std::abs(cross_diffusion) /
                             (sdr_arrs[nbx](i, j, k) + 1e-15)) *
                        delta_t;

                } else {
                    sdr_src_arrs[nbx](i, j, k) =
                        production_omega + cross_diffusion;

                    sdr_diss_arrs[nbx](i, j, k) =
                        -rho_arrs[nbx](i, j, k) * beta * blankTerrain *
                            sdr_arrs[nbx](i, j, k) * sdr_arrs[nbx](i, j, k) +
                        sdr_diss_amb;

                    sdr_lhs_arrs[nbx](i, j, k) =
                        0.5 * rho_arrs[nbx](i, j, k) * blankTerrain * beta *
                        sdr_arrs[nbx](i, j, k) * delta_t;
                }
            });
    }
    amrex::Gpu::streamSynchronize();

    mu_turb.fillpatch(this->m_sim.time().current_time());
}

template <typename Transport>
void KOmegaSSTTerrain<Transport>::update_scalar_diff(
    Field& deff, const std::string& name)
{
    BL_PROFILE("amr-wind::" + this->identifier() + "::update_scalar_diff");

    auto lam_mu = (this->m_transport).mu();
    const auto& mu_turb = this->mu_turb();

    const bool has_terrain =
        this->m_sim.repo().int_field_exists("terrain_blank");

    auto* const m_terrain_blank =
        (has_terrain) ? &this->m_sim.repo().get_int_field("terrain_blank")
                      : nullptr;

    if (name == pde::TKE::var_name()) {
        const amrex::Real sigma_k1 = this->m_sigma_k1;
        const amrex::Real sigma_k2 = this->m_sigma_k2;
        const auto& repo = deff.repo();
        const int nlevels = repo.num_active_levels();
        for (int lev = 0; lev < nlevels; ++lev) {
            const auto& blank_arrs =
                has_terrain ? (*m_terrain_blank)(lev).const_arrays()
                            : amrex::MultiArray4<const int>();
            const auto& lam_mu_arrs = (*lam_mu)(lev).const_arrays();
            const auto& mu_arrs = mu_turb(lev).const_arrays();
            const auto& f1_arrs = (this->m_f1)(lev).const_arrays();
            const auto& deff_arrs = deff(lev).arrays();
            amrex::ParallelFor(
                deff(lev),
                [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
                    const amrex::Real blankTerrain =
                        (has_terrain) ? 1 - blank_arrs[nbx](i, j, k, 0) : 1.0;
                    deff_arrs[nbx](i, j, k) =
                        lam_mu_arrs[nbx](i, j, k) +
                        (f1_arrs[nbx](i, j, k) * (sigma_k1 - sigma_k2) +
                         sigma_k2) *
                            mu_arrs[nbx](i, j, k);
                    deff_arrs[nbx](i, j, k) *= blankTerrain;
                });
        }
        amrex::Gpu::streamSynchronize();
    } else if (name == pde::SDR::var_name()) {
        const amrex::Real sigma_omega1 = this->m_sigma_omega1;
        const amrex::Real sigma_omega2 = this->m_sigma_omega2;
        const auto& repo = deff.repo();
        const int nlevels = repo.num_active_levels();
        for (int lev = 0; lev < nlevels; ++lev) {
            const auto& blank_arrs =
                has_terrain ? (*m_terrain_blank)(lev).const_arrays()
                            : amrex::MultiArray4<const int>();
            const auto& lam_mu_arrs = (*lam_mu)(lev).const_arrays();
            const auto& mu_arrs = mu_turb(lev).const_arrays();
            const auto& f1_arrs = (this->m_f1)(lev).const_arrays();
            const auto& deff_arrs = deff(lev).arrays();
            amrex::ParallelFor(
                deff(lev),
                [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
                    const amrex::Real blankTerrain =
                        (has_terrain) ? 1 - blank_arrs[nbx](i, j, k, 0) : 1.0;
                    deff_arrs[nbx](i, j, k) =
                        lam_mu_arrs[nbx](i, j, k) +
                        (f1_arrs[nbx](i, j, k) * (sigma_omega1 - sigma_omega2) +
                         sigma_omega2) *
                            mu_arrs[nbx](i, j, k);
                    deff_arrs[nbx](i, j, k) *= blankTerrain;
                });
        }
        amrex::Gpu::streamSynchronize();
    } else {
        amrex::Abort(
            "KOmegaSSTTerrain:update_scalar_diff not implemented for field " +
            name);
    }
}

template <typename Transport>
void KOmegaSSTTerrain<Transport>::post_advance_work()
{

    BL_PROFILE("amr-wind::" + this->identifier() + "::post_advance_work");
}

} // namespace turbulence

INSTANTIATE_TURBULENCE_MODEL(KOmegaSSTTerrain);

} // namespace amr_wind
