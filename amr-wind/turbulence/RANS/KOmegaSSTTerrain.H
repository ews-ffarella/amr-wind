#ifndef KOMEGASSTTERRAIN_H
#define KOMEGASSTTERRAIN_H

#include <string>
#include "amr-wind/turbulence/TurbModelBase.H"

namespace amr_wind::turbulence {

/** K-Omega-SST RANS turbulence model
 *
 * This also serves as the base class for all k-omega type RANS models
 *
 * Turbulence model described in
 *
 * Menter, F., Esch, T.,
 * "Elements of Industrial Heat Transfer Prediction",
 * 16th Brazilian Congress of Mechanical Engineering (COBEM),
 * Nov. 2001.
 *
 * with updated coefficients from
 *
 * Menter, F. R., Kuntz, M., and Langtry, R.,
 * "Ten Years of Industrial Experience with the SST Turbulence Model",
 * Turbulence, Heat and Mass Transfer 4, 2003,
 * pp. 625 - 632.
 *
 * Also see
 *
 * http://turbmodels.larc.nasa.gov/sst.html
 *
 */
template <typename Transport>
class KOmegaSSTTerrain : public TurbModelBase<Transport>
{
public:
    static std::string identifier()
    {
        return "KOmegaSSTTerrain-" + Transport::identifier();
    }

    explicit KOmegaSSTTerrain(CFDSim& sim);

    std::string model_name() const override { return "KOmegaSSTTerrain"; }

    //! Update the turbulent viscosity field
    void update_turbulent_viscosity(
        const FieldState fstate, const DiffusionType diff_type) override;

    //! Do any post advance work
    void post_advance_work() override;

    // TODO: Add support for terrain
    //! Update the effective thermal diffusivity field
    // void update_alphaeff(Field& alphaeff) override;

    //! Update the effective scalar diffusivity field
    void update_scalar_diff(Field& deff, const std::string& name) override;

    //! Parse turbulence model coefficients
    void parse_model_coeffs() override;

    //! Return turbulence model coefficients
    TurbulenceModel::CoeffsDictType model_coeffs() const override;

private:
    Field& m_vel;

    Field& m_f1;
    Field& m_shear_prod;
    Field& m_buoy_term;
    Field& m_diss;
    Field& m_sdr_src;
    Field& m_sdr_diss;
    Field& m_rho;
    // Field& m_walldist // Replaced by terrain_height for now

    Field* m_tke{nullptr};
    Field* m_sdr{nullptr};

    //! Turbulence constants
    amrex::Real m_beta_star{0.09};
    amrex::Real m_alpha1{0.5555555555555556};
    amrex::Real m_alpha2{0.44};
    amrex::Real m_beta1{0.075};
    amrex::Real m_beta2{0.0828};
    amrex::Real m_sigma_k1{0.85};
    amrex::Real m_sigma_k2{1.0};
    amrex::Real m_sigma_omega1{0.5};
    amrex::Real m_sigma_omega2{0.856};
    amrex::Real m_a1{0.31};
    amrex::Real m_tke_amb{0.0};
    amrex::Real m_sdr_amb{0.0};

    //! Buoyancy modified constants
    bool m_include_buoyancy{false};
    Field& m_temperature;
    amrex::Real m_buoyancy_factor = 0.0;
    amrex::Real m_sigma_t{0.85};

    //! Gravity vector (m/s^2)
    amrex::Vector<amrex::Real> m_gravity{0.0, 0.0, -9.81};

    // TODO: Add support for terrain
    amrex::Real m_surf_flux{0};
};

} // namespace amr_wind::turbulence

#endif /* KOMEGASSTTERRAIN_H */
