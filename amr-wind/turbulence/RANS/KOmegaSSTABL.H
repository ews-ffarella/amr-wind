#ifndef KOMEGASSTABL_H
#define KOMEGASSTABL_H

#include <string>
#include "amr-wind/turbulence/RANS/KOmegaSST.H"

namespace amr_wind::turbulence {

/** K-Omega-SST RANS model for ABL
 *
 *
 *
 */
template <typename Transport>
class KOmegaSSTABL : public KOmegaSST<Transport>
{
public:
    static std::string identifier()
    {
        return "KOmegaSSTABL-" + Transport::identifier();
    }

    explicit KOmegaSSTABL(CFDSim& sim);

    ~KOmegaSSTABL() override;

    std::string model_name() const override { return "KOmegaSSTABL"; }

    //! Update the turbulent viscosity field
    void update_turbulent_viscosity(
        const FieldState fstate, const DiffusionType diff_type) override;

    //! No post advance work for this model
    void post_advance_work() override {}

    //! Update the effective scalar diffusivity field
    void update_scalar_diff(Field& deff, const std::string& name) override;

    //! Parse turbulence model coefficients
    void parse_model_coeffs() override;

    //! Return turbulence model coefficients
    TurbulenceModel::CoeffsDictType model_coeffs() const override;

protected:
    //! Turbulence constants
    //  Add extra vars in here
};

} // namespace amr_wind::turbulence

#endif /* KOMEGASSTABL_H */