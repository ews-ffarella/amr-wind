#ifndef KWSSTTERRAINSRC_H
#define KWSSTTERRAINSRC_H

#include "amr-wind/transport_models/TransportModel.H"
#include "amr-wind/equation_systems/tke/TKESource.H"
#include "amr-wind/turbulence/RANS/KransTerrainBase.H"
#include "amr-wind/utilities/constants.H"
#include "amr-wind/wind_energy/MOData.H"

namespace amr_wind::pde::tke {

class KOmegaSSTTerrainTKESrc
    : public TKESource::Register<KOmegaSSTTerrainTKESrc>
    , public pde::KransTerrainBase
{
public:
    static std::string identifier() { return "KOmegaSSTTerrainTKESrc"; }

    explicit KOmegaSSTTerrainTKESrc(const CFDSim& /*sim*/);

    ~KOmegaSSTTerrainTKESrc() override;

    void parse_coeffs();

    void operator()(
        const int lev,
        const amrex::MFIter& mfi,
        const amrex::Box& bx,
        const FieldState fstate,
        const amrex::Array4<amrex::Real>& src_term) const override;

private:
    Field& m_shear_prod;
    Field& m_diss;
    Field& m_buoy_term;
    Field& m_tke;
    amrex::Real m_Cmu{1.0 / 5.48 / 5.48}; // 0.033 (atmospheric)
};

} // namespace amr_wind::pde::tke

#endif /* KWSSTTERRAINSRC_H */
