#ifndef KOMEGASTTERRAINSDRSRC_H
#define KOMEGASTTERRAINSDRSRC_H

#include "amr-wind/transport_models/TransportModel.H"
#include "amr-wind/equation_systems/tke/TKESource.H"
#include "amr-wind/turbulence/RANS/KOmegaSSTTerrainBase.H"
#include "amr-wind/utilities/constants.H"
#include "amr-wind/wind_energy/MOData.H"

namespace amr_wind::pde::tke {

class KOmegaSTTerrainSDRSrc
    : public TKESource::Register<KOmegaSTTerrainSDRSrc>
    , public pde::KwSSTTerrainBase
{
public:
    static std::string identifier() { return "KOmegaSTTerrainSDRSrc"; }

    explicit KOmegaSTTerrainSDRSrc(const CFDSim& /*sim*/);

    ~KOmegaSTTerrainSDRSrc() override;

    void operator()(
        const int lev,
        const amrex::MFIter& mfi,
        const amrex::Box& bx,
        const FieldState fstate,
        const amrex::Array4<amrex::Real>& src_term) const override;

private:
    Field& m_sdr_src;
    Field& m_sdr_diss;
    Field& m_sdr;
};

} // namespace amr_wind::pde::tke

#endif /* KOMEGASTTERRAINSDRSRC_H */
