#ifndef KOmegaSSTTerrainSDRSRC_H
#define KOmegaSSTTerrainSDRSRC_H

#include "amr-wind/transport_models/TransportModel.H"
#include "amr-wind/equation_systems/sdr/SDRSource.H"
#include "amr-wind/turbulence/RANS/KransTerrainBase.H"
#include "amr-wind/utilities/constants.H"
#include "amr-wind/wind_energy/MOData.H"

namespace amr_wind::pde::tke {

class KOmegaSSTTerrainSDRSrc
    : public SDRSource::Register<KOmegaSSTTerrainSDRSrc>
    , public pde::KransTerrainBase
{
public:
    static std::string identifier() { return "KOmegaSSTTerrainSDRSrc"; }

    explicit KOmegaSSTTerrainSDRSrc(const CFDSim& /*sim*/);

    ~KOmegaSSTTerrainSDRSrc() override;

    void parse_coeffs();

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
    amrex::Real m_Cmu{1.0 / 5.48 / 5.48}; // 0.033 (atmospheric)
};

} // namespace amr_wind::pde::tke

#endif /* KOmegaSSTTerrainSDRSRC_H */
