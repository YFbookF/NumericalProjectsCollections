#pragma once

inline
void fresnelToCoeffs(const mitsuba::Spectrum& eta, const mitsuba::Spectrum& kappa,
                     mitsuba::Spectrum* coeffs)
{
    for(int i=0; i<3; ++i) {
        float t_coeffs[4];
        fresnelToCoeffs(eta[i], kappa[i], t_coeffs);
        coeffs[0][i] = t_coeffs[0];
        coeffs[1][i] = t_coeffs[1];
        coeffs[2][i] = t_coeffs[2];
        coeffs[3][i] = t_coeffs[3];
   }
}

inline
void GulbransenToIOR(const mitsuba::Spectrum& r, const mitsuba::Spectrum& g,
                     mitsuba::Spectrum& n, mitsuba::Spectrum& k)
{
    for(uint8_t s=0; s<3; ++s) {
        n[s] = g[s]*(1.0-r[s])/(1.0+r[s]) + (1.0-g[s])*(1.0+sqrt(r[s]))/(1.0-sqrt(r[s]));
        float v = (r[s]*(1.0+n[s])*(1.0+n[s]) - (n[s]-1.0)*(n[s]-1.0)) / (1.0-r[s]);
        k[s] = sqrt( std::max<float>(0.0f, v) );
    }
}


enum FresnelType {
    FRESNEL_SCHLICK,
    FRESNEL_OURS,
    FRESNEL_REFERENCE,
};

inline
mitsuba::Spectrum fresnelSchlick(mitsuba::Float cosT, mitsuba::Spectrum R0) {
    mitsuba::Float oCosT = 1.0-cosT;
    mitsuba::Float oCosT2 = oCosT*oCosT;
    mitsuba::Float oCosT5 = oCosT2*oCosT2*oCosT;
    mitsuba::Spectrum a = R0;
    mitsuba::Spectrum b = mitsuba::Spectrum(1.0)-R0;
    b.clampNegative();
    return a + b*oCosT5;
}
