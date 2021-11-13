
//////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the Henyey-Greenstein phase function.
/// 主要用来算BSSRDF
Spectrum hg(Float cosTheta, const Spectrum &g) {
    Spectrum temp = Spectrum(1) + g * g + 2 * g * cosTheta;
    return INV_FOURPI * (Spectrum(1) - g * g) / (temp * temp.sqrt());
}
