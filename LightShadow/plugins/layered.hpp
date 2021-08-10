// Mitsuba includes
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/render/bsdf.h>

// STL includes
#include <fstream>
#include <sstream>
#include <vector>

// LookupIOR
#include "ior.h"
#include <math.h>

/* Helping functions
 */
template <typename T>
inline T _clamp(T x, T m, T M) { return std::min<T>(std::max<T>(x, m), M); }
template <typename T>
inline T sqr(T x) { return x * x; }

/* On OSX, I cannot compile Mitsuba with full c++11. Some features are thus
 * missing. Such as `to_string` ...
 */
#if defined(__APPLE__)
namespace std
{
template <typename T>
std::string to_string(const T &value)
{
    std::stringstream ss;
    ss << value;
    return ss.str();
}
} // namespace std
#endif

/* Roughness to linear space conversions
 */

#define CONST_A 1.28809776
#define CONST_B 1.31699416

inline float roughnessToVariance(float a)
{
    a = _clamp<float>(a, 0.0, 0.9999);
    float aPow = powf(a, CONST_A);
    return logf(1 + (CONST_B * aPow) / (1 - aPow));
}

inline float varianceToRoughness(float v)
{
    float c = expf(v) - 1.0f;
    if (isinf(c)) return 1.0;
    return powf(c / (c + CONST_B), 1.0f / CONST_A);
}

MTS_NAMESPACE_BEGIN

void parseLayers(const Properties &props,
                 int &nb_layers,
                 std::vector<Spectrum> &m_etas,
                 std::vector<Spectrum> &m_kappas,
                 std::vector<Float> &m_alphaUs,
                 std::vector<Float> &m_alphaVs)
{

    // File Resolver for getting materials etas an kappas
    ref<FileResolver> fResolver = Thread::getThread()->getFileResolver();

    /* Parse a layered structure */
    nb_layers = props.getInteger("nb_layers", 0);
    // AssertEx(nb_layers > 0, "layered must have at least one layer");

    // Add the external IOR
    Float extEta = lookupIOR(props, "extEta", "air");
    m_etas.push_back(Spectrum(extEta));
    m_kappas.push_back(Spectrum(0.0));

    // Add the layers IOR and interfaces
    for (int k = 0; k < nb_layers; ++k)
    {
        std::string index = std::to_string(k);
        std::string name;

        // Adding a rough surface parameters
        name = std::string("eta_") + index;
        Spectrum eta_k = props.getSpectrum(name, Spectrum(1.0f));

        name = std::string("kappa_") + index;
        Spectrum kappa_k = props.getSpectrum(name, Spectrum(0.0f));

        name = std::string("alphaU_") + index;
        Float alphaU_k = props.getFloat(name, 0.0f);

        name = std::string("alphaV_") + index;
        Float alphaV_k = props.getFloat(name, 0.0f);

        name = std::string("material_") + index;
        std::string material_k = props.getString(name, "None");

        // if a material is specified override eta and kappa
        if (material_k.compare("None") != 0)
        {
            /* IOR (real part) (default: Cu, copper) */
            eta_k.fromContinuousSpectrum(InterpolatedSpectrum(fResolver->resolve("data/ior/" + material_k + ".eta.spd")));

            /* IOR (complex part) (default: Cu, copper) */
            kappa_k.fromContinuousSpectrum(InterpolatedSpectrum(fResolver->resolve("data/ior/" + material_k + ".k.spd")));
        }

        // Update roughness
        m_etas.push_back(eta_k);
        m_kappas.push_back(kappa_k);
        m_alphaUs.push_back(alphaU_k);
        m_alphaVs.push_back(alphaV_k);
    }

    // Print the resulting microstructure
    SLog(EInfo, "");
    SLog(EInfo, "Adding Exterior IOR");
    SLog(EInfo, " + n = %s", m_etas[0].toString().c_str());
    SLog(EInfo, " + k = %s", m_kappas[0].toString().c_str());
    SLog(EInfo, "");

    for (int k = 0; k < nb_layers; ++k)
    {
        SLog(EInfo, "Adding layer %d", k);
        SLog(EInfo, " + n = %s", m_etas[k + 1].toString().c_str());
        SLog(EInfo, " + k = %s", m_kappas[k + 1].toString().c_str());
        SLog(EInfo, " + alphaU = %f", m_alphaUs[k]);
        SLog(EInfo, " + alphaV = %f", m_alphaVs[k]);
        SLog(EInfo, "");
    }
}

void parseLayers(const Properties &props,
                 int &nb_layers,
                 std::vector<ref<const Texture>> &m_tex_etas,
                 std::vector<ref<const Texture>> &m_tex_kappas,
                 std::vector<ref<const Texture>> &m_tex_alphaUs,
                 std::vector<ref<const Texture>> &m_tex_alphaVs)
{

    // File Resolver for getting materials etas an kappas
    ref<FileResolver> fResolver = Thread::getThread()->getFileResolver();

    /* Parse a layered structure */
    nb_layers = props.getInteger("nb_layers", 0);
    // AssertEx(nb_layers > 0, "layered must have at least one layer");

    // Add the external IOR
    Float extEta = lookupIOR(props, "extEta", "air");
    m_tex_etas.push_back(new ConstantSpectrumTexture(Spectrum(extEta)));
    m_tex_kappas.push_back(new ConstantSpectrumTexture(Spectrum(0.0)));

    // Add the layers IOR and interfaces
    for (int k = 0; k < nb_layers; ++k)
    {
        std::string index = std::to_string(k);
        std::string name;

        // Adding a rough surface parameters
        name = std::string("eta_") + index;
        Spectrum eta_k = props.getSpectrum(name, Spectrum(1.0f));

        name = std::string("kappa_") + index;
        Spectrum kappa_k = props.getSpectrum(name, Spectrum(0.0f));

        name = std::string("alphaU_") + index;
        Float alphaU_k = props.getFloat(name, 0.0f);

        name = std::string("alphaV_") + index;
        Float alphaV_k = props.getFloat(name, 0.0f);

        name = std::string("material_") + index;
        std::string material_k = props.getString(name, "None");

        // if a material is specified override eta and kappa
        if (material_k.compare("None") != 0)
        {
            /* IOR (real part) (default: Cu, copper) */
            eta_k.fromContinuousSpectrum(InterpolatedSpectrum(fResolver->resolve("data/ior/" + material_k + ".eta.spd")));

            /* IOR (complex part) (default: Cu, copper) */
            kappa_k.fromContinuousSpectrum(InterpolatedSpectrum(fResolver->resolve("data/ior/" + material_k + ".k.spd")));
        }

        // Update roughness
        m_tex_etas.push_back(new ConstantSpectrumTexture(eta_k));
        m_tex_kappas.push_back(new ConstantSpectrumTexture(kappa_k));
        m_tex_alphaUs.push_back(new ConstantFloatTexture(alphaU_k));
        m_tex_alphaVs.push_back(new ConstantFloatTexture(alphaV_k));
    }

    // Print the resulting microstructure
    SLog(EInfo, "");
    SLog(EInfo, "Adding Exterior IOR");
    SLog(EInfo, " + n = %s", m_tex_etas[0]->getAverage().toString().c_str());
    SLog(EInfo, " + k = %s", m_tex_kappas[0]->getAverage().toString().c_str());
    SLog(EInfo, "");

    for (int k = 0; k < nb_layers; ++k)
    {
        SLog(EInfo, "Adding layer %d", k);
        SLog(EInfo, " + n = %s", m_tex_etas[k + 1]->getAverage().toString().c_str());
        SLog(EInfo, " + k = %s", m_tex_kappas[k + 1]->getAverage().toString().c_str());
        SLog(EInfo, " + alphaU = %f", m_tex_alphaUs[k]->getAverage());
        SLog(EInfo, " + alphaV = %f", m_tex_alphaVs[k]->getAverage());
        SLog(EInfo, "");
    }
}

MTS_NAMESPACE_END
