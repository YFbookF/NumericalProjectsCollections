#include <mitsuba/core/fresolver.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/hw/basicshader.h>
#include "microfacet.h"
#include "ior.h"

#include "fresnel_curves.hpp"
#include "fresnel_mapping.hpp"
#include "fresnel_helpers.hpp"

MTS_NAMESPACE_BEGIN

class RoughConductor : public BSDF {
public:

    RoughConductor(const Properties &props) : BSDF(props) {
        m_specularReflectance = new ConstantSpectrumTexture(
            props.getSpectrum("specularReflectance", Spectrum(1.0f)));

        updateCoeffs(props);
        std::cout << "<<<<INFO>>>> Fitted coeffs:"
                  << "<<<<INFO>>>> c_0 = [" << m_coeffs[0][0] << ", " << m_coeffs[0][1] << ", " << m_coeffs[0][2] << "], "
                  << std::endl
                  << "<<<<INFO>>>> c_1 = [" << m_coeffs[1][0] << ", " << m_coeffs[1][1] << ", " << m_coeffs[1][2] << "], "
                  << std::endl
                  << "<<<<INFO>>>> c_2 = [" << m_coeffs[2][0] << ", " << m_coeffs[2][1] << ", " << m_coeffs[2][2] << "], "
                  << std::endl
                  << "<<<<INFO>>>> c_3 = [" << m_coeffs[3][0] << ", " << m_coeffs[3][1] << ", " << m_coeffs[3][2] << "], "
                  << std::endl;

        MicrofacetDistribution distr(props);
        m_type = distr.getType();
        m_sampleVisible = distr.getSampleVisible();

        m_alphaU = new ConstantFloatTexture(distr.getAlphaU());
        if (distr.getAlphaU() == distr.getAlphaV())
            m_alphaV = m_alphaU;
        else
            m_alphaV = new ConstantFloatTexture(distr.getAlphaV());
    }

    void updateCoeffs(const Properties &props) {
        ref<FileResolver> fResolver = Thread::getThread()->getFileResolver();

       // Only parsing a material that defines precomputed tables of IOR
       if( props.hasProperty("material")) {
          std::string materialName = props.getString("material", "Cu");
          Spectrum intEta, intK;
          intEta.fromContinuousSpectrum(InterpolatedSpectrum(
                   fResolver->resolve("data/ior/" + materialName + ".eta.spd")));
          intK.fromContinuousSpectrum(InterpolatedSpectrum(
                   fResolver->resolve("data/ior/" + materialName + ".k.spd")));
          fresnelToCoeffs(intEta, intK, m_coeffs);
          return;
        }

       // Look for a set of (r,b) parameters
       if( props.hasProperty("r") && props.hasProperty("b") ) {
          Spectrum r = props.getSpectrum("r", Spectrum(0.5));
          Spectrum b = props.getSpectrum("b", Spectrum(1.0));
          std::cout << "<<<<INFO>>>> Using (r,b) = " << r[0] << ", " << b[0] << std::endl;
          for(int i=0; i<3; ++i) {
             float coeffs[4];
             evaluateMapping(r[i], b[i], coeffs);
             m_coeffs[0][i] = coeffs[0];
             m_coeffs[1][i] = coeffs[1];
             m_coeffs[2][i] = coeffs[2];
             m_coeffs[3][i] = coeffs[3];
          }
          return;
       }

       // Look for a set of (r,g) parameters
       if( props.hasProperty("r") && props.hasProperty("g") ) {
          Spectrum r = props.getSpectrum("r", Spectrum(0.5));
          Spectrum g = props.getSpectrum("g", Spectrum(1.0));
          std::cout << "<<<<INFO>>>> Using (r,g) = " << r[0] << ", " << g[0] << std::endl;
          Spectrum eta, kappa;
          GulbransenToIOR(r, g, eta, kappa);
          std::cout << "<<<<INFO>>>>       (n,k) = " << eta[0] << ", " << kappa[0] << std::endl;
          fresnelToCoeffs(eta, kappa, m_coeffs);
          return;
       }

       // Look for a set of (eta,kappa) parameters
       if( props.hasProperty("eta") && props.hasProperty("kappa") ) {
          Spectrum eta   = props.getSpectrum("eta", Spectrum(1.5));
          Spectrum kappa = props.getSpectrum("kappa", Spectrum(0.0));
          fresnelToCoeffs(eta, kappa, m_coeffs);
          return;
       }
    }

    RoughConductor(Stream *stream, InstanceManager *manager)
     : BSDF(stream, manager) {
        m_type = (MicrofacetDistribution::EType) stream->readUInt();
        m_sampleVisible = stream->readBool();
        m_alphaU = static_cast<Texture *>(manager->getInstance(stream));
        m_alphaV = static_cast<Texture *>(manager->getInstance(stream));
        m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
        m_coeffs[0] = Spectrum(stream);
        m_coeffs[1] = Spectrum(stream);
        m_coeffs[2] = Spectrum(stream);
        m_coeffs[3] = Spectrum(stream);

        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);

        stream->writeUInt((uint32_t) m_type);
        stream->writeBool(m_sampleVisible);
        manager->serialize(stream, m_alphaU.get());
        manager->serialize(stream, m_alphaV.get());
        manager->serialize(stream, m_specularReflectance.get());
        m_coeffs[0].serialize(stream);
        m_coeffs[1].serialize(stream);
        m_coeffs[2].serialize(stream);
        m_coeffs[3].serialize(stream);
    }

    void configure() {
        unsigned int extraFlags = 0;
        if (m_alphaU != m_alphaV)
            extraFlags |= EAnisotropic;

        if (!m_alphaU->isConstant() || !m_alphaV->isConstant() ||
            !m_specularReflectance->isConstant())
            extraFlags |= ESpatiallyVarying;

        m_components.clear();
        m_components.push_back(EGlossyReflection | EFrontSide | extraFlags);

        /* Verify the input parameters and fix them if necessary */
        m_specularReflectance = ensureEnergyConservation(
            m_specularReflectance, "specularReflectance", 1.0f);

        m_usesRayDifferentials =
            m_alphaU->usesRayDifferentials() ||
            m_alphaV->usesRayDifferentials() ||
            m_specularReflectance->usesRayDifferentials();

        BSDF::configure();
    }

    inline Spectrum evalFresnel(Float cosT) const
    {
       Spectrum F;
       Float v[4];
       evaluateFresnelCurves(cosT, v);

       F = m_coeffs[0]*v[0] + m_coeffs[1]*v[1] + m_coeffs[2]*v[2] + m_coeffs[3]*v[3];
       F.clampNegative();
       return F;
    }

    /// Helper function: reflect \c wi with respect to a given surface normal
    inline Vector reflect(const Vector &wi, const Normal &m) const {
        return 2 * dot(wi, m) * Vector(m) - wi;
    }

    Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        /* Stop if this component was not requested */
        if (measure != ESolidAngle ||
            Frame::cosTheta(bRec.wi) <= 0 ||
            Frame::cosTheta(bRec.wo) <= 0 ||
            ((bRec.component != -1 && bRec.component != 0) ||
            !(bRec.typeMask & EGlossyReflection)))
            return Spectrum(0.0f);

        /* Calculate the reflection half-vector */
        Vector H = normalize(bRec.wo+bRec.wi);

        /* Construct the microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(
            m_type,
            m_alphaU->eval(bRec.its).average(),
            m_alphaV->eval(bRec.its).average(),
            m_sampleVisible
        );

        /* Evaluate the microfacet normal distribution */
        const Float D = distr.eval(H);
        if (D == 0)
            return Spectrum(0.0f);

        /* Fresnel factor */
        const Spectrum F = evalFresnel(dot(bRec.wi, H)) *
            m_specularReflectance->eval(bRec.its);

        /* Smith's shadow-masking function */
        const Float G = distr.G(bRec.wi, bRec.wo, H);

        /* Calculate the total amount of reflection */
        Float model = D * G / (4.0f * Frame::cosTheta(bRec.wi));

        return F * model;
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        if (measure != ESolidAngle ||
            Frame::cosTheta(bRec.wi) <= 0 ||
            Frame::cosTheta(bRec.wo) <= 0 ||
            ((bRec.component != -1 && bRec.component != 0) ||
            !(bRec.typeMask & EGlossyReflection)))
            return 0.0f;

        /* Calculate the reflection half-vector */
        Vector H = normalize(bRec.wo+bRec.wi);

        /* Construct the microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(
            m_type,
            m_alphaU->eval(bRec.its).average(),
            m_alphaV->eval(bRec.its).average(),
            m_sampleVisible
        );

        if (m_sampleVisible)
            return distr.eval(H) * distr.smithG1(bRec.wi, H)
                / (4.0f * Frame::cosTheta(bRec.wi));
        else
            return distr.pdf(bRec.wi, H) / (4 * absDot(bRec.wo, H));
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
        if (Frame::cosTheta(bRec.wi) < 0 ||
            ((bRec.component != -1 && bRec.component != 0) ||
            !(bRec.typeMask & EGlossyReflection)))
            return Spectrum(0.0f);

        /* Construct the microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(
            m_type,
            m_alphaU->eval(bRec.its).average(),
            m_alphaV->eval(bRec.its).average(),
            m_sampleVisible
        );

        /* Sample M, the microfacet normal */
        Float pdf;
        Normal m = distr.sample(bRec.wi, sample, pdf);

        if (pdf == 0)
            return Spectrum(0.0f);

        /* Perfect specular reflection based on the microfacet normal */
        bRec.wo = reflect(bRec.wi, m);
        bRec.eta = 1.0f;
        bRec.sampledComponent = 0;
        bRec.sampledType = EGlossyReflection;

        /* Side check */
        if (Frame::cosTheta(bRec.wo) <= 0)
            return Spectrum(0.0f);

        Spectrum F = evalFresnel(dot(bRec.wi, m)) * m_specularReflectance->eval(bRec.its);

        Float weight;
        if (m_sampleVisible) {
            weight = distr.smithG1(bRec.wo, m);
        } else {
            weight = distr.eval(m) * distr.G(bRec.wi, bRec.wo, m)
                * dot(bRec.wi, m) / (pdf * Frame::cosTheta(bRec.wi));
        }

        return F * weight;
    }

    Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &sample) const {
        if (Frame::cosTheta(bRec.wi) < 0 ||
            ((bRec.component != -1 && bRec.component != 0) ||
            !(bRec.typeMask & EGlossyReflection)))
            return Spectrum(0.0f);

        /* Construct the microfacet distribution matching the
           roughness values at the current surface position. */
        MicrofacetDistribution distr(
            m_type,
            m_alphaU->eval(bRec.its).average(),
            m_alphaV->eval(bRec.its).average(),
            m_sampleVisible
        );

        /* Sample M, the microfacet normal */
        Normal m = distr.sample(bRec.wi, sample, pdf);

        if (pdf == 0)
            return Spectrum(0.0f);

        /* Perfect specular reflection based on the microfacet normal */
        bRec.wo = reflect(bRec.wi, m);
        bRec.eta = 1.0f;
        bRec.sampledComponent = 0;
        bRec.sampledType = EGlossyReflection;

        /* Side check */
        if (Frame::cosTheta(bRec.wo) <= 0)
            return Spectrum(0.0f);

        Spectrum F = evalFresnel(dot(bRec.wi, m)) * m_specularReflectance->eval(bRec.its);

        Float weight;
        if (m_sampleVisible) {
            weight = distr.smithG1(bRec.wo, m);
        } else {
            weight = distr.eval(m) * distr.G(bRec.wi, bRec.wo, m)
                * dot(bRec.wi, m) / (pdf * Frame::cosTheta(bRec.wi));
        }

        /* Jacobian of the half-direction mapping */
        pdf /= 4.0f * dot(bRec.wo, m);

        return F * weight;
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
            if (name == "alpha")
                m_alphaU = m_alphaV = static_cast<Texture *>(child);
            else if (name == "alphaU")
                m_alphaU = static_cast<Texture *>(child);
            else if (name == "alphaV")
                m_alphaV = static_cast<Texture *>(child);
            else if (name == "specularReflectance")
                m_specularReflectance = static_cast<Texture *>(child);
            else
                BSDF::addChild(name, child);
        } else {
            BSDF::addChild(name, child);
        }
    }

    Float getRoughness(const Intersection &its, int component) const {
        return 0.5f * (m_alphaU->eval(its).average()
            + m_alphaV->eval(its).average());
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "RoughConductor[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  distribution = " << MicrofacetDistribution::distributionName(m_type) << "," << endl
            << "  sampleVisible = " << m_sampleVisible << "," << endl
            << "  alphaU = " << indent(m_alphaU->toString()) << "," << endl
            << "  alphaV = " << indent(m_alphaV->toString()) << "," << endl
            << "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
            << "]";
        return oss.str();
    }

    Shader *createShader(Renderer *renderer) const;

    MTS_DECLARE_CLASS()
private:
    MicrofacetDistribution::EType m_type;
    ref<Texture> m_specularReflectance;
    ref<Texture> m_alphaU, m_alphaV;
    bool m_sampleVisible;

    Spectrum m_coeffs[4];
};

/**
 * GLSL port of the rough conductor shader. This version is much more
 * approximate -- it only supports the Ashikhmin-Shirley distribution,
 * does everything in RGB, and it uses the Schlick approximation to the
 * Fresnel reflectance of conductors. When the roughness is lower than
 * \alpha < 0.2, the shader clamps it to 0.2 so that it will still perform
 * reasonably well in a VPL-based preview.
 */
class RoughConductorShader : public Shader {
public:
    RoughConductorShader(Renderer *renderer, const Texture *specularReflectance,
            const Texture *alphaU, const Texture *alphaV, const Spectrum &R0) : Shader(renderer, EBSDFShader),
            m_specularReflectance(specularReflectance), m_alphaU(alphaU), m_alphaV(alphaV) {
        m_specularReflectanceShader = renderer->registerShaderForResource(m_specularReflectance.get());
        m_alphaUShader = renderer->registerShaderForResource(m_alphaU.get());
        m_alphaVShader = renderer->registerShaderForResource(m_alphaV.get());

        /* Compute the reflectance at perpendicular incidence */
        m_R0 = R0;
    }

    bool isComplete() const {
        return m_specularReflectanceShader.get() != NULL &&
               m_alphaUShader.get() != NULL &&
               m_alphaVShader.get() != NULL;
    }

    void putDependencies(std::vector<Shader *> &deps) {
        deps.push_back(m_specularReflectanceShader.get());
        deps.push_back(m_alphaUShader.get());
        deps.push_back(m_alphaVShader.get());
    }

    void cleanup(Renderer *renderer) {
        renderer->unregisterShaderForResource(m_specularReflectance.get());
        renderer->unregisterShaderForResource(m_alphaU.get());
        renderer->unregisterShaderForResource(m_alphaV.get());
    }

    void resolve(const GPUProgram *program, const std::string &evalName, std::vector<int> &parameterIDs) const {
        parameterIDs.push_back(program->getParameterID(evalName + "_R0", false));
    }

    void bind(GPUProgram *program, const std::vector<int> &parameterIDs, int &textureUnitOffset) const {
        program->setParameter(parameterIDs[0], m_R0);
    }

    void generateCode(std::ostringstream &oss,
            const std::string &evalName,
            const std::vector<std::string> &depNames) const {
        oss << "uniform vec3 " << evalName << "_R0;" << endl
            << endl
            << "float " << evalName << "_D(vec3 m, float alphaU, float alphaV) {" << endl
            << "    float ct = cosTheta(m), ds = 1-ct*ct;" << endl
            << "    if (ds <= 0.0)" << endl
            << "        return 0.0f;" << endl
            << "    alphaU = 2 / (alphaU * alphaU) - 2;" << endl
            << "    alphaV = 2 / (alphaV * alphaV) - 2;" << endl
            << "    float exponent = (alphaU*m.x*m.x + alphaV*m.y*m.y)/ds;" << endl
            << "    return sqrt((alphaU+2) * (alphaV+2)) * 0.15915 * pow(ct, exponent);" << endl
            << "}" << endl
            << endl
            << "float " << evalName << "_G(vec3 m, vec3 wi, vec3 wo) {" << endl
            << "    if ((dot(wi, m) * cosTheta(wi)) <= 0 || " << endl
            << "        (dot(wo, m) * cosTheta(wo)) <= 0)" << endl
            << "        return 0.0;" << endl
            << "    float nDotM = cosTheta(m);" << endl
            << "    return min(1.0, min(" << endl
            << "        abs(2 * nDotM * cosTheta(wo) / dot(wo, m))," << endl
            << "        abs(2 * nDotM * cosTheta(wi) / dot(wi, m))));" << endl
            << "}" << endl
            << endl
            << "vec3 " << evalName << "_schlick(float ct) {" << endl
            << "    float ctSqr = ct*ct, ct5 = ctSqr*ctSqr*ct;" << endl
            << "    return " << evalName << "_R0 + (vec3(1.0) - " << evalName << "_R0) * ct5;" << endl
            << "}" << endl
            << endl
            << "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
            << "   if (cosTheta(wi) <= 0 || cosTheta(wo) <= 0)" << endl
            << "        return vec3(0.0);" << endl
            << "   vec3 H = normalize(wi + wo);" << endl
            << "   vec3 reflectance = " << depNames[0] << "(uv);" << endl
            << "   float alphaU = max(0.2, " << depNames[1] << "(uv).r);" << endl
            << "   float alphaV = max(0.2, " << depNames[2] << "(uv).r);" << endl
            << "   float D = " << evalName << "_D(H, alphaU, alphaV)" << ";" << endl
            << "   float G = " << evalName << "_G(H, wi, wo);" << endl
            << "   vec3 F = " << evalName << "_schlick(1-dot(wi, H));" << endl
            << "   return reflectance * F * (D * G / (4*cosTheta(wi)));" << endl
            << "}" << endl
            << endl
            << "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
            << "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
            << "        return vec3(0.0);" << endl
            << "    return " << evalName << "_R0 * inv_pi * inv_pi * cosTheta(wo);"<< endl
            << "}" << endl;
    }
    MTS_DECLARE_CLASS()
private:
    ref<const Texture> m_specularReflectance;
    ref<const Texture> m_alphaU;
    ref<const Texture> m_alphaV;
    ref<Shader> m_specularReflectanceShader;
    ref<Shader> m_alphaUShader;
    ref<Shader> m_alphaVShader;
    Spectrum m_R0;
};

Shader *RoughConductor::createShader(Renderer *renderer) const {
    return new RoughConductorShader(renderer,
        m_specularReflectance.get(), m_alphaU.get(), m_alphaV.get(), m_coeffs[0]);
}

MTS_IMPLEMENT_CLASS(RoughConductorShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(RoughConductor, false, BSDF)
MTS_EXPORT_PLUGIN(RoughConductor, "Rough conductor BRDF");
MTS_NAMESPACE_END
