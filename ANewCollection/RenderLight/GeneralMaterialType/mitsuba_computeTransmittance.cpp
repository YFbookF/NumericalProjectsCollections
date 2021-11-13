https://github.com/mitsuba-renderer/mitsuba
class PrecomputeTransmittance : public Utility {
public:
    Float *computeTransmittance(const char *name, Float ior, Float alpha,
            size_t resolution, Float &diffTrans, int inverted) {
        Properties bsdfProps(alpha == 0 ? "dielectric" : "roughdielectric");
        if (inverted) {
            bsdfProps.setFloat("intIOR", 1.00);
            bsdfProps.setFloat("extIOR", ior);
        } else {
            bsdfProps.setFloat("extIOR", 1.00);
            bsdfProps.setFloat("intIOR", ior);
        }
        bsdfProps.setFloat("alpha", alpha);
        bsdfProps.setString("distribution", name);
        ref<BSDF> bsdf = static_cast<BSDF *>(
                PluginManager::getInstance()->createObject(bsdfProps));

        Float stepSize = 1.0f / (resolution-1);
        Float error;

        NDIntegrator intTransmittance(1, 2, 50000, 0, 1e-6f);
        NDIntegrator intDiffTransmittance(1, 1, 50000, 0, 1e-6f);
        Float *transmittances = new Float[resolution];

        for (size_t i=0; i<resolution; ++i) {
            Float t = i * stepSize;
            if (i == 0) /* Don't go all the way to zero */
                t = stepSize/10;

            Float cosTheta = std::pow(t, (Float) 4.0f);

            Vector wi(math::safe_sqrt(1-cosTheta*cosTheta), 0, cosTheta);

            Float min[2] = {0, 0}, max[2] = {1, 1};
            intTransmittance.integrateVectorized(
                boost::bind(&transmittanceIntegrand, bsdf, wi, _1, _2, _3),
                min, max, &transmittances[i], &error, NULL);
        }

        Float min[1] = { 0 }, max[1] = { 1 };
        intDiffTransmittance.integrateVectorized(
            boost::bind(&diffTransmittanceIntegrand, transmittances, resolution, _1, _2, _3),
            min, max, &diffTrans, &error, NULL);

        if (alpha == 0.0f)
            cout << diffTrans << " vs " << 1-fresnelDiffuseReflectance(inverted ? (1.0f / ior) : ior) << endl;

        return transmittances;
    }