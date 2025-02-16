class SimpleVolumetricPathTracer : public MonteCarloIntegrator {
public:
    SimpleVolumetricPathTracer(const Properties &props) : MonteCarloIntegrator(props) { }

    /// Unserialize from a binary data stream
    SimpleVolumetricPathTracer(Stream *stream, InstanceManager *manager)
     : MonteCarloIntegrator(stream, manager) { }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        MediumSamplingRecord mRec;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        bool nullChain = true, scattered = false;
        Float eta = 1.0f;

        /* Perform the first ray intersection (or ignore if the
           intersection has already been provided). */
        rRec.rayIntersect(ray);
        Spectrum throughput(1.0f);

        if (m_maxDepth == 1)
            rRec.type &= RadianceQueryRecord::EEmittedRadiance;

        /**
         * Note: the logic regarding maximum path depth may appear a bit
         * strange. This is necessary to get this integrator's output to
         * exactly match the output of other integrators under all settings
         * of this parameter.
         */
        while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
            /* ==================================================================== */
            /*                 Radiative Transfer Equation sampling                 */
            /* ==================================================================== */
            if (rRec.medium && rRec.medium->sampleDistance(Ray(ray, 0, its.t), mRec, rRec.sampler)) {
                /* Sample the integral
                   \int_x^y tau(x, x') [ \sigma_s \int_{S^2} \rho(\omega,\omega') L(x,\omega') d\omega' ] dx'
                */
                const PhaseFunction *phase = rRec.medium->getPhaseFunction();

                throughput *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;

                /* ==================================================================== */
                /*                     Direct illumination sampling                     */
                /* ==================================================================== */

                /* Estimate the single scattering component if this is requested */
                if (rRec.type & RadianceQueryRecord::EDirectMediumRadiance) {
                    DirectSamplingRecord dRec(mRec.p, mRec.time);
                    int maxInteractions = m_maxDepth - rRec.depth - 1;

                    Spectrum value = scene->sampleAttenuatedEmitterDirect(
                            dRec, rRec.medium, maxInteractions,
                            rRec.nextSample2D(), rRec.sampler);

                    if (!value.isZero())
                        Li += throughput * value * phase->eval(
                                PhaseFunctionSamplingRecord(mRec, -ray.d, dRec.d));
                }

                /* Stop if multiple scattering was not requested, or if the path gets too long */
                if ((rRec.depth + 1 >= m_maxDepth && m_maxDepth > 0) ||
                    !(rRec.type & RadianceQueryRecord::EIndirectMediumRadiance))
                    break;

                /* ==================================================================== */
                /*             Phase function sampling / Multiple scattering            */
                /* ==================================================================== */

                PhaseFunctionSamplingRecord pRec(mRec, -ray.d);
                Float phaseVal = phase->sample(pRec, rRec.sampler);
                if (phaseVal == 0)
                    break;
                throughput *= phaseVal;

                /* Trace a ray in this direction */
                ray = Ray(mRec.p, pRec.wo, ray.time);
                ray.mint = 0;
                scene->rayIntersect(ray, its);
                nullChain = false;
                scattered = true;
            } else {
                /* Sample
                    tau(x, y) * (Surface integral). This happens with probability mRec.pdfFailure
                    Account for this and multiply by the proper per-color-channel transmittance.
                */

                if (rRec.medium)
                    throughput *= mRec.transmittance / mRec.pdfFailure;

                if (!its.isValid()) {
                    /* If no intersection could be found, possibly return
                       attenuated radiance from a background luminaire */
                    if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                        && (!m_hideEmitters || scattered)) {
                        Spectrum value = throughput * scene->evalEnvironment(ray);
                        if (rRec.medium)
                            value *= rRec.medium->evalTransmittance(ray);
                        Li += value;
                    }
                    break;
                }

                /* Possibly include emitted radiance if requested */
                if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                    && (!m_hideEmitters || scattered))
                    Li += throughput * its.Le(-ray.d);

                /* Include radiance from a subsurface integrator if requested */
                if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
                    Li += throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);

                /* Prevent light leaks due to the use of shading normals */
                Float wiDotGeoN = -dot(its.geoFrame.n, ray.d),
                      wiDotShN  = Frame::cosTheta(its.wi);
                if (m_strictNormals && wiDotGeoN * wiDotShN < 0)
                    break;

                /* ==================================================================== */
                /*                     Direct illumination sampling                     */
                /* ==================================================================== */

                const BSDF *bsdf = its.getBSDF(ray);

                /* Estimate the direct illumination if this is requested */
                if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
                        (bsdf->getType() & BSDF::ESmooth)) {
                    DirectSamplingRecord dRec(its);
                    int maxInteractions = m_maxDepth - rRec.depth - 1;

                    Spectrum value = scene->sampleAttenuatedEmitterDirect(
                            dRec, its, rRec.medium, maxInteractions,
                            rRec.nextSample2D(), rRec.sampler);

                    if (!value.isZero()) {
                        /* Allocate a record for querying the BSDF */
                        BSDFSamplingRecord bRec(its, its.toLocal(dRec.d));
                        bRec.sampler = rRec.sampler;

                        Float woDotGeoN = dot(its.geoFrame.n, dRec.d);
                        /* Prevent light leaks due to the use of shading normals */
                        if (!m_strictNormals ||
                            woDotGeoN * Frame::cosTheta(bRec.wo) > 0)
                            Li += throughput * value * bsdf->eval(bRec);
                    }
                }

                /* ==================================================================== */
                /*                   BSDF sampling / Multiple scattering                */
                /* ==================================================================== */

                /* Sample BSDF * cos(theta) */
                BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
                Spectrum bsdfVal = bsdf->sample(bRec, rRec.nextSample2D());
                if (bsdfVal.isZero())
                    break;

                /* Recursively gather indirect illumination? */
                int recursiveType = 0;
                if ((rRec.depth + 1 < m_maxDepth || m_maxDepth < 0) &&
                    (rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                    recursiveType |= RadianceQueryRecord::ERadianceNoEmission;

                /* Recursively gather direct illumination? This is a bit more
                   complicated by the fact that this integrator can create connection
                   through index-matched medium transitions (ENull scattering events) */
                if ((rRec.depth < m_maxDepth || m_maxDepth < 0) &&
                    (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) &&
                    (bRec.sampledType & BSDF::EDelta) &&
                    (!(bRec.sampledType & BSDF::ENull) || nullChain)) {
                    recursiveType |= RadianceQueryRecord::EEmittedRadiance;
                    nullChain = true;
                } else {
                    nullChain &= bRec.sampledType == BSDF::ENull;
                }

                /* Potentially stop the recursion if there is nothing more to do */
                if (recursiveType == 0)
                    break;
                rRec.type = recursiveType;

                /* Prevent light leaks due to the use of shading normals */
                const Vector wo = its.toWorld(bRec.wo);
                Float woDotGeoN = dot(its.geoFrame.n, wo);
                if (woDotGeoN * Frame::cosTheta(bRec.wo) <= 0 && m_strictNormals)
                    break;

                /* Keep track of the throughput, medium, and relative
                   refractive index along the path */
                throughput *= bsdfVal;
                eta *= bRec.eta;
                if (its.isMediumTransition())
                    rRec.medium = its.getTargetMedium(wo);

                /* In the next iteration, trace a ray in this direction */
                ray = Ray(its.p, wo, ray.time);
                scene->rayIntersect(ray, its);
                scattered |= bRec.sampledType != BSDF::ENull;
            }

            if (rRec.depth++ >= m_rrDepth) {
                /* Russian roulette: try to keep path weights equal to one,
                   while accounting for the solid angle compression at refractive
                   index boundaries. Stop with at least some probability to avoid
                   getting stuck (e.g. due to total internal reflection) */

                Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
                if (rRec.nextSample1D() >= q)
                    break;
                throughput /= q;
            }
        }
        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;
        return Li;
    }
