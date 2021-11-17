#include "pseries_ratio.h"
#include <tgmath.h>
https://github.com/ZackMisso/TransmittanceEstimation
Pseries_Ratio::Pseries_Ratio() : Estimator() { }

Float Pseries_Ratio::T(TransmittanceQuaryRecord& rec, Sampler* sampler) const {
    Float sample = sampler->next1D();

    Float maj = rec.extFunc->calculateMajorant((rec.b - rec.a) / 2.0 + rec.a);
    Float tau = maj * (rec.b - rec.a);
    Float prob = exp(-tau);
    Float cdf = prob;
    Float Tr = 1.0;

    int i = 1;

    for (; cdf < sample; cdf += prob, ++i)
    {
        Float x = sampleUniFF(rec, sampler);
        Tr *= 1.0 - rec.extFunc->calculateExtinction(x, rec.extCalls) / maj;
        prob *= tau / Float(i);
    }

    rec.transmittance = Tr;

    return Tr;
}

string Pseries_Ratio::getName() const {
    return "pseries_ratio";
}
