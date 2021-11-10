//https://github.com/tflsguoyu/layeredbsdf
//phase function (normalized as a pdf)
    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        Spectrum result;

        Float weight = std::min((Float) 1.0f, std::max((Float) 0.0f,
            m_weight->eval(bRec.its).average()));

        if (bRec.component == -1) {
            return
                m_bsdfs[0]->pdf(bRec, measure) * (1-weight) +
                m_bsdfs[1]->pdf(bRec, measure) * weight;
        } else {
            /* Pick out an individual component */
            int idx = m_indices[bRec.component].first;
            if (idx == 0)
                weight = 1-weight;
            BSDFSamplingRecord bRec2(bRec);
            bRec2.component = m_indices[bRec.component].second;
            return m_bsdfs[idx]->pdf(bRec2, measure) * weight;
        }
    }