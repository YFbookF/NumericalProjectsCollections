https://github.com/mitsuba-renderer/mitsuba
https://github.com/mitsuba-renderer/mitsuba
void IrradianceOctree::propagate(OctreeNode *node) {
    IrradianceSample &repr = node->data;

    /* Initialize the cluster values */
    repr.E = Spectrum(0.0f);
    repr.area = 0.0f;
    repr.p = Point(0.0f, 0.0f, 0.0f);
    Float weightSum = 0.0f;

    if (node->leaf) {
        /* Inner node */
        for (uint32_t i=0; i<node->count; ++i) {
            const IrradianceSample &sample = m_items[i+node->offset];
            repr.E += sample.E * sample.area;
            repr.area += sample.area;
            Float weight = sample.E.getLuminance() * sample.area;
            repr.p += sample.p * weight;
            weightSum += weight;
        }
        statsNumSamples += node->count;
    } else {
        /* Inner node */
        for (int i=0; i<8; i++) {
            OctreeNode *child = node->children[i];
            if (!child)
                continue;
            propagate(child);
            repr.E += child->data.E * child->data.area;
            repr.area += child->data.area;
            Float weight = child->data.E.getLuminance() * child->data.area;
            repr.p += child->data.p * weight;
            weightSum += weight;
        }
    }
    if (repr.area != 0)
        repr.E /= repr.area;
    if (weightSum != 0)
        repr.p /= weightSum;

    ++statsNumNodes;
}
