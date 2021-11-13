
https://github.com/mitsuba-renderer/mitsuba
    /**
     * Computes a bloom filter based on
     *
     * "Physically-Based Glare Effects for Digital Images" by
     * Greg Spencer, Peter Shirley, Kurt Zimmerman and Donald P. Greenberg
     * SIGGRAPH 1995
     */
    ref<Bitmap> computeBloomFilter(int size, Float fov) {
        ref<Bitmap> bitmap = new Bitmap(Bitmap::ELuminance, Bitmap::EFloat, Vector2i(size));

        Float scale       = 2.f / (size - 1),
              halfLength  = std::tan(.5f * degToRad(fov));

        Float *ptr = bitmap->getFloatData();
        double sum = 0;

        for (int y=0; y<size; ++y) {
            for (int x=0; x<size; ++x) {
                Float xf = x*scale - 1,
                      yf = y*scale - 1,
                      r = std::sqrt(xf*xf+yf*yf),
                      angle = radToDeg(std::atan(r * halfLength)),
                      tmp   = angle + 0.02f,
                      f0 = 2.61e6f * math::fastexp(-2500*angle*angle),
                      f1 = 20.91f / (tmp*tmp*tmp),
                      f2 = 72.37f / (tmp*tmp),
                      f  = 0.384f*f0 + 0.478f*f1 + 0.138f*f2;

                *ptr++ = f;
                sum += f;
            }
        }
        ptr = bitmap->getFloatData();
        Float normalization = (Float) (1/sum);
        for (int i=0; i<size*size; ++i)
            *ptr++ *= normalization;

        return bitmap;
    }