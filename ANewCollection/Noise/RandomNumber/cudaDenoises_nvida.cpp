//https://github.com/ZheyuanXie/Project3-CUDA-Path-Tracer
//////////
/* Random number generator */
// reference: https://research.nvidia.com/publication/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A

// Generates a seed for a random number generator from 2 inputs plus a backoff
__host__ __device__
unsigned int initRand(unsigned int val0, unsigned int val1, unsigned int backoff = 16)
{
    unsigned int v0 = val0, v1 = val1, s0 = 0;

    for (unsigned int n = 0; n < backoff; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

// Takes our seed, updates it, and returns a pseudorandom float in [0..1]
__host__ __device__
float nextRand(unsigned int& s)
{
    s = (1664525u * s + 1013904223u);
    return float(s & 0x00FFFFFF) / float(0x01000000);
}