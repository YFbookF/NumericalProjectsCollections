Real-time Interactive Simulation of Smoke Using Discrete Integrable Vortex
Filaments.  

voriticity form
$$
\dot \omega =curl(\omega \times u)
$$
如果想反过来找u，可以用下面这种方式
$$
u(x) = -\frac{1}{4\pi}\int_{\mathcal{R^3}}\frac{x-z}{||x-z||^2}\times w(z)dz
$$
This is an integral over the whole of R3, but with the restriction to vorticity fields that are supported on tubular neighbourhoods of closed space curves γk, Equation (2) reduces to a sum of line integrals – the Biot-Savart law:  
$$
u(x) = \sum_k -\frac{\Gamma_k}{4\pi}\oint\frac{x-\gamma_k(s)}{||x-\gamma k(s)||^3} \times \gamma_k' (s)ds
$$
代码如下

```
inline Vec3 FilamentKernel(const Vec3& pos, const vector<VortexRing>& rings, const vector<BasicParticleData>& fp, Real reg, Real cutoff, Real scale) {
	const Real strength = 0.25 / M_PI * scale;
	const Real a2 = square(reg);
	const Real cutoff2 = square(cutoff);
	const Real mindist = 1e-6;
	Vec3 u(_0);
	
	for (size_t i=0; i<rings.size(); i++) {
		const VortexRing& r = rings[i];
		if (r.flag & ParticleBase::PDELETE) continue;
		
		const int N = r.isClosed ? (r.size()) : (r.size()-1);
		const Real str = strength * r.circulation;
		for (int j=0; j<N; j++) {
			const Vec3 r0 = fp[r.idx0(j)].pos - pos;
			const Vec3 r1 = fp[r.idx1(j)].pos - pos;
			const Real r0_2 = normSquare(r0), r1_2 = normSquare(r1);
			if (r0_2 > cutoff2 || r1_2 > cutoff2 || r0_2 < mindist || r1_2 < mindist)
				continue;
			
			const Vec3 e = getNormalized(r1-r0);
			const Real r0n = 1.0f/sqrt(a2+r0_2);
			const Real r1n = 1.0f/sqrt(a2+r1_2);
			const Vec3 cp = cross(r0,e);
			const Real A = str * (dot(r1,e)*r1n - dot(r0,e)*r0n) / (a2 + normSquare(cp));
			u += A * cp;
		}
	}
	return u;
}
```

