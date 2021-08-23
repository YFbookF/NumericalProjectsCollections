http://ntoken.com/pubs.html#Thuerey_2016_ofblend

**Synthetic Turbulence using Artificial Boundary Layers**

代码简洁，注释详细，难以运行

split和merge看不懂

湍流公式写得不错

不完全Cholesky和Mod

wavelet

三线性速度梯度

```
	// calculate velocity gradient matrix at particles position by trilinear interpolation
	inline Vec3 velGradient(Grid<Vec3>* grid, const Vec3& pos, int idx)
	{
		Vec3 res;
		for (int n = 0; n < 3; n++)
		{
			// translate grid (MAC)
			Vec3 cpos(pos);
			cpos[n] += .5; cpos[idx] -= 0.5;
			nVec3i id(vec2I(cpos)), nid(id + 1), nd(id), nnd(nid);
			Vec3 a(cpos - vec2R(id)), na(-a + 1.);
			nd[n]++; nnd[n]++;

			// trilinear interpolation
			res[n] = na.x * na.y * na.z * ((*grid)(nd.x, nd.y, nd.z)[idx] - (*grid)(id.x, id.y, id.z)[idx]) +
				na.x * na.y * a.z * ((*grid)(nd.x, nd.y, nnd.z)[idx] - (*grid)(id.x, id.y, nid.z)[idx]) +
				na.x * a.y * na.z * ((*grid)(nd.x, nnd.y, nd.z)[idx] - (*grid)(id.x, nid.y, id.z)[idx]) +
				na.x * a.y * a.z * ((*grid)(nd.x, nnd.y, nnd.z)[idx] - (*grid)(id.x, nid.y, nid.z)[idx]) +
				a.x * na.y * na.z * ((*grid)(nnd.x, nd.y, nd.z)[idx] - (*grid)(nid.x, id.y, id.z)[idx]) +
				a.x * na.y * a.z * ((*grid)(nnd.x, nd.y, nnd.z)[idx] - (*grid)(nid.x, id.y, nid.z)[idx]) +
				a.x * a.y * na.z * ((*grid)(nnd.x, nnd.y, nd.z)[idx] - (*grid)(nid.x, nid.y, id.z)[idx]) +
				a.x * a.y * a.z * ((*grid)(nnd.x, nnd.y, nnd.z)[idx] - (*grid)(nid.x, nid.y, nid.z)[idx]);
		}
		return res;
	}
```

VortexStretch

```
	// perform vortex stretching
	void VortexParticle::vortexStretch(Grid<Vec3>* pVel, Real dt, Real mult)
	{
		Vec3 spos = mPos * mult;
		// gradient
		Vec3 s;
		s.x = dot(mStrength, velGradient(pVel, spos, 0));
		s.y = dot(mStrength, velGradient(pVel, spos, 1));
		s.z = dot(mStrength, velGradient(pVel, spos, 2));

		Vec3 nstr = mStrength + dt * s;
		Real oldStr2 = normNoSqrt(mStrength);
		Real newStr2 = normNoSqrt(nstr);

		// do not allow the magnitude to grow
		if (newStr2 > MININV2 && (newStr2 > oldStr2 || msStretchingPreserve))
			nstr *= sqrt(oldStr2 / newStr2);

		// apply vorticity update
		mStrength = nstr;
	}
```

