```
uint32_t UpdateForces(Particle* particles, uint32_t numParticles, 
	const Triangle* triangles, Element* elements, uint32_t numTriangles,
   	Vec2 gravity, float lameLambda, float lameMu, float damp, float drag, float dt, 
	FractureEvent* fractures, uint32_t maxFractures, float toughness, float yield, float creep)
{

	for (uint32_t i=0; i < numParticles; ++i)
	{
		particles[i].f += particles[i].invMass>0.0f?(gravity/particles[i].invMass):Vec2(0.0f) - drag*particles[i].v;
	}

	uint32_t numFractures = 0;

	for (uint32_t i=0; i < numTriangles; ++i)
	{
		const Triangle& tri = triangles[i];
		Element& elem = elements[i];

		// read particles into a local array
		Vec2 x[3] = { particles[tri.i].p, particles[tri.j].p, particles[tri.k].p };
		Vec2 v[3] = { particles[tri.i].v, particles[tri.j].v, particles[tri.k].v };

		if (1)
		{
			Matrix22 f = CalcDeformation(x, elem.mInvDm);
			Matrix22 q = QRDecomposition(f);

			// strain 
			Matrix22 e = CalcCauchyStrainTensor(Transpose(q)*f);
			//if (FrobeniusNorm(e) > 0.2f)
			//	printf("%f\n", FrobeniusNorm(e));

			// update plastic strain
			float ef = FrobeniusNorm(e);
		
			//if (ef > yield)
			//	printf("%f\n", ef);

			if (ef > yield)
				elem.mEp += dt*creep*e;
			
			const float epmax = 0.6f;	
			if (ef > epmax)	
				elem.mEp *= epmax / ef;  

			// adjust strain
			e -= elem.mEp;

			Matrix22 s = CalcStressTensor(e, lameLambda, lameMu);

			// damping forces	
			Matrix22 dfdt = CalcDeformation(v, elem.mInvDm);
			Matrix22 dedt = CalcCauchyStrainTensorDt(Transpose(q)*dfdt);
			Matrix22 dsdt = CalcStressTensor(dedt, damp, damp);

			Matrix22 p = s + dsdt;
		
			/*	
			static int z = 0;
		   	if (1)
			{	
				Vec2 c = (x[0]+x[1]+x[2])/3.0f;
				ShowMatrix(e, q, c);
			}
			++z;	
			*/

			float e1, e2;
			EigenDecompose(p, e1, e2);

			float me = max(e1, e2);

			if (me > toughness && numFractures < maxFractures)
			{
				// calculate Eigenvector corresponding to max Eigenvalue
				Vec2 ev = q*Normalize(Vec2(p(0,1), me-p(0,0)));

				// pick a random vertex to split on
				uint32_t splitNode = rand()%3;

				// don't fracture immovable nodes
				if (particles[GetVertex(tri, splitNode)].invMass == 0.0f)
					break;

				// fracture plane perpendicular to ev
				Vec3 p(ev.x, ev.y, -Dot(ev, particles[GetVertex(tri, splitNode)].p));

				FractureEvent f = { i, splitNode, p };

				fractures[numFractures++] = f;
			}

			// calculate force on each edge due to stress and distribute to the nodes
			Vec2 f1 = q*p*elem.mB[0];
			Vec2 f2 = q*p*elem.mB[1];
			Vec2 f3 = q*p*elem.mB[2];
		
			particles[tri.i].f -= f1/3.0f;
			particles[tri.j].f -= f2/3.0f;
			particles[tri.k].f -= f3/3.0f;
		}
		else
		{
			Matrix22 f = CalcDeformation(x, elem.mInvDm);

			// elastic forces
			Matrix22 e = CalcGreenStrainTensor(f);	
			Matrix22 s = CalcStressTensor(e, lameLambda, lameMu);
	
			// damping forces	
			Matrix22 dfdt = CalcDeformation(v, elem.mInvDm);
			Matrix22 dedt = CalcGreenStrainTensorDt(f, dfdt);
			Matrix22 dsdt = CalcStressTensor(dedt, damp, damp);

			Matrix22 p = s + dsdt;

			float det;	
			Matrix22 finv = Inverse(Transpose(f), det);

			Vec2 f1 = p*(finv*elem.mB[0]);
			Vec2 f2 = p*(finv*elem.mB[1]);
			Vec2 f3 = p*(finv*elem.mB[2]);

			particles[tri.i].f -= f1/3.0f;
			particles[tri.j].f -= f2/3.0f;
			particles[tri.k].f -= f3/3.0f;
		}
	}

	return numFractures;
}
```

剩下的公式计算，写得很清楚

```
float FrobeniusNorm(const Matrix22& m)
{
	float f = 0.0f;

	for (uint32_t i=0; i < 2; ++i)
		for (uint32_t j=0; j < 2; ++j)
			f += m(i, j)*m(i, j);

	return sqrtf(f);
}

// deformation gradient
Matrix22 CalcDeformation(const Vec2 x[3], const Matrix22& invM)
{	
	Vec2 e1 = x[1]-x[0]; 
	Vec2 e2 = x[2]-x[0]; 

	Matrix22 m(e1, e2);

	// mapping from material coordinates to world coordinates	
	Matrix22 f = m*invM; 	
	return f;
}

// calculate Green's non-linear strain tensor
Matrix22 CalcGreenStrainTensor(const Matrix22& f)
{
	Matrix22 e = 0.5f*(f*Transpose(f) - Matrix22::Identity());
	return e;
}

// calculate time derivative of Green's strain
Matrix22 CalcGreenStrainTensorDt(const Matrix22& f, const Matrix22& dfdt)
{
	Matrix22 e = 0.5f*(f*Transpose(dfdt) + dfdt*Transpose(f));
	return e;
}

// calculate Cauchy's linear strain tensor
Matrix22 CalcCauchyStrainTensor(const Matrix22& f)
{
	Matrix22 e = 0.5f*(f + Transpose(f)) - Matrix22::Identity();
	return e;
}

// calculate time derivative of Cauchy's strain tensor
Matrix22 CalcCauchyStrainTensorDt(const Matrix22& dfdt)
{
	Matrix22 e = 0.5f*(dfdt + Transpose(dfdt));
	return e;
}

// calculate isotropic Hookean stress tensor, lambda and mu are the Lame parameters
Matrix22 CalcStressTensor(const Matrix22& e, float lambda, float mu)
{
	Matrix22 s = lambda*Trace(e)*Matrix22::Identity() + mu*2.0f*e;
	return s;
}
```

mInvM定义如下

```
	Element(const Vec2 x[3])
	{
		Vec2 e1 = x[1]-x[0]; 
		Vec2 e2 = x[2]-x[0]; 
		Vec2 e3 = x[2]-x[1];

		Matrix22 m(e1, e2);

		float det;	
		mInvDm = Inverse(m, det);
```

旋转如下

```
// returns the ccw perpendicular vector 
template <typename T>
CUDA_CALLABLE XVector2<T> PerpCCW(const XVector2<T>& v)
{
	return XVector2<T>(-v.y, v.x);
}

template <typename T>
CUDA_CALLABLE XVector2<T> PerpCW(const XVector2<T>& v)
{
	return XVector2<T>(v.y, -v.x);
}
```

```
CUDA_CALLABLE inline Matrix22 Inverse(const Matrix22& m, float& det)
{
	det = Determinant(m); 

	if (fabsf(det) > FLT_EPSILON)
	{
		Matrix22 inv;
		inv(0,0) =  m(1,1);
		inv(1,1) =  m(0,0);
		inv(0,1) = -m(0,1);
		inv(1,0) = -m(1,0);

		return Multiply(1.0f/det, inv);	
	}
	else
	{
		det = 0.0f;
		return m;
	}
}

CUDA_CALLABLE inline Matrix22 QRDecomposition(const Matrix22& m)
{
	Vec2 a = Normalize(m.cols[0]);
	Matrix22 q(a, PerpCCW(a)); 

	return q;
}

CUDA_CALLABLE inline Matrix22 PolarDecomposition(const Matrix22& m)
{
	/*
	//iterative method
	 
	float det;
	Matrix22 q = m;

	for (int i=0; i < 4; ++i)
	{
		q = 0.5f*(q + Inverse(Transpose(q), det));
	}
	*/	

	Matrix22 q = m + Matrix22(m(1,1), -m(1,0), -m(0,1), m(0,0));

	float s = Length(q.cols[0]);
	q.cols[0] /= s;
	q.cols[1] /= s;

	return q;
}


```

