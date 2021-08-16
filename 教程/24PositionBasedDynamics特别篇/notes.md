布料，除了距离外，还有有限元限制

ClothCollisionDemo.cpp

```
bool PositionBasedDynamics::init_FEMTriangleConstraint(	
	const Vector3r &p0,
	const Vector3r &p1,
	const Vector3r &p2,
	Real &area, 
	Matrix2r &invRestMat)
{
	Vector3r normal0 = (p1 - p0).cross(p2 - p0);
	area = normal0.norm() * static_cast<Real>(0.5);

	Vector3r axis0_1 = p1 - p0;
	axis0_1.normalize();
	Vector3r axis0_2 = normal0.cross(axis0_1);
	axis0_2.normalize();

	Vector2r p[3];
	p[0] = Vector2r(p0.dot(axis0_2), p0.dot(axis0_1));
	p[1] = Vector2r(p1.dot(axis0_2), p1.dot(axis0_1));
	p[2] = Vector2r(p2.dot(axis0_2), p2.dot(axis0_1));

	Matrix2r P;
	P(0, 0) = p[0][0] - p[2][0];
	P(1, 0) = p[0][1] - p[2][1];
	P(0, 1) = p[1][0] - p[2][0];
	P(1, 1) = p[1][1] - p[2][1];

	const Real det = P.determinant();
	if (fabs(det) > eps)
	{
		invRestMat = P.inverse();
		return true;
	}
	return false;
}
```

特别推荐

**Dynamic Deformables:**

**Implementation and Production**

**Practicalities**

首先是St.Venant Kirchhoff ，还是GreenStrain
$$
\psi _{stVK,stretch} = \frac{1}{2}||\bold F^T \bold F - \bold I||^2_F
$$

```
// epsilon = 0.5(F^T * F - I)
	Matrix2r epsilon;
	epsilon(0,0) = static_cast<Real>(0.5)*(F(0,0) * F(0,0) + F(1,0) * F(1,0) + F(2,0) * F(2,0) - static_cast<Real>(1.0));		// xx
	epsilon(1,1) = static_cast<Real>(0.5)*(F(0,1) * F(0,1) + F(1,1) * F(1,1) + F(2,1) * F(2,1) - static_cast<Real>(1.0));		// yy
	epsilon(0,1) = static_cast<Real>(0.5)*(F(0,0) * F(0,1) + F(1,0) * F(1,1) + F(2,0) * F(2,1));			// xy
	epsilon(1,0) = epsilon(0,1);
```

再记一遍，三个黎曼不变量，主要是如果把这三个不变量算出来后，计算别的量就像踩死地上的蚂蚁一样简单。

https://en.wikipedia.org/wiki/Alternative_stress_measures

First Piola - Kirchhoff
$$
\bold P = J \sigma \bold F^{-1}
$$

```
// P(F) = det(F) * C*E * F^-T => E = green strain
	Matrix2r stress;
	stress(0,0) = C(0,0) * epsilon(0,0) + C(0,1) * epsilon(1,1) + C(0,2) * epsilon(0,1);
	stress(1,1) = C(1,0) * epsilon(0,0) + C(1,1) * epsilon(1,1) + C(1,2) * epsilon(0,1);
	stress(0,1) = C(2,0) * epsilon(0,0) + C(2,1) * epsilon(1,1) + C(2,2) * epsilon(0,1);
	stress(1,0) = stress(0,1);

	const Eigen::Matrix<Real, 3, 2> piolaKirchhoffStres = F * stress;
```

其中Kirchhoff stress
$$
\tau = J \sigma
$$
J是F是det

C是 Orthotropic elasticity tensor

