```
// Compute the curvature binormal for a vertex between two edges with tangents
// e0 and e1, respectively
// (edge tangent vectors not necessarily normalized)
template<typename Real_>
Vec3_T<Real_> curvatureBinormal(const Vec3_T<Real_> &e0, const Vec3_T<Real_> &e1) {
    return e0.cross(e1) * (2.0 / (e0.norm() * e1.norm() + e0.dot(e1)));
}
```

https://github.com/jpanetta/ElasticRods 注释详细

https://github.com/Reimilia/DiscreteElasticRods

A new version of Bishop frame and an application to spherical images

https://www.youtube.com/watch?v=VIqA8U9ozIA
$$
\frac{d\vec B}{d s} = \frac{d(\vec T \times \vec N)}{ds} = \kappa \vec N \times \vec N + \vec T \times \frac{d \vec N}{ds} = \vec T \times \frac{d \vec N}{ds}
$$
记住
$$
if |\vec r(t)| = 1 \qquad \vec r'(t) \cdot \vec{r}(t) = 0
$$
那么
$$
\frac{d\vec B}{d s} \cdot \vec{T} = \frac{d\vec B}{d s} \cdot \vec{B} = 0 \qquad   
$$
那么
$$
-\frac{d\vec B}{ds} = \tau \vec{N} \qquad-\frac{d\vec B}{ds} \cdot \vec{N} = \tau \vec{N} \cdot \vec{N} = \tau(torsion)
$$
cuvature代表的曲线如何弯曲，torsion代表平面如何弯曲，这就是Frenet-Serret
$$
\frac{d \vec{T}}{ds} = \kappa \vec N \qquad \frac{d \vec{N}}{ds} = -\kappa \vec T + \tau \vec B \qquad \frac{d \vec{B}}{ds} = -\tau \vec N 
$$
bishop frame
$$
\frac{d\vec T}{ds} = \kappa_1 \vec M_1 + \kappa_2\vec M_2 \qquad \frac{d\vec M_1}{ds} = -\kappa_1 \vec T\qquad \frac{d\vec M_2}{ds} = -\kappa_2 \vec T
$$
注意**A Characterization for Bishop Equations of Parallel Curves according**

**to Bishop Frame in**
$$
\bold T = \bold T \\
\bold N = \cos\theta(s)\bold M_1 + \sin\theta(s)\bold M_2\\
\bold B = -\sin\theta(s)\bold M_1 + \cos\theta(s)\bold M_1
$$
而discreteElasticRod里的公式长这样
$$
\bold m_1^i = \cos\theta^i \cdot \bold u^i + \sin \theta^i \cdot \bold v^i\\
\bold m_2^i = - \sin \theta^i \cdot \bold u^i + \cos \theta^i \cdot \bold v^i
$$

```
void ElasticRod::computeMaterialFrame(const ColumnVector & theta,
	std::vector<mg::Vec3D>&io_m1,
	std::vector<mg::Vec3D>&io_m2) const
{
	mg::Real sinQ, cosQ;
	mg::Vec3D m1, m2;
	for (unsigned i = 0; i < io_m1.size(); ++i)
	{
		cosQ = std::cos(theta(i));
		sinQ = std::sqrt(1 - cosQ * cosQ);

		m1 = cosQ * io_m1[i] + sinQ * io_m2[i];
		m2 = -sinQ * io_m1[i] + cosQ * io_m2[i];

		io_m1[i] = m1;
		io_m2[i] = m2;
	}
}
```

用DiscreteParallelTransport算出BishopFrame

https://github.com/Reimilia/DiscreteElasticRods

```
void ElasticRod::updateBishopFrame()
{
	/*	
	Compute Discrete Parallel Transportation to update Bishop Frame for centerline
	Move coordinate frame via the rotational matrix
	*/

	int nRods = (int)rods.size();
	rods[0].t = nodes[1].pos - nodes[0].pos;
	rods[0].t = rods[0].t / rods[0].t.norm();
	rods[0].u = Eigen::Vector3d(rods[0].t[2] - rods[0].t[1], rods[0].t[0] - rods[0].t[2], rods[0].t[1] - rods[0].t[0]);
	rods[0].u = rods[0].u / rods[0].u.norm();
	rods[0].v = rods[0].t.cross(rods[0].u);
	rods[0].v /= rods[0].v.norm();
	

	// Now compute Bishop frame
	for (int i = 1; i < nRods; i++)
	{
		rods[i].t = nodes[i + 1].pos - nodes[i].pos;
		rods[i].t = (rods[i].t) / (rods[i].t).norm();
		Eigen::Vector3d n = (rods[i - 1].t).cross(rods[i].t);

		// Watchout!
		if (n.norm() < 1e-10)
		{
			rods[i].u = rods[i - 1].u;
		}
		else
		{
			if (rods[i].t.dot(rods[i - 1].t) > 0)
			{
				rods[i].u = VectorMath::rotationMatrix(n*asin(n.norm()) / n.norm()) * rods[i - 1].u;
			}
			else
			{
				rods[i].u = VectorMath::rotationMatrix(n*(M_PI - asin(n.norm())) / n.norm()) * rods[i - 1].u;
			}
			
		}
		//rods[i].u = VectorMath::rotationMatrix(stencils[i-1].kb) * rods[i-1].u;
		
		rods[i].u = (rods[i].u) / (rods[i].u).norm();

		rods[i].v = (rods[i].t).cross(rods[i].u);
	}
}
```

对应公式2

```
void ElasticRod::updateMaterialCurvature()
{
	/*
	Compute material curvatures
	*/
	int nRods = (int)rods.size();
	for (int i = 0; i < nRods; ++i)
	{
		Eigen::Vector3d m1 = cos(rods[i].theta)* rods[i].u + sin(rods[i].theta)* rods[i].v;
		Eigen::Vector3d m2 = -sin(rods[i].theta)* rods[i].u + cos(rods[i].theta)* rods[i].v;
		if (i < nRods - 1)
		{
			stencils[i].prevCurvature = Eigen::Vector2d((stencils[i].kb).dot(m2), -(stencils[i].kb).dot(m1));
		}
		if (i > 0)
		{
			stencils[i - 1].nextCurvature = Eigen::Vector2d((stencils[i - 1].kb).dot(m2), -(stencils[i - 1].kb).dot(m1));
		}
		
	}
}
```

bend能量
$$
E_{bend}(\Gamma) = \sum_{i=1}^n \frac{1}{2l_i}\sum_{j=i-1}^i(\omega_i^j - \overline\omega_i^j)^T\overline B^j(\omega_i^j - \overline \omega_i^j)
$$

```
		double e1 = (stencils[i].prevCurvature - restCurvature.col(2 * i)).dot(rods[i].bendingModulus * (stencils[i].prevCurvature - restCurvature.col(2 * i)));
		double e2 = (stencils[i].nextCurvature - restCurvature.col(2 * i + 1)).dot(rods[i + 1].bendingModulus * (stencils[i].nextCurvature - restCurvature.col(2 * i + 1)));
		energy += (e1 + e2) / 2 / stencils[i].restlength;
```

$$
E_{twist}(\Gamma) = \sum_{i=1}^n\beta\frac{(\theta^i - \theta^{i-1})^2}{l_i} = \sum_{i=1}^n\frac{\beta m_i^2}{l_i}
$$

```
		energy += beta * (rods[i + 1].theta - rods[i].theta) * (rods[i + 1].theta - rods[i].theta) / stencils[i].restlength;
		
```

叫kb的意思是因为curvature binormal
$$
(\kappa \bold b)_i = \frac{2\bold e^{i-1} \times \bold e^i}{|\overline {\bold e}^{i-1}||\overline {\bold e}^i| + \bold e^{i-1} \cdot \bold e^i}
$$
然而么有直接算的，只有算导数的

```
		Eigen::Matrix3d dkb1, dkb2;
		e1 = nodes[k + 1].pos - nodes[k].pos;
		e2 = nodes[k + 2].pos - nodes[k + 1].pos;
		dkb1 = (2 * VectorMath::crossProductMatrix(e2) + stencils[k].kb * e2.transpose()) / (restLength[k] * restLength[k + 1] + e1.dot(e2));
		dkb2 = (2 * VectorMath::crossProductMatrix(e1) - stencils[k].kb * e1.transpose()) / (restLength[k] * restLength[k + 1] + e1.dot(e2));
```

书对应的代码

```
void RodAnisoForce::computeNablaKappa()
{
  for (int i = 1; i < m_rod.nv()-1; ++i) {

    const Vec3d& kb = m_rod.getCurvatureBinormal(i);
    const Vec3d& e0 = m_rod.getEdge(i-1);
    Scalar len0 = m_rod.getEdgeLength(i-1);
    const Vec3d& e1 = m_rod.getEdge(i);
    Scalar len1 = m_rod.getEdgeLength(i);

    Mat3d& nk0 = _nk[3*(i-1)+0];
    crossMat(nk0, e1);
    nk0 *= 2;
    nk0 += outerProd(kb,e1);
    nk0 *= 1.0/(len0*len1 + e0.dot(e1));

    Mat3d& nk2 = _nk[3*(i-1)+2];
    crossMat(nk2, e0);
    nk2 *= 2;
    nk2 -= outerProd(kb,e0);
    nk2 *= 1.0/(len0*len1 + e0.dot(e1));

    Mat3d& nk1 = _nk[3*(i-1)+1];
    nk1 = nk0+nk2;
    nk1 *= -1;
  }
}
```

