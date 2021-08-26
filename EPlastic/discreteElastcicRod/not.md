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

