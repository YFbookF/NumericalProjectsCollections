### Euler’s theory of the elastica   

assumes that the bending moment in the
rod is linearly proportional to the curvature of the centerline of the rod, that the
deformation of the rod is planar, and that cross-sections that were normal to the
centerline in an undeformed configuration remain normal to the centerline as the
rod is deformed.   

4.2.1

因为t^k和t^{k-1}是单位向量。那么操作符P，定义为将t^{k-1}转换为t^k的操作，实现上用旋转矩阵，旋转轴是b_k即binormal，这个binomrmal 与 t^k X t^{k+1}平行，也就是binormal的离散版本。

重写写一遍
$$
\bold b_k = \frac{\bold t^{k-1} \times \bold t^k}{||\bold t^{k-1} \times \bold t^k||} \qquad \cos(\psi_k)= \bold t^k \cdot \bold t^{k-1} \qquad P = \bold R(\psi_k,\bold b_k)
$$
那么curvature binormal定义如下
$$
(\kappa \bold b)_k = \kappa_k \bold b_k = \frac{2\bold t^{k-1} \times \bold t^k}{1 + \bold t^{k-1} \cdot \bold t^k}
$$
我们也可以使用darboux vector
$$
\kappa_k \bold b_k = 2\tan(\frac{\psi_k}{2})\bold b_k
$$
To model the deformation of a long slender (i.e., rod-like) body, the behavior of
the cross-sections must be accounted for. In modern theories of rods, deformable
vector fields, which are known as directors, are associated with each point on the
material curve that is used to model the centerline of the rod.  

代码来源书

```
void RodAnisoForce::computeNablaPsi()
{
  for (int i = 1; i < m_rod.nv()-1; ++i) {

    const Vec3d& kb = m_rod.getCurvatureBinormal(i);

    Scalar len0 = m_rod.getEdgeLength(i-1);
    Scalar len1 = m_rod.getEdgeLength(i);

    Vec3d np0 = 0.5/len0 * kb;
    Vec3d np2 = -0.5/len1 * kb;

    _np[3*(i-1)+0] = np0;
    _np[3*(i-1)+1] = -np0-np2;
    _np[3*(i-1)+2] = np2;

  }
}
```

对应论文公式(9)
$$
\nabla_{i-1}\psi_{i} = \frac{(\kappa \bold b)_i}{2|\overline e^{i-1}|} \qquad \nabla_{i+1}\psi_{i} = \frac{(\kappa \bold b)_i}{2|\overline e^{i}|} \qquad \nabla_i \psi_i = -(\nabla_{i-1} + \nabla_{i+1})\psi_i 
$$

```
void RodAnisoForce::computeNablaPsi()
{
  for (int i = 1; i < m_rod.nv()-1; ++i) {

    const Vec3d& kb = m_rod.getCurvatureBinormal(i);

    Scalar len0 = m_rod.getEdgeLength(i-1);
    Scalar len1 = m_rod.getEdgeLength(i);

    Vec3d np0 = 0.5/len0 * kb;
    Vec3d np2 = -0.5/len1 * kb;

    _np[3*(i-1)+0] = np0;
    _np[3*(i-1)+1] = -np0-np2;
    _np[3*(i-1)+2] = np2;
  }
}
```

书公式6.17
$$
\delta(\kappa \bold b)_k = \frac{2\delta \bold e^{k-1} \times \bold e^k}{||\bold e^{k-1}|||\bold e^k|| + \bold e^{k-1} \cdot \bold e^k}
$$
论文公式10.9
$$
\nabla_{i+1}(\kappa \bold b)_i = \frac{2|\bold e^{i-1}| - (\kappa \bold b_i)(\bold e^{i-1})^T}{|\overline {\bold e}^{i-1}||\overline {\bold e}^i| + \bold e^{i-1}\cdot \bold e^i}
$$


```
// finite difference approximation of grad (kb)_i with respect to vertex v
void RodAnisoForce::fdNablaKappa(Mat3d& nk, int i, int v)
{
  Scalar h = 1.0e-4;
  Vec3d e0 = m_rod.getEdge(i-1); Scalar len0 = m_rod.getEdgeLength(i-1);
  Vec3d e1 = m_rod.getEdge(i); Scalar len1 = m_rod.getEdgeLength(i);

  Scalar dir0 = 0, dir1 = 0;
  if (v == i-1) { dir0 = -1; dir1 = 0; }
  else if (v == i+1) { dir0 = 0; dir1 = 1; }
  else if (i == v) { dir0 = 1; dir1 = -1; }
  else assert(0);

  for (int k = 0; k < 3; ++k) {

    e0[k] += dir0*h; e1[k] += dir1*h;
    Vec3d kbp = 2.0/(len0*len1+e0.dot(e1)) * e0.cross(e1);

    e0[k] -= 2*dir0*h; e1[k] -= 2*dir1*h;
    Vec3d kbm = 2.0/(len0*len1+e0.dot(e1)) * e0.cross(e1);

    e0[k] += dir0*h; e1[k] += dir1*h;

    for (int cmp = 0; cmp < 3; ++cmp)
      nk(cmp,k) = (kbp[cmp]-kbm[cmp])/(2*h);
  }
}

```

计算能量

```
  for (int i = 1; i < m_rod.nv() - 1; ++i) {

    Mat2d B = (getB(i - 1) + getB(i)) / 2.0;
    Scalar len = getRefVertexLength(i);

    //Scalar m = m_rod.twist(i);
    //E += _kt * m * m / (2.0 * len);

    for (int j = i - 1; j <= i; ++j) {

      const Vec2d& w = getOmega(i, j);
      const Vec2d& w0 = getOmegaBar(i, j);
      //const Mat2d& I = m_rod.I(j);

      E += 1.0 / (4.0 * len) * (w - w0).dot(B * (w - w0));
    }
  }
```

论文公式7.几
$$
\frac{\partial^2}{(\partial \theta^j)^2}W_i = \frac{1}{\overline l_i}(\bold \omega_i^j)^TJ^T\overline B^jJ(\omega_i^j) - \frac{1}{\overline l_i}(\omega_i^j)^T\overline B^j(\omega_i^j - \overline \omega_i^j)
$$

```
  // \frac{ \partial^2 E }{ \partial^2\theta^j }
  Mat2d JBJt =  J * B * Jt;
  for (int j = idx - 1; j <= idx; ++j) {
    int j2 = j + 1 - idx;
    const Vec2d& omega = getOmega(idx, j);
    const Vec2d& omegaBar = getOmegaBar(idx, j);
    jac(3 + 4 * j2, 3 + 4 * j2) =
      -0.5 / len * (omega.dot(JBJt * omega)
                    - (omega - omegaBar).dot(B * omega));
  }
  jac(3,7) = jac(7,3) = 0;
```

$$
\frac{\partial E}{\partial \bold x_i} = \sum_{k=1}^n \frac{1}{\overline l_k}\sum_{j=k-1}^k(\nabla_i \omega_k^j)^T\overline B^j(\omega_k^j - \overline \omega_k^j)
$$

不敢确定有一个

```
inline void RodBendingForceSym::localJacobian(MatXd& localJ,
                                              const vertex_handle& vh)
{
  Mat2d B = getB(vh);
  Scalar len = getRefVertexLength(vh);

  const Vec2d& kappa = getKappa(vh);
  const Vec2d& kappaBar = getKappaBar(vh);
  const MatXd& gradKappa = getGradKappa(vh);

  localJ = -1.0 / len * gradKappa * B * gradKappa.transpose();

  const pair<MatXd, MatXd>& hessKappa = getHessKappa(vh);
  Vec2d temp = -1.0 / len * (kappa - kappaBar).transpose() * B;
  localJ += temp(0) * hessKappa.first + temp(1) * hessKappa.second;
}
```

ParallelTransport

ElasticRods版本

```
        Vec3_T<Real_> sinThetaAxis = t0.cross(t1);
    Real_ cosTheta = t0.dot(t1);
    Real_ den = 1 + cosTheta;
    return (sinThetaAxis.dot(v) / (1 + cosTheta)) * sinThetaAxis
        + sinThetaAxis.cross(v)
        + cosTheta * v;
```

APrime版本

```
function d = parallel_transport(u, t1, t2)
b = cross(t1, t2);
if (norm(b) == 0 ) 
    d = u;
else
    b = b / norm(b);
    b = b - dot(b,t1) * t1;
    b = b / norm(b);
    b = b - dot(b,t2) * t2;
    b = b / norm(b);
    n1 = cross(t1, b);
    n2 = cross(t2, b);
    d = dot(u,t1) * t2 + dot(u, n1) * n2 + dot(u, b) * b;
end
end
```

Discrete版本

```
void ElasticRod::parallelTransportFrame(const mg::Vec3D& e0, const mg::Vec3D& e1,
                                        mg::Vec3D& io_u) const
{
	mg::Vec3D axis = 2 * mg::cross(e0, e1) /
	        (e0.length() * e1.length() + mg::dot(e0, e1));

//    here sinPhi and cosPhi are derived from the length of axis ||axis|| = 2 * tan( phi/2 )
	double sinPhi, cosPhi;
	double magnitude = mg::dot(axis, axis);
	extractSinAndCos(magnitude, sinPhi, cosPhi);
	assert( cosPhi >= 0 && cosPhi <= 1 );

	if ( (1 - cosPhi) < mg::ERR )
	{
		io_u = mg::cross(e1, io_u);
		io_u = mg::normalize( mg::cross(io_u, e1) );
		return;
	}
	mg::Quaternion q(cosPhi, sinPhi * mg::normalize(axis));
	mg::Quaternion p(0, io_u);
	p = q * p * mg::conjugate(q);

	io_u.set(p[1], p[2], p[3]);
	io_u.normalize();
}
```

$$
\Chi
$$

