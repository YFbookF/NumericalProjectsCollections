# 主

拉格朗日乘子
$$
\lambda = -\frac{\bold C(p)}{|\nabla_p \bold C(p)|^2}
$$
位移向量
$$
\Delta \bold p = \lambda \nabla_p \bold C(p) = -\frac{\bold C(p)}{|\nabla \bold C(p)|^2}\nabla \bold C_p(p)
$$


# 距离约束

$$
\bold C(\bold x_0,\bold x_1) = |\bold x_0 - \bold x_1| - \bold d
$$

```
	computeMatrixK(c0, invMass0, x0, inertiaInverseW0, K1);
	computeMatrixK(c1, invMass1, x1, inertiaInverseW1, K2);
```

$$
J_0 = \begin{bmatrix}1 & 0  \end{bmatrix} \qquad J_1 = \begin{bmatrix}-1 & 0  \end{bmatrix} \qquad dir = \begin{bmatrix}-1 & 0  \end{bmatrix}
$$

K = 2,  C = 

```
const Real C = (length - restLength);
```

col0 就是那个摆臂的向量

col2 就是刚体上副点的位置

```
	return PositionBasedRigidBodyDynamics::update_DistanceJoint(
		rb1.getPosition(),
		rb1.getRotation(),
		rb2.getPosition(),
		rb2.getRotation(),
		m_jointInfo);
```

J 就是壁

position based dynamics

```
	jointInfo.col(0) = rot0T * (pos0 - x0);
	jointInfo.col(1) = rot1T * (pos1 - x1);
	jointInfo.col(2) = pos0;
	jointInfo.col(3) = pos1;
```

别弄那么难，首先是两个线段，每个线段都有一个主点，坐标为世界坐标，然后一个副点，坐标为相对于主点的初始本地坐标。然后还有一个旋转矩阵，代表副点到主点的轴的旋转角度。

比如
$$
\bold p_{00} = \begin{bmatrix} 0 \\ 0\end{bmatrix} \qquad \bold p_{01} = \begin{bmatrix} 1 \\ 0\end{bmatrix} \qquad \bold R_{0}  = \begin{bmatrix} 1 & 0 \\ 0 & 1\end{bmatrix}
$$
注意p00是世界坐标，p01是副点初始情况下相对于主点的本地坐标。注意旋转矩阵为正交矩阵，所以旋转矩阵的逆等于转置。任意时刻，副点的世界坐标
$$
p_{01world} = \bold p _{00world} + \bold R_0(\bold p_{01local} - \bold p_{00local})
$$
那么第一行和第二行，其实就是p00的世界坐标减去p01的世界坐标的那个向量了，

pos0 和 pos1 可能是副点

距离约束

E:\mycode\physicsEngine\JoltPhysics-master\Jolt\Physics\Constraints\DistanceConstraint.cpp

```
float distance = (mWorldSpacePosition2 - mWorldSpacePosition1).Dot(mWorldSpaceNormal);
```

还有一处我看不懂

```
// Update world space positions (the bodies may have moved)
	mWorldSpacePosition1 = mBody1->GetCenterOfMassTransform() * mLocalSpacePosition1;
	mWorldSpacePosition2 = mBody2->GetCenterOfMassTransform() * mLocalSpacePosition2;

	// Calculate world space normal
	Vec3 delta = mWorldSpacePosition2 - mWorldSpacePosition1;
	float delta_len = delta.Length();
	if (delta_len > 0.0f)
		mWorldSpaceNormal = delta / delta_len;

	// Calculate points relative to body
	// r1 + u = (p1 - x1) + (p2 - p1) = p2 - x1
	Vec3 r1_plus_u = mWorldSpacePosition2 - mBody1->GetCenterOfMassPosition();
	Vec3 r2 = mWorldSpacePosition2 - mBody2->GetCenterOfMassPosition();
```

FEBIO

```

	// get the current position of the two nodes
	vec3d ra = nodea.m_rt; // 
	vec3d rb = nodeb.m_rt;

	// calculate the Lagrange multipler
	double l = (ra - rb).norm();
	double Lm = m_Lm + m_eps*(l - m_l0);

	// calculate force
	vec3d Fc = (ra - rb)*(Lm/l);
	
```

记住此时拉格朗日乘子
$$
\lambda = \frac{|\bold p_1 - \bold p_2| - d}{w_1 + w_2}
$$
![image-20211224202946329](E:\mycode\collection\新教程\image-20211224202946329.png)

E:\mycode\positionBased\rigid-body-simulation-with-extended-position-based-dynamics-master\Core.hpp

```
void SolvePositionConstraints(Config &c) {
  auto h = c.dt / c.numSubSteps;
  auto h2 = h * h;
  for (auto &&b : c.posConstraints) {
    auto &eli = c.bodies[b.i];
    auto &elj = c.bodies[b.j];
    auto [n, c] = b.n_c();
    auto wi = 1 / eli.m + blaze::trans(blaze::cross(b.ri, n)) * eli.Iinv() *
                              blaze::cross(b.ri, n);
    auto wj = 1 / elj.m + blaze::trans(blaze::cross(b.rj, n)) * elj.Iinv() *
                              blaze::cross(b.rj, n);
    auto alphac = b.compliance / h2;
    auto dlambda = (-c - alphac * b.lambda) / (wi + wj + alphac);
    // b.lambda += dlambda;
    auto p = dlambda * n;
    eli.x += p / eli.m;
    elj.x -= p / elj.m;
    eli.q += 0.5 * to4(eli.Iinv() * blaze::cross(b.ri, p)) * eli.q;
    elj.q -= 0.5 * to4(elj.Iinv() * blaze::cross(b.rj, p)) * elj.q;
  }
}
```

E:\mycode\positionBased\xpbd-master\src\xpbd.cpp

```
  void Solve(CApplication& app, float dt){
    GLfloat   inv_mass1         = m_Particle1->GetInvMass();
    GLfloat   inv_mass2         = m_Particle2->GetInvMass();
    GLfloat   sum_mass          = inv_mass1 + inv_mass2;
    if (sum_mass == 0.0f) { return; }
    glm::vec3 p1_minus_p2       = m_Particle1->GetPosition() - m_Particle2->GetPosition();
    GLfloat   distance          = glm::length(p1_minus_p2);
    GLfloat   constraint        = distance - m_RestLength; // Cj(x)
    glm::vec3 correction_vector;
    if (app.m_Mode != eModePBD) { // XPBD
      m_Compliance = MODE_COMPLIANCE[app.m_Mode];
      m_Compliance /= dt * dt;    // a~
      GLfloat dlambda           = (-constraint - m_Compliance * m_Lambda) / (sum_mass + m_Compliance); // eq.18
              correction_vector = dlambda * p1_minus_p2 / (distance + FLT_EPSILON);                    // eq.17
      m_Lambda += dlambda;
    } else {                      // normal PBD
              correction_vector = m_Stiffness * glm::normalize(p1_minus_p2) * -constraint/ sum_mass;   // eq. 1
    }
    m_Particle1->AddPosition(+inv_mass1 * correction_vector);
    m_Particle2->AddPosition(-inv_mass2 * correction_vector);
  }
```

E:\mycode\collection\ANewCollection\ClassicMechanics\Joint\CAE_DistanceJoint.cpp

看不懂，摆烂

```
    Eigen::Vector deltaURel = mTargetURel - uRel;
    deltaURel = deltaURel.dot(direction) * direction;
```

E:\mycode\physicsEngine\PhysX-4.1\physx\source\physxextensions\src\ExtDistanceJoint.cpp

```
	if(distance < EPS_REAL)
		direction = PxVec3(1.0f, 0.0f, 0.0f);

	Px1DConstraint* c = constraints;

	const PxVec3 angular0 = ch.getRa().cross(direction);
	const PxVec3 angular1 = ch.getRb().cross(direction);

	setupContraint(*c, direction, angular0, angular1, data); 
```

E:\mycode\collection\ANewCollection\ClassicMechanics\Joint\qbox_distantConstraint.cpp

```
  // compute gradient at r
  D3vector r12(r1-r2);
  D3vector g1,g2;
  g1 = 2.0 * r12;
  g2 = -g1;
  const double norm2 = g1*g1 + g2*g2;
  assert(norm2>=0.0);

  // if the gradient is too small, do not attempt correction
  if ( norm2 < 1.e-6 ) return true;
  const double proj = v1 * g1 + v2 * g2;
  const double err = fabs(proj)/sqrt(norm2);
```

E:\mycode\collection\ANewCollection\ClassicMechanics\Joint\ema_distantConstraint.cpp

```
		setJacobian(jacobian, p1, q1);

		A = calcA(jacobian);
		B = calcB(jacobian, v);
		C = calcC(p1, q1, L);
		B = B + C;

		for(j=0; j<iter; j++)
		{
			delta_lambda = (float)(B - A*lambda)/A;
			lambda = lambda + delta_lambda;
		}

		calcV2(v2, jacobian, v, lambda);
		p2 = calcX2(p1, v2[0]); 
		q2 = calcX2(q1, v2[1]);

		*post_dst = p2;
		*vd2 = v2[0];
		*post_pos = q2;
		*vp2 = v2[1];
```



# 角速度约束

E:\mycode\positionBased\rigid-body-simulation-with-extended-position-based-dynamics-master\Core.hpp

```
void SolveAngularConstraints(Config &c) {
  auto h = c.dt / c.numSubSteps;
  auto h2 = h * h;
  for (auto &&b : c.angularConstraints) {
    auto &eli = c.bodies[b.i];
    auto &elj = c.bodies[b.j];
    auto [n, t] = b.n_t();
    auto wi = blaze::trans(n) * eli.Iinv() * n;
    auto wj = blaze::trans(n) * elj.Iinv() * n;
    auto alphac = b.compliance / h2;
    auto dlambda = (-t - alphac * b.lambda) / (wi + wj + alphac);
    b.lambda += dlambda;
    auto p = dlambda * n;
    eli.q += 0.5 * to4(eli.Iinv() * p) * eli.q;
    elj.q -= 0.5 * to4(elj.Iinv() * p) * elj.q;
  }
}
```



# 弯曲约束



![image-20211224203249437](E:\mycode\collection\新教程\image-20211224203249437.png)

# 固定约束

cylinder joint 

E:\mycode\positionBased\FEBio-634176cd551cd3518695c688e9aef04d2cf9b8a7\FEBioMech\FERigidCylindricalJoint.cpp



https://github.com/zetan/ConstrainedDynamics

```
void CircleConstrainForce::ApplyForce(Particle* particle){
	float fx = Vector3D::DotProduct(particle->getForce(), particle->getPos());
	float vv = Vector3D::DotProduct(particle->getVelocity(), particle->getVelocity());
	float xx = Vector3D::DotProduct(particle->getPos(), particle->getPos());

	double kdForce = kd * (xx - 1)/ 2;
	double ksForce = ks * Vector3D::DotProduct(particle->getPos(), particle->getVelocity());

	float lamda = -1 * (fx + particle->getMass() * vv + kdForce + ksForce) / xx;
	constrainForce = Vector3D::Scale(particle->getPos(), lamda);
	particle->setForce(Vector3D::Add(particle->getForce(), constrainForce));
}
```

![image-20211224201909435](E:\mycode\collection\新教程\image-20211224201909435.png)