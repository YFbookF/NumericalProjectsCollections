```
	void ImplicitNewmark::update_quantities(const Eigen::VectorXd &x)
	{
		Eigen::VectorXd tmp = x_prev + dt() * (v_prev + dt() * (0.5 - beta) * a_prev);
		v_prev += dt() * (1 - gamma) * a_prev;	 // vᵗ + h(1 - γ)aᵗ
		a_prev = (x - tmp) / (beta * dt() * dt()); // aᵗ⁺¹ = ...
		v_prev += dt() * gamma * a_prev;		   // hγaᵗ⁺¹
		x_prev = x;
	}

```

但我觉得这是骗人的，这明明是显式的~

下面的deformation和strain和stress的关系式终于看懂了

```
const Eigen::MatrixXd strain = (displacement_grad + displacement_grad.transpose()) / 2;
			const Eigen::MatrixXd stress = 2 * mu * strain + lambda * strain.trace() * Eigen::MatrixXd::Identity(size(), size());
```

下面能量这种关系是没见过

```
			const T val = mu * (strain.transpose() * strain).trace() + lambda / 2 * strain.trace() * strain.trace();

			energy += val * da(p);
```

velocity gradient
$$
\bold L = \dot{\bold F}\bold F^{-1}
$$
strain rate tensor given as the symmetric part of L
$$
\dot{e} = \frac{1}{2}[\bold L + \bold L^T]
$$
krichhoff stress
$$
\tau = (det \bold F)\sigma
$$
以及

```
# energy density w 2 def EnergyDensity (F,mu , lmbda ):
3 a = 0.5* mu -0.125* lmbda
4 b = 0.125* lmbda
5 c = 0.125* lmbda
6 d = 0.5* lmbda +mu
7 e = 1.5* mu +0.125* lmbda
8 return a*tr(F.T*F) +b*tr( cof (F.T*F)) \
9 +c* det(F) **2 -d*ln( det (F) ) -e
```

Numerical Aspects in Optimal Control

of Elasticity Models with Large Deformations

```
energy += (stress_tensor * strain).trace() * da(p);
```

https://engcourses-uofa.ca/books/introduction-to-solid-mechanics/displacement-and-strain/the-deformation-and-the-displacement-gradients/

超级棒

NeoHookeanElasticity.cpp

```
const T log_det_j = log(polyfem::determinant(def_grad));
			const T val = mu / 2 * ((def_grad.transpose() * def_grad).trace() - size() - 2 * log_det_j) + lambda / 2 * log_det_j * log_det_j;

			energy += val * da(p);
```

按照https://en.wikipedia.org/wiki/Neo-Hookean_solid 上的解释，可压缩NH的模型的能量密度为
$$
W = C_1(I_1 - 3 - 2\ln J) + D_1(J-1)^2
$$
对于线弹性来说，可以设为
$$
C_1 = \frac{\mu}{2} \qquad D_1 = \frac{\kappa}{2}
$$
stress，看不懂，找不到InJ.

```
//stress = mu (F - F^{-T}) + lambda ln J F^{-T}
			//stress = mu * (def_grad - def_grad^{-T}) + lambda ln (det def_grad) def_grad^{-T}
			Eigen::MatrixXd stress_tensor = mu * (def_grad - FmT) + lambda * std::log(def_grad.determinant()) * FmT;

			//stess = (mu displacement_grad + lambda ln(J) I)/J
			// Eigen::MatrixXd stress_tensor = (mu_/J) * displacement_grad + (lambda_/J) * std::log(J)  * Eigen::MatrixXd::Identity(size(), size());

			all.row(p) = fun(stress_tensor);
```

VON-MISES

```
double von_mises_stress_for_stress_tensor(const Eigen::MatrixXd &stress)
	{
		double von_mises_stress;

		if (stress.rows() == 3)
		{
			von_mises_stress = 0.5 * (stress(0, 0) - stress(1, 1)) * (stress(0, 0) - stress(1, 1)) + 3.0 * stress(0, 1) * stress(1, 0);
			von_mises_stress += 0.5 * (stress(2, 2) - stress(1, 1)) * (stress(2, 2) - stress(1, 1)) + 3.0 * stress(2, 1) * stress(2, 1);
			von_mises_stress += 0.5 * (stress(2, 2) - stress(0, 0)) * (stress(2, 2) - stress(0, 0)) + 3.0 * stress(2, 0) * stress(2, 0);
		}
		else
		{
			// von_mises_stress = ( stress(0, 0) - stress(1, 1) ) * ( stress(0, 0) - stress(1, 1) ) + 3.0  *  stress(0, 1) * stress(1, 0);
			von_mises_stress = stress(0, 0) * stress(0, 0) - stress(0, 0) * stress(1, 1) + stress(1, 1) * stress(1, 1) + 3.0 * stress(0, 1) * stress(1, 0);
		}

		von_mises_stress = sqrt(fabs(von_mises_stress));

		return von_mises_stress;
	}
```

