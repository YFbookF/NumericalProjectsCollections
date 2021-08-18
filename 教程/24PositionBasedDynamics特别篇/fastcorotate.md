这个库真的棒

volume constraint
$$
C_t(\bold x) = det(\bold F_t) - 1 = \frac{[(\bold x_1 - \bold x_0) \times (\bold x_2 -\bold x_0)](\bold x_3 - \bold x_0)}{6V_t} - 1
$$

```
Vector3f8 d1 = p[1] - p[0];
			Vector3f8 d2 = p[2] - p[0];
			Vector3f8 d3 = p[3] - p[0];
			Scalarf8 volume = (d1 % d2) * d3 * (1.0f / 6.0f);
```



Larange Multipliers
$$
\Delta \kappa_t = \frac{C_t - \alpha_t \kappa_t^n}{\sum_{i=0}^3 \frac{1}{m_i}||\nabla_{\bold x_i} C_I||^2 + \alpha_t}
$$

```
Scalarf8 delta_kappa =
				inv_mass_phases[phase][constraint][0] * grad0.lengthSquared() +
				inv_mass_phases[phase][constraint][1] * grad1.lengthSquared() +
				inv_mass_phases[phase][constraint][2] * grad2.lengthSquared() +
				inv_mass_phases[phase][constraint][3] * grad3.lengthSquared() +
				alpha;
			
			delta_kappa = (restVol - volume - alpha * kappa) / blend(abs(delta_kappa) < eps, 1.0f, delta_kappa);
			kappa = kappa + delta_kappa;
```

