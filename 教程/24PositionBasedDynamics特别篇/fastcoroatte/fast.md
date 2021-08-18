极分解如下polar decomposition
$$
F = RS
$$
RS都是对称，半正定矩阵。但是当矩阵包含reflection上面的公式就很不靠谱了。因此可以使用SVD分解，然后计算
$$
\bold R = \bold U \bold V^T
$$
SVD分解不是唯一迭代，

 a stan

dard convention is to choose the singular values to be positive and

to arrange them in descending order.

**Invertible Finite Elements For Robust Simulation of Large**

**Deformation**

注意GreenStrain
$$
\bold G = \frac{1}{2}(\bold F^T \bold F - \bold I)
$$
可以基于此计算stress 和 forces。但是，对于所有的正交变换来说，G是不变的，那么元素inversion就不可能探测到。并且G在形变中是非线性的，and it is

therefore more diffificult to interpret the large deformation behavior of a constitutive model based on **G** than one based on **F**, which is linearly related to deformation. T
$$

$$


```
        iden = np.zeros((3,3))
        iden[0,0] = iden[1,1] = iden[2,2] = 1
        strain = (Ftilde + np.transpose(Ftilde)) / 2 - iden
        tr = strain[0,0] + strain[1,1] + strain[2,2]
        stress = lam * tr * iden + 2 * mu * strain
        
        Xdot = np.zeros((3,3))
        Xdot[0,:] = v1 - v0
        Xdot[1,:] = v2 - v0
        Xdot[2,:] = v3 - v0
        
        Fdot = np.dot(basis[ie,:,:],Xdot)
        Fdottilde = np.dot(Fdot,np.transpose(Q))
```

first Piola-Kirchhoff stress P的一种解释是，将根据区域计算权重的法向量，投射到世界空间。

假设我们知道了Cauchy Stress sgima 或者 第二 Pk力S，我们也可以计算出权重
$$
\bold P= \bold F \bold S \qquad \bold P = det(\bold F)\sigma \bold F^{-T}
$$

```
	// P(F) = F(2 mu E + lambda tr(E)I) => E = green strain
	const Real trace = epsilon(0, 0) + epsilon(1, 1) + epsilon(2, 2);
	const Real ltrace = lambda*trace;
	sigma = epsilon * 2.0*mu;
	sigma(0, 0) += ltrace;
	sigma(1, 1) += ltrace;
	sigma(2, 2) += ltrace;
	sigma = F * sigma;

	Real psi = 0.0;
	for (unsigned char j = 0; j < 3; j++)
		for (unsigned char k = 0; k < 3; k++)
			psi += epsilon(j, k) * epsilon(j, k);
	psi = mu*psi + static_cast<Real>(0.5)*lambda * trace*trace;
	energy = restVolume * psi;
```

由于旋转并并不会导致形变，所以
$$
\bold P = \bold P(\bold U \bold F) \qquad \bold U \bold P(\bold F)
$$
Hessian 矩阵

Eigen Space of Mesh Distortion Energy Hessian
$$
\bold F = \bold U \Sigma \bold V^T
$$

```

```



U和V是旋转矩阵，所以可以使用parameterize distortion kernel。看着不像啊，sigma应该是指Cauchy Stress
$$
\psi(\bold x) := \psi(\sigma_0(\bold x),\sigma_1(\bold x),\sigma_2(\bold x))
$$
Hessian矩阵
$$
\bold H = \begin{bmatrix} \frac{\partial^2 \psi}{\partial \bold x \partial  \bold x} \end{bmatrix}
$$
