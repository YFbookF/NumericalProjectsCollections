突然想起来这玩意是纯LEVELSET啊，牛啊

我们希望最小化thin-plate energy能量
$$
E = \frac{1}{2}\int_{\Omega}(\phi_{xx}^2 + \phi_{yy}^2 + \phi_{zz}^2 + 2\phi_{xy}^2 + 2\phi_{yz}^2 + 2\phi_{zx}^2)dxdydz
$$
变分导数成为下面这个东西
$$
\phi_t = -\nabla^4\phi||\nabla \phi|| = \\
-(\phi_{xxxx} + \phi_{yyyy} + \phi_{zzzz} + 2\phi_{xxyy} + 2\phi_{yyzz} + 2\phi_{zzxx})||\nabla \phi||
$$
积分一下就是
$$
\phi^{t+\Delta t} = \phi^t - \Delta t \nabla ^4 \phi||\nabla \phi||
$$
很可惜这是个水平集，后面绝对值就是1，上式稍微变换一下就是
$$
(\bold I + \Delta t \nabla^4)\phi^{t + \Delta t} = \phi^t
$$
怎么解，当然是Laplacian加biharmonic。不过我本人认为这个应该叫做Poisson。

```
bool SmoothingGrid::computeLaplacianCG(const SlArray3D<double>& x, SlArray3D<unsigned char>& marked) {
	for (int i = 1; i < nx - 1; i++) {
		for (int j = 1; j < ny - 1; j++) {
			for (int k = 1; k < nz - 1; k++) {
				if (marked(i, j, k)) {
					laplacian(i, j, k) = (x(i + 1, j, k) + x(i - 1, j, k) + x(i, j + 1, k)
						+ x(i, j - 1, k) + x(i, j, k + 1) + x(i, j, k - 1)
						- 6 * x(i, j, k));
				}
			}
		}
	}
	return true;
}

bool SmoothingGrid::applyBiharmonic(const SlArray3D<double>& x, SlArray3D<double>& y, SlArray3D<unsigned char>& markedB, double dt) const {
	double factor = dt / sqr(h) * sqr(h);
	for (int i = 2; i < nx - 2; i++) {
		for (int j = 2; j < ny - 2; j++) {
			for (int k = 2; k < nz - 2; k++) {
				if (markedB(i, j, k)) {
				// 注意下面这个x，这玩意就是1
					y(i, j, k) = x(i, j, k) + factor * (laplacian(i + 1, j, k) + laplacian(i - 1, j, k) + laplacian(i, j + 1, k)
						+ laplacian(i, j - 1, k) + laplacian(i, j, k + 1) + laplacian(i, j, k - 1)
						- 6 * laplacian(i, j, k));
				}
			}
		}
	}

	return true;
}

	computeLaplacianCG(x, markedL);
	applyBiharmonic(x, q, markedB, dt);
```

