https://github.com/davvm/clothSim

首先计算ShearEnergy。x1x2是世界坐标系，而uv是本地坐标系。w则将坐标系映射到世界坐标系。


$$
\Delta \bold x_1 = \bold w_u\Delta u_1 + \bold w_v \Delta v_1 \\
\Delta \bold x_2 = \bold w_u\Delta u_2 + \bold w_v \Delta v_2
$$
那么列方程
$$
\begin{bmatrix} \Delta u_1 & \Delta u_2 \\ \Delta v_1 & \Delta v_2 \end{bmatrix} \begin{bmatrix} \bold w_u \\ \bold w_v\end{bmatrix} = \begin{bmatrix} \Delta \bold x_1 \\ \Delta \bold x_2\end{bmatrix}
$$
我还是习惯左乘一点。反正就是求上面的wu和wv

```
Vector2 duv1 = uv1 - uv0; 
Vector2 duv2 = uv2 - uv0; 

Real du1 = duv1[0]; 
Real dv1 = duv1[1];
Real du2 = duv2[0];
Real dv2 = duv2[1];

// triangle area in reference pose:
a = Real(0.5) * (du1 * dv2 - du2 * dv1);
// trangle tangents in reference directions:
wu = ((p1 - p0) * dv2 - (p2 - p0) * dv1) / (2 * a);
wv = (-(p1 - p0) * du2 + (p2 - p0) * du1) / (2 * a);
```

然后w还要分别对三个空间坐标中的点求导，比如
$$
\frac{\partial \bold w_u}{\partial \bold x_1}
$$

```
	// first derivatives of uv tangents:
		dwudP0Scalar = (dv1 - dv2) / (2 * a);
		dwudP1Scalar = dv2 / (2 * a);
		dwudP2Scalar = -dv1 / (2 * a);

		dwvdP0Scalar = (du2 - du1) / (2 * a);
		dwvdP1Scalar = -du2 / (2 * a);
		dwvdP2Scalar = du1 / (2 * a);
```

ShearCondition.cpp

情况函数C是一个标量哦、
$$
C(\bold x) = \alpha \bold w_u(\bold x)^T \bold w_v(\bold x)
$$

```
Real wuNorm = wu.norm();
	Real wvNorm = wv.norm();

	wuHat = wu / wuNorm;
	wvHat = wv / wvNorm;

	// energy condition:
	C = a * wu.dot(wv);
```

继续弄它的导数，同样是对世界坐标中三个点求导，三个求导结果都是3x1向量
$$
\frac{\partial C(\bold x)}{\partial \bold x_i} = \alpha(\frac{\partial \bold w_u}{\partial \bold x_i} \bold w_v + \frac{\partial \bold w_v}{\partial \bold x_i} \bold w_u)
$$

```
	dCdP0 = a * (dwudP0Scalar * wv + dwvdP0Scalar * wu);
	dCdP1 = a * (dwudP1Scalar * wv + dwvdP1Scalar * wu);
	dCdP2 = a * (dwudP2Scalar * wv + dwvdP2Scalar * wu);
```

然后是二阶导，有九个求导结果，都是3x3矩阵。

```
	d2CdP0dP0 = 2 * a * dwudP0Scalar * dwvdP0Scalar * Matrix3::Identity();
	d2CdP0dP1 = a * (dwudP0Scalar * dwvdP1Scalar + dwvdP0Scalar * dwudP1Scalar) * Matrix3::Identity();
	d2CdP0dP2 = a * (dwudP0Scalar * dwvdP2Scalar + dwvdP0Scalar * dwudP2Scalar) * Matrix3::Identity();

	d2CdP1dP0 = a * (dwudP1Scalar * dwvdP0Scalar + dwvdP1Scalar * dwudP0Scalar) * Matrix3::Identity();
	d2CdP1dP1 = 2 * a * dwvdP1Scalar * dwudP1Scalar * Matrix3::Identity();
	d2CdP1dP2 = a * (dwudP1Scalar * dwvdP2Scalar + dwvdP1Scalar * dwudP2Scalar) * Matrix3::Identity();

	d2CdP2dP0 = a * (dwudP2Scalar * dwvdP0Scalar + dwvdP2Scalar * dwudP0Scalar) * Matrix3::Identity();
	d2CdP2dP1 = a * (dwudP2Scalar * dwvdP1Scalar + dwvdP2Scalar * dwudP1Scalar) * Matrix3::Identity();
	d2CdP2dP2 = 2 * a * dwvdP2Scalar * dwudP2Scalar * Matrix3::Identity();
```

C对时间求导，不知道咋来的

```
dCdt = dCdP0.dot(v0) + dCdP1.dot(v1) + dCdP2.dot(v2);
```

### ShearCondition迭代开始

首先算能量函数
$$
\bold E(\bold x) = \frac{k}{2}\bold C(\bold x)^T\bold C(\bold x)
$$
然后算力，f是个3x3矩阵
$$
\bold f_i = -\frac{\partial E}{\partial \bold x_i} = -k\frac{\partial \bold C(\bold x)}{\partial \bold x_i}\bold C(\bold x)
$$

```
forces.segment<3>(3 * m_inds[0]) -= k * q.C * q.dCdP0;
forces.segment<3>(3 * m_inds[1]) -= k * q.C * q.dCdP1;
forces.segment<3>(3 * m_inds[2]) -= k * q.C * q.dCdP2;
```

然后为组装矩阵做准备，算力的导数dfdx

```
	// compute force derivatives and insert them into the sparse matrix:
	Matrix3 df0dP0 = -k * (q.dCdP0 * q.dCdP0.transpose() + q.C * q.d2CdP0dP0);
	Matrix3 df0dP1 = -k * (q.dCdP0 * q.dCdP1.transpose() + q.C * q.d2CdP0dP1);
	Matrix3 df0dP2 = -k * (q.dCdP0 * q.dCdP2.transpose() + q.C * q.d2CdP0dP2);

	Matrix3 df1dP0 = -k * (q.dCdP1 * q.dCdP0.transpose() + q.C * q.d2CdP1dP0);
	Matrix3 df1dP1 = -k * (q.dCdP1 * q.dCdP1.transpose() + q.C * q.d2CdP1dP1);
	Matrix3 df1dP2 = -k * (q.dCdP1 * q.dCdP2.transpose() + q.C * q.d2CdP1dP2);

	Matrix3 df2dP0 = -k * (q.dCdP2 * q.dCdP0.transpose() + q.C * q.d2CdP2dP0);
	Matrix3 df2dP1 = -k * (q.dCdP2 * q.dCdP1.transpose() + q.C * q.d2CdP2dP1);
	Matrix3 df2dP2 = -k * (q.dCdP2 * q.dCdP2.transpose() + q.C * q.d2CdP2dP2);
```

### 然后算damping

$$
\bold d = -k_d \frac{\partial \bold C(\bold x)}{\partial \bold x}\dot{\bold C}(\bold x)
$$

```
	dampingForces.segment<3>(3 * m_inds[0]) -= d * q.dCdt * q.dCdP0;
	dampingForces.segment<3>(3 * m_inds[1]) -= d * q.dCdt * q.dCdP1;
	dampingForces.segment<3>(3 * m_inds[2]) -= d * q.dCdt * q.dCdP2;
```

它的导为
$$
\frac{\partial \bold d_i}{\partial \bold x_j} = -k_d(\frac{\partial \bold C(\bold x)}{\partial \bold x_i}\frac{\partial \dot{\bold C}(x)}{\partial \bold x_j} + \frac{\partial^2 \bold C(\bold x)}{\partial \bold x_i \partial \bold x_j} \dot{\bold C}(x))
$$

```
	Matrix3 dfd0dV0 = -d * (q.dCdP0 * q.dCdP0.transpose());
	Matrix3 dfd0dV1 = -d * (q.dCdP0 * q.dCdP1.transpose());
	Matrix3 dfd0dV2 = -d * (q.dCdP0 * q.dCdP2.transpose());

	Matrix3 dfd1dV0 = -d * (q.dCdP1 * q.dCdP0.transpose());
	Matrix3 dfd1dV1 = -d * (q.dCdP1 * q.dCdP1.transpose());
	Matrix3 dfd1dV2 = -d * (q.dCdP1 * q.dCdP2.transpose());

	Matrix3 dfd2dV0 = -d * (q.dCdP2 * q.dCdP0.transpose());
	Matrix3 dfd2dV1 = -d * (q.dCdP2 * q.dCdP1.transpose());
	Matrix3 dfd2dV2 = -d * (q.dCdP2 * q.dCdP2.transpose());
	
	Matrix3 dD0dP0Pseudo = -d * (q.d2CdP0dP0 * q.dCdt);
	Matrix3 dD1dP0Pseudo = -d * (q.d2CdP1dP0 * q.dCdt);
	Matrix3 dD2dP0Pseudo = -d * (q.d2CdP2dP0 * q.dCdt);

	Matrix3 dD0dP1Pseudo = -d * (q.d2CdP0dP1 * q.dCdt);
	Matrix3 dD1dP1Pseudo = -d * (q.d2CdP1dP1 * q.dCdt);
	Matrix3 dD2dP1Pseudo = -d * (q.d2CdP2dP1 * q.dCdt);

	Matrix3 dD0dP2Pseudo = -d * (q.d2CdP0dP2 * q.dCdt);
	Matrix3 dD1dP2Pseudo = -d * (q.d2CdP1dP2 * q.dCdt);
	Matrix3 dD2dP2Pseudo = -d * (q.d2CdP2dP2 * q.dCdt);
```

### Stretch

它的Condition有俩
$$
\bold C(\bold x) = \alpha \begin{bmatrix} ||\bold w_u(\bold x)|| - b_u \\ ||\bold w_v(\bold x)|| - b_v \end{bmatrix}
$$

```
	// first derivatives of condition quantities:
	Real wuNorm = wu.norm();
	dC0dP0 = a * dwudP0 * wu / wuNorm;
	dC0dP1 = a * dwudP1 * wu / wuNorm;
	dC0dP2 = a * dwudP2 * wu / wuNorm;

	Real wvNorm = wv.norm();
	dC1dP0 = a * dwvdP0 * wv / wvNorm;
	dC1dP1 = a * dwvdP1 * wv / wvNorm;
	dC1dP2 = a * dwvdP2 * wv / wvNorm;

	// condition quantities:
	C0 = a * (wuNorm - bu);
	C1 = a * (wvNorm - bv);
```

原文里有句话"Given a condition C(x) which we want to be zero"，这种语句经常在各种计算力学和图形学资料中出现。为了达到这一目标，我们所用的方法一般都与牛顿迭代法很类似，每帧更新时迭代一次。再回顾一下牛顿迭代
$$
\bold x^{k+1} = \bold x^{k} - \frac{\bold F(\bold x)}{\bold F'(\bold x)}
$$
不过在这里，等式右边第二项不再是C(x)/C'(x)，而是
$$
h(\bold v + h(\bold M^{-1}(\bold f_0 + \frac{\partial \bold f_0}{\partial \bold x}\Delta \bold x + \frac{\partial \bold f}{\partial \bold v}\Delta \bold v))) 
$$
当C(x)为零，物体已经恢复原状，没有一点剪切或拉伸力了了。
$$
\bold f_i = -\frac{\partial E}{\partial \bold x_i} = -k\frac{\partial \bold C(\bold x)}{\partial \bold x_i}\bold C(\bold x)
$$

### 手算案例

接下来用一个简单的例子来确保结果是正确的。比如一个三角形，它三个顶点的uv坐标是及uv差是
$$
uv_1 = \begin{bmatrix} 0 \\ 0\end{bmatrix} \qquad uv_2 = \begin{bmatrix} 1\\0\end{bmatrix}\qquad uv_3 = \begin{bmatrix} 0\\1\end{bmatrix} \qquad \Delta uv_{21} = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \qquad \Delta uv_{31} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$
它的世界坐标为及世界坐标差是
$$
\bold x_1 = \begin{bmatrix} 0\\0\\1\end{bmatrix} \qquad \bold x_2 = \begin{bmatrix} 1\\1\\1\end{bmatrix}\qquad \bold x_2 = \begin{bmatrix} 0\\2\\1\end{bmatrix}  \qquad \Delta x_{21} =\begin{bmatrix} 1\\1\\0\end{bmatrix} \qquad \Delta x_{31} = \begin{bmatrix} 0\\2\\0\end{bmatrix}
$$
解方程得到
$$
\bold w_u = \begin{bmatrix} 1\\1\\0\end{bmatrix} \qquad \bold w_v = \begin{bmatrix} 0\\2\\0\end{bmatrix}
$$
那么对于剪切的情况来说，不为零。
$$
C(\bold x) = \alpha \bold w_u(\bold x)^T \bold w_v(\bold x) = 2\alpha
$$
假设更换初始的duv，让，那么剪切情况就变为零了。
$$
\qquad \Delta uv_{21} = \begin{bmatrix} 1 \\ 0\end{bmatrix} \qquad \Delta uv_{31} = \begin{bmatrix} 1 \\ 1 \end{bmatrix} \qquad \bold w_u = \begin{bmatrix} 1\\-1\\0\end{bmatrix} \qquad \bold w_v = \begin{bmatrix} 0\\2\\0\end{bmatrix}
$$
拉伸的bu和bv是这样的，一般是1。如果想类似garment那样稍微拉伸一点，那么就设得稍微大一点。如果想想sleeve那样稍微小一点，就设的稍微小一点。那现在设为1。按照之前
$$
||\bold w_u|| = \sqrt{2} \qquad ||\bold w_v|| = 2
$$


### 