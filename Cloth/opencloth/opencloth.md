stress-strain relations
$$
\begin{bmatrix} \varepsilon_{x} \\ \varepsilon_{y} \\ \varepsilon_{z} \\ 2\varepsilon_{yz} \\ 2\varepsilon_{xz} \\ 2\varepsilon_{xy}\end{bmatrix} = \frac{1}{E}\begin{bmatrix} 1 & -\nu & -\nu & 0 & 0 & 0 \\ -\nu & 1 & -\nu & 0 & 0 & 0  \\-\nu & -\nu & 1 & 0 &0  & 0  \\ 0 & 0 & 0 & 2(1+\nu) & 0 & 0 \\ 0 & 0& 0 & 0 & 2(1+\nu) & 0  \\0 & 0 & 0& 0 & 0 & 2(1+\nu) \end{bmatrix}\begin{bmatrix} \sigma_{x} \\ \sigma_{y} \\\sigma_{z} \\ \tau_{yz} \\ \tau_{xz} \\ \tau_{xy}\end{bmatrix} 
$$

```
void ComputeStrainAndStress()
{
	for (int i = 0; i < total_points; i++) {
		glm::mat3 Jtr = glm::transpose(J[i]);
		epsilon[i] = (Jtr * J[i]) - I;		// formula 3, Green-Saint Venant non-linear tensor

		glm::mat3& e = epsilon[i];
		glm::mat3& s = sigma[i];

		s[0][0] = D.x * e[0][0] + D.y * e[1][1] + D.y * e[2][2];
		s[1][1] = D.y * e[0][0] + D.x * e[1][1] + D.y * e[2][2];
		s[2][2] = D.y * e[0][0] + D.y * e[1][1] + D.x * e[2][2];

		s[0][1] = D.z * e[0][1];
		s[1][2] = D.z * e[1][2];
		s[2][0] = D.z * e[2][0];

		s[0][2] = s[2][0];
		s[1][0] = s[0][1];
		s[2][1] = s[1][2];
	}
}
```

https://www.sciencedirect.com/topics/engineering/stress-strain-relations
$$
\begin{bmatrix} \sigma_{x} \\ \sigma_{y} \\\sigma_{z} \\ \tau_{yz} \\ \tau_{xz} \\ \tau_{xy}\end{bmatrix} =   \frac{E}{(1+\nu)(1-2\nu)} 
\begin{bmatrix} 1 - \nu & \nu & \nu & 0 & 0 & 0 \\ \nu & 1-\nu & \nu & 0 & 0 & 0  \\-\nu & \nu & 1- \nu & 0 &0  & 0  \\ 0 & 0 & 0 & (1-2\nu)/2 & 0 & 0 \\ 0 & 0& 0 & 0 & (1-2\nu)/2 & 0  \\0 & 0 & 0& 0 & 0 & (1-2\nu)/2  \end{bmatrix}\begin{bmatrix} \varepsilon_{x} \\ \varepsilon_{y} \\ \varepsilon_{z} \\ 2\varepsilon_{yz} \\ 2\varepsilon_{xz} \\ 2\varepsilon_{xy}\end{bmatrix}
$$
https://cn.comsol.com/multiphysics/analysis-of-deformation

只不过这玩意用的是前者

```
void ComputeJacobians()
{
	for (int i = 0; i < total_points; i++) {
		vector<neighbor>& pNeighbor = neighbors[i];
		for (size_t j = 0; j < pNeighbor.size(); j++)
			U[pNeighbor[j].j] = X[pNeighbor[j].j] - Xi[pNeighbor[j].j];

		glm::mat3 B = glm::mat3(0);		// help matrix used to compute the sum in Eq. 15

		// reset du and du_tr
		glm::mat3 du = glm::mat3(0);
		glm::mat3 du_tr = glm::mat3(0);

		for (size_t j = 0; j < pNeighbor.size(); j++)
		{
			glm::mat3 Bj = glm::mat3(0);
			//Eq. 15 right hand side terms with A_inv
			Bj = glm::outerProduct(U[pNeighbor[j].j] - U[i], pNeighbor[j].rdist * pNeighbor[j].w);
			B += Bj;
		}
		B = glm::transpose(B);

		du = Minv[i] * B;	// Eq. 15 page 4
		du_tr = glm::transpose(du);
		J[i] = glm::mat3(1);
		J[i] += du_tr;		// Eq. 1
	}
}
```

记住，我们的核心的任务，就是通过施加在物体的力，计算出每个点的位移。

如果我们想知道每个点的位移，就得知道每个块的应变