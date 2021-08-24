for hyperelastic materials，应力是应变的导数，由应变的不变量构成。比如第一Piola-Kirchhoff应力，它是应变对形变梯度求导得到的
$$
\bold P = \frac{\partial \psi}{\partial \bold F}
$$
nodal力可以由能量定义，因此可以写出下面的等式
$$
\bold f = \frac{\partial \psi}{\partial x} \qquad \frac{\partial^2 \psi}{\partial x^2} | \Delta x_k = - \frac{\partial \psi}{\partial x}|x_k
$$
因此，全局刚度矩阵-df/dx总是对称的，因为能量能去二阶导。

**Invertible Finite Elements For Robust Simulation of Large**

**Deformation**

**Robust Quasistatic Finite Elements and Flesh Simulation**
$$
\bold A = \begin{bmatrix} \alpha_{11} + \beta_{11} + \gamma_{11} & \gamma_{12} & \gamma_{13} \\ \gamma_{12} & \alpha_{22} + \beta_{22} + \gamma_{22} & \gamma_{23} \\ \gamma_{13} & \gamma_{23} & \alpha_{33} + \beta_{33} + \gamma_{33} \end{bmatrix}\bold B = \begin{bmatrix}\alpha_{ij} & \beta_{ij} \\ \beta_{ij} & \alpha_{ij}\end{bmatrix}
$$
并且
$$
\alpha_{ij} = 2\psi_I + 4(\sigma^2_i + \sigma_j^2)\psi_{II}
$$

```
  double alpha11 = 2.0 * gradient[0] + 8.0 * sigma1square * gradient[1];
  double alpha22 = 2.0 * gradient[0] + 8.0 * sigma2square * gradient[1];
  double alpha33 = 2.0 * gradient[0] + 8.0 * sigma3square * gradient[1];
  double alpha12 = 2.0 * gradient[0] + 4.0 * (sigma1square+sigma2square) * gradient[1];
  double alpha13 = 2.0 * gradient[0] + 4.0 * (sigma1square+sigma3square) * gradient[1];
  double alpha23 = 2.0 * gradient[0] + 4.0 * (sigma2square+sigma3square) * gradient[1];
```

果然是对三个黎曼不变量求导...
$$
\beta_{ij} = 4\sigma_{i}\sigma_j \psi_{II} - \frac{2III\psi_{III}}{\sigma_i\sigma_j}
$$

```
  double beta11 = 4.0 * sigma1square * gradient[1] - (2.0 * invariants[2] * gradient[2]) / sigma1square;
  double beta22 = 4.0 * sigma2square * gradient[1] - (2.0 * invariants[2] * gradient[2]) / sigma2square;
  double beta33 = 4.0 * sigma3square * gradient[1] - (2.0 * invariants[2] * gradient[2]) / sigma3square;
  double beta12 = 4.0 * sigma[0] * sigma[1] * gradient[1] - (2.0 * invariants[2] * gradient[2]) / (sigma[0] * sigma[1]);
  double beta13 = 4.0 * sigma[0] * sigma[2] * gradient[1] - (2.0 * invariants[2] * gradient[2]) / (sigma[0] * sigma[2]);
  double beta23 = 4.0 * sigma[1] * sigma[2] * gradient[1] - (2.0 * invariants[2] * gradient[2]) / (sigma[1] * sigma[2]);
```

$$
\gamma_{ij} = \begin{bmatrix} 2\sigma_i & 4\sigma_i^3 & \frac{2III}{\sigma_i} \end{bmatrix}\frac{\partial^2 \psi}{\partial (I,II,III)^2}\begin{bmatrix} 2\sigma_j \\ 4\sigma_j^3 \\ \frac{2III}{\sigma_j} \end{bmatrix} + \frac{4III\psi_{III}}{\sigma_i\sigma_j}
$$

```
inline double IsotropicHyperelasticFEM::gammaValue(int i, int j, double sigma[3], double invariants[3], double gradient[3], double hessian[6])
{
  /*
    The hessian is in order (11,12,13,22,23,33)
    | 11 12 13 |   | 0 1 2 |
    | 21 22 23 | = | 1 3 4 |
    | 31 32 33 |   | 2 4 5 |
  */

  double tempGammaVec1[3];
  tempGammaVec1[0] = 2.0 * sigma[i];
  tempGammaVec1[1] = 4.0 * sigma[i] * sigma[i] * sigma[i];
  tempGammaVec1[2] = 2.0 * invariants[2] / sigma[i];
  double tempGammaVec2[3];
  tempGammaVec2[0] = 2.0 * sigma[j];
  tempGammaVec2[1] = 4.0 * sigma[j] * sigma[j] * sigma[j];
  tempGammaVec2[2] = 2.0 * invariants[2] / sigma[j];
  double productResult[3];
  productResult[0] = (tempGammaVec2[0] * hessian[0] + tempGammaVec2[1] * hessian[1] + 
		      tempGammaVec2[2] * hessian[2]);
  productResult[1] = (tempGammaVec2[0] * hessian[1] + tempGammaVec2[1] * hessian[3] + 
		      tempGammaVec2[2] * hessian[4]);
  productResult[2] = (tempGammaVec2[0] * hessian[2] + tempGammaVec2[1] * hessian[4] + 
		      tempGammaVec2[2] * hessian[5]);
  return (tempGammaVec1[0] * productResult[0] + tempGammaVec1[1] * productResult[1] +
	  tempGammaVec1[2] * productResult[2] + 4.0 * invariants[2] * gradient[2] / (sigma[i] * sigma[j]));
}
```

接下来计算
$$
\frac{\partial \bold P}{\partial \bold F}|_{\hat {\bold F}}
$$

```
  dPdF_atFhat[tensor9x9Index(0, 0, 0, 0)] = x1111;
  dPdF_atFhat[tensor9x9Index(0, 0, 1, 1)] = x2211;
  dPdF_atFhat[tensor9x9Index(0, 0, 2, 2)] = x3311;

  dPdF_atFhat[tensor9x9Index(1, 1, 0, 0)] = x2211;
  dPdF_atFhat[tensor9x9Index(1, 1, 1, 1)] = x2222;
  dPdF_atFhat[tensor9x9Index(1, 1, 2, 2)] = x3322;

  dPdF_atFhat[tensor9x9Index(2, 2, 0, 0)] = x3311;
  dPdF_atFhat[tensor9x9Index(2, 2, 1, 1)] = x3322;
  dPdF_atFhat[tensor9x9Index(2, 2, 2, 2)] = x3333;

  dPdF_atFhat[tensor9x9Index(0, 1, 0, 1)] = x2121;
  dPdF_atFhat[tensor9x9Index(0, 1, 1, 0)] = x2112;

  dPdF_atFhat[tensor9x9Index(1, 0, 0, 1)] = x2112;
  dPdF_atFhat[tensor9x9Index(1, 0, 1, 0)] = x2121;

  dPdF_atFhat[tensor9x9Index(0, 2, 0, 2)] = x3131;
  dPdF_atFhat[tensor9x9Index(0, 2, 2, 0)] = x3113;

  dPdF_atFhat[tensor9x9Index(2, 0, 0, 2)] = x3113;
  dPdF_atFhat[tensor9x9Index(2, 0, 2, 0)] = x3131;

  dPdF_atFhat[tensor9x9Index(1, 2, 1, 2)] = x3232;
  dPdF_atFhat[tensor9x9Index(1, 2, 2, 1)] = x3223;

  dPdF_atFhat[tensor9x9Index(2, 1, 1, 2)] = x3223;
  dPdF_atFhat[tensor9x9Index(2, 1, 2, 1)] = x3232;

```

diagonal Piola stress tensor。居然找不到相应的介绍
$$
P_{diag} = (\frac{\partial I}{\partial \lambda})^T\frac{\partial \psi}{\partial I}
$$

```
// compute diagonal Piola stress tensor from the three principal stretches
void IsotropicHyperelasticFEM::ComputeDiagonalPFromStretches(int elementIndex, double *lambda, double *PDiag)
{
  double invariants[3];

  double lambda2[3] = {lambda[0] * lambda[0], lambda[1] * lambda[1], lambda[2] * lambda[2]};
  double IC = lambda2[0] + lambda2[1] + lambda2[2];
  double IIC = lambda2[0] * lambda2[0] + lambda2[1] * lambda2[1] + lambda2[2] * lambda2[2];
  double IIIC = lambda2[0] * lambda2[1] * lambda2[2];

  invariants[0] = IC;
  invariants[1] = IIC;
  invariants[2] = IIIC;

  double dPsidI[3];

  isotropicMaterial->ComputeEnergyGradient(elementIndex, invariants, dPsidI);

  // PDiag = [ dI / dlambda ]^T * dPsidI

  double mat[9];
  mat[0] = 2.0 * lambda[0];
  mat[1] = 2.0 * lambda[1];
  mat[2] = 2.0 * lambda[2];
  mat[3] = 4.0 * lambda[0] * lambda[0] * lambda[0];
  mat[4] = 4.0 * lambda[1] * lambda[1] * lambda[1];
  mat[5] = 4.0 * lambda[2] * lambda[2] * lambda[2];
  mat[6] = 2.0 * lambda[0] * lambda2[1] * lambda2[2];
  mat[7] = 2.0 * lambda[1] * lambda2[0] * lambda2[2];
  mat[8] = 2.0 * lambda[2] * lambda2[0] * lambda2[1];

  Mat3d matM(mat);
  Vec3d dPsidIV(dPsidI);
  Vec3d result;

  result = trans(matM) * dPsidIV;
  result.convertToArray(PDiag);
}
```

irving et al. 04

force on a node i due to a single tetrahedron incident to it is
$$
\bold g_i = -\bold P(A_1\bold N_1 + A_2 \bold N_2 + A_3 \bold N_3)
$$
AN是根据面积计算的权重。
$$
\bold g_i = \bold P \bold b_i
$$

```
      /*
        we compute the nodal forces by G=PBm as described in 
        section 4 of [Irving 04]
      */
      // multiply by 4 because each tet has 4 vertices
      Vec3d forceUpdateA = P * areaWeightedVertexNormals[4 * el + 0];
      Vec3d forceUpdateB = P * areaWeightedVertexNormals[4 * el + 1];
      Vec3d forceUpdateC = P * areaWeightedVertexNormals[4 * el + 2];
      Vec3d forceUpdateD = P * areaWeightedVertexNormals[4 * el + 3];
      
          areaWeightedVertexNormals[4 * el + 0] = (acbArea * acbNormal + adcArea * adcNormal + abdArea * abdNormal) / 3.0;
    areaWeightedVertexNormals[4 * el + 1] = (acbArea * acbNormal + abdArea * abdNormal + bcdArea * bcdNormal) / 3.0;
    areaWeightedVertexNormals[4 * el + 2] = (acbArea * acbNormal + adcArea * adcNormal + bcdArea * bcdNormal) / 3.0;
    areaWeightedVertexNormals[4 * el + 3] = (adcArea * adcNormal + abdArea * abdNormal + bcdArea * bcdNormal) / 3.0;

```

如果我们将F对角化，使用U和V的话
$$
\bold F = \bold U \hat {\bold F} \bold V^T
$$
进而转化为
$$
\bold P = \bold P(\bold F) = \bold U \bold P(\hat {\bold F})\bold V^T
$$

```
      double pHat[3];
      ComputeDiagonalPFromStretches(el, fHat, pHat); // calls the isotropic material to compute the diagonal P tensor, given the principal stretches in fHat
      Vec3d pHatv(pHat);

      // This is the 1st equation in p3 section 5 of [Irving 04]
      // P = Us[el] * diag(pHat) * trans(Vs[el])
      Mat3d P = Us[el];
      P.multiplyDiagRight(pHatv);
      P = P * trans(Vs[el]);
      //isotropicHyperelasticFEM.cpp line 494
```

而且

Invertible Isotropic Hyperelasticity using SVD Gradients
$$
\bold K = \frac{\partial \bold G}{\partial \bold u} =  \frac{\partial \bold G}{\partial \bold F}\frac{\partial \bold F}{\partial \bold u}=  \frac{\partial \bold P}{\partial \bold F}\bold B_m\frac{\partial \bold F}{\partial \bold u}
$$

它算得应该是这个
$$
\frac{\partial \bold P}{\partial \bold F} =\bold U\{\frac{\partial \bold P}{\partial \bold F}|_{\hat{\bold F}}:\bold U^T \delta \bold F\bold V\}\bold V^T :\delta \bold F
$$

```
eiejVector[column] = 1.0;
    Mat3d ei_ej(eiejVector);
    Mat3d ut_eiej_v = UT * ei_ej * (Vs[el]);
    double ut_eiej_v_TeranVector[9]; //in Teran order
    ut_eiej_v_TeranVector[rowMajorMatrixToTeran[0]] = ut_eiej_v[0][0];
    ...
     tempResult += dPdF_atFhat[innerRow * 9 + innerColumn] *
                      ut_eiej_v_TeranVector[innerColumn];
      }
      dPdF_resultVector[teranToRowMajorMatrix[innerRow]] = tempResult;
      ...
          Mat3d dPdF_resultMatrix(dPdF_resultVector);
    Mat3d u_dpdf_vt = (Us[el]) * dPdF_resultMatrix * VT;
    dPdF[column + 0] = u_dpdf_vt[0][0];
```

