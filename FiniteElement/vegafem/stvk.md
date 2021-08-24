**Interactive Skeleton-Driven Dynamic Deformations**

StVKTetHighMemoryABCD.cpp

弹性势能elastic potential energy
$$
V = G\int_{\Omega}\{ \frac{\nu}{1-2\nu}tr^2(e) + \delta^{ij} \delta^{kl}e_{ik}e_{jl}\}d\Omega
$$
ABCD是算它的梯度的，其中
$$
A = \int_{\Omega}(\frac{\partial \phi^{a}}{\partial \bold x} \otimes \frac{\partial \phi^b}{\partial \bold x})d\Omega \\
B = \int_{\Omega}(\frac{\partial \phi^{a}}{\partial \bold x} \cdot \frac{\partial \phi^b}{\partial \bold x})d\Omega \\
C = \int_{\Omega}\frac{\partial \phi^a}{\partial \bold x}(\frac{\partial \phi^b}{\partial \bold x} \cdot \frac{\partial \phi^c}{\partial \bold x})d\Omega\\
D = \int_{\Omega}(\frac{\partial \phi^a}{\partial \bold x} \cdot \frac{\partial \phi^b}{\partial \bold x})(\frac{\partial \phi^c}{\partial \bold x} \cdot \frac{\partial \phi^d}{\partial \bold x})d\Omega
$$

```
  // A
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      A[i][j] = volume * tensorProduct(Phig[i], Phig[j]);

  // B
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      B[i][j] = volume * dot(Phig[i], Phig[j]);    

  // C
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      for (int k=0; k<4; k++)
        C[i][j][k] = volume * dot(Phig[j], Phig[k]) * Phig[i];

  // D
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      for (int k=0; k<4; k++)
        for (int l=0; l<4; l++)
          D[i][j][k][l] = volume * dot(Phig[i],Phig[j]) * dot(Phig[k],Phig[l]);
```

