https://github.com/YuCrazing/Taichi/blob/6d0d3f67ea3561a5ba24037e173484f229f269cd/fem_3d_imp/fem-3d-implicit.py#L189

```
 def F(self, i): # deformation gradient
        return self.D(i) @ self.B[i]
```

这里D是D_X，B是d_x

这个是bw08

```
    @ti.func
    def Psi(self, i): # (strain) energy density
        F = self.F(i)
        J = max(F.determinant(), 0.01)
        return self.mu / 2 * ( (F @ F.transpose()).trace() - self.dim ) - self.mu * ti.log(J) + self.la / 2 * ti.log(J)**2
```

femsimulation 第24页
$$
\bold P(\bold F) = \mu(\bold F - \mu\bold F^{-T}) + \lambda \log(J)\bold F^{-T}
$$


```
def P(self, i):
        F = self.F(i)
        F_T@ti.kernel
    def compute_force(self):

        for i in range(self.vn):
            self.force[i] = ti.Vector([0, -self.node_mass * self.g, 0])

        for i in range(self.en):

            P = self.P(i)

            H = - self.element_volume[i] * (P @ self.B[i].transpose())

            h1 = ti.Vector([H[0, 0], H[1, 0], H[2, 0]])
            h2 = ti.Vector([H[0, 1], H[1, 1], H[2, 1]])
            h3 = ti.Vector([H[0, 2], H[1, 2], H[2, 2]])

            a = self.element[i][0]
            b = self.element[i][1]
            c = self.element[i][2]
            d = self.element[i][3]


            self.force[a] += h1
            self.force[b] += h2
            self.force[c] += h3
            self.force[d] += - (h1 + h2 + h3)= F.inverse().transpose()
        J = max(F.determinant(), 0.01)
        return self.mu * (F - F_T) + self.la * ti.log(J) * F_T
```

femsimulation 第32页第二个框

```
F = self.F(e)
            F_1 = F.inverse()
            F_1_T = F_1.transpose()
            J = max(F.determinant(), 0.01)

            for n in range(4):
                for dim in range(self.dim):
                    for i in ti.static(range(self.dim)):
                        for j in ti.static(range(self.dim)):
                            
                            # dF/dF_{ij}
                            dF = ti.Matrix([[0.0, 0.0, 0.0], 
                                            [0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0]])
                            dF[i, j] = 1

                            # dF^T/dF_{ij}
                            dF_T = dF.transpose()

                            # Tr( F^{-1} dF/dF_{ij} )
                            dTr = F_1_T[i, j]

                            dP_dFij = self.mu * dF + (self.mu - self.la * ti.log(J)) * F_1_T @ dF_T @ F_1_T + self.la * dTr * F_1_T
                            dFij_ndim = self.dF[e, n, dim][i, j]

                            self.dP[e, n, dim] += dP_dFij * dFij_ndim

            for n in range(4):
                for dim in range(self.dim):
                    self.dH[e, n, dim] = - self.element_volume[e] * self.dP[e, n, dim] @ self.B[e].transpose()

```

