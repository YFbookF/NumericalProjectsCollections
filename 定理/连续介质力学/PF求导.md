The Material Point Method for Simulating Continuum Materials Chenfanfu Jiang∗1, Craig Schroedery2, Joseph Teranz1,3, Alexey Stomakhinx3, and Andrew Selle{3  

例如对于CoRotate Model
$$
\Psi = \mu \sum_{i=1}^d(\sigma_i - 1)^2 + \frac{\lambda}{2}(J - 1)^2
$$
求导开始
$$
\frac{\partial }{\partial \bold F}\sum_{i=1}^d\sigma_i^2 = 2\bold F \qquad \frac{\partial }{\partial \bold F}\sum_{i=1}^d \sigma_i = \bold R
$$
我并不觉得它可以show
$$
\bold P(\bold F) = \frac{\partial \Psi}{\partial \bold F}(\bold F) = 2\mu(\bold F - \bold R) + \lambda (J - 1)J\bold F^{-1}
$$
而二次导
$$
\delta (\frac{\partial \Psi}{\partial \bold F}) = 2\mu \delta \bold F - 2\mu \delta \bold R + \lambda J \bold F^{-T}\delta J + \lambda(J - 1)\delta(J\bold F^{-T})\\
$$
The Material Point Method for Simulating Continuum Materials Chenfanfu Jiang∗1, Craig Schroedery2, Joseph Teranz1,3, Alexey Stomakhinx3, and Andrew Selle{3  

p27

拉格朗日视角下的守恒

质量守恒
$$
\bold R(\bold X,t)J(\bold X,t) = R(\bold X,0)
$$
动量守恒
$$
R(\bold X,0)\frac{\partial \bold V}{\partial t} = \nabla^{\bold X} \cdot \bold P + R(\bold X,0)\bold g
$$
R is the lagrangian mass density
$$
\frac{\partial }{\partial t}(R(\bold X,t)J(\bold X,t)) = 0 = \frac{\partial R}{\partial t}J + R\frac{\partial J}{\partial t}
$$
notice that also
$$
\frac{\partial J}{\partial t} = \frac{\partial J}{\partial F_{ij}}\frac{\partial F_{ij}}{\partial t} = JF^{-1}_{ji}\frac{\partial \bold V_i}{\partial \bold X_j} = JF_{ji}^{-1}\frac{\partial v_i}{\partial x_i}F_{kj} = J \delta_{ik}\frac{\partial v_i}{\partial x_k} = J\frac{\partial v_i}{\partial x_i}
$$
![image-20211105234039226](E:\mycode\collection\定理\连续介质力学\image-20211105234039226.png)

```
            Vector2 new_vel = new Vector2(0.0f,0.0f);
            float new_C00 = 0;
            float new_C01 = 0;
            float new_C10 = 0;
            float new_C11 = 0;

            for (int j = 0; j < 3; j++)
            {
                for (int i = 0; i < 3; i++)
                {
                    if (basex + i < 0 || basex + i > GridSize)
                        continue;
                    if (basey + j < 0 || basey + j > GridSize)
                        continue;
                    int idx = (basey + j) * GridSize + basex + i;
                    float weight = wx[i] * wy[j];
                    float dposx = (i - fx) * dx;
                    float dposy = (j - fy) * dx;

                    new_vel += weight * GridVel[idx];

                    new_C00 += 4 * weight / dx / dx * GridVel[idx].x * dposx;
                    new_C01 += 4 * weight / dx / dx * GridVel[idx].x * dposy;
                    new_C10 += 4 * weight / dx / dx * GridVel[idx].y * dposx;
                    new_C11 += 4 * weight / dx / dx * GridVel[idx].y * dposy;
                }
            }
            ParticleVel[p] = new_vel;
            ParticlePos[p] += dt * new_vel;
            ParticleC00[p] = new_C00;
            ParticleC01[p] = new_C01;
            ParticleC10[p] = new_C10;
            ParticleC11[p] = new_C11;
            ParticleJ[p] *= (1 + dt * (new_C00 + new_C11));
```

Cp is just the euler velocity gradient
$$
\bold C_p = \sum_i \bold v_i (\frac{\partial N_i}{\partial x}(x_p))^T
$$
apic

![image-20211105234638634](E:\mycode\collection\定理\连续介质力学\image-20211105234638634.png)

![image-20211105234649671](E:\mycode\collection\定理\连续介质力学\image-20211105234649671.png)

=========================Strain Based Dynamics  

bending
$$
\phi = \arccos(\frac{(\bold p_3 - \bold p_1)\times (\bold p_4 - \bold p_1)}{|(\bold p_3 - \bold p_1) \times (\bold p_4 - \bold p_1)|} \cdot\frac{(\bold p_4 - \bold p_2)\times (\bold p_3 - \bold p_2)}{|(\bold p_4 - \bold p_2) \times (\bold p_3 - \bold p_2)|} )
$$
求导
$$
\nabla_p \phi = |\bold e|\bold n_1 \qquad \nabla_{\bold p_2}\phi = |\bold e|\bold n_2 \\
\nabla_{\bold p_3}\phi = \frac{(\bold p_1 - \bold p_4)\cdot \bold e}{|\bold e|}\bold n_1 + \frac{(\bold p_2 - \bold p_4)\cdot {\bold e}}{|\bold e|}\bold n_2
$$
where
$$
\bold e = \bold p_4 - \bold p_3
$$
volume / Area Conservation Constraints
$$
C(\bold p_1,\bold p_2,\bold p_3) = \det(\bold P) - \det(\bold Q) = \bold p_1^T(\bold p_2 \times \bold p_3) - \bold q_1^T(\bold q_2 \times \bold q_3)
$$
P is now posiition, Q is material position
$$
\nabla_{\bold p_1}C = \bold p_2 \times \bold p_3\\
\nabla_{\bold p_2}C = \bold p_3 \times \bold p_1\\
\nabla_{\bold p_3}C = \bold p_1 \times \bold p_2\\
\nabla_{\bold p_0}C = -\bold p_2 \times \bold p_3-\bold p_3 \times \bold p_1-\bold p_1 \times \bold p_2
$$
triangle perservation
$$
C(\bold p_1,\bold p_2) = |\bold p_1 \times \bold p_2|^2 - |\bold q_1 \times \bold q_2|^2
$$
its derivatives are
$$
\nabla_{\bold p_1}C = 2\bold p_2 \times (\bold p_1 \times \bold p_2)\\
\nabla_{\bold p_2}C = 2\bold p_1 \times (\bold p_2 \times \bold p_1)\\
\nabla_{\bold p_0}C = -2\bold p_2 \times (\bold p_1 \times \bold p_2)- 2\bold p_1 \times (\bold p_2 \times \bold p_1)
$$
===============Practical notes on implementing derivatives

===============Editing Fluid Animation using Flow Interpolation  

![image-20211116121518227](E:\mycode\collection\定理\连续介质力学\image-20211116121518227.png)

=================https://github.com/Azmisov/snow
$$
J ： \R^{2\times2} = \det(\bold F) = F_{11}F_{22} - F_{21}F_{12} \\
\frac{\partial J}{\partial \bold F}(\hat{\bold F}) = J(\hat{\bold F})\hat{\bold F}^{-T} \qquad \delta J(\hat{\bold F},\delta{\bold F}) = J(\hat{\bold F})\hat F_{ji}^{-1}\delta F_{ij}
$$
迹
$$
\frac{\partial Tr}{\partial F_{kl}}(\bold F) = \delta_{kl} \qquad \delta Tr(\hat{\bold F},\delta {\bold F}) = \delta_{kl}\delta F_{kl} = \delta F_{kk} = \delta F_{ii}
$$
![image-20211126130135249](E:\mycode\collection\定理\连续介质力学\image-20211126130135249.png)

