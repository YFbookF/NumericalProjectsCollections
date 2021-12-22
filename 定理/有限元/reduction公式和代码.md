Skipping Steps in Deformable Simulation with Online Model Reduction  

Dimensional model reduction in non-linear nite element dynamics of solids and structures  

FEM Simulation of 3D Deformable Solids: A practitioner’s guide to theory, discretization and model reduction.  

E:

```

```

\mycode\Elastic\MaterialEditor-master\src\fitmat.cpp

```
UpdateStiffness(K, Efull.data());
        solver.compute(K);
        ubat =  solver.solve(fbat); // ubat = u* = K^(-1) * f

        for(int i = 0; i < m_visDofs.size(); i++)
            dubat[i] = ubat[m_visDofs[i]] - m_visDisp[i]; // du = u* - u

        energy = 0.5 * dubat.squaredNorm(); // || K^(-1) * f - u ||^2


        if(grad == nullptr) return;

        for(int i = 0; i < ne; i++)
            grad[i] = 0.0;

        for(int ir = 0; ir < ne; ir++)
        {
            UpdateStiffness(DK, m_Phi.data() + ir * ne); // compute each p(K)/p(q_i)
                                                        // p denote partial derivative
            DVector Df = DK * ubat; // DK: p(K)/p(q_i)   ubat: K^(-1) * f
            Df = solver.solve(Df); // Df = K^(-1) * p(K)/p(q_i) * K^(-1) * f

            for(int i = 0; i < m_visDofs.size(); i++)
                grad[ir] += -Df[m_visDofs[i]] * dubat[i]; // chain rule
        }
```

Interactive Material Design Using Model Reduction  

看不懂

![image-20211219095037113](E:\mycode\collection\定理\有限元\image-20211219095037113.png)

Akantu

E:\mycode\Elastic\Akantu-master\extra_packages\extra-materials\src\material_damage\material_iterative_stiffness_reduction.cc

```

          /// increment the counter of stiffness reduction steps
          *reduction_it += 1;
          if (*reduction_it == this->max_reductions)
            *dam_it = this->max_damage;
          else {
            /// update the damage on this quad
            *dam_it =
                1. - (1. / std::pow(this->reduction_constant, *reduction_it));
            /// update the stiffness on this quad
            *Sc_it = (*eps_u_it) * (1. - (*dam_it)) * this->E * (*D_it) /
                     ((1. - (*dam_it)) * this->E + (*D_it));
          }
          nb_damaged_elements += 1;
```

vegafem

```
  // u = U*q
  SynthesizeVector(3*n1, r1, U1, q, u);

  stVKInternalForces->ComputeForces(u,forces);
```

==============delfem2

```
double Cinv[ndim][ndim];
  const double p3C = DetInv_Mat3(Cinv,C);

  const double tmp1 = 1.0 / pow(p3C, 1.0 / 3.0);
  const double tmp2 = 1.0 / pow(p3C, 2.0 / 3.0);
  const double pi1C = p1C*tmp1; // 1st reduced invariant
  const double pi2C = p2C*tmp2; // 2nd reduced invariant
  const double W = c1*(pi1C-3.) + c2*(pi2C-3.);

  { // compute 2nd Piola-Kirchhoff tensor here
    double S[ndim][ndim]; // 2nd Piola-Kirchhoff tensor
    for (unsigned int idim = 0; idim < ndim; idim++) {
      for (unsigned int jdim = 0; jdim < ndim; jdim++) {
        S[idim][jdim] =
            - 2.0 * c2 * tmp2 * C[idim][jdim]
            - 2.0 * (c1 * pi1C + c2 * 2.0 * pi2C) / 3.0 * Cinv[idim][jdim];
      }
    }
    {
      const double dtmp1 = 2.0 * c1 * tmp1 + 2.0 * c2 * tmp2 * p1C;
      S[0][0] += dtmp1;
      S[1][1] += dtmp1;
      S[2][2] += dtmp1;
    }
    { // 2nd piola-kirchhoff tensor is symmetric. Here extracting 6 independent elements.
      dWdC2[0] = S[0][0];
      dWdC2[1] = S[1][1];
      dWdC2[2] = S[2][2];
      dWdC2[3] = S[0][1];
      dWdC2[4] = S[1][2];
      dWdC2[5] = S[2][0];
    }
  }
  
```

