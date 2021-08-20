**Matching Fluid Simulation Elements to Surface Geometry and Topology**

Voronoi三角形，三角形级别的融合变换。挺复杂的。

值得一看共轭梯度法

CGNR

MINRES

不完全Cholesky分解

Volume 体积

```
double signed_volume(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3)
{
   // Equivalent to triple(x1-x0, x2-x0, x3-x0), six times the signed volume of the tetrahedron.
   // But, for robustness, we want the result (up to sign) to be independent of the ordering.
   // And want it as accurate as possible...
   // But all that stuff is hard, so let's just use the common assumption that all coordinates are >0,
   // and do something reasonably accurate in fp.

   // This formula does almost four times too much multiplication, but if the coordinates are non-negative
   // it suffers in a minimal way from cancellation error.
   return ( x0[0]*(x1[1]*x3[2]+x3[1]*x2[2]+x2[1]*x1[2])
           +x1[0]*(x2[1]*x3[2]+x3[1]*x0[2]+x0[1]*x2[2])
           +x2[0]*(x3[1]*x1[2]+x1[1]*x0[2]+x0[1]*x3[2])
           +x3[0]*(x1[1]*x2[2]+x2[1]*x0[2]+x0[1]*x1[2]) )

        - ( x0[0]*(x2[1]*x3[2]+x3[1]*x1[2]+x1[1]*x2[2])
           +x1[0]*(x3[1]*x2[2]+x2[1]*x0[2]+x0[1]*x3[2])
           +x2[0]*(x1[1]*x3[2]+x3[1]*x0[2]+x0[1]*x1[2])
           +x3[0]*(x2[1]*x1[2]+x1[1]*x0[2]+x0[1]*x2[2]) );
}
```

解三次方程

```
for(unsigned int i=1; i<interval_times.size(); ++i){
      double tlo=interval_times[i-1], thi=interval_times[i], tmid;
      double vlo=interval_values[i-1], vhi=interval_values[i], vmid;
      if((vlo<0 && vhi>0) || (vlo>0 && vhi<0)){
         // start off with secant approximation (in case the cubic is actually linear)
         double alpha=vhi/(vhi-vlo);
         tmid=alpha*tlo+(1-alpha)*thi;
         for(int iteration=0; iteration<50; ++iteration){
            vmid=signed_volume((1-tmid)*x0+tmid*xnew0, (1-tmid)*x1+tmid*xnew1,
                               (1-tmid)*x2+tmid*xnew2, (1-tmid)*x3+tmid*xnew3);
            if(std::fabs(vmid)<1e-2*convergence_tol) break;
            if((vlo<0 && vmid>0) || (vlo>0 && vmid<0)){ // if sign change between lo and mid
               thi=tmid;
               vhi=vmid;
            }else{ // otherwise sign change between hi and mid
               tlo=tmid;
               vlo=vmid;
            }
            if(iteration%2) alpha=0.5; // sometimes go with bisection to guarantee we make progress
            else alpha=vhi/(vhi-vlo); // other times go with secant to hopefully get there fast
            tmid=alpha*tlo+(1-alpha)*thi;
         }
         possible_times.push_back(tmid);
      }
   }
```

