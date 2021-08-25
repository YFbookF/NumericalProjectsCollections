```
Vector3d energy_derivative(int type, Vector3d S)
{
    
    Vector3d es;
    double J;
    double l1;
    double l2;
    
    switch(type)
    {
        case 0: //arap
            es[0] = 2 * (S[0] - 1);
            es[1] = 2 * (S[1] - 1);
            es[2] = 2 * (S[2] - 1);
            break;
            
        case 1: // mips
            es[0] = 1.0 / S[1] - S[1] / (S[0] * S[0]);
            es[1] = 1.0 / S[0] - S[0] / (S[1] * S[1]);
            break;
            
        case 2: // iso
            es[0] = 2 * S[0] - 2.0 / (S[0] * S[0] * S[0]);
            es[1] = 2 * S[1] - 2.0 / (S[1] * S[1] * S[1]);
            es[2] = 2 * S[2] - 2.0 / (S[2] * S[2] * S[2]);
            break;
            
        case 3: // amips
            es[0] = amips_param * exp(amips_param * (S[0] / S[1] + S[1] / S[0])) * (1.0 / S[1] - S[1] / (S[0] * S[0]));
            es[1] = amips_param * exp(amips_param * (S[0] / S[1] + S[1] / S[0])) * (1.0 / S[0] - S[0] / (S[1] * S[1]));
            break;
            
        case 4: // conf
            es[0] = 2 * S[0] / (S[1] * S[1]);
            es[1] = -2 * S[0] * S[0] / (S[1] * S[1] * S[1]);
            break;
            
        case 5:
            J = S[0] * S[1] * S[2];
            
            J = max(J, 1e-8);
            
            l1 = S[0] * S[0] + S[1] * S[1] + S[2] * S[2];
            l2 = S[0] * S[0] * S[1] * S[1] + S[1] * S[1] * S[2] * S[2] + S[2] * S[2] * S[0] * S[0];
            
            //es[0] = 2 * (J - 1) * S[1];
            //es[1] = 2 * (J - 1) * S[0];
            
            es[0] = c1_param * (-2.0 / 3.0 * pow(J, -5.0 / 3.0) * S[1] * S[2] * l1 + pow(J, -2.0 / 3.0) * 2 * S[0]) 
                    + c2_param * (-4.0 / 3.0 * pow(J, -7.0 / 3.0) * S[1] * S[2] * l2 + pow(J, -4.0 / 3.0) * 2 * (S[1] * S[1] + S[2] * S[2]) * S[0]) 
                    + d1_param * 2 * (J - 1) * S[1] * S[2];
            
            es[1] = c1_param * (-2.0 / 3.0 * pow(J, -5.0 / 3.0) * S[2] * S[0] * l1 + pow(J, -2.0 / 3.0) * 2 * S[1]) 
                    + c2_param * (-4.0 / 3.0 * pow(J, -7.0 / 3.0) * S[2] * S[0] * l2 + pow(J, -4.0 / 3.0) * 2 * (S[0] * S[0] + S[2] * S[2]) * S[1]) 
                    + d1_param * 2 * (J - 1) * S[2] * S[0];
            
            es[2] = c1_param * (-2.0 / 3.0 * pow(J, -5.0 / 3.0) * S[0] * S[1] * l1 + pow(J, -2.0 / 3.0) * 2 * S[2]) 
                    + c2_param * (-4.0 / 3.0 * pow(J, -7.0 / 3.0) * S[0] * S[1] * l2 + pow(J, -4.0 / 3.0) * 2 * (S[0] * S[0] + S[1] * S[1]) * S[2]) 
                    + d1_param * 2 * (J - 1) * S[0] * S[1];

            break;
    }
    
    return es;

}


double energy_hessian(double s1, double s2, double s3, double s1j, double s1k, double s2j, double s2k, double s3j, double s3k, double s1jk, double s2jk, double s3jk, int type)
{
    
    double hessian;
    double v1, v2, v3;
     
    switch(type)
    {
        case 0: //arap
            v1 = 2 * ((s1 - 1) * s1jk + (s2 - 1) * s2jk + (s3 - 1) * s3jk);
            v2 = 2 * (s1j * s1k + s2j * s2k + s3j * s3k);
            hessian = v1 + v2;
            break;
            
        case 1: // mips
            
            break;
            
        case 2: // iso
            
            v1 = 2 * s1k * s1j + 2 * s1 * s1jk;
            v2 = 2 * s2k * s2j + 2 * s2 * s2jk;
            v3 = 2 * s3k * s3j + 2 * s3 * s3jk;
            
            v1 += 6 * s1k * s1j / pow(s1, 4) - 2 * s1jk / pow(s1, 3);
            v2 += 6 * s2k * s2j / pow(s2, 4) - 2 * s2jk / pow(s2, 3);
            v3 += 6 * s3k * s3j / pow(s3, 4) - 2 * s3jk / pow(s3, 3);
            
            hessian = v1 + v2 + v3;
            break;
            
        case 3: // amips
            
            break;
            
        case 4: // conf
            
            break;
            
        case 5: // gmr
            
            double J = s1 * s2 * s3;
            double l1 = s1 * s1 + s2 * s2 + s3 * s3;
            double l2 = s1 * s1 * s2 * s2 + s2 * s2 * s3 * s3 + s3 * s3 * s1 * s1;
            
            double Jj = s1j * s2 * s3 + s1 * s2j * s3 + s1 * s2 * s3j;
            double Jk = s1k * s2 * s3 + s1 * s2k * s3 + s1 * s2 * s3k;
            
            double l1j = 2 * (s1 * s1j + s2 * s2j + s3 * s3j);
            double l1k = 2 * (s1 * s1k + s2 * s2k + s3 * s3k);
            
            double l2j = 2 * (s1 * s1j * s2 * s2 + s2 * s2j * s1 * s1 + s2 * s2j * s3 * s3 + s3 * s3j * s2 * s2 + s3 * s3j * s1 * s1 + s1 * s1j * s3 * s3);
            double l2k = 2 * (s1 * s1k * s2 * s2 + s2 * s2k * s1 * s1 + s2 * s2k * s3 * s3 + s3 * s3k * s2 * s2 + s3 * s3k * s1 * s1 + s1 * s1k * s3 * s3);
            
            double Jjk = (s1jk * s2 * s3 + s1j * s2k * s3 + s1j * s2 * s3k) + (s1k * s2j * s3 + s1 * s2jk * s3 + s1 * s2j * s3k) + (s1k * s2 * s3j + s1 * s2k * s3j + s1 * s2 * s3jk);
            
            double l1jk = 2 * (s1k * s1j + s1 * s1jk + s2k * s2j + s2 * s2jk + s3k * s3j + s3 * s3jk);
            
            double l2jk = s1k * s1j * s2 * s2 + s1 * s1jk * s2 * s2 + 2 * s1 * s1j * s2 * s2k;
            l2jk += s2k * s2j * s1 * s1 + s2 * s2jk * s1 * s1 + 2 * s2 * s2j * s1 * s1k;
            l2jk += s2k * s2j * s3 * s3 + s2 * s2jk * s3 * s3 + 2 * s2 * s2j * s3 * s3k;
            l2jk += s3k * s3j * s2 * s2 + s3 * s3jk * s2 * s2 + 2 * s3 * s3j * s2 * s2k;
            l2jk += s3k * s3j * s1 * s1 + s3 * s3jk * s1 * s1 + 2 * s3 * s3j * s1 * s1k;
            l2jk += s1k * s1j * s3 * s3 + s1 * s1jk * s3 * s3 + 2 * s1 * s1j * s3 * s3k;
            l2jk *= 2;
                         
            hessian = c1_param * (-2.0 / 3.0 * pow(J, -5.0 / 3.0) * Jk * l1j + pow(J, -2.0 / 3.0) * l1jk + 10.0 / 9.0 * pow(J, -8.0 / 3.0) * Jk * Jj * l1 - 2.0 / 3.0 * pow(J, -5.0 / 3.0) * Jjk * l1 - 2.0 / 3.0 * pow(J, -5.0 / 3.0) * Jj * l1k);              
            hessian += c2_param * (-4.0 / 3.0 * pow(J, -7.0 / 3.0) * Jk * l2j + pow(J, -4.0 / 3.0) * l2jk + 28.0 / 9.0 * pow(J, -10.0 / 3.0) * Jk * Jj * l2 - 4.0 / 3.0 * pow(J, -7.0 / 3.0) * Jjk * l2 - 4.0 / 3.0 * pow(J, -7.0 / 3.0) * Jj * l2k);              
            hessian += 2 * d1_param * (Jk * Jj + J * Jjk - Jjk);              
            
            break;
    }
    
    return hessian;

}

```

Scalable Locally Injective Mapping

AsRigidAsPossible
$$
\mathcal{D}_{ARAP}(\bold J_f(\bold x)) = ||\bold J_f(\bold x) - \bold R(\bold J_f(\bold x))||_F^2
$$
也可以这么写
$$
||\bold J - \bold R||_F^2 = ||\bold S_J - \bold I||_F^2
$$

```
case 0: //arap
            es[0] = 2 * (S[0] - 1);
            es[1] = 2 * (S[1] - 1);
            es[2] = 2 * (S[2] - 1);
            break;
case 0: //arap
            v1 = 2 * ((s1 - 1) * s1jk + (s2 - 1) * s2jk + (s3 - 1) * s3jk);
            v2 = 2 * (s1j * s1k + s2j * s2k + s3j * s3k);
            hessian = v1 + v2;
            break;
```

ISO
$$
\mathcal{D}_{iso}(\bold J) = \exp(s \cdot \mathcal{D}_{iso}(\bold J))\\
\mathcal{D}_{iso}(\bold J) = \frac{1}{2}(\frac{tr(\bold J^T\bold J)}{det(\bold J)} + \frac{1}{2}(det(\bold J) + det(\bold J^{-1})))
$$

```
 case 2: // iso
            es[0] = 2 * S[0] - 2.0 / (S[0] * S[0] * S[0]);
            es[1] = 2 * S[1] - 2.0 / (S[1] * S[1] * S[1]);
            es[2] = 2 * S[2] - 2.0 / (S[2] * S[2] * S[2]);
            break;
            
        case 3: // amips
            es[0] = amips_param * exp(amips_param * (S[0] / S[1] + S[1] / S[0])) * (1.0 / S[1] - S[1] / (S[0] * S[0]));
            es[1] = amips_param * exp(amips_param * (S[0] / S[1] + S[1] / S[0])) * (1.0 / S[0] - S[0] / (S[1] * S[1]));
            break;
```

conformal distortions
$$
\mathcal{D}(\bold J) = \frac{tr(\bold J^)}{det(\bold J)^{2/d}}
$$
