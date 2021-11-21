Mixed Finite Elements for Variational Surface Modeling  

https://igl.ethz.ch/projects/mixed-fem/

biharm
$$
\begin{bmatrix} -M^d & L_{\Omega,\Omega} \\ L_{\Omega,\Omega} & 0\end{bmatrix}\begin{bmatrix} \bold v \\ \bold u\end{bmatrix} = \begin{bmatrix} -L\bold u - L \bold u \\ 0\end{bmatrix}
$$

```
  if strcmp(reduction,'flatten')
    % system matrix for the system with x as variable only
    % obtained by eliminating y = M^{-1} (S_{all,Omega} - rhs_Dx)
    A = S(Omega,all) * (M(all,all) \ S(all,Omega)); 
  else % full matrix
    % system matrix for x,y variables
    n_Omega = size(Omega,2);
    Z_Omega_Omega = sparse(n_Omega, n_Omega);
    A = [ -M(all,all)      S(  all,Omega);   ...
           S(Omega,all)    Z_Omega_Omega  ];     
  end
```

