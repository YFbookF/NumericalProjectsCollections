Classical Mechanics An Introduction by Dieter Strauch (z-lib.org)

Steiner`s theorem gives a relation between the inertia tensor relative to the center mass S and that with respect to an arbitrary point Q. i.e.
$$
\Theta^Q = \Theta^S + M(R^21 - \bold R \otimes \bold R)
$$
比如对于三维正方体，如果旋转重心在中间，那么
$$
\Theta_{ij}^{(S)} = \begin{bmatrix} \frac{1}{12}M(L_x^2 + L_y^2 + L_z^2 - L_i^2) & i = j \\ 0 & i \neq j\end{bmatrix}\\
\Theta_{ij}^{(Q)} = \begin{bmatrix} \frac{1}{3}M(L_x^2 + L_y^2 + L_z^2 - L_i^2) & i = j \\ -\frac{1}{4}ML_iL_j & i \neq j\end{bmatrix}
$$
注意
$$
R = \frac{1}{2}(L_x,L_y,L_z)
$$
那么
$$
\Theta_{ij}^{(Q)} - \Theta_{ij}^{(S)} = M(R^21 - \bold R \otimes \bold R) \\= \frac{M}{4}[(L_x^2 + L_y^2 + L_z^2)\begin{bmatrix}1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1\end{bmatrix} - \begin{bmatrix}L_x^2 & L_xL_y & L_xL_z \\ L_yL_x & L_y^2 & L_yL_z \\ L_zL_x & L_xL_y & L_z^2\end{bmatrix}]\\
= \frac{M}{4}\begin{bmatrix}L_y^2 + L_x^2 & -L_xL_y & -L_xL_z \\ -L_yL_x & L_x^2 + L_z^2 & -L_yL_z \\ -L_zL_x & -L_xL_y & L_x^2 + L_y^2\end{bmatrix}
$$
