HOUSEHOLDER REDUCTION
Per Brinch Hansen  

高斯消元法，如果If the pivot element aii is very small, the scaling factor becomes very large and we may end up subtracting very large reals from very small ones. This makes the results highly inaccurate. The numerical instability of Gaussian elimination can be reduced by pivoting, a rearrangement of the rows and columns which makes the pivot element as large as possible  

reflection

很显然
$$
f\bold v = \bold a - \bold b
$$
紧接着就有
$$
||\bold a||^2 = ||\bold b||^2 \\
= (\bold a - f\bold v)^T(\bold a - f\bold v) \\
= \bold a ^T\bold a - f\bold a^T\bold v - f\bold v^T\bold a + f^2 \bold v^T\bold v \\
= ||\bold a||^2 - 2f\bold v^T \bold a + f^2
$$
那么也就是
$$
f = 2\bold v^T\bold a = -2\bold v^T\bold b
$$
也就是
$$
\bold b = \bold a - \bold v f = \bold I \bold a - \bold v(2\bold v^T\bold a) = (\bold I - 2\bold v \bold v^T)\bold a
$$
那也就是
$$
\bold b = \bold H \bold a \qquad \bold H =(\bold I - 2\bold v \bold v^T)
$$
