==================

Point Based Animation of Elastic, Plastic and Melting Object  

第一步
$$
u(\bold x_i + \Delta \bold x_i) = \bold u_i + \nabla_u|\bold x_i
$$
其中
$$
\tilde u_j = u_i + \nabla u|_{\bold x_i} \cdot \bold x_{ij} 
$$
第三
$$
e = \sum_j(\tilde u_j - u_j)^2 w_{ij} \qquad \bold x_{ij} = \bold x_j - \bold x_i
$$
第四
$$
Ax = b \qquad (\sum_j \bold x_{ij}\bold x_{ij}^T w_{ij})\nabla u|_{\bold x_i} = \sum_j(u_j - u_i)\bold x_{ij}w_{ij}
$$
看不懂，告辞

比如原来两个点[0,1],[1,0]，现在他们移动到[0,2]和[1,1]，形变梯度是

那么变形梯度
$$
\bold X = \begin{bmatrix}0 & 1 \\ 1 & 0 \end{bmatrix} \qquad \bold x = \begin{bmatrix}0 & 1 \\ 2 & 1 \end{bmatrix} \qquad \bold F = \begin{bmatrix}1 & 0 \\ 1 & 2 \end{bmatrix} = \bold I + \nabla \bold u^T = \begin{bmatrix}1 & 0 \\ 0 & 1 \end{bmatrix} + \begin{bmatrix}0 & 0 \\ 1 & 1 \end{bmatrix}
$$
用上面的来说
$$
\begin{bmatrix} 1 & -1 \end{bmatrix}\begin{bmatrix} 1\\-1 \end{bmatrix}x = 
$$
应变能
$$
\varepsilon = \bold J^T \bold J - \bold I = \nabla \bold u + \nabla \bold u^T + \nabla \bold u \nabla \bold u^T
$$
==========matlab有限元结构动力学

![image-20211102141122903](D:\定理\数学\image-20211102141122903.png)
