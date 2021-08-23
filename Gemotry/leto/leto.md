网格粒子拓扑优化

https://github.com/xuan-li/LETO

strain energy of the material under a displacement field u is defined as
$$
e(\rho,\bold u) = \int_{\Omega} \psi(\bold F)d\bold X
$$
psi是弹性能量密度，由连续模型决定
$$
\bold F = \frac{\partial \bold x}{\partial \bold X} = \bold I + \frac{\partial \bold u}{\partial \bold X}
$$
对于线弹性来说
$$
\bold \psi_L(\bold F) = \mu||\varepsilon (\bold F)||^2 + \frac{\lambda}{2}tr(\varepsilon(\bold F))^2
$$
NeoHooke
$$
\psi_{NH}(\bold F) = \frac{\mu}{2}(tr(\bold F^T\bold F) - d) - \mu \log J + \frac{\lambda}{2}(\log J)^2
$$
