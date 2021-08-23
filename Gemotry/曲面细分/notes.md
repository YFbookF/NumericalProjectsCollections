chaikin方法
$$
p_{2i}^{k+1} = \frac{3}{4}p_i^k + \frac{1}{4}p_{i+1}^k\\
p_{2i+1}^{k+1} = \frac{1}{4}p_i^k + \frac{3}{4}p_{i+1}^k
$$
cubic spline
$$
p_{2i}^{k+1} = \frac{1}{4}p_i^k + \frac{1}{2}p_{i+1}^k\\
p_{2i+1}^{k+1} = \frac{1}{8}p_i^k + \frac{3}{4}p_{i+1}^k + \frac{1}{8}p_{i+2}^k
$$
butterfly
$$
q^{k+1} = u(p_1^k + p_2^k) + v(p_3^k + p_4^k) - w(p_5^k + p_6^k + p_7^k + p_8^k)
$$
nbflip/edgecollapse.cpp