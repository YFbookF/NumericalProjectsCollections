==========BSSRDF Explorer: A rendering framework for the BSSRDF  

The distribution of phtotons exiting through the surface gives rise to a radially symmetric reflectance called reduced reflectance
$$
R_d(\vec x) =\int_{\Omega}L_r(\vec x,\vec w')f_t(\vec x,\vec w')d\vec w'
$$
ft is the bidirectional transmittance distrbution function BTDF.

In BSSRDF Sd cannot be constructed exactly from Rd, many methods in graphics approximate Sd using the reflectance distribution profile and a directionally dependent Fresnel transmission term
$$
S_d = \frac{1}{\pi}F_t(\vec x_i,\vec w_i) R_d(\vec x_o - \vec x_i)\frac{F_t(\vec x_o,\vec w_o)}{4C_{\phi}(1/\eta)}
$$
4 C phi is an approximate normalization factor