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

=================Path tracing in Production
Part 1: Modern Path Tracing  

Partial sub-paths starting from the light, generating using the light tracing algorithm, can be connected to
partial eye-path to create a complete path, which is called bidirectional path tracing (����) . An example
connecting only a path of length twowith the camera path isshown in itssimplest form in figure 3(c),which
can improve situations where light sources concentrate their emission onto a small region. Full ���� first
builds independent paths from the eye and from the light source and then creates new paths by connecting
all their respective sub-paths. Each index that defines at which vertex a connection between the eye and
light subpath is made corresponds to a sampling technique.  