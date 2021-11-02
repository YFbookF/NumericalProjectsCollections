=============A Novel Plastic Phase-Field Method for Ductile Fracture with GPU  

phase-field theory
$$
\mathcal{W} = \int_{\Omega^0}\hat \Psi d\bold X + G_c \int_{\Gamma}d\bold X
$$
W is the total free energy, Gc is the fracture toughness, Gamma is the discontinuous crack set

the intergral of the fracture surface is replaced by an approximate volume integral as
$$
\mathcal W^s(s) = G_c = \int_{\Omega^0}(\frac{1}{4l}(1 - s)^2 + l|\nabla s|^2)d\bold X
$$
the phase field value s represents the material state from healthy s = 1 to damaged s < 1, s=  0 denoteds the material is completely broken

![image-20211102170210139](D:\定理\断裂力学\image-20211102170210139.png)

================CD-MPM: Continuum Damage Material Point Methods for Dynamic
Fracture Animation  

Griffith`s theory of fracture defines the total free energy to be the summation of the elastic potential and the released energy at crack surfaces
$$
\xi = \int \hat \Psi(\bold F)d\bold X + \int G d\bold X
$$
hat psi is a degraded (damaged) finite-strain hyperelastic energy density function. G is critical energy release rate, also called fracture toughness of the material.

