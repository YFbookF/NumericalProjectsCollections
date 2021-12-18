===============Simulation of High-Resolution Granular Media  

As discussed earlier, there are two basic ways of modeling
granular particles: either as spherical or non-spherical particles. Spherical particles have shown limitations inherent to
their own geometry, such as the inability to properly model
the maximum angle of repose in piles, or other phenomena
induced by static friction [LH93]. Non-spherical particles  

were investigated as a way to solve those problems. Fig. 2
shows a close-up of sand grains, whose irregular shape has a
large influence on their macroscopic behavior  

![image-20211206142244959](E:\mycode\collection\定理\流体\image-20211206142244959.png)

In reality, a granular medium is governed by microscopic
mechanical behavior, i.e., the interactions between millions
of grains. The microscopic state of the particles could then
be defined by the contact forces acting on the individual particles. However, the behavior of granular flow at a
macroscopic level can be qualitatively characterized even
by relatively rough models with coarsely sampled particles [BKMS07]. In particular, the macroscopic behavior of
a granular medium is characterized by its relative density,
which is defined by the void ratio [AVS03]. Intuitively, a
coarse granular simulation, such as the one described in
Section 3, will maintain the qualitative behavior of a highresolution simulation provided that the porosity and shape of
the grains are maintained.  

==============Simulation of Granular Flows with the Material Point Method  (Philip Alsop )

离散元比较精确，但很耗时

Bagnold (1954) first identified many of the characteristics of granular flows
[4], by shearing grains immersed in a viscous fluid and in dry conditions.
The MiDi group (2004) conducted a number of experiments and simulations
to investigate the intermediate, \liquid" dense flow regime, introducing the
µ(I) rheology, describing a relation between the confinement timescale and
the deformation timescale,   
$$
\bold I = \frac{\dot \gamma \bold d}{\sqrt{\bold P / \bold \rho}}
$$
Further, Jop et al. [14] proposed the following friction law for µ(I) to match
experimental results, where the friction begins from the static value µs at
low values of I and converges to a limit value µ2 at high values of I:  
$$
\mu(\bold I ) = \bold \mu_s + \frac{\bold \mu_2 - \bold \mu_s}{\bold I_0/\bold I + 1}
$$
The advantage of studying the collapse of a dry granular column by gravity onto a flat surface is that a granular transitional flow is produced that
may display the three different phase behaviors as the individual grains fall,
dissipate energy at the base of the column, and eject outwards to form an
outflow.  

***<u>*They observe that for large aspect ratios, the free fall dynamics induce*
*a collapse front which accelerates, then travels at constant velocity before*
*decelerating to a stop, proposing to relate the potential energy of the column*
to the problem dynamics, including the collapse duration as:***</u>  
$$
\bold T_{\infin} \approx 2.25\bold T_0 = 2.25\sqrt{(\frac{2\bold H_0}{\bold g})}
$$
![image-20211218094733390](E:\mycode\collection\定理\流体\image-20211218094733390.png)

![image-20211218094742676](E:\mycode\collection\定理\流体\image-20211218094742676.png)

Mass Scaling may be applied to scale the density and hence the critical timestep in slow-process problems, but is not applicable to the granular collapse
problem. Damping is available as an artificial method to reduce the energy
in the system, although its use in dynamic problems is not recommended.
Solowski and Sloan [20] experimented with the use of damping as a strategy
to match the experimental runout, noting that the Mohr-Coulomb model
did not sufficiently dissipate enough energy on its own. In this case a small
amount of damping (ie. 5%) might be appropriate to stabilize the stresses,
however we have not applied any damping in our results.  

As noted by others, the Mohr-Coulomb model does not sufficiently dissipate
energy in the problem and the runout is overestimated as compared to the
experimental results. The use of the local µ(I) model in mpm-2d with µs =
tan(φs) = tan(33) and µ2 = tan(φs + 12) gives results that are slightly closer
to those measured experimentally  

============Three-dimensional granular flow continuum modeling via
material point method with hyperelastic nonlocal granular
fluidity  



Plastic models can suffer from rate-independency
[33] and in some cases, they may have issues with modeling strain hardening [16]  

