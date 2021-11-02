Non-iterative Computation of Contact Forces
for Deformable Objects  

Newton’s third law states that for two entities in contact, the force that the first entity exerts on the second
entity must be opposite equal to the force exerted by the
second entity on the first entity. As a consequence, the
sum of contact forces must always be 0. We denote this
situation as force equilibrium.  

![image-20211030145409636](D:\定理\连续介质力学\image-20211030145409636.png)

Incremental Potential Contact: Intersection- and Inversion-free, Large-Deformation Dynamics  

The addition of accurate friction with stiction only increases the
computational challenge for time stepping deformation [Wriggers
1995]. Friction forces are tightly coupled to the computation of both
deformation and the contact forces that prevent intersection. These
side conditions are generally formulated by their own governing
variational Maximal Dissipation Principle (MDP) [Goyal et al. 1991;
Moreau 1973] and thus introduce additional nonlinear, nonsmooth
and asymmetric relationships to dynamics. In transitions between
sticking and sliding modes large, nonsmooth jumps in both magnitude and direction are made possible by frictional contact model.
Asymmetry, in turn, is a direct consequence of MDP: frictional forces
are not uniquely defined by the velocities they oppose, and are also
determined by additional consistency conditions and constraints,
e.g., Coulomb’s law. One critical consequence is that there is no
well-defined potential that we can add to an IP to directly produce
contact friction via minimization.  

Strain Limiting for Soft Finger Contact Simulation  

![image-20211030164113700](D:\定理\连续介质力学\image-20211030164113700.png)