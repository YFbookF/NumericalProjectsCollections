==================Fast Implicit Simulation of Flexible Trees  

A rod is modelled as a finite set of vertices 
$$
x^0 ,...,x^n \in \R^3
$$
together with edge torsion scalars 
$$
\psi^0,...,\psi^{n-1}
$$
Those are interlaced into an array of 4n + 3 degrees of freedom. An orthonormal reference frame is associated to each of the vertices 
$$
x^0,...,x^{n-1}
$$
. The first vector of each frame coincides with the tangent vector
$$
\tau = \frac{(x^{i+1} - x^{i})}{||x^{i+1} - x^i||}
$$
the second vector m is, initially, chosen arbitrarily2 in the normal plane and the third one n completes the orthonormal basis. As the rod vertices change in time, the reference frames evolve by parallel transport of the tangent vector.

The torsion degree of freedom records the angle between the material frame attached to each vertex and its reference frame. Thanks to this set-up, it is shown in [3] that the bending, twisting and stretching internal forces at any vertex depend only on itself and the neighbouring vertices and torsions. Thus the Jacobian ∇ f is  band-limited (width 3 + 1 + 3 + 1 + 3 = 11). This model allows anisotropic cross sections. Leaves, or grass blades for instance, can be simulated as very flat elliptic frustums.  

