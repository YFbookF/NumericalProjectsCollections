=================Linear-Time Dynamics using Lagrange Multipliers  

2.1 Why Reduced Coordinates?
There are certainly valid reasons for preferring a reduced-coordinate
approach over a multiplier approach. In particular, if the n d.o.f.’s
left to the system is very much smaller than the c d.o.f.’s removed
by the constraints, a reduced-coordinate approach is clearly called
for. Even if c and n are linearly related the use of generalized coordinates eliminates the “drifting” problem that multiplier methods
have. (For example, two links which are supposed to remain connected will have a tendency to drift apart somewhat when a multiplier approach is used.) Such drift is partly a consequence of numerical errors in computing the multipliers, but stems mostly from
the inevitable errors of numerical integration during the simulation.
Constraint stabilization techniques [4, 3] are used to help combat
this problem.2 The use of generalized coordinates eliminates this
worry completely, since generalized coordinates only express configurations which exactly satisfy the constraints. There is anecdotal
evidence that the use of generalized coordinates thus allows simulations to proceed faster, not because evaluation of the generalized
coordinate accelerations is faster, but because larger timesteps can
be taken by the integrator. This may well be true. More importantly,
for the case of articulated figures, we know that with a reducedcoordinate approach, linear-time performance is achievable.  