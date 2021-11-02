==============Crash Course in BRDF Implementation  

How much light reflects away (or scatters into) the material is described by Fresnel equations.
Light incident under grazing angles is more likely to be reflected, which creates an effect sometimes called the “Fresnel reflections” (see Figure 4)  

resnel term 𝐹 determines how much light will be reflected off the surface, effectively telling
us how much light will contribute to evaluated BRDF. The remaining part (1 - 𝐹) will be passed to underlying material layer (e.g., the diffuse BRDF, or transmission BTDF). Our implementation so far only discusses two layers (specular and diffuse), but it is possible to create complex materials with many layers.   

