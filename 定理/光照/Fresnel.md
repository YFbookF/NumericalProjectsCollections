==============Crash Course in BRDF Implementation  

How much light reflects away (or scatters into) the material is described by Fresnel equations.
Light incident under grazing angles is more likely to be reflected, which creates an effect sometimes called the “Fresnel reflections” (see Figure 4)  

resnel term 𝐹 determines how much light will be reflected off the surface, effectively telling
us how much light will contribute to evaluated BRDF. The remaining part (1 - 𝐹) will be passed to underlying material layer (e.g., the diffuse BRDF, or transmission BTDF). Our implementation so far only discusses two layers (specular and diffuse), but it is possible to create complex materials with many layers.   

=================

https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/

Dieletric-Conductor interface
$$
R_p = R_s\frac{\cos^2 \theta(a^2 + b^2) - 2a\cos \theta \sin^2 \theta + \sin^4 \theta}{\cos^2 \theta(a^2 + b^2) + 2a\cos \theta\sin^2 \theta + \sin^4\theta}
$$

```
float3 FresnelDieletricConductor(float3 Eta, float3 Etak, float CosTheta)
{  
   float CosTheta2 = CosTheta * CosTheta;
   float SinTheta2 = 1 - CosTheta2;
   float3 Eta2 = Eta * Eta;
   float3 Etak2 = Etak * Etak;

   float3 t0 = Eta2 - Etak2 - SinTheta2;
   float3 a2plusb2 = sqrt(t0 * t0 + 4 * Eta2 * Etak2);
   float3 t1 = a2plusb2 + CosTheta2;
   float3 a = sqrt(0.5f * (a2plusb2 + t0));
   float3 t2 = 2 * a * CosTheta;
   float3 Rs = (t1 - t2) / (t1 + t2);

   float3 t3 = CosTheta2 * a2plusb2 + SinTheta2 * SinTheta2;
   float3 t4 = t2 * SinTheta2;   
   float3 Rp = Rs * (t3 - t4) / (t3 + t4);

   return 0.5 * (Rp + Rs);
}
```

或者也可以这么写

//https://github.com/linusmossberg/monte-carlo-ray-tracer，n1和n2是折射率，costheta是角度，返回的reflectance R what we are looking for when we want to calculate the percentage of reflection and transmission

R 是反射比，T是透射比

https://zhuanlan.zhihu.com/p/303168568

<img src="E:\mycode\collection\定理\光照\image-20211111105523811.png" alt="image-20211111105523811" style="zoom:50%;" />

计算如下
$$
R_{\perp} = (\frac{\sin(\theta_t - \theta_i)}{\sin(\theta_t + \theta_i)})^2 \qquad T_{\perp} = 1 - R_{\perp} \\ R_{||} = (\frac{\tan(\theta_t - \theta_i)}{\tan(\theta_t + \theta_i)})^2 \qquad T_{||} = 1 - R_{||}
$$


```
// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
glm::dvec3 Fresnel::conductor(double n1, ComplexIOR* n2, double cos_theta)
{
    double cos_theta2 = pow2(cos_theta);
    double sin_theta2 = 1.0 - cos_theta2;

    glm::dvec3 eta2 = pow2(n2->real / n1);
    glm::dvec3 eta_k2 = pow2(n2->imaginary / n1);

    glm::dvec3 t0 = eta2 - eta_k2 - sin_theta2;
    glm::dvec3 a2_p_b2 = glm::sqrt(pow2(t0) + 4.0 * eta2 * eta_k2);
    glm::dvec3 t1 = a2_p_b2 + cos_theta2;
    glm::dvec3 t2 = 2.0 * cos_theta * glm::sqrt(0.5 * (a2_p_b2 + t0));
    glm::dvec3 r_perpendicual = (t1 - t2) / (t1 + t2); 

    glm::dvec3 t3 = cos_theta2 * a2_p_b2 + pow2(sin_theta2);
    glm::dvec3 t4 = t2 * sin_theta2;
    glm::dvec3 r_parallel = r_perpendicual * (t3 - t4) / (t3 + t4); 

    return (r_parallel + r_perpendicual) * 0.5;
}
```

非导体如下

**Dielectric-Dielectric interface**

An equivalent expression can be derive from the [1] formulation by setting kt = 0
$$
g = \sqrt{\eta + \cos^2 \theta - 1} \qquad R = \frac{(g-c)^2}{2(g+c)^2}(1+\frac{(c(g+c)-1)^2}{(c(g-c)+1)^2})
$$


```
// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
double Fresnel::dielectric(double n1, double n2, double cos_theta)
{
    double g2 = pow2(n2 / n1) + pow2(cos_theta) - 1.0;

    if (g2 < 0.0) return 1.0;

    double g = std::sqrt(g2);
    double g_p_c = g + cos_theta;
    double g_m_c = g - cos_theta;

    return 0.5 * pow2(g_m_c / g_p_c) * (1.0 + pow2((g_p_c * cos_theta - 1.0) / (g_m_c * cos_theta + 1.0)));
}

```

//https://github.com/peterkutz/GPUPathTracer

Fresnel的意义

```
		bool reflectFromSurface = (doSpecular && rouletteRandomFloat < computeFresnel(bestNormal, incident, incidentMedium.refractiveIndex, transmittedMedium.refractiveIndex, reflectionDirection, transmissionDirection).reflectionCoefficient);
		if (reflectFromSurface)
		{	ray reflected from the surface. }
		else if (bestMaterial.hasTransmission)
		{	 Ray transmitted and refracted. }
		else
		{	Ray did not reflect from the surface, so consider emission and take a diffuse sample. 	}
```

Fresnel 要传三个参数

```
__host__ __device__
	float3
	computeReflectionDirection(const float3 &normal, const float3 &incident)
{
	return 2.0 * dot(normal, incident) * normal - incident;
}

__host__ __device__
	float3
	computeTransmissionDirection(const float3 &normal, const float3 &incident, float refractiveIndexIncident, float refractiveIndexTransmitted)
{
	// Snell's Law:
	// Copied from Photorealizer.

	float cosTheta1 = dot(normal, incident);

	float n1_n2 = refractiveIndexIncident / refractiveIndexTransmitted;

	float radicand = 1 - pow(n1_n2, 2) * (1 - pow(cosTheta1, 2));
	if (radicand < 0)
		return make_float3(0, 0, 0); // Return value???????????????????????????????????????
	float cosTheta2 = sqrt(radicand);

	if (cosTheta1 > 0)
	{ // normal and incident are on same side of the surface.
		return n1_n2 * (-1 * incident) + (n1_n2 * cosTheta1 - cosTheta2) * normal;
	}
	else
	{ // normal and incident are on opposite sides of the surface.
		return n1_n2 * (-1 * incident) + (n1_n2 * cosTheta1 + cosTheta2) * normal;
	}
}

```

=============pbrt

The Fresnel equations describe the amount of light reflected from a surface;   
$$
r_{||} = \frac{\eta_t \cos \theta_i - \eta_i \cos \theta_t}{\eta_t \cos \theta_i + \eta_i \cos \theta_t}\\
r_{\perp} = \frac{\eta_i \cos \theta_i - \eta_t \cos \theta_t}{\eta_i \cos \theta_i + \eta_i \cos \theta_t}
$$
For For unpolarized light, the Fresnel reflectance is  
$$
F_r = \frac{1}{2} (r_{\perp} ^2 + r_{||}^2 )
$$
To find the cosine of the transmitted angle, cosThetaT, it is first necessary to determine if
the incident direction is on the outside of the medium or inside it, so that the two indices
of refraction can be interpreted appropriately.

The sign of the cosine of the incident angle indicates on which side of the medium the
incident ray lies (Figure 8.5). If the cosine is between 0 and 1, the ray is on the outside,
and if the cosine is between -1 and 0, the ray is on the inside. The parameters etaI and
etaT are adjusted such that etaI has the index of refraction of the incident medium, and
thus it is ensured that cosThetaI is nonnegative.  

```
// Fresnel Inline Functions
PBRT_CPU_GPU inline Float FrDielectric(Float cosTheta_i, Float eta) {
    cosTheta_i = Clamp(cosTheta_i, -1, 1);
    // Potentially flip interface orientation for Fresnel equations
    // if wo is inside , swap the eta and costheta
    if (cosTheta_i < 0) {
        eta = 1 / eta;
        cosTheta_i = -cosTheta_i;
    }

    // Compute $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
    Float sin2Theta_i = 1 - Sqr(cosTheta_i);
    Float sin2Theta_t = sin2Theta_i / Sqr(eta);
    if (sin2Theta_t >= 1)
        return 1.f;
    Float cosTheta_t = SafeSqrt(1 - sin2Theta_t);

    Float r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    Float r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
    return (Sqr(r_parl) + Sqr(r_perp)) / 2;
}
```

看看Fresnel算出的东西用到哪里去了吧？

```
Float R = FrDielectric(CosTheta(wo), eta), T = 1 - R;
        // Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
        Float pr = R, pt = T;
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            pr = 0;
        if (!(sampleFlags & BxDFReflTransFlags::Transmission))
            pt = 0;
        if (pr == 0 && pt == 0)
            return {};

        if (uc < pr / (pr + pt)) {
            // Sample perfect specular dielectric BRDF
            Vector3f wi(-wo.x, -wo.y, wo.z);
            SampledSpectrum fr(R / AbsCosTheta(wi));
            return BSDFSample(fr, wi, pr / (pr + pt), BxDFFlags::SpecularReflection);

        } else {
```

=========================Realistic Image Synthesis

For smooth homogenous metals and dielectrics the amount of light reflected can be derived from Maxwell`s equations, and the result is the Fresnel Equation.
