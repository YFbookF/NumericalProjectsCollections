Position-Free Monte Carlo Simulation for Arbitrary Layered BSDFs  

BSDF分为几个步骤，

第一步，采样

蒙特卡洛积分的基本形式
$$
\int_a^b f(x)dx \approx \frac{1}{N}\sum_{i=1}^N\frac{f(X_i)}{pdf(X_i)}
$$
路径追踪中的积分问题
$$
L_o(p,\omega_o) \approx \frac{1}{N}\sum_{i=1}^N\frac{L_i(p,w_i)f_r(p,w_i,w_o)(n\cdot w_i)}{p(\omega_i)}
$$
bsdf samplig，是根据概率密度分布函数，由ingoing方向绘制outgoing

https://github.com/linusmossberg/monte-carlo-ray-tracer

```
bool Interaction::sampleBSDF(glm::dvec3& bsdf_absIdotN, double& pdf, Ray& new_ray, bool flux) const
{
    new_ray = Ray(*this);

    glm::dvec3 wi = shading_cs.to(new_ray.direction);
    
    if ((new_ray.refraction && wi.z >= 0.0) || (!new_ray.refraction && wi.z <= 0.0))
    {
        // 直接返回radiance
        return false;
    }

    glm::dvec3 wo = shading_cs.to(out);
bsdf_absIdotN = BSDF(wo, wi, pdf, flux, new_ray.dirac_delta) * std::abs(wi.z);
    return pdf > 0.0;
}

```

wo是output，wi是input

注意最后几步

```
   
    double F = Fresnel::dielectric(n1, n2, cos_theta);

    double pdf_s, pdf_d;
    glm::dvec3 brdf_s = material->specularReflection(wi, wo, pdf_s);
    glm::dvec3 brdf_d = material->diffuseReflection(wi, wo, pdf_d);
        double pdf_t = pdf_s;
    glm::dvec3 btdf = brdf_s;
    if (F < 1.0)
    {
        btdf = material->specularTransmission(wi, wo, n1, n2, pdf_t, inside, flux);
    }
   if (wi_dirac_delta)
    {
        // wi guaranteed to be direction of ray spawned by interaction
        if (type == REFLECT)
        {
        	// 如果是反射的话，那么pdf就是反射比，分子就是高光反射
            pdf = R;
            return brdf_s * F;
        }
        else
        {
            pdf = T * (1.0 - R);
            return btdf * T * (1.0 - F);
        }
    }
    else if (!material->rough_specular)
    {
        pdf = pdf_d * (1.0 - R) * (1.0 - T);
        return brdf_d * (1.0 - F) * (1.0 - T);
    }
    pdf = glm::mix(glm::mix(pdf_d, pdf_t, T), pdf_s, R);
    return glm::mix(glm::mix(brdf_d, btdf, T), brdf_s, F);

```

R 是反射比，T是透射比，F是电介质常数

https://zhuanlan.zhihu.com/p/303168568

<img src="E:\mycode\collection\定理\光照\image-20211111105523811.png" alt="image-20211111105523811" style="zoom:50%;" />

计算如下
$$
R_{\perp} = (\frac{\sin(\theta_t - \theta_i)}{\sin(\theta_t + \theta_i)})^2 \qquad T_{\perp} = 1 - R_{\perp} \\ R_{||} = (\frac{\tan(\theta_t - \theta_i)}{\tan(\theta_t + \theta_i)})^2 \qquad T_{||} = 1 - R_{||}
$$


```
    if (material->rough_specular)
        specular_normal = shading_cs.from(material->visibleMicrofacet(shading_cs.to(out)));
    else
        specular_normal = shading_normal;

    // Specular reflect probability
    R = Fresnel::dielectric(n1, n2, glm::dot(specular_normal, out));

    // Transmission probability once ray has passed through specular layer
    T = material->transparency;

    // Ensure that both reflection and transmission has a chance to contribute for arbitrary wi, 
    // even if importance sampled ray spawned by interaction is e.g. total internal reflection.
    if (material->rough_specular)
    {
        R = glm::clamp(R, 0.1, 0.9);
    }
```

BSDF最后会更新到这里算throughput

```
        if (!interaction.sampleBSDF(bsdf_absIdotN, ls.bsdf_pdf, ray))
        {
            return radiance;
        }

        throughput *= bsdf_absIdotN / ls.bsdf_pdf;

/*
    double survive = glm::compMax(throughput) * ray.refraction_scale;

    if (survive == 0.0) return true;

    if (ray.diffuse_depth > min_ray_depth || ray.depth > min_priority_ray_depth)
    {
        survive = std::min(0.95, survive);

        if (!Random::trial(survive))
        {
            return true;
        }
        throughput /= survive;
    }
    return false;
*/

        if (absorb(ray, throughput))
        {
            return radiance;
        }
```

