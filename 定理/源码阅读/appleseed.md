有关光线追踪光照的概念介绍，各种文章，博客，书籍已经介绍得很不错了，比如physics based rendering。但是这些文章很多仅仅是给出一堆公式和图，有的不错的教程倒会给出代码，但也仅仅是自己的实现方向。如果想要博采众家之长，显然是不够的，因此我收集了一些开源渲染器中一些函数的实现方向，通过横向比较来发现它们的特点。

还有个问题，自学非常容易跑偏方向。所以非常需要这些开源渲染器的源码作为参考，避免自己犯一些低级错误。



本文大约

函数大致思路

sample

输入：给定outgoing光线的方向，给出光线的模型，即漫反射或镜面反射或其它

输出：所对应的可能的incoming的光线的方向

这个函数在很多开源渲染库中，都不仅仅是计算入射光线方向的，一般都是顺便把下面的evaluate函数的功能也实现了。毕竟sample函数本身很简单，漫反射就是半球cos采样，反射就是完美反射，折射用菲涅尔算算就行了。

evaluate

输入：给出outgoing和incoming两条光线，给出光照模型

输出：这一对光线的可能值，也就是这一对光线的brdf，或者说给出incoming光线的强度，打到表面后，还剩下多少会成为outgoing光线的强度

pdf 

输入：给定incoming光线的方向，表面法向量及其它参数，给定光照模型

输出：重要性采样中，这条incoming光线的重要性或者说是权重，也就是概率密度函数。也可以直接返回一，这样就是平均权重了，不过返回一的话蒙特卡洛积分就积不准了。

不说废话，直接来看首先是

## Diffuse/Lambertian

#### appleseed

appleseed库的diffusebtdf.cpp中的，要不要把判断函数加上来？

sample函数直接进行半球cos采样即可

```
Vector3f wi = sample_hemisphere_cosine(s);

```

evaluate函数的核心代码如下，说明appleseed库的diffuse材质其实机会

```

```

pdf函数核心代码如下，如果入射光线与出射光线在同一半球，那么成功发生漫反射，否则发生折射。但是此材质不允许折射存在，所以返回零。

```
const float cos_in = dot(incoming, n);
const float cos_on = dot(outgoing, n);
if (cos_in * cos_on < 0.0f){
// Return the probability density of the sampled direction.
return std::abs(cos_in) / PI;
}else{
// No transmission in the same hemisphere as outgoing.
return 0.0f;}
```

#### mitsuba库

mitsuba中，也很简单。但是值得一提的是，这里计算bsdf的时候，物体表面上每一点的m_reflectance 参数都是不同的。

```
wi = squareToCosineHemisphere();
f = m_reflectance->evalAtPoint(bRec.its) * (INV_PI * Frame::cosTheta(bRec.wo));
squareToCosineHemispherePdf = INV_PI * Frame::cosTheta(wi);
```

#### tungsten库

在LambertBsdf.cpp中，原理与之前一样????

```
event.wo  = SampleWarp::cosineHemisphere(event.sampler->next2D());
f = albedo(event.info)*INV_PI*event.wo.z();
if (!event.requestedLobe.test(BsdfLobes::DiffuseReflectionLobe))
        f = Vec3f(0.0f);
if (event.wi.z() <= 0.0f || event.wo.z() <= 0.0f)
        f = Vec3f(0.0f);
event.pdf = SampleWarp::cosineHemispherePdf(event.wo);
```

#### 

## Specular

applesedd库的specularbrdf.cpp中的sample函数是这么写的

第一行的公式哪里来的？妈的一行公式都看不懂

```
const float cos_theta_i = dot(outgoing.get_value(), shading_normal);
const float sin_theta_i2 = 1.0f - square(cos_theta_i);
const float sin_theta_t2 = sin_theta_i2 * square(values->m_precomputed.m_eta);
const float cos_theta_t2 = 1.0f - sin_theta_t2;
if (cos_theta_t2 < 0.0f){
	incoming = reflect(outgoing.get_value(), shading_normal); // 仅反射
}else{
    const float cos_theta_t = std::sqrt(cos_theta_t2); 
    float fresnel_reflection = fresnel_reflectance_dielectric( // 计算Fresnel参数
    	fresnel_reflection,1.0f / values->m_precomputed.m_eta,
    	std::abs(cos_theta_i),cos_theta_t);
    fresnel_reflection *= values->m_fresnel_multiplier;
    if (rand01() < fresnel_reflection){ // 如果一个随机数小于反射的可能性，那么仍然是反射
        incoming = reflect(outgoing.get_value(), shading_normal);
    }else{  // 否则计算折射方向
        const float eta = values->m_precomputed.m_eta; 
        incoming = cos_theta_i > 0.0f
                            ? (eta * cos_theta_i - cos_theta_t) * shading_normal - eta * outgoing.get_value()
                            : (eta * cos_theta_i + cos_theta_t) * shading_normal - eta * outgoing.get_value();
}}
```

关于Fresnel的那个函数可以看

evaluate函数和pdf函数都返回零。

## Reflection/Mirror

#### luxcore库

很简单的镜面反射，不过很多渲染库喜欢使用本地坐标，也许是这样方便一些？

```
*localSampledDir = Vector(-localFixedDir.x, -localFixedDir.y, localFixedDir.z);
fr =  Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
*pdfW = 1.f;
```

#### tungsten库

在MirrorBsdf.cpp中直接使用albedo值作为bsdf值。

```
event.wo = Vec3f(-event.wi.x(), -event.wi.y(), event.wi.z());
fr = albedo(event.info);
event.pdf = 1.0f;
```

#### yocto库

在yocto_shading.h中的sample_reflective()，有使用微表面模型和不使用的两种版本。不使用的版本与上面的一样，

微表面模型，所以在采样函数中需要使用微表面模型计算中间向量

```
sample_reflective = reflect(outgoing, up_normal);
eval_reflective = fresnel_conductor(
      reflectivity_to_eta(color), {0, 0, 0}, up_normal, outgoing);
sample_reflective_pdf = 1;
```

使用微表面模型的较为麻烦，请参见代码。

## Glossy材质

#### luxcore库



#### yocto库

在yocto_shading.h中的sample_glossy()，假设会发生镜面反射和漫反射。

sample函数中使用了微表面模型，因此计算镜面反射时，中间向量需要通过微表面法线分布来计算。

```
auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
if (rnl < fresnel_dielectric(ior, up_normal, outgoing)) {
    auto halfway  = sample_microfacet(roughness, up_normal, rn);
    auto incoming = reflect(outgoing, halfway);
    if (!same_hemisphere(up_normal, outgoing, incoming)) return {0, 0, 0};
    return incoming;
} else {
    return sample_hemisphere_cos(up_normal, rn); }
```

evaluate函数，同样使用微表面模型

```
return color * (1 - F1) / pif * abs(dot(up_normal, incoming)) +
	vec3f{1, 1, 1} * F * D * G /(4 * dot(up_normal, outgoing) * dot(up_normal, 
	incoming)) *abs(dot(up_normal, incoming));
```

pdf函数，同样使用微表面模型

```
return F * sample_microfacet_pdf(roughness, up_normal, halfway) /
             (4 * abs(dot(outgoing, halfway))) +
         (1 - F) * sample_hemisphere_cos_pdf(up_normal, incoming);
```

#### libYafaRay库

在material_glossy.cc库，同样假设会发生镜面反射和漫反射中一种，但是还考虑各项同性或各项异性的镜面反射，在一次采样中，镜面反射和漫反射会相互影响，镜面反射使用微表面模型，漫反射使用OrenNayar。

```
if(use_diffuse){
	if(s_1 < s_p_diffuse){
		wi = sample::cosHemisphere(n, sp.nu_, sp.nv_, s_1, s.s_2_);
		if(use_glossy){
		    // 使用带各项异性的微表面模型
			if(anisotropic_){
				s.pdf_ = s.pdf_ * cur_p_diffuse +asAnisoPdf() * (1.f - 
							cur_p_diffuse);
				glossy =asAnisoD() * schlickFresnel() / asDivisor();
			}else{
					s.pdf_ = s.pdf_ * cur_p_diffuse + blinnPdf()* (1.f - 
								cur_p_diffuse);
					glossy = blinnD() * schlickFresnel() / asDivisor();}
		scolor = glossy * getShaderColor();
		add_col = diffuseReflect();
		if(oren_nayar_){add_col *= orenNayar();}
		scolor += add_col;
		return scolor;
	}
```

pdf函数中，高光也会使用带各项异性的微表面模型，漫反射则是普通的漫反射

```
if(use_diffuse){
	pdf = std::abs(wi * n);
	if(use_glossy){
		if(anisotropic_){
			const Vec3 hs(h * sp.nu_, h * sp.nv_, cos_n_h);
			pdf = pdf * cur_p_diffuse + asAnisoPdf() * (1.f - cur_p_diffuse);
		}else pdf = pdf * cur_p_diffuse + blinnPdf() * (1.f - cur_p_diffuse);
	}
	return pdf;}
```



## Translucent/DiffuseTransmission

这种特殊的材质假设光线击中物体表面后，会透射过去并在相反的表面进行漫反射。

#### mitsuba库

在.cpp，仅认为光线会全部透射过去并在相反的表面进行漫反射。

sample函数中对于这个写得很清楚。这里用的是本地坐标，算起来很方便。

```
bRec.wo = warp::squareToCosineHemisphere(sample);
if (Frame::cosTheta(bRec.wi) > 0)bRec.wo.z *= -1;
```

evaluate函数与普通的漫反射没什么不同，仅仅是判断光线存在性时，如果入射光线与出射光线点积大于零，那么这次折射就不可能发生。

```
if (!(bRec.typeMask & EDiffuseTransmission) || measure != ESolidAngle
            || Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0)
            return Spectrum(0.0f);
return m_transmittance->eval(bRec.its)
            * (INV_PI * std::abs(Frame::cosTheta(bRec.wo)));
```

pdf函数也是与普通漫反射一样

```
return std::abs(Frame::cosTheta(bRec.wo)) * INV_PI;
```

#### tungsten库

在DiffuseTransmissionBsdf.cpp，认为光线可能会透射，也可能不会。但之后都会发生漫反射。

sample函数

```
event.wo = SampleWarp::cosineHemisphere(event.sampler->next2D());
event.wo.z() = std::copysign(event.wo.z(), event.wi.z());
if (transmit)event.wo.z() = -event.wo.z();
```

evaluate函数，在光线没有透射的时候，为了能量守恒，因子是1 - T。

```
float factor = event.wi.z()*event.wo.z() < 0.0f ? _transmittance : 1.0f - _transmittance;
return albedo(event.info)*factor*INV_PI*std::abs(event.wo.z());
```

pdf函数，同样在光线没透射的时候，要遵循能力守恒原则。

```
float factor = event.wi.z()*event.wo.z() < 0.0f ? transmittanceProbability : 1.0f - transmittanceProbability;
return factor*SampleWarp::cosineHemispherePdf(event.wo);
```

#### yocto库

采样函数就是简单的cos半球采样

```
auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
return sample_hemisphere_cos(-up_normal, rn);
//注意与下面的标准漫反射采样对比，半透明材质用的是反的法向量
//sample_hemisphere_cos(up_normal, rn);
```

evaluate采样

```
if (dot(normal, incoming) * dot(normal, outgoing) >= 0) return {0, 0, 0};
//注意与下面的标准漫反射合法性检测对比
//if (dot(normal, incoming) * dot(normal, outgoing) <= 0) return {0, 0, 0};
return color / pif * abs(dot(normal, incoming));
```

概率密度函数

```
if (dot(normal, incoming) * dot(normal, outgoing) >= 0) return 0;
auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
return sample_hemisphere_cos_pdf(-up_normal, incoming);
```



## Metal材质

#### luxcore库

对于金属材质，luxcore用的是微表面模型。入射光线为镜面反射，不过反射时需要用的的中间向量h来源于shclick模型。代码在metal2.cpp中。

```
SchlickDistribution_SampleH(roughness, anisotropy, u0, u1, &wh, &d, &specPdf);
const float cosWH = Dot(localFixedDir, wh);
*localSampledDir = 2.f * cosWH * wh - localFixedDir;
```

计算bsdf的时候，然后是哪个微表面模型？

```
float factor = (d / specPdf) * G * fabsf(cosWH);
if (!hitPoint.fromLight) factor /= coso;
else factor /= cosi;
fr =  factor * F;
```

然后算概率密度的时候，主要由三个参数控制，分别是粗糙度roughness，中间向量wh，以及各项异性度anisotropy

```
cosTheta = fabsf(wh.z);
const float h = sqrtf(wh.x* wh.x+ wh.y * wh.y);
SchlickA = 1;
if (h > 0.f) {
		const float w = (anisotropy > 0.f ? H.x : H.y) / h;
		const float p = 1.f - fabsf(anisotropy);
		SchlickA = sqrtf(p / (p * p + w * w * (1.f - p * p)));
}
SchlickZ = 0
if (roughness > 0.f) {
		const float cosNH2 = cosTheta * cosTheta;
		// expanded for increased numerical stability
		const float d = cosNH2 * roughness + (1.f - cosNH2);
		// use double division to avoid overflow in d*d product
		SchlickZ =  (roughness / d) / d;
}
pdf = SchlickZ * SchlickA;
```



## Conductor材质

与电解质相反，导体不透射任何光线，所以在mitsuba中，虽然也使用了Fresnel导体函数，但是透射部分并未出现。

#### mitsuba

sample函数中入射光线为标准镜面反射。

evaluate函数中，只返回一种可能值，即物体表面被击中点的高光反射率，乘上Fresnel导体系数。没有透射或折射影响。

```
return m_specularReflectance->eval(bRec.its) *
            fresnelConductorExact(Frame::cosTheta(bRec.wi), m_eta, m_k);
```

pdf函数中，判断以一光线是否不太离谱，然后返回0或1。

```
if (!sampleReflection || measure != EDiscrete ||Frame::cosTheta(bRec.wi) <= 0 ||
Frame::cosTheta(bRec.wo) <= 0 ||std::abs(dot(reflect(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
       return 0.0f;
return 1.0f;
```

#### tungsten库

在ConductorBsdf.cpp中，采样函数仍然用本地坐标

```
event.wo = Vec3f(-event.wi.x(), -event.wi.y(), event.wi.z());
```

evaluate函数中，用的是材质的albedo乘上菲涅尔导体系数

```
event.weight = albedo(event.info)*Fresnel::conductorReflectance(_eta, _k, event.wi.z());
```

pdf函数中返回1。

## Dielectric材质

与导体相反，电解质既反射光线又折射光线。常见的电解质表面例如空气与水。

#### mitsuba

概率判断，先判断入射光线方向和出射光线方向是否一致，如果一致则发生高光反射，否则发射折射。

sample函数中，如果发生反射，那么生成镜面反射入射光线，如果发生折射，根据index算就行了。

```
/// Refraction in local coordinates
inline Vector refract(const Vector &wi, Float cosThetaT) const {
        Float scale = -(cosThetaT < 0 ? m_invEta : m_eta);
        return Vector(scale*wi.x, scale*wi.y, cosThetaT);
}
```

evaluate函数中，那么直接让高光反射率乘上入射光线的Fresnel函数即可。如果发生折射，则还需要乘上缩放系数。

```
if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0) {
    return m_specularReflectance->eval(bRec.its) * F;
} else {
    /* Radiance must be scaled to account for the solid angle compression
    that occurs when crossing the interface. */
    Float factor = (bRec.mode == ERadiance)? (cosThetaT < 0 ? m_invEta : m_eta) : 1.0f;
    return m_specularTransmittance->eval(bRec.its)  * factor * factor * (1 - F);}
```

pdf函数中，mitsuba库比较特殊，认为高光反射和折射可能同时存在，虽然实际上只能采样一种，但还是要将它们乘上系数菲涅尔系数。

```
if (sampleTransmission && sampleReflection) {
            if (sample.x <= F) 	{pdf = F;}
            else				{pdf = 1 - F;}
else if (sampleReflection) 		{pdf = 1;}
else if	(sampleTransmission)	{pdf = 1;}
```

#### tungesten库

概率判断，算计算Fresnel反射率。如果表面上这一点既允许发生高光反射又允许折射，那么最终反射率就是Fresnel反射率。如果仅允许发射高光，那么反射率为1，反之为零。

sample函数，反射和折射用的是本地坐标

```
float F = Fresnel::dielectricReflectance(eta, std::abs(event.wi.z()), cosThetaT);
float reflectionProbability;
if (sampleR && sampleT)reflectionProbability = F;
else if (sampleR)reflectionProbability = 1.0f;
else if (sampleT) reflectionProbability = 0.0f;
else return false;
if (event.sampler->nextBoolean(reflectionProbability)) {
        event.wo = Vec3f(-event.wi.x(), -event.wi.y(), event.wi.z());
}else{
		event.wo = Vec3f(-event.wi.x()*eta, -event.wi.y()*eta, 
		-std::copysign(cosThetaT, event.wi.z()));}
```

evaluate函数，也是判断入射光线与出射光线的角度。这里的高光反射率叫做albedo。

```
if (event.wi.z()*event.wo.z() >= 0.0f) {
        if (evalR && checkReflectionConstraint(event.wi, event.wo))
            return F*albedo(event.info);
        else return Vec3f(0.0f);
} else {
        if (evalT && checkRefractionConstraint(event.wi, event.wo, eta, cosThetaT))
            return (1.0f - F)*albedo(event.info);
        else return Vec3f(0.0f);}
```

pdf函数很简单，与反射概率呈线性关系

```
if (event.sampler->nextBoolean(reflectionProbability)) {
	event.pdf = reflectionProbability;
}else{
	event.pdf = 1.0f - reflectionProbability;
}
```



## RoughDielectric

这种材质常用的表面是粗糙的电介质，如空气与玻璃。

mitsuba

使用了微表面模型的材质

sample函数中，入射光线分为反射与折射。

evaluate函数中，用的是微表面模型。折射部分参考Microfacet Models for Refraction through Rough Surfaces'' by Walter et al.

```
/* Evaluate the microfacet normal distribution */
const Float D = distr.eval(H);
 if (D == 0)return Spectrum(0.0f);
const Float F = fresnelDielectricExt(dot(bRec.wi, H), m_eta);
/* Smith's shadow-masking function */
const Float G = distr.G(bRec.wi, bRec.wo, H);
if (reflect) {
	return m_specularReflectance->eval(bRec.its) *  F * D * G /
                (4.0f * std::abs(Frame::cosTheta(bRec.wi)));
} else {
    Float eta = Frame::cosTheta(bRec.wi) > 0.0f ? m_eta : m_invEta;
	/* Calculate the total amount of transmission */
    Float sqrtDenom = dot(bRec.wi, H) + eta * dot(bRec.wo, H);
    Float value = ((1 - F) * D * G * eta * eta* dot(bRec.wi, H) * dot(bRec.wo, H)) /
                (Frame::cosTheta(bRec.wi) * sqrtDenom * sqrtDenom);
	/* Missing term in the original paper: account for the solid angle compression 	
		when tracing radiance -- this is necessary for bidirectional methods */
    Float factor = (bRec.mode == ERadiance)
    ? (Frame::cosTheta(bRec.wi) > 0 ? m_invEta : m_eta) : 1.0f;
return m_specularTransmittance->eval(bRec.its)* std::abs(value * factor * factor);}
```



## Matte

matte，sheen和shiny是三种布料的材质。

matte类似于粗布，粗表面，没有反射高光。以审美的角度来看，如果你想让一件物体不太被人注意，隐藏起来，就可以使用matte材质。

sheen则是反射一点高光的布料。以审美的角度来看，如果你想让一件物体能够被人注意，就可以使用sheen材质。根据https://insideoutstyleblog.com/2008/10/fabric-should-i-choose-matte-sheen-or-shine.html，如果你觉得自己今天的脸蛋感到比较满意，就可以穿一条sheen材质的领带。那么大家就会被高光吸引，进而留意到你的惊世骇俗的脸蛋了。

shiny则是反射很多高光的布料。如果你想让一件物体快速被人注意，成为舞台上的主角，就可以使用shiny材质。不过显然，shiny材质用的太多也不好。

不过在pbrt第三版说，可以一个参数sigma来控制材质的粗糙程度，如果sigma为零，那么就是lambetian反射模型，否则就是OrenNayar模型。不过也有很多开源渲染引擎并没有这个控制参数，而是直接使用了OrenNayar模型，但是材质名字仍然叫matte。

## Sheen

首先看看appleseed中的sheen

sample函数中，入射光线方向是在半球上的uniform采样。注意之前diffusebtdf是cosine采样，为什么？

```
const Vector2f s = rand01<Vector2f>();
const Vector3f wi = sample_hemisphere_uniform(s);
```

evalute和pdf函数都是返回1 / pi

```
return  1 / M_PI;
```

注意

#### blender

在bsdf_principled_sheen.h中，采样是直接使用半球采样的。evaluate计算方式很简单

```
float NdotL = dot(N, L);
float NdotV = dot(N, V);
if (NdotL < 0 || NdotV < 0) {
	*pdf = 0.0f;
    return make_float3(0.0f, 0.0f, 0.0f);
}
*pdf = fmaxf(dot(N, L), 0.0f) * M_1_PI_F;
float LdotH = dot(L, H);
float value = schlick_fresnel(LdotH) * NdotL;
return make_float3(value, value, value);
```



## Plasticity

#### appleseed

首先是sample中

```
const float alpha = microfacet_alpha_from_roughness(values->m_roughness);
//return std::max(0.001f, roughness * roughness);
// 菲涅尔方程会告诉我们被反射的光线所占入射光线的百分比
const float Fo = fresnel_reflectance(wo, m, values->m_precomputed.m_eta);
// 相当于F 再乘一个系数，就是塑料材质下高光所占比
const float specular_weight = Fo * m_specular_weight;
const float diffuse_weight = (1.0f - Fo) * m_diffuse_weight;
const float total_weight = specular_weight + diffuse_weight;
const float specular_probability  == 0.0f ? 1.0f : specular_weight / total_weight;
//微表面法向量，用的微表面模型中的GGX

//如果当前应该高光反射
if(rand01() < specular_probability){
	const Vector3f m = GGXMDF::sample(wo, Vector2f(rand01(), rand01()), alpha);
	wi = reflect(wo,m);
	//计算高光反射的概率密度函数
	const float cos_wom = dot(wo, m);
	const float jacobian = 1.0f / (4.0f * std::abs(cos_wom));
	const float probability = jacobian * GGXMDF::pdf(wo, m, alpha, alpha) 											*specular_probability;
	//如果确实会发生高光反射
	if (probability > 1.0e-6f){
		 const float denom = std::abs(4.0f * wo.y * wi.y);
         const float D = GGXMDF::D(m, alpha);
         const float G = GGXMDF::G(wi, wo, m, alpha);
         m_beauty = specular_reflectance * Fo * D * G / denom;
    }
}else{ //如果当前应该漫反射
	wi = sample_hemisphere_cosine(Vector2f(s[0], s[1]));
	// 计算漫反射的概率密度函数
	const float probability = wi.y / M_PI * (1.0f - specular_probability);
	// 如果确实会发生漫反射
	if (probability > 1.0e-6f){
		const float Fi = fresnel_reflectance(wi, m, values->m_precomputed.m_eta);
		const float eta2 = square(eta);
            const float fdr = fresnel_internal_diffuse_reflectance(1.0f / eta);
            const float T = (1.0f - Fo) * (1.0f - Fi);
            for (size_t i = 0, e = Spectrum::size(); i < e; ++i)
            {
                const float pd = diffuse_reflectance[i];
                const float non_linear_term = 1.0f - lerp(1.0f, pd, internal_scattering) * fdr;
                m_beauty[i] = (T * pd * eta2 / M_PI) / non_linear_term;
            }
	}
}
```



然后在第四行计算出射光线的菲涅尔系数。其实appleseed库的菲涅尔系数计算很简单。其它库有更全面的，所以之后再说。

同样，当光线照射到塑料材质上，可能发生高光反射，也可能发生漫反射，这取决于塑料物体自身的参数。在appleseed中这些参数包括高光反射权重m_specular_weight以及漫反射权重，在上面代码第6行到第8行可看到它们的作用。

高光反射的话，入射光线方向仍然用简单的反射函数计算。本次高光反射是否真的会发生，还取决于概率密度函数是否大于零。如果真的发生了，那么就利用微表面模型计算m_beauty值。用的是简单的Cook-Torrance reflectance model ：
$$
\rho = \frac{FDG}{4\cos \theta_i \cos \theta_o}
$$
m_beauty最终会被转换为m_throughput值，这个值最终要作为灯光的强度的参数计算，你也可以用vscode打开appleseed的源代码文件夹然后去搜索下面的代码

```
vertex.m_throughput *= sample.m_value.m_beauty;
```

所以现在暂时先不管。这个值也就是pbrt第三版第8.1.1节的reflectance的rho值，也就是光线真正被反射的参数。在learnopengl的pbr理论那一章，每个像素着色器颜色的计算方式要用到
$$
L_o(p,\omega_o) = \int\limits_{\Omega} 
    	(k_d\frac{c}{\pi} + k_s\frac{DFG}{4(\omega_o \cdot n)(\omega_i \cdot n)})
    	L_i(p,\omega_i) n \cdot \omega_i  d\omega_i
$$
这样就熟悉多了。而塑料的材质的漫反射，首先具体理论请参见appleseed库所参考的论文A Physically-Based Reflectance Model Combining
Reflection and Diffraction 。总之第一层透明层的transmission计算如下
$$
T = (1 - F(\theta_i))(1 - F(\theta_o))
$$
然后计算reflectance
$$
\rho = T\frac{\rho_d}{\pi}(1 - \frac{\rho_d}{\pi}F_{dr}(\frac{1}{\eta}))^{-1}
$$
这样就和之前的代码对上了。不过这里还是将每个spectrum分开计算了，为什么？

而evaluate函数和pdf函数返回的都是概率密度，该高光高光，该漫反射漫反射，计算方法和上面的一样

```
jacobian = 1.0f / (4.0f * std::abs(cos_wom));
specular_pdf =  jacobian * GGXMDF::pdf(wo, m, alpha, alpha);
diffuse_pdf = std::abs(wi.y) / M_PI;
```

#### mitsuba库

sample函数中，仍然是漫反射是标准的半球cos采样，高光反射是完美镜面反射。

evaluate函数中，如果某次采样中，既可能发射高光反射又可能发生漫反射，就要计算高光反射发生的可能性。计算高光可能性时，用的仍然是和appleseed库一样的方法。最后漫反射的时候注意计算非线性项，在appleseed中叫做nonlinear_term，在mitsuba中叫做diff。

如果既可能发射高光反射又可能发生漫反射，那么如果本次采样计算高光的话，还要让最终bsdf值除以高光发生的概率，这在别的库中是没有的。

```
if (hasDiffuse && hasSpecular) {
	Float probSpecular = (Fi*m_specularSamplingWeight) /
          (Fi*m_specularSamplingWeight +(1-Fi) * (1-m_specularSamplingWeight));
	if (sample.x < probSpecular) {
		return m_specularReflectance->eval(bRec.its)* Fi / probSpecular;
    } else {
    	Float Fo = fresnelDielectricExt(Frame::cosTheta(bRec.wo), m_eta);
	 	Spectrum diff = m_diffuseReflectance->eval(bRec.its);
        if (m_nonlinear)	diff /= Spectrum(1.0f) - diff*m_fdrInt;
        else  diff /= 1 - m_fdrInt;
		return diff * (m_invEta2 * (1-Fi) * (1-Fo) / (1-probSpecular));
    }
```

pdf函数基本上就是高光反射的概率。

```
if (Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0)
    {   pdf = 0.0f; return;}
Float Fi = fresnelDielectricExt(Frame::cosTheta(bRec.wi), m_eta);
pdf = (Fi*m_specularSamplingWeight) /
	(Fi*m_specularSamplingWeight +(1-Fi) * (1-m_specularSamplingWeight));
```

#### tungsten

在PlasticBsdf.cpp，概率判断使用电介质Fresnel乘上权重

```
float Fi = Fresnel::dielectricReflectance(eta, wi.z());
float substrateWeight = _avgTransmittance*(1.0f - Fi);
float specularWeight = Fi;
if (sampleR && sampleT)
        specularProbability = specularWeight/(specularWeight + substrateWeight);
else if (sampleR)	specularProbability = 1.0f;
else if (sampleT)	specularProbability = 0.0f;
else	return false;
```

sample函数中，仍然是漫反射是标准的半球cos采样，高光反射是完美镜面反射。

evaluate函数与之前两个库是一样的，不过参数名称不一样。

```
float Fi = Fresnel::dielectricReflectance(eta, event.wi.z());
float Fo = Fresnel::dielectricReflectance(eta, event.wo.z());
if (evalR && checkReflectionConstraint(event.wi, event.wo)) {
    return Vec3f(Fi);
} else if (evalT) {
    Vec3f diffuseAlbedo = albedo(event.info);
	Vec3f brdf = ((1.0f - Fi)*(1.0f - Fo)*eta*eta*event.wo.z()*INV_PI)*
                (diffuseAlbedo/(1.0f - diffuseAlbedo*_diffuseFresnel));
	if (_scaledSigmaA.max() > 0.0f)
       	brdf *= std::exp(_scaledSigmaA*(-1.0f/event.wo.z() -1.0f/event.wi.z()));
    return brdf;
} else {return Vec3f(0.0f);}
```

pdf函数仍然有五种可能，如果需要同时采样反射和透射，那么就和Fresnel参数有关，否则就是简单的半球采样或1和0。

```
if (sampleR && sampleT) {
   if (checkReflectionConstraint(event.wi, event.wo))return specularProbability;
   else	return SampleWarp::cosineHemispherePdf(event.wo)*(1.0f - 		
    		specularProbability);
} else if (sampleT) { return SampleWarp::cosineHemispherePdf(event.wo);
					//return std::abs(wo.z())*INV_PI;
} else if (sampleR) {
	return checkReflectionConstraint(event.wi, event.wo) ? 1.0f : 0.0f;
} else {	return 0.0f;}
```

#### yocto

在yocto_shading.h的sample_transparent()，光线碰到表面后可能会镜面反射，或直接被弹回去。这有点不太像塑料了，所以名称为透明物体。

采样有两种可供选择，包括微表面采样和普通采样。微表面采样如下

```
if (rnl < fresnel_dielectric(ior, halfway, outgoing)) {
    auto incoming = reflect(outgoing, halfway);
} else {
    auto reflected = reflect(outgoing, halfway);
    auto incoming  = -reflect(reflected, up_normal);
```

普通采样的入射光线如下

```
if (rnl < fresnel_dielectric(ior, up_normal, outgoing)) {
    return reflect(outgoing, up_normal);
} else { return -outgoing;}
```

这么一看显然是前者更加真实一些了。不使用微表面模型的话，evaluate函数计算如下

```
if (dot(normal, incoming) * dot(normal, outgoing) >= 0) {
    return vec3f{1, 1, 1} * fresnel_dielectric(ior, up_normal, outgoing);
  } else {
    return color * (1 - fresnel_dielectric(ior, up_normal, outgoing));
  }
```

概率密度函数如下

```
if (dot(normal, incoming) * dot(normal, outgoing) >= 0) {
    return fresnel_dielectric(ior, up_normal, outgoing);
  } else {
    return 1 - fresnel_dielectric(ior, up_normal, outgoing);
  }
```



## Ashikhmin-Shirley



```
bool is_diffuse = rand01() < diffuse_weight;
if(is_diffuse)
{
	incoming = sample_hemisphere_cosine(Vector2f(rand01(), rand1())); // 入射光线方向
	h = normalize(incoming + outgoing.get_value()); // halfway vector
}else{
	if(isotropic)
	{
		exp = m_nu;
	}else
	{
		const float phi = sample_anisotropic_glossy(m_k, s[0]);
		exp = m_nu * cos_phi * cos_phi + m_nv * sin_phi * sin_phi;
	}
	const float cos_theta = std::pow(1.0f - rand01(), 1.0f / (exp + 1.0f));
    const float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);
    h = local_geometry.m_shading_basis.transform_to_parent(
    Vector3f::make_unit_vector(cos_theta, sin_theta, cos_phi, sin_phi));
	incoming = reflect(outgoing.get_value(), h); 
}
const float cos_in = std::abs(dot(incoming, normal));
const float cos_on = std::abs(dot(outgoing, normal));
const float cos_oh = std::min(std::abs(dot(outgoing, h)), 1.0f);
const float cos_hn = std::abs(dot(h, normal));
if(is_diffuse)
{
	const float a = 1.0f - pow5(1.0f - 0.5f * cos_in);
    const float b = 1.0f - pow5(1.0f - 0.5f * cos_on);
    sample_diffuse = mat_kd * a * b;
    diffuse_pdf = cos_in / M_PI;
}else
{
	onst float num = mat_kg * std::pow(cos_hn, exp);
    const float den = cos_oh * (cos_in + cos_on - cos_in * cos_on);
    sample_glossy = fresnel_reflectance_dielectric_schlick() * num / den;
    glossy_pdf = num / den;
}
sample_beauty = sample_diffuse + sample_glossy;
```

方向的选择依然是半球采样和完美高光反射，不过为了计算reflectance，也就是rho，在计算各项异性高光反射时需要花点心思。在appleseed库提到的参考论文中，高光反射率是这么算的
$$
\rho_s(\bold k_1,\bold k_2) = \frac{\sqrt{(n_u + 1)(n_v + 1)}}{8\pi}\frac{(\bold n \cdot \bold h)*term}{(\bold h \cdot \bold k)\max((\bold n \cdot \bold k_1),(\bold n \cdot \bold k_2))}F((\bold k,\bold h))
$$
对于各项异性来说
$$
term = \frac{n_u(\bold h \bold u)^2 + n_v(\bold h \bold v)^2}{1 - (\bold h \bold n)^2}
$$
对于各项同性来说
$$
term = n_u
$$
其中n_u和n_v都是参数，n_u越大，代表高光在水平方向上越狭窄，n_v越大代表在竖直方向上越狭窄。

<img src="E:\mycode\collection\定理\源码阅读\image-20211121141818155.png" alt="image-20211121141818155" style="zoom: 67%;" />

而evaluate函数和pdf函数返回的都是概率密度，计算方法和上面的一样

```
const float probability = diffuse_weight * pdf_diffuse + glossy_weight * pdf_glossy;
return probability;
```

#### blender

在bsdf_ashikhmin_shirley.h中，

```
if (n_x == n_y) {/* isotropic */
      float e = n_x;
      float lobe = powf(HdotN, e);
      float norm = (n_x + 1.0f) / (8.0f * M_PI_F);
      out = NdotO * norm * lobe * pump;
      /* this is p_h / 4(H.I)  (conversion from 'wh measure' to 'wi measure', eq. 				8 in paper). */
      *pdf = norm * lobe / HdotI;
}else {/* anisotropic */
      if (HdotN < 1.0f) {
        float e = (n_x * HdotX * HdotX + n_y * HdotY * HdotY) / (1.0f - HdotN * HdotN);
        lobe = powf(HdotN, e);}
      else {lobe = 1.0f; }
      float norm = sqrtf((n_x + 1.0f) * (n_y + 1.0f)) / (8.0f * M_PI_F);
      out = NdotO * norm * lobe * pump;
      *pdf = norm * lobe / HdotI;
    }
```



## Azimuth

Azimuthal scattering function. Assumes perfectly smooth reflection in

tungsten中RoughWireBcsdf.cpp

## Blinn-Phong

#### appleseed

采样随便采采

```
    const float cos_theta = std::pow(1.0f - rand01(), 1.0f / (alpha_x + 2.0f));
    const float sin_theta = std::sqrt(std::max(0.0f, 1.0f - square(cos_theta)));
    phi = TwoPi<float>() * rand01();
    wi = Vector3(cos_phi * sin_theta,cos_theta,sin_phi * sin_theta);
```

evaluate和pdf函数则根据Blinn微表面模型计算概率密度

```
//wo是出射光线方向，m是微表面模型的法线方向
cos_vm = dot(wo, m);
G1 = std::min(1.0f, 2.0f * std::abs(m.y * wo.y / cos_vm));
D = (rand01() + 2.0f) * std::pow(std::abs(m.y), rand01()) * 0.5 / M_PI;
pdf_visible_normals = G1 * D
pdf = pdf_visible_normals;
```

#### tungsten

在PhongBsdf.cpp，假设材质可能需要采样glossy和diffuse，因此采样光线方向略有不同

```
if (sampleGlossy) {
	Vec2f xi = event.sampler->next2D();
	float phi      = xi.x()*TWO_PI;
    float cosTheta = std::pow(xi.y(), _invExponent);
    float sinTheta = std::sqrt(max(0.0f, 1.0f - cosTheta*cosTheta));
	Vec3f woLocal(std::cos(phi)*sinTheta, std::sin(phi)*sinTheta, cosTheta);
} else {
    event.wo = SampleWarp::cosineHemisphere(event.sampler->next2D());
    event.sampledLobe = BsdfLobes::DiffuseReflectionLobe;
}
```

evaluate函数也会根据是采样diffuse还是glossy来决定最终值。

```
float result = 0.0f;
if (evalDiffuse)	result += _diffuseRatio*INV_PI;
if (evalGlossy) {	 float cosTheta = Vec3f(-event.wi.x(), -event.wi.y(), 
									event.wi.z()).dot(event.wo);
if (cosTheta > 0.0f) result += std::pow(cosTheta, _exponent)*_brdfFactor*(1.0f - 
								_diffuseRatio);}
return albedo(event.info)*event.wo.z()*result;
```

pdf函数也是如此

```
float result = 0.0f;
if (evalGlossy) {	
	float cosTheta = Vec3f(-event.wi.x(), -event.wi.y(), 
									event.wi.z()).dot(event.wo);
	if (cosTheta > 0.0f) result += std::pow(cosTheta, _exponent)*_pdfFactor;}
if (evalDiffuse && evalGlossy)
        result = result*(1.0f - _diffuseRatio) + 
        		_diffuseRatio*SampleWarp::cosineHemispherePdf(event.wo);
else if (evalDiffuse)
        result = SampleWarp::cosineHemispherePdf(event.wo);
return result;
```



## 薄膜ThinFilm材质

这假设材质为单层非常薄的材质，内部仍然可以反射和折射，甚至会有光谱干涉的发生。

#### tungsten库

在ThinSheetBsdf.cpp中，假设材质为单层非常薄的材质，内部仍然可以反射和折射，甚至会有光谱干涉的发生。

sample函数中，入射光线为本地反射

```
event.wo = Vec3f(-event.wi.x(), -event.wi.y(), event.wi.z());
```

evaluate函数，使用的叫transmittance，所以我搞不清楚到底是反射还是透射。

```
if (_enableInterference) {
        transmittance = 1.0f - Fresnel::thinFilmReflectanceInterference(1.0f/_ior,
                std::abs(event.wi.z()), thickness*500.0f, cosThetaT);
 else {
        transmittance = Vec3f(1.0f - Fresnel::thinFilmReflectance(1.0f/_ior, std::abs(event.wi.z()), cosThetaT));}
if (_sigmaA != 0.0f && cosThetaT > 0.0f)
        transmittance *= std::exp(-_sigmaA*(thickness*2.0f/cosThetaT));
return transmittance;
```

上面的薄膜Fresnel计算较为复杂，参加库中的Fresnel.hpp和http://www.gamedev.net/page/resources/_/technical/graphics-programming-and-theory/thin-film-interference-for-computer-graphics-r2962。



## Disney

经典的迪士尼采样，入射光线仍然是半球cosine采样。其中evaluate函数使用了disney的BRDF explorer库里的Hanrahan-Krueger BRDF来模拟各项同性

```
const float cos_on = dot(n, outgoing);
const float cos_in = dot(n, incoming);
const float cos_ih = dot(incoming, h);
const float fl = schlick_fresnel(cos_in);
const float fv = schlick_fresnel(cos_on);
float fd = 0.0f;
if (values->m_subsurface != 1.0f)
{
       const float fd90 = 0.5f + 2.0f * square(cos_ih) * values->m_roughness;
       fd = mix(1.0f, fd90, fl) * mix(1.0f, fd90, fv);
}
if (values->m_subsurface > 0.0f)
{
    // Based on Hanrahan-Krueger BRDF approximation of isotropic BSRDF.
    // The 1.25 scale is used to (roughly) preserve albedo.
    // Fss90 is used to "flatten" retroreflection based on roughness.
    const float fss90 = square(cos_ih) * values->m_roughness;
    const float fss = mix(1.0f, fss90, fl) * mix(1.0f, fss90, fv);
    const float ss = 1.25f * (fss * (1.0f / (std::abs(cos_on) + std::abs(cos_in)) - 0.5f) + 0.5f);
    fd = mix(fd, ss, values->m_subsurface);
}
value = m_base_color * fd / M_PI * (1.0f - values->m_metallic);
```

概率密度函数仍然是简单的漫反射概率密度函数，即

```
pdf = abs(cos_in) / M_PI;
```

#### blender

在bsdf_principled_diffuse.h中，入射光线使用cos半球采样。

evaluate函数请参考"Physically Based Shading at Disney" (2012)与"Extending the Disney BRDF to a BSDF with Integrated Subsurface Scattering" (2015)。

```
const float FV = schlick_fresnel(NdotV);
const float FL = schlick_fresnel(NdotL);
float f = 0.0f;
/* Lambertian component. */
if (bsdf->components & (PRINCIPLED_DIFFUSE_FULL | PRINCIPLED_DIFFUSE_LAMBERT)) {
    f += (1.0f - 0.5f * FV) * (1.0f - 0.5f * FL);
}else if (bsdf->components & PRINCIPLED_DIFFUSE_LAMBERT_EXIT) {
    f += (1.0f - 0.5f * FL);}
/* Retro-reflection component. */
if (bsdf->components & (PRINCIPLED_DIFFUSE_FULL | PRINCIPLED_DIFFUSE_RETRO_REFLECTION)) {
    /* H = normalize(L + V);  // Bisector of an angle between L and V
     * LH2 = 2 * dot(L, H)^2 = 2cos(x)^2 = cos(2x) + 1 = dot(L, V) + 1,
     * half-angle x between L and V is at most 90 deg. */
    const float LH2 = dot(L, V) + 1;
    const float RR = bsdf->roughness * LH2;
    f += RR * (FL + FV + FL * FV * (RR - 1.0f));}
float value = M_1_PI_F * NdotL * f;
return make_float3(value, value, value);
```

pdf也很简单，

```
if (dot(N, omega_in) > 0.0f) {
    *pdf = fmaxf(dot(N, omega_in), 0.0f) * M_1_PI_F;
}else {*pdf = 0.0f; }
```



## Glass

如果说塑料材质是光线在漫反射和高光反射中二选一，那么玻璃材质就是光线在折射中和高光反射二选一了。

#### appleseed

在appleseed还考虑各项同性或异性的玻璃材质

```
const float eta = wo.y > 0.0f
                    ? values->m_ior / values->m_precomputed.m_outside_ior
                    : values->m_precomputed.m_outside_ior / values->m_ior;
if (APPLESEED_UNLIKELY(eta == 1.0f)) return;
const float square_roughness = m_roughness * m_roughness;
if (m_anisotropy >= 0.0f) 
{
	const float aspect = std::sqrt(1.0f - m_anisotropy * 0.9f);
    alpha_x = std::max(0.001f, square_roughness / aspect);
    alpha_y = std::max(0.001f, square_roughness * aspect);
}
else
{
    const float aspect = std::sqrt(1.0f + anisotropy * 0.9f);
    alpha_x = std::max(0.001f, square_roughness * aspect);
    alpha_y = std::max(0.001f, square_roughness / aspect);
}
const Vector3f m = GGXMDF::sample(wo, Vector2f(rand01(), rand01()), alpha_x, alpha_y);
const float rcp_eta = 1.0f / eta;
const float cos_wom = clamp(dot(wo, m), -1.0f, 1.0f);
const float F = fresnel_reflectance(cos_wom, rcp_eta, cos_theta_t);
const float r_probability = choose_reflection_probability(F);
bool is_reflection = rand01() < r_probability;
if(is_reflection)
{
	wi = improve_normalization(reflect(wo, m));
	// If incoming and outgoing are on different sides of the surface, 
	// this is not a reflection.
    if (wi.y * wo.y <= 0.0f)return;
    const float denom = std::abs(4.0f * wo.y * wi.y);
    const float D = GGXMDF::D(m, alpha_x, alpha_y);
    const float G = GGXMDF::G(wi, wo, m, alpha_x, alpha_y);
    sample_beauty = mat_reflection_color * F * D * G / denom;
}else
{
	//计算折射方向
	wi = normalize(cos_wom > 0.0f
                ? (rcp_eta * cos_wom - cos_theta_t) * m - rcp_eta * wo
                : (rcp_eta * cos_wom + cos_theta_t) * m - rcp_eta * wo)
	// 计算m_beauty
	const float cos_ih = dot(m, wi);
    const float cos_oh = dot(m, wo);
    const float dots = (cos_ih * cos_oh) / (wi.y * wo.y);
    sqrt_denom = cos_oh + eta * cos_ih;
    const float D = GGXMDF::D(m, alpha_x, alpha_y);
    const float G = GGXMDF::G(wi, wo, m, alpha_x, alpha_y);
    const float T = 1 - F; // F 是被反射的部分，那么T就是被折射的部分
    m_beauty = mat_refraction_color * std::abs(dots) * T * D * G / square(sqrt_denom);
}
```

接下来的BSDF值与pdf值的计算是一样，也是反射和折射分开计算。注意在塑料材质，漫反射和高光反射的概率密度最后要相加在一起，但在玻璃材质中，它们并没有加在一起。

```
if(is_reflection)
{
	jacobian = 1.0f / (4.0f * std::abs(cos_oh));
	pdf = r_probability * jacobian * GGXMDF::pdf(wo, m, alpha_x, alpha_y);
}else
{
	const float sqrt_denom = cos_oh + eta * cos_ih;
    if (std::abs(sqrt_denom) < 1.0e-6f)return 0.0f;
    const float jacobian = std::abs(cos_ih) * square(eta / sqrt_denom);
    pdf =  (1 - r_probability) * jacobian * GGXMDF::pdf(wo, m, alpha_x, alpha_y);
}
```

上面的代码中，反射部分的bsdf来源于Microfacet Models for Refraction through Rough Surfaces  的第15个公式，即
$$
f_r^m(\bold i,\bold o,\bold m) = F(\bold i,\bold m)\frac{\delta_{w_m}(\bold h_r,\bold m)}{4(\bold i \cdot \bold h_r)^2}
$$
折射的部分则来源于第18个公式
$$
f_t^m(\bold i,\bold o,\bold m) = (1-F(\bold i,\bold m))\frac{\delta_{w_m}(\bold h_t,\bold m)\eta_o^2}{(\eta_i(\bold i \cdot \bold h_t) + \eta_o(\bold o \cdot \bold h_t))^2}
$$
appleseed库中的r_probability值其实类似于fresnel值。在之前塑料材料的地方说过，受到物体本身折射反射的权重的影响。

```
const float r_probability = F * reflection_weight;
const float t_probability = (1.0f - F) * refraction_weight;
const float sum_probabilities = r_probability + t_probability;
return sum_probabilities != 0.0f ? r_probability / sum_probabilities : 1.0f;
```

#### luxcore

luxcore中的玻璃材质特点为，它娘的要算wavelength ! 包括archglass和glass

#### yocto库

采样函数参加yocto_shading.h中的sample_refractive()函数。

简洁的yocto库直接使用Fresnel函数判断是反射还是折射。

```
if (rnl < fresnel_dielectric(rel_ior, up_normal, outgoing)) {
    return reflect(outgoing, up_normal);
} else {
    return refract(outgoing, up_normal, 1 / rel_ior);
}
// yocto_math.h
inline vec3f refract(const vec3f& w, const vec3f& n, float inv_eta) {
  auto cosine = dot(n, w);
  auto k      = 1 + inv_eta * inv_eta * (cosine * cosine - 1);
  if (k < 0) return {0, 0, 0};  // tir
  return -w * inv_eta + (inv_eta * cosine - sqrt(k)) * n;
}
```

evaluate函数也几乎是直接使用Fresnel函数

```
auto entering  = dot(normal, outgoing) >= 0;
auto up_normal = entering ? normal : -normal;
auto rel_ior   = entering ? ior : (1 / ior);
if (dot(normal, incoming) * dot(normal, outgoing) >= 0) {
    return vec3f{1, 1, 1} * fresnel_dielectric(rel_ior, up_normal, outgoing);
} else {
    return vec3f{1, 1, 1} * (1 / (rel_ior * rel_ior)) *
           (1 - fresnel_dielectric(rel_ior, up_normal, outgoing));}
```

概率密度函数直接使用Fresnel函数

```
if (dot(normal, incoming) * dot(normal, outgoing) >= 0) {
    return fresnel_dielectric(rel_ior, up_normal, outgoing);
  } else {
    return (1 - fresnel_dielectric(rel_ior, up_normal, outgoing));
  }
```

yocto库还有一种使用了微表面模型的玻璃材质，仍然很简洁，值得一读。

#### libyafaray

使用了只考虑各项同性的微表面模型。没什么可写的，值得一读。

## OrenNayar材质

这个材质的入射光线方向与概率密度函数与diffuse一样。不过evaluate函数将bsdf分为直接照明和互反射照明。

#### appleseed库

光线方向仍然是半球cos采样，概率密度函数也仍然是简单的cos_in / pi。 

而在evaluate 中计算反射率的代码如下

```
if(mat_roughness != 0.0)
{
    // Direct illumination component.
    Lr1 = reflectance * reflectance_multiplier / M_PI *  (C1
         + delta_cos_phi * C2 * std::tan(beta)
          + (1.0f - std::abs(delta_cos_phi)) * C3 * std::tan(0.5f * (alpha + beta)));
    // Add interreflection component.
    Lr2 = m_reflectance * m_reflectance * 0.17f * square(reflectance_multiplier) 
                    * sigma2 / (sigma2 + 0.13f)/  M_PI
                    * (1.0f - delta_cos_phi * square(2.0f * beta * RcpPi<float>()));
    sample_beauty = Lr1 + Lr2
}else
{   // Revert to Lambertian when roughness is zero.
	sample_beauty = m_reflectance / M_PI;  
}

```

pbrt书上与appleseed所用公式并不相同。appleseed所用公式来自Generalization of Lambert’s Reflectance Model 。这个公式将bsdf分为直接照明和互反射照明，同样采用近似方法，直接照明近似为
$$
L_r^1 = \frac{\rho}{\pi}E_0\cos \theta_i[C_1 + \cos(\phi_r -\phi_i)C_2\tan\beta + (1-|\cos(\phi_r - \phi_i|)C_3 \tan(\frac{\alpha + \beta}{2})]
$$
互反射近似为
$$
L_r^2 = 0.17\frac{\rho^2}{\pi}\cos \theta_i \frac{\sigma^2}{\sigma^2 + 0.13}(1 - \cos(\phi_r - \phi_i)(\frac{2\beta}{\pi})^2)
$$
#### blender

在bsdf_oren_nayar.h中，似乎公式不太一样

```
float nl = max(dot(n, l), 0.0f);
  float nv = max(dot(n, v), 0.0f);
  float t = dot(l, v) - nl * nv;

  if (t > 0.0f)
    t /= max(nl, nv) + FLT_MIN;
  float is = nl * (bsdf->a + bsdf->b * t);
  return make_float3(is, is, is);
```



#### LuxCore库

在LuxCore库中，orennaya材质放在了roughmatte.cpp中，采样函数中，入射方向为cos半球采样。

evaluate函数用的本上pbrt书上所介绍的orennaya方程
$$
f_r = \frac{\rho}{\pi}(C_1 + C_2 \max(0,\cos(\phi_i - \phi_o)))\sin \alpha \tan \beta
$$
代码在roughmatte.cpp中

```
const float coef = (A + B * maxcos * sinthetai * sinthetao / max(fabsf(CosTheta(*localSampledDir)), fabsf(CosTheta(localFixedDir))));
if (hitPoint.fromLight)
		return Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * (coef * fabsf(localFixedDir.z / localSampledDir->z));
else
		return Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * coef;
```

概率密度函数则根据光线是否来自光源判断了一下

```
if (directPdfW)
	*directPdfW = fabsf((hitPoint.fromLight ? localEyeDir.z : localLightDir.z) / M_PI 
if (reversePdfW)
	*reversePdfW = fabsf((hitPoint.fromLight ? localLightDir.z : localEyeDir.z) / M_PI 
```

可以看到其实和appleseed用的函数差不多。

#### mitsuba库

roughdiffuse

这个库计算evaluate时，用了两种方法，分别是brdf版本的快速近似，与appleseed版本的标准近似。

```
if(m_useFastApprox)
{
	return m_reflectance->eval(bRec.its)* (INV_PI * Frame::cosTheta(bRec.wo) * (A + B
                * std::max(cosPhiDiff, (Float) 0.0f) * sinAlpha * tanBeta));
}else
{
	Spectrum rho = m_reflectance->eval(bRec.its),
         snglScat = rho * (C1 + cosPhiDiff * C2 * tanBeta +
         (1.0f - std::abs(cosPhiDiff)) * C3 * tanHalf),
         dblScat = rho * rho * (C4 * (1.0f - cosPhiDiff*tmp3*tmp3));
	return  (snglScat + dblScat) * (INV_PI * Frame::cosTheta(bRec.wo));
}

```

#### tungsten库

在OrenNayarBsdf.cpp中，所采用的策略与pbrt书第三版策略是一样的，同样根据概率判断，足够光滑则均匀半球采样，否则cos半球采样。

```
float ratio = clamp(roughness, 0.01f, 1.0f);
    if (event.sampler->nextBoolean(ratio))
        event.wo  = SampleWarp::uniformHemisphere(event.sampler->next2D());
    else
        event.wo  = SampleWarp::cosineHemisphere(event.sampler->next2D());
```

evaluate函数与众不同，不仅使用了较为复杂的近似，还为直接照明和互反射照明添加了不同的因子

```
float fr1 = (C1 + cosDeltaPhi*C2*std::tan(beta) + (1.0f - 
			std::abs(cosDeltaPhi))*C3*std::tan(0.5f*(alpha + beta)));
float fr2 = 0.17f*sigmaSq/(sigmaSq + 0.13f)*(1.0f - 
			cosDeltaPhi*sqr((2.0f*INV_PI)*beta));
Vec3f diffuseAlbedo = albedo(event.info);
return (diffuseAlbedo*fr1 + diffuseAlbedo*diffuseAlbedo*fr2)*wo.z()*INV_PI;
```

pdf函数

```
event.pdf = SampleWarp::uniformHemispherePdf(event.wo)*ratio + SampleWarp::cosineHemispherePdf(event.wo)*(1.0f - ratio);
```



## 布料Cloth

filament用的是光栅化着色器下的微表面模型，详见shading_model_cloth.fs。

falcor的ClothBRDF.slang给出一些参考论文，包括 \- Neubelt and Pettineo 2013, "Crafting a Next-gen Material Pipeline for The Order: 1886".

  \- Estevez and Kulla 2017, "Production Friendly Microfacet Sheen BRDF".

## 头发Hair材质

#### tungesten

在LambetianFiberBcsdf.cpp中，

```
min(std::sqrt(max(1.0f - x*x, 0.0f)), 1.0f);
```

还有另一种HairBcsdf.cpp中，计算较为复杂，代码注释详细，细节请看论文Light Scattering from Human Hair Fibers。

#### blender

在bsdf_hair.h和bsdf_principle_hair，好像很复杂

falcor

在HairChiang16.slang中，参考论文为Hair BCSDF from "A Practical and Controllable Hair and Fur Model for Production Path Tracing", Chiang et al. 2016。

## 天鹅绒Velvet材质

velvet材质算是布料的一种。

在luxcore，入射光线法向为半球上cos采样，概率密度函数参见之前的orennaya材质。其evaluate函数返回的结果如下

```
const float cosv = -Dot(localFixedDir, *localSampledDir);
// Compute phase function
const float B = 3.0f * cosv;
float p = 1.0f + A1 * cosv + A2 * 0.5f * (B * cosv - 1.0f) + A3 * 0.5 * (5.0f * cosv * cosv * cosv - B);
p = p / (4.0f * M_PI);
p = (p * delta) / (hitPoint.fromLight ? fabsf(localSampledDir->z) : fabsf(localFixedDir.z));
// Clamp the BRDF (page 7)
if (p > 1.0f)p = 1.0f;
else if (p < 0.0f)p = 0.0f;
return Kd->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * (p / *pdfW);
```

然而我并没有找到上面的公式在哪篇论文里。

#### blender

在bsdf_ashikhmin_velvet.h中，入射光线方向为半球均匀采样，

evaluate核心代码如下

```
float fac1 = 2 * fabsf(cosNHdivHO * cosNO);
float fac2 = 2 * fabsf(cosNHdivHO * cosNI);
float sinNH2 = 1 - cosNH * cosNH;
float sinNH4 = sinNH2 * sinNH2;
float cotangent2 = (cosNH * cosNH) / sinNH2;
float D = expf(-cotangent2 * m_invsigma2) * m_invsigma2 * M_1_PI_F / sinNH4;
float G = min(1.0f, min(fac1, fac2));  // TODO: derive G from D analytically
float out = 0.25f * (D * G) / cosNO;
*pdf = 0.5f * M_1_PI_F;
return make_float3(out, out, out);
```



## Irawan

用于模拟woven的材质

#### mitsuba

sample函数和pdf函数和漫反射的一样，不过evaluate函数很复杂，具体请参考mitsuba库的irawan.cpp，代码中注释很详细，且附有参考论文，这里就不贴了。

广义微表面

blender中的bsdf_microfacet.h

```
* Based on paper from Wenzel Jakob
   * An Improved Visible Normal Sampling Routine for the Beckmann Distribution
   *
   * http://www.mitsuba-renderer.org/~wenzel/files/visnormal.pdf
   *
   * Reformulation from OpenShadingLanguage which avoids using inverse
   * trigonometric functions.
   */
```



# 多层材质

多层材质在生活中很常见，大部分外面都是一层薄半透明材质，加上里面的一般的普通材质。比如车漆和汽水罐。

## Translucent材质

半透明材质即外面一层薄透明材质，里面一层不透明材质。

#### luxcore库

在LuxCore库中有个很特别的RoughMatteTranslucent材质，也就是半透明的粗表面。kr即为反射，kt就是透射

```
const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f) * 
// Energy conservation
		(Spectrum(1.f) - kr);
const Spectrum result = (INV_PI * fabsf(localLightDir.z) *
		(A + B * maxcos * sinthetai * sinthetao / max(fabsf(CosTheta(localLightDir)), fabsf(CosTheta(localEyeDir)))));
if (localLightDir.z * localEyeDir.z > 0.f) {
	*event = DIFFUSE | REFLECT;
	return kr * result;
} else {
	*event = DIFFUSE | TRANSMIT;
	return kt * result;
}
```

注意上面第7行，用的是灯光照射方向与摄像机方向的dot 是否大于零，大于零则说明方向相同，那么反射。否则投射。但是luxcore也在sample函数用另一种方式，即随机数判断是反射还是折射。

对于这种半透明粗表面，luxcore库的概率密度函数比较独特，如下

```
if (!isKrBlack) { if (!isKtBlack)weight = .5f;
                  else weight = 1.f;
} else {
		if (!isKtBlack)weight = 0.f;
		else { 	if (directPdfW) *directPdfW = 0.f;
				if (reversePdfW)*reversePdfW = 0.f;
				return;}}
const bool relfected = (Sgn(CosTheta(localLightDir)) == Sgn(CosTheta(localEyeDir)));
weight = relfected ? weight : (1.f - weight);
if (directPdfW)
*directPdfW = fabsf((hitPoint.fromLight ? localEyeDir.z : localLightDir.z) * (weight * INV_PI));
```

如果kr和kt都不为零，那么各自的权重就为0.5。

## ClearCoat材质

filament用的是光栅化着色器下的微表面模型，详见shading_model_standard.fs。注释很详细。

#### tungsten

在SmoothCoatBsdf.cpp中，概率判断仍然用电介质Fresnel加上权重判断

```
float Fi = Fresnel::dielectricReflectance(eta, wi.z(), cosThetaTi);
float substrateWeight = _avgTransmittance*(1.0f - Fi);
float specularWeight = Fi;
if (sampleR && sampleT)
        specularProbability = specularWeight/(specularWeight + substrateWeight);
else if (sampleR)	specularProbability = 1.0f;
else if (sampleT)	specularProbability = 0.0f;
else	return false;
```

入射光线方向没用微表面模型，所以反射仍然是镜面反射，折射需要乘上折射率，都为本地坐标系。

```
event.wo = Vec3f(-wi.x(), -wi.y(), wi.z());
event.wo = Vec3f(event.wo.x()*_ior, event.wo.y()*_ior, cosThetaTo);
```

evaulate函数，看不懂

```
if (evalR && checkReflectionConstraint(event.wi, event.wo)) {
     return Vec3f(Fi);
} else if (evalT) {
    Vec3f wiSubstrate(wi.x()*eta, wi.y()*eta, std::copysign(cosThetaTi, wi.z()));
    Vec3f woSubstrate(wo.x()*eta, wo.y()*eta, std::copysign(cosThetaTo, wo.z()));
	float laplacian = eta*eta*wo.z()/cosThetaTo;
	Vec3f substrateF = _substrate->eval(event.makeWarpedQuery(wiSubstrate, 
						woSubstrate));
    if (_scaledSigmaA.max() > 0.0f)
         substrateF *= std::exp(_scaledSigmaA*(-1.0f/cosThetaTo - 
						1.0f/cosThetaTi));
    return laplacian*(1.0f - Fi)*(1.0f - Fo)*substrateF;
} else {return Vec3f(0.0f);}
```

pdf函数，

```
event.pdf *= (1.0f - specularProbability)*eta*eta*cosThetaTo/cosThetaSubstrate;
```



## ThinDielectric材质

mitsuba库限定材质？

总之就是一层电解质包着另一层薄电解质，内部那层电介质因为很薄所以忽略折射。例如空气包着玻璃球。

sample函数中，光线可能是完美反射或是透射，不过这里的透射有点奇怪

```
/// Transmission in local coordinates
inline Vector transmit(const Vector &wi) const {return -wi;}
```

光线进入外面那层电介质时，可能反射也可能透射，如果反射的话，那么概率为R。

透射的部分碰到内部那层电介质，也可能被反射。本次被反射的光线，遇到外面那层电介质的边界，再透射一次，就离开了物体，概率为TRT。

没能离开物体的光线，再发生一次反射，一次透射，同样能离开物体，概率为TRRRT。

复读一遍，第三次仍然没能离开物体的光线，再发生一次反射，一次透射，同样能离开物体，概率为TRRRRRT。

光线可以发生很多次反射，所以总的离开物体的概率，相当于发生总反射的概率是
$$
R' = R + TRT + TR^3T + ... = R + \frac{T^2R}{1 - R^2}
$$
这样一来mitsuba的thindielectric.cpp就很容易看懂了。

```
Float R = fresnelDielectricExt(std::abs(Frame::cosTheta(bRec.wi)), m_eta), T = 1-R;
if (R < 1)R += T*T * R / (1-R*R);
if (sampleTransmission && sampleReflection) {
     if (sample.x <= R) {
            Rec.wo = reflect(bRec.wi);
            f = m_specularReflectance->eval(bRec.its);
            pdf = R;
     }else{
     		bRec.wo = transmit(bRec.wi);
     		f = m_specularTransmittance->eval(bRec.its);
     		pdf = 1 - R;
     }
else if(sampleReflection)
{
	Rec.wo = reflect(bRec.wi);
    f = m_specularReflectance->eval(bRec.its) * R;
    pdf = 1;
}else if(sampleTransmission)
{
	bRec.wo = transmit(bRec.wi);
	f = m_specularTransmittance->eval(bRec.its) * (1-R);
	pdf = 1;
}
```



# 特殊材质

这里收集的一些材质，包括双面材质，混合材质

## TwoSided

双面材质，即光线从击中物体的前后表面时，会表现出不同的材质效果。

在luxcore库中，仅仅是简单的判断一下击中的是物体的正面还是后面，再调用相应材质的函数。函数在twosided.cpp中。

```
if (hitPoint.intoObject)
		return frontMat->Evaluate(hitPoint, localLightDir, localEyeDir, event, directPdfW, reversePdfW);
	else
		return backMat->Evaluate(hitPoint, localLightDir, localEyeDir, event, directPdfW, reversePdfW);
```

## 混合blend/Mix材质

混合材质，就是光线击中的某点上，既有某种材质，也有另一种材质，不过百分比不同而已。

#### filament库

bsdf以及概率密度函数都是乘上两种材质的权重即可

```
m_bsdfs[0]->eval(bRec, measure) * (1-weight) +
m_bsdfs[1]->eval(bRec, measure) * weight;
```

#### tungsten库

在MixedBsdf.cpp中，采样时，根据两种材质的权重，来选择一个采样入射光线。

而bsdf以及概率密度则结合两种材质，并乘上不同的权重。

```
if (event.sampler->nextBoolean(ratio)) {event.sampler->nextBoolean(ratio)}
else{_bsdf1->sample(event)}
f = albedo(event.info)*(_bsdf0->eval(event)*ratio + _bsdf1->eval(event)*(1.0f - ratio));
pdf = _bsdf0->pdf(event)*ratio + _bsdf1->pdf(event)*(1.0f - ratio);
```

## 全透明Transparency材质

外层全透明，内层为其它材质。光线经过全透明时不发生任何反应。

#### tungsten库

直接调用内层材质的sample，evaluate与pdf函数。

#### yocto库

sample函数，参见yocto_shading.h中的sample_passthrough();

```
return -outgoing;
```

evaluate函数

```
if (dot(normal, incoming) * dot(normal, outgoing) >= 0) {
    return vec3f{0, 0, 0};
  } else {
    return vec3f{1, 1, 1};
  }
```

pdf函数

```
if (dot(normal, incoming) * dot(normal, outgoing) >= 0) {
    return 0;
  } else {
    return 1;
  }
```



# 参考

## appleseed

![image-20211121131744735](E:\mycode\collection\定理\源码阅读\image-20211121131744735.png)

简介：基于物理的全局光照渲染引擎。代码里的注释很详细，值得初学者学习。

官网：https://appleseedhq.net/

说明：本篇以appleseed库的2021年3月28号的提交为参考，这个版本的地址为，https://github.com/appleseedhq/appleseed/tree/1ba62025b5db722e179a2219d8d366c34bfaa342

## Filament

![image-20211121231837183](E:\mycode\collection\定理\源码阅读\image-20211121231837183.png)

简介：极受欢迎的跨平台的基于物理的实时渲染库，github星星12k，fork数1k。不过源代码中注释很少，不怎么容易看懂。甚至有一份官方文档介绍Filament的源码，即physc

## Luxcore

## Mitsuba2

![image-20211121232349530](E:\mycode\collection\定理\源码阅读\image-20211121232349530.png)

简介：以研究为导向的基于物理的渲染引擎。代码中注释非常详细，值得初学者学习。以研究为导向，说的是很多论文的代码都是基于mitsuba改进的，我在github上见到好几个了。

## Yocto

简介：代码注释详细，且写的简洁紧凑，很适合初学者学习。在yocto_shading.h中记载了各种导体金属的折射率。
