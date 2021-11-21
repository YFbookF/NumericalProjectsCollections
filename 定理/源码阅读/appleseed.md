有关光线追踪光照的概念介绍，各种文章，博客，书籍已经介绍得很不错了，比如physics based rendering。但是这些文章很多仅仅是给出一堆公式和图，有的不错的教程倒会给出代码，但也仅仅是自己的实现方向。如果想要博采众家之长，显然是不够的，因此我收集了一些开源渲染器中一些函数的实现方向，通过横向比较来发现它们的特点。

还有个问题，就是作为初学者看不懂文章的时候，好不容易搞到一份源代码，但是抄也不是，不抄也不是。因为不抄的话，自己是完全想不到代码怎么写的。抄的话，以自己有限的水平也只能复制粘贴了，学不到太多东西。

首先是在pathtracer.h的process_bounce函数中采样bsdf，然后计算on_first_diffuse_bounce

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

## Diffuse

#### appleseed

appleseed库的diffusebtdf.cpp中的sample_f核心代码是这么写的

```
Vector3f wi = sample_hemisphere_cosine(s);
const float probability = wi.y / M_PI;
assert(probability > 0.0f);
if (probability > 1.0e-6f){wi.y = -wi.y}
```

第一行在半球上进行cosine采样，第二行咋来的？第四行

evaluate函数的核心代码如下

```
const float cos_in = dot(incoming, n);
const float cos_on = dot(outgoing, n);
if (cos_in * cos_on < 0.0f){
// Return the probability density of the sampled direction.
return std::abs(cos_in) * RcpPi<float>();
}else{
// No transmission in the same hemisphere as outgoing.
return 0.0f;}
```

应该很容易看懂？喵的incoming和outgoing究竟是哪个方向？

pdf函数与上面evaluate函数的核心代码是一样的？为什么？

## Lambetian

appleseed中，lambertian的很简单，入射光线仍然是半球cos采样，reflectance的计算是因子除以pi，bsdf和pdf的计算都是cos_in 除以pi

```
const Vector3f wi = sample_hemisphere_cosine(s);
m_beauty = m_reflectance * m_reflectance_multiplier / M_PI;
f_bsdf = std::abs(dot(wi, n)) / M_PI;
pdf = std::abs(dot(wi, n)) / M_PI;
```

mitsuba中，也很简单。但是值得一提的是，这里计算bsdf的时候，物体表面上每一点的m_reflectance 参数都是不同的。

```
wi = squareToCosineHemisphere();
f = m_reflectance->evalAtPoint(bRec.its) * (INV_PI * Frame::cosTheta(bRec.wo));
squareToCosineHemispherePdf = INV_PI * Frame::cosTheta(wi);
```



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

## Reflection材质

#### luxcore库

很简单的镜面反射，不过很多渲染库喜欢使用本地坐标，也许是这样方便一些？

```
*localSampledDir = Vector(-localFixedDir.x, -localFixedDir.y, localFixedDir.z);
fr =  Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
*pdfW = 1.f;
```

## Glossy材质

#### luxcore库

glossy则是不那么完美的反射

<img src="E:\mycode\collection\定理\源码阅读\image-20211121225733930.png" alt="image-20211121225733930" style="zoom: 25%;" />

上图的右上就是glossy，左下就是完美反射

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

## Plasticity

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
const Vector3f m = GGXMDF::sample(wo, Vector2f(rand01(), rand01()), alpha);
//如果当前应该高光反射
if(rand01() < specular_probability){
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

首先在第一行根据pbrt第三版第8.4.2节，我们先将roughness 转换为 alpha，数值越小，代表表面越光滑接近完美反射。

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
这样就熟悉多了。而塑料的材质的漫反射，首先假设塑料由两层构成，第一场为透明层，第二层为漫反射不透明层，具体理论请参见appleseed库所参考的论文A Physically-Based Reflectance Model Combining
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
acobian = 1.0f / (4.0f * std::abs(cos_wom));
specular_pdf =  jacobian * GGXMDF::pdf(wo, m, alpha, alpha);
diffuse_pdf = std::abs(wi.y) / M_PI;
//specular_probability 也和上份代码一样
pdf = lerp(diffuse_pdf, specular_pdf, specular_probability);
```

#### mitsuba库

这个库计算塑料的方法很特别，允许在一次采样中漫反射和高光反射同时存在



evaluate函数是这么计算的，可以看到比appleseed简单不少。

```
if (hasSpecular) {
	/* Check if the provided direction pair matches an ideal
    specular reflection; tolerate some roundoff errors */
    if (std::abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon)
          return m_specularReflectance->eval(bRec.its) * Fi;
} else if (hasDiffuse) {
    Float Fo = fresnelDielectricExt(Frame::cosTheta(bRec.wo), m_eta);
	Spectrum diff = m_diffuseReflectance->eval(bRec.its);
	if (m_nonlinear)diff /= Spectrum(1.0f) - diff * m_fdrInt;
    else  diff /= 1 - m_fdrInt;
	return diff * (warp::squareToCosineHemispherePdf(bRec.wo)
                * m_invEta2 * (1-Fi) * (1-Fo));}
```

概率密度函数是这么计算的

```
if (Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0)
    {   pdf = 0.0f; return;}
Float Fi = fresnelDielectricExt(Frame::cosTheta(bRec.wi), m_eta);
pdf = (Fi*m_specularSamplingWeight) /
	(Fi*m_specularSamplingWeight +(1-Fi) * (1-m_specularSamplingWeight));
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

## Blinn-Phong

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

## OrenNayar材质

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

## Velvet

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

# 特殊材质

这里收集的一些材质，包括双面材质，混合材质

### TwoSided

双面材质，即光线从击中物体的前后表面时，会表现出不同的材质效果。

在luxcore库中，仅仅是简单的判断一下击中的是物体的正面还是后面，再调用相应材质的函数。函数在twosided.cpp中。

```
if (hitPoint.intoObject)
		return frontMat->Evaluate(hitPoint, localLightDir, localEyeDir, event, directPdfW, reversePdfW);
	else
		return backMat->Evaluate(hitPoint, localLightDir, localEyeDir, event, directPdfW, reversePdfW);
```

## blend

混合材质，就是光线击中的某点上，既有某种材质，也有另一种材质，不过百分比不同而已。

#### filament

bsdf以及概率密度函数都是乘上两种材质的权重即可

```
m_bsdfs[0]->eval(bRec, measure) * (1-weight) +
m_bsdfs[1]->eval(bRec, measure) * weight;
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

