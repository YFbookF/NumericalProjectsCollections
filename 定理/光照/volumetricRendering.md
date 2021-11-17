==========BSSRDF Explorer: A rendering framework for the BSSRDF  

Volumetric Scattering  

吸收，吸收的radiance会被转换为热量，但我们可以直接让它消失即可。x is the current position along the ray, w is the direction ray, and radiance ray
$$
(\vec w \cdot \nabla)L(\vec x,\vec w) = -\sigma_a L(\vec x,\vec w)
$$
Emission，介质自己会发光。
$$
(\vec w \cdot \nabla)L(\vec x,\vec w) = Q(\vec x,\vec w)
$$
OutScattering. At each step, the radiance may also be reduced due to light be scattered into other directions than w
$$
(\vec w \cdot \nabla)L(\vec x,\vec w) = -\sigma_s L(\vec x,\vec w)
$$
InScattering 其它方向的光也有可能照射到这个方向上来
$$
(\vec w \cdot \nabla)L(\vec x,\vec w) = \int_{\Omega}p(\vec w',\vec w)L(\vec x,\vec w')d\vec w'
$$
p is phase function. The phase function describes the angular distribution of light intensity being scattering.

A simple, but naive and inefficient implementation of the trace function is given as follows. It
is a straightforward application of Monte Carlo to the RTE:  

\1. Start with a position ~x and direction ~!
\2. Pick a random distance s to travel before the next scattering event
\3. Advance by s along ~! and pick a new random direction ~!0 on the sphere
\4. Modify the path weight by extinction e-σts, scattering coefficient σs and phase function
p(~! · ~!0)
\5. Repeat steps 2-4 until the surface is hit
\6. Modify the path weight by the transmission term ft  

但这种方法并不是很好，尤其是对高吸收率的材料的时候，需要超大样本数才能收敛。

=============Volumetric Path Tracing   Steve Marschner  Cornell University  CS 6630 Fall 2015, 29 October  

用蒙特卡洛法好是好，但是对于inhomogeneous 或者高散射的物体来说，就好了。

The intergral form of the volume rendering volume

![image-20211111201459891](E:\mycode\collection\定理\光照\image-20211111201459891.png)

![image-20211111201423560](E:\mycode\collection\定理\光照\image-20211111201423560.png)

如果只有Emission，那么tau = 1，那么sigmas 和 sigmaa = 0
$$
L(x,w) = \int_{x}^y \varepsilon(x',w)dx'
$$
If we use uniform sampling with respect to distance along the ray, we have the estimator
$$
g = \varepsilon(x',w)||x - y||
$$
如果只有Homogeneous 的 absorption，那么

vareps = sigmas = 0，并且
$$
L(\bold x,w) = \tau(\bold x,\bold y)L_e(\bold y,w)
$$
没有积分，也无需蒙特卡洛。

只有Emission和homogeneous absorption，那么我们暂且还能积分
$$
L(x,w) = \int_x^y \tau(x,x')\varepsilon(x',w)dx' + \tau(x,y)L_e(y,w)
$$
for uniform sampling，
$$
g = ||x - y||exp(-\sigma_n||x - x'||)\varepsilon(x',w)
$$
但是如果有衰减的话，我们就不能统一采样。但是我们可以重要性采样衰减。homogeneous介质的衰减是
$$
\tau(s) = \exp(-\sigma_t s)
$$
where s is arc length along the ary。我们需要找到一个pdf，与tau(s)。我们只需要归一化就行了，就得到了pdf
$$
p(s) = \frac{\tau(s)}{\int_0^{\infin}\tau(t)dt} = \frac{\exp(-\sigma_t s)}{-\exp(-\sigma_t t/\sigma_t)}_{t=0} = \sigma_t \exp(-\sigma_t s)
$$
然后算cumulative distribution function cdf
$$
P(s) = \int_0^s p(s')ds' = (-\exp(-\sigma_t s')_0^s) = 1 - \exp(-\sigma_t s)
$$
然后不知道在算什么东西
$$
\xi = 1 - \exp(-\sigma_ts) \qquad s = -\frac{\ln(1-\xi)}{\sigma_t} \qquad s = -\frac{\ln s}{\sigma_t}
$$

```
function radiance_estimator(x,w)
	xi = rand();
	s = - ln(xi)/sigma_t
	if s < s_max
		return vareps * (x - s * w) / sigma_t
	return L
```

======================pbrt

要算的积分式子
$$
L_i(p,w) = T_r(p_0 \rightarrow p)L_0(p_0,-w) + \int_0^t T_r(p+tw \rightarrow p)L_s(p+ tw,-w)dt
$$
p = p + tmax w 就是撞击点。这时候有两种情况，首先如果不采样[0,tmax]直接，那么我们只需要推算第一项。

我们假设p(t)是光线前进一个单位距离时碰到的粒子程度，
$$
p_{surf} = 1 - \int_0^{tmax} p_t(t)dt
$$
同时sigma_t 是衰减因子，随波长变化。
$$
p_{surf} = \frac{1}{n}\sum_{i=1}^n \exp(-\sigma_t^i t_{max})
$$

```
// Compute the transmittance and sampling density
    Spectrum Tr = Exp(-sigma_t * std::min(t, MaxFloat) * ray.d.Length());

    // Return weighting factor for scattering from homogeneous medium
    Spectrum density = sampledMedium ? (sigma_t * Tr) : Tr;
    Float pdf = 0;
    for (int i = 0; i < Spectrum::nSamples; ++i) pdf += density[i];
    pdf *= 1 / (Float)Spectrum::nSamples;
    if (pdf == 0) {
        CHECK(Tr.IsBlack());
        pdf = 1;
    }
    return sampledMedium ? (Tr * sigma_s / pdf) : (Tr / pdf);

```

采样完后是一个数字，最终被作为因子，乘上采样灯光的系数上去了

```
// Sample the participating medium, if present
        MediumInteraction mi;
        if (ray.medium) beta *= ray.medium->Sample(ray, sampler, arena, &mi);
        if (beta.IsBlack()) break;

        // Handle an interaction with a medium or a surface
        if (mi.IsValid()) {
            // Terminate path if ray escaped or _maxDepth_ was reached
            if (bounces >= maxDepth) break;

            ++volumeInteractions;
            // Handle scattering at point in medium for volumetric path tracer
            const Distribution1D *lightDistrib =
                lightDistribution->Lookup(mi.p);
            L += beta * UniformSampleOneLight(mi, scene, arena, sampler, true,
                                              lightDistrib);

            Vector3f wo = -ray.d, wi;
            mi.phase->Sample_p(wo, &wi, sampler.Get2D());
            ray = mi.SpawnRay(wi);
            specularBounce = false;
        } else {
```

注意sigma t是在一个特殊的spectral channel 里面。而sigma_a 是吸收 absorb，而sigma_s 是发射 scattering，它俩之和sigma t也就是transmittance 是固定的，注意Transmittance 是透光率

就比如Grid Distance

```
while (true) {
        t -= std::log(1 - sampler.Get1D()) * invMaxDensity / sigma_t;
        if (t >= tMax) break;
        if (Density(ray(t)) * invMaxDensity > sampler.Get1D()) {
            // Populate _mi_ with medium interaction information and return
            PhaseFunction *phase = ARENA_ALLOC(arena, HenyeyGreenstein)(g);
            *mi = MediumInteraction(rWorld(t), -rWorld.d, rWorld.time, this,
                                    phase);
            return sigma_s / sigma_t;
        }
    }
```

$$
t_i = t_{i-1} - \frac{\ln(1 - \xi)}{\sigma_t}
$$

当t > tmax，说明光已经离开了介质，此时返回的透光率是1。Density 用的是三线性插值，也就是本地粒子的密度。Get1D是个随机。

注意光线强度随指数衰减，那么，那么直接算个1/sigma_T就行了吗？可能它不是无限远，仅仅是从0积分到t

而且sigma_t 以及density 是数组，需要把各个波长的衰减都考虑进去
$$
t = -\frac{\ln(1 - \xi)}{\sigma_t} \qquad p_t(t) = \sigma_te^{-\sigma_t t}
$$


```
 bool sampledMedium = t < ray.tMax;
    if (sampledMedium)
        *mi = MediumInteraction(ray(t), -ray.d, ray.time, this,
                                ARENA_ALLOC(arena, HenyeyGreenstein)(g));

    // Compute the transmittance and sampling density
    Spectrum Tr = Exp(-sigma_t * std::min(t, MaxFloat) * ray.d.Length());

    // Return weighting factor for scattering from homogeneous medium
    Spectrum density = sampledMedium ? (sigma_t * Tr) : Tr;
    // 注意这是个float，意味着我们平均了各个波长
    Float pdf = 0;
    for (int i = 0; i < Spectrum::nSamples; ++i) pdf += density[i];
    pdf *= 1 / (Float)Spectrum::nSamples;
    if (pdf == 0) {
        CHECK(Tr.IsBlack());
        pdf = 1;
    }
    return sampledMedium ? (Tr * sigma_s / pdf) : (Tr / pdf);
```

======================Volumetric Skin and Fabric Shading at Framestore  

homogeneous 是介质各处的参数都相同，但是heterogeneous 不一定相同。

=================Path tracing in Production
Part 1: Modern Path Tracing  

In a homogeneous medium, the change in radiance along the ray is proportional to the extinction coefficient μt. This simple fact allows us to derive the transmittance along the ray:  
$$
T(t) = \exp(-\mu_t t)
$$
which is known as Beer’s Law. This is a smooth function with a rapid falloff. This is what makes distant objects in a volume appear darker (while scattering is what leads to the hazy appearance). The natural choice when sampling a distance for further interaction is to choose a point proportionally to this transmittance function. Normalizing the transmittance T(t) into a pdf we obtain:  
$$
p(t) = \frac{T(s)}{\int_0^{\theta}T(s)ds} = \mu_t T(t)
$$
This gives us the following cdf c (t) and sample generation function t (ξ):  
$$
c(t) = \int_0^t p(s)ds = 1- \exp(-\mu_t t) \qquad t(\xi) = -\frac{\log(1 - \xi)}{\mu_t}
$$
然后我们再生成一个0到1的随机数xi，就可以知道光线在哪里相交了。假如光线并非全在介质内，而是[a,b]范围里才有介质，那么

![image-20211116203045931](E:\mycode\collection\定理\光照\image-20211116203045931.png)

hetergeneous，如果介质的 mu_t 不再是常数，那么透光率计算如下
$$
T(t) = \exp  (-\int_0^t \mu_t [\bold o + s\bold d]ds)
$$
Even in homogeneous media,the nice simplifications between path contribution and pdf are no longer possiblewhen the densities are colors because the pdf itself must always be ascalar. Thesolution to this problem is to incorporatemultiple importancesampling to combine the probability of sampling from eachwavelengthseparately  

![image-20211116203806317](E:\mycode\collection\定理\光照\image-20211116203806317.png)

======================Practical Illumination from Flames  

渲染方程如下

![image-20211116232851953](E:\mycode\collection\定理\光照\image-20211116232851953.png)

在一段无限小的，方向w0里，radiance的变化率，是那一小段的emission以及从所有方向来的scattering

![image-20211116233026432](E:\mycode\collection\定理\光照\image-20211116233026432.png)

如果我们忽略散射的话，那么sigma_t = sigma_a，也就是
$$
L'(t) + \sigma_AL(t) = \varepsilon(t)
$$
其解析解为，注意1号方程下面那个方程，这是假设在isotropic介质中，scattering 为零。

![image-20211116233256118](E:\mycode\collection\定理\光照\image-20211116233256118.png)

![image-20211116233326522](E:\mycode\collection\定理\光照\image-20211116233326522.png)

![image-20211116233708643](E:\mycode\collection\定理\光照\image-20211116233708643.png)

===================================Multiple Importance Sampling for Emissive Effects  

If the energy emitted from volumes is fairly low, it can be treated as any normal emissive object;
however if the emission becomes strong, we need to treat the volume as a light source with its own
sampling strategy, in order to reduce variance  

![image-20211117090753065](E:\mycode\collection\定理\光照\image-20211117090753065.png)

==================================

```
// Transmittance
float T = 1.f;
for (int step = 0; step < numsteps; ++step) {
float rho = m_buffer.trilinearInterpolation(raypos);
T *= std::exp(rhomult * rho);
if (T < 1e-8)
break;
raypos += stepdir;
}
```

我们可以model absorption
$$
L_o = L_i + dL \qquad dL = -\sigma_a L_i dt
$$
Lo 是进入介质前的incoming radiance，Lo是离开介质后的outgoing radiance

model emission
$$
L_o = L_i + L_e \qquad L_e = \sigma_e dt
$$
model scattering，Given a light source S, whose function S(p, ω') describes the quantity of light arriving at point p from direction ω', we can formulate the scattering interaction as:  
$$
L_o = L_i + dL_{in} + dL_{out} \qquad dL_{in} = \sigma_s p(w,w')S(p,w')\qquad dL_{out} = -\sigma_s L_idt
$$
This
means that a ray traveling through a medium with σs = 0.1 will travel on average a distance of 0.1-1 = 10
units before a scattering event occurs. This distance can also be referred to as the scattering length.  

The function p(ω, ω') is called the phase function, and the next section will detail what it is and how it
affects the scattering interaction.  

体素预处理光照

```
void voxelizeLights(const Scene &scene, const std::vector<Light> &lights,
VoxelBuffer &lightBuffer)
{
BBox dims = lightBuffer.dims();
for (int k = dims.min.z; k <= dims.max.z; ++k) {
for (int j = dims.min.y; j <= dims.max.y; ++j) {
for (int i = dims.min.x; i <= dims.max.x; ++i) {
V3f vsP = discreteToContinuous(i, j, k);
V3f wsP;
lightBuffer.mapping().voxelToWorld(vsP, wsP);
Color incomingLight = 0.0f;
for (int light = 0; light < lights.size(); ++light) {
float intensity = lights[light].intensity(wsP);
// Raymarch toward light to compute occlusion. This is a costly operation.
float occlusion = computeOcclusion(lights[light].wsPosition(), wsP);
incomingLight += intensity * (1.0 - occlusion);
}
lightBuffer.lvalue(i, j, k) = incomingLight;
}
}
}
}
```

Technically, we could voxelize this data set and use it in beauty renders the same way as in the previous
section, but a more efficient way to store this data is to leave it in its native form, i.e. a monotonically
decreasing function per pixel seen from the light source. When storing it in this way, it is equivalent to
the deep shadow maps technique described by Tom Lokovic and Eric Veach in their paper Deep Shadow
Maps [Lokovic, 2000].  

=================Production Volume Rendering
SIGGRAPH 2017 Course  

As a photon travels through a volume, it may collide with the particles making up the volume. These
collisions dene the radiance distribution throughout a volume. Because it is not feasible to model
each and every particle in a volume, they are treated as collision probability elds; essentially, particle
collisions are stochastically instantiated  

Phase Function. The phase function fp„x; ω; ω0” is the angular distribution of radiance scattered and
is usually modeled as a 1D function of the angle θ between the two directions ω and ω0. Phase functions
need to be normalized over the sphere:  
$$
\int f_p(x,w,w')d\theta = 1 \qquad f_p(x,\theta) = \frac{1}{4\pi}
$$
Volumes that are isotropic have an equal probability of scattering incoming light in any
direction, and have an associated phase function:  

Anisotropic volumes can exhibit complicated phase functions which can be accurately modeled by
using the Mie solution to Maxwell’s equations (Mie scattering), or by using the Rayleigh approximation.
As an alternative to these expensive functions, in production volume rendering, the most widely used
phase function is the Henyey-Greenstein phase function  

对于吸收，w是光线前进的方向，x就是光线的起点，对于一线段光线柱来说，它的radiance的变化量就是一个吸收参数，乘上那个地方的radiance
$$
(w \cdot \nabla)L  = -\sigma_a(x)L(\bold x,w)
$$
外散射乘上一个参数就行了，发光也是乘参数，内散射倒不用乘参数，而是要积分

![image-20211117124347855](E:\mycode\collection\定理\光照\image-20211117124347855.png)

我们只处理光子碰到粒子后的反应，接下来只需要采样光子的自由路径距离t，就是用随机数xi算的那个自由路径t。但是只有当CDF可逆时，我们才能采样自由路径。

咱不用raymarch，咱用close-form tracking

![image-20211117125340371](E:\mycode\collection\定理\光照\image-20211117125340371.png)

atmosphere and fog are
almost always modeled using homogeneous volumes.  

closed-form tracking can be used for path-traced subsurface
scattering to render skin or any other subsurface scattering.  

对于hetermogoenous，需要处理碰撞

![image-20211117133927679](E:\mycode\collection\定理\光照\image-20211117133927679.png)

```
for (int i = 0; i < trials; ++i)
        {
            Float trans = 0.0;
            Float cost_tmp = 0.0;

            Sampler* base_sampler = sampler->copy();
            base_sampler->reseed(simple_hash(base_seed + (i * resolution * resolution * resolution)));
            base_sampler->setNumSamples(samples);

            for (int j = 0; j < samples; ++j)
            {
                TransmittanceQuaryRecord rec(ext, 0.0, 1.0, -1, samples);
                trans += est->T(rec, base_sampler);
                cost_tmp += Float(rec.extCalls);

                base_sampler->nextSample();
            }

            trans /= Float(samples);
            cost_tmp /= Float(samples);

            if (est->getName() == "expected") i = trials;

            Float mean_new = mean + (1.0 / Float(i+1)) * (trans - mean);
            var = var + (trans - mean) * (trans - mean_new);
            mean = mean_new;

            cost += cost_tmp;
            delete base_sampler;
        }
```

Trans计算

```
Float RatioTracking::T(TransmittanceQuaryRecord& rec, Sampler* sampler) const {
    Float x = rec.a;
    Float T = 1;

    do {
        // Sample a random value y with pdf(y) = k * exp(-ky) set x = a + y
        Float k = rec.extFunc->calculateMajorant(x);
        x += sampleExpFF(rec, sampler, k);

        if (x >= rec.b) break;

        T *= (1.0 - rec.extFunc->calculateExtinction(x, rec.extCalls) / k);
    } while(true);

    rec.transmittance = T;

    return T;
}

```

=================Weighted Delta-Tracking in Scattering Media  

![image-20211117144414864](E:\mycode\collection\定理\光照\image-20211117144414864.png)

=================Production Volume Rendering
SIGGRAPH 2017 Course  

As hinted at in the previous section, there is no single preferred integration technique for volumes that
works well across every dierent type of participating media. Simple cases like homogeneous absorption
can be implemented by a straight forward beam transmittance calculation added to the throughput
calculation of a path tracing algorithm. More complex cases involving light scattering through low
albedo media (e.g. smoke) typically require algorithms which only need single scattering, whereas the
most complicated types of volumes, such as clouds, may require high amounts of multiple scattering and
complicated anisotropic phase functions. Rather than building a number of dierent volume integration
models into a single lighting integrator, it is helpful to decouple 

```

```

volume integration from the general
light integration problem: the integration domain can be broken into smaller volumetric subdomains
when participating media is involved, and control of integration of those domains given to smaller
volume modules.  

```
Color L = Color(0.0);
Color throughput = Color(1.0);
Ray ray = pickCameraDirection();
if (rs.GetNearestHit(ray, P, g))
continue;
int j = 0;
while (j < maxPathLength)
{
    ShadingContext *ctx = rs.CreateShadingContext(P, ray, g);
    Material *m = g.GetMaterial();
    BSDF *bsdf = m->CreateBSDF(*ctx);
    // Perform direct lighting on the surface
    L += throughput * directLighting();
    // Compute direction of indirect ray
    float pdf;
    Color Ls;
    Vector sampleDirection;
    bsdf->GenerateSample(rs, sampleDirection, Ls, pdf);
    throughput *= (Ls / pdf);
    Ray nextRay(ray);
    nextRay.org = P;
    nextRay.dir = sampleDirection;
    if (!rs.GetNearestHit(nextRay, P, g))
    break;
    ray = nextRay;
    j++;
}
```

上面是简单的path tracing，但是接下来我们可以使用

```
Volume *volume = 0;
if (m->HasVolume()) {
    // Did we go transmit through the surface? V is the
    // direction away from the point P on the surface.
    float VdotN = ctx->GetV().Dot(ctx->GetN());
    float dirDotN = sampleDirection.Dot(ctx->GetN());
    bool transmit = (VdotN < 0.0) != (dirDotN < 0.0);
    if (transmit) {
        // We transmitted through the surface. Check dot
        // product between the sample direction and the
        // surface normal N to see whether we entered or
        // exited the volume media
        bool entered = dirDotN < 0.0f;
        if (entered) {
        	nextRay.EnterMaterial(m);
        } else {
       		 nextRay.ExitMaterial(m);
        }
    }
    volume = nextRay.GetVolume(*ctx);
}
if (volume) {
    Color Lv;
    Color transmittance;
    float weight;
    if (!volume->Integrate(rs, nextRay, Lv, transmittance, weight, P, nextRay,
    g))
    break;
    L += weight * throughput * Lv;
    throughput *= transmittance;
} else {
	if (!rs.GetNearestHit(nextRay, P, g))
		break;
}
```

Integrate可以使用如下

```
virtual bool Integrate(RendererServices &rs, const Ray &wi, Color &L, Color
&transmittance, Color &weight, Point &P, Ray &wo, Geometry &g) {
    if (!rs.GetNearestHit(Ray(m_ctx.GetP(), wi.dir), P, g))
        return false;
        L = Color(0.0);
        transmittance = Transmittance(rs, P, m_ctx.GetP());
        weight = Color(1.0);
        wo = Ray(P, wi.dir);
        return true;
}
virtual Color Transmittance(RendererServices &rs, const Point &P0, const Point &P1) {
    float distance = Vector(P0 - P1).Length();
    return Color(exp(m_absorption.r * -distance), exp(m_absorption.g * -distance),
    exp(m_absorption.b * -distance));
}
```

single scattering

```
class SingleScatterHomogeneousVolume: public Volume {
public:
SingleScatterHomogeneousVolume(Color &scatteringAlbedo, Color &extinction,
ShadingContext &ctx) :
Volume(ctx), m_scatteringAlbedo(scatteringAlbedo), m_extinction(extinction) {}
virtual bool Integrate(RendererServices &rs, const Ray &wi, Color &L, Color
&transmittance, Color &weight, Point &P, Ray &wo, Geometry &g) {
if (!rs.GetNearestHit(Ray(m_ctx.GetP(), wi.dir), P, g))
return false;
// Transmittance over the entire interval
transmittance = Transmittance(rs, P, m_ctx.GetP());
// Compute sample location for scattering, based on the PDF
// normalized to the total transmission
float xi = rs.GenerateRandomNumber();
float scatterDistance = -logf(1.0f - xi * (1.0f - transmittance.ChannelAvg())) /
m_extinction.ChannelAvg();
// Set up shading context to be at the scatter location
Point Pscatter = m_ctx.GetP() + scatterDistance * wi.dir;
m_ctx.SetP(Pscatter);
m_ctx.RecomputeInputs();
// Compute direct lighting with light sampling and phase function sampling
IsotropicPhaseBSDF phaseBSDF(m_ctx);
L = Color(0.0);
Color lightL, bsdfL, beamTransmittance;
float lightPdf, bsdfPdf;
Vector sampleDirection;
rs.GenerateLightSample(m_ctx, sampleDirection, lightL, lightPdf, beamTransmittance);
phaseBSDF.EvaluateSample(rs, sampleDirection, bsdfL, bsdfPdf);
L += lightL * bsdfL * beamTransmittance * rs.MISWeight(1, lightPdf, 1, bsdfPdf) /
lightPdf;
phaseBSDF.GenerateSample(rs, sampleDirection, bsdfL, bsdfPdf);
rs.EvaluateLightSample(m_ctx, sampleDirection, lightL, lightPdf, beamTransmittance);
L += lightL * bsdfL * beamTransmittance * rs.MISWeight(1, lightPdf, 1, bsdfPdf) /
bsdfPdf;
Color Tr(exp(m_extinction.r * -scatterDistance), exp(m_extinction.g *
-scatterDistance), exp(m_extinction.b * -scatterDistance));
L *= (m_extinction * m_scatteringAlbedo * Tr);
// This weight is 1 over the PDF normalized to the total transmission
weight = (1 - transmittance) / (Tr * m_extinction);
wo = Ray(P, wi.dir);
return true;
}
virtual Color Transmittance(RendererServices &rs, const Point &P0, const Point &P1) {
float distance = Vector(P0 - P1).Length();
return Color(exp(m_extinction.r * -distance), exp(m_extinction.g * -distance),
exp(m_extinction.b * -distance));
}
protected:
const Color m_scatteringAlbedo;
const Color m_extinction;
};
```

