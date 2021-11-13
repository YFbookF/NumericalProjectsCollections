//BSSRDF Explorer: A rendering framework for the BSSRDF
analytic radial no_wi no_wo

vec3 lu;
vec3 alphaPrime;
vec3 SigmaTr;
float A;
void init()
{
	float Fdr = -1.44/(IOR * IOR) + 0.71/IOR + 0.668 + 0.0636*IOR;
	A = (1.0 + Fdr) / (1 - Fdr);
	vec3 SigmaSPrime = SigmaS *(1.0 - G);
	vec3 SigmaTPrime = SigmaSPrime + SigmaA;
	SigmaTr = sqrt(3 * SigmaA * SigmaTPrime);
	Lu = 1.0 / SigmaTPrime;
}

vec3 DiffuseReflectance(vec3 AlphaPrime)
{
	vec3 T = sqrt(3.0 * (1.0 - AlphaPrime));
	return AlphaPrime * 0.5 * (1.0 - exp(-4/3*A*T)) * exp(-T);
}

vec3 Bssrdf(vec3 Xi, vec Wi, vec3 Xo,vec3 Wo)
{
	vec3 R = vec3(dot(Xi - Xo,Xi - Xo));
	vec3 Zr = Lu;
	vec3 Zv = Lu * (1.0 + 4.0 / 3.0 * A);
	vec3 Dr = sqrt(R + zr * zr);
	vec3 Dv = sqrt(R + zv * zv);
	vec3 C1 = Zr * (SigmaTr + 1.0 / Dr);
	vec3 C2 = Zv * (SigmaTr + 1.0 / Dv);
	vec3 FluenceR = C1 * exp(-SigmaTr * Dr) / (Dr * Dr);
	vec3 FluenceV = C2 * exp(-SigmaTr * Dv) / (Dv * Dv);
	return Multiplier / (4.0 * Pi) * AlphaPrime * (FluenceR + FluenceV);
}