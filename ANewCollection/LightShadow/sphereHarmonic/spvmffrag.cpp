//http://www.cs.columbia.edu/cg/normalmap/SH.frag
//Notice: this sample vMF shader is mainly for reference purposes, therefore 
//doesn't guarantee compilability or optimal performance. 


// Variables passed in by the vertex shader
varying vec3 LightPos;
varying vec3 VertPos;
varying vec3 n;
varying vec3 t;
varying vec3 b;


// Varialbes passed in by the programe
uniform sampler2D vMFmap1;
uniform sampler2D vMFmap2;
uniform sampler2D vMFmap3;
uniform sampler2D vMFmap4;
uniform sampler2D vMFmap5;
uniform sampler2D vMFmap6;
uniform int SpecularExponent;

// Fragment main function
void main( void )
{  
	vec3 v;
	// Eye direction in the vertex's local coordinate
	v.x = dot(EyePos, t);
	v.y = dot(EyePos, b);
	v.z = dot(EyePos, n);
	vec3  EyeDir = normalize(v);

	// Light direction in the vertex's local coordinate
	v.x = dot(LightPos, t);
	v.y = dot(LightPos, b);
	v.z = dot(LightPos, n);
	vec3 LightDir = normalize(v);

	// Half angle in the vertex's local coordiante
	vec3 halfAngle = normalize(-EyeDir + LightDir);

	// Calcualte mipmap level
	const float delta = 1.0 / 11.0;
	float level = texture2D(Mipmap,texcoord).r / delta;

	// Look up vMF lobes (6 lobes here) in 2D textures
	vec4 coeffs[6];
	coeffs[0] = texture2D(vMFmap1, gl_TexCoord[0].st);
	coeffs[1] = texture2D(vMFmap2, gl_TexCoord[0].st);
	coeffs[2] = texture2D(vMFmap3, gl_TexCoord[0].st);
	coeffs[3] = texture2D(vMFmap4, gl_TexCoord[0].st);
	coeffs[4] = texture2D(vMFmap5, gl_TexCoord[0].st);
	coeffs[5] = texture2D(vMFmap6, gl_TexCoord[0].st);

	// Add up shading contributions from all lobes	
	vec3 result = render(coeffs,halfAngle,LightDir);

  	gl_FragColor = vec4(result, 1.0);
}

// Helper function that adds up the shading contributions from all lobes
vec3 render(vec4 coeffs[6], vec3 halfAngle, vec3 lightAngle) 
{
  	vec3 ret = vec3(0.0);
	for (int i=0; i<6; i++) {
		ret += evaluatePeak(coeffs[i], colors[i].xyz, halfAngle, lightAngle);
  	}
  
  return ret;
}

// Helper frunction that computes shading for one lobe
vec3 evaluatePeak(vec4 peak, vec3 halfAngle, vec3 lightAngle) 
{
	// vMF parameters alpha, mu are stored in the texture
	// parameter kappa can be computed from mu
	float alpha = peak.x;
	vec3 mu = 2.0 * peak.yzw - 1.0;
	mu /= alpha;
	float r = length(mu);
	mu = normalize(mu);
	float kappa = (3.0*r - r*r*r)/(1.0 - r*r);

	// Calcualte shading per equation 30 in the paper
	float sprime = kappa*SpecularExponent / (kappa+SpecularExponent);
	float norm = (sprime + 1.0) / (2.0 * pi);
 	float ret = pow(ret,sprime);
	ret *= norm;
	ret *= max(dot(lightAngle,mu),0.0);

	return ret;
}