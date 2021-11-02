//http://www.cs.columbia.edu/cg/normalmap/SH.frag
//Notice: this sample vMF shader is mainly for reference purposes, therefore 
//doesn't guarantee compilability or optimal performance. 


// Variables passed in by the vertex shader
varying vec3 LightPos;
varying vec3 EyePos;
varying vec3 n;
varying vec3 t;
varying vec3 b;

// Varialbes passed in by the programe
uniform sampler2D SHmap1;
uniform sampler2D SHmap2;
uniform sampler2D SHmap3;
uniform sampler2D SHmap4;
uniform vec2 SHmapRange;
uniform sampler2D Ylm1;
uniform sampler2D Ylm2;
uniform sampler2D Ylm3;
uniform sampler2D Ylm4;
uniform vec2 YlmRange;

uniform sampler2D Mtl;
uniform vec2 MtlRange;
uniform mat4 Llm1;
uniform mat4 Llm2;
uniform mat4 Llm3;
uniform mat4 Llm4;
uniform vec4 rho1;
uniform vec4 rho2;

uniform vec3 diffColor;
uniform float exposure;
const float pi = 3.141592653589793;
const float a0 = 3.141593;
const float a1 = 2.094395;
const float a2 = 0.785398;
const float a3 = 0.0;
const float a4 = -0.130900;
const float a5 = 0.0;
const float a6 = 0.049087;
const float a7 = 0.0;

// Fragment main function
void main( void )
{
	vec3 v;
	// Eye direction in the vertex's local coordinate
    	v.x = dot(EyePos, t);
    	v.y = dot(EyePos, b);
    	v.z = dot(EyePos, n);
    	vec3 EyeDir = normalize(v);

	// Light direction in the vertex's local coordinate
    	v.x = dot(LightDir, t);
    	v.y = dot(LightDir, b);
    	v.z = dot(LightDir, n);
    	vec3 LightDir = normalize(v);
    
    	mat4 ylm1=mat4(0.0); mat4 ylm2=mat4(0.0); mat4 ylm3=mat4(0.0); mat4 ylm4=mat4(0.0);
    	mat4 sh1;  mat4 sh2;  mat4 sh3;  mat4 sh4;
    	vec2 coord0; vec2 coord1; vec2 coord2; vec2 coord3;
    	vec4 col1; vec4 col2; vec4 col3; vec4 col4;
    
	// Computer half angle
	vec3 HalfAngle = normalize(-EyeDir + LightDir);
	float theta_h = acos( HalfAngle.z );
	float phi_h = atan( HalfAngle.y, HalfAngle.x );

    	// Calculate Ylm texture coordinates
	vec2 YlmCoord = vec2(phi_h, theta_h);
    	YlmCoord *= vec2(1.0 / (2.0 * pi), 1.0 / pi);
	YlmCoord.x += 0.5;
    	coord0 = YlmCoord;
    	coord0.t = fract(coord0.t) * 0.125 + 0.0625;
	coord1 = coord0;
    	coord1.t = coord1.t + 0.25;
    	coord2 = coord1;
    	coord2.t = coord2.t + 0.25;
    	coord3 = coord2;
    	coord3.t = coord3.t + 0.25;
    	float fScale=YlmRange[1]-YlmRange[0];
    	float fOffset=YlmRange[0];
        
	// Ylm texture look up, each texture contain 16 coefficients, therefore 64 in total for four textures
	col1 = texture2D(Ylm1, coord0,-0.0);
    	col2 = texture2D(Ylm1, coord1,-0.0);
    	col3 = texture2D(Ylm1, coord2,-0.0);
    	col4 = texture2D(Ylm1, coord3,-0.0);
    	ylm1 = mat4(col1,col2,col3,col4);
    	ylm1 = fScale * ylm1 +fOffset;
    	col1 = texture2D(Ylm2, coord0,-0.0);
    	col2 = texture2D(Ylm2, coord1,-0.0);
    	col3 = texture2D(Ylm2, coord2,-0.0);
    	col4 = texture2D(Ylm2, coord3,-0.0);
    	ylm2 = mat4(col1,col2,col3,col4);
    	ylm2 = fScale * ylm2 +fOffset;
    	col1 = texture2D(Ylm3, coord0,-0.0);
    	col2 = texture2D(Ylm3, coord1,-0.0);
    	col3 = texture2D(Ylm3, coord2,-0.0);
    	col4 = texture2D(Ylm3, coord3,-0.0);
    	ylm3 = mat4(col1,col2,col3,col4);
    	ylm3 = fScale * ylm3 +fOffset;
    	col1 = texture2D(Ylm4, coord0,-0.0);
	col2 = texture2D(Ylm4, coord1,-0.0);
    	col3 = texture2D(Ylm4, coord2,-0.0);
    	col4 = texture2D(Ylm4, coord3,-0.0);
    	ylm4 = mat4(col1,col2,col3,col4);
    	ylm4 = fScale * ylm4 +fOffset;

    	// Calculate normal map texture coordinates
    	coord0 = gl_TexCoord[0].st;
    	coord0.t = fract(coord0.t) * 0.125 + 0.0625;
    	coord1 = coord0;
    	coord1.t = coord1.t + 0.25;
    	coord2 = coord1;
    	coord2.t = coord2.t + 0.25;
    	coord3 = coord2;
    	coord3.t = coord3.t + 0.25;
    	fScale=SHmapRange[1]-SHmapRange[0];
    	fOffset=SHmapRange[0];
    	
	// Normal map texture look up, each texture contain 16 coefficients, therefore 64 in total for four textures
    	col1 = texture2D(SHmap1, coord0, 0.0);
    	col2 = texture2D(SHmap1, coord1, 0.0);
    	col3 = texture2D(SHmap1, coord2, 0.0);
    	col4 = texture2D(SHmap1, coord3, 0.0);
    	sh1 = mat4(col1,col2,col3,col4);    
    	sh1 = fScale * sh1 +fOffset;
    	col1 = texture2D(SHmap2, coord0, 0.0);
    	col2 = texture2D(SHmap2, coord1, 0.0);
    	col3 = texture2D(SHmap2, coord2, 0.0);
    	col4 = texture2D(SHmap2, coord3, 0.0);
    	sh2 = mat4(col1,col2,col3,col4);    
    	sh2 = fScale * sh2 +fOffset;
    	col1 = texture2D(SHmap3, coord0, 0.0);
    	col2 = texture2D(SHmap3, coord1, 0.0);
    	col3 = texture2D(SHmap3, coord2, 0.0);
    	col4 = texture2D(SHmap3, coord3, 0.0);
    	sh3 = mat4(col1,col2,col3,col4);    
    	sh3 = fScale * sh3 +fOffset;
    	col1 = texture2D(SHmap4, coord0, 0.0);
    	col2 = texture2D(SHmap4, coord1, 0.0);
    	col3 = texture2D(SHmap4, coord2, 0.0);
    	col4 = texture2D(SHmap4, coord3, 0.0);
    	sh4 = mat4(col1,col2,col3,col4);
    	sh4 = fScale * sh4 +fOffset;
    
    	// brdf coefficients
    	float b0 = rho1.x;
    	float b1 = rho1.y;
    	float b2 = rho1.z;
    	float b3 = rho1.w;
    	float b4 = rho2.x;
    	float b5 = rho2.y;
    	float b6 = rho2.z;
    	float b7 = rho2.w;
    	mat4 B1 = mat4(b0,b1,b1,b1,
    	      	b2,b2,b2,b2,
			b2,b3,b3,b3,
     	       	b3,b3,b3,b3);
    	mat4 B2 = mat4(b4,b4,b4,b4,
            	b4,b4,b4,b4,
            	b4,b5,b5,b5,
            	b5,b5,b5,b5);
    	mat4 B3 = mat4(b5,b5,b5,b5,
            	b6,b6,b6,b6,
            	b6,b6,b6,b6,
            	b6,b6,b6,b6);
    	mat4 B4 = mat4(b6,b7,b7,b7,
            	b7,b7,b7,b7,
            	b7,b7,b7,b7,
            	b7,b7,b7,b7);
        
   	// dot product , equation 12 in the paper
    	mat4 spec1 = matrixCompMult(matrixCompMult(sh1, ylm1), B1);
    	mat4 spec2 = matrixCompMult(matrixCompMult(sh2, ylm2), B2);
    	mat4 spec3 = matrixCompMult(matrixCompMult(sh3, ylm3), B3);
    	mat4 spec4 = matrixCompMult(matrixCompMult(sh4, ylm4), B4);
    	mat4 specm = spec1 + spec2 + spec3 + spec4;
    	float spec  = specm[0][0] + specm[0][1] + specm[0][2] + specm[0][3] +
    	specm[1][0] + specm[1][1] + specm[1][2] + specm[1][3] +
    	specm[2][0] + specm[2][1] + specm[2][2] + specm[2][3] +
    	specm[3][0] + specm[3][1] + specm[3][2] + specm[3][3];
 

	vec3 result = vec3(max(spec,0.0));

	gl_FragColor = vec4(result,0.0);
 }
