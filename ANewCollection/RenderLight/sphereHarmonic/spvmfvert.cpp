//http://www.cs.columbia.edu/cg/normalmap/SH.frag
//Notice: this sample vMF shader is mainly for reference purposes, therefore 
//doesn't guarantee compilability or optimal performance. 

// Variables passed to fragment shader
varying vec3 LightPos;
varying vec3 EyePos;
varying vec3 n;
varying vec3 t;
varying vec3 b;

// Varialbes passed in by the programe
uniform vec3 LightPosition;
attribute vec3 Tangent;

// Vertex shader main function
void main( void )
{	
	// Standard transformation
	EyePos = vec3 (gl_ModelViewMatrix * gl_Vertex);
	LightPos = vec3 (gl_ModelViewMatrix * vec4(LightPosition,0.0));

	gl_Position = ftransform();
	gl_TexCoord[0] = gl_MultiTexCoord0;


	// Vertex's local coordinate axis
	n = normalize(gl_NormalMatrix * gl_Normal);
	t = normalize(gl_NormalMatrix * Tangent);
	b = cross(n, t);
}
