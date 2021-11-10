//pbrt
THREE.ShaderChunk[ 'pathtracing_physical_sky_functions' ] = `
float RayleighPhase(float cosTheta)
{
	return THREE_OVER_SIXTEENPI * (1.0 + (cosTheta * cosTheta));
}
float hgPhase(float cosTheta, float g)
{
        float g2 = g * g;
        float inverse = 1.0 / pow(max(0.0, 1.0 - 2.0 * g * cosTheta + g2), 1.5);
	return ONE_OVER_FOURPI * ((1.0 - g2) * inverse);
}
vec3 totalMie()
{
	float c = (0.2 * TURBIDITY) * 10E-18;
	return 0.434 * c * MIE_CONST;
}
float SunIntensity(float zenithAngleCos)
{
	zenithAngleCos = clamp( zenithAngleCos, -1.0, 1.0 );
	return SUN_POWER * max( 0.0, 1.0 - pow( E, -( ( CUTOFF_ANGLE - acos( zenithAngleCos ) ) / STEEPNESS ) ) );
}
vec3 Get_Sky_Color(Ray r, vec3 sunDirection)
{
	vec3 viewDirection = normalize(r.direction);
	
	/* most of the following code is borrowed from the three.js shader file: SkyShader.js */
    	// Cosine angles
	float cosViewSunAngle = dot(viewDirection, normalize(sunDirection));
    	float cosSunUpAngle = dot(UP_VECTOR, normalize(sunDirection)); // allowed to be negative: + is daytime, - is nighttime
    	float cosUpViewAngle = dot(UP_VECTOR, viewDirection);
	
        // Get sun intensity based on how high in the sky it is
    	float sunE = SunIntensity(cosSunUpAngle);
        
	// extinction (absorbtion + out scattering)
	// rayleigh coefficients
    	vec3 rayleighAtX = TOTAL_RAYLEIGH * RAYLEIGH_COEFFICIENT;
    
	// mie coefficients
	vec3 mieAtX = totalMie() * MIE_COEFFICIENT;  
    
	// optical length
	float zenithAngle = acos( max( 0.0, dot( UP_VECTOR, viewDirection ) ) );
	float inverse = 1.0 / ( cos( zenithAngle ) + 0.15 * pow( 93.885 - ( ( zenithAngle * 180.0 ) / PI ), -1.253 ) );
	float rayleighOpticalLength = RAYLEIGH_ZENITH_LENGTH * inverse;
	float mieOpticalLength = MIE_ZENITH_LENGTH * inverse;
	// combined extinction factor	
	vec3 Fex = exp(-(rayleighAtX * rayleighOpticalLength + mieAtX * mieOpticalLength));
	// in scattering
	vec3 betaRTheta = rayleighAtX * RayleighPhase(cosViewSunAngle * 0.5 + 0.5);
	vec3 betaMTheta = mieAtX * hgPhase(cosViewSunAngle, MIE_DIRECTIONAL_G);
	
	vec3 Lin = pow( sunE * ( ( betaRTheta + betaMTheta ) / ( rayleighAtX + mieAtX ) ) * ( 1.0 - Fex ), vec3( 1.5 ) );
	Lin *= mix( vec3( 1.0 ), pow( sunE * ( ( betaRTheta + betaMTheta ) / ( rayleighAtX + mieAtX ) ) * Fex, vec3( 1.0 / 2.0 ) ), clamp( pow( 1.0 - cosSunUpAngle, 5.0 ), 0.0, 1.0 ) );
	// nightsky
	float theta = acos( viewDirection.y ); // elevation --> y-axis, [-pi/2, pi/2]
	float phi = atan( viewDirection.z, viewDirection.x ); // azimuth --> x-axis [-pi/2, pi/2]
	vec2 uv = vec2( phi, theta ) / vec2( 2.0 * PI, PI ) + vec2( 0.5, 0.0 );
	vec3 L0 = vec3( 0.1 ) * Fex;
	// composition + solar disc
	float sundisk = smoothstep( SUN_ANGULAR_DIAMETER_COS, SUN_ANGULAR_DIAMETER_COS + 0.00002, cosViewSunAngle );
	L0 += ( sunE * 19000.0 * Fex ) * sundisk;
	vec3 texColor = ( Lin + L0 ) * 0.04 + vec3( 0.0, 0.0003, 0.00075 );
	float sunfade = 1.0 - clamp( 1.0 - exp( ( sunDirection.y / 450000.0 ) ), 0.0, 1.0 );
	vec3 retColor = pow( texColor, vec3( 1.0 / ( 1.2 + ( 1.2 * sunfade ) ) ) );
	return retColor;
}