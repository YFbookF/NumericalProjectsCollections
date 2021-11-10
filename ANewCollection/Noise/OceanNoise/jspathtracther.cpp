//https://github.com/erichlof/THREE.js-PathTracing-Renderer
float hash( vec2 p )
{
	float h = dot(p,vec2(127.1,311.7));	
    	return fract(sin(h)*43758.5453123);
}

float noise( in vec2 p )
{
	vec2 i = floor( p );
	vec2 f = fract( p );	
	vec2 u = f*f*(3.0-2.0*f);
	return -1.0+2.0*mix( mix( hash( i + vec2(0.0,0.0) ), 
		     hash( i + vec2(1.0,0.0) ), u.x),
		mix( hash( i + vec2(0.0,1.0) ), 
		     hash( i + vec2(1.0,1.0) ), u.x), u.y);
}

float ocean_octave( vec2 uv, float choppy )
{
	uv += noise(uv);        
	vec2 wv = 1.0 - abs(sin(uv));
	vec2 swv = abs(cos(uv));    
	wv = mix(wv, swv, wv);
	return pow(1.0 - pow(wv.x * wv.y, 0.65), choppy);
}

float getOceanWaterHeight( vec3 p )
{
	float freq = WATER_FREQ;
	float amp = 1.0;
	float choppy = WATER_CHOPPY;
	float sea_time = uTime * WATER_SPEED;
	
	vec2 uv = p.xz * WATER_SAMPLE_SCALE; 
	//uv.x *= 0.75;
	float h, d = 0.0;    
	for(int i = 0; i < 1; i++)
	{        
		d =  ocean_octave((uv + sea_time) * freq, choppy);
		d += ocean_octave((uv - sea_time) * freq, choppy);
		h += d * amp;     
		uv *= M1; 
		freq *= 1.9; 
		amp *= 0.22;
		choppy = mix(choppy, 1.0, 0.2);
	}

	return h * WATER_WAVE_HEIGHT + uWaterLevel;
}

float getOceanWaterHeight_Detail( vec3 p )
{
	float freq = WATER_FREQ;
	float amp = 1.0;
	float choppy = WATER_CHOPPY;
	float sea_time = uTime * WATER_SPEED;
	
	vec2 uv = p.xz * WATER_SAMPLE_SCALE; 
	//uv.x *= 0.75;
	float h, d = 0.0;    
	for(int i = 0; i < 4; i++)
	{        
		d =  ocean_octave((uv + sea_time) * freq, choppy);
		d += ocean_octave((uv - sea_time) * freq, choppy);
		h += d * amp;     
		uv *= M1; 
		freq *= 1.9; 
		amp *= 0.22;
		choppy = mix(choppy, 1.0, 0.2);
	}

	return h * WATER_WAVE_HEIGHT + uWaterLevel;
}


float OceanIntersect()
{
	vec3 pos = rayOrigin;
	vec3 dir = normalize(rayDirection);
	float h = 0.0;
	float t = 0.0;
	
	for(int i = 0; i < 200; i++)
	{
		h = abs(pos.y - getOceanWaterHeight(pos));
		if (t > TERRAIN_FAR || h < 1.0) break;
		t += h;
		pos += dir * h; 
	}
	return (h <= 1.0) ? t : INFINITY;
}

vec3 ocean_calcNormal( vec3 pos, float t )
{
	vec3 eps = vec3(1.0, 0.0, 0.0);
	
	return normalize( vec3( getOceanWaterHeight_Detail(pos-eps.xyy) - getOceanWaterHeight_Detail(pos+eps.xyy),
			  	eps.x * 2.0,
			  	getOceanWaterHeight_Detail(pos-eps.yyx) - getOceanWaterHeight_Detail(pos+eps.yyx) ) );
}

