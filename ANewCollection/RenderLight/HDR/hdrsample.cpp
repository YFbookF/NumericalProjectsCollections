//https://github.com/erichlof/THREE.js-PathTracing-Renderer
vec3 Get_HDR_Color(vec3 rDirection)
{
	vec2 sampleUV;
	//sampleUV.y = asin(clamp(rDirection.y, -1.0, 1.0)) * ONE_OVER_PI + 0.5;
	///sampleUV.x = (1.0 + atan(rDirection.x, -rDirection.z) * ONE_OVER_PI) * 0.5;
	sampleUV.x = atan(rDirection.x, -rDirection.z) * ONE_OVER_TWO_PI + 0.5;
  	sampleUV.y = acos(rDirection.y) * ONE_OVER_PI;
	vec4 texData = texture(tHDRTexture, sampleUV);

	vec3 texColor = vec3(RGBEToLinear(texData));
	// texColor = texData.a > 0.57 ? vec3(100) : vec3(0);
	// return texColor;
	return texColor * uHDRI_Exposure;
}
