
// gpu pro lighting
// World - space position of volumetric texture texel
float3 worldPosition = CalcWorldPositionFromCoords(dispatchThreadID.xyz);
// Thickness of slice -- non - constant due to exponential slice
// distribution
float layerThickness = ComputeLayerThickness(dispatchThreadID.z);
// Estimated density of participating medium at given point
float dustDensity = CalculateDensityFunction(worldPosition);
// Scattering coefficient
float scattering = g_VolumetricFogScatteringCoefficient * dustDensity * layerThickness;
// Absorption coefficient
float absorption = g_VolumetricFogAbsorptionCoefficient * dustDensity * layerThickness;
// Normalized view direction
float3 viewDirection = normalize(worldPosition - g_WorldEyePos.xyz);
float3 lighting = 0.0 f;
// Lighting section BEGIN
// Adding all contributing lights radiance and multiplying it by
// a phase function -- volumetric fog equivalent of BRDFs
lighting += GetSunLightingRadiance(worldPosition) * GetPhaseFunction(viewDirection, g_SunDirection,
                                                                     g_VolumetricFogPhaseAnisotropy);
lighting += GetAmbientConvolvedWithPhaseFunction(worldPosition,
                                                 viewDirection, g_VolumetricFogPhaseAnisotropy);
[loop] for (int lightIndex = 0; lightIndex < g_LightsCount; ++lightIndex)
{
    float3 localLightDirection =
        GetLocalLightDirection(lightIndex, worldPosition);
    lighting += GetLocalLightRadiance(lightIndex, worldPosition) * GetPhaseFunction(viewDirection, localLightDirection,
                                                                                    g_VolumetricFogPhaseAnisotropy);
}
// Lighting section END
// Finally , we apply some potentially non - white fog scattering albedo
color lighting *= g_FogAlbedo;
// Final in - scattering is product of outgoing radiance and scattering
// coefficients , while extinction is sum of scattering and absorption
float4 finalOutValue = float4(lighting * scattering, scattering + absorption);