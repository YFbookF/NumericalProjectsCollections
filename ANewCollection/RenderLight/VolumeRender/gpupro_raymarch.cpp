//gpu pro lighting
// One step of numerical solution to the light
// scattering equation
float4 AccumulateScattering(in float4 colorAndDensityFront,
                            in float4 colorAndDensityBack)
{
    // rgb = in - scattered light accumulated so far ,
    // a = accumulated scattering coefficient
    float3 light = colorAndDensityFront.rgb + saturate(
                                                  exp(-colorAndDensityFront.a)) *
                                                  colorAndDensityBack.rgb;
    return float4(light.rgb, colorAndDensityFront.a +
                                 colorAndDensityBack.a);
}
}
// Writing out final scattering values
}
void WriteOutput(in uint3 pos, in float4 colorAndDensity)
{
    // final value rgb = in - scattered light accumulated so far ,
    // a = scene light transmittance
    float4 finalValue = float4(colorAndDensity.rgb,
                               exp(-colorAndDensity.a));
    OutputTexture[pos].rgba = finalValue;
}
void RayMarchThroughVolume(uint3 dispatchThreadID)
{
    float4 currentSliceValue = InputTexture[uint3(dispatchThreadID.xy, 0)];
    WriteOutput(uint3(dispatchThreadID.xy, 0), currentSliceValue);
    for (uint z = 1; z < VOLUME{\_} DEPTH; z++)

    {
        uint3 volumePosition =uint3(dispatchThreadID.xy, z);

        float4 nextValue = InputTexture[volumePosition];

        currentSliceValue =AccumulateScattering(currentSliceValue, nextValue);

        WriteOutput(volumePosition, currentSliceValue);
    }
}
// Read volumetric in - scattering and transmittance
float4 scatteringInformation = tex3D(VolumetricFogSampler,
                                     positionInVolume);
float3 inScattering = scatteringInformation.rgb;
float transmittance = scatteringInformation.a;
// Apply to lit pixel
float3 finalPixelColor = pixelColorWithoutFog * transmittance.xxx + inScattering;