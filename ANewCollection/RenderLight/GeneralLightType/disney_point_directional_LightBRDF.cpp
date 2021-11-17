// disney brdf explorer
vec3 computeWithDirectionalLight( vec3 surfPt, vec3 incidentVector, vec3 viewVec, vec3 normal, vec3 tangent, vec3 bitangent )
{
    // evaluate the BRDF
    vec3 b = max( BRDF( incidentVector, viewVec, normal, tangent, bitangent ), vec3(0.0) );

    // multiply in the cosine factor
    if (useNDotL != 0)
        b *= dot( normal, incidentVector );

    return b;
}


vec3 computeWithPointLight( vec3 surfPt, vec3 incidentVector, vec3 viewVec, vec3 normal, vec3 tangent, vec3 bitangent )
{
    // compute the point light vector
    vec3 toLight = (incidentVector * lightDistanceFromCenter) - surfPt;
    float pointToLightDist = length( toLight );
    toLight /= pointToLightDist;


    // evaluate the BRDF
    vec3 b = max( BRDF( toLight, viewVec, normal, tangent, bitangent ), vec3(0.0) );

    // multiply in the cosine factor
    if (useNDotL != 0)
        b *= dot( normal, toLight );

    // multiply in the falloff
    b *= (1.0 / (pointToLightDist*pointToLightDist));

    return b;
}