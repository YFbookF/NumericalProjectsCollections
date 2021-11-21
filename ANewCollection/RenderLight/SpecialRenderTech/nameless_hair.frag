#version 450
https://github.com/santipaprika/vulkan-hair-simulator
#define PI 3.1415926538

layout (location = 0) in vec3 fragColor;
layout (location = 1) in vec3 directionWS;
layout (location = 2) in vec3 positionWS;

layout (location = 0) out vec4 outColor;

layout (binding = 0) uniform UniformBufferObject {
    mat4 projectionView;
    mat4 model;
    mat4 normalMatrix;
    vec3 camPos;
} ubo;

layout(binding = 1) uniform sampler2D texSampler;

layout(push_constant) uniform Push {
    float brightness;
} push;

const vec3 DIRECTION_TO_LIGHT = normalize(vec3(3.0, -3.0, -1.0));
vec3 AMBIENT = vec3(0.0);


vec4 getAmbientAndDiffuse(vec4 lightColor0, vec4 diffuseColor, float angleLT)
{
    return (lightColor0 * diffuseColor * sin(angleLT) + vec4(AMBIENT,1.0));
}

vec4 getSpecular(vec4 lightColor0, vec4 specularColor, float angleLT, float angleVT, float specPower)
{
    float angleLTComp = PI - angleLT;
    return specularColor * pow(abs(cos(angleLTComp - angleVT)), specPower) * specularColor;
}

void main() {
    
    vec3 L = DIRECTION_TO_LIGHT;
    vec3 V = normalize(ubo.camPos - positionWS);
    vec3 T = normalize(directionWS);
    float angleLT = acos(dot(L,T));
    float angleVT = acos(dot(V,T));

    vec4 diffuseColor = vec4(fragColor, 1.0);
    vec4 specularColor = vec4(1.0);
    vec4 lightColor = vec4(1.0);
    vec4 diffuse = getAmbientAndDiffuse(lightColor, diffuseColor, angleLT);
    vec4 specular = getSpecular(lightColor, specularColor, angleLT, angleVT, 3);

    float kd = 0.8;
    float ks = 0.2;
    outColor = kd * diffuse + ks * specular;
}
