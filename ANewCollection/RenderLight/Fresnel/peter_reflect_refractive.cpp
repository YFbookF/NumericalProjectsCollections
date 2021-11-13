//https://github.com/peterkutz/GPUPathTracer
__host__ __device__
	Fresnel
	computeFresnel(const float3 &normal, const float3 &incident, float refractiveIndexIncident, float refractiveIndexTransmitted, const float3 &reflectionDirection, const float3 &transmissionDirection)
{
	Fresnel fresnel;

	// First, check for total internal reflection:
	if (length(transmissionDirection) <= 0.12345 || dot(normal, transmissionDirection) > 0)
	{ // The length == 0 thing is how we're handling TIR right now.
		// Total internal reflection!
		fresnel.reflectionCoefficient = 1;
		fresnel.transmissionCoefficient = 0;
		return fresnel;
	}

	// Real Fresnel equations:
	// Copied from Photorealizer.
	float cosThetaIncident = dot(normal, incident);
	float cosThetaTransmitted = dot(-1 * normal, transmissionDirection);
	float reflectionCoefficientSPolarized = pow((refractiveIndexIncident * cosThetaIncident - refractiveIndexTransmitted * cosThetaTransmitted) / (refractiveIndexIncident * cosThetaIncident + refractiveIndexTransmitted * cosThetaTransmitted), 2);
	float reflectionCoefficientPPolarized = pow((refractiveIndexIncident * cosThetaTransmitted - refractiveIndexTransmitted * cosThetaIncident) / (refractiveIndexIncident * cosThetaTransmitted + refractiveIndexTransmitted * cosThetaIncident), 2);
	float reflectionCoefficientUnpolarized = (reflectionCoefficientSPolarized + reflectionCoefficientPPolarized) / 2.0; // Equal mix.
	//
	fresnel.reflectionCoefficient = reflectionCoefficientUnpolarized;
	fresnel.transmissionCoefficient = 1 - fresnel.reflectionCoefficient;
	return fresnel;

	/*
	// Shlick's approximation including expression for R0 and modification for transmission found at http://www.bramz.net/data/writings/reflection_transmission.pdf
	// TODO: IMPLEMENT ACTUAL FRESNEL EQUATIONS!
	float R0 = pow( (refractiveIndexIncident - refractiveIndexTransmitted) / (refractiveIndexIncident + refractiveIndexTransmitted), 2 ); // For Schlick's approximation.
	float cosTheta;
	if (refractiveIndexIncident <= refractiveIndexTransmitted) {
		cosTheta = dot(normal, incident);
	} else {
		cosTheta = dot(-1 * normal, transmissionDirection); // ???
	}
	fresnel.reflectionCoefficient = R0 + (1.0 - R0) * pow(1.0 - cosTheta, 5); // Costly pow function might make this slower than actual Fresnel equations. TODO: USE ACTUAL FRESNEL EQUATIONS!
	fresnel.transmissionCoefficient = 1.0 - fresnel.reflectionCoefficient;
	return fresnel;
	*/
}