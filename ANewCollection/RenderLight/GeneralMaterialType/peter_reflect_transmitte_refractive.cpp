//https://github.com/peterkutz/GPUPathTracer
		bool doSpecular = (bestMaterial.medium.refractiveIndex > 1.0); // TODO: Move?
		float rouletteRandomFloat = uniformDistribution(rng);
		// TODO: Fix long conditional, and maybe lots of temporary variables.
		// TODO: Optimize total internal reflection case (no random number necessary in that case).
		bool reflectFromSurface = (doSpecular && rouletteRandomFloat < computeFresnel(bestNormal, incident, incidentMedium.refractiveIndex, transmittedMedium.refractiveIndex, reflectionDirection, transmissionDirection).reflectionCoefficient);
		if (reflectFromSurface)
		{
			// Ray reflected from the surface. Trace a ray in the reflection direction.

			// TODO: Use Russian roulette instead of simple multipliers! (Selecting between diffuse sample and no sample (absorption) in this case.)
			notAbsorbedColors[pixelIndex] *= bestMaterial.specularColor;

			Ray nextRay;
			nextRay.origin = bestIntersectionPoint + biasVector;
			nextRay.direction = reflectionDirection;
			rays[pixelIndex] = nextRay; // Only assigning to global memory ray once, for better performance.
		}
		else if (bestMaterial.hasTransmission)
		{
			// Ray transmitted and refracted.

			// The ray has passed into a new medium!
			absorptionAndScattering[pixelIndex] = transmittedMedium.absorptionAndScatteringProperties;

			Ray nextRay;
			nextRay.origin = bestIntersectionPoint - biasVector; // Bias ray in the other direction because it's transmitted!!!
			nextRay.direction = transmissionDirection;
			rays[pixelIndex] = nextRay; // Only assigning to global memory ray once, for better performance.
		}
		else
		{
			// Ray did not reflect from the surface, so consider emission and take a diffuse sample.

			// TODO: Use Russian roulette instead of simple multipliers! (Selecting between diffuse sample and no sample (absorption) in this case.)
			accumulatedColors[pixelIndex] += notAbsorbedColors[pixelIndex] * bestMaterial.emittedColor;
			notAbsorbedColors[pixelIndex] *= bestMaterial.diffuseColor;

			// Choose a new ray direction:
			float randomFloat1 = uniformDistribution(rng);
			float randomFloat2 = uniformDistribution(rng);
			Ray nextRay;
			nextRay.origin = bestIntersectionPoint + biasVector;
			nextRay.direction = randomCosineWeightedDirectionInHemisphere(bestNormal, randomFloat1, randomFloat2);
			rays[pixelIndex] = nextRay; // Only assigning to global memory ray once, for better performance.
		}