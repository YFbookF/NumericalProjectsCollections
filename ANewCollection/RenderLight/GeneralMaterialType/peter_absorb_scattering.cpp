//https://github.com/peterkutz/GPUPathTracer
	// ABSORPTION AND SCATTERING:
	{ // BEGIN SCOPE.
		AbsorptionAndScatteringProperties currentAbsorptionAndScattering = absorptionAndScattering[pixelIndex];
		#define ZERO_ABSORPTION_EPSILON 0.00001
		if ( currentAbsorptionAndScattering.reducedScatteringCoefficient > 0 || dot(currentAbsorptionAndScattering.absorptionCoefficient, currentAbsorptionAndScattering.absorptionCoefficient) > ZERO_ABSORPTION_EPSILON ) { // The dot product with itself is equivalent to the squre of the length.
			float randomFloatForScatteringDistance = uniformDistribution(rng);
			float scatteringDistance = -log(randomFloatForScatteringDistance) / absorptionAndScattering[pixelIndex].reducedScatteringCoefficient;
			if (scatteringDistance < bestT) {
				// Both absorption and scattering.

				// Scatter the ray:
				Ray nextRay;
				nextRay.origin = positionAlongRay(currentRay, scatteringDistance);
				float randomFloatForScatteringDirection1 = uniformDistribution(rng);
				float randomFloatForScatteringDirection2 = uniformDistribution(rng);
				nextRay.direction = randomDirectionInSphere(randomFloatForScatteringDirection1, randomFloatForScatteringDirection2); // Isoptropic scattering!
				rays[pixelIndex] = nextRay; // Only assigning to global memory ray once, for better performance.

				// Compute how much light was absorbed along the ray before it was scattered:
				notAbsorbedColors[pixelIndex] *= computeTransmission(currentAbsorptionAndScattering.absorptionCoefficient, scatteringDistance);

				// DUPLICATE CODE:
				// To assist Thrust stream compaction, set this activePixel to -1 if the ray weight is now zero:
				if (length(notAbsorbedColors[pixelIndex]) <= MIN_RAY_WEIGHT) { // TODO: Faster: dot product of a vector with itself is the same as its length squared.
					activePixels[activePixelIndex] = -1;
				}

				// That's it for this iteration!
				return; // IMPORTANT!
			} else {
				// Just absorption.

				notAbsorbedColors[pixelIndex] *= computeTransmission(currentAbsorptionAndScattering.absorptionCoefficient, bestT);

				// Now proceed to compute interaction with intersected object and whatnot!
			}
		}