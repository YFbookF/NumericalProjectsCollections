//https://github.com/henrikdahlberg/GPUPathTracer
	__device__ inline glm::vec3 ReflectionDir(const glm::vec3 &normal,
											  const glm::vec3 &incident) {
		return 2.0f * dot(normal, incident) * normal - incident;
	}

	__device__ glm::vec3 TransmissionDir(const glm::vec3 &normal,
										 const glm::vec3 &incident,
										 float eta1, float eta2) {
		float cosTheta1 = dot(normal, incident);
		float r = eta1 / eta2;

		float radicand = 1.0f - powf(r, 2.0f) * (1.0f - powf(cosTheta1, 2.0f));

		if (radicand < 0.0f) { // total internal reflection
			return glm::vec3(0.0f); //temp, dont know what to do here
		}

		float cosTheta2 = sqrtf(radicand);
		return r*(-1.0f*incident) + (r*cosTheta1 - cosTheta2)*normal;
	}
