//https://github.com/henrikdahlberg/GPUPathTracer
	//////////////////////////////////////////////////////////////////////////
	// Device Kernels
	//////////////////////////////////////////////////////////////////////////
	__device__ glm::vec3 HemisphereCosSample(const glm::vec3 &normal,
											 const float r1,
											 const float r2) {
		float c = sqrtf(r1);
		float phi = r2 * M_2PI;

		glm::vec3 w = fabs(normal.x) < M_SQRT1_3 ? glm::vec3(1, 0, 0) : (fabs(normal.y) < M_SQRT1_3 ? glm::vec3(0, 1, 0) : glm::vec3(0, 0, 1));
		glm::vec3 u = normalize(cross(normal, w));
		glm::vec3 v = cross(normal, u);

		return sqrtf(1.0f - r1) * normal + (cosf(phi) * c * u) + (sinf(phi) * c * v);

	}