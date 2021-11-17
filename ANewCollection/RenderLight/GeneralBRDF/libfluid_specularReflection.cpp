https://github.com/lukedan/libfluid

		spectrum specular_reflection_brdf::f(vec3d, vec3d, transport_mode) const {
			return spectrum();
		}

		outgoing_ray_sample specular_reflection_brdf::sample_f(vec3d norm_in, vec2d, transport_mode) const {
			outgoing_ray_sample result;
			result.norm_out_direction_tangent.x = -norm_in.x;
			result.norm_out_direction_tangent.y = norm_in.y;
			result.norm_out_direction_tangent.z = -norm_in.z;
			result.pdf = 1.0;
			result.reflectance = reflectance / std::abs(norm_in.y); // cancel out Lambertian term
			return result;
		}

		double specular_reflection_brdf::pdf(vec3d, vec3d) const {
			return 0.0;
		}

		bool specular_reflection_brdf::is_delta() const {
			return true;
		}
