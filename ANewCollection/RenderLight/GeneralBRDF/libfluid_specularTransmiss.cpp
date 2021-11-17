https://github.com/lukedan/libfluid
spectrum specular_transmission_bsdf::f(vec3d, vec3d, transport_mode) const {
			return spectrum();
		}

		outgoing_ray_sample specular_transmission_bsdf::sample_f(
			vec3d norm_in, vec2d random, transport_mode mode
		) const {
			outgoing_ray_sample result;
			double eta_in = 1.0, eta_out = index_of_refraction, cos_in = norm_in.y, sign = 1.0;
			if (cos_in < 0.0) {
				std::swap(eta_in, eta_out);
				cos_in = -cos_in;
				sign = -1.0;
			}
			double eta = eta_in / eta_out;
			double sin2_out = (1.0 - cos_in * cos_in) * eta * eta;
			if (sin2_out >= 1.0) { // total internal reflection
				result.norm_out_direction_tangent = vec3d(-norm_in.x, norm_in.y, -norm_in.z);
				result.pdf = 1.0;
				result.reflectance = skin / cos_in;
				return result;
			}
			double cos_out = std::sqrt(1.0 - sin2_out);
			double fres = fresnel::dielectric(cos_in, cos_out, eta_in, eta_out);
			if (random.x > fres) { // refraction
				result.norm_out_direction_tangent = -eta * norm_in;
				result.norm_out_direction_tangent.y += (eta * cos_in - cos_out) * sign;
				result.pdf = 1.0 - fres;
				result.reflectance = (1.0 - fres) * skin / cos_out;
				if (mode == transport_mode::radiance) {
					result.reflectance *= eta * eta;
				}
			} else { // reflection
				result.norm_out_direction_tangent = vec3d(-norm_in.x, norm_in.y, -norm_in.z);
				result.pdf = fres;
				result.reflectance = fres * skin / cos_in;
			}
			return result;
		}

		double specular_transmission_bsdf::pdf(vec3d, vec3d) const {
			return 0.0;
		}

		bool specular_transmission_bsdf::is_delta() const {
			return true;
		}
	}