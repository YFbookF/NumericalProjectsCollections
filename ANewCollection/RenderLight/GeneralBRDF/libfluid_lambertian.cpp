https://github.com/lukedan/libfluid
		spectrum lambertian_reflection_brdf::f(vec3d in, vec3d out, transport_mode) const {
			return in.y * out.y > 0.0 ? reflectance / constants::pi : spectrum();
		}

		outgoing_ray_sample lambertian_reflection_brdf::sample_f(
			vec3d norm_in, vec2d random, transport_mode mode
		) const {
			outgoing_ray_sample result;
			result.norm_out_direction_tangent = warping::unit_hemisphere_from_unit_square_cosine(random);
			result.pdf = warping::pdf_unit_hemisphere_from_unit_square_cosine(result.norm_out_direction_tangent);
			if constexpr (double_sided) {
				if (norm_in.y < 0.0) {
					result.norm_out_direction_tangent.y = -result.norm_out_direction_tangent.y;
				}
			}
			result.reflectance = f(norm_in, result.norm_out_direction_tangent, mode);
			return result;
		}

		double lambertian_reflection_brdf::pdf(vec3d norm_in, vec3d norm_out) const {
			if constexpr (double_sided) {
				if ((norm_in.y > 0) == (norm_out.y > 0)) {
					norm_out.y = std::abs(norm_out.y);
					return warping::pdf_unit_hemisphere_from_unit_square_cosine(norm_out);
				}
				return 0.0;
			} else {
				return warping::pdf_unit_hemisphere_from_unit_square_cosine(norm_out);;
			}
		}

		bool lambertian_reflection_brdf::is_delta() const {
			return false;
		}