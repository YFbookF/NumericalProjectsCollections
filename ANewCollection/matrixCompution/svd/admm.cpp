//Projective Dynamics: Fast Simulation of Hyperelastic Models with Dynamic Constraints
//http://www.github.com/mattoverby/admm-elastic
void HyperElasticTet::prox( VecX &zi ){
	typedef Matrix<double,9,1> Vec9;
	typedef Matrix<double,3,3> Mat3;
	Prox *problem = get_problem();

	Mat3 F = Map<Mat3>(zi.data());
	Vec3 S; Mat3 U, V;
	signed_svd( F, S, U, V );
	problem->set_x0(S);

	// If everything is very low, It is collapsed to a point and the minimize
	// will likely fail. So we'll just inflate it a bit.
	const double eps = 1e-6;
	if( std::abs(S[0]) < eps && std::abs(S[1]) < eps && std::abs(S[2]) < eps ){
		S[0] = eps; S[1] = eps; S[2] = eps;
	}

	if( S[2] < 0.0 ){ S[2] = -S[2]; }

	solver.minimize( *problem, S );
	Mat3 matp = U * S.asDiagonal() * V.transpose();
	zi = Map<Vec9>(matp.data());
}
// Relevent papers:
// Computing the Singular Value Decomposition of 3x3 matrices with minimal branching and elementary floating point operations, McAdams et al.
// Energetically Consistent Invertible Elasticity, Stomakhin et al.
// Invertible Finite Elements For Robust Simulation of Large Deformation, Irving et al.
//
	// Projection, Singular Values, SVD's U, SVD's V
	template <typename T>
	static inline void signed_svd( const fsvd::Mat3<T> &F, fsvd::Vec3<T> &S, fsvd::Mat3<T> &U, fsvd::Mat3<T> &V ){
		using namespace Eigen;

		JacobiSVD< fsvd::Mat3<T> > svd( F, ComputeFullU | ComputeFullV );
		S = svd.singularValues();
		U = svd.matrixU();
		V = svd.matrixV();
		fsvd::Mat3<T> J = Matrix3d::Identity();
		J(2,2) = -1.0;

		// Check for inversion: U
		if( U.determinant() < 0.0 ){
			U = U * J;
			S[2] = -S[2];
		}

		// Check for inversion: V
		if( V.determinant() < 0.0 ){
			fsvd::Mat3<T> Vt = V.transpose();
			Vt = J * Vt;
			V = Vt.transpose();
			S[2] = -S[2];
		}

	} // end signed svd