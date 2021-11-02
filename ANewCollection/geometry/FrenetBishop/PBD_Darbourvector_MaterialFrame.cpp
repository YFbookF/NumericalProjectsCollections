
//  Single rod element:
	//      3   4		//ghost points
	//		|	|
	//  --0---1---2--	// rod points

	Vector3r darboux_vector;
	Matrix3r d0, d1;

	PositionBasedElasticRods::computeMaterialFrame(p0, p1, p3, d0);
	PositionBasedElasticRods::computeMaterialFrame(p1, p2, p4, d1);

	PositionBasedElasticRods::computeDarbouxVector(d0, d1, midEdgeLength, darboux_vector);
	
// ----------------------------------------------------------------------------------------------
bool PositionBasedElasticRods::computeMaterialFrame(
	const Vector3r& p0, 
	const Vector3r& p1, 
	const Vector3r& p2, 
	Matrix3r& frame)
{
	frame.col(2) = (p1 - p0);
	frame.col(2).normalize();

	frame.col(1) = (frame.col(2).cross(p2 - p0));
	frame.col(1).normalize();

	frame.col(0) = frame.col(1).cross(frame.col(2));
	return true;
}
bool PositionBasedElasticRods::computeDarbouxVector(const Matrix3r& dA, const Matrix3r& dB, const Real mid_edge_length, Vector3r& darboux_vector)
{
	Real factor = static_cast<Real>(1.0) + dA.col(0).dot(dB.col(0)) + dA.col(1).dot(dB.col(1)) + dA.col(2).dot(dB.col(2));

	factor = static_cast<Real>(2.0) / (mid_edge_length * factor);

	for (int c = 0; c < 3; ++c)
	{
		const int i = permutation[c][0];
		const int j = permutation[c][1];
		const int k = permutation[c][2];
		darboux_vector[i] = dA.col(j).dot(dB.col(k)) - dA.col(k).dot(dB.col(j));
	}
	darboux_vector *= factor;
	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedElasticRods::computeMaterialFrameDerivative(
	const Vector3r& p0, const Vector3r& p1, const Vector3r& p2, const Matrix3r& d,
	Matrix3r& d1p0, Matrix3r& d1p1, Matrix3r& d1p2,
	Matrix3r& d2p0, Matrix3r& d2p1, Matrix3r& d2p2,
	Matrix3r& d3p0, Matrix3r& d3p1, Matrix3r& d3p2)
{
	//////////////////////////////////////////////////////////////////////////
	// d3pi
	//////////////////////////////////////////////////////////////////////////
	const Vector3r p01 = p1 - p0;
	Real length_p01 = p01.norm();

	d3p0.col(0) = d.col(2)[0] * d.col(2);
	d3p0.col(1) = d.col(2)[1] * d.col(2);
	d3p0.col(2) = d.col(2)[2] * d.col(2);

	d3p0.col(0)[0] -= 1.0;
	d3p0.col(1)[1] -= 1.0;
	d3p0.col(2)[2] -= 1.0;

	d3p0.col(0) *= (static_cast<Real>(1.0) / length_p01);
	d3p0.col(1) *= (static_cast<Real>(1.0) / length_p01);
	d3p0.col(2) *= (static_cast<Real>(1.0) / length_p01);

	d3p1.col(0) = -d3p0.col(0);
	d3p1.col(1) = -d3p0.col(1);
	d3p1.col(2) = -d3p0.col(2);

	d3p2.col(0).setZero();
	d3p2.col(1).setZero();
	d3p2.col(2).setZero();

	//////////////////////////////////////////////////////////////////////////
	// d2pi
	//////////////////////////////////////////////////////////////////////////
	const Vector3r p02 = p2 - p0;
	const Vector3r p01_cross_p02 = p01.cross(p02);

	const Real length_cross = p01_cross_p02.norm();

	Matrix3r mat;
	mat.col(0) = d.col(1)[0] * d.col(1);
	mat.col(1) = d.col(1)[1] * d.col(1);
	mat.col(2) = d.col(1)[2] * d.col(1);

	mat.col(0)[0] -= 1.0;
	mat.col(1)[1] -= 1.0;
	mat.col(2)[2] -= 1.0;

	mat.col(0) *= (-static_cast<Real>(1.0) / length_cross);
	mat.col(1) *= (-static_cast<Real>(1.0) / length_cross);
	mat.col(2) *= (-static_cast<Real>(1.0) / length_cross);

	Matrix3r product_matrix;
	MathFunctions::crossProductMatrix(p2 - p1, product_matrix);
	d2p0 = mat * product_matrix;

	MathFunctions::crossProductMatrix(p0 - p2, product_matrix);
	d2p1 = mat * product_matrix;

	MathFunctions::crossProductMatrix(p1 - p0, product_matrix);
	d2p2 = mat * product_matrix;

	//////////////////////////////////////////////////////////////////////////
	// d1pi
	//////////////////////////////////////////////////////////////////////////
	Matrix3r product_mat_d3;
	Matrix3r product_mat_d2;
	MathFunctions::crossProductMatrix(d.col(2), product_mat_d3);
	MathFunctions::crossProductMatrix(d.col(1), product_mat_d2);

	d1p0 = product_mat_d2 * d3p0 - product_mat_d3 * d2p0;
	d1p1 = product_mat_d2 * d3p1 - product_mat_d3 * d2p1;
	d1p2 = -product_mat_d3 * d2p2;
	return true;
}

// ----------------------------------------------------------------------------------------------
bool PositionBasedElasticRods::computeDarbouxGradient(
	const Vector3r& darboux_vector, const Real length,
	const Matrix3r& da, const Matrix3r& db,
	const Matrix3r dajpi[3][3], const Matrix3r dbjpi[3][3],
	//const Vector3r& bendAndTwistKs,
	Matrix3r& omega_pa, Matrix3r& omega_pb, Matrix3r& omega_pc, Matrix3r& omega_pd, Matrix3r& omega_pe
	)
{
	Real X = static_cast<Real>(1.0) + da.col(0).dot(db.col(0)) + da.col(1).dot(db.col(1)) + da.col(2).dot(db.col(2));
	X = static_cast<Real>(2.0) / (length * X);

	for (int c = 0; c < 3; ++c) 
	{
		const int i = permutation[c][0];
		const int j = permutation[c][1];
		const int k = permutation[c][2];
		// pa
		{
			Vector3r term1(0,0,0);
			Vector3r term2(0,0,0);
			Vector3r tmp(0,0,0);

			// first term
			term1 = dajpi[j][0].transpose() * db.col(k);
			tmp =   dajpi[k][0].transpose() * db.col(j);
			term1 = term1 - tmp;
			// second term
			for (int n = 0; n < 3; ++n) 
			{
				tmp = dajpi[n][0].transpose() * db.col(n);
				term2 = term2 + tmp;
			}
			omega_pa.col(i) = X * (term1-(0.5 * darboux_vector[i] * length) * term2);
			//omega_pa.col(i) *= bendAndTwistKs[i];
		}
		// pb
		{
			Vector3r term1(0, 0, 0);
			Vector3r term2(0, 0, 0);
			Vector3r tmp(0, 0, 0);
			// first term
			term1 = dajpi[j][1].transpose() * db.col(k);
			tmp =   dajpi[k][1].transpose() * db.col(j);
			term1 = term1 - tmp;
			// third term
			tmp = dbjpi[j][0].transpose() * da.col(k);
			term1 = term1 - tmp;
			
			tmp = dbjpi[k][0].transpose() * da.col(j);
			term1 = term1 + tmp;

			// second term
			for (int n = 0; n < 3; ++n) 
			{
				tmp = dajpi[n][1].transpose() * db.col(n);
				term2 = term2 + tmp;
				
				tmp = dbjpi[n][0].transpose() * da.col(n);
				term2 = term2 + tmp;
			}
			omega_pb.col(i) = X * (term1-(0.5 * darboux_vector[i] * length) * term2);
			//omega_pb.col(i) *= bendAndTwistKs[i];
		}
		// pc
		{
			Vector3r term1(0, 0, 0);
			Vector3r term2(0, 0, 0);
			Vector3r tmp(0, 0, 0);
			
			// first term
			term1 = dbjpi[j][1].transpose() * da.col(k);
			tmp =   dbjpi[k][1].transpose() * da.col(j);
			term1 = term1 - tmp;

			// second term
			for (int n = 0; n < 3; ++n) 
			{
				tmp = dbjpi[n][1].transpose() * da.col(n);
				term2 = term2 + tmp;
			}
			omega_pc.col(i) = -X*(term1+(0.5 * darboux_vector[i] * length) * term2);
			//omega_pc.col(i) *= bendAndTwistKs[i];
		}
		// pd
		{
			Vector3r term1(0, 0, 0);
			Vector3r term2(0, 0, 0);
			Vector3r tmp(0, 0, 0);
			// first term
			term1 = dajpi[j][2].transpose() * db.col(k);
			tmp =   dajpi[k][2].transpose() * db.col(j);
			term1 = term1 - tmp;
			// second term
			for (int n = 0; n < 3; ++n) {
				tmp = dajpi[n][2].transpose() * db.col(n);
				term2 = term2 + tmp;
			}
			omega_pd.col(i) = X*(term1-(0.5 * darboux_vector[i] * length) * term2);
			//omega_pd.col(i) *= bendAndTwistKs[i];
		}
		// pe
		{
			Vector3r term1(0, 0, 0);
			Vector3r term2(0, 0, 0);
			Vector3r tmp(0, 0, 0);
			// first term
			term1 = dbjpi[j][2].transpose() * da.col(k);
			tmp = dbjpi[k][2].transpose() * da.col(j);
			term1 -= tmp;
			
			// second term
			for (int n = 0; n < 3; ++n) 
			{	
				tmp = dbjpi[n][2].transpose() * da.col(n);
				term2 += tmp;
			}

			omega_pe.col(i) = -X*(term1+(0.5 * darboux_vector[i] * length) * term2);
			//omega_pe.col(i) *= bendAndTwistKs[i];
		}
	}
	return true;
}
// ----------------------------------------------------------------------------------------------