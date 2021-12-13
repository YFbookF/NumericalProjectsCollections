https://github.com/Azmisov/snow
void Particle::applyPlasticity(){
	Matrix2f f_all = def_elastic * def_plastic;
	//We compute the SVD decomposition
	//The singular values (basically a scale transform) tell us if 
	//the particle has exceeded critical stretch/compression
	def_elastic.svd(&svd_w, &svd_e, &svd_v);
	Matrix2f svd_v_trans = svd_v.transpose();
	//Clamp singular values to within elastic region
	for (int i=0; i<2; i++){
		if (svd_e[i] < CRIT_COMPRESS)
			svd_e[i] = CRIT_COMPRESS;
		else if (svd_e[i] > CRIT_STRETCH)
			svd_e[i] = CRIT_STRETCH;
	}
#if ENABLE_IMPLICIT
	//Compute polar decomposition, from clamped SVD
	polar_r.setData(svd_w*svd_v_trans);
	polar_s.setData(svd_v);
	polar_s.diag_product(svd_e);
	polar_s.setData(polar_s*svd_v_trans);
#endif
	
	//Recompute elastic and plastic gradient
	//We're basically just putting the SVD back together again
	Matrix2f v_cpy(svd_v), w_cpy(svd_w);
	v_cpy.diag_product_inv(svd_e);
	w_cpy.diag_product(svd_e);
	def_plastic = v_cpy*svd_w.transpose()*f_all;
	def_elastic = w_cpy*svd_v.transpose();
}
const Matrix2f Particle::energyDerivative(){
	//Adjust lame parameters to account for hardening
	float harden = exp(HARDENING*(1-def_plastic.determinant())),
		Je = svd_e.product();
	//This is the co-rotational term
	Matrix2f temp = 2*mu*(def_elastic - svd_w*svd_v.transpose())*def_elastic.transpose();
	//Add in the primary contour term
	temp.diag_sum(lambda*Je*(Je-1));
	//Add hardening and volume
	return volume * harden * temp;
}