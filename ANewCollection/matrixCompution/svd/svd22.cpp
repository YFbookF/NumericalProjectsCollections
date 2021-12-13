https://github.com/Azmisov/snow
void Matrix2f::svd(Matrix2f* w, Vector2f* e, Matrix2f* v) const{
	/* Probably not the fastest, but I can't find any simple algorithms
		Got most of the derivation from:
			http://www.ualberta.ca/~mlipsett/ENGM541/Readings/svd_ellis.pdf
			www.imm.dtu.dk/pubdb/views/edoc_download.php/3274/pdf/imm3274.pdf
			https://github.com/victorliu/Cgeom/blob/master/geom_la.c (geom_matsvd2d method)
	*/
	//If it is diagonal, SVD is trivial
	if (fabs(data[0][1] - data[1][0]) < MATRIX_EPSILON && fabs(data[0][1]) < MATRIX_EPSILON){
		w->setData(data[0][0] < 0 ? -1 : 1, 0, 0, data[1][1] < 0 ? -1 : 1);
		e->setData(fabs(data[0][0]), fabs(data[1][1]));
		v->loadIdentity();
	}
	//Otherwise, we need to compute A^T*A
	else{
		float j = data[0][0]*data[0][0] + data[0][1]*data[0][1],
			k = data[1][0]*data[1][0] + data[1][1]*data[1][1],
			v_c = data[0][0]*data[1][0] + data[0][1]*data[1][1];
		//Check to see if A^T*A is diagonal
		if (fabs(v_c) < MATRIX_EPSILON){
			float s1 = sqrt(j),
				s2 = fabs(j-k) < MATRIX_EPSILON ? s1 : sqrt(k);
			e->setData(s1, s2);
			v->loadIdentity();
			w->setData(
				data[0][0]/s1, data[1][0]/s2,
				data[0][1]/s1, data[1][1]/s2
			);
		}
		//Otherwise, solve quadratic for eigenvalues
		else{
			float jmk = j-k,
				jpk = j+k,
				root = sqrt(jmk*jmk + 4*v_c*v_c),
				eig = (jpk+root)/2,
				s1 = sqrt(eig),
				s2 = fabs(root) < MATRIX_EPSILON ? s1 : sqrt((jpk-root)/2);
			e->setData(s1, s2);
			//Use eigenvectors of A^T*A as V
			float v_s = eig-j,
				len = sqrt(v_s*v_s + v_c*v_c);
			v_c /= len;
			v_s /= len;
			v->setData(v_c, -v_s, v_s, v_c);
			//Compute w matrix as Av/s
			w->setData(
				(data[0][0]*v_c + data[1][0]*v_s)/s1,
				(data[1][0]*v_c - data[0][0]*v_s)/s2,
				(data[0][1]*v_c + data[1][1]*v_s)/s1,
				(data[1][1]*v_c - data[0][1]*v_s)/s2
			);
		}
	}
}