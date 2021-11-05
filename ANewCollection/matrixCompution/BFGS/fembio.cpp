//https://github.com/kbronik2017/FEM_fracture_contact_mechanics/blob/master/FECore/BFGSSolver.cpp
bool BFGSSolver::Update(double s, vector<double>& ui, vector<double>& R0, vector<double>& R1)
{
	int i;
	double dg, dh,dgi, c, r;
	double *vn, *wn;

	int neq = ui.size();

	// calculate the BFGS update vectors
	for (i=0; i<neq; ++i)	
	{
		m_D[i] = s*ui[i];
		m_G[i] = R0[i] - R1[i];
		m_H[i] = R0[i]*s;
	}

	dg = m_D*m_G;
	dh = m_D*m_H;
	dgi = 1.0 / dg;
	r = dg/dh;

	// check to see if this is still a pos definite update
//	if (r <= 0) 
//	{
//		return false;
//	}

	// calculate the condition number
//	c = sqrt(r);
	c = sqrt(fabs(r));

	// make sure c is less than the the maximum.
	if (c > m_cmax) return false;

	vn = m_V[m_nups];
	wn = m_W[m_nups];

	// TODO: There might be a bug here. Check signs!
	for (i=0; i<neq; ++i)	
	{
		vn[i] = -m_H[i]*c - m_G[i];
		wn[i] = m_D[i]*dgi;
	}

	// increment update counter
	++m_nups;

	return true;
}