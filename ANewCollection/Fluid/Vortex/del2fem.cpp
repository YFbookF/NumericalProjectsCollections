// ---------------------------------------------

void delfem2::BEM_VortexSheet_Coeff_0th(
	double aC[4], 
	const CVec3d& x0,
	const CVec3d& x1,
	const CVec3d& x2,
	const CVec3d& y0,
	const CVec3d& y1,
	const CVec3d& y2,
	[[maybe_unused]] const CVec3d& velo,
	int ngauss)
{
  assert(ngauss>=0&&ngauss<6);
  const int nint = NIntTriGauss[ngauss]; // number of integral points
  //
  CVec3d xm = (x0+x1+x2)*0.3333333333333333333;
  CVec3d nx = Normal(x0, x1, x2);
//  const double areax = nx.Length()*0.5; // area
  nx.normalize(); // unit normal
  const CVec3d ux = (x1-x0).normalized();
  const CVec3d vx = nx^ux;
  //
  CVec3d ny = Normal(y0, y1, y2);
  const double areay = ny.norm()*0.5; // area
  ny.normalize(); // unit normal
  const CVec3d uy = (y1-y0).normalized();
  const CVec3d vy = ny^uy;
  //
  aC[0] = aC[1] = aC[2] = aC[3] = 0.0;
  for (int iint = 0; iint<nint; iint++){
    double r0 = TriGauss[ngauss][iint][0];
    double r1 = TriGauss[ngauss][iint][1];
    double r2 = 1.0-r0-r1;
    double wb = TriGauss[ngauss][iint][2];
    CVec3d yb = r0*y0+r1*y1+r2*y2;
    CVec3d r = (xm-yb);
    double len = r.norm();
//    double G = -1.0/(4*M_PI*len);
//    double dGdn = -(r*ny)/(4*M_PI*len*len*len);
    CVec3d dGdy = -r/(4*M_PI*len*len*len);
    CVec3d pvycdGdy = -(vy^r)/(4*M_PI*len*len*len); // +vy ^ dGdy = (ny^uy)^dGdy
    CVec3d muycdGdy = +(uy^r)/(4*M_PI*len*len*len); // -uy ^ dGdy = (ny^vy)^dGdy
    {
      aC[0] += wb*areay*(pvycdGdy.dot(ux));
      aC[1] += wb*areay*(muycdGdy.dot(ux));
      aC[2] += wb*areay*(pvycdGdy.dot(vx));
      aC[3] += wb*areay*(muycdGdy.dot(vx));
    } 
  }
}

void delfem2::makeLinearSystem_VortexSheet_Order0th(
    std::vector<double>& A,
    std::vector<double>& f,
    // --
    const CVec3d& velo,
    int ngauss,
    const std::vector<double>& aXYZ,
    const std::vector<int>& aTri)
{
  const int nt = (int)aTri.size()/3;
  A.assign(4*nt*nt, 0.0);
  f.assign(2*nt, 0.0);
  for (int it = 0; it<nt; ++it){
    const int ip0 = aTri[it*3+0];
    const int ip1 = aTri[it*3+1];
    const int ip2 = aTri[it*3+2];
    const CVec3d p0(aXYZ[ip0*3+0], aXYZ[ip0*3+1], aXYZ[ip0*3+2]);
    const CVec3d p1(aXYZ[ip1*3+0], aXYZ[ip1*3+1], aXYZ[ip1*3+2]);
    const CVec3d p2(aXYZ[ip2*3+0], aXYZ[ip2*3+1], aXYZ[ip2*3+2]);
    {
      const CVec3d nx = Normal(p0, p1, p2).normalized();
      const CVec3d ux = (p1-p0).normalized();
      const CVec3d vx = (nx^ux);
      f[it*2+0] = ux.dot(velo);
      f[it*2+1] = vx.dot(velo);
    }
    for (int jt = 0; jt<nt; ++jt){
      if (it==jt) continue;
      const int jq0 = aTri[jt*3+0];
      const int jq1 = aTri[jt*3+1];
      const int jq2 = aTri[jt*3+2];
      const CVec3d q0(aXYZ[jq0*3+0], aXYZ[jq0*3+1], aXYZ[jq0*3+2]);
      const CVec3d q1(aXYZ[jq1*3+0], aXYZ[jq1*3+1], aXYZ[jq1*3+2]);
      const CVec3d q2(aXYZ[jq2*3+0], aXYZ[jq2*3+1], aXYZ[jq2*3+2]);
      double aC[4];
      BEM_VortexSheet_Coeff_0th(aC, 
        p0, p1, p2,
        q0, q1, q2, 
        velo, ngauss);
      A[(2*it+0)*(2*nt)+(2*jt+0)] = aC[0];
      A[(2*it+0)*(2*nt)+(2*jt+1)] = aC[1];
      A[(2*it+1)*(2*nt)+(2*jt+0)] = aC[2];
      A[(2*it+1)*(2*nt)+(2*jt+1)] = aC[3];
    }
    A[(2*it+0)*(2*nt)+(2*it+0)] = 0.5;
    A[(2*it+0)*(2*nt)+(2*it+1)] = 0.0;
    A[(2*it+1)*(2*nt)+(2*it+0)] = 0.0;
    A[(2*it+1)*(2*nt)+(2*it+1)] = 0.5;
  }
}

delfem2::CVec3d delfem2::evaluateField_VortexSheet_Order0th
(const CVec3d& pos,
 const std::vector<double>& aValSrf,
 //
 int ngauss,
 const std::vector<double>& aXYZ,
 const std::vector<int>& aTri, 
 int jtri_exclude)
{
  assert(ngauss>=0&&ngauss<6);
  const size_t nt = aTri.size()/3;
  CVec3d velo_res(0,0,0);
  for (unsigned int jt = 0; jt<nt; ++jt){
    if ((int)jt==jtri_exclude){ continue; }
    const int jq0 = aTri[jt*3+0];
    const int jq1 = aTri[jt*3+1];
    const int jq2 = aTri[jt*3+2];
    const CVec3d q0(aXYZ[jq0*3+0], aXYZ[jq0*3+1], aXYZ[jq0*3+2]);
    const CVec3d q1(aXYZ[jq1*3+0], aXYZ[jq1*3+1], aXYZ[jq1*3+2]);
    const CVec3d q2(aXYZ[jq2*3+0], aXYZ[jq2*3+1], aXYZ[jq2*3+2]);
    CVec3d ny = Normal(q0, q1, q2);
    const double areay = ny.norm()*0.5; // area
    ny.normalize(); // unit normal
    const CVec3d uy = (q1-q0).normalized();
    const CVec3d vy = ny^uy;
    const int nint = NIntTriGauss[ngauss]; // number of integral points
    for (int iint = 0; iint<nint; iint++){
      const double r0 = TriGauss[ngauss][iint][0];
      const double r1 = TriGauss[ngauss][iint][1];
      const double r2 = 1.0-r0-r1;
      const double wb = TriGauss[ngauss][iint][2];
      CVec3d yb = r0*q0+r1*q1+r2*q2;
      CVec3d r = (pos-yb);
      const double len = r.norm();
      //    double G = -1.0/(4*M_PI*len);
      //    double dGdn = -(r*ny)/(4*M_PI*len*len*len);
      //    CVector3 dGdy = -r/(4*M_PI*len*len*len);
      CVec3d pvycdGdy = -(vy^r)/(4*M_PI*len*len*len); // +vy ^ dGdy = (ny^uy)^dGdy
      CVec3d muycdGdy = +(uy^r)/(4*M_PI*len*len*len); // -uy ^ dGdy = (ny^vy)^dGdy
      velo_res -= wb*areay*(pvycdGdy*aValSrf[jt*2+0]+muycdGdy*aValSrf[jt*2+1]);
    }
  }
  return velo_res;
}
