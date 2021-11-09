void delfem2::makeLinearSystem_PotentialFlow_Order1st(
    std::vector<double>& A,
    std::vector<double>& f,
    //
    const CVec3d& velo,
    int ngauss,
    const std::vector<double>& aXYZ,
    const std::vector<int>& aTri,
    const std::vector<double>& aSolidAngle)
{
  const int np = (int)aXYZ.size()/3;
  A.assign(np*np, 0.0);
  f.assign(np, 0.0);
  for (int ip = 0; ip<np; ip++){
    CVec3d p(aXYZ[ip*3+0], aXYZ[ip*3+1], aXYZ[ip*3+2]);
    for (int jtri = 0; jtri<(int)aTri.size()/3; ++jtri){
      const int jq0 = aTri[jtri*3+0];
      const int jq1 = aTri[jtri*3+1];
      const int jq2 = aTri[jtri*3+2];
      const CVec3d q0(aXYZ[jq0*3+0], aXYZ[jq0*3+1], aXYZ[jq0*3+2]);
      const CVec3d q1(aXYZ[jq1*3+0], aXYZ[jq1*3+1], aXYZ[jq1*3+2]);
      const CVec3d q2(aXYZ[jq2*3+0], aXYZ[jq2*3+1], aXYZ[jq2*3+2]);
      CVec3d n = Normal(q0, q1, q2);
      const double area = n.norm()*0.5; // area
      n.normalize(); // unit normal
      n *= -1; // outward pointing vector
      double dC[3] = { 0, 0, 0 };
      double df = 0;
      const int nint = NIntTriGauss[ngauss]; // number of integral points
      for (int iint = 0; iint<nint; iint++){
        double r0 = TriGauss[ngauss][iint][0];
        double r1 = TriGauss[ngauss][iint][1];
        double r2 = 1.0-r0-r1;
        double wb = TriGauss[ngauss][iint][2];
        CVec3d yb = r0*q0+r1*q1+r2*q2;
        CVec3d v = (p-yb);
        double len = v.norm();
        double G = 1.0/(4*M_PI*len);
        double dGdn = (v.dot(n))/(4*M_PI*len*len*len);
        {
          double wav = wb*area*dGdn;
          dC[0] += r0*wav; 
          dC[1] += r1*wav; 
          dC[2] += r2*wav; 
        }
        {
          double vnyb = -n.dot(velo);
          double val = vnyb*G;
          df += wb*area*val; 
        }
      }
      A[ip*np+jq0] += dC[0];
      A[ip*np+jq1] += dC[1];
      A[ip*np+jq2] += dC[2];
      f[ip] += df;
    }
    A[ip*np+ip] += aSolidAngle[ip]/(4*M_PI);
    //A[ip*np+ip] += aSolidAngle[ip];
  }
}

delfem2::CVec3d delfem2::evaluateField_PotentialFlow_Order1st(
    double& phi_pos,
    const CVec3d& pos,
    const CVec3d& velo_inf,
    int ngauss,
    const std::vector<double>& aValSrf,
    const std::vector<double>& aXYZ,
    const std::vector<int>& aTri)
{
  const unsigned int np = static_cast<unsigned int>(aXYZ.size()/3);
  if (aValSrf.size()!=np){ return CVec3d(0, 0, 0); }
  CVec3d gradphi_pos = CVec3d(0, 0, 0);
  phi_pos = 0;
  for (unsigned int jtri = 0; jtri<aTri.size()/3; ++jtri){
    const int jq0 = aTri[jtri*3+0];
    const int jq1 = aTri[jtri*3+1];
    const int jq2 = aTri[jtri*3+2];
    const CVec3d q0(aXYZ[jq0*3+0], aXYZ[jq0*3+1], aXYZ[jq0*3+2]);
    const CVec3d q1(aXYZ[jq1*3+0], aXYZ[jq1*3+1], aXYZ[jq1*3+2]);
    const CVec3d q2(aXYZ[jq2*3+0], aXYZ[jq2*3+1], aXYZ[jq2*3+2]);
    assert(ngauss>=0&&ngauss<6);
    CVec3d n = Normal(q0, q1, q2);
    const double area = n.norm()*0.5; // area
    n.normalize(); // unit normal
    n *= -1; // outward normal 
    const int nint = NIntTriGauss[ngauss]; // number of integral points
    for (int iint = 0; iint<nint; iint++){
      double r0 = TriGauss[ngauss][iint][0];
      double r1 = TriGauss[ngauss][iint][1];
      double r2 = 1.0-r0-r1;
      double wb = TriGauss[ngauss][iint][2];
      CVec3d yb = r0*q0+r1*q1+r2*q2;
      double phiyb = r0*aValSrf[jq0]+r1*aValSrf[jq1]+r2*aValSrf[jq2];
      CVec3d v = (pos-yb);
      double len = v.norm();
      double G = 1.0/(4*M_PI*len);
      double dGdn = (v.dot(n))/(4*M_PI*len*len*len);
      CVec3d dGdx = -v/(4*M_PI*len*len*len);
      CVec3d dGdndx = (1/(4*M_PI*len*len*len))*n-(3*(v.dot(n))/(4*M_PI*len*len*len*len*len))*v;
      double vnyb = -n.dot(velo_inf);
      {
        double phyx = dGdn*phiyb-G*vnyb;
        phi_pos -= wb*area*phyx;
      }
      {
        CVec3d gradphyx = dGdndx*phiyb-dGdx*vnyb;
        gradphi_pos -= wb*area*gradphyx;
      }
    }
  }
  gradphi_pos += velo_inf;
  return gradphi_pos;
}


void delfem2::makeLinearSystem_PotentialFlow_Order0th(
    std::vector<double>& A,
    std::vector<double>& f,
    //
    const CVec3d& velo_inf,
    int ngauss,
    const std::vector<double>& aXYZ,
    const std::vector<unsigned int> &aTri)
{
  const size_t nt = aTri.size()/3;
  A.assign(nt*nt, 0.0);
  f.assign(nt, 0.0);
  for (unsigned int it = 0; it<nt; it++){
    const CVec3d pm = MidPoint(it, aTri, aXYZ);
    for (unsigned int jt = 0; jt<nt; ++jt){
      if (it==jt) continue;
      const unsigned int jq0 = aTri[jt*3+0];
      const unsigned int jq1 = aTri[jt*3+1];
      const unsigned int jq2 = aTri[jt*3+2];
      const CVec3d q0(aXYZ[jq0*3+0], aXYZ[jq0*3+1], aXYZ[jq0*3+2]);
      const CVec3d q1(aXYZ[jq1*3+0], aXYZ[jq1*3+1], aXYZ[jq1*3+2]);
      const CVec3d q2(aXYZ[jq2*3+0], aXYZ[jq2*3+1], aXYZ[jq2*3+2]);
      CVec3d ny = Normal(q0, q1, q2);
      const double area = ny.norm()*0.5; // area
      ny.normalize(); // unit normal
      ny *= -1; // it is pointing outward to the domain
      double aC = 0;
      double df = 0;
      const int nint = NIntTriGauss[ngauss]; // number of integral points
      for (int iint = 0; iint<nint; iint++){
        double r0 = TriGauss[ngauss][iint][0];
        double r1 = TriGauss[ngauss][iint][1];
        double r2 = 1.0-r0-r1;
        double wb = TriGauss[ngauss][iint][2];
        CVec3d yb = r0*q0+r1*q1+r2*q2;
        CVec3d r = (pm-yb);
        double len = r.norm();
        double G = 1.0/(4*M_PI*len);
        double dGdn = (r.dot(ny))/(4*M_PI*len*len*len);
        {
          double wav = wb*area*dGdn;
          aC += wav;  // should be plus
        }
        {
          double vnyb = -ny.dot(velo_inf);
          double val = vnyb*G;
          df += wb*area*val;  // should be plus
        }
      }
      A[it*nt+jt] = aC;
      f[it] += df;
    }
    A[it*nt+it] += 0.5; 
  }
  /*
  {
    double sum = 0;
    for (int jt = 0; jt<nt; ++jt){
      double row = 0;
      for (int it = 0; it<nt; ++it){
        row += A[it*nt+jt];
      }
      sum += row;
      std::cout<<"hoge"<<jt<<" "<<row<<std::endl;
    }
    std::cout<<"sum: "<<sum<<std::endl;
  }
  */
}


void delfem2::evaluateField_PotentialFlow_Order0th(
    double& phi_pos,
    CVec3d& gradphi_pos,
    //
    const CVec3d& pos,
    const CVec3d& velo_inf,
    int ngauss,
    const std::vector<double>& aValTri,
    const std::vector<double>& aXYZ,
    const std::vector<unsigned int> &aTri)
{
  const unsigned int nt = static_cast<unsigned int>(aTri.size()/3);
  if (aValTri.size()!=nt){
    gradphi_pos = CVec3d(0,0,0);
    return;
  }
  gradphi_pos = CVec3d(0, 0, 0);
  phi_pos = 0;
  for (unsigned int jtri = 0; jtri<nt; ++jtri){
    const int jq0 = aTri[jtri*3+0];
    const int jq1 = aTri[jtri*3+1];
    const int jq2 = aTri[jtri*3+2];
    const CVec3d q0(aXYZ[jq0*3+0], aXYZ[jq0*3+1], aXYZ[jq0*3+2]);
    const CVec3d q1(aXYZ[jq1*3+0], aXYZ[jq1*3+1], aXYZ[jq1*3+2]);
    const CVec3d q2(aXYZ[jq2*3+0], aXYZ[jq2*3+1], aXYZ[jq2*3+2]);
    CVec3d ny = Normal(q0, q1, q2);
    const double area = ny.norm()*0.5; // area
//    const double elen = sqrt(area*2)*0.5;
    ny.normalize(); // unit normal
    ny *= -1; // normal pointing outward
    const double phiy = aValTri[jtri];
    const int nint = NIntTriGauss[ngauss]; // number of integral points
    for (int iint = 0; iint<nint; iint++){
      const double r0 = TriGauss[ngauss][iint][0];
      const double r1 = TriGauss[ngauss][iint][1];
      const double r2 = 1.0-r0-r1;
      const double wb = TriGauss[ngauss][iint][2];
      const CVec3d yb = r0*q0+r1*q1+r2*q2;
      const CVec3d r = (pos-yb);
      const double len = r.norm();
      double G = 1.0/(4*M_PI*len);
      double dGdny = (r.dot(ny))/(4*M_PI*len*len*len);
      CVec3d dGdx = -r/(4*M_PI*len*len*len);
      CVec3d dGdnydx = (1/(4*M_PI*len*len*len))*ny-(3*(r.dot(ny))/(4*M_PI*len*len*len*len*len))*r;
      ///
//      const double reg = 1.0-exp(-(len*len*len)/(elen*elen*elen));
//      G *= reg;
//      dGdny *= reg;
//      dGdx *= reg;
//      dGdnydx *= reg;
      ////
      const double vnyb = -ny.dot(velo_inf);
      {
        double phyx = -dGdny*phiy+G*vnyb;
        phi_pos += wb*area*phyx; // should be plus
      }
      {
        CVec3d gradphyx = -dGdnydx*phiy+dGdx*vnyb;
        gradphi_pos += wb*area*gradphyx; // should be plus
      }
    }
  }
  gradphi_pos += velo_inf;
}
