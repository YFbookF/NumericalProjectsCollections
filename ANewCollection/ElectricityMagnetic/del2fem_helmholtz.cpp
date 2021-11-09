// -----------------------------------

std::complex<double> delfem2::evaluateField_Helmholtz_Order0th(
    const std::vector<std::complex<double>>& aSol,
    const CVec3d& p,
    const CVec3d& pos_source,
    double k, // wave number
    //
    double Adm, // admittance
    const std::vector<unsigned int> &aTri,
    const std::vector<double>& aXYZ,
    bool is_inverted_norm)
{
  using COMPLEX = std::complex<double>;
  const COMPLEX IMG(0.0, 1.0);
  std::complex<double> c1;
  {
    double rs = (p-pos_source).norm();
    c1 = exp(rs*k*IMG)/(4*M_PI*rs);
  }
  int ntri = (int)aTri.size()/3;
  for (int jtri = 0; jtri<ntri; jtri++){
    CVec3d pmj = MidPoint(jtri, aTri, aXYZ);
    double rm = (p-pmj).norm();
    CVec3d n = bem::NormalTri(jtri, aTri, aXYZ);
    if (is_inverted_norm){ n *= -1; }
    double area = n.norm()*0.5;
    n.normalize();
    std::complex<double> G = exp(rm*k*IMG)/(4*M_PI*rm);
    std::complex<double> dGdr = G*(IMG*k-1.0/rm);
    double drdn = (1.0/rm)*((p-pmj).dot(n));
    c1 -= area*aSol[jtri]*(dGdr*drdn-IMG*k*Adm*G);
  }
  return c1;
}

//////////////////////////////////////////////////////////////////////////////////////

void delfem2::Helmholtz_TransferOrder1st_PntTri
(std::complex<double> aC[3],
const CVec3d& p0,
const CVec3d& q0, const CVec3d& q1, const CVec3d& q2,
double k, double beta,
int ngauss)
{
  using COMPLEX = std::complex<double>;
  const COMPLEX IMG(0.0, 1.0);
  assert(ngauss>=0&&ngauss<3);
  const int nint = NIntTriGauss[ngauss]; // number of integral points
  CVec3d n = Normal(q0, q1, q2);
  const double a = n.norm()*0.5; // area
  n.normalize(); // unit normal
  aC[0] = aC[1] = aC[2] = COMPLEX(0, 0);
  for (int iint = 0; iint<nint; iint++){
    double r0 = TriGauss[ngauss][iint][0];
    double r1 = TriGauss[ngauss][iint][1];
    double r2 = 1.0-r0-r1;
    double w = TriGauss[ngauss][iint][2];
    CVec3d v = p0-(r0*q0+r1*q1+r2*q2);
    double d = v.norm();  
    COMPLEX G = exp(COMPLEX(0, k*d))/(4.0*M_PI*d);
    COMPLEX val = G*(-IMG*k*beta+v.dot(n)/(d*d)*COMPLEX(1.0, -k*d));
    const COMPLEX wav = w*a*val;
    aC[0] += r0*wav;
    aC[1] += r1*wav;
    aC[2] += r2*wav;
  }
}

std::complex<double> delfem2::evaluateField_Helmholtz_Order1st(
    const std::vector<std::complex<double>>& aSol,
    const CVec3d& p,
    const CVec3d& pos_source,
    double k, // wave number
    double beta, // admittance
    const std::vector<int>& aTri,
    const std::vector<double>& aXYZ,
    [[maybe_unused]] bool is_inverted_norm,
    int ngauss)
{
  using COMPLEX = std::complex<double>;
  const COMPLEX IMG(0.0, 1.0);
  COMPLEX c1;
  {
    double rs = (p-pos_source).norm();
    c1 = exp(rs*k*IMG)/(4*M_PI*rs);
  }
  int ntri = (int)aTri.size()/3;
  for (int jtri = 0; jtri<ntri; jtri++){
    const int jn0 = aTri[jtri*3+0];
    const int jn1 = aTri[jtri*3+1];
    const int jn2 = aTri[jtri*3+2];
    CVec3d q0(aXYZ[jn0*3+0], aXYZ[jn0*3+1], aXYZ[jn0*3+2]);
    CVec3d q1(aXYZ[jn1*3+0], aXYZ[jn1*3+1], aXYZ[jn1*3+2]);
    CVec3d q2(aXYZ[jn2*3+0], aXYZ[jn2*3+1], aXYZ[jn2*3+2]);
    COMPLEX aC[3];  Helmholtz_TransferOrder1st_PntTri(aC, p, q0, q1, q2, k, beta, ngauss);
    c1 -= aC[0]*aSol[jn0]+aC[1]*aSol[jn1]+aC[2]*aSol[jn2];
  }
  return c1;
}
