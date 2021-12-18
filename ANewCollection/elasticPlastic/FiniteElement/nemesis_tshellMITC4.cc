/*******************************************************************************
* nemesis. an experimental finite element code.                                *
* Copyright (C) 2004-2011 F.E.Karaoulanis [http://www.nemesis-project.org]     *
*                                                                              *
* This program is free software; you can redistribute it and/or modify         *
* it under the terms of the GNU General Public License version 3, as           *
* published by the Free Software Foundation.                                   *
*                                                                              *
* This program is distributed in the hope that it will be useful,              *
* but WITHOUT ANY WARRANTY; without even the implied warranty of               *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
* GNU General Public License for more details.                                 *
*                                                                              *
* You should have received a copy of the GNU General Public License            *
* along with this program.  If not, see < http://www.gnu.org/licenses/>.       *
*******************************************************************************/

// *****************************************************************************
// $LastChangedDate$
// $LastChangedRevision$
// $LastChangedBy$
// $HeadURL$
// Author(s): F.E. Karaoulanis (fkar@nemesis-project.org)
// *****************************************************************************

#include "elements/shellMITC4.h"
#include <algorithm>
#include "group/group_data.h"
#include "main/nemesis_debug.h"
#include "material/matpoint.h"
#include "material/shell_material.h"
#include "node/node.h"

double ShellMITC4::detJ_[4];
double ShellMITC4::shp_[3][4][4];

/**
 * Default constructor.
 */
ShellMITC4::ShellMITC4()
    : materials_(0) {
}

/**
 * Constructor.
 */
ShellMITC4::ShellMITC4(int id, std::vector<Node*> nodes,
                               ShellMaterial* material)
    : Element(id, nodes),
      materials_(4) {
  // Get nodal data
  myNodalIDs.resize(4);
  myNodalIDs[0] = nodes_[0]->get_id();
  myNodalIDs[1] = nodes_[1]->get_id();
  myNodalIDs[2] = nodes_[2]->get_id();
  myNodalIDs[3] = nodes_[3]->get_id();

  // Set local nodal dofs
  myLocalNodalDofs.resize(6);
  myLocalNodalDofs[0] = 0;
  myLocalNodalDofs[1] = 1;
  myLocalNodalDofs[2] = 2;
  myLocalNodalDofs[3] = 3;
  myLocalNodalDofs[4] = 4;
  myLocalNodalDofs[5] = 5;

  // Handle common info: Start -------------------------------------------------
  // Find own matrix and vector
  myMatrix = theStaticMatrices[24];
  myVector = theStaticVectors[24];
  // Get nodal coordinates
  /// @todo replace with const references
  x.Resize(4, 3);
  for (int i = 0; i < 4; i++) {
    x(i, 0) = nodes_[i]->get_x1();
    x(i, 1) = nodes_[i]->get_x2();
    x(i, 2) = nodes_[i]->get_x3();
  }
  // Inform the nodes that the corresponding Dof's must be activated
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 6; j++)
      nodes_[i]->addDofToNode(myLocalNodalDofs[j]);
  // Load vector
  P.Resize(24, 0.);
  // Self weight
  G.Resize(24, 0.);
  myMaterial = material;
  this->AssignGravityLoads();
  // Handle common info: End ---------------------------------------------------

  // Materials
  materials_[0] = material->get_clone();
  materials_[1] = material->get_clone();
  materials_[2] = material->get_clone();
  materials_[3] = material->get_clone();
  
  // Gauss points
  gps_[0][0] = -0.577350269189626;
  gps_[0][1] = -0.577350269189626;
  gps_[1][0] = +0.577350269189626;
  gps_[1][1] = -0.577350269189626;
  gps_[2][0] = +0.577350269189626;
  gps_[2][1] = +0.577350269189626;
  gps_[3][0] = -0.577350269189626;
  gps_[3][1] = +0.577350269189626;

  // Find Ktt
  Vector s(6);
  const Matrix& C = material->get_C();
  s[0] = C(0, 0);
  s[1] = C(1, 1);
  s[2] = C(2, 2);
  s[3] = C(1, 0);
  s[4] = C(2, 1);
  s[5] = C(2, 0);
  Vector eig = s.Eigenvalues();
  Ktt_ = std::min(eig[0], std::min(eig[1], eig[2]));
  // Basis
  this->FindBasis();
}

/**
 * Destructor.
 */
ShellMITC4::~ShellMITC4() {
  Containers::vector_delete(&materials_);
}

/**
 * Element stiffness matrix.
 */
const Matrix& ShellMITC4::get_K() {
  // Static variables and references
  Matrix &K=*myMatrix;
  K.Clear();
  this->FindShapeFunctions();
  
  Matrix Bmemb(3, 2);
  Matrix Bbend(3, 2);
  Matrix Bshea(2, 3);
  Vector Da(6);
  Vector Db(6);
  Matrix Ba(8, 6);
  Matrix Bb(8, 6);

  for (unsigned k = 0; k < 4; k++) {
    const Matrix& C = materials_[k]->get_C();
    for (unsigned a = 0; a < 4; a++) {
      this->FormBm(&Bmemb, a, k);
      this->FormBb(&Bbend, a, k);
      this->FormBs(&Bshea, a, k);
      Bbend *= -1;
      this->FormB(&Ba, Bmemb, Bbend, Bshea);
      this->FormBd(&Da, a, k);
      for (unsigned b = 0; b < 4; b++) {
        this->FormBm(&Bmemb, b, k);
        this->FormBb(&Bbend, b, k);
        this->FormBs(&Bshea, b, k);
        this->FormB(&Bb, Bmemb, Bbend, Bshea);
        this->FormBd(&Db, b, k);
        ///@todo Remove the following if changed in the Matrix operator=
        Matrix BB(6, 6, 0.);
        BB = ((Transpose(Ba) * C) * Bb) * detJ_[k];
        for (unsigned i = 0; i < 6; i++) {
          for (unsigned j = 0; j < 6; j++) {
            K(6 * a + i, 6 * b + j) += BB(i, j) + Ktt_ * Da[i] * Db[j] * detJ_[k];
          }
        }
      }
    }
  }

  // Multiply by facK (active/deactivated)
  double facK = groupdata_->active ? groupdata_->factor_K : 1e-7;
  K *= facK; 
  return K;
}

/**
 * Element mass matrix.
 */
const Matrix& ShellMITC4::get_M() {
  Matrix &M=*myMatrix;
  M.Clear();
  return M;
}

/**
 * Element residual vector.
 */
const Vector& ShellMITC4::get_R() {
  // Static variables and references
  static Vector sigma(6);
  static Matrix Ba(6, 3);
  Vector Du(6);
  Vector& R=*myVector;
  R.Clear();
  // Quick return if not active
  if (!(groupdata_->active)) {
    return R;
  }
  // Get factors
  double facS = groupdata_->factor_S;
  double facG = groupdata_->factor_G;
  double facP = groupdata_->factor_P;
  
  // Find shape functions for all GaussPoints (double shp[node][N, i][GPoint])
  this->FindShapeFunctions();

  // R = facS*Fint - facG*SelfWeigth - facP*ElementalLoads
  for (unsigned k = 0; k < 4; k++) {
    // drilling stress
    double eps_drill = 0.;
    for (unsigned a = 0; a < 4; a++) {
      Vector Da(6);
      this->FormBd(&Da, a, k);
      Du  = nodes_[a]->get_disp_trial();
      Du -= nodes_[a]->get_disp_convg();
      eps_drill += Da * Du;
    }
    double tau_drill = Ktt_ * eps_drill;
    
    // residual vector
    Matrix Bmemb(3, 2);
    Matrix Bbend(3, 2);
    Matrix Bshea(2, 3);
    Vector Da(6);
    Matrix Ba(8, 6);
    for (unsigned a = 0; a < 4; a++) {
      this->FormBm(&Bmemb, a, k);
      this->FormBb(&Bbend, a, k);
      this->FormBs(&Bshea, a, k);
      Bbend *= -1.0;
      this->FormB(&Ba, Bmemb, Bbend, Bshea);
      this->FormBd(&Da, a, k);
      for (unsigned i = 0; i < 6; i++) {
        Vector sigma = materials_[k]->get_stress();
        R[6 * a + i] += facS * ((Transpose(Ba) * sigma)[i] * detJ_[k] + Da[i] * tau_drill * detJ_[k]);
      }
    }
  }

  // Gravity
  Matrix T(3, 3);
  T(0, 0) = v1_[0];
  T(0, 1) = v1_[1];
  T(0, 2) = v1_[2];
  T(1, 0) = v2_[0];
  T(1, 1) = v2_[1];
  T(1, 2) = v2_[2];
  T(2, 0) = v3_[0];
  T(2, 1) = v3_[1];
  T(2, 2) = v3_[2];      
  Vector bl = T * b;
  Vector F(24, 0.);
  for (unsigned k = 0; k < 4; k++) {
    for (unsigned a = 0; a < 4; a++) {
      for (unsigned i = 0; i < 3; i++) {
        F[6 * a + i] += shp_[2][a][k] * bl[i] * detJ_[k] * materials_[k]->get_thickness();
      }
    }
  }
  for (unsigned a = 0; a < 4; a++) {
    Vector F0(3);
    for (unsigned i = 0; i < 3; i++) {
      F0[i] = F[6 * a + i];
    }
    F0 = Transpose(T) * F0;
    for (unsigned i = 0; i < 3; i++) {
      F[6 * a + i] = F0[i];
    }  
  }          
  F *= facG;
  R -= F;

  // -facP*ElementalLoads
  R-=facP*P;
  //char id_str[64];
  //snprintf(id_str, 64, "%d", id_);
  //report(R, id_str);
  return R;
}

/**
 * Element update.
 */
void ShellMITC4::Update(const double Dt) {

  // Static variables and references
  static Vector u(24);
  static Vector epsilon(8);
  static Matrix B;
  Vector Du(6);
  // Quick return if inactive
  if (!(groupdata_->active)) {
    return;
  }
  
  for (unsigned k = 0; k < 4; k++) {
    epsilon.Clear();
    for (unsigned a = 0; a < 4; a++) {
      Matrix Bmemb(3, 2);
      Matrix Bbend(3, 2);
      Matrix Bshea(2, 3);
      // Vector Da(6);
      this->FormBm(&Bmemb, a, k);
      this->FormBb(&Bbend, a, k);
      this->FormBs(&Bshea, a, k);
      Matrix Ba(8, 6);
      this->FormB(&Ba, Bmemb, Bbend, Bshea);
      Du  = nodes_[a]->get_disp_trial();
      Du -= nodes_[a]->get_disp_convg();
      epsilon += Ba * Du;
    }
    materials_[k]->set_strain(epsilon);
  }
}


/**
 * Element commit.
 */
void ShellMITC4::Commit() {
  for (unsigned int i = 0; i < materials_.size(); i++)
    materials_[i]->Commit();
}

/**
 * Element shape functions.
 */
void ShellMITC4::FindShapeFunctions() {

  double s[4] = {-0.5,  0.5, 0.5, -0.5};
  double t[4] = {-0.5, -0.5, 0.5,  0.5};

  for (unsigned k = 0; k < 4; k++) {
    for (unsigned i = 0; i < 4; i++) {
      shp_[2][i][k] = (0.5 + s[i] * gps_[k][0]) * ( 0.5 + t[i] * gps_[k][1]);
      shp_[0][i][k] = s[i] * (0.5 + t[i] * gps_[k][1]);
      shp_[1][i][k] = t[i] * (0.5 + s[i] * gps_[k][0]);
    }
    
    double J[2][2];
    for (unsigned i = 0; i < 2; i++) {
      for (unsigned j = 0; j < 2; j++) {
        J[i][j] = 0.;
        for (unsigned m = 0; m < 4; m++) {
          J[i][j] +=  xl_[i][m] * shp_[j][m][k];
        }
      }
    }

    detJ_[k] = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    double Jinv[2][2];
    Jinv[0][0] =  J[1][1] / detJ_[k];
    Jinv[0][1] = -J[0][1] / detJ_[k];
    Jinv[1][0] = -J[1][0] / detJ_[k];
    Jinv[1][1] =  J[0][0] / detJ_[k];

    for (unsigned i = 0; i < 4; i++) {
      double temp   = shp_[0][i][k] * Jinv[0][0] + shp_[1][i][k] * Jinv[1][0];
      shp_[1][i][k] = shp_[0][i][k] * Jinv[0][1] + shp_[1][i][k] * Jinv[1][1];
      shp_[0][i][k] = temp;
    }
  }
}

/**
 * Initial stresses.
 */
void ShellMITC4::AddInitialStresses(int direction,
                                 double h1, double s1,
                                 double h2, double s2, double K0) {
}

/**
 * Element stress recovery.
 */
void ShellMITC4::recoverStresses() {
}

void ShellMITC4::FindBasis() {
  v1_.Resize(3);
  v1_[0] = 0.5 * (x(2, 0) + x(1, 0) - x(3, 0) - x(0, 0));
  v1_[1] = 0.5 * (x(2, 1) + x(1, 1) - x(3, 1) - x(0, 1));
  v1_[2] = 0.5 * (x(2, 2) + x(1, 2) - x(3, 2) - x(0, 2));
  v2_.Resize(3);
  v2_[0] = 0.5 * (x(3, 0) + x(2, 0) - x(1, 0) - x(0, 0));
  v2_[1] = 0.5 * (x(3, 1) + x(2, 1) - x(1, 1) - x(0, 1));
  v2_[2] = 0.5 * (x(3, 2) + x(2, 2) - x(1, 2) - x(0, 2));
  v1_.Normalize();
  double alpha = v1_ * v2_;
  v2_ -= alpha * v1_;
  v2_.Normalize();
  ///@todo Remove Resize if is set in vector operator=.
  v3_.Resize(3);
  v3_ = Cross(v1_, v2_);
  for (unsigned i = 0; i < 4; i++) {
    xl_[0][i] = x(i, 0) * v1_[0] + x(i, 1) * v1_[1] + x(i, 2) * v1_[2];
    xl_[1][i] = x(i, 0) * v2_[0] + x(i, 1) * v2_[1] + x(i, 2) * v2_[2];
  }
}

void ShellMITC4::FormBm(Matrix* Bm, int node, int gp) {
  Bm->Clear();
  (*Bm)(0, 0) = shp_[0][node][gp];
  (*Bm)(1, 1) = shp_[1][node][gp];
  (*Bm)(2, 0) = shp_[1][node][gp];
  (*Bm)(2, 1) = shp_[0][node][gp];
}

void ShellMITC4::FormBb(Matrix* Bb, int node, int gp) {
  Bb->Clear();
  (*Bb)(0, 1) = -shp_[0][node][gp];
  (*Bb)(1, 0) =  shp_[1][node][gp];
  (*Bb)(2, 0) =  shp_[0][node][gp];
  (*Bb)(2, 1) = -shp_[1][node][gp];
}

void ShellMITC4::FormBd(Vector* Bd, int node, int gp) {
  Bd->Clear();
  double B1 = -0.5 * shp_[1][node][gp];
  double B2 = +0.5 * shp_[0][node][gp];
  double B6 =       -shp_[2][node][gp];
  (*Bd)[0] = B1 * v1_[0] + B2 * v2_[0];
  (*Bd)[1] = B1 * v1_[1] + B2 * v2_[1];
  (*Bd)[2] = B1 * v1_[2] + B2 * v2_[2];
  (*Bd)[3] = B6 * v3_[0];
  (*Bd)[4] = B6 * v3_[1];
  (*Bd)[5] = B6 * v3_[2];
}

void ShellMITC4::FormBs(Matrix* Bs, int node, int gp) {
  double dx34 = xl_[0][2] - xl_[0][3];
  double dy34 = xl_[1][2] - xl_[1][3];
  double dx21 = xl_[0][1] - xl_[0][0];
  double dy21 = xl_[1][1] - xl_[1][0];
  double dx32 = xl_[0][2] - xl_[0][1];
  double dy32 = xl_[1][2] - xl_[1][1];
  double dx41 = xl_[0][3] - xl_[0][0];
  double dy41 = xl_[1][3] - xl_[1][0];
  
  Matrix G(4, 12, 0.);
  G( 0,  0) = -0.50;
  G( 0,  1) = -0.25 * dy41;
  G( 0,  2) =  0.25 * dx41;
  G( 0,  9) =  0.50;
  G( 0, 10) = -0.25 * dy41;
  G( 0, 11) =  0.25 * dx41;
  G( 1,  0) = -0.50;
  G( 1,  1) = -0.25 * dy21;
  G( 1,  2) =  0.25 * dx21;
  G( 1,  3) =  0.50;
  G( 1,  4) = -0.25 * dy21;
  G( 1,  5) =  0.25 * dx21;
  G( 2,  3) = -0.50;
  G( 2,  4) = -0.25 * dy32;
  G( 2,  5) =  0.25 * dx32;
  G( 2,  6) =  0.50;
  G( 2,  7) = -0.25 * dy32;
  G( 2,  8) =  0.25 * dx32;
  G( 3,  6) =  0.50;
  G( 3,  7) = -0.25 * dy34;
  G( 3,  8) =  0.25 * dx34;
  G( 3,  9) = -0.50;
  G( 3, 10) = -0.25 * dy34;
  G( 3, 11) =  0.25 * dx34;

  double Ax = -xl_[0][0] + xl_[0][1] + xl_[0][2] - xl_[0][3];
  double Bx =  xl_[0][0] - xl_[0][1] + xl_[0][2] - xl_[0][3];
  double Cx = -xl_[0][0] - xl_[0][1] + xl_[0][2] + xl_[0][3];

  double Ay = -xl_[1][0] + xl_[1][1] + xl_[1][2] - xl_[1][3];
  double By =  xl_[1][0] - xl_[1][1] + xl_[1][2] - xl_[1][3];
  double Cy = -xl_[1][0] - xl_[1][1] + xl_[1][2] + xl_[1][3];

  double alph = atan(Ay / Ax);
  double beta = 0.5 * num::pi - atan(Cx / Cy);
  Matrix Rot(2, 2, 0.);
  Rot(0, 0) =  sin(beta);
  Rot(0, 1) = -sin(alph);
  Rot(1, 0) = -cos(beta);
  Rot(1, 1) =  cos(alph);
     
  Matrix Ms(2, 4, 0.);
  Ms(1, 0) = 1.0 - gps_[gp][0];
  Ms(0, 1) = 1.0 - gps_[gp][1];
  Ms(1, 2) = 1.0 + gps_[gp][0];
  Ms(0, 3) = 1.0 + gps_[gp][1];
  
  Matrix Bsv(2, 12, 0.);
  Bsv = Ms * G;

  double r1 = Cx + gps_[gp][0] * Bx;
  double r2 = Ax + gps_[gp][0] * Bx;
  double r3 = Cy + gps_[gp][1] * By;
  r1 = r1 * r1 + r3 * r3;
  r1 = sqrt(r1);
  r3 = Ay + gps_[gp][1] * By;
  r2 = r2 * r2 + r3 * r3;
  r2 = sqrt(r2);

  for (unsigned j = 0; j < 12; j++) {
    Bsv(0, j) = Bsv(0, j) * r1 / (8.0 * detJ_[gp]);
    Bsv(1, j) = Bsv(1, j) * r2 / (8.0 * detJ_[gp]);
  }
  Matrix Bss(2, 12, 0.); 
  Bss = Rot * Bsv;
  
  for (unsigned p = 0; p < 3; p++) {
    (*Bs)(0, p) = Bss(0, node * 3 + p);
    (*Bs)(1, p) = Bss(1, node * 3 + p);
  }
}


void ShellMITC4::FormB(Matrix* B, const Matrix& Bm, const Matrix& Bb,
                       const Matrix& Bs) {
  
  Matrix Gmem = Matrix(2, 3, 0.);
  Gmem(0, 0) = v1_[0];
  Gmem(0, 1) = v1_[1];
  Gmem(0, 2) = v1_[2];
  Gmem(1, 0) = v2_[0];
  Gmem(1, 1) = v2_[1];
  Gmem(1, 2) = v2_[2];
  Matrix BmembraneShell(3, 3, 0.);
  BmembraneShell = Bm * Gmem;
 
  Matrix Gbend(2, 3, 0.);
  Gbend = Gmem;
  Matrix BbendShell(3, 3, 0.);
  BbendShell = Bb * Gbend;
  
  Matrix Gshear(3, 6, 0.);
  Gshear(0, 0) = v3_[0];
  Gshear(0, 1) = v3_[1];
  Gshear(0, 2) = v3_[2];
  Gshear(1, 3) = v1_[0];
  Gshear(1, 4) = v1_[1];
  Gshear(1, 5) = v1_[2];
  Gshear(2, 3) = v2_[0];
  Gshear(2, 4) = v2_[1];
  Gshear(2, 5) = v2_[2];

  Matrix BshearShell(2, 6, 0.);
  BshearShell = Bs * Gshear;

  B->Clear();

  // membrane terms 
  for (unsigned p = 0; p < 3; p++) {
    for (unsigned q = 0; q < 3; q++) {
      (*B)(p, q) = BmembraneShell(p, q);
    }
  }
  
  // bending terms
  for (unsigned p = 3; p < 6; p++) {
    for (unsigned q = 3; q < 6; q++) {
      (*B)(p, q) = BbendShell(p - 3, q - 3);
    }
  }

  // shear terms 
  for (unsigned p = 0; p < 2; p++) {
    for (unsigned q = 0; q < 6; q++) {
      (*B)(p + 6, q) = BshearShell(p, q);
    }
  }
}
