/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*  This file is part of the library KASKADE 7                               */
/*    see http://www.zib.de/projects/kaskade7-finite-element-toolbox         */
/*                                                                           */
/*  Copyright (C) 2013-2013 Zuse Institute Berlin                            */
/*  Copyright (C) 2013-2013 U Milan                                          */
/*                                                                           */
/*  KASKADE 7 is distributed under the terms of the ZIB Academic License.    */
/*    see $KASKADE/academic.txt                                              */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <iostream>

#include "dune/grid/config.h"

#include "fem/fixdune.hh"
#include "fem/diffops/membraneModels.hh"

#include "utilities/detailed_exception.hh"

namespace Kaskade 
{
    AlievPanfilov::AlievPanfilov(): MembraneModelBase<AlievPanfilov,1>("Aliev-Panfilov",0,0),
      a(0.1), eps1(0.01), ga(8.0), gs(8.0) , mu1(0.07), mu2(0.3)
    {
    }
 
    double AlievPanfilov::current(double u, Gating const& v_) const
    {
      double v = v_[0];
      return -(ga*u*(u-a)*(u-1.0) + u*v);
    }
    
    double AlievPanfilov::current_du(double u, Gating const& v_) const
    {
      double v = v_[0];
      return std::min(-ga*( (2*u-a)*(u-1.0) + u*(u-a) ) - v, 0.0); // why the min here?
    }
    
    AlievPanfilov::Gating AlievPanfilov::current_dv(double u, Gating const& ) const
    {
      return -u;
    }
    
    AlievPanfilov::Gating AlievPanfilov::gatingRhs(double u, Gating const& v_) const
    {
      double v = v_[0];
      return 0.25*(eps1+mu1*v/(u+mu2))*(-v-gs*u*(u-a-1.0));
    }
    
    AlievPanfilov::Gating AlievPanfilov::gatingRhs_du(double u, Gating const& v_) const
    {
      double v = v_[0];
      return 0.25* ( - mu1*v/((u+mu2)*(u+mu2))*(-v-gs*u*(u-a-1)) - ( eps1+mu1*v/(u+mu2))*(gs*(2*u-a-1)) );       
    }
    
    AlievPanfilov::GatingJacobian AlievPanfilov::gatingRhs_dv(double u, Gating const& v_) const
    {
      double v = v_[0];
      return std::min(0.25*( mu1/(u+mu2)*(-v-gs*u*(u-a-1)) - (eps1+mu1*v/(u+mu2)) ), 0.0);  // why the min here?
    }

  //------------------------------------------------------------------------------------------  
  
  namespace {
    
    TenTusscher::Gating vRest()
    {
      TenTusscher::Gating v;
      v[0] = 3.373e-5;
      v[1] = 9.755e-1;
      v[2] = 9.953e-1;
      v[3] = 7.888e-1;
      v[4] = 3.640e+0;
      v[5] = 1.260e-4;
      v[6] = 3.600e-4;
      v[7] = 9.073e-1;
      v[8] = 7.444e-1;
      v[9] = 7.045e-1;
      v[10] = 1.720e-3;
      v[11] = 1.3689e+2;
      v[12] = 6.21e-3;
      v[13] = 4.712e-1;
      v[14] = 9.500e-3;
      v[15] = 8.604e+0;
      v[16] = 2.4200e-8;
      v[17] = 9.99998e-1;
      return v;
    }
  }
  
  TenTusscher::TenTusscher(): MembraneModelBase<TenTusscher,18>("ten Tusscher",-8.523e+1,vRest())
  {}
    
  double TenTusscher::current(double u, Gating const& v) const
  {

    // State variables
    double V=u;
    double d=v[0];
    double f2=v[1];
    double fCass=v[2];
    double pf=v[3];
    // double Ca_sr=v[4]; unused...
    double Ca_i=v[5];
    double Ca_ss=v[6];
    // double Rel=v[7]; unused...
    double h=v[8];
    double j=v[9];
    double m=v[10];
    double K_i=v[11];
    double xr1=v[12];
    double xr2=v[13];
    double xs=v[14];
    double Na_i=v[15];
    double pr=v[16];
    double s=v[17];

    // Constants
    double g_K1 = 5.405;
    double F = 96485.3415;
    double R = 8314.472;
    double T = 310.0;
    double K_o = 5.4;
    double g_to = 0.294;
    double g_Kr = 0.153;
    double P_kna = 0.03;
    double Na_o = 140.0; 
    double Ca_o = 2.0;
    double g_CaL = 0.0000398;
    double P_NaK = 2.724;
    double K_mk = 1.0;
    double K_mNa = 40.0;
    double g_Na = 14.838;
    double g_bna = 0.00029;
    double K_NaCa = 1000.0;
    double gamma = 0.35;
    double Km_Nai = 87.5;
    double Km_Ca = 1.38;
    double K_sat = 0.1;
    double g_bca = 0.000592;
    double g_pK = 0.0146;
    double K_pCa = 0.0005;
    double g_pCa = 0.1238;
    double g_Ks = 0.392;
    double alpha = 2.5;
 
    // Compute
    double E_K = R*T/F*log(K_o/K_i);
    double alpha_K1 = 0.1/(1.0+exp(0.06*(V-E_K-200.0)));
    double beta_K1 = (3.0*exp(0.0002*(V-E_K+100.0))+exp(0.1*(V-E_K-10.0)))/(1.0+exp(-0.5*(V-E_K)));
    double xK1_inf = alpha_K1/(alpha_K1+beta_K1);
    double i_K1 = g_K1*xK1_inf*sqrt(K_o/5.4)*(V-E_K);
    double i_to = g_to*pr*s*(V-E_K);
    double i_Kr = g_Kr*sqrt(K_o/5.4)*xr1*xr2*(V-E_K);
    double E_Ks = R*T/F*log((K_o+P_kna*Na_o)/(K_i+P_kna*Na_i));
    double i_Ks = g_Ks*xs*xs*(V-E_Ks);
    double i_CaL = g_CaL*d*pf*f2*fCass*4.0*(V-15.0)*pow(F,2)/(R*T)*(0.25*Ca_ss*exp(2*(V-15.0)*F/(R*T))-Ca_o)/(exp(2*(V-15.0)*F/(R*T))-1.0);
    double i_NaK = P_NaK*K_o/(K_o+K_mk)*Na_i/(Na_i+K_mNa)/(1.0+0.1245*exp(-0.1*V*F/(R*T))+0.0353*exp(-V*F/(R*T)));
    double E_Na = R*T/F*log(Na_o/Na_i);
    double i_Na = g_Na*m*m*m*h*j*(V-E_Na);
    double i_b_Na = g_bna*(V-E_Na);
    double i_NaCa = K_NaCa*(exp(gamma*V*F/(R*T))*pow(Na_i,3)*Ca_o-         exp((gamma-1.0)*V*F/(R*T))*pow(Na_o,3)*Ca_i*alpha)/((pow(Km_Nai,3)+pow(Na_o,3))*(Km_Ca+Ca_o)*(1.0+K_sat*exp((gamma-1.0)*V*F/(R*T))));
    double E_Ca = 0.5*R*T/F*log(Ca_o/Ca_i);
    double i_b_Ca = g_bca*(V-E_Ca);
    double i_p_K = g_pK*(V-E_K)/(1.0+exp((25.0-V)/5.98));
    double i_p_Ca = g_pCa*Ca_i/(Ca_i+K_pCa);
    
    double dV_dt = -(i_K1+i_to+i_Kr+i_Ks+i_CaL+i_NaK+i_Na+i_b_Na+i_NaCa+i_b_Ca+i_p_K+i_p_Ca);

    return dV_dt;
  }
  
  
  
  TenTusscher::Gating TenTusscher::gatingRhs(double u, Gating const& v) const
  {
 
    // State variables
    double V=u;
    double d=v[0];
    double f2=v[1];
    double fCass=v[2];
    double pf=v[3];
    double Ca_sr=v[4];
    double Ca_i=v[5];
    double Ca_ss=v[6];
    double Rel=v[7];
    double h=v[8];
    double j=v[9];
    double m=v[10];
    double K_i=v[11];
    double xr1=v[12];
    double xr2=v[13];
    double xs=v[14];
    double Na_i=v[15];
    double pr=v[16];
    double s=v[17];        

    // Constants
    double g_CaL = 0.0000398;
    double g_bca = 0.000592;
    double Buf_c = 0.2;
    double Buf_sr = 10.0;
    double Buf_ss = 0.4;
    double Ca_o = 2.0;
    double EC = 1.5;
    double K_buf_c = 0.001;
    double K_buf_sr = 0.3;
    double K_buf_ss = 0.00025;
    double K_up = 0.00025;
    double V_leak = 0.00036;
    double V_rel = 0.102;
    double V_sr = 0.001094;
    double V_ss = 0.00005468;
    double V_xfer = 0.0038;
    double Vmax_up = 0.006375;
    double k1_prime = 0.15;
    double k2_prime = 0.045;
    double k3 = 0.06;
    double k4 = 0.005;
    double max_sr = 2.5;
    double min_sr = 1.0;
    double K_pCa = 0.0005;
    double g_pCa = 0.1238;
    double g_Na = 14.838;
    double g_K1 = 5.405;
    double Cm = 0.185;
    double F = 96485.3415;
    double R = 8314.472;
    double T = 310.0;
    double V_c = 0.016404;
    double K_o = 5.4;
    double g_pK = 0.0146;
    double g_Kr = 0.153;
    double P_kna = 0.03;
    double g_Ks = 0.392;
    double g_bna = 0.00029;
    double K_NaCa = 1000.0;
    double K_sat = 0.1;
    double Km_Ca = 1.38;
    double Km_Nai = 87.5;
    double alpha = 2.5;
    double gamma = 0.35;
    double Na_o = 140.0;
    double K_mNa = 40.0;
    double K_mk = 1.0;
    double P_NaK = 2.724;
    double g_to = 0.294;

    // Compute
    double i_CaL = g_CaL*d*pf*f2*fCass*4.0*(V-15.0)*pow(F,2)/(R*T)*(0.25*Ca_ss*exp(2*(V-15.0)*F/(R*T))-Ca_o)/(exp(2*(V-15.0)*F/(R*T))-1.0);
    double d_inf = 1.0/(1.0+exp((-8.0-V)/7.5));
    double alpha_d = 1.4/(1.0+exp((-35.0-V)/13.0))+0.25;
    double beta_d = 1.4/(1.0+exp((V+5.0)/5.0));
    double gamma_d = 1.0/(1.0+exp((50.0-V)/20.0));
    double tau_d = 1.0*alpha_d*beta_d+gamma_d;
    double dd_dt=(d_inf-d)/tau_d;

    double f2_inf = 0.67/(1.0+exp((V+35.0)/7.0))+0.33;
    double tau_f2 = 562.0*exp(-pow(V+27.0,2)/240.0)+31.0/(1.0+exp((25.0-V)/10.0))+80.0/(1.0+exp((V+30.0)/10.0));
    double df2_dt=(f2_inf-f2)/tau_f2;

    double fCass_inf = 0.60/(1.0+pow((Ca_ss/0.05),2))+0.4;
    double tau_fCass = 80.0/(1.0+pow((Ca_ss/0.05),2))+2.0;
    double dfCass_dt=(fCass_inf-fCass)/tau_fCass;

    double f_inf = 1.0/(1.0+exp((V+20.0)/7.0));
    double tau_f = 1102.50*exp(-pow((V+27.0),2)/225.0)+200.0/(1.0+exp((13.0-V)/10.0))+180.0/(1.0+exp((V+30.0)/10.0))+20.0;
    double dpf_dt=(f_inf-pf)/tau_f;

    double E_Ca = 0.5*R*T/F*log(Ca_o/Ca_i);
    double i_b_Ca = g_bca*(V-E_Ca);
    double kcasr = max_sr-(max_sr-min_sr)/(1.0+pow((EC/Ca_sr),2));
    double k1 = k1_prime/kcasr;
    double O = k1*pow(Ca_ss,2)*Rel/(k3+k1*pow(Ca_ss,2));
    double i_rel = V_rel*O*(Ca_sr-Ca_ss);
    double i_up = Vmax_up/(1.0+pow(K_up,2)/pow(Ca_i,2));
    double i_leak = V_leak*(Ca_sr-Ca_i);
    double i_xfer = V_xfer*(Ca_ss-Ca_i);
    double k2 = k2_prime*kcasr;
    double dRel_dt = -k2*Ca_ss*Rel+k4*(1.0-Rel);

    double Ca_i_bufc = 1.0/(1.0+Buf_c*K_buf_c/pow((Ca_i+K_buf_c),2));
    double Ca_sr_bufsr = 1.0/(1.0+Buf_sr*K_buf_sr/pow((Ca_sr+K_buf_sr),2));
    double Ca_ss_bufss = 1.0/(1.0+Buf_ss*K_buf_ss/pow((Ca_ss+K_buf_ss),2));
    double i_p_Ca = g_pCa*Ca_i/(Ca_i+K_pCa);
    double i_NaCa = K_NaCa*(exp(gamma*V*F/(R*T))*pow(Na_i,3)*Ca_o-exp((gamma-1.0)*V*F/(R*T))*pow(Na_o,3)*Ca_i*alpha)/((pow(Km_Nai,3)+pow(Na_o,3))*(Km_Ca+Ca_o)*(1.0+K_sat*exp((gamma-1.0)*V*F/(R*T))));
    double dCa_i_dt = Ca_i_bufc*((i_leak-i_up)*V_sr/V_c+i_xfer-1.0*(i_b_Ca+i_p_Ca-2.0*i_NaCa)*Cm/(2.0*1.0*V_c*F));

    double dCa_sr_dt = Ca_sr_bufsr*(i_up-(i_rel+i_leak));

    double dCa_ss_dt = Ca_ss_bufss*(-1.0*i_CaL*Cm/(2.0*1.0*V_ss*F)+i_rel*V_sr/V_ss-i_xfer*V_c/V_ss);

    double E_Na = R*T/F*log(Na_o/Na_i);
    double i_Na = g_Na*m*m*m*h*j*(V-E_Na);
    double h_inf = 1.0/pow((1.0+exp((V+71.55)/7.43)),2);

    double alpha_h = 0.0;
    if(V<-40.0)
      alpha_h = 0.057*exp(-(V+80.0)/6.8);
    else
      alpha_h = 0.0;
   
    double beta_h = 0.0; 
    if(V<-40.0)
      beta_h = 2.7*exp(0.079*V)+310000.0*exp(0.3485*V);
    else
      beta_h = 0.77/(0.13*(1.0+exp((V+10.66)/(-11.1))));

    double tau_h = 1.0/(alpha_h+beta_h);
    double dh_dt=(h_inf-h)/tau_h;

    double j_inf = 1.0/pow((1.0+exp((V+71.55)/7.43)),2);

    double alpha_j = 0.0;
    if(V<-40.0)
      alpha_j = (-25428.0*exp(0.2444*V)-6.948e-6*exp(-0.04391*V))*(V+37.78)/1.0/(1.0+exp(0.311*(V+79.23)));
    else
      alpha_j = 0.0;

    double beta_j = 0.0;
    if(V<-40.0)
      beta_j = 0.02424*exp(-0.01052*V)/(1.0+exp(-0.1378*(V+40.14)));
    else
      beta_j = 0.6*exp(0.057*V)/(1.0+exp(-0.1*(V+32.0)));

    double tau_j = 1.0/(alpha_j+beta_j);
    double dj_dt=(j_inf-j)/tau_j;

    double m_inf = 1.0/pow((1.0+exp((-56.86-V)/9.03)),2);
    double alpha_m = 1.0/(1.0+exp((-60.0-V)/5.0));
    double beta_m = 0.1/(1.0+exp((V+35.0)/5.0))+0.1/(1.0+exp((V-50.0)/200.0));
    double tau_m = 1.0*alpha_m*beta_m;
    double dm_dt=(m_inf-m)/tau_m;

    double xr1_inf = 1.0/(1.0+exp((-26.0-V)/7.0));
    double alpha_xr1 = 450.0/(1.0+exp((-45.0-V)/10.0));
    double beta_xr1 = 6.0/(1.0+exp((V+30.0)/11.50));
    double tau_xr1 = 1.0*alpha_xr1*beta_xr1;
    double dxr1_dt=(xr1_inf-xr1)/tau_xr1;

    double xr2_inf = 1.0/(1.0+exp((V+88.0)/24.0));
    double alpha_xr2 = 3.0/(1.0+exp((-60.0-V)/20.0));
    double beta_xr2 = 1.12/(1.0+exp((V-60.0)/20.0));
    double tau_xr2 = 1.0*alpha_xr2*beta_xr2;
    double dxr2_dt=(xr2_inf-xr2)/tau_xr2;

    double xs_inf = 1.0/(1.0+exp((-5.0-V)/14.0));
    double alpha_xs = 1400.0/sqrt(1.0+exp((5.0-V)/6.0));
    double beta_xs = 1.0/(1.0+exp((V-35.0)/15.0));
    double tau_xs = 1.0*alpha_xs*beta_xs+80.0;
    double dxs_dt=(xs_inf-xs)/tau_xs;

    double r_inf = 1.0/(1.0+exp((20.0-V)/6.0));
    double tau_r = 9.5*exp(-pow((V+40.0),2)/1800.0)+0.8;
    double dpr_dt=(r_inf-pr)/tau_r;

    double s_inf = 1.0/(1.0+exp((V+20.0)/5.0));
    double tau_s = 85.0*exp(-pow((V+45.0),2)/320.0)+5.0/(1.0+exp((V-20.0)/5.0))+3.0;
    double ds_dt=(s_inf-s)/tau_s;

    // I_K1 current
    double E_K = R*T/F*log(K_o/K_i);
    double alpha_K1 = 0.1/(1.0+exp(0.06*(V-E_K-200.0)));
    double beta_K1 = (3.0*exp(0.0002*(V-E_K+100.0))+exp(0.1*(V-E_K-10.0)))/(1.0+exp(-0.5*(V-E_K)));
    double xK1_inf = alpha_K1/(alpha_K1+beta_K1);
    double i_K1 = g_K1*xK1_inf*sqrt(K_o/5.4)*(V-E_K);

    double i_to = g_to*pr*s*(V-E_K);
    double i_Kr = g_Kr*sqrt(K_o/5.4)*xr1*xr2*(V-E_K);
    double E_Ks = R*T/F*log((K_o+P_kna*Na_o)/(K_i+P_kna*Na_i));
    double i_Ks = g_Ks*xs*xs*(V-E_Ks);
    double i_NaK = P_NaK*K_o/(K_o+K_mk)*Na_i/(Na_i+K_mNa)/(1.0+0.1245*exp(-0.1*V*F/(R*T))+0.0353*exp(-V*F/(R*T)));
    double i_b_Na = g_bna*(V-E_Na);
    double i_p_K = g_pK*(V-E_K)/(1.0+exp((25.0-V)/5.98));

    double dK_i_dt = -1.0*(i_K1+i_to+i_Kr+i_Ks+i_p_K-2.0*i_NaK)/(1.0*V_c*F)*Cm;

    double dNa_i_dt = -1.0*(i_Na+i_b_Na+3.0*i_NaK+3.0*i_NaCa)/(1.0*V_c*F)*Cm;

    // RHS
    Gating dv_dt;
    
    dv_dt[0]=dd_dt;
    dv_dt[1]=df2_dt;
    dv_dt[2]=dfCass_dt;
    dv_dt[3]=dpf_dt;
    dv_dt[4]=dCa_sr_dt;
    dv_dt[5]=dCa_i_dt;
    dv_dt[6]=dCa_ss_dt;
    dv_dt[7]=dRel_dt;
    dv_dt[8]=dh_dt;
    dv_dt[9]=dj_dt;
    dv_dt[10]=dm_dt;
    dv_dt[11]=dK_i_dt;
    dv_dt[12]=dxr1_dt;
    dv_dt[13]=dxr2_dt;
    dv_dt[14]=dxs_dt;
    dv_dt[15]=dNa_i_dt;
    dv_dt[16]=dpr_dt;
    dv_dt[17]=ds_dt;
    
    for (int i=0; i<nGating; ++i)
      if (std::isnan(dv_dt[i]))
//         throw DetailedException("NaN occured in ten Tusscher computation",__FILE__,__LINE__);
        std::cerr << "dv_dt[" << i << "] = NaN\n";

    return dv_dt;
  }
  
  
};
