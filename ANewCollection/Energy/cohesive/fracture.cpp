//https://github.com/ElsevierSoftwareX/SOFTX-D-21-00041/blob/main/src/interface_laws/rate_and_state_law.cc
/* -------------------------------------------------------------------------- */
void RateAndStateLaw::computeCohesiveForces(
    std::vector<NodalField *> &cohesion, bool predicting) {
  unsigned int nb = this->mesh.getNbNodes();

  // find forces needed to close normal gap
  this->interface->closingNormalGapForce(cohesion[1], predicting);
  // double * coh_1_p = cohesion[1]->storage();

  // find force needed to maintain shear gap
  this->interface->maintainShearGapForce(cohesion);
  double * coh_0_p = cohesion[0]->storage();

  // interface properties
  const HalfSpace & top = this->interface->getTop();
  const HalfSpace & bot = this->interface->getBot();
  const Material & mat_top = top.getMaterial();
  const Material & mat_bot = bot.getMaterial();
  double fact_top = mat_top.getCs() / mat_top.getShearModulus();
  double fact_bot = mat_bot.getCs() / mat_bot.getShearModulus();
  double fact_both = fact_top + fact_bot;

  const double *a_p = this->a->storage();
  const double *b_p = this->b->storage();
  const double *Dc_p = this->Dc->storage();
  const double *sigma_p = this->sigma->storage();
  const double *int0top_p = top.getInternal(0)->storage();
  const double *int0bot_p = bot.getInternal(0)->storage();
  const double *ext0_p = this->interface->getLoad(0)->storage();

  std::vector<NodalField *> gap_velo = this->interface->getBufferField();
  // compute delta_dot: do not use v* here -- predicting = false
  this->interface->computeGapVelocity(gap_velo, false);
  // pass true to compute slip rate only in shear directions vectorially
  this->interface->computeNorm(gap_velo, *(this->V), true);
  const double * V_p = this->V->storage();

  // compute theta
  this->computeTheta(predicting ? this->theta_pc : this->theta, this->V);
  double * theta_p = (predicting ? this->theta_pc : this->theta)->storage();

  double * iterations_p = this->iterations->storage();
  double * rel_error_p = this->rel_error->storage();
  // double * abs_error_p = this->abs_error->storage();

  // solve tau_coh using Newton-Raphson
  // V is then solved in Interface::advanceTimeStep()
  for (unsigned int i = 0; i < nb; ++i) {
    double Z;
    if (evolution_law == EvolutionLaw::SlipLawWithStrongRateWeakening){
      Z = 0.5 / V0 * std::exp(theta_p[i] / a_p[i]);
    } else {
      Z = 0.5 / V0 * std::exp((f0 + b_p[i] * std::log(V0 * theta_p[i] / Dc_p[i])) / a_p[i]);
    }

    // initial guess
    double v_prev = V_p[i];
    double tau, dtau, F, dF, v;
    double rel_change = 1;
    double rel_tol = 1e-8;
    unsigned max_iter = 1000;
    unsigned min_iter = 0;
    unsigned iter = 0;
    unsigned sign_change_count = 0;
    // Newton-Raphson
    do {
      ++iter;
      tau = a_p[i] * sigma_p[i] * std::asinh(Z * (v_prev + Vplate));
      dtau = a_p[i] * sigma_p[i] * Z / std::sqrt(1.0 + Z * Z * (v_prev + Vplate) * (v_prev + Vplate));
      F = fact_both * (ext0_p[i] - tau) + fact_top * int0top_p[i] +
          fact_bot * int0bot_p[i] - v_prev;
      dF = -fact_both * dtau - 1.0;
      v = v_prev - F / dF;
      rel_change = std::abs(F / dF / v_prev);
      // catching infinite loop when v is jumping across 0 more than 4 times
      // set v to 0 (Vp) and continue the iterations
      if ((v_prev + Vplate) * (v + Vplate) < 0) ++sign_change_count;
      if (sign_change_count >= 4) {
        v = -Vplate;
        sign_change_count = 0;
        rel_change = 1.0;
      }
      if (!std::isfinite(v)) {
        v = -Vplate;
        rel_change = 1.0;
      } else if (!std::isfinite(rel_change)) {
        rel_change = 1.0;
      }
      if (!std::isfinite(v)){
        v = -Vplate;
        rel_change = 1;
      }
      if (!std::isfinite(rel_change)) rel_change = 1.0;
      v_prev = v;
    } while ((rel_change > rel_tol && iter < max_iter) || iter < min_iter);
    if (iter == max_iter) {
      throw std::runtime_error("Newton-Raphson not converged in RateAndStateLaw::computeCohesiveForces");
    }
    tau = a_p[i] * sigma_p[i] * std::asinh(Z * (v + Vplate));
    coh_0_p[i] = tau;
    iterations_p[i] = iter;
    rel_error_p[i] = rel_change;
  }
}
