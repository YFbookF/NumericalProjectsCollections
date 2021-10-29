//constraint.cpp
double IneqCon::energy(double value)
{
    double v = violation(value);
    return stiff * v * v * v / ::magic.repulsion_thickness / 6;
}
double IneqCon::energy_grad(double value)
{
    return -stiff * sq(violation(value)) / ::magic.repulsion_thickness / 2;
}
double IneqCon::energy_hess(double value)
{
    return stiff * violation(value) / ::magic.repulsion_thickness;
}