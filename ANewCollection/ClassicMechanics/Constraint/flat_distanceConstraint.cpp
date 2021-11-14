// huyuanming-FLAT
class DistanceConstraint : public Constraint {
private:
	Vector2D p;
	double L;
	Shape *shape;
	Vector2D r;
	Graphics *graphics;
public:
	DistanceConstraint(Vector2D p, double L, Shape *shape, Vector2D r, Graphics *graphics = 0) : p(p), L(L), shape(shape), r(r), graphics(graphics) {}
	void Process() {
		Vector2D q = shape->centerPosition + shape->transformToWorld(r);
		if (graphics)
			graphics->DrawLine((int)p.x, (int)p.y, (int)q.x, (int)q.y, RGB(128, 0, 127));
		Vector2D n = p - q;
		double l = n.GetLength();
		if (l < eps) return;
		n.Normalize();
		double v = n * shape->PointVelocity(q);
		double J = ((l - L) * 0.3 / timeInterval - v) / (1.0 / shape->mass + sqr(r % n) / shape->momentOfInertia);
		shape->AddImpluse(q, n * J);
	}
	Constraint *Copy() {
		return new DistanceConstraint(*this);
	}
};
