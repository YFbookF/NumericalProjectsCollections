//transform.cpp
Transformation Transformation::operator/(double s) const
{
    return (*this) * (1. / s);
}

Quaternion Quaternion::operator+(const Quaternion& other) const
{
    Quaternion q;
    q.v = this->v + other.v;
    q.s = this->s + other.s;
    return q;
}

Quaternion Quaternion::operator-(const Quaternion& other) const
{
    Quaternion q;
    q.v = this->v - other.v;
    q.s = this->s - other.s;
    return q;
}

Quaternion Quaternion::operator-() const
{
    Quaternion q;
    q.v = -this->v;
    q.s = -this->s;
    return q;
}

Quaternion Quaternion::operator*(const Quaternion& other) const
{
    Quaternion q;
    q.v = (this->s * other.v) + (other.s * this->v) + cross(this->v, other.v);
    q.s = (this->s * other.s) - dot(this->v, other.v);
    return q;
}

Quaternion Quaternion::operator*(double s) const
{
    Quaternion q;
    q.v = this->v * s;
    q.s = this->s * s;
    return q;
}

Quaternion Quaternion::operator/(double s) const
{
    return (*this) * (1. / s);
}

Vec3 Quaternion::rotate(const Vec3& x) const
{
    return x * (sq(s) - dot(v, v)) + 2. * v * dot(v, x) + 2. * cross(v, x) * s;
}

Quaternion inverse(const Quaternion& q)
{
    Quaternion in;
    double divisor = norm2(q);
    in.s = q.s / divisor;
    in.v = -q.v / divisor;
    return in;
}

Quaternion Quaternion::from_axisangle(const Vec3& axis, double angle)
{
    Quaternion q;
    if (angle == 0) {
        q.s = 1;
        q.v = Vec3(0);
    } else {
        q.s = cos(angle / 2);
        q.v = sin(angle / 2) * normalize(axis);
    }
    return q;
}

pair<Vec3, double> Quaternion::to_axisangle() const
{
    double angle = 2 * acos(s);
    Vec3 axis;
    if (angle == 0) {
        axis = Vec3(1);
    } else {
        axis = v / sqrt(1.0 - s * s);
    }
    return pair<Vec3, double>(axis, angle);
}
