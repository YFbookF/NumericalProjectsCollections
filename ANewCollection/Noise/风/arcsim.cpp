//physics.cpp
Vec3 wind_force(const Face* face, const Wind& wind)
{
    Vec3 vface = (face->v[0]->node->v + face->v[1]->node->v + face->v[2]->node->v) / 3.;
    Vec3 vrel = wind.velocity - vface;
    double vn = dot(face->n, vrel);
    Vec3 vt = vrel - vn * face->n;
    return wind.density * face->a * abs(vn) * vn * face->n + wind.drag * face->a * vt;
}