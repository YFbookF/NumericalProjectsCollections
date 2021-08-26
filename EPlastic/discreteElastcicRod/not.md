```
// Compute the curvature binormal for a vertex between two edges with tangents
// e0 and e1, respectively
// (edge tangent vectors not necessarily normalized)
template<typename Real_>
Vec3_T<Real_> curvatureBinormal(const Vec3_T<Real_> &e0, const Vec3_T<Real_> &e1) {
    return e0.cross(e1) * (2.0 / (e0.norm() * e1.norm() + e0.dot(e1)));
}
```

https://github.com/jpanetta/ElasticRods 注释详细

https://github.com/Reimilia/DiscreteElasticRods