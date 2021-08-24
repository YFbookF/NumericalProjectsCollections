**Explicit Exact Formulas for the 3-D Tetrahedron Inertia Tensor** 

**in Terms of its Vertex Coordinates** 

inertia tensor E of a body D with respect to the three axes
$$
\bold E_Q = \begin{bmatrix} a & -b' & -c'\\ -b' & b & -a' \\ -c' & -a' & c \end{bmatrix}
$$
那么mu是density of medium
$$
a = \int_D \mu(y^2 + z^2)dD\\
b = \int\mu(x^2 + z^2)dD\\
c = \int_D \mu(x^2 + y^2)dD\\
a' = \int\mu yzdD\\
b' = \int_D \mu xzdD\\
c' = \int\mu xydD\\
$$

```
void TetMesh::getElementInertiaTensor(int el, Mat3d & inertiaTensor) const
{
  Vec3d a = getVertex(el, 0);
  Vec3d b = getVertex(el, 1);
  Vec3d c = getVertex(el, 2);
  Vec3d d = getVertex(el, 3);

  Vec3d center = getElementCenter(el);
  a -= center;
  b -= center;
  c -= center;
  d -= center;

  double absdetJ = 6.0 * getElementVolume(el);

  double x1 = a[0], x2 = b[0], x3 = c[0], x4 = d[0];
  double y1 = a[1], y2 = b[1], y3 = c[1], y4 = d[1];
  double z1 = a[2], z2 = b[2], z3 = c[2], z4 = d[2];

  double A = absdetJ * (y1*y1 + y1*y2 + y2*y2 + y1*y3 + y2*y3 + y3*y3 + y1*y4 + y2*y4 + y3*y4 + y4*y4 + z1*z1 + z1 * z2 + z2 * z2 + z1 * z3 + z2 * z3 + z3 * z3 + z1 * z4 + z2 * z4 + z3 * z4 + z4 * z4) / 60.0;

  double B = absdetJ * (x1*x1 + x1*x2 + x2*x2 + x1*x3 + x2*x3 + x3*x3 + x1*x4 + x2*x4 + x3*x4 + x4*x4 + z1*z1 + z1 * z2 + z2 * z2 + z1 * z3 + z2 * z3 + z3 * z3 + z1 * z4 + z2 * z4 + z3 * z4 + z4 * z4) / 60.0;

  double C = absdetJ * (x1*x1 + x1*x2 + x2*x2 + x1*x3 + x2*x3 + x3*x3 + x1*x4 + x2*x4 + x3*x4 + x4*x4 + y1*y1 + y1 * y2 + y2 * y2 + y1 * y3 + y2 * y3 + y3 * y3 + y1 * y4 + y2 * y4 + y3 * y4 + y4 * y4) / 60.0;

  double Ap = absdetJ * (2*y1*z1 + y2*z1 + y3*z1 + y4*z1 + y1*z2 + 2*y2*z2 + y3*z2 + y4*z2 + y1*z3 + y2*z3 + 2*y3*z3 + y4*z3 + y1*z4 + y2*z4 + y3*z4 + 2*y4*z4) / 120.0;

  double Bp = absdetJ * (2*x1*z1 + x2*z1 + x3*z1 + x4*z1 + x1*z2 + 2*x2*z2 + x3*z2 + x4*z2 + x1*z3 + x2*z3 + 2*x3*z3 + x4*z3 + x1*z4 + x2*z4 + x3*z4 + 2*x4*z4) / 120.0;

  double Cp = absdetJ * (2*x1*y1 + x2*y1 + x3*y1 + x4*y1 + x1*y2 + 2*x2*y2 + x3*y2 + x4*y2 + x1*y3 + x2*y3 + 2*x3*y3 + x4*y3 + x1*y4 + x2*y4 + x3*y4 + 2*x4*y4) / 120.0;

  inertiaTensor = Mat3d(A, -Bp, -Cp,   -Bp, B, -Ap,   -Cp, -Ap, C);
}
```

四面体的det，除以6就是体积

```
double TetMesh::getTetDeterminant(const Vec3d & a, const Vec3d & b, const Vec3d & c, const Vec3d & d)
{
  // computes det(A), for the 4x4 matrix A
  //     [ 1 a ]
  // A = [ 1 b ]
  //     [ 1 c ]
  //     [ 1 d ]
  // It can be shown that det(A) = dot(d - a, cross(b - a, c - a))
  // When det(A) > 0, the tet has positive orientation.
  // When det(A) = 0, the tet is degenerate.
  // When det(A) < 0, the tet has negative orientation.

  return dot(d - a, cross(b - a, c - a));
}
```

