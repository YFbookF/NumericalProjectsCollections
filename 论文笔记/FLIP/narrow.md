http://www.tkim.graphics/CPT/source.html

```
///////////////////////////////////////////////////////////////////////
// from "Level Set Surface Editing Operators", Museth et al. 2002
// "Geometric Surface Processing via Normal Maps", Tasdizen 2003
///////////////////////////////////////////////////////////////////////
void FIELD_3D::principalCurvatures(FIELD_3D& minCurvature, FIELD_3D& maxCurvature) const
{
#pragma omp parallel
#pragma omp for  schedule(static)
	for (int z = 0; z < _zRes; z++)
		for (int y = 0; y < _yRes; y++)
			for (int x = 0; x < _xRes; x++)
			{
				int index = x + y * _xRes + z * _slabSize;

				MATRIX3 N = Dnormal(x, y, z);
				VEC3F n = normal(x, y, z);
				MATRIX3 outer = MATRIX3::outer_product(n, n);

				MATRIX3 B = N * (MATRIX3::I() - outer);

				Real D = sqrt(B.squaredSum());
				Real H = trace(B) * 0.5;

				Real discrim = D * D * 0.5 - H * H;

				if (discrim < 0.0)
				{
					minCurvature[index] = 0;
					maxCurvature[index] = 0;
					continue;
				}

				Real root = sqrt(discrim);
				Real k1 = H + root;
				Real k2 = H - root;

				maxCurvature[index] = (fabs(k1) > fabs(k2)) ? k1 : k2;
				minCurvature[index] = (fabs(k1) > fabs(k2)) ? k2 : k1;
			}
}
```

