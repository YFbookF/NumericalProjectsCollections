A Robust Method to Extract the Rotational Part of Deformations  

难得一见的附代码的论文

它说的正确的旋转就是
$$
\bold R <= exp(\frac{\sum_i \bold r_i \times \bold a_i}{|\sum _ i\bold r \cdot \bold a_i| + \varepsilon})\bold R
$$

```
	for (unsigned int iter = 0; iter < maxIter; iter++)
	{
		Matrix3r R = q.matrix();
		Vector3r omega = (R.col(0).cross(A.col(0)) + R.col(1).cross(A.col(1)) + R.col(2).cross(A.col(2))) * 
			(1.0 / fabs(R.col(0).dot(A.col(0)) + R.col(1).dot(A.col(1)) + R.col(2).dot(A.col(2)) + 1.0e-9));
		Real w = omega.norm();
		if (w < 1.0e-9)
			break;
		q = Quaternionr(AngleAxisr(w, (1.0 / w) * omega)) *	q;
		q.normalize();
	}
```

