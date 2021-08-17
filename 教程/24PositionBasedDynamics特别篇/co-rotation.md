A Robust Method to Extract the Rotational Part of Deformations  

Understanding the deformation gradient in Abaqus and key
guidelines for anisotropic hyperelastic user material subroutines
(UMATs)  

https://github.com/LallyLabTCD/localBasisAbaqus

https://github.com/mmacklin/sandbox

计算deformationGradient，小x是世界坐标，X是物质坐标
$$
\bold F(\bold X) = \frac{\partial \bold x}{\partial \bold X}
$$
这么说肯定记不住了，看看sandbox的代码

```
// deformation gradient
Matrix22 CalcDeformation(const Vec2 x[3], const Matrix22& invM)
{	
	Vec2 e1 = x[1]-x[0]; 
	Vec2 e2 = x[2]-x[0]; 

	Matrix22 m(e1, e2);

	// mapping from material coordinates to world coordinates	
	Matrix22 f = m*invM; 	
	return f;
}
```

sandbox的代码的超级有价值，特别是有限元部分

Invertible Finite Elements For Robust Simulation of Large
Deformation  