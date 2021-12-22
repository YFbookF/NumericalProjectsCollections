# CHRONO

E:\mycode\Elastic\chrono-develop\src\chrono\fea\ChContinuumMaterial.cpp

```
void ChContinuumPlasticVonMises::ComputeReturnMapping(ChStrainTensor<>& mplasticstrainflow,
                                                      const ChStrainTensor<>& mincrementstrain,
                                                      const ChStrainTensor<>& mlastelasticstrain,
                                                      const ChStrainTensor<>& mlastplasticstrain) const {
    ChStrainTensor<> guesselstrain(mlastelasticstrain);
    guesselstrain += mincrementstrain;  // assume increment is all elastic

    double vonm = guesselstrain.GetEquivalentVonMises();
    if (vonm > this->elastic_yeld) {
        ChVoightTensor<> mdev;
        guesselstrain.GetDeviatoricPart(mdev);
        mplasticstrainflow = mdev * ((vonm - this->elastic_yeld) / (vonm));
    } else {
        mplasticstrainflow.setZero();
    }
}

```

![image-20211218215837422](E:\mycode\collection\定理\弹性力学\image-20211218215837422.png)

D:\图形学书籍\图形学书籍\有限元\非线性专项\Nonlinear Continuum Mechanics for Finite Element Analysis, 2nd Edition by Javier Bonet, Richard D. Wood (z-lib.org).pdf

![image-20211220093001200](E:\mycode\collection\定理\弹性力学\image-20211220093001200.png)

![image-20211220093218210](E:\mycode\collection\定理\弹性力学\image-20211220093218210.png)

![image-20211220093254382](E:\mycode\collection\定理\弹性力学\image-20211220093254382.png)

![image-20211220100531000](E:\mycode\collection\定理\弹性力学\image-20211220100531000.png)

==========================

D:\图形学书籍\图形学书籍\固体物理\Computational Continuum Mechanics by Ahmed A. Shabana (z-lib.org).pdf

![image-20211220102153318](E:\mycode\collection\定理\弹性力学\image-20211220102153318.png)

D:\图形学书籍\论文\Anisotropic Elastoplasticity for Cloth, Knit and Hair Frictional Contact.pdf

![image-20211221224808522](E:\mycode\collection\定理\弹性力学\image-20211221224808522.png)

https://github.com/2iw31Zhv/AnisotropicElastoplasticity

```
for (int f = 0; f < mesh_->elementDirections_1.rows(); ++f)
		{

			Matrix3d virtualElementDirection;
			virtualElementDirection.col(0) = mesh_->elementDirections_1.row(f);
			virtualElementDirection.col(1) = mesh_->elementDirections_2.row(f);
			virtualElementDirection.col(2) = mesh_->elementDirections_3.row(f);

			Matrix3d Q, R;
			GramSchmidtOrthonomalization(Q, R, virtualElementDirection);

			// main return mapping algorithm

			if (R(2, 2) > 1.0)
			{
				R(2, 2) = 1.0;
				R(0, 2) = R(1, 2) = 0.0;
			}
			else
			{
				double normalForce = mesh_->stiffness * (R(2, 2) - 1.0) * (R(2, 2) - 1.0);
				double shearForce = mesh_->shearStiffness * sqrt(
					R(0, 2) * R(0, 2) + R(1, 2) * R(1, 2));


				if (shearForce > mesh_->frictionCoeff * normalForce)
				{
					R(0, 2) *= mesh_->frictionCoeff * normalForce / shearForce;
					R(1, 2) *= mesh_->frictionCoeff * normalForce / shearForce;
				}
			}

			Vector3d d3 = Q * R.col(2);
			mesh_->elementDirections_3.row(f) = d3;
```

==================chrono

```

void ChContinuumPlasticVonMises::ComputeReturnMapping(ChStrainTensor<>& mplasticstrainflow,
                                                      const ChStrainTensor<>& mincrementstrain,
                                                      const ChStrainTensor<>& mlastelasticstrain,
                                                      const ChStrainTensor<>& mlastplasticstrain) const {
    ChStrainTensor<> guesselstrain(mlastelasticstrain);
    guesselstrain += mincrementstrain;  // assume increment is all elastic

    double vonm = guesselstrain.GetEquivalentVonMises();
    if (vonm > this->elastic_yeld) {
        ChVoightTensor<> mdev;
        guesselstrain.GetDeviatoricPart(mdev);
        mplasticstrainflow = mdev * ((vonm - this->elastic_yeld) / (vonm));
    } else {
        mplasticstrainflow.setZero();
    }
}

```

=====================

D:\图形学书籍\图形学书籍\固体布料数学\裂缝\Computational Methods for Plasticity Theory and Applications by EA de Souza Neto, Prof. D Periæ, Prof. DRJ Owen (z-lib.org).pdf

![image-20211222105127308](E:\mycode\collection\定理\弹性力学\image-20211222105127308.png)

![image-20211222105141570](E:\mycode\collection\定理\弹性力学\image-20211222105141570.png)
