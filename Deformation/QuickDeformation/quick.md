https://github.com/JoshWolper/quik-deform

没写完，但是关于tetConstraint部分写得很不错

2维旋转矩阵，用QR分解，三维SVD即可，EIGEN库的SVD需要稍微处理一下

解参数也用不完全cholesky，用到了simplellt

之后除了弹簧质点系统，还有有限元系统。也许不应该叫有限元，但PositionBasedDynamic库就是这么叫的，不管了。

```
	//Now have to iterate over all constraints to add to our b term!
	for (int i = 0; i < constraints.size(); i++) {

		double w = constraints[i]->getW();
		MatrixXd S = constraints[i]->getS();
		MatrixXd Stranspose = S.transpose();
		MatrixXd A = constraints[i]->getA();
		MatrixXd Atranspose = A.transpose(); //need to do this because of the way transpose works! (does not replace original)
		MatrixXd B = constraints[i]->getB();
		MatrixXd p = constraints[i]->getP();

		b += (w * Stranspose * Atranspose * B * p); //add term to b for this constraint

	}
```

