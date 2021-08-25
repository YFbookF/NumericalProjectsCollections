Scalable-Locally-Injective-Mappings-master

BCQN

HOT

SymmetricDirichlet.cpp



```
do
	{

		const double relativeEpsilon = 0.0001 * max(1.0, x.norm());

		if (grad.norm() < relativeEpsilon)
			break;

		//Algorithm 7.4 (L-BFGS two-loop recursion)
		q = grad;
		const int k = min(m, iter);

		// for i k − 1, k − 2, . . . , k − m
		for (int i = k - 1; i >= 0; i--)
		{
			// alpha_i <- rho_i*s_i^T*q
			const double rho = 1.0 / static_cast<Eigen::VectorXd>(sVector.col(i)).dot(static_cast<Eigen::VectorXd>(yVector.col(i)));
			alpha(i) = rho * static_cast<Eigen::VectorXd>(sVector.col(i)).dot(q);
			// q <- q - alpha_i*y_i
			q = q - alpha(i) * yVector.col(i);
		}
		// r <- H_k^0*q
		q = H0k * q;
		//for i k − m, k − m + 1, . . . , k − 1
		for (int i = 0; i < k; i++)
		{
			// beta <- rho_i * y_i^T * r
			const double rho = 1.0 / static_cast<Eigen::VectorXd>(sVector.col(i)).dot(static_cast<Eigen::VectorXd>(yVector.col(i)));
			const double beta = rho * static_cast<Eigen::VectorXd>(yVector.col(i)).dot(q);
			// r <- r + s_i * ( alpha_i - beta)
			q = q + sVector.col(i) * (alpha(i) - beta);
		}
		// stop with result "H_k*f_f'=q"

		// any issues with the descent direction ?
		double descent = grad.dot(q);
		double alpha_init = 1.0 / grad.norm();
		if (descent < 0.0001 * relativeEpsilon) {
			cout << "hopa!" << endl; //int blabla; cin >> blabla;
			q = grad;
			iter = 0;
			alpha_init = 1.0;
		}

		// find steplength
		//const double rate = WolfeRule::linesearch(x, -q,  FunctionValue, FunctionGradient, alpha_init) ;
		//x = x - rate * q;

		// update guess
		//q=q/q.norm(); //TODO: remove me!
		
```

**Large-scale L-BFGS using MapReduce**

