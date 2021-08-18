```
bool PositionBasedElasticRods::computeDarbouxVector(const Matrix3r& dA, const Matrix3r& dB, const Real mid_edge_length, Vector3r& darboux_vector)
{
	Real factor = static_cast<Real>(1.0) + dA.col(0).dot(dB.col(0)) + dA.col(1).dot(dB.col(1)) + dA.col(2).dot(dB.col(2));

	factor = static_cast<Real>(2.0) / (mid_edge_length * factor);

	for (int c = 0; c < 3; ++c)
	{
		const int i = permutation[c][0];
		const int j = permutation[c][1];
		const int k = permutation[c][2];
		darboux_vector[i] = dA.col(j).dot(dB.col(k)) - dA.col(k).dot(dB.col(j));
	}
	darboux_vector *= factor;
	return true;
}

```

modified discrete Darboux vector
$$
\Omega_i = \frac{4}{l}\frac{vect(\bold Q)_I}{1 + tr\bold Q} = \frac{2}{l}\frac{\bold d_e^j \cdot \bold d_{e+1}^k - \bold d_{e}^k \cdot \bold d_{e+1}^j}{1 + \sum_{n=1}^3 \bold d_e^n \cdot \bold d_{e+1}^n}
$$
Position-based Elastic Rods  