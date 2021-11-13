
bool MassProperties::DecomposePrincipalMomentsOfInertia(Mat44 &outRotation, Vec3 &outDiagonal) const
{
	// Using eigendecomposition to get the principal components of the inertia tensor
	// See: https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix
	Matrix<3, 3> inertia;
	inertia.CopyPart(mInertia, 0, 0, 3, 3, 0, 0);
	Matrix<3, 3> eigen_vec = Matrix<3, 3>::sIdentity();
	Vector<3> eigen_val;
	if (!EigenValueSymmetric(inertia, eigen_vec, eigen_val))
		return false;

	// Sort so that the biggest value goes first
	int indices[] = { 0, 1, 2 };
	sort(indices, indices + 3, [&eigen_val](int inLeft, int inRight) -> bool { return eigen_val[inLeft] > eigen_val[inRight]; });
		
	// Convert to a regular Mat44 and Vec3
	outRotation = Mat44::sIdentity();
	for (int i = 0; i < 3; ++i)
	{
		outRotation.SetColumn3(i, Vec3(reinterpret_cast<Float3 &>(eigen_vec.GetColumn(indices[i]))));
		outDiagonal.SetComponent(i, eigen_val[indices[i]]);
	}

	// Make sure that the rotation matrix is a right handed matrix
	if (outRotation.GetAxisX().Cross(outRotation.GetAxisY()).Dot(outRotation.GetAxisZ()) < 0.0f)
		outRotation.SetAxisZ(-outRotation.GetAxisZ());

#ifdef JPH_ENABLE_ASSERTS
	// Validate that the solution is correct, for each axis we want to make sure that the difference in inertia is
	// smaller than some fraction of the inertia itself in that axis
	Mat44 new_inertia = outRotation * Mat44::sScale(outDiagonal) * outRotation.Inversed();
	for (int i = 0; i < 3; ++i)
		JPH_ASSERT(new_inertia.GetColumn3(i).IsClose(mInertia.GetColumn3(i), mInertia.GetColumn3(i).LengthSq() * 1.0e-10f));
#endif

	return true;
}