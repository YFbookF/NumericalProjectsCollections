//bullet
static inline void b3SolveFriction(b3ContactConstraint4& cs,
								   const b3Vector3& posA, b3Vector3& linVelA, b3Vector3& angVelA, float invMassA, const b3Matrix3x3& invInertiaA,
								   const b3Vector3& posB, b3Vector3& linVelB, b3Vector3& angVelB, float invMassB, const b3Matrix3x3& invInertiaB,
								   float maxRambdaDt[4], float minRambdaDt[4])
{
	if (cs.m_fJacCoeffInv[0] == 0 && cs.m_fJacCoeffInv[0] == 0) return;
	const b3Vector3& center = (const b3Vector3&)cs.m_center;

	b3Vector3 n = -(const b3Vector3&)cs.m_linear;

	b3Vector3 tangent[2];

	b3PlaneSpace1(n, tangent[0], tangent[1]);

	b3Vector3 angular0, angular1, linear;
	b3Vector3 r0 = center - posA;
	b3Vector3 r1 = center - posB;
	for (int i = 0; i < 2; i++)
	{
		b3SetLinearAndAngular(tangent[i], r0, r1, linear, angular0, angular1);
		float rambdaDt = b3CalcRelVel(linear, -linear, angular0, angular1,
									  linVelA, angVelA, linVelB, angVelB);
		rambdaDt *= cs.m_fJacCoeffInv[i];

		{
			float prevSum = cs.m_fAppliedRambdaDt[i];
			float updated = prevSum;
			updated += rambdaDt;
			updated = b3Max(updated, minRambdaDt[i]);
			updated = b3Min(updated, maxRambdaDt[i]);
			rambdaDt = updated - prevSum;
			cs.m_fAppliedRambdaDt[i] = updated;
		}

		b3Vector3 linImp0 = invMassA * linear * rambdaDt;
		b3Vector3 linImp1 = invMassB * (-linear) * rambdaDt;
		b3Vector3 angImp0 = (invInertiaA * angular0) * rambdaDt;
		b3Vector3 angImp1 = (invInertiaB * angular1) * rambdaDt;
#ifdef _WIN32
		b3Assert(_finite(linImp0.getX()));
		b3Assert(_finite(linImp1.getX()));
#endif
		linVelA += linImp0;
		angVelA += angImp0;
		linVelB += linImp1;
		angVelB += angImp1;
	}

	{  //	angular damping for point constraint
		b3Vector3 ab = (posB - posA).normalized();
		b3Vector3 ac = (center - posA).normalized();
		if (b3Dot(ab, ac) > 0.95f || (invMassA == 0.f || invMassB == 0.f))
		{
			float angNA = b3Dot(n, angVelA);
			float angNB = b3Dot(n, angVelB);

			angVelA -= (angNA * 0.1f) * n;
			angVelB -= (angNB * 0.1f) * n;
		}
	}
}