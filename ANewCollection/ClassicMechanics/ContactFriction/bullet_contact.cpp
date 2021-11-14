//bullet
static inline void b3SolveContact(b3ContactConstraint4& cs,
								  const b3Vector3& posA, b3Vector3& linVelA, b3Vector3& angVelA, float invMassA, const b3Matrix3x3& invInertiaA,
								  const b3Vector3& posB, b3Vector3& linVelB, b3Vector3& angVelB, float invMassB, const b3Matrix3x3& invInertiaB,
								  float maxRambdaDt[4], float minRambdaDt[4])
{
	b3Vector3 dLinVelA;
	dLinVelA.setZero();
	b3Vector3 dAngVelA;
	dAngVelA.setZero();
	b3Vector3 dLinVelB;
	dLinVelB.setZero();
	b3Vector3 dAngVelB;
	dAngVelB.setZero();

	for (int ic = 0; ic < 4; ic++)
	{
		//	dont necessary because this makes change to 0
		if (cs.m_jacCoeffInv[ic] == 0.f) continue;

		{
			b3Vector3 angular0, angular1, linear;
			b3Vector3 r0 = cs.m_worldPos[ic] - (b3Vector3&)posA;
			b3Vector3 r1 = cs.m_worldPos[ic] - (b3Vector3&)posB;
			b3SetLinearAndAngular((const b3Vector3&)-cs.m_linear, (const b3Vector3&)r0, (const b3Vector3&)r1, linear, angular0, angular1);

			float rambdaDt = b3CalcRelVel((const b3Vector3&)cs.m_linear, (const b3Vector3&)-cs.m_linear, angular0, angular1,
										  linVelA, angVelA, linVelB, angVelB) +
							 cs.m_b[ic];
			rambdaDt *= cs.m_jacCoeffInv[ic];

			{
				float prevSum = cs.m_appliedRambdaDt[ic];
				float updated = prevSum;
				updated += rambdaDt;
				updated = b3Max(updated, minRambdaDt[ic]);
				updated = b3Min(updated, maxRambdaDt[ic]);
				rambdaDt = updated - prevSum;
				cs.m_appliedRambdaDt[ic] = updated;
			}

			b3Vector3 linImp0 = invMassA * linear * rambdaDt;
			b3Vector3 linImp1 = invMassB * (-linear) * rambdaDt;
			b3Vector3 angImp0 = (invInertiaA * angular0) * rambdaDt;
			b3Vector3 angImp1 = (invInertiaB * angular1) * rambdaDt;
#ifdef _WIN32
			b3Assert(_finite(linImp0.getX()));
			b3Assert(_finite(linImp1.getX()));
#endif
			{
				linVelA += linImp0;
				angVelA += angImp0;
				linVelB += linImp1;
				angVelB += angImp1;
			}
		}
	}
}
