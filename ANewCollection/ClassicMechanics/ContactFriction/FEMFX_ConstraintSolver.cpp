//https://github.com/GPUOpen-Effects/FEMFX
 // Solve one constraint (block) row in Projected Gauss-Seidel iteration.
    // Update error norm used in convergence test: inf norm of solution change.
    static FM_FORCE_INLINE void FmPgsIteration3Row(
        FmSolverIterationNorms* FM_RESTRICT norms,
        FmSVector3* FM_RESTRICT lambda3,
        FmSVector3* FM_RESTRICT JTlambda,
        const FmConstraintJacobian& FM_RESTRICT J,
        const FmSVector3* FM_RESTRICT pgsRhs,
        const FmSMatrix3* FM_RESTRICT DAinv,
        const FmSVector3* FM_RESTRICT totalLambda3,
        float omega,
        uint passIdx,
        uint rowIdx)
    {
        FmConstraintParams& constraintParams = J.params[rowIdx];
        float frictionCoeff = constraintParams.frictionCoeff;
        FmSolverConstraintType type = (FmSolverConstraintType)constraintParams.type;
        uint8_t flags = constraintParams.flags;

        FmSVector3 diagInv = constraintParams.diagInverse[passIdx];

        FmSVector3 rowResult = pgsRhs[rowIdx];
        FmSVector3 lambda3Orig = lambda3[rowIdx];

        // Get current total lambda from previous outer iterations, and add with current lambda before projection
        FmSVector3 totLambda3 = (totalLambda3 == NULL) ? FmInitSVector3(0.0f) : totalLambda3[rowIdx];

        FmSMatrix3* jacobianSubmats = (FmSMatrix3*)((uint8_t*)J.submats + constraintParams.jacobianSubmatsOffset);
        uint* jacobianIndices = (uint*)((uint8_t*)J.indices + constraintParams.jacobianIndicesOffset);
        uint rowSize = constraintParams.jacobiansNumStates;

        for (uint i = 0; i < rowSize; i++)
        {
            FmSMatrix3 submat = jacobianSubmats[i];
            uint idx = jacobianIndices[i];
            FmSMatrix3 DAinv_idx = DAinv[idx];
            FmSVector3 JTlambda_idx = JTlambda[idx];

            rowResult -= mul(submat, mul(DAinv_idx, JTlambda_idx));
        }

        // rowResult is now c - B_lambda3 * lambda
        // Finish GS iteration for row
        rowResult = diagInv * rowResult + lambda3Orig;

        // Under-relaxation for stability benefit
        rowResult = (1.0f - omega) * lambda3Orig + omega * rowResult;

        rowResult += totLambda3;

        float lambda = rowResult.x;
        float gamma1 = rowResult.y;
        float gamma2 = rowResult.z;

        if (type == FM_SOLVER_CONSTRAINT_TYPE_3D_NORMAL2DFRICTION)
        {
            // Normal and friction force projections.
            lambda = FmMaxFloat(lambda, 0.0f);

            float maxFriction = frictionCoeff * lambda;

#if FM_PROJECT_FRICTION_TO_CIRCLE
            // Projection to circle
            float length = sqrtf(gamma1*gamma1 + gamma2*gamma2);
            if (length > maxFriction)
            {
                float scale = maxFriction / length;
                gamma1 *= scale;
                gamma2 *= scale;
            }
#else
            // Projection to box
            gamma1 = FmMinFloat(gamma1, maxFriction);
            gamma1 = FmMaxFloat(gamma1, -maxFriction);
            gamma2 = FmMinFloat(gamma2, maxFriction);
            gamma2 = FmMaxFloat(gamma2, -maxFriction);
#endif
        }
        else if (type == FM_SOLVER_CONSTRAINT_TYPE_3D_JOINT1DFRICTION)
        {
            float maxFriction = frictionCoeff;
            lambda = FmMinFloat(lambda,  maxFriction);
            lambda = FmMaxFloat(lambda, -maxFriction);
        }
        else
        {
            if ((flags & FM_SOLVER_CONSTRAINT_FLAG_NONNEG0) != 0)
            {
                lambda = FmMaxFloat(lambda, 0.0f);
            }
            if ((flags & FM_SOLVER_CONSTRAINT_FLAG_NONNEG1) != 0)
            {
                gamma1 = FmMaxFloat(gamma1, 0.0f);
            }
            if ((flags & FM_SOLVER_CONSTRAINT_FLAG_NONNEG2) != 0)
            {
                gamma2 = FmMaxFloat(gamma2, 0.0f);
            }
        }

        FmSVector3 lambda3Update = FmInitSVector3(lambda, gamma1, gamma2) - totLambda3;

        // Update J^T * lambda given change in lambda values
        FmSVector3 deltaLambda = lambda3Update - lambda3Orig;

#if FM_CONSTRAINT_SOLVER_CONVERGENCE_TEST
        norms->Update(deltaLambda, lambda3Update);
#else
        (void)norms;
#endif

        const float epsilon = FLT_EPSILON*FLT_EPSILON;
        if (lengthSqr(deltaLambda) > epsilon)
        {
            for (uint i = 0; i < rowSize; i++)
            {
                FmSMatrix3 rowSubmat = jacobianSubmats[i];
                uint outputRowIdx = jacobianIndices[i];
#if 1
                JTlambda[outputRowIdx] += FmTransposeMul(rowSubmat, deltaLambda);
#else
                FmSMatrix3 JTSubmat = transpose(rowSubmat);

                JTlambda[outputRowIdx] += mul(JTSubmat, deltaLambda);
#endif
            }

            // update lambda
            lambda3[rowIdx] = lambda3Update;
        }
    }