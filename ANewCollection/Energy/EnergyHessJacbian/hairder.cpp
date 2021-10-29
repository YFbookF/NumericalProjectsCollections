//https://github.com/pielet/Hair-DER
// Jacobian of the Force <==>  - Hessian of the Energy
    static void accumulate( TripletXs& hessianOfEnergy, const StrandForce& strand )
    {
        typename ForceT::LocalJacobianType localJ;
        for( IndexType vtx = ForceT::s_first; vtx < strand.getNumVertices() - ForceT::s_last; ++vtx )
        {
            ForceT::computeLocal( localJ, strand, vtx );

            if( localJ.rows() > 6 ){ // (Bending & Twisting)
                for( IndexType r = 0; r < localJ.rows(); ++r )
                {
                    for( IndexType c = 0; c < localJ.cols(); ++c )
                    {
                        if( isSmall( localJ(r,c) )  ) continue;
                        hessianOfEnergy.push_back( Triplets( (vtx - 1) * 4 + r, (vtx - 1) * 4 + c, localJ(r,c) ) );
                    }
                }
            }
            else{ // Stretch
                int trCount = 0;
                for( IndexType r = 0; r < localJ.rows(); ++r ){
                    if( r == 3 ){ // skip twist dof
                        ++trCount;
                    }                    
                    int tcCount = 0;
                    for( IndexType c = 0; c < localJ.cols(); ++c ){
                        if( c == 3 ){ // skip twist dof
                            ++tcCount;
                        }                        
                        if( isSmall( localJ(r,c) )  ) continue;
                        hessianOfEnergy.push_back( Triplets( vtx * 4 + r + trCount, vtx * 4 + c + tcCount, localJ(r,c) ) );
                    }
                }
            }
        }

template<typename ViscousT>
void BendingForce<ViscousT>::computeLocal(Eigen::Matrix<scalar, 11, 11>& localJ,
        const StrandForce& strand, const IndexType vtx )
{
    localJ = strand.m_strandState->m_bendingProducts[vtx];

    if( strand.m_requiresExactForceJacobian )
    {
        const Mat2& bendingMatrixBase = strand.m_strandParams->bendingMatrixBase();
        const Vec2& kappaBar = ViscousT::kappaBar( strand, vtx );
        const Vec2& kappa = strand.m_strandState->m_kappas[vtx];
        const std::pair<LocalJacobianType, LocalJacobianType>& hessKappa = strand.m_strandState->m_hessKappas[vtx];
        const Vec2& temp = bendingMatrixBase * ( kappa - kappaBar );

        localJ += temp( 0 ) * hessKappa.first + temp( 1 ) * hessKappa.second;
    }

    const scalar ilen = strand.m_invVoronoiLengths[vtx];
    localJ *= -ilen * ViscousT::bendingCoefficient( strand, vtx );
}

template<typename ViscousT>
void StretchingForce<ViscousT>::computeLocal(Eigen::Matrix<scalar, 6, 6>& localJ,
        const StrandForce& strand, const IndexType vtx )
{
    const scalar ks = ViscousT::ks( strand, vtx );
    const scalar restLength = ViscousT::ellBar( strand, vtx );

    const scalar length = strand.m_strandState->m_lengths[vtx];
    const Vec3& edge = strand.m_strandState->m_tangents[ vtx ];

    bool useApprox = !strand.m_requiresExactForceJacobian && length < restLength;

    Mat3 M ;
    if( useApprox ){
        M = ks / restLength * ( edge * edge.transpose() );
    }
    else{
        M = ks
                * ( ( 1.0 / restLength - 1.0 / length ) * Mat3::Identity()
                    + 1.0 / length * ( edge * edge.transpose() ) );
    }

    localJ.block<3, 3>( 0, 0 ) = localJ.block<3, 3>( 3, 3 ) = -M;
    localJ.block<3, 3>( 0, 3 ) = localJ.block<3, 3>( 3, 0 ) = M;
}

template<typename ViscousT>
void TwistingForce<ViscousT>::computeLocal( Eigen::Matrix<scalar, 11, 11>& localJ,
        const StrandForce& strand, const IndexType vtx )
{
    const scalar kt = ViscousT::kt( strand, vtx );
    const scalar ilen = strand.m_invVoronoiLengths[vtx];
    const Mat11& gradTwistSquared = strand.m_strandState->m_gradTwistsSquared[vtx];

    localJ = -kt * ilen * gradTwistSquared;
    if( strand.m_requiresExactForceJacobian )
    {
        const scalar undeformedTwist = ViscousT::thetaBar( strand, vtx );
        const scalar twist = strand.m_strandState->m_twists[vtx];
        const Mat11& hessTwist = strand.m_strandState->m_hessTwists[vtx];
        localJ += -kt * ilen * ( twist - undeformedTwist ) * hessTwist;
    }
}