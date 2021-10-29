/**
 https://github.com/pielet/Hair-DER
 * \brief Rotates a vector
 *
 * \param [in] v vector to rotate
 * \param [in] z normalized vector on the rotation axis
 * \param [in] theta rotation angle
 *
 */
template<typename scalarT>
inline void rotateAxisAngle( typename Eigen::Matrix<scalarT, 3, 1> & v,
        const typename Eigen::Matrix<scalarT, 3, 1> & z, const scalarT theta )
{
    assert( isApproxUnit( z ) );

    if( theta == 0 )
        return;

    const scalarT c = cos( theta );
    const scalarT s = sin( theta );

    v = c * v + s * z.cross( v ) + z.dot( v ) * ( 1.0 - c ) * z;
}