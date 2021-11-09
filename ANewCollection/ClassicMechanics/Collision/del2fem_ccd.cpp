// compute impulse force
void SelfCollisionImpulse_CCD(
    std::vector<double>& aUVWm, // (in,out)velocity
    //
    double delta,
    [[maybe_unused]] double stiffness,
    double dt,
    double mass,
    const std::vector<double>& aXYZ,
    [[maybe_unused]] const std::vector<unsigned int>& aTri,
    const std::vector<dfm2::CContactElement>& aContactElem)
{
  for(const auto & ce : aContactElem){
    const int ino0 = ce.ino0;
    const int ino1 = ce.ino1;
    const int ino2 = ce.ino2;
    const int ino3 = ce.ino3;
    dfm2::CVec3d p0( aXYZ[ ino0*3+0], aXYZ[ ino0*3+1], aXYZ[ ino0*3+2] );
    dfm2::CVec3d p1( aXYZ[ ino1*3+0], aXYZ[ ino1*3+1], aXYZ[ ino1*3+2] );
    dfm2::CVec3d p2( aXYZ[ ino2*3+0], aXYZ[ ino2*3+1], aXYZ[ ino2*3+2] );
    dfm2::CVec3d p3( aXYZ[ ino3*3+0], aXYZ[ ino3*3+1], aXYZ[ ino3*3+2] );
    dfm2::CVec3d v0( aUVWm[ino0*3+0], aUVWm[ino0*3+1], aUVWm[ino0*3+2] );
    dfm2::CVec3d v1( aUVWm[ino1*3+0], aUVWm[ino1*3+1], aUVWm[ino1*3+2] );
    dfm2::CVec3d v2( aUVWm[ino2*3+0], aUVWm[ino2*3+1], aUVWm[ino2*3+2] );
    dfm2::CVec3d v3( aUVWm[ino3*3+0], aUVWm[ino3*3+1], aUVWm[ino3*3+2] );
    double t;
    {
      bool res = FindCoplanerInterp(
          t,
          p0,p1,p2,p3, p0+v0,p1+v1,p2+v2,p3+v3);
      if( !res ) continue;
      assert( t >= 0 && t <= 1 );
    }
    if( ce.is_fv ){ // face-vtx
      double w0,w1;
      {        
        dfm2::CVec3d p0m = p0 + t*v0;
        dfm2::CVec3d p1m = p1 + t*v1;
        dfm2::CVec3d p2m = p2 + t*v2;
        dfm2::CVec3d p3m = p3 + t*v3;
        double dist = DistanceFaceVertex(p0m, p1m, p2m, p3m, w0,w1);
        if( w0 < 0 || w0 > 1 ) continue;
        if( w1 < 0 || w1 > 1 ) continue;
        if( dist > delta ) continue;
      }
      double w2 = 1.0 - w0 - w1;
      dfm2::CVec3d pc = w0*p0 + w1*p1 + w2*p2;
      dfm2::CVec3d norm = p3 - pc; norm.normalize();
      double rel_v = Dot(v3-w0*v0-w1*v1-w2*v2,norm); // relative velocity (positive if separating)
      if( rel_v > 0.1*delta/dt ) continue; // separating
      double imp = mass*(0.1*delta/dt-rel_v);
      double imp_mod = 2*imp/(1.0+w0*w0+w1*w1+w2*w2);
      imp_mod /= mass;
      imp_mod *= 0.1;
      aUVWm[ino0*3+0] += -norm.x*imp_mod*w0;
      aUVWm[ino0*3+1] += -norm.y*imp_mod*w0;
      aUVWm[ino0*3+2] += -norm.z*imp_mod*w0;
      aUVWm[ino1*3+0] += -norm.x*imp_mod*w1;
      aUVWm[ino1*3+1] += -norm.y*imp_mod*w1;
      aUVWm[ino1*3+2] += -norm.z*imp_mod*w1;
      aUVWm[ino2*3+0] += -norm.x*imp_mod*w2;
      aUVWm[ino2*3+1] += -norm.y*imp_mod*w2;
      aUVWm[ino2*3+2] += -norm.z*imp_mod*w2;
      aUVWm[ino3*3+0] += +norm.x*imp_mod;
      aUVWm[ino3*3+1] += +norm.y*imp_mod;
      aUVWm[ino3*3+2] += +norm.z*imp_mod;
    }
    else{ // edge-edge
      double w01,w23;
      {
        dfm2::CVec3d p0m = p0 + t*v0;
        dfm2::CVec3d p1m = p1 + t*v1;
        dfm2::CVec3d p2m = p2 + t*v2;
        dfm2::CVec3d p3m = p3 + t*v3;
        double dist = DistanceEdgeEdge(p0m, p1m, p2m, p3m, w01,w23);
        if( w01 < 0 || w01 > 1 ) continue;
        if( w23 < 0 || w23 > 1 ) continue;
        if( dist > delta ) continue;
      }      
      dfm2::CVec3d c01 = (1-w01)*p0 + w01*p1;
      dfm2::CVec3d c23 = (1-w23)*p2 + w23*p3;
      dfm2::CVec3d norm = (c23-c01); norm.normalize();
      double rel_v = Dot((1-w23)*v2+w23*v3-(1-w01)*v0-w01*v1,norm);
      if( rel_v > 0.1*delta/dt ) continue; // separating
      double imp = mass*(0.1*delta/dt-rel_v); // reasonable
      double imp_mod = 2*imp/( w01*w01+(1-w01)*(1-w01) + w23*w23+(1-w23)*(1-w23) );
      imp_mod /= mass;
      imp_mod *= 0.1;
      aUVWm[ino0*3+0] += -norm.x*imp_mod*(1-w01);
      aUVWm[ino0*3+1] += -norm.y*imp_mod*(1-w01);
      aUVWm[ino0*3+2] += -norm.z*imp_mod*(1-w01);
      aUVWm[ino1*3+0] += -norm.x*imp_mod*w01;
      aUVWm[ino1*3+1] += -norm.y*imp_mod*w01;
      aUVWm[ino1*3+2] += -norm.z*imp_mod*w01;
      aUVWm[ino2*3+0] += +norm.x*imp_mod*(1-w23);
      aUVWm[ino2*3+1] += +norm.y*imp_mod*(1-w23);
      aUVWm[ino2*3+2] += +norm.z*imp_mod*(1-w23);
      aUVWm[ino3*3+0] += +norm.x*imp_mod*w23;
      aUVWm[ino3*3+1] += +norm.y*imp_mod*w23;
      aUVWm[ino3*3+2] += +norm.z*imp_mod*w23;
    }
  }
}
