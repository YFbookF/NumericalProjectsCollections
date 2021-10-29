
https://github.com/benjones/strainLimitingForClusteredShapeMatching/blob/master/twistingPlane.cpp

void TwistingPlane::twistParticle(Particle& particle, double timeElapsed) const {
   if (timeElapsed > lifetime) return;
   //std::cout<<timeElapsed<<" "<<lifetime<<std::endl;
   if(outside(particle)){
      //will rotate around support point
      const double sos = normal.dot(normal);
      const Eigen::Vector3d supportPoint{normal.x()*offset/sos,
         normal.y()*offset/sos,
         normal.z()*offset/sos};

      //project current position into the plane, compute dir vector from
      //supportPoint to particle position
      Eigen::Vector3d posInPlane = particle.position - supportPoint; 
      posInPlane = posInPlane - particle.position.dot(normal) * normal;

      //compute tangent space, to get velocity components
      Eigen::Vector3d tangent1, tangent2;

      tangent1 = normal.cross(Eigen::Vector3d{1,0,0});
      if(tangent1.isZero(1e-3)){
         tangent1 = normal.cross(Eigen::Vector3d{0,0,1});
         if(tangent1.isZero(1e-3)){
            tangent1 = normal.cross(Eigen::Vector3d{0,1,0});
         }
      }
      tangent1.normalize();

      tangent2 = normal.cross(tangent1);
      tangent2.normalize(); //probably not necessary

      //dot posInPlane with current tangent space to compute rotation velocity
      double t2_rot = angularVelocity * posInPlane.dot(tangent1);
      double t1_rot = angularVelocity * posInPlane.dot(tangent2);
      
      particle.velocity = -t1_rot * tangent1 + t2_rot * tangent2;
   }
}
