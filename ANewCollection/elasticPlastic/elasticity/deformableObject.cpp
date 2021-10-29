#include "DeformableObject.hpp"
#include <fstream>
#include <random>
#include <iostream>
#include <json/json.h>
//https://github.com/benjones/adaptiveDeformables/blob/master/DeformableObject.cpp
#include "AccelerationGrid.hpp"
#include "quatSVD.hpp"

#include "KDTree.hpp"
#include "SampleElimination.hpp"

DeformableObject::DeformableObject(const Json::Value& jv){


  std::string filename = jv["particleFile"].asString();
  std::ifstream ins(filename);


  lambda = getOrThrow<double>(jv, "lambda");
  mu = getOrThrow<double>(jv, "mu");
  dampingFactor = getOrThrow<double>(jv,"dampingFactor");
  density = getOrThrow<double>(jv, "density");

  
  //hierarchyLevels = getOrThrow<int>(jv, "hierarchyLevels");
  //parentsPerParticle = getOrThrow<int>(jv, "parentsPerParticle");
  neighborsPerParticle = getOrThrow<int>(jv, "neighborsPerParticle");

  
  //scalingVarianceThreshold = getOrThrow<double>(jv, "scalingVarianceThreshold");
  //angularVarianceThreshold = getOrThrow<double>(jv, "angularVarianceThreshold");


  particleSize = getOrThrow<double>(jv, "particleSize");

  rbfDelta = getOrThrow<double>(jv, "rbfDelta");
  
  Vec3 translation = Vec3::Zero();
  if(jv.isMember("translation")){
	translation.x() = jv["translation"][0].asDouble();
	translation.y() = jv["translation"][1].asDouble();
	translation.z() = jv["translation"][2].asDouble();
  }
  
  
  size_t numParticles;
  ins.read(reinterpret_cast<char*>(&numParticles), sizeof(numParticles));

  particles.resize(numParticles);
  for(size_t i = 0; i < numParticles; ++i){

	ins.read(reinterpret_cast<char*>(particles[i].position.data()), 3*sizeof(double));

	particles[i].position += translation;
	particles[i].velocity = Vec3::Zero();
  }

  computeNeighbors();
  computeBasisAndVolume();

  //computeHierarchy();

  renderInfos.resize(particles.size());
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(0.0, 360);
  
  for(auto& ri : renderInfos){
	ri.color = hsv2rgb(dis(gen), 0.75, 0.8);
	ri.size = particleSize;
  }

  /*
  auto pSize = particleSize;
  for(const auto& level : hierarchy){
	for(auto i : level){
	  renderInfos[i].size = pSize;
	}
	pSize /= 2;
	}*/
  
}


std::vector<int> kMeans(const DeformableObject& d,  const std::vector<int>& indices, int n){


  std::random_device rd;
  std::mt19937 gen(rd());

  std::vector<int> centers = indices;
  std::shuffle(centers.begin(), centers.end(), gen);
  centers.resize(n);

  std::sort(centers.begin(), centers.end());

  /*  for(auto c : centers){
	std::cout << c << std::endl;
	}*/
  //number of points, and sum of those points positions
  std::vector<std::pair<int, Vec3> > centerInfo;
  

  for(int iter = 0; iter < 5; ++iter){
	auto oldCenters = centers;
	centerInfo.assign(centers.size(), std::pair<int, Vec3>(0, Vec3::Zero()));
	
	//assign each particle to nearest center
	for(const auto& p : d.particles){

	  int nc = std::distance(centers.begin(),
		  std::min_element(centers.begin(), centers.end(),
			  [&d, &p](int a, int b){
				return (d.particles[a].position - p.position).squaredNorm() <
				(d.particles[b].position - p.position).squaredNorm();
			  }));

	  ++centerInfo[nc].first;
	  centerInfo[nc].second += p.position;
	
	}


	//find particle nearest this sum

	for(auto i = 0; i < centers.size(); ++i){

	  Vec3 com = centerInfo[i].second/centerInfo[i].first;
	  //std::cout << "com: " << com << std::endl;
	  centers[i] = *std::min_element(indices.begin(), indices.end(),
		  [&d, &com](int a, int b){
			return (com - d.particles[a].position).squaredNorm() <
			(com - d.particles[b].position).squaredNorm();
		  });
	  
	}
	if(centers == oldCenters){
	  //std::cout << "converged in " << iter << " iters" << std::endl;
	  break;
	}
  }
  std::sort(centers.begin(), centers.end());
  return centers;
}
 

void DeformableObject::computeNeighbors(){

  std::vector<Vec3> positions(particles.size());
  for(auto i = 0; i < particles.size(); ++i){ positions[i] = particles[i].position;}
  
  KDTree<Vec3, 3> kdTree(positions);
  
  for(int i = 0; i < particles.size(); ++i){
	auto& p = particles[i];
	

	auto neighbors = kdTree.KNN(positions, positions[i], neighborsPerParticle + 1);

	
	auto it = std::find(neighbors.begin(), neighbors.end(), i);
	assert(it != neighbors.end());
	neighbors.erase(it);

	
	
	assert(neighbors.size() == neighborsPerParticle);
	p.neighbors.resize(neighbors.size());
	//set the radius to be the distance to the furthest neighbor
	//since KNN stores stuff in a max heap, front is the farthest
	p.kernelRadius = 2*(particles[neighbors.front()].position - p.position).norm();
	double wSum = 0;
	for(auto j = 0; j < neighbors.size(); ++j){
	  p.neighbors[j].index = neighbors[j];
	  p.neighbors[j].uij = particles[neighbors[j]].position - p.position;
	  p.neighbors[j].wij = poly6(p.neighbors[j].uij.squaredNorm(), p.kernelRadius);
	  wSum += p.neighbors[j].wij;

	}
	for(auto & n : p.neighbors){
	  n.wij /= wSum;
	}
	
  }
  RBFInit();
}

void DeformableObject::applyGravity(double dt){
  for(auto& p : particles){
	p.velocity -= dt*Vec3(0, 9.81, 0);
  }
}


Mat3 DeformableObject::computeDeformationGradient(int pIndex) const{
  auto& p = particles[pIndex];

  Mat3 ux = Mat3::Zero();
  for(const auto& n : p.neighbors){
	ux += n.wij*(particles[n.index].position - p.position)*n.uij.transpose();
  }
  
  return ux*p.Ainv;
  
}

void DeformableObject::applyElasticForces(double dt){

  forces.assign(particles.size(), Vec3::Zero());
  
  for(int i = 0; i < particles.size(); ++i){
	auto& p = particles[i];

	//auto F = computeDeformationGradient(i);
	auto F = computeDeformationGradientRBF(i);
	auto svd = QuatSVD::svd(F);

	//Eigen::JacobiSVD<Mat3> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Vec3 FHat = svd.S;//singularValues();
	Vec3 diagStress = lambda*(FHat(0) + FHat(1) + FHat(2) - 3)*Vec3(1,1,1) +
	  2*mu*(FHat - Vec3(1,1,1));
	
	Mat3 rotStress = svd.U.toRotationMatrix()*diagStress.asDiagonal()*(svd.V.toRotationMatrix().transpose());

	Mat3 forceCommon = dt*p.volume*rotStress*p.Ainv;
	
	for(const auto& n : p.neighbors){
	  Vec3 fi = forceCommon*(n.wij*n.uij);
	  forces[i] += fi;
	  forces[n.index] -= fi;
	}
	
  }

  //now modify velocities
  for(int i = 0; i < particles.size(); ++i){
	particles[i].velocity += forces[i]/(particles[i].volume*density);
  }

  assertFinite();
}




/*void DeformableObject::applyElasticForcesAdaptive(double dt){
  forces.assign(particles.size(), Vec3::Zero());
  for(auto i = 0; i < hierarchy[0].size(); ++i){
	auto index = hierarchy[0][i];
	auto& p = particles[index];
	Mat3 F = computeDeformationGradient(index);
	p.svd = QuatSVD::svd(F);
	
  }
  //not interpolate for the rest
  int interpedCount = 0, notInterpedCount = 0, badAngles = 0, badScales = 0;
  std::vector<QuatSVD::EigenSVD<double> > svds;
  std::vector<double> weights;
  
  for(int level = 1; level < hierarchy.size(); ++level){
	for(int i = 0; i < hierarchy[level].size(); ++i){
	  auto index = hierarchy[level][i];
	  //	  std::cout << "index: " << index << std::endl;
	  auto& p = particles[index];
	  svds.resize(p.parents.size());
	  weights.resize(p.parents.size());
	  Vec3 sSquared = Vec3::Zero();
	  double weightSum = 0;
	  
	  //	  std::cout << "SVDs" << std::endl;
	  for(auto j = 0; j < p.parents.size(); ++j){
		svds[j] = particles[p.parents[j].index].svd;
		weights[j] = p.parents[j].wij;
		weightSum += weights[j];
		sSquared.x() += weights[j]*svds[j].S.x()*svds[j].S.x();
		sSquared.y() += weights[j]*svds[j].S.y()*svds[j].S.y();
		sSquared.z() += weights[j]*svds[j].S.z()*svds[j].S.z();
		//		std::cout << svds[j].U << std::endl
		//				  << svds[j].S << std::endl
		//				  << svds[j].V << std::endl;
		  
		
	  }
	  sSquared /= (weightSum);
	  auto interpedPolar = interpolateSVDsPolar(svds, weights);
	  
	  //	  Mat3 interpedF = interpedSVD.U.toRotationMatrix()*interpedSVD.S.asDiagonal()*
	  //		interpedSVD.V.toRotationMatrix().transpose();
	  
		//	  double error = (interpedF - computeDeformationGradient(index)).norm();
		
	  //std::cout << "sSquared: " << sSquared << " interped S: " << interpedSVD.S << std::endl;
	  
	  double normalizedVariance = sSquared.sum()/(interpedPolar.S.squaredNorm()) - 1;
	  normalizedVariance /= parentsPerParticle;
	  assert(normalizedVariance > -1e-5); //variance should be positive...
	  //std::cout << "nv: " <<  normalizedVariance << std::endl;
	  double angleError = 0;
	  //Quat totalR = interpedSVD.U*interpedSVD.V.conjugate();
	  
	  for(const auto& svd: svds){
		angleError += square(interpedPolar.R.angularDistance(
				svd.U*svd.V.conjugate()));
	  }
	  //std::cout << "angle error: " << angleError << std::endl;
	  
	  auto badScale = normalizedVariance > scalingVarianceThreshold;
	  auto badAngle = angleError > angularVarianceThreshold;
	  if(badScale) ++badScales;
	  if(badAngle) ++badAngles;
	  
	  if(badScale || badAngle){
		p.svd = QuatSVD::svd(computeDeformationGradient(index));
		notInterpedCount++;
	  } else {
		p.svd = QuatSVD::svdFromPolar(interpedPolar.R, interpedPolar.S);
		interpedCount++;
	  }
	  
	}
  }
  std::cout << "interped: " << interpedCount << " not: " << notInterpedCount << " bad scales: " << badScales << " bad angles: " << badAngles << std::endl;  
  //compute forces based on the deformation
  for(auto i = 0; i < particles.size(); ++i){
	auto& p = particles[i];
	
	Vec3 FHat = p.svd.S;
	Vec3 diagStress = lambda*(FHat(0) + FHat(1) + FHat(2) - 3)*Vec3(1,1,1) +
	  2*mu*(FHat - Vec3(1,1,1));
	Mat3 rotStress = p.svd.U.toRotationMatrix()*
	  diagStress.asDiagonal()*
	  p.svd.V.toRotationMatrix().transpose();
	Mat3 forceCommon = p.volume*rotStress*p.Ainv;
	
	for(const auto& n : p.neighbors){
	  Vec3 fi = forceCommon*(n.wij*n.uij);
	  forces[i] += fi;
	  forces[n.index] -= fi;
	}
	
  }
  
  
  //now modify velocities
  for(int i = 0; i < particles.size(); ++i){
	particles[i].velocity += forces[i]/(particles[i].volume*density);
  }
 
  }*/


void DeformableObject::updatePositions(double dt){
  for(auto& p : particles){
	p.position += dt*p.velocity;
  }
  
}


void DeformableObject::computeBasisAndVolume(){
  
  for(auto& p : particles){
	Mat3 A = Mat3::Zero();
	double wSum = 0;

	for(const auto& n : p.neighbors){
	  A += n.wij* n.uij* n.uij.transpose();
	  //	  wSum += n.wij;
	}
	p.Ainv = A.inverse();
	assert(p.Ainv.allFinite());
	p.volume = std::sqrt(A.determinant()); ///std::pow(wSum, 3));
  }
  
}


void DeformableObject::bounceOffGround(){
  for(auto& p : particles){
	if(p.position.y() < 0){
	  p.position.y() = 0;
	  p.velocity.y() = 0;
	}
  }
}

void DeformableObject::dump(const std::string& filename) const{
  std::ofstream outs(filename, std::ios::binary);
  if(!outs){
	throw std::runtime_error("couldn't open output file");
  }
  
  size_t np = particles.size();

  outs.write(reinterpret_cast<const char*>(&np), sizeof(np));

  for(const auto& p : particles){
	outs.write(reinterpret_cast<const char*>(p.position.data()), 3*sizeof(p.position.x()));
  }
}

/*void DeformableObject::writeHierarchy(const std::string&filename) const{
  std::ofstream outs(filename);
  outs << hierarchy.size() << std::endl;
  for(const auto& level : hierarchy){
	outs << level.size() << std::endl;
	for(auto i : level){
	  outs << i << ' ';
	}
	outs << std::endl;
  }
  
  }*/


void DeformableObject::springDamping(double dt){
  std::cout << "damping" << std::endl;
  //actually damping impulses
  dampedVelocities.assign(particles.size(), Vec3::Zero());
  for(auto i = 0; i < particles.size(); ++i){
	const auto& p = particles[i];

	for(const auto& n : p.neighbors){
	  const auto& np = particles[n.index];
	  
	  Vec3 xij = np.position - p.position;
	  if(xij.squaredNorm() > 1e-3){
		xij.normalize();
	  } else {
		xij = Vec3::Zero();
	  }
	  
	  Vec3 vij = np.velocity - p.velocity;

	  //project onto xij if reasonable.
	  Vec3 dampedSpring= 0.5*dt*n.wij*dampingFactor*xij.dot(vij)*xij;
	  
	  
	  dampedVelocities[i] += dampedSpring;
	  dampedVelocities[n.index] -= dampedSpring;
	  if(!dampedSpring.allFinite()){
		std::cout << "nan!!! bad!!! " << i << std::endl;
		std::cout << n.index << std::endl;
		exit(1);
	  }
	}
  }

  for(auto i = 0; i < particles.size(); ++i){
	particles[i].velocity += dampedVelocities[i];
  }
  

}

void DeformableObject::damp(double dt){
  dampedVelocities.assign(particles.size(), Vec3::Zero());
  for(auto i = 0; i < particles.size(); ++i){
	auto& p = particles[i];
	Vec3 vel = Vec3::Zero();
	//double wSum = 0;
	for(const auto& n : p.neighbors){
	  vel += n.wij*particles[n.index].velocity;
	  //wSum += n.wij;
	}
	//vel /= wSum;
	dampedVelocities[i] = vel; 
  }

  for(auto i = 0; i < particles.size(); ++i){
	Vec3 impulse = 0.5*dt*dampingFactor*(dampedVelocities[i] - particles[i].velocity);
	for(const auto& n : particles[i].neighbors){
	  particles[i].velocity += n.wij*impulse;
	  particles[n.index].velocity -= n.wij*impulse;

	}
  }
  /*
  Vec3 dp = Vec3::Zero();
  double ns = 0;
  for(const auto& v : dampedVelocities){
	dp += v;
	ns += v.squaredNorm();
  }
  std::cout << "squared norm: " << ns << " dp : " << dp << std::endl;
  */
}




/*void DeformableObject::computeHierarchy(){
  hierarchy.resize(hierarchyLevels);
  int n = particles.size();
  std::vector<int> indices(particles.size());
  std::iota(indices.begin(), indices.end(), 0);
  //todo: do better
  std::vector<Vec3> positions(particles.size());
  std::transform(particles.begin(), particles.end(), positions.begin(),
	  [](const Particle& p){ return p.position;});
  
  //
  //hierarchy.back() = kMeans(*this, indices, n);
  hierarchy.back() = indices;
  
  for(int level = hierarchyLevels - 2; level >= 0; --level){
	n = n/4;
	std::cout << "level " << level << " has " << n << " particles " << std::endl;
	hierarchy[level] = kMeans(*this, hierarchy[level +1], n);
	//hierarchy[level] = sampleEliminate(positions, hierarchy[level + 1], n);
	//remove these from the next highest level (they shoudl already be removed from the lower ones
	for(int i  = 0; i < hierarchy[level].size(); ++i){
	  std::remove(hierarchy[level +1].begin(),
		  hierarchy[level +1].begin() + hierarchy[level +1].size() - i,
		  hierarchy[level][i]);
	}
	//erase the points at this level
	hierarchy[level+1].erase(
		hierarchy[level +1].begin() + (hierarchy[level + 1].size() - hierarchy[level].size()),
		hierarchy[level +1].end());
	
  }
  for(const auto& level: hierarchy){
	std::cout << "level" << std::endl;
	for(auto i : level){
	  std::cout << i << ' ';
	}
	std::cout << std::endl;
  }
  //compute parents, starting from level0, which has no parents
  for(auto levelIndex = 1; levelIndex < hierarchy.size(); ++levelIndex){
	const auto& level = hierarchy[levelIndex];
	//todo compute fastGrid for the parent level
	
	for(int i : level){
	  assert(particles[i].parents.empty());
	  //find the closest parents
	  auto myPosition = particles[i].position;
	  auto myRadius = particles[i].kernelRadius;
	  
	  auto start = hierarchy[levelIndex -1].begin();
	  auto mid = parentsPerParticle <= hierarchy[levelIndex -1].size() ?
		hierarchy[levelIndex -1].begin() + parentsPerParticle :
		hierarchy[levelIndex -1].end();
	  
	  std::partial_sort(start, mid,
		  hierarchy[levelIndex -1].end(),
		  [this,&myPosition](int a, int b){
			return (particles[a].position - myPosition).squaredNorm() <
			  (particles[b].position - myPosition).squaredNorm();
		  });
	  particles[i].parents.resize(mid - start);
	  std::transform(start, mid, particles[i].parents.begin(),
		  [this, &myPosition, &myRadius](int a){
			return Parent{a,
				1.0/(myPosition - particles[a].position).squaredNorm()};
					
		  });
	  
	}
	
  }
}*/


void DeformableObject::dumpWithColor(const std::string& filename) const{

    std::ofstream outs(filename, std::ios::binary);
  if(!outs){
	throw std::runtime_error("couldn't open output file");
  }
  
  size_t np = particles.size();

  outs.write(reinterpret_cast<const char*>(&np), sizeof(np));

  for(auto i = 0; i < particles.size(); ++i){
	const auto& p = particles[i];
	outs.write(reinterpret_cast<const char*>(p.position.data()), 3*sizeof(p.position.x()));
	outs.write(reinterpret_cast<const char*>(renderInfos.data() + i), 4*sizeof(double));
  }

  
  
}


void DeformableObject::applyElasticForcesNoOvershoot(double dt){

  forces.assign(particles.size(), Vec3::Zero());


  int negAlphas = 0;
  int bigAlphas = 0;
  int fine= 0;

  
  for(int i = 0; i < particles.size(); ++i){
	auto& p = particles[i];

	auto F = computeDeformationGradient(i);
	auto svd = QuatSVD::svd(F);

	Vec3 FHat = svd.S;
	Vec3 diagStress = lambda*(FHat(0) + FHat(1) + FHat(2) - 3)*Vec3(1,1,1) +
	  2*mu*(FHat - Vec3(1,1,1));
	
	Mat3 rotStress = svd.U.toRotationMatrix()*diagStress.asDiagonal()*(svd.V.toRotationMatrix().transpose());

	Mat3 forceCommon = dt*p.volume*rotStress*p.Ainv;

	//From the polar decomp
	Quat R = svd.U*svd.V.conjugate();
	
	for(const auto& n : p.neighbors){

	  Vec3 goalPosition = R*n.uij;
	  Vec3 err = goalPosition - (particles[n.index].position - p.position);
	  
	  
	  Vec3 impulse  = forceCommon*(n.wij*n.uij);
	  auto sqNorm = impulse.squaredNorm()*dt*dt;

	  
	  //applying the impulse to i, and its negative to n.index
	  double alpha = sqNorm > 1e-6 ? dt*err.dot(-impulse)/sqNorm : 1;
	  if(alpha < 0){
		alpha = 0;
		++negAlphas;
	  } else if(alpha > 1){
		alpha = 1;
		++bigAlphas;
	  } else {
		++fine;
	  }
	  
	  forces[i] += alpha*impulse;
	  forces[n.index] -= alpha*impulse;
	}
	
  }

  //now modify velocities
  for(int i = 0; i < particles.size(); ++i){
	particles[i].velocity += forces[i]/(particles[i].volume*density);
  }

  std::cout << "fine " << fine << " negAlphas: " << negAlphas << " hugeAlphas: " << bigAlphas << std::endl;
  
  assertFinite();
}


void DeformableObject::RBFInit(){

  const int M = 4; //1 + x + y + z
  const int m = 3;
  
  std::vector<Vec3> positions(particles.size());
  for(auto i = 0; i < particles.size(); ++i){ positions[i] = particles[i].position;}
  
  KDTree<Vec3, 3> kdTree(positions);  

  std::vector<bool> indicesInBall(positions.size(), false);
  for(int i = 0; i < positions.size(); ++i){
	if(!indicesInBall[i]){
	  auto neighbors = kdTree.KNN(positions, positions[i], neighborsPerParticle);
	  auto nRadius = (positions[neighbors[0]] - positions[i]).norm();
	  std::vector<int> ballMembers;
	  for(auto n : neighbors){
		if( (positions[n] - positions[i]).norm() <= (1 - rbfDelta)*nRadius){
		  ballMembers.push_back(n);
		  indicesInBall[n] = true;
		}
	  }
	  //Assemble matrices

	  //A is symmetric, so only store the lower triangular part
	  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(neighbors.size() + 4, neighbors.size() + 4);
	  Eigen::MatrixXd bx = Eigen::MatrixXd::Zero(neighbors.size() + 4, ballMembers.size());
	  Eigen::MatrixXd by = Eigen::MatrixXd::Zero(neighbors.size() + 4, ballMembers.size());
	  Eigen::MatrixXd bz = Eigen::MatrixXd::Zero(neighbors.size() + 4, ballMembers.size());

	  for(int r = 0; r < neighbors.size(); ++r){
		for(int c = 0; c < r; ++c){
		  double dist = std::pow((positions[neighbors[r]] - positions[neighbors[c]]).norm(), m);
		  A(r,c) = dist;
		}

		//fill in the monomials
		A(neighbors.size(), r) = 1;
		A(neighbors.size() +1, r) = positions[neighbors[r]].x();
		A(neighbors.size() +2, r) = positions[neighbors[r]].y();
		A(neighbors.size() +3, r) = positions[neighbors[r]].z();

		//RHSs
		for(int c = 0; c < ballMembers.size(); ++c){
		  double scale = 3*(positions[neighbors[r]] - positions[ballMembers[c]]).squaredNorm();
		  bx(r, c) = scale*positions[neighbors[r]].x();
		  by(r, c) = scale*positions[neighbors[r]].y();
		  bz(r, c) = scale*positions[neighbors[r]].z();
		}
	  }

	  //the B phi block at the bottom
	  for(int c = 0; c < ballMembers.size(); ++c){
		bx(neighbors.size() + 1, c) = 1;
		by(neighbors.size() + 2, c) = 1;
		bz(neighbors.size() + 3, c) = 1;
	  }

	  Eigen::PartialPivLU<Eigen::MatrixXd> decomp(A.selfadjointView<Eigen::Lower>());
	  Eigen::MatrixXd ddx = decomp.solve(bx);
	  Eigen::MatrixXd ddy = decomp.solve(by);
	  Eigen::MatrixXd ddz = decomp.solve(bz);

	  for(auto bi = 0; bi < ballMembers.size(); ++bi){
		int b = ballMembers[bi];
		particles[b].rbfIndices.resize(neighbors.size());
		particles[b].rbfWeights.resize(neighbors.size());

		for(int pi = 0; pi < neighbors.size(); ++pi){
		  particles[b].rbfIndices[pi] = neighbors[pi];
		  particles[b].rbfWeights[pi] = Vec3(ddx(pi, bi), ddy(pi, bi), ddz(pi, bi));

		}
	  }
	}
  }
  
}

Mat3 DeformableObject::computeDeformationGradientRBF(int pIndex) const{

  Mat3 ret = Mat3::Zero();
  const auto& p = particles[pIndex];
  for(int i = 0; i < p.rbfIndices.size(); ++i){
	ret.col(0) += p.rbfWeights[i].x()*particles[p.rbfIndices[i]].position;
	ret.col(1) += p.rbfWeights[i].y()*particles[p.rbfIndices[i]].position;
	ret.col(2) += p.rbfWeights[i].z()*particles[p.rbfIndices[i]].position;
  }
  return ret;
}