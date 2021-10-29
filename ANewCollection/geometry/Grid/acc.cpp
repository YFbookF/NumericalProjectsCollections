//#include "accelerationGrid.h"
//template madness
//https://github.com/benjones/adaptiveDeformables/blob/master/AccelerationGrid.cpp
#include <numeric>
#include <limits>
#include <iostream>


template <typename ParticleType, typename Px>
void AccelerationGrid<ParticleType, Px>::updateGrid(const std::vector<ParticleType>& particles){
  
  assert(numBuckets > 0);
  const auto getter = Px{};
  
  Vec3 bbMin = 
	std::accumulate(particles.begin(), particles.end(),
		Vec3{std::numeric_limits<double>::max(),
			std::numeric_limits<double>::max(),
			std::numeric_limits<double>::max()},
		[&getter](Vec3 best, const ParticleType& p){
		  return Vec3{std::min(best(0),getter(p)(0)),
					  std::min(best(1),getter(p)(1)),
					  std::min(best(2),getter(p)(2))};
		  
		});
  
  
  
  Vec3 bbMax = 
	std::accumulate(particles.begin(), particles.end(),
		Vec3{std::numeric_limits<double>::min(),
			std::numeric_limits<double>::min(),
			std::numeric_limits<double>::min()},
		[&getter](Vec3 best, const ParticleType& p){
		  return Vec3{std::max(best(0),getter(p)(0)),
					  std::max(best(1),getter(p)(1)),
					  std::max(best(2),getter(p)(2))};
		  
		});
  
  delta = (bbMax - bbMin)/numBuckets;
  //give us a little slack space
  bbMin -= 0.05*delta;
  bbMax += 0.05*delta;
  origin = bbMin;
  delta = (bbMax - bbMin)/numBuckets;
  
  grid.assign(numBuckets*numBuckets*numBuckets, {});

  for(auto i = 0; i < particles.size(); ++i){
	const auto bucket = getBucket(getter(particles[i]));
	grid[index(bucket)].push_back(i);
  }
}

template <typename ParticleType, typename Px>
template <typename RGetter>
void AccelerationGrid<ParticleType, Px>::updateGridWithRadii(
	const std::vector<ParticleType>& particles, RGetter&& rGetter){
  
  assert(numBuckets > 0);

  const auto getter = Px{};

  Vec3 bbMin = 
	std::accumulate(particles.begin(), particles.end(),
		Vec3{std::numeric_limits<double>::max(),
			std::numeric_limits<double>::max(),
			std::numeric_limits<double>::max()},
		[&getter](Vec3 best, const ParticleType& p){
		  return Vec3{std::min(best(0),getter(p)(0)),
					  std::min(best(1),getter(p)(1)),
					  std::min(best(2),getter(p)(2))};
		  
		});
  
  
  
  Vec3 bbMax = 
	std::accumulate(particles.begin(), particles.end(),
		Vec3{std::numeric_limits<double>::min(),
			std::numeric_limits<double>::min(),
			std::numeric_limits<double>::min()},
		[&getter](Vec3 best, const ParticleType& p){
		  return Vec3{std::max(best(0),getter(p)(0)),
					  std::max(best(1),getter(p)(1)),
					  std::max(best(2),getter(p)(2))};
		  
		});
  
  delta = (bbMax - bbMin)/numBuckets;
  //give us a little slack space
  bbMin -= 0.05*delta;
  bbMax += 0.05*delta;
  origin = bbMin;
  delta = (bbMax - bbMin)/numBuckets;
  
  grid.assign(numBuckets*numBuckets*numBuckets, {});

  for(int pIndex = 0; pIndex < particles.size(); ++pIndex){
	const auto& p = particles[pIndex];
	const auto& pos = getter(p);

	//conservative estimate of how many nearby buckets to splat to
	const double radius = rGetter(p);
	
	const auto minBucket = getBucket(pos.array() - radius);
	const auto maxBucket = getBucket(pos.array() + radius);

	for(auto i  = minBucket(0); i < maxBucket(0) + 1; ++i){
	  for(auto j = minBucket(1); j < maxBucket(1) + 1; ++j){
		for(auto k = minBucket(2); k < maxBucket(2) +1; ++k){
		  //		std::cout << "i " << i << " j " << j << " k " << k << std::endl;
		  //		std::cout << "index: " << index(i,j,k) << std::endl;
		  //		std::cout << "grid size: " << grid.size() << std::endl;
		  grid[index(i,j,k)].push_back(pIndex);
		}
	  }
	}
  } 

}

template<typename T>
std::pair<T, T> makeSortedPair(T a, T b){
  return a < b ? std::make_pair(a, b) : std::make_pair(b,a);
}

template<typename ParticleType, typename Px>
std::vector<std::pair<int, int> >
AccelerationGrid<ParticleType, Px>::getPotentialPairs() const{

  std::vector<std::pair<int, int> > ret;
  for(auto& bucket : grid){
	for(auto i = 0; i < bucket.size(); ++i){
	  for(auto j  = 0; j < i + 1, bucket.size(); ++j){
		ret.push_back(makeSortedPair(bucket[i], bucket[j]));
	  }
	}
  }
  std::sort(ret.begin(), ret.end());
  ret.erase(std::unique(ret.begin(), ret.end()),
	  ret.end());
  return ret;
  
}



template <typename ParticleType, typename Px>
std::vector<int> 
AccelerationGrid<ParticleType, Px>::
getNearestNeighbors(const std::vector<ParticleType>& particles, 
					const Vec3& x, double r) const {
  
  //const auto bucket = getBucket(p);
  
  const auto rSquared = r*r;
  //const auto p = Px{}(particles[pIndex]);

  const auto getter = Px{};
  
  std::vector<int> ret;
  
  const auto minBucket = getBucket(x.array() - r);
  const auto maxBucket = getBucket(x.array() + r);
  
  for(auto i  = minBucket(0); i < maxBucket(0) +1; ++i){
	for(auto j = minBucket(1); j < maxBucket(1) + 1; ++j){
	  for(auto k = minBucket(2); k < maxBucket(2) +1; ++k){
		//		std::cout << "i " << i << " j " << j << " k " << k << std::endl;
		//		std::cout << "index: " << index(i,j,k) << std::endl;
		//		std::cout << "grid size: " << grid.size() << std::endl;
		for(auto q : grid[index(i,j,k)]){
		  if ((x - getter(particles[q])).squaredNorm() <= rSquared){
			ret.push_back(q);
		  }
		}
	  }
	}
  }
  
  
  return ret;
}