#include "math/primitives.h"
#include <float.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

/**************************************************
 * mykmeans()
 * A variation on kmeans where all clustered pts must
 * be within mindist.
 *
 * pts: a list of points (size nPts)
 * nPts: the # of points in pts
 * nClusters: the desired k # of clusters
 * mindist: minimum distance of cluster
 * maxIters: maximum # of iterations
 * clusters: (output) array of clusters (size nClusters)
 * nclustered: (output) array of # of clustered points (size nClusters)
 * membership: (output) array of point-cluster membership (size nPts)
 *     stores -1 if i'th point is unclustered
 *************************************************/
int mykmeans(const Vector3* pts, int nPts, int nClusters,
	    double mindist, int maxIters,
	    Vector3* clusters, int* nclustered, int* membership)
{
  int i,j,k;

  k = nClusters;

  double* dist = new double [nPts];
  int num_valid_clusters = 0;

  //we'll use distance squared
  mindist = mindist*mindist;

  Vector3 temp;

  //initialize clusters to k random points
  for(i=0; i<k; i++) {
    clusters[i] = pts[rand()%nPts];
  }

  for(int iter = 0; iter < maxIters; iter++) {
    //we have new clusters, initialize their inclusion to 0
    for(i=0; i<k; i++)
      nclustered[i] = 0;

    //calculate membership
    for(i=0; i<nPts; i++) {
      membership[i] = -1;
      dist[i] = mindist;
      for(j=0; j<k; j++) {
	//calc distance from this cluster, if it can/should be clustered, do so
	temp = pts[i] - clusters[j];
	double cdist = dot(temp,temp);
	if(cdist < dist[i]) {
	  membership[i] = j;
	  dist[i] = cdist;
	}
      }
      if(membership[i] != -1)
	nclustered[membership[i]]++;
    }
    
    //update clusters with the mean of the values clustered by them
    //if a cluster has no membership, toss it and find a new one
    for(i=0; i<nPts; i++) {
      if(membership[i] != -1)
	clusters[membership[i]] += pts[i];
    }
    num_valid_clusters = 0;
    for(i=0; i<k; i++) {
      if(nclustered[i]) {    //some clustered points in this cluster, keep it
	//its new position is the mean of all clustered points
	clusters[i] /= nclustered[i]+1;
	num_valid_clusters++;
      }
      else {
	clusters[i] = pts[rand()%nPts];
      }
    }
  }

  delete [] dist;

  //We're done!

  return nClusters;
}
