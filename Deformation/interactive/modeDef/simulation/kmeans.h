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
	     Vector3* clusters, int* nclustered, int* membership);
