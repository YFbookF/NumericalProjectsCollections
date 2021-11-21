
#include fast_marching_method.h
 https://github.com/nmwsharp/variational-surface-cutting/blob/master/core/src/fast_marching_method.cpp
#include queue
#include tuple


namespace GC {

VertexDatadouble FMMDistance(GeometryEuclidean geometry, const stdvectorstdpairVertexPtr, double& initialDistances)
{

    HalfedgeMesh mesh = geometry-getMesh();

     Necessary geometric quantities
    EdgeDatadouble edgeLengths; geometry-getEdgeLengths(edgeLengths);
    HalfedgeDatadouble oppAngles; geometry-getHalfedgeAngles(oppAngles);

    return FMMDistance(mesh, initialDistances, edgeLengths, oppAngles);
}
    
VertexDatadouble FMMDistance(HalfedgeMesh mesh, const stdvectorstdpairVertexPtr, double& initialDistances,
                               const EdgeDatadouble& edgeLengths, const HalfedgeDatadouble& oppAngles)
{

    typedef stdpairdouble, VertexPtr Entry;

     Initialize 
    VertexDatadouble distances(mesh, stdnumeric_limitsdoubleinfinity());
    VertexDatachar finalized(mesh, false);

    stdpriority_queueEntry, stdvectorEntry, stdgreaterEntry frontierPQ;
    for(auto& x  initialDistances) {
        frontierPQ.push(stdmake_pair(x.second, x.first));
    }
    size_t nFound = 0;
    size_t nVert = mesh-nVertices();
    
     Search
    while(nFound  nVert && !frontierPQ.empty()) {

         Pop the nearest element
        Entry currPair = frontierPQ.top();
        frontierPQ.pop();
        VertexPtr currV = currPair.second;
        double currDist = currPair.first;


         Accept it if not stale
        if(finalized[currV]) {
            continue;
        }
        distances[currV] = currDist;
        finalized[currV] = true;
        nFound++;


         Add any eligible neighbors
        for(HalfedgePtr he  currV.incomingHalfedges()) {
            VertexPtr neighVert = he.vertex();

             Add with length
            if(!finalized[neighVert]) {
                double newDist = currDist + edgeLengths[he.edge()];
                if(newDist  distances[neighVert]) {
                    frontierPQ.push(stdmake_pair(currDist + edgeLengths[he.edge()], neighVert));
                    distances[neighVert] = newDist;
                }
                continue;
            }

             Check the third point of the left triangle straddling this edge
            if(he.isReal()) { 
                VertexPtr newVert = he.next().next().vertex();
                if(!finalized[newVert]) {

                     Compute the distance
                    double lenB = edgeLengths[he.next().next().edge()];
                    double distB = currDist;
                    double lenA = edgeLengths[he.next().edge()];
                    double distA = distances[neighVert];
                    double theta = oppAngles[he];
                    double newDist = eikonalDistanceSubroutine(lenA, lenB, theta, distA, distB);

                    if(newDist  distances[newVert]) {
                        frontierPQ.push(stdmake_pair(newDist, newVert));
                        distances[newVert] = newDist;
                    }
                }
            }

             Check the third point of the right triangle straddling this edge
            HalfedgePtr heT = he.twin();
            if(heT.isReal()) { 
                VertexPtr newVert = heT.next().next().vertex();
                if(!finalized[newVert]) {

                     Compute the distance
                    double lenB = edgeLengths[heT.next().edge()];
                    double distB = currDist;
                    double lenA = edgeLengths[heT.next().next().edge()];
                    double distA = distances[neighVert];
                    double theta = oppAngles[heT];
                    double newDist = eikonalDistanceSubroutine(lenA, lenB, theta, distA, distB);
    
                    if(newDist  distances[newVert]) {
                        frontierPQ.push(stdmake_pair(newDist, newVert));
                        distances[newVert] = newDist;
                    }
                }
            }
            
        }

    }

    return distances;
}


 The super fun quadratic distance function in the Fast Marching Method on triangle meshes
 TODO parameter c isn't actually defined in paper, so I guessed that it was an error
double eikonalDistanceSubroutine(double a, double b, double theta, double dA, double dB) {


    if(theta = PI2.0) {
        double u = dB - dA;
        double cTheta = stdcos(theta);
        double sTheta2 = 1.0 - cThetacTheta;

         Quadratic equation
        double quadA = aa + bb - 2abcTheta;
        double quadB = 2bu  (acTheta - b);
        double quadC = bb  (uu - aasTheta2);
        double sqrtVal = stdsqrt(quadBquadB - 4quadAquadC);
         double tVals[] = {(-quadB + sqrtVal)  (2quadA),         seems to always be the first one
                           (-quadB - sqrtVal)  (2quadA)};

        double t = (-quadB + sqrtVal)  (2quadA);
        if(u  t && acTheta  b(t-u)t && b(t-u)t  a  cTheta) {
            return dA + t;
        } else {
            return stdmin(b + dA, a + dB);
        }

    } 
     Custom by Nick to get acceptable results in obtuse triangles without fancy unfolding 
    else {

        double maxDist = stdmax(dA, dB);  all points on base are less than this far away, by convexity
        double c = stdsqrt(aa + bb - 2abstdcos(theta));
        double area = 0.5  stdsin(theta)  a  b;
        double altitude = 2  area  c;  distance to base, must be inside triangle since obtuse
        double baseDist = maxDist + altitude;

        return stdmin({b + dA, a + dB, baseDist});
    }

}


}  namespace GC