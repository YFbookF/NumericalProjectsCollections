    //mitsuba
	void getNormalDerivative(const Intersection &its,
            Vector &dndu, Vector &dndv, bool shadingFrame) const {

        const std::vector<Float> &times = m_kdtree->getTimes();
        int frameIndex = m_kdtree->findFrame(its.time);
        Float alpha = std::max((Float) 0.0f, std::min((Float) 1.0f,
            (its.time - times[frameIndex])
            / (times[frameIndex + 1] - times[frameIndex])));

        uint32_t primIndex = its.primIndex, shapeIndex = its.other;
        const TriMesh *trimesh0 = m_kdtree->getMesh(frameIndex,   shapeIndex);
        const TriMesh *trimesh1 = m_kdtree->getMesh(frameIndex+1, shapeIndex);
        const Point *vertexPositions0 = trimesh0->getVertexPositions();
        const Point *vertexPositions1 = trimesh1->getVertexPositions();
        const Point2 *vertexTexcoords0 = trimesh0->getVertexTexcoords();
        const Point2 *vertexTexcoords1 = trimesh1->getVertexTexcoords();
        const Normal *vertexNormals0 = trimesh0->getVertexNormals();
        const Normal *vertexNormals1 = trimesh1->getVertexNormals();

        if (!vertexNormals0 || !vertexNormals1) {
            dndu = dndv = Vector(0.0f);
        } else {
            const Triangle &tri = trimesh0->getTriangles()[primIndex];
            uint32_t idx0 = tri.idx[0],
                     idx1 = tri.idx[1],
                     idx2 = tri.idx[2];

            const Point
                p0 = (1-alpha)*vertexPositions0[idx0] + alpha*vertexPositions1[idx0],
                p1 = (1-alpha)*vertexPositions0[idx1] + alpha*vertexPositions1[idx1],
                p2 = (1-alpha)*vertexPositions0[idx2] + alpha*vertexPositions1[idx2];

            /* Recompute the barycentric coordinates, since 'its.uv' may have been
               overwritten with coordinates of the texture "parameterization". */
            Vector rel = its.p - p0, du = p1 - p0, dv = p2 - p0;

            Float b1  = dot(du, rel), b2 = dot(dv, rel), /* Normal equations */
                  a11 = dot(du, du), a12 = dot(du, dv),
                  a22 = dot(dv, dv),
                  det = a11 * a22 - a12 * a12;

            if (det == 0) {
                dndu = dndv = Vector(0.0f);
                return;
            }

            Float invDet = 1.0f / det,
                  u = ( a22 * b1 - a12 * b2) * invDet,
                  v = (-a12 * b1 + a11 * b2) * invDet,
                  w = 1 - u - v;

            const Normal
                n0 = normalize((1-alpha)*vertexNormals0[idx0] + alpha*vertexNormals1[idx0]),
                n1 = normalize((1-alpha)*vertexNormals0[idx1] + alpha*vertexNormals1[idx1]),
                n2 = normalize((1-alpha)*vertexNormals0[idx2] + alpha*vertexNormals1[idx2]);

            /* Now compute the derivative of "normalize(u*n1 + v*n2 + (1-u-v)*n0)"
               with respect to [u, v] in the local triangle parameterization.

               Since d/du [f(u)/|f(u)|] = [d/du f(u)]/|f(u)|
                 - f(u)/|f(u)|^3 <f(u), d/du f(u)>, this results in
            */

            Normal N(u * n1 + v * n2 + w * n0);
            Float il = 1.0f / N.length(); N *= il;

            dndu = (n1 - n0) * il; dndu -= N * dot(N, dndu);
            dndv = (n2 - n0) * il; dndv -= N * dot(N, dndv);

            if (vertexTexcoords0 && vertexTexcoords1) {
                /* Compute derivatives with respect to a specified texture
                   UV parameterization.  */
                const Point2
                    uv0 = (1-alpha)*vertexTexcoords0[idx0] + alpha*vertexTexcoords1[idx0],
                    uv1 = (1-alpha)*vertexTexcoords0[idx1] + alpha*vertexTexcoords1[idx1],
                    uv2 = (1-alpha)*vertexTexcoords0[idx2] + alpha*vertexTexcoords1[idx2];

                Vector2 duv1 = uv1 - uv0, duv2 = uv2 - uv0;

                det = duv1.x * duv2.y - duv1.y * duv2.x;

                if (det == 0) {
                    dndu = dndv = Vector(0.0f);
                    return;
                }

                invDet = 1.0f / det;
                Vector dndu_ = ( duv2.y * dndu - duv1.y * dndv) * invDet;
                Vector dndv_ = (-duv2.x * dndu + duv1.x * dndv) * invDet;
                dndu = dndu_; dndv = dndv_;
            }
        }
    }

