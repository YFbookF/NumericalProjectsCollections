https://github.com/sethlu/renderbox-snow/tree/master/src
void
SnowSolver::implicitVelocityIntegrationMatrix(std::vector<glm::dvec3> &Av_next, std::vector<glm::dvec3> const &v_next) {
    LOG_ASSERT(Av_next.size() == v_next.size() && v_next.size() == gridNodes.size());

    auto numGridNodes = gridNodes.size();
    auto numParticleNodes = particleNodes.size();

    // x^n+1

    std::vector<glm::dvec3> x_next(numGridNodes);

    for (auto i = 0; i < numGridNodes; i++) {
        x_next[i] = gridNodes[i].position + delta_t * v_next[i];
    }

    // del_f

    std::vector<glm::dvec3> del_f(numGridNodes);

    for (auto p = 0; p < numParticleNodes; p++) {
        auto const &particleNode = particleNodes[p];
        auto gmin = glm::ivec3((particleNode.position / h) - glm::dvec3(1));

        // del_deformElastic

        glm::dmat3 del_deformElastic{};

        // Nearby weighted grid nodes
        for (unsigned int i = 0; i < 64; i++) {
            auto gx = gmin.x + i / 16;
            auto gy = gmin.y + (i / 4) % 4;
            auto gz = gmin.z + i % 4;
            if (!isValidGridNode(gx, gy, gz)) continue;
            auto &gridNode = this->gridNode(gx, gy, gz);

            del_deformElastic += glm::outerProduct(v_next[getGridNodeIndex(gx, gy, gz)],
                                                   particleNode.nabla_weight[i]);

        }

        del_deformElastic = delta_t * del_deformElastic * particleNode.deformElastic;

        // del_polarRotDeformElastic

        glm::dmat3 r, s;
        polarDecompose(particleNode.deformElastic, r, s);

        auto rtdf_dftr = (glm::transpose(r) * del_deformElastic - glm::transpose(del_deformElastic) * r);
        auto rtdr = glm::inverse(glm::dmat3(s[0][0] + s[1][1], s[2][1], -s[2][0],
                                            s[1][2], s[0][0] + s[2][2], s[0][1],
                                            -s[2][0], s[1][0], s[2][2] + s[1][1])) *
                    glm::dvec3(rtdf_dftr[1][0], rtdf_dftr[2][0], rtdf_dftr[2][1]);

        auto del_polarRotDeformElastic =
                r * glm::dmat3(0, -rtdr.x, -rtdr.y,
                               rtdr.x, 0, -rtdr.z,
                               rtdr.y, rtdr.z, 0);

        // jp, je, mu, lambda

        auto jp = glm::determinant(particleNode.deformPlastic);
        auto je = glm::determinant(particleNode.deformElastic);

        auto e = exp(hardeningCoefficient * (1 - jp));
        auto mu = mu0 * e;
        auto lambda = lambda0 * e;

        auto cofactor_deformElastic = je * glm::transpose(glm::inverse(particleNode.deformElastic));

        // del_je
        // FIXME: Better variable name?

        // Take Frobenius inner product
        auto del_je = ddot(cofactor_deformElastic, del_deformElastic);

        // del_cofactor_deformElastic

        auto &cde = cofactor_deformElastic;

        auto del_cofactor_deformElastic = glm::dmat3(
                ddot(glm::dmat3(0, 0, 0,
                                0, cde[2][2], -cde[2][1],
                                0, -cde[1][2], cde[1][1]),
                     del_deformElastic),
                ddot(glm::dmat3(0, 0, 0,
                                -cde[2][2], 0, cde[2][0],
                                cde[1][2], 0, -cde[1][0]),
                     del_deformElastic),
                ddot(glm::dmat3(0, 0, 0,
                                cde[2][1], -cde[2][0], 0,
                                -cde[1][1], cde[1][0], 0),
                     del_deformElastic),

                ddot(glm::dmat3(0, -cde[2][2], cde[2][1],
                                0, 0, 0,
                                0, cde[0][2], -cde[0][1]),
                     del_deformElastic),
                ddot(glm::dmat3(cde[2][2], 0, -cde[2][0],
                                0, 0, 0,
                                -cde[0][2], 0, cde[0][0]),
                     del_deformElastic),
                ddot(glm::dmat3(-cde[2][1], cde[2][0], 0,
                                0, 0, 0,
                                cde[0][1], -cde[0][0], 0),
                     del_deformElastic),

                ddot(glm::dmat3(0, cde[1][2], -cde[1][1],
                                0, -cde[0][2], cde[0][1],
                                0, 0, 0),
                     del_deformElastic),
                ddot(glm::dmat3(-cde[1][2], 0, cde[1][0],
                                cde[0][2], 0, -cde[0][0],
                                0, 0, 0),
                     del_deformElastic),
                ddot(glm::dmat3(cde[1][1], -cde[1][0], 0,
                                -cde[0][1], cde[0][0], 0,
                                0, 0, 0),
                     del_deformElastic));

        // Accumulate to del_f

        auto unweightedDelForce =
                -particleNode.volume0 * (2 * mu * (del_deformElastic - del_polarRotDeformElastic) +
                                         lambda * (cofactor_deformElastic * del_je +
                                                   (je - 1) * del_cofactor_deformElastic)) *
                glm::transpose(particleNode.deformElastic);

        // Nearby weighted grid nodes
        for (unsigned int i = 0; i < 64; i++) {
            auto gx = gmin.x + i / 16;
            auto gy = gmin.y + (i / 4) % 4;
            auto gz = gmin.z + i % 4;
            if (!isValidGridNode(gx, gy, gz)) continue;
            auto &gridNode = this->gridNode(gx, gy, gz);

            del_f[getGridNodeIndex(gx, gy, gz)] += unweightedDelForce * particleNode.nabla_weight[i];

        }

    }

    // Av_next

    for (auto i = 0; i < numGridNodes; i++) {
        Av_next[i] = v_next[i];
        if (gridNodes[i].mass > 0) {
            Av_next[i] -= beta * delta_t * del_f[i] / gridNodes[i].mass;
        }
    }

}