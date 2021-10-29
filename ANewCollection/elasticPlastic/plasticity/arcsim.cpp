//plasticity.cpp
void plastic_update(Cloth& cloth)
{
    Mesh& mesh = cloth.mesh;
    for (size_t f = 0; f < mesh.faces.size(); f++) {
        Face* face = mesh.faces[f];
        double S_yield = face->material->yield_curv;
        Mat3x3 S_total = curvature<WS>(face);
        Mat3x3 S_elastic = S_total - face->Sp_bend;
        double dS = norm_F(S_elastic);
        if (dS > S_yield) {
            face->Sp_bend += S_elastic / dS * (dS - S_yield);
            face->damage += dS / S_yield - 1;
        }
    }
    recompute_edge_plasticity(cloth.mesh);
}
void recompute_edge_plasticity(Mesh& mesh)
{
    for (int e = 0; e < (int)mesh.edges.size(); e++) {
        mesh.edges[e]->theta_ideal = 0;
        mesh.edges[e]->damage = 0;
    }
    for (int f = 0; f < (int)mesh.faces.size(); f++) {
        const Face* face = mesh.faces[f];
        Vec3 theta = face_to_edges(face->Sp_bend, face);
        for (int e = 0; e < 3; e++) {
            face->adje[e]->theta_ideal += theta[e];
            face->adje[e]->damage += face->damage;
        }
    }
    for (int e = 0; e < (int)mesh.edges.size(); e++) {
        Edge* edge = mesh.edges[e];
        if (edge->adjf[0] && edge->adjf[1]) { // edge has two adjacent faces
            edge->theta_ideal /= 2;
            edge->damage /= 2;
        }
    }
}