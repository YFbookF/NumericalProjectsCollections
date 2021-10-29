//mesh.cpp
void compute_ws_data(Node* node)
{
    node->n = normal<WS>(node);
    Mat3x3 C(0), C0(0);
    double sum = 0;
    for (size_t v = 0; v < node->verts.size(); v++) {
        const vector<Face*>& adjfs = node->verts[v]->adjf;
        for (size_t i = 0; i < adjfs.size(); i++) {
            Face const* face = adjfs[i];
            C += face->a / 3 * curvature<WS>(face);
            C0 += face->a / 3 * curvature<MS>(face);
            sum += face->a;
        }
    }
    Eig<3> eig = eigen_decomposition((C - C0) / sum);
    for (int i = 0; i < 3; i++)
        eig.l[i] = fabs(eig.l[i]);
    node->curvature = eig.Q * diag(eig.l) * eig.Q.t();
}