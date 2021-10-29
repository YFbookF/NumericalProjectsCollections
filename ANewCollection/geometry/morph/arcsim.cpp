//physics.cpp
void add_morph_forces(const Cloth& cloth, const Morph& morph, double t,
    double dt, vector<Vec3>& fext, vector<Mat3x3>& Jext)
{
    const Mesh& mesh = cloth.mesh;
    for (int v = 0; v < (int)mesh.verts.size(); v++) {
        const Vert* vert = mesh.verts[v];
        Vec3 x = morph.pos(t, vert->u);
        double stiffness = exp(morph.log_stiffness.pos(t));
        //Vec3 n = vert->node->n;
        double s = stiffness * vert->node->a;
        // // lower stiffness in tangential direction
        // Mat3x3 k = s*outer(n,n) + (s/10)*(Mat3x3(1) - outer(n,n));
        Mat3x3 k = Mat3x3(s);
        double c = sqrt(s * vert->node->m); // subcritical damping
        Mat3x3 d = c / s * k;
        fext[vert->node->index] -= k * (vert->node->x - x);
        fext[vert->node->index] -= d * vert->node->v;
        Jext[vert->node->index] -= k + d / dt;
    }
}
