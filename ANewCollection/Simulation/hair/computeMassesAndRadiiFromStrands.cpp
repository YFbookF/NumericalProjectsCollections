//https://github.com/pielet/Hair-DER CompliantImplicitEuler
void CompliantImplicitEuler::computeMassesAndRadiiFromStrands()
{
	for (int si = 0; si < m_strand_num; ++si) {
		StrandForce* const strand = dynamic_cast<StrandForce*>(m_strand_force[si]);
		if (strand == NULL) continue;

		if (strand->m_strandParams->m_straightHairs != 1.0) {
			Vec2Array& kappas = strand->alterRestKappas();
			for (int k = 0; k < kappas.size(); ++k) {
				kappas[k] *= strand->m_strandParams->m_straightHairs;
			}
		}

		for (int v = 0; v < strand->getNumVertices(); ++v) {
			const int global_vtx = getStartIndex(strand->m_globalIndex) + v;
			const int global_edx = global_vtx - si;
			const int global_dof = getDof(global_vtx);
			const scalar r = strand->m_strandParams->getRadius(v, strand->getNumVertices());

			m_radii[global_vtx] = r;
			m_m.segment<3>(global_dof).setConstant(strand->m_vertexMasses[v]);

			if (v < strand->getNumEdges()) {
				m_edge_to_hair[global_edx] = si;
				// Edge radius, edge's should be indexed the same as 
				m_edge_radii[global_edx] = r;

				// Twist Mass (Second moment of inertia * length)
				const scalar mass = strand->m_strandParams->m_density * M_PI * r * r * strand->m_restLengths[v];
				scalar vtm = 0.25 * mass * 2 * r * r;
				m_m[global_dof + 3] = vtm;
			}
		}
	}
}