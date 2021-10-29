//breaking.cpp
void perform_breaking(Mesh& mesh)
{
    // compute sigma and separation strength for complete mesh
    for (size_t i = 0; i < mesh.faces.size(); i++)
        mesh.faces[i]->sigma = compute_sigma(mesh.faces[i]);

    MeshSubset subset;
    for (size_t i = 0; i < mesh.nodes.size(); i++) {
        mesh.nodes[i]->flag &= ~Node::FlagResolveMax;
        if (separation_strength(mesh.nodes[i], 0, false) > 1)
            subset.active_nodes.push_back(mesh.nodes[i]);
    }
    if (subset.active_nodes.empty())
        return;

    vector<AccelStruct*> obs_accs = create_accel_structs(sim.obstacle_meshes, false);

    // fracture sub-stepping
    for (int num = 0; num < ::magic.max_cracks && !subset.active_nodes.empty(); num++) {
        subset.set_flag(Node::FlagMayBreak);
        subset.grow(SUPPORT_RINGS);
        subset.set_flag(Node::FlagResolveMax);

        // dynamic remeshing
        map<Node*, Plane> planes = nearest_obstacle_planes(subset.get_all_nodes(), obs_accs);
        dynamic_remesh(subset, planes);
        subset.update_support();

        // physics update
        local_physics_step(subset);

        // recompute separation strength
        SplitNode cur_split(0);
        priority_queue<SplitNode> split;
        for (size_t i = 0; i < subset.active_nodes.size(); i++)
            if ((subset.active_nodes[i]->flag & Node::FlagMayBreak) && separation_strength(subset.active_nodes[i], &cur_split, false) > 1)
                split.push(cur_split);

        // rewind
        for (size_t i = 0; i < subset.active_nodes.size(); i++)
            subset.active_nodes[i]->x = subset.active_nodes[i]->x0;
        for (size_t i = 0; i < subset.support_nodes.size(); i++)
            subset.support_nodes[i]->x = subset.support_nodes[i]->x0;

        //subset.debug();
        subset.clear_flag(Node::FlagMayBreak);
        subset.active_nodes.clear();

        // break biggest one, put the rest into the zone for next round
        while (!split.empty()) {
            cur_split = split.top();
            split.pop();
            if (subset.active_nodes.empty()) {
                break_node(cur_split, subset);
            } else
                subset.active_nodes.push_back(cur_split.node);
        }
    }

    destroy_accel_structs(obs_accs);

    compute_ms_data(mesh);
}
