// https://github.com/xelatihy/yocto-gl/
// Update bvh
static void refit_bvh(vector<bvh_node>& nodes, const vector<int>& primitives,
    const vector<bbox3f>& bboxes) {
  for (auto nodeid = (int)nodes.size() - 1; nodeid >= 0; nodeid--) {
    auto& node = nodes[nodeid];
    node.bbox  = invalidb3f;
    if (node.internal) {
      for (auto idx = 0; idx < 2; idx++) {
        node.bbox = merge(node.bbox, nodes[node.start + idx].bbox);
      }
    } else {
      for (auto idx = 0; idx < node.num; idx++) {
        node.bbox = merge(node.bbox, bboxes[primitives[node.start + idx]]);
      }
    }
  }
}