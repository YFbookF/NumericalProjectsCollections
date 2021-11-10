// https://github.com/xelatihy/yocto-gl/
// Splits a BVH node using the balance heuristic. Returns split position and
// axis.
[[maybe_unused]] static pair<int, int> split_balanced(vector<int>& primitives,
    const vector<bbox3f>& bboxes, const vector<vec3f>& centers, int start,
    int end) {
  // compute primitives bounds and size
  auto cbbox = invalidb3f;
  for (auto i = start; i < end; i++)
    cbbox = merge(cbbox, centers[primitives[i]]);
  auto csize = cbbox.max - cbbox.min;
  if (csize == vec3f{0, 0, 0}) return {(start + end) / 2, 0};

  // split along largest
  auto axis = 0;
  if (csize.x >= csize.y && csize.x >= csize.z) axis = 0;
  if (csize.y >= csize.x && csize.y >= csize.z) axis = 1;
  if (csize.z >= csize.x && csize.z >= csize.y) axis = 2;

  // balanced tree split: find the largest axis of the
  // bounding box and split along this one right in the middle
  auto middle = (start + end) / 2;
  std::nth_element(primitives.data() + start, primitives.data() + middle,
      primitives.data() + end,
      [axis, &centers](auto primitive_a, auto primitive_b) {
        return centers[primitive_a][axis] < centers[primitive_b][axis];
      });

  // if we were not able to split, just break the primitives in half
  if (middle == start || middle == end) return {(start + end) / 2, axis};

  // done
  return {middle, axis};
}