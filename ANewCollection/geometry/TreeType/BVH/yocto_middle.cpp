// https://github.com/xelatihy/yocto-gl/
// Splits a BVH node using the middle heuristic. Returns split position and
// axis.
static pair<int, int> split_middle(vector<int>& primitives,
    const vector<bbox3f>& bboxes, const vector<vec3f>& centers, int start,
    int end) {
  // compute primintive bounds and size
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

  // split the space in the middle along the largest axis
  auto split = center(cbbox)[axis];
  auto middle =
      (int)(std::partition(primitives.data() + start, primitives.data() + end,
                [axis, split, &centers](auto primitive) {
                  return centers[primitive][axis] < split;
                }) -
            primitives.data());

  // if we were not able to split, just break the primitives in half
  if (middle == start || middle == end) return {(start + end) / 2, axis};

  // done
  return {middle, axis};
}