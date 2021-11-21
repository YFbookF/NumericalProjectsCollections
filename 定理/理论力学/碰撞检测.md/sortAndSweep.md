sortAndSwepp

https://developer.nvidia.com/blog/thinking-parallel-part-i-collision-detection-gpu/

In order to find the overlapping ranges, the algorithm collects the start points (S1, S2, S3) and end points (E1, E2, E3) of the ranges into an array, and sorts them along the axis. For each object, it then sweeps the list from the object’s start and end points (e.g. S2 and E2) and identifies all objects whose start point lies between them (e.g. S3). For each pair of objects found this way, the algorithm further checks their 3D bounding boxes for overlap, and reports the overlapping pairs as potential collisions.

我们可以使用并行radix sort来算

