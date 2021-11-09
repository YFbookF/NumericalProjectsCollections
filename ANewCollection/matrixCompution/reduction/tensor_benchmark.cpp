 //Projective Dynamics: Fast Simulation of Hyperelastic Models with Dynamic Constraints
//http://www.github.com/mattoverby/admm-elastic
 // Row reduction
  void rowReduction(int num_iters) {
    Eigen::array<TensorIndex, 2> input_size;
    input_size[0] = k_;
    input_size[1] = n_;
    const TensorMap<Tensor<T, 2, 0, TensorIndex>, Eigen::Aligned> B(b_, input_size);
    Eigen::array<TensorIndex, 1> output_size;
    output_size[0] = n_;
    TensorMap<Tensor<T, 1, 0, TensorIndex>, Eigen::Aligned> C(c_, output_size);

#ifndef EIGEN_HAS_INDEX_LIST
    Eigen::array<TensorIndex, 1> sum_along_dim;
    sum_along_dim[0] = 0;
#else
    // Take advantage of cxx11 to give the compiler information it can use to
    // optimize the code.
    Eigen::IndexList<Eigen::type2index<0>> sum_along_dim;
#endif

    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      C.device(device_) = B.sum(sum_along_dim);
    }
    // Record the number of FLOP executed per second (assuming one operation
    // per value)
    finalizeBenchmark(static_cast<int64_t>(k_) * n_ * num_iters);
  }

  // Column reduction
  void colReduction(int num_iters) {
    Eigen::array<TensorIndex, 2> input_size;
    input_size[0] = k_;
    input_size[1] = n_;
    const TensorMap<Tensor<T, 2, 0, TensorIndex>, Eigen::Aligned> B(
        b_, input_size);
    Eigen::array<TensorIndex, 1> output_size;
    output_size[0] = k_;
    TensorMap<Tensor<T, 1, 0, TensorIndex>, Eigen::Aligned> C(
        c_, output_size);

#ifndef EIGEN_HAS_INDEX_LIST
    Eigen::array<TensorIndex, 1> sum_along_dim;
    sum_along_dim[0] = 1;
#else
    // Take advantage of cxx11 to give the compiler information it can use to
    // optimize the code.
    Eigen::IndexList<Eigen::type2index<1>> sum_along_dim;
#endif

    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      C.device(device_) = B.sum(sum_along_dim);
    }
    // Record the number of FLOP executed per second (assuming one operation
    // per value)
    finalizeBenchmark(static_cast<int64_t>(k_) * n_ * num_iters);
  }

  // Full reduction
  void fullReduction(int num_iters) {
    Eigen::array<TensorIndex, 2> input_size;
    input_size[0] = k_;
    input_size[1] = n_;
    const TensorMap<Tensor<T, 2, 0, TensorIndex>, Eigen::Aligned> B(
        b_, input_size);
    Eigen::array<TensorIndex, 0> output_size;
    TensorMap<Tensor<T, 0, 0, TensorIndex>, Eigen::Aligned> C(
        c_, output_size);

    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      C.device(device_) = B.sum();
    }
    // Record the number of FLOP executed per second (assuming one operation
    // per value)
    finalizeBenchmark(static_cast<int64_t>(k_) * n_ * num_iters);
  }
