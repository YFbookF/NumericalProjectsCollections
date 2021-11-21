============A Comprehensive Framework for Rendering Layered Materials  

```
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, layer1.fourierOrders(), 1),
            [&](const tbb::blocked_range<size_t> &range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    const LayerMode &l1 = layer1[i], &l2 = layer2[i];
                    LayerMode &lo = output[i];

                    /* Gain for downward radiation */
                    Eigen::SparseLU<MatrixS, Eigen::AMDOrdering<int>> G_tb;
                    G_tb.compute(I - l1.reflectionBottom * l2.reflectionTop);

                    /* Gain for upward radiation */
                    Eigen::SparseLU<MatrixS, Eigen::AMDOrdering<int>> G_bt;
                    G_bt.compute(I - l2.reflectionTop * l1.reflectionBottom);

                    /* Transmission at the bottom due to illumination at the top */
                    MatrixS result = G_tb.solve(l1.transmissionTopBottom);
                    MatrixS Ttb = l2.transmissionTopBottom * result;

                    /* Reflection at the bottom */
                    MatrixS temp = l1.reflectionBottom * l2.transmissionBottomTop;
                    result = G_tb.solve(temp);
                    MatrixS Rb = l2.reflectionBottom + l2.transmissionTopBottom * result;

                    /* Transmission at the top due to illumination at the bottom */
                    result = G_bt.solve(l2.transmissionBottomTop);
                    MatrixS Tbt = l1.transmissionBottomTop * result;

                    /* Reflection at the top */
                    temp = l2.reflectionTop * l1.transmissionTopBottom;
                    result = G_bt.solve(temp);
                    MatrixS Rt = l1.reflectionTop + l1.transmissionBottomTop * result;

                    #if defined(DROP_THRESHOLD)
                        Ttb.prune((Float) 1, (Float) DROP_THRESHOLD);
                        Tbt.prune((Float) 1, (Float) DROP_THRESHOLD);
                        Rb.prune((Float) 1, (Float) DROP_THRESHOLD);
                        Rt.prune((Float) 1, (Float) DROP_THRESHOLD);
                    #endif

                    lo.transmissionTopBottom = Ttb;
                    lo.transmissionBottomTop = Tbt;
                    lo.reflectionTop = Rt;
                    lo.reflectionBottom = Rb;
```

![image-20211116151646443](E:\mycode\collection\定理\论文解读\image-20211116151646443.png)