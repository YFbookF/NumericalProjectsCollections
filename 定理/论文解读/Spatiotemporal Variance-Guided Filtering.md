https://github.com/ZheyuanXie/CUDA-Path-Tracer-Denoising

Spatiotemporal Variance-Guided Filtering: Real-Time Reconstruction for Path-Traced Global Illumination  

我们的光照边界停止函数，的关键地方是能够自动适应重归一化照明，基于标准求导。，需要少量协方差以及求导。这可能带来瑕疵。于是我们先用3x3 Gaussian Kernel ，可以极高重建质量。

The weight function typically combines geometrical and color based geometrical and color based edge-stopping function

Our novel weight function instead uses depth, world space normals, as well as the luminance of the filter input 
$$
w_i(p,q) = w_z \cdot w_n \cdot w_l
$$


```
// Edge-stopping weights
float wl = expf(-glm::distance(lp, lq) / (sqrt(var) * sigma_c + 1e-6));
float wn = min(1.0f, expf(-glm::distance(np, nq) / (sigma_n + 1e-6)));
float wx = min(1.0f, expf(-glm::distance(pp, pq) / (sigma_x + 1e-6)));

// filter weights
int k = (2 + i) + (2 + j) * 5;
float weight = h[k] * wl * wn * wx;
weights_sum += weight;
weights_squared_sum += weight * weight;
color_sum += (colorin[q] * weight);
variance_sum += (variance[q] * weight * weight);
```

我们推测本地深度模型使用裁剪空间深度的屏幕空间导数，也就是
$$
w_z = \exp(-\frac{|z(p) - z(q)|}{\sigma|\nabla z(p)  \cdot(p -q)| + \varepsilon})
$$
wz 是 depth，有钱就是wx

edge - stopping function on world space normals
$$
w_n = max(0,n(p) \cdot n(q))^{\sigma_n}
$$
Spatiotemporal Variance-Guided Filtering  
$$
w_l = \exp(-\frac{-|l_i(p) - l_i(q)|}{\sigma_l \sqrt{g_{3\times3}(Var(l_i(p)))} + \varepsilon})
$$
Since the luminance variance tends to reduce with subsequent lter iterations, the inuence of wl grows with each iteration, preventing overblurring.  注意用了GuassianKernel.

```
    // 3x3 Gaussian kernel
    float gaussian[9] = { 1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
                          1.0 / 8.0,  1.0 / 4.0, 1.0 / 8.0,
                          1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0 };

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < res.x && y < res.y) {
        int p = x + y * res.x;
        int step = 1 << level;
        
        float var;
        // perform 3x3 gaussian blur on variance
        if (blur_variance) {
            float sum = 0.0f;
            float sumw = 0.0f;
            glm::ivec2 g[9] = { glm::ivec2(-1, -1), glm::ivec2(0, -1), glm::ivec2(1, -1),
                               glm::ivec2(-1, 0),  glm::ivec2(0, 0),  glm::ivec2(1, 0),
                               glm::ivec2(-1, 1),  glm::ivec2(0, 1),  glm::ivec2(1, 1) };
            for (int sampleIdx = 0; sampleIdx < 9; sampleIdx++) {
                glm::ivec2 loc = glm::ivec2(x, y) + g[sampleIdx];
                if (loc.x >= 0 && loc.y >= 0 && loc.x < res.x && loc.y < res.y) {
                    sum += gaussian[sampleIdx] * variance[loc.x + loc.y * res.x];
                    sumw += gaussian[sampleIdx];
                }
            }
            var = max(sum / sumw, 0.0f);
        } else {
            var = max(variance[p], 0.0f);
        }
```

each step of an edge-aware a-trous wavelet decompostion using 5x5 cross -bilateral filter with weight function w between pixels
$$
\hat c_{i+1}(p) = \frac{\sum h(q) \cdot w(p,q) \cdot c(q)}{\sum h(q) \cdot w(p,q)}
$$


```
 // filter weights
                    int k = (2 + i) + (2 + j) * 5;
                    float weight = h[k] * wl * wn * wx;
                    weights_sum += weight;
                    weights_squared_sum += weight * weight;
                    color_sum += (colorin[q] * weight);
                    variance_sum += (variance[q] * weight * weight);
                }
            }
        }

        // update color and variance
        if (weights_sum > 10e-6) {
            colorout[p] = color_sum / weights_sum;
            variance[p] = variance_sum / weights_squared_sum;
        } else {
            colorout[p] = colorin[p];
        }
```

不过的代码协方差是怎么算的？

```
            if (valid) {
                // calculate alpha values that controls fade
                float color_alpha = max(1.0f / (float)(N + 1), color_alpha_min);
                float moment_alpha = max(1.0f / (float)(N + 1), moment_alpha_min);

                // incresase history length
                history_length_update[p] = (int)prevHistoryLength + 1;

                // color accumulation
                color_acc[p] = current_color[p] * color_alpha + prevColor * (1.0f - color_alpha);

                // moment accumulation
                float first_moment = moment_alpha * prevMoments.x + (1.0f - moment_alpha) * luminance;
                float second_moment = moment_alpha * prevMoments.y + (1.0f - moment_alpha) * luminance * luminance;
                moment_acc[p] = glm::vec2(first_moment, second_moment);

                // calculate variance from moments
                float variance = second_moment - first_moment * first_moment;
                variacne_out[p] = variance > 0.0f ? variance : 0.0f;
                return;
            }
```

