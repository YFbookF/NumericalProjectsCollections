//// https://github.com/xelatihy/yocto-gl/
// Approximate color of blackbody radiation from wavelength in nm.
inline vec3f blackbody_to_rgb(float temperature) {
  // clamp to valid range
  auto t = clamp(temperature, 1667.0f, 25000.0f) / 1000.0f;
  // compute x
  auto x = 0.0f;
  if (temperature < 4000.0f) {
    x = -0.2661239f * 1 / (t * t * t) - 0.2343589f * 1 / (t * t) +
        0.8776956f * (1 / t) + 0.179910f;
  } else {
    x = -3.0258469f * 1 / (t * t * t) + 2.1070379f * 1 / (t * t) +
        0.2226347f * (1 / t) + 0.240390f;
  }
  // compute y
  auto y = 0.0f;
  if (temperature < 2222.0f) {
    y = -1.1063814f * (x * x * x) - 1.34811020f * (x * x) + 2.18555832f * x -
        0.20219683f;
  } else if (temperature < 4000.0f) {
    y = -0.9549476f * (x * x * x) - 1.37418593f * (x * x) + 2.09137015f * x -
        0.16748867f;
  } else {
    y = +3.0817580f * (x * x * x) - 5.87338670f * (x * x) + 3.75112997f * x -
        0.37001483f;
  }
  return xyz_to_rgb(xyY_to_xyz({x, y, 1}));
}
