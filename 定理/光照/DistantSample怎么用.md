https://github.com/sakanaman/kumo_viewer

```
 t += distant_sample(max_t, rnd(rand_state));
 __gpu__ float distant_sample(float sigma_t, float u)
{
    return -logf(1 - u)/sigma_t;
}
```

