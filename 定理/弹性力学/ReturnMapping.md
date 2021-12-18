# CHRONO

E:\mycode\Elastic\chrono-develop\src\chrono\fea\ChContinuumMaterial.cpp

```
void ChContinuumPlasticVonMises::ComputeReturnMapping(ChStrainTensor<>& mplasticstrainflow,
                                                      const ChStrainTensor<>& mincrementstrain,
                                                      const ChStrainTensor<>& mlastelasticstrain,
                                                      const ChStrainTensor<>& mlastplasticstrain) const {
    ChStrainTensor<> guesselstrain(mlastelasticstrain);
    guesselstrain += mincrementstrain;  // assume increment is all elastic

    double vonm = guesselstrain.GetEquivalentVonMises();
    if (vonm > this->elastic_yeld) {
        ChVoightTensor<> mdev;
        guesselstrain.GetDeviatoricPart(mdev);
        mplasticstrainflow = mdev * ((vonm - this->elastic_yeld) / (vonm));
    } else {
        mplasticstrainflow.setZero();
    }
}

```

![image-20211218215837422](E:\mycode\collection\定理\弹性力学\image-20211218215837422.png)

