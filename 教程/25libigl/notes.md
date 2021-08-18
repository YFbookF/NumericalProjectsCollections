```
    //Kill non-interior DOFs
    VecXd Mint = M.diagonal();
    std::vector<std::vector<int> > bdryLoop;
    igl::boundary_loop(F,bdryLoop);
    for(const std::vector<int>& loop : bdryLoop)
        for(const int& bdryVert : loop)
            Mint(bdryVert) = 0.;

    //Invert Mint
    for(int i=0; i<Mint.rows(); ++i)
        if(Mint(i) > 0)
            Mint(i) = 1./Mint(i);

    //Repeat Mint to form diaginal matrix
    DiagMat stackedMinv = Mint.replicate(dim*dim,1).asDiagonal();

    //Compute squared Hessian
    SparseMat H;
    igl::hessian(V,F,H);
    Q = H.transpose()*stackedMinv*H;
```

enrygy Hess