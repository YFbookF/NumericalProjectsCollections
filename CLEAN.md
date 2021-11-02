

### FiniteElement



**cracksDeepBlue**

包括efficient optimization of reliability-constrained structural design problems including interval uncertainty以及Source code for proposed Multi-Scale FEM Crack Growth model 以及Software code for A dynamic discretization method for reliability inference in dynamic bayesian networks的源码



**mixed**

Mixed Finite Elements for Variational Surface Modelinghttps://igl.ethz.ch/projects/mixed-fem/



### Bubble

**AcousticBubbles**

Toward Animating Water with Complex Acoustic Bubbleshttps://www.cs.cornell.edu/projects/Sound/bubbles/

boling

**A simple boiling module**http://www.tkim.graphics/

### BoundaryElement

BoundaryElementProgramming

The Boundary Element Method with Programming这本书的源代码

### FluidSolidInteraction

### FireFlameSmoke

compress

***Compressing Fluid Subspaces\***http://www.tkim.graphics/COMPRESSING/source.html

resim

***Subspace Fluid Re-Simulation\***http://www.tkim.graphics/RESIM/

### LinearSystem

multipcg

https://www.cs.ubc.ca/~rbridson/ MATLAB source for the multi-preconditioned conjugate gradient algorithm

autodiff

 Automatic Differentiation of Moving Least Squares https://graphics.stanford.edu/courses/cs348c-17-fall/

parallel

****A parallel multigrid Poisson solver for fluids simulation on large grids****http://pages.cs.wisc.edu/~sifakis/project_pages/mgpcg.html

floatingPoint

*Computing the Singular Value Decomposition of 3x3 matrices with minimal branching and elementary floating point operations\*http://pages.cs.wisc.edu/~sifakis/project_pages/svd.html

derivate

"Practical notes on implementing derivatives"https://www.cs.ucr.edu/~craigs/research.html

piston

An introduction to fluid-structure interaction:
application to the piston problem http://www.utc.fr/~elefra02/ifs/

### FrontTracking

LevelsetSculpt

https://www.cs.ubc.ca/~rbridson/  a C++ simple sculpting program based on level sets, multi-resolution point splatting, etc. See my PhD thesis for a write-up on it.

mathingFluid

A 2D, C++ implementation of our voronoi fluid simulator is available here 

### Siggraph

Highly Adaptive Liquid Simulations on Tetrahedral Meshes

### Volumetric

volfill

Volfill: a hole filler based on volumetric diffusionhttp://graphics.stanford.edu/software/volfill/

diffusion

Volumetric Modeling with Diffusion Surfaceshttps://igl.ethz.ch/projects/diffusion-surfaces/ 未解压

### Gemotry

**BlueNoiseSampling**

http://graphics.uni-konstanz.de/publikationen/index.html#y2005

Blue Noise Sampling With Controlled Aliasing

**blueNoise**

Blue Noise through Optimal Transport
de Goes, Breeden, Ostromoukhov, Desbrun
SIGGRAPH Asia / ACM Transactions on Graphics (2012)http://fernandodegoes.org/

**simplification**

An optimal transport approach to robust reconstruction and simplification of 2D shapes de Goes, Cohen-Steiner, Alliez, Desbrun
SGP / Computer Graphics Forum (2011)http://fernandodegoes.org/

**Adapt Dynamic Meshes**

A Simple and Flexible Framework to Adapt Dynamic Meshes
de Goes, Goldenstein, Velho Computer & Graphics (2008)http://fernandodegoes.org/

Laplacian Surface Editing

Laplacian Surface Editinghttps://igl.ethz.ch/projects/Laplacian-mesh-processing/Laplacian-mesh-editing/index.php

InterferenceAware

Interference Aware Geometric Modelinghttps://cims.nyu.edu/gcl/daniele.html

c2spline

C2 Splines Covering Polar Configurations.https://ashishmyles.com/research.shtml

surfacePatch

Tri/Quad/Pent Surface Patch Construction and Rendering on the GPUhttps://www.cise.ufl.edu/research/SurfLab/research.shtml#08poly

phong

Phong Deformation: A better C0 interpolant for embedded deformationhttp://graphics.stanford.edu/~djames/publications/

days

**A massive fractal in days, not years.**

### FLIP_MPM

simpleflip

https://www.cs.ubc.ca/~rbridson/。文件写入和展示画面部分有问题，不过对我只看变量的我来说等于没问题。

narrowBand

http://ntoken.com/pubs.html#Thuerey_2016_ofblend **Narrow Band FLIP for Liquid Simulations**

### Physics

明明整个库就是物理模拟的库，为什么还要专门有个叫做“物理”的文件夹呢？

numericalConservation

https://archive.siam.org/books/cs18/ Numerical methods for Conservation laws: From Analysis to Algorithms

**manyworlds**

http://chris.twi.gg/software/mwbCode/ Many-Worlds Browsing for Control of Multibody Dynamics http://graphics.stanford.edu/~djames/publications/

### Sound

precomputed transfer

http://graphics.cs.cmu.edu/projects/pat/Precomputed Acoustic Transfer: Output-sensitive, accurate sound generation for geometrically complex vibration sources。项目已无法构建。

### WaterWaveFluid

curlNoise

http://graphics.stanford.edu/courses/cs348c-16-fall/ Robert Bridson, Jim Houriham, Marcus Nordenstam, Curl-noise for procedural fluid flow, ACM Transactions on Graphics (TOG), v.26 n.3, July 2007.

waveletsTurbulence

论文名称：Wavelet Turbulence for Fluid Simulation

项目地址： https://www.cs.cornell.edu/~tedkim/WTURB/

值得一看：比perlinNoise更好的turbulenceNoise。heatEquation计算方法。三维Voritity计算方法。共轭梯度解算压力

构建指南：我把noiseFFT从项目中移除了，并且注释掉image.h中的几个函数内容。即可成功运行。

复刻进度：主体部分完成，turbulenceNoise没写完，因为涉及到一大堆LU分解，特征值计算，我还不熟悉

FEMFluid

https://www.pplusplus.lima-city.de/lib/data/femfluid/FEM%20Fluid%20Source.zip

flux

A Flux-Interpolated Advection Scheme for Fluid Simulationhttps://ryichando.graphics/

perceptual

Perceptual Evaluation of Liquid Simulation Methodshttps://ge.in.tum.de/publications/2017-sig-um/

**Synthetic**

Synthetic Turbulence using Artificial Boundary Layershttp://ntoken.com/pubs.html#Thuerey_2016_ofblend

close

***Closest Point Turbulence Source Code\***http://www.tkim.graphics/CPT/source.html

eigenFluid

http://www.tkim.graphics/    **Scalable laplacian eigenfluids.**

构建指南：我直接cmakeLists.txt里的find glut package删除了。

### Lights&Shadow

virtualSphericalLights

Virtual Spherical Lights for Many-Light Rendering of Glossy Sceneshttp://miloshasan.net/

# 本仓库未收录的代码

没收录的原因是因为这些代码存在于github/gitlab/gitee上，短时间内不会消失。或者这些代码是在太大了，超过10mb

### Deformation

论文名称：*Cubica:\*a toolkit for subspace deformations

Optimizing Cubature for Efficient Integration of Subspace Deformations

Skipping Steps in Deformable Simulation with Online Model Reduction

Physics-based Character Skinning using Multi-Domain Subspace Deformations

项目地址：http://www.tkim.graphics/cubica/ 

http://graphics.stanford.edu/~djames/publications/

代码地址：http://www.tkim.graphics/cubica/src/cubica-1.0.tar.gz

### FiniteElement

项目名称：A polyvalent C++/python FEM library

代码地址：https://github.com/polyfem/polyfem/

论文名称：**A Large Scale Comparison of Tetrahedral and Hexahedral Elements for Finite Element Analysis**

项目地址：https://cims.nyu.edu/gcl/daniele.html

代码地址：https://github.com/polyfem/tet-vs-hex

论文名称：****Poly-Spline Finite Element Method****

项目地址：https://cims.nyu.edu/gcl/daniele.html

代码地址：https://polyfem.github.io/

### FluidSolidInteraction

论文名称：A Multi-Scale Model for Simulating Liquid-Fabric Interactions

项目地址：https://cs.uwaterloo.ca/~c2batty/

代码地址：https://github.com/nepluno/libWetCloth

论文名称：**Eulerian solid-fluid coupling**

项目地址：http://www.tkim.graphics/

代码地址：https://github.com/yunteng/EULERSF

### Gemotry

论文名称：**Convolutional Wasserstein Distances: Efficient Optimal Transportation on Geometric Domains**
Solomon, de Goes, Peyré, Cuturi, Butscher, Nguyen, Du, Guibas

项目地址：http://fernandodegoes.org/

代码地址：https://github.com/gpeyre/2015-SIGGRAPH-convolutional-ot

论文名称：Ziwei Zhu and Changxi Zheng
*Differentiable scattering matrix for optimization of photonic structures*.
Optics Express, 28 (25), 2020

项目地址：http://www.cs.columbia.edu/~cxz/publications.php

代码地址：https://github.com/Columbia-Computational-X-Lab/DiffSMat

论文名称：**Fast Tile-Based Adaptive Sampling with User-Specified Fourier Spectra**
Wachtel, Pilleboue, Coeurjolly, Breeden, Singh, Cathelin, de Goes, Desbrun, Ostromoukhov

项目地址：http://fernandodegoes.org/

代码地址：http://github.com/polyhex-sampling/sampler

论文名称：**Geometry Processing with Discrete Exterior Calculus**
Crane, de Goes, Desbrun, Schroeder
**SIGGRAPH Courses (2013)**

项目地址：http://fernandodegoes.org/

代码地址：https://github.com/dgpdec/course

论文名称：**Lifting Simplices to Find Injectivity**
ACM Transactions on Graphics (SIGGRAPH 2020)

项目地址：http://dannykaufman.io/

代码地址：https://github.com/duxingyi-charles/lifting_simplices_to_find_injectivity

论文名称：**OptCuts: Joint Optimization of Surface Cuts and Parameterization**
ACM Transactions on Graphics (SIGGRAPH Asia 2018)

项目地址：http://dannykaufman.io/

代码地址：https://github.com/liminchen/OptCuts

论文名称：**Blended Cured Quasi-Newton for Distortion Optimization**
ACM Transactions on Graphics (SIGGRAPH 2018)

项目地址：http://dannykaufman.io/

代码地址：https://github.com/mike323zyf/BCQN

论文名称：**Fast Linking Numbers for Topology Verification of Loopy Structures**

项目地址：https://www.antequ.net/

代码地址：https://github.com/antequ/fastlinkingnumbers/

论文名称：**Effect of Geometric Sharpness on Translucent Material Perception**

Bei Xiao, Shuang Zhao, Ioannis Gkioulekas, Wenyan Bi, and Kavita Bala
*Journal of Vision, 20(7), July 2020*

项目地址：https://shuangz.com/publications.htm

代码地址：https://github.com/BumbleBee0819/OnlinePsychophysicsExperiment_AsymmetricMatching

论文名称：An Adaptive Virtual Node Algorithm With Robust Mesh Cutting

项目地址：https://www.math.ucla.edu/~cffjiang/publications.html

代码地址：https://github.com/loopstring/3d-cutter

论文名称：**Fast Tetrahedral Meshing in the Wild**

项目地址：https://cims.nyu.edu/gcl/daniele.html

代码地址：https://github.com/wildmeshing/fTetWild

论文名称：****Exact and Efficient Polyhedral Envelope Containment Check****

项目地址：https://cims.nyu.edu/gcl/daniele.html

代码地址：https://github.com/wangbolun300/fast-envelope

论文名称：Seamless: Seam erasure and seam-aware decoupling of shape from mesh resolution

项目地址：https://cragl.cs.gmu.edu/seamless/

代码地址：https://github.com/zfergus/seam-erasure

https://github.com/songrun/SeamAwareDecimater

论文名称：**Half-Space Power Diagrams and Discrete Surface Offsets**

项目地址：https://cims.nyu.edu/gcl/daniele.html

代码地址：https://github.com/geometryprocessing/voroffset

论文名称：**TriWild: Robust Triangulation with Curve Constraints**

项目地址：https://cims.nyu.edu/gcl/daniele.html

代码地址：https://github.com/wildmeshing/TriWild

论文名称：****Progressive Embedding****

项目地址：https://cims.nyu.edu/gcl/daniele.html

代码地址：https://github.com/hankstag/progressive_embedding

论文名称：**Feature Preserving Octree-Based Hexahedral Meshing**

项目地址：https://cims.nyu.edu/gcl/daniele.html

代码地址：https://github.com/gaoxifeng/Feature-Preserving-Octree-Hex-Meshin

论文名称：****Decoupling Simulation Accuracy from Mesh Quality****

项目地址：https://cims.nyu.edu/gcl/daniele.html

代码地址：https://github.com/polyfem/polyfem

论文名称：****Tetrahedral Meshing in the Wild****

项目地址：https://cims.nyu.edu/gcl/daniele.html

代码地址：https://github.com/Yixin-Hu/TetWild

论文名称：****Stitch Meshing****

项目地址：https://cims.nyu.edu/gcl/daniele.html

代码地址：https://github.com/kuiwuchn/stitchMeshingg

论文名称：****Stitch Meshing****

项目地址：https://cims.nyu.edu/gcl/daniele.html

代码地址：https://github.com/kuiwuchn/stitchMeshingg

论文名称：**Robust Hex-Dominant Mesh Generation using Field-Guided Polyhedral Agglomeration**

项目地址：https://cims.nyu.edu/gcl/daniele.html

代码地址：https://github.com/gaoxifeng/robust_hex_dominant_meshing

论文名称：****Instant Field-Aligned Meshes****

项目地址：https://cims.nyu.edu/gcl/daniele.html

代码地址：https://github.com/wjakob/instant-meshes

论文名称：Weighted Averages on Surfaces ACM SIGGRAPH 2013

项目地址：https://igl.ethz.ch/projects/wa/ 

代码地址：https://igl.ethz.ch/projects/wa/WA-code.zip

论文名称：**TriWild: Robust Triangulation with Curve Constraints**

项目地址：https://gaoxifeng.github.io/

代码地址：https://github.com/wildmeshing/TriWild

论文名称：**Robust Structure Simplification for Hex Re-meshing**

项目地址：http://graphics.cs.uh.edu/zdeng/

代码地址：https://github.com/gaoxifeng/Robust-Hexahedral-Re-Meshing

论文名称：**Hexahedral Mesh Generation, Optimization, and Evaluation**

项目地址：http://graphics.cs.uh.edu/zdeng/

代码地址：https://github.com/gaoxifeng/Evaluation_code_SGP2017

论文名称：**Conformal Mesh Deformations with Möbius Transformations**

项目地址：https://webspace.science.uu.nl/~vaxma001/

代码地址：https://github.com/avaxman/MoebiusCode

论文名称：****Hierarchical Functional Maps between Subdivision Surfaces****

项目地址：https://webspace.science.uu.nl/~vaxma001/

代码地址：http://www.cs.technion.ac.il/~mirela/code/hfm.zip

论文名称：**Discrete Laplacians on General Polygonal Meshes**

项目地址：http://num.math.uni-goettingen.de/~wardetzky/

代码地址：http://cybertron.cg.tu-berlin.de/polymesh

论文名称：**Data-Driven Interactive Quadrangulation**

项目地址：http://vcg.isti.cnr.it/Publications/2015/MTPPSPC15/

代码地址：..

论文名称：Developing fractal curves

项目地址：https://naml.us/

代码地址：https://github.com/otherlab/fractal

论文名称：You Can Find Geodesic Paths in Triangle Meshes by Just Flipping Edges

项目地址：https://nmwsharp.com/research/flip-geodesics/

代码地址：https://github.com/nmwsharp/potpourri3d#mesh-geodesic-paths

### Cloth

论文名称：A Wave Optics Based Fiber Scattering Model

项目地址：https://mandyxmq.github.io/research/wavefiber.html

代码地址：https://github.com/mandyxmq/WaveOpticsFiber

论文名称：Fitting Procedural Yarn Models for Realistic Cloth Rendering

项目地址：https://www.cs.cornell.edu/projects/ctcloth/

代码地址：http://www.cs.cornell.edu/~kb/projects/ctcloth/code_data_v2.zip

论文名称：Structure-aware Synthesis for Predictive Woven Fabric Appearance

项目地址：https://www.cs.cornell.edu/projects/ctcloth/

代码地址：https://www.cs.cornell.edu/projects/ctcloth/data/

### Skin&Seleketon

论文名称：Accurate and Efficient Lighting for Skinned Models

项目地址：http://vcg.isti.cnr.it/deformFactors/

代码地址：http://vcg.isti.cnr.it/deformFactors/deformFactors_source.zip

论文名称：Smooth Skinning Decomposition with Rigid Bones

项目地址：http://graphics.cs.uh.edu/zdeng/

代码地址：https://github.com/electronicarts/dem-bones

论文名称：Implicit Skinning: Real-Time Skin Deformation with Contact Modeling

项目地址：他人实现

代码地址：https://github.com/likangning93/HRBF-Skin

### Water&Flow&Fluid

论文名称：Functional Optimization of Fluidic Devices with Differentiable Stokes Flow

代码地址：https://github.com/mit-gfx/diff_stokes_flow

项目名称：Finite volume solver for incompressible multiphase flows with surface tension. Foaming flows in complex geometries.

代码地址：https://github.com/cselab/aphros

论文名称：Efficient And Conservative Fluids Using Bidirectional Mapping

项目地址：https://www.math.ucla.edu/~cffjiang/publications.html

代码地址：https://github.com/ziyinq/Bimocq

论文名称：*The Reduced Immersed Method for Real-Time Fluid-Elastic Solid Interaction and Contact Simulation*

项目地址：https://graphics.tudelft.nl/~klaus/

代码地址：https://github.com/RotateMe/RIM

论文名称：*An Adaptive Variational Finite Difference Framework for Efficient Symmetric Octree Viscosity*

项目地址：https://cs.uwaterloo.ca/~c2batty/

代码地址：https://github.com/rgoldade/AdaptiveViscositySolver

论文名称：3D Liquid Simulator code

项目地址：https://cs.uwaterloo.ca/~c2batty/

代码地址：https://github.com/christopherbatty/Fluid3D

论文名称：2D Variational Viscosity code

项目地址：https://cs.uwaterloo.ca/~c2batty/

代码地址：https://github.com/christopherbatty/VariationalViscosity2D

论文名称：2D Variational Pressure Projection with Rigid Body Coupling code

项目地址：https://cs.uwaterloo.ca/~c2batty/

代码地址：https://github.com/christopherbatty/FluidRigidCoupling2D

论文名称：Fluid mixing

项目地址：https://sites.google.com/view/valentinresseguier/projects

代码地址：https://github.com/vressegu/mixing

论文名称：Models under location uncertainty

项目地址：https://sites.google.com/view/valentinresseguier/projects

代码地址：https://github.com/vressegu/LU_SALT_SelfSim

https://github.com/vressegu/sqgmu

### BubbleFilmFoam

论文名称：*Constraint Bubbles and Affine Regions: Reduced Fluid Models for Efficient Immersed Bubbles and Flexible Spatial Coarsening*

项目地址：https://cs.uwaterloo.ca/~c2batty/

代码地址：https://github.com/rgoldade/ReducedFluids

论文名称：*A Hyperbolic Geometric Flow for Evolving Films and Foams*

项目地址：https://ryichando.graphics/

代码地址：https://github.com/sdsgisd/HGF

### Deformation

论文名称：Incremental Potential Contact:
Intersection- and Inversion-free Large Deformation Dynamics

项目地址：https://ipc-sim.github.io/

代码地址：https://github.com/ipc-sim/ipc-toolkit

论文名称：Decomposed Optimization Time Integrator for Large-Step Elastodynamics

项目地址：http://dannykaufman.io/projects/DOT/DOT.html

代码地址：https://github.com/penn-graphics-research/DOT

论文名称：An Asymptotic Numerical Method for Inverse Elastic Shape Design

项目地址：http://kunzhou.net/zjugaps/ANMdesign/

代码地址：http://kunzhou.net/zjugaps/ANMdesign/code/opensrc.zip

论文名称：A Multi-Scale Model for Coupling Strands with Shear-Dependent Liquid

项目地址：http://www.cs.columbia.edu/cg/creamystrand/

代码地址：https://github.com/nepluno/creamystrand

论文名称：PlasticineLab: A Soft-Body Manipulation Benchmark with Differentiable Physics

代码地址：https://github.com/lwkobe/PlasticineLab

论文名称：Decomposed Optimization Time Integrator For Large-Step Elastodynamics

项目地址：https://www.math.ucla.edu/~cffjiang/publications.html

代码地址：https://github.com/liminchen/DOT

论文名称：Locally Injective Mappings

项目地址：https://igl.ethz.ch/projects/LIM/

代码地址：https://igl.ethz.ch/projects/LIM/LIM-2013-code.zip

### Physics

论文名称：A Large Scale Benchmark and an Inclusion-Based Algorithm for Continuous Collision Detection

项目地址：https://continuous-collision-detection.github.io/

代码地址：https://github.com/Continuous-Collision-Detection

论文名称：**REDMAX: Efficient and Flexible Approach for Articulated Dynamics**
ACM Transactions on Graphics (SIGGRAPH 2019)

项目地址：http://dannykaufman.io/

代码地址：https://github.com/sueda/redmax

论文名称：**Dynamics-Aware Numerical Coarsening For Fabrication Design**
ACM Transactions on Graphics (SIGGRAPH 2017)

项目地址：http://dannykaufman.io/

代码地址：https://github.com/desaic/DAC

项目名称：Tiny Differentiable Simulator is a header-only C++ physics library with zero dependencies.【最近快被各种依赖各种库搞哭了】

代码地址：https://github.com/google-research/tiny-differentiable-simulator

### Math

论文名称：**Border-Peeling Clustering**
Hadar Averbuch-Elor, Nadav Bar, Daniel Cohen-Or
IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2019

项目地址：https://www.cs.cornell.edu/~hadarelor/

代码地址：https://github.com/nadavbar/BorderPeelingClustering

论文名称：*Fast Approximation of Laplace–Beltrami Eigenproblems*

项目地址：https://graphics.tudelft.nl/~klaus/

代码地址：https://github.com/a-nasikun/FastSpectrum

论文名称：**Physics-Based Differentiable Rendering**
**A Comprehensive Introduction** SIGGRAPH 2020 Course

项目地址：https://shuangz.com/courses/pbdr-course-sg20/

代码地址：https://github.com/BachiLi/diffrender_tutorials

论文名称：**Differentiable Vector Graphics Rasterization for Editing and Learning**

项目地址：https://people.csail.mit.edu/tzumao/diffvg/

代码地址：https://github.com/BachiLi/diffvg

### Hair

论文名称：**Physical Based Hair Simulation**

项目地址：https://pielet.github.io/

代码地址：https://github.com/pielet/Hair-DER

论文名称：Capturing hair assemblies fiber by fiber

项目地址：https://www.cs.cornell.edu/projects/HairCapture/

代码地址：..

### Fire&Flame&Smoke

论文名称：Animating Fire with Sound ACM SIGGRAPH 2011

项目地址：https://www.cs.cornell.edu/projects/Sound/fire/#source

代码地址：请看项目地址

论文名称：**SPGrid: A Sparse Paged Grid structure applied to adaptive smoke simulation**

项目地址：http://pages.cs.wisc.edu/~sifakis/project_pages/SPGrid.html

代码地址：http://pages.cs.wisc.edu/~sifakis/project_pages/SIGGRAPH_ASIA_Code_Release.zip

论文名称：**Resolving Fluid Boundary Layers with. Particle Strength Exchange and Weak Adaptivity(siggraph 2016)**

项目地址：https://zhxx1987.github.io/#cod

代码地址：https://github.com/zhxx1987/IVOCK

论文名称：A PPPM fast Summation Method for Fluids and beyond

项目地址：https://zhxx1987.github.io/#cod

代码地址：https://github.com/zhxx1987/PPPM_VORTEX_Bouyant_Flow_sample

论文名称：FLIP-liquidSolverwithAnMGPCGpressureSolver

项目地址：https://zhxx1987.github.io/#cod

代码地址：https://github.com/zhxx1987/tbb_liquid_amgpcg

构建指南：未成功，提示应用程序错误

论文名称：Interpolations of Smoke and Liquid Simulations

项目地址：http://www.ntoken.com/proj_ofblend.html

代码地址：https://github.com/thunil/ofblend

### FLIP&MPM

论文名称：Lagrangieyn-eulerian Multi-density topology optimization with the material point method

项目地址：https://www.math.ucla.edu/~minchen/

代码地址：https://github.com/xuan-li/LETO

项目简介：MATLAB代码

论文名称：IQ-MPM: An Interface Quadrature Material Point Method for Non-sticky Strongly Two-Way Coupled Nonlinear Solids and Fluids

AnisoMPM: Animating Anisotropic Damage Mechanics

项目地址：https://www.math.ucla.edu/~minchen/  

https://www.math.ucla.edu/~cffjiang/publications.html

代码地址：https://github.com/penn-graphics-research/ziran2020

论文名称：Hierarchical Optimization Time Integration for CFL-rate MPM Stepping

Silly Rubber: An Implicit Material Point Method for Simulating Non-equilibrated Viscoelastic and Elastoplastic Solids

项目地址：https://www.math.ucla.edu/~minchen/

代码地址：https://github.com/littlemine/HOT

论文名称：CD-MPM: Continuum Damage Material Point Methods for Dynamic Fracture Animation

项目地址：https://www.math.ucla.edu/~minchen/

代码地址：https://github.com/squarefk/ziran2019

论文名称：A Massively Parallel And Scalable Multi-GPU Material Point Method

项目地址：https://www.math.ucla.edu/~cffjiang/publications.html

代码地址：https://github.com/penn-graphics-research/claymore

论文名称：CD-MPM: Continuum Damage Material Point Methods For Dynamic Fracture Animation

Silly Rubber: An Implicit Material Point Method For Simulating Non-Equilibrated Viscoelastic And Elastoplastic Solids

项目地址：https://www.math.ucla.edu/~cffjiang/publications.html

代码地址：https://github.com/squarefk/ziran2019

### Light&Shadow

论文名称：**Differentiable Monte Carlo Ray Tracing through Edge Sampling**

项目地址：https://people.csail.mit.edu/tzumao/diffrt/

代码地址：https://github.com/BachiLi/redner

论文名称：**** Position-Free Monte Carlo Simulation for Arbitrary Layered BSDFs****

项目地址：http://miloshasan.net/

代码地址：https://github.com/tflsguoyu/layeredbsdf

论文名称：Implementing the Render Cache and the Edge-and-Point Image on Graphics Hardware

项目地址：https://www.cs.cornell.edu/~kb/projects/epigpu/

代码地址：请看项目地址

论文名称：**A Differential Theory of Radiative Transfer**

项目地址：https://shuangz.com/projects/dtrt-sa19/

代码地址：https://github.com/uci-rendering/dtrt

论文名称：****Bijective Projection in a Shell****

项目地址：https://cims.nyu.edu/gcl/daniele.html

代码地址：https://github.com/jiangzhongshi/bijective-projection-shell

论文名称：*Hyper-Reduced Projective Dynamics*
Christopher Brandt, Elmar Eisemann, Klaus Hildebrandt
ACM Transactions on Graphics 37(4) Article 80 (SIGGRAPH 2018)

项目地址：https://graphics.tudelft.nl/~klaus/

代码地址：https://graphics.tudelft.nl/~klaus/code/HRPD.zip

# 还有一些代码没下载的

代码实在太多啦，还没来得及看呢

http://graphics.berkeley.edu/resources/index.html

http://www.cs.cmu.edu/~kmcrane/ 几十份和几何变形论文代码，慢慢看吧...

https://www.cs.ubc.ca/~rbridson/

https://nmwsharp.com/ 

http://www.cs.columbia.edu/cg/creamystrand/

http://isgwww.cs.uni-magdeburg.de/graphics/#publications

https://www.cs.ubc.ca/~greif/Publications/Greif_Publications.html

http://people.csail.mit.edu/fredo/ 这些代码都是关于光照阴影的

https://ipc-sim.github.io/C-IPC/

https://www.math.ucla.edu/~minchen/

http://dannykaufman.io/

https://www.guandaoyang.com/

https://shuangz.com/publications.htm 这位大佬有许多光照的源码，不过本库并不太关注光照，所以就没怎么收录

https://people.csail.mit.edu/sbangaru/#publications 同样是光照的

https://cseweb.ucsd.edu/~ravir/

http://www-labs.iro.umontreal.ca/~bmpix/ 关于几何的很多很可爱

http://www.cs.columbia.edu/~cxz/publications.php

https://www.cs.cornell.edu/projects/Sound/

http://graphics.ucsd.edu/~henrik/

https://www.math.ucla.edu/~cffjiang/publications.html 一大堆物质点法的文章和代码，大佬中的大佬

https://www.math.ucla.edu/~minchen/

https://cims.nyu.edu/gcl/daniele.html 代码实在太多啦

https://igl.ethz.ch/code/ 这里也全是代码

https://webspace.science.uu.nl/~vaxma001/

https://zhipeiyan.github.io/

http://vcg.isti.cnr.it/~cignoni/

https://cs.uwaterloo.ca/~c2batty/

http://gamma.cs.unc.edu/software/#collision 碰撞检测的一大堆

http://www.tkim.graphics/

https://www.cwimd.nl/doku.php?id=codes:start

https://people.engr.tamu.edu/schaefer/research/index.html

http://www.nobuyuki-umetani.com/

https://koyama.xyz/#Publication

### 未成功打开

http://barbic.usc.edu/vega/form.php?VegaFEM-v3.1.zip

姓名要填 anonymous

https://sites.me.ucsb.edu/~fgibou/

http://physbam.stanford.edu/~mlentine/

https://research.adobe.com/person/qingnan-zhou/

https://cragl.cs.gmu.edu/seamless/

http://web.uvic.ca/~teseo/

https://w2.mat.ucsb.edu/qiaodong/

https://www.physicsbasedanimation.com/resources-courses/

### 没代码但仍然很不错的

http://www-sop.inria.fr/reves/publis/gdindex.php?pg=5

https://sgvr.kaist.ac.kr/~sungeui/

http://www.skinning.org/

### 未整理，代码实在太多了，其中很大一部分又运行不起来

http://www.cs.columbia.edu/cg/normalmap/index.html

http://www.cs.columbia.edu/cg/mb/ 运动模糊

http://www.cs.cornell.edu/projects/HarmonicFluids/ 水声

https://www.cs.cornell.edu/projects/Sound/proxy/ 噪声

https://cragl.cs.gmu.edu/pixelate/ 奥观海像素画，没卵用

http://www.cs.columbia.edu/cg/ACM/ 需要发邮件申请

http://www.cs.columbia.edu/cg/SC/ 衣物交互

https://cs.uwaterloo.ca/~c2batty/

https://koyama.xyz/project/ExampleBasedShapeMatching/index.html 弹性

https://koyama.xyz/project/color_unblending/ 图像处理

https://koyama.xyz/project/ViewDependentRodSimulation/index.html 弹性，unity

http://www.cs.columbia.edu/cg/multitracker/

http://www.cs.columbia.edu/cg/liquidhair/ 头发

http://poisson.cs.uwaterloo.ca/stokes/ 流体houdini 插件

http://www.cs.columbia.edu/cg/wetcloth/ 流体加布料

http://www.cs.ubc.ca/research/thinskinelastodynamics/ MAPLE code

A New MATLAB and Octave Interface to a Popular Magnetics Finite Element Code

https://cragl.cs.gmu.edu/singleimage/

http://masc.cs.gmu.edu/wiki/CVF

https://cragl.cs.gmu.edu/splineskin/

https://www.dgp.toronto.edu/projects/fast-rotation-fitting/

https://www.dgp.toronto.edu/projects/opening-and-closing-surfaces/

https://github.com/rarora7777/VolumetricTruss matlab,居然还要装一堆恶心的库

https://www.dgp.toronto.edu/projects/spectral-coarsening/ matlab

https://www.dgp.toronto.edu/projects/cubic-stylization/

https://cragl.cs.gmu.edu/selfsimilar/

http://www.cs.columbia.edu/cg/nested-cages/

https://www.dgp.toronto.edu/projects/rigid-body-compression/

https://odedstein.com/projects/hessians/

https://www.dgp.toronto.edu/projects/fast-winding-numbers/

https://www.dgp.toronto.edu/projects/paparazzi/

https://www.dgp.toronto.edu/projects/latent-space-dynamics/

https://www.dgp.toronto.edu/projects/michell/

https://www.cs.utah.edu/~ladislav/schuller13local/schuller13local.html

https://www.cs.utah.edu/~ladislav/jacobson12fast/jacobson12fast.html

https://www.cs.utah.edu/~ladislav/schuller13local/schuller13local.html 运行稍微需要一点魄力，因为这是matlab 和 c++ 联合编译的

https://lgg.epfl.ch/code.php

https://simtk.org/search/search.php?srch=&search=search&type_of_search=soft&sort=date&page=0 神奇的仿真软件下载网站，虽然大部分都没什么用

https://www.cs.utah.edu/~ladislav/liu16fast/liu16fast.html

https://www.cs.utah.edu/~ladislav/sahillioglu16detail/sahillioglu16detail.html

https://www.cs.utah.edu/~ladislav/liu17towards/liu17towards.html

https://www.cs.utah.edu/~ladislav/liu13fast/source_code_vs_build.zip

https://www.dgp.toronto.edu/projects/deconstructed-domains/

https://www.dgp.toronto.edu/projects/fast-rotation-fitting/

https://github.com/sgsellan/swept-volumes

http://graphics.stanford.edu/projects/ccclde/index.htm

https://graphics.cs.kuleuven.be/publications/phdniels/index.html

https://gitlab.com/thin-films-gpu/thin-films-app

https://web.cse.ohio-state.edu/~wang.3602/publications.html

http://www.cse.chalmers.se/~marcof/#publications_selected

http://give.zju.edu.cn/en/memberIntro/tm.html 碰撞检测很多

https://animation.rwth-aachen.de/person/1/

http://people.csail.mit.edu/aschulz/optCAD/index.html 200MB的代码...CAD

http://solidmechanics.org/FEA.htm 强大的有限元分析matlab

https://www.cs.columbia.edu/cg/codes.php

https://github.com/alecjacobson/geometry-processing-deformation