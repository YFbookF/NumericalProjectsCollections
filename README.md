# NumericalProjectsCollections

这个库专门用于搜集那些难于搜集的代码，这些代码大部分存在于个人网站或教授主页或项目主页上，网站很难找，并且随时会挂掉。

我发现，只看论文或书本，很可能绞尽脑汁也想不出到底是怎么实现的，自己写一个甚至可能连大方向都错了。但有了源码参考后，难度瞬间下降90%。如果还能把源码运行起来，那么难度已经约等于零了。至于那些没配套源码的论文或书本，就先暂时当它们不存在了。

顺便吐槽一下，有的论文表面看起来没开源代码，实际上打开项目页就能看到。论文通常由几个作者写成，有的作者在自己网页上没开源，但可能另一个作者却开源了同一篇论文的代码。

有的源码是matlab，运行起来很简单。而有的c++代码依赖各种库，不容易运行起来。而后者才是占这些源码中的多数。很多代码都是论文中实现的简化版。

我学习的方法主要是，把这些c++的,matlab的,julia的,fortran的代码，手工翻译成python代码，也就是用python重写一遍，这样就能理解代码的50%，剩下的数学原理之后再慢慢思考。

另外有的项目代码实在太多了，一下子看懂不可能。于是只好化整为零，把原代码的几小部分实现出来。卡住超过半天的话立马换下一个项目。反正项目多的是。

预计这个项目将会很大，所以只收录那些看起来随时会消失的源码。至于看起来经常更新的项目主页，或github上的代码就仅仅给出链接了。

另一个图形学源码收录网站http://kesen.realtimerendering.com/

PaperWithCodes虽然收录了很多论文的源码，但大多都是机器学习方面的。

持续更新维护中...

# 本仓库收录代码【重新整理】

### 【Deformation】

**[dynamicDeformables]**

论文名称：Dynamic Deformables: Implementation and Production Practicalities

项目网站：http://www.tkim.graphics/DYNAMIC_DEFORMABLES/

值得一看：

源代码：已收录

学习指南：*SIGGRAPH Courses, 2020*。先看很不错的Courses Notes再看代码。文风形式很有趣。详细地推导了St. Venant-Kirchhoff 和很多公式。

构建指南：matlab

**[biharmonic]**

论文名称：Biharmonic distance

项目网站：http://www.hakenberg.de/diffgeo/differential_geometry.htm#Spin

源代码：已收录matlab

**[spin]**

论文名称：Spin Transformations of Discrete Surfaces

项目网站：http://www.hakenberg.de/diffgeo/differential_geometry.htm#Spin

源代码：已收录matlab

**[interactive]**

论文名称：Interactive Deformation Using Modal Analysis with Constraints

项目网站：http://graphics.berkeley.edu/papers/Hauser-IDU-2003-06/

值得一看：

源代码：已收录,6MB

学习指南：还没来得及学

构建指南：c++

**[]**

论文名称：Strain Limiting for Clustered Shape Matching

项目网站：https://cal.cs.umbc.edu/Papers/Bargteil-2014-SLF/

值得一看：

源代码：https://github.com/benjones/strainLimitingForClusteredShapeMatching

学习指南：还没来得及学

### 【Geometry】

**[Developability]**

论文名称： Developability of Triangle Meshes

项目网站：http://www.cs.cmu.edu/~kmcrane/Projects/DiscreteDevelopable/

源代码：https://github.com/odedstein/DevelopabilityOfTriangleMeshes

构建指南：c++，把本库cmake文件“add_subdirectory("${LIBIGL_INCLUDE_DIR}/../shared/cmake" "libigl")”这一句改成“add_subdirectory("${LIBIGL_INCLUDE_DIR}/../cmake" "libigl")”，然后手工下载libigl库。编译成功但没运行起来

**[trival]**

论文名称： Developability of Triangle Meshes

项目网站：http://www.cs.cmu.edu/~kmcrane/Projects/TrivialConnections/

**[spin]**

论文名称： Spin Transformations of Discrete Surfaces

项目网站：http://www.cs.cmu.edu/~kmcrane/Projects/SpinTransformations/

**[]**

论文名称：AMS Short Course on Discrete Differential Geometry

项目网站：http://geometry.cs.cmu.edu/ddgshortcourse/index.html

构建指南：javascript，一键运行

**[]**

论文名称：Hexahedral Quality Evaluation via SOS Relaxations

项目网站：https://people.csail.mit.edu/jsolomon/#research

源代码：https://github.com/zoemarschner/SOS-hex

**[]**

论文名称：Dynamical Optimal Transport on Discrete Surfaces

项目网站：https://people.csail.mit.edu/jsolomon/#research

源代码：https://github.com/HugoLav/DynamicalOTSurfaces

**[]**

论文名称：Isometry-Aware Preconditioning for Mesh Parameterization

项目网站：https://people.csail.mit.edu/jsolomon/#research

源代码：https://github.com/sebastian-claici/AKVFParam

**[]**

论文名称："Earth Mover’s Distances on Discrete Surfaces." SIGGRAPH 2014,

项目网站：https://people.csail.mit.edu/jsolomon/#research

源代码：已收录matlab

### 【Graphics】

**[proceduralShadrBandlimiting]**

论文名称： Approximate Program Smoothing Using Mean-Variance Statistics, with Application to Procedural Shader Bandlimiting

项目网站：https://www.cs.princeton.edu/~yutingy/docs/eg_2018.html

值得一看：

源代码：https://github.com/yyuting/approximate_program_smoothing

构建指南：c++

### 【FiniteElement】

**GUOJIATONG**

《有限元与MTALAB程序设计程序文件》这本书的源代码

**[Prof_C_CarStencen]**

论文名称：****

项目网站： https://www2.mathematik.hu-berlin.de/~cc/cc_homepage/software/software.shtml

值得一看：

源代码：本仓库已收录

构建指南：

学习指南：

**[NodalGalerkin_Book]**

论文名称：Nodal Discontinuous Galerkin Methods Algorithms, Analysis, and Applications by Jan S. Hesthaven, Tim Warburton

项目网站： http://www.tkim.graphics/COMPRESSING/source.html

值得一看：

源代码：本仓库已收录

构建指南：matlab

学习指南：书本的配套源代码，本书有中译本《交点间断Galerkin方法：算法，分析和应用》

**[jburkardt]**

项目网站： https://people.sc.fsu.edu/~jburkardt/m_src/

值得一看：这个网站收录了相当多的代码，比较实用的包括一维谱方法，Galerkin方法

源代码：本仓库已收录

构建指南：matlab

**[galerkin0]**

文章地址：Notes on Galerkin methods for incompressible flow simulation. 

项目网站： https://www.mcs.anl.gov/~fischer/me528/

值得一看：配套讲义也在这个网页上

源代码：本仓库已收录

构建指南：matlab



### 【FireFlameSmoke】

**[compress]**

论文名称：**Compressing Fluid Subspaces**

项目网站： http://www.tkim.graphics/COMPRESSING/source.html

值得一看：用三维余弦变换压缩三维数据，waveletNoise其实就是waveTurbulenceNoise

源代码：本仓库已收录

构建指南：c++，未成功

学习指南：

### 【FrontTracking】

**[skin]**

论文名称：A Level-set Method for Skinning Animated Particle Data

项目网站： https://cal.cs.umbc.edu/Papers/Bhattacharya-2015-ALM/

值得一看：

源代码：本仓库已收录

构建指南：只有很少的c++文件

学习指南：

### 【Image】

**[melding]**

论文名称：**Image Melding: Combining Inconsistent Images**
**using Patch-based Synthesis**

项目网站：https://web.ece.ucsb.edu/~psen/melding

值得一看：

源代码：本仓库已收录

构建指南：matlab

[NeuralTextureSynthesis]

论文名称：**Image Melding: Combining Inconsistent Images**
**using Patch-based Synthesis**

项目网站：**Stable and Controllable Neural Texture Synthesis and Style Transfer Using Histogram Losses**

值得一看：

源代码：https://github.com/pierre-wilmot/NeuralTextureSynthesis

构建指南：python

### 【LightShadow】

[photonbeam]

论文名称：Photon Beam Diffusion: A Hybrid Monte Carlo Method for Subsurface Scattering

项目网站：**https://graphics.pixar.com/library/**

值得一看：

源代码：已收录

构建指南：matlab

### 【Math】

[Eigen]

项目网站：https://eigen.tuxfamily.org/index.php?title=Main_Page

值得一看：被很多c++项目使用的超受欢迎的数学库，里面有很多数值算法的实现

源代码：gitlab

构建指南：c++，无外部依赖，cmake一遍成功

[]

书籍名称：numerical recipes in c

值得一看：附源代码的书都是好书

[levenberg]

文章名称：The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems  

项目简介：The Levenberg-Marquardt algorithm的matlab实现

### 【Physics】

**[PhysicsBasedAnimation]**

论文名称：An Introduction to Physics-based Animation

项目网站：https://cal.cs.umbc.edu/Courses/PhysicsBasedAnimation/

值得一看：显式有限元，隐式有限元，及对比。

源代码：本仓库已收录2019版的。2018版需自行下载。

学习指南：比较简单，已写成相应版本的python代码。

构建指南：c++，只用了Eigen库，直接拉到Eigen库中去。读取文件的文件名用绝对路径，.json里读取网格的路径也要改成绝对路径。花了一个小时终于搞定。

**[]**

论文名称：**ARCSim: Adaptive Refining and Coarsening Simulator**

项目网站：http://graphics.berkeley.edu/resources/ARCSim/

值得一看：开源的小巧物理模拟软件

[tinydiff]

代码地址：https://github.com/google-research/tiny-differentiable-simulator

简介：GoogleSearch的可微分物理，不过看不懂究竟在搞啥

**[largeccd]**

论文名称：**A Large Scale Benchmark and an Inclusion-Based Algorithm for Continuous Collision Detection**

源代码：https://github.com/Continuous-Collision-Detection/Tight-Inclusion

**[exact]**

论文名称：**Efficient Geometrically Exact Continuous Collision Detection**

项目网站：https://www.cs.ubc.ca/~rbridson/

值得一看：

源代码：收录

构建指南：

**[simplediff]

代码地址：https://github.com/BachiLi/diffrender_tutorials

简介：单c++文件，无外部依赖。容易运行。可微分光线追踪。仓库作者的另一仓库也是可微分物理

### 【Plastic】

**[localRemehing]**

论文名称：**Dynamic Local Remeshing for Elastoplastic Simulation**

项目网站：http://graphics.berkeley.edu/papers/Wicke-DLR-2010-07/

值得一看：

源代码：已收录

构建指南：c++，构建非常简单，CMake一遍过，只需解决win10下没用sys/time.h的问题。然而还没解决

**[sharpKevlinlets]**

论文名称：Sharp Kelvinlets: Elastic Deformations with Cusps and Localized Falloffs

Dynamic Kelvinlets: Secondary Motions based on Fundamental Solutions of Elastodynamics

项目网站：http://fernandodegoes.org/ 

值得一看：代码挺简单的，仅仅是平板变个形。如何做到文章里的效果还需要思考。以及如何把顶点数据输出成obj三维文件。

源代码：已收录。此库已包含了dynamicKevlinlets 的代码

构建指南：只用了Eigen库。外加从网上下载个getopt.h和getopt.c即可解决编译问题。运行输出成功。

[**anisotropicHyperelasticity**]

论文名称：Anisotropic Elasticity for Inversion-Safety and Element Rehabilitation

项目网站：https://graphics.pixar.com/library/indexAuthorFernando_de_Goes.html

值得一看：first Piola-Kirchhoff stress解析解和数值解

源代码：已收录

构建指南：matlab

### 【SkinSkeletonBone】

[**NeoHookeanFlesh**]

论文名称：Stable Neo-Hookean Flesh Simulation

项目网站： https://graphics.pixar.com/library/indexAuthorFernando_de_Goes.html

值得一看：c++ 项目结构和注释写得很简洁易懂。

源代码：本仓库已收录

构建指南：只用了Eigen库，无其他依赖。一遍构建成功。

学习指南：代码文件居然用模型网格的是立方体，比某些二维项目好多了。代码所作的就是拉伸了这个立方体，导致立方体中间截面变小，就像在拉面团。记得看Readme.md手动写命令参数。改编成python很有挑战性。目前写了一半，弄不懂的地方太多了。





### 【WaterFluidFlow】

**[eigenFluid]**

论文名称：Scalable laplacian eigenfluids.

项目网站： http://www.tkim.graphics/    

值得一看：用余弦傅里叶变换解泊松方程[未完成]，用modified imcomplete cholesky decomposition 解泊松方程[python]。粒子没有速度这个属性，需要前进的时候，直接插值网格。

源代码：本仓库已收录

构建指南：未成功，GLUT_glut_LIBRARY找不到

学习指南：代码文件多而复杂。未完成。主体看不懂建议直接看LaplaicanEigen，逻辑相似而更简洁。

相似代码：LaplacianEigen 

**[LaplacianEigen]**

论文名称：Fluid Dynamics using Laplacian Eigenfunctions

项目网站： http://www.dgp.toronto.edu/~tyler/fluids/

值得一看：涡量输运方程求解

源代码：本仓库已收录

学习指南：主代码只有几百行，非常容易。主体部分完成。

构建指南：java，不打算构建

[**Clebsch**]

论文名称：Clebsch Gauge Fluid

项目网站： https://y-sq.github.io/proj/clebsch_gauge_fluid/

值得一看：

源代码：本仓库已收录，不过少了许多文件

[**voronoi**]

论文名称：Matching Fluid Simulation Elements to Surface Geometry

项目网站： http://www.cs.ubc.ca/labs/imager/tr/2010/MatchingSimulationToSurface/BBB2010.html

值得一看：

源代码：本仓库已收录

[**acurrateviscosity**]

论文名称：Accurate Viscous Free Surfaces for Buckling, Coiling, and Rotating Liquids

项目网站： http://www.cs.ubc.ca/labs/imager/tr/2008/Batty_ViscousFluids/

值得一看：

源代码：本仓库已收录

http://www.cs.columbia.edu/cg/surfaceliquids/

http://www.cs.columbia.edu/cg/surfaceliquids/code/

# 本仓库收录的代码

### Deformation

**subspace**

Optimizing Cubature for Efficient Integration of Subspace Deformationshttps://www.cs.cornell.edu/~stevenan。构建难度：高。这是个c++项目，有十几个文件，却连main函数都没有。







**Analytic Eigensystems**

Analytic Eigensystems for Isotropic Distortion Energieshttps://graphics.pixar.com/library/indexAuthorFernando_de_Goes.html



**boundedBiharmonic**

Bounded Biharmonic Weights for Real-Time Deformationhttps://igl.ethz.ch/projects/bbw/

**shapeAware**

Smooth Shape-Aware Functions with Controlled Extremahttps://igl.ethz.ch/projects/monotonic/

**Practicalities**

http://www.tkim.graphics/DYNAMIC_DEFORMABLES/**Dynamic Deformables:
Implementation and Production Practicalities**
*SIGGRAPH Courses, 2020*

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

https://github.com/tflsguoyu/layeredbsdf

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

### New

http://www.oliviermercier.com/

https://web.cse.ohio-state.edu/~wang.3602/publications.html

http://visualcomputing.ist.ac.at/publications/2020/HYLC/

https://elrnv.com/projects/frictional-contact-on-smooth-elastic-solids/

https://shuangz.com/publications.htm#ctcloth-sg16

https://git.ist.ac.at/gsperl/HYLC/-/tree/master/solver/src/simulation/statics/constraints