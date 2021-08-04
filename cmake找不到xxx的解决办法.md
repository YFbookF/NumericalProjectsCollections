以这个库https://github.com/rgoldade/ClothSim 为例

cmake首先报错CMAKE CANNOT FIND FREEGLUT

第一步：下载最新版freeglut并解压

第二步：cmake这个freeglut库

第三步：打开生成.sln，运行AllBUILD，这步用来生成静态库动态库之类的

第四步：打开原项目的FindFreeglut.cmake，它可能长这样

```
FIND_PATH(
  FREEGLUT_INCLUDE_DIR GL/freeglut.h
  ${CMAKE_INCLUDE_PATH}
  $ENV{include}
  ${OPENGL_INCLUDE_DIR}
  /usr/include
  /usr/local/include
)
FIND_LIBRARY(
  FREEGLUT_LIBRARY
  NAMES freeglut_staticd freeglutd
  PATH
    
    ${CMAKE_LIBRARY_PATH}
    $ENV{lib}
    /usr/lib
    /usr/local/lib
)

IF (FREEGLUT_INCLUDE_DIR AND FREEGLUT_LIBRARY)
   SET(FREEGLUT_FOUND TRUE)
ENDIF (FREEGLUT_INCLUDE_DIR AND FREEGLUT_LIBRARY)
```

那个freeglut.h就是它要找的文件。同理下面的freeglut.staticd.dll和freeglutd.dll也是它要找的文件。不要管那个GL/，它根本不会按照这个路径找。反正找不到文件就保错

这样就很好解决了，set两句

```
FIND_PATH(
  FREEGLUT_INCLUDE_DIR GL/freeglut.h
  ${CMAKE_INCLUDE_PATH}
  $ENV{include}
  ${OPENGL_INCLUDE_DIR}
  /usr/include
  /usr/local/include
)
FIND_LIBRARY(
  FREEGLUT_LIBRARY
  NAMES freeglut_staticd freeglutd
  PATH
    
    ${CMAKE_LIBRARY_PATH}
    $ENV{lib}
    /usr/lib
    /usr/local/lib
)
set(FREEGLUT_INCLUDE_DIR "E:/CollectionCodes/freeglut-3.2.1/freeglut-3.2.1/include/GL")
set(FREEGLUT_LIBRARY "E:/CollectionCodes/freeglut-3.2.1/freeglut-3.2.1/lib/Debug")
IF (FREEGLUT_INCLUDE_DIR AND FREEGLUT_LIBRARY)
   SET(FREEGLUT_FOUND TRUE)
ENDIF (FREEGLUT_INCLUDE_DIR AND FREEGLUT_LIBRARY)
```

困扰了一个月的问题终于解决了...

吐槽一句vcpkg下载个nuget下载半天下载不了

接下来是CANNOT FIND TBB

首先找github的oneapiTBB

然后在findtbb.cmake添加下面三句

```
set(TBB_INCLUDE_DIRS "E:/software/oneTBB-master/include")
set(TBB_LIBRARIES "E:/software/oneTBB-master/msvc_19.29_cxx_64_md_debug")

```

仍然报错？查看错误信息

```
CMake Error at cmake/FindTBB.cmake:290 (set_target_properties):
  set_target_properties called with incorrect number of arguments.
Call Stack (most recent call first):
  CMakeLists.txt:38 (find_package)
```

查看FindTBB.cmake第290行

```
IMPORTED_LOCATION              ${TBB_LIBRARIES_DEBUG}
```

原因是没有指定TBB_LIBRARIES_DEBUG，那么指定即可

```
set(TBB_LIBRARIES_DEBUG "E:/software/oneTBB-master/msvc_19.29_cxx_64_md_debug")
```

然后没有googletest的问题，github克隆一份过来，保证External/googletest下面找的到cmakelists.txt即可

运行继续报错，找不到Eigen库，直接手动查找

```
#include "../../../../eigen-master/Eigen/Dense"
#include "../../../../eigen-master/Eigen/Geometry"
#include "../../../../eigen-master/Eigen/Sparse"
```

然而还是失败了，找不到

#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

恕我直言，我用everything根本就没找到这两个东西