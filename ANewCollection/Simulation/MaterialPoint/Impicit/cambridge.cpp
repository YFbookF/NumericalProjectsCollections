https://github.com/bssrdf/MPM-Implicit/blob/develop/Implicit/cartesian/src/main.cpp
// *****************************************************************************
//                  2D / 3D Implicit/Explicit Material Point Method
//                      Author: (Implicit MPM) Shyamini Kularathna
//                             University Of Cambridge
//              Author: (Explicit MPM) Samila Bandara and Krishna Kumar
//                             University of Cambridge
//
// Version: 1.0
//
// Dependecy: CORLIB - Template Library by Fehmi Cirak, University of Cambridge
//            BOOST, TCLAP, OPENMP , EIGEN and VTK
// *****************************************************************************

#include <omp.h>
#include <sys/stat.h>
#include <time.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <parallel/algorithm>
#include <sstream>
#include <string>
#include <vector>

// Boost Header Files
#include <boost/algorithm/string.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

// Eigen Header Files
#include <Eigen/Dense>

// TCLAP for Input Argument Parsing
#include "tclap/CmdLine.h"

// CORLIB Header files - Fehmi Cirak
#include "corlib/PropertiesParser.hpp"
#include "corlib/fuzzyEqual.hpp"
#include "corlib/linalg.hpp"
#include "corlib/Sensor.hpp"
#include "corlib/verify.hpp"
#include "corlib/Quadrature.hpp"
#include "corlib/IntegrandAdaptor.hpp"
#include "corlib/Integrator.hpp"
#include "corlib/SystemSolve.hpp"
#include "corlib/DofFunctors.hpp"
#include "corlib/MatrixAssembler.hpp"
#include "corlib/VectorAssembler.hpp"
#include "corlib/BodyForce.hpp"

// MPM Material Header Files
#include "cartesian/material/MaterialBaseMpm.hpp"
#include "cartesian/material/MaterialContainerMpm.hpp"

// MPM Header File
#include "cartesian/src/MpmItems.hpp"

// Function Declaration
// String Function to Create Output FileNames
std::string ResultStringFn(std::string, int, int);

//******************************************************************************
//******************************************************************************

int main(int argc, char* argv[]) {

// MPM 2D/3D is defined in MpmItems.hpp
#ifdef _MPM2D_
    std::cout << "\n\tTwo-Dimensional Material Point Method" << std::endl;
    std::cout << "\tUniversity of Cambridge - Version 0.1.3 \n";
#else
    std::cout << "\n\tThree-Dimensional Material Point Method" << std::endl;
    std::cout << "\tUniversity of Cambridge - Version 0.1.3 \n" << std::endl;
#endif

    // User input for folder name which includes inputFiles
    std::string inputFileName, inputFolderName;
    std::string cWorkingDir;

    // TCLAP input argument parsing inside exception handling
    try {

      TCLAP::CmdLine cmd("Material Point Method - Bingham Model (University of Cambridge)", ' ', "0.1.3");

      TCLAP::ValueArg<std::string> foldernameArg("f", "inputFolder", "Input Folder Name", true, "", "Input Folder Name");
      cmd.add(foldernameArg);

      TCLAP::ValueArg<std::string> cwdArg("d", "currentDir", "Current Working Directory", false, "", "Working_Directory");
      cmd.add(cwdArg);

      cmd.parse(argc, argv);
      inputFolderName = foldernameArg.getValue();
      cWorkingDir = cwdArg.getValue();

    }
    catch (TCLAP::ArgException &e) {  // catch any exceptions
      std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }
    
    // Specify input and current working directory
    boost::algorithm::trim(cWorkingDir);
    if (cWorkingDir.empty()) cWorkingDir = "./bin/";

    boost::algorithm::trim(inputFolderName);
    std::cout << "\nLoading input files from: " << cWorkingDir + inputFolderName << std::endl;

//------------------------------------------------------------------------------
                 // Declare input files and input variables
    
    // Declare variables for input stream parsing
    std::string inputMeshFileName, inputSubMeshFileName, constraintsFileName,
                particleMaterialFileName, inputSoilParticleFileName,
                inputWaterParticleFileName, materialFileName, outputFileName,
                initStressSoilPFileName, tractionsSoilPFileName;

    // Declare input variables
    double dt, boundaryMiu;
    double soilParticleSpacingX, soilParticleSpacingY, soilParticleSpacingZ;
    double soilParticleSpacing = 0.;
    std::vector<double> soilParticleSpacing_;
    unsigned int numberOfSteps, numberOfSubStepsOS;
    bool gravityFlag;
    unsigned writeCountOS = 0;

//------------------------------------------------------------------------------
            // Property parsers for input files and input variables
    
    // Property parser for input files
    corlib::PropertiesParser *prop = new corlib::PropertiesParser;
    prop->registerPropertiesVar("inputMeshFileName", inputMeshFileName);
    prop->registerPropertiesVar("inputSubMeshFileName", inputSubMeshFileName);
    prop->registerPropertiesVar("constraintsFileName", constraintsFileName);
    prop->registerPropertiesVar("inputSoilParticleFileName", inputSoilParticleFileName);
    prop->registerPropertiesVar("initStressSoilPFileName", initStressSoilPFileName);
    prop->registerPropertiesVar("tractionsSoilPFileName", tractionsSoilPFileName);
    // prop->registerPropertiesVar("inputWaterParticleFileName",inputWaterParticleFileName);
    prop->registerPropertiesVar("materialFileName", materialFileName);
    prop->registerPropertiesVar("particleMaterialFileName", particleMaterialFileName);

    // Property parser for variables
    prop->registerPropertiesVar("gravityFlag", gravityFlag);
    prop->registerPropertiesVar("boundaryFrictionMiu", boundaryMiu);
    prop->registerPropertiesVar("soilParticleSpacing", soilParticleSpacing);
    prop->registerPropertiesVar("soilParticleSpacingX", soilParticleSpacingX);
    prop->registerPropertiesVar("soilParticleSpacingY", soilParticleSpacingY);
    prop->registerPropertiesVar("soilParticleSpacingZ", soilParticleSpacingZ);
    prop->registerPropertiesVar("timeInterval", dt);
    prop->registerPropertiesVar("numberOfSteps", numberOfSteps);
    prop->registerPropertiesVar("numberOfSubStepsOS", numberOfSubStepsOS);
    
//------------------------------------------------------------------------------
                    // read variables from the input.dat file

    inputFileName = cWorkingDir + inputFolderName +"/inputFiles/input.dat";
    std::ifstream inputFile(inputFileName.c_str());
    FSI_VERIFY(inputFile.is_open());
    prop -> readValues(inputFile);
    delete prop;
    inputFile.close();
    
    // This should be modified to be read as a vector instead of 3 doubles in the input file
    if (soilParticleSpacing > 0.) {
      std::cout << "\nUniform Soil Particle Spacing is defined as:" << soilParticleSpacing << std::endl;
      if (mpm::DIM == 2) {
        soilParticleSpacingX = soilParticleSpacingY = soilParticleSpacing;
        soilParticleSpacingZ = 1.;
      } else {
        soilParticleSpacingX = soilParticleSpacingY = soilParticleSpacingZ = soilParticleSpacing;
      }
    }
    soilParticleSpacing_.push_back(soilParticleSpacingX);
    soilParticleSpacing_.push_back(soilParticleSpacingY);
    soilParticleSpacing_.push_back(soilParticleSpacingZ);
 
//---------------------------------------------------------------------------   
                             // read material

    mpm::material::MaterialContainerMpm* mc = mpm::material::MaterialContainerMpm::instance();
    materialFileName = cWorkingDir + materialFileName;
    std::ifstream mf(materialFileName.c_str());
    FTL_VERIFY(mf.is_open());  // read verify.hpp file to include new methods
    mc -> readMaterialStream(mf);
    mf.close();

//------------------------------------------------------------------------------
   // Typedef and defining MPM Dimension, Degree Of Freedom and loading template

    typedef mpm::MpmItems<mpm::DIM, mpm::DOF, mpm::NUMNODES, mpm::SHAPEFN> MpmItemsType;
    typedef mpm::MeshMpm<MpmItemsType> MeshType;
    typedef MeshType* MeshPtr;

    typedef mpm::SubMeshMpm<MpmItemsType> SubMeshType;
    typedef mpm::ElementMpm<MpmItemsType> ElementType;
    typedef mpm::NodeMpm<MpmItemsType> NodeType;
    typedef mpm::SolverMpm<MpmItemsType> SolverType;
    
    typedef mpm::ParticleSoilMpm<MpmItemsType> SoilParticleType;
    typedef mpm::ParticleCloudMpm<MpmItemsType, SoilParticleType> ParticleCloudSoilType;
    typedef ParticleCloudSoilType* ParticleCloudSoilPtr;
    typedef ParticleCloudSoilType::ParticleVec SoilParticleVec;
    
    typedef mpm::ParticleWaterMpm<MpmItemsType> WaterParticleType;
    typedef mpm::ParticleCloudMpm<MpmItemsType, WaterParticleType> ParticleCloudWaterType;
    typedef ParticleCloudWaterType* ParticleCloudWaterPtr;
    typedef ParticleCloudWaterType::ParticleVec WaterParticleVec;
    
    typedef mpm::material::MaterialBaseMpm MaterialType;
    typedef mpm::material::MaterialBaseMpm* MaterialTypePtr;
    
    typedef boost::numeric::ublas::bounded_vector<double, mpm::DOF> VecDof;
    typedef boost::numeric::ublas::bounded_vector<double, mpm::DIM> VecDim; 
    typedef boost::numeric::ublas::bounded_vector<double, 6> Vec6x1;
//--------------------------------------------------------------------------
                // Opening the files for input parameters
    
    // Defining file names for input parameters
    inputMeshFileName = cWorkingDir + inputMeshFileName;
    inputSubMeshFileName =  cWorkingDir + inputSubMeshFileName;
    constraintsFileName =  cWorkingDir + constraintsFileName;
    inputSoilParticleFileName =  cWorkingDir + inputSoilParticleFileName;
    initStressSoilPFileName =  cWorkingDir + initStressSoilPFileName;
    tractionsSoilPFileName =  cWorkingDir + tractionsSoilPFileName;
    particleMaterialFileName = cWorkingDir + particleMaterialFileName;
    
    // Defining a stream name for each input variable
    std::ifstream inputMeshStream(inputMeshFileName.c_str());
    std::ifstream inputSubMeshStream(inputSubMeshFileName.c_str());
    std::ifstream constraintsStream(constraintsFileName.c_str());
    std::ifstream inputSoilParticleStream(inputSoilParticleFileName.c_str());
    std::ifstream initStressSoilPStream(initStressSoilPFileName.c_str());
    std::ifstream tractionsSoilPStream(tractionsSoilPFileName.c_str());
    std::ifstream particleMaterialStream(particleMaterialFileName.c_str());
    // std::ifstream inputWaterParticleStream(inputWaterParticleFileName.c_str());
    
    // Verify if the input files have been read properly
    FSI_VERIFY(inputMeshStream.is_open());
    FSI_VERIFY(inputSubMeshStream.is_open());
    FSI_VERIFY(constraintsStream.is_open());
    FSI_VERIFY(inputSoilParticleStream.is_open());
    FSI_VERIFY(initStressSoilPStream.is_open());
    FSI_VERIFY(tractionsSoilPStream.is_open());
    FSI_VERIFY(particleMaterialStream.is_open());
    // FSI_VERIFY(inputWaterParticleStream.is_open());
 
//-----------------------------------------------------------------------------                           // Local Time and Runtime
 
    double beginning_of_time, loop_start_time;
    loop_start_time = beginning_of_time = omp_get_wtime(); 
    time_t rawtime;
    time(&rawtime);
    std::cout << "\nAnalysis started at: " << asctime(localtime(&rawtime)) << std::endl;
    std::cout << "Model properties: dimension = " << mpm::DIM << " , dof = " << mpm::DOF  << " , Nodes = " << mpm::NUMNODES <<std::endl;

//------------------------------------------------------------------------------
                             // Setting Gravity 

    boost::numeric::ublas::bounded_vector<double, mpm::DIM> G;
    G.clear();
    if (gravityFlag) G[mpm::DIM - 1] = -9.81;
    
//------------------------------------------------------------------------------
             // Defining objects for mesh, particles and the solver

    // Defining the mesh from the input stream
    MeshType mesh(inputMeshStream, inputSubMeshStream);
    MeshPtr meshPtr = &mesh;

    // Defining the material points from the input stream
    ParticleCloudSoilType particleCloudSoil(inputSoilParticleStream);   
    ParticleCloudSoilPtr particleCloudSoilPtr = &particleCloudSoil;
    
    // Definig the solver type (Explicit or Implicit)
    SolverType solver;

//------------------------------------------------------------------------------
          // Read input files and save initial data of particles and nodes

    // Read and save constrains for velocity & acceleration with friction at nodes
    mesh.readConstraintsExplicit(constraintsStream);
    
    // obtain the list of pointers to the materials
    std::vector<MaterialTypePtr> materials = mc -> getMaterials();
    
    // Associate Particles to Materials
    particleCloudSoil.associateParticleToMaterial(particleMaterialStream);

    // cache material to all elements
    particleCloudSoil.iterateOverParticles(boost::bind(&SoilParticleType::cacheMaterial, _1, materials, soilParticleSpacing_));
    
    // Read initial stresses of particles
    particleCloudSoil.setInitialStressParticles(initStressSoilPStream);
    
    // Read traction pressures of particles
    particleCloudSoil.readTractionBoundary(tractionsSoilPStream);

//--------------------------------------------------------------------------    
                // Writing Results and Specifying Output Directory

    std::string ResultsDir;
    ResultsDir = cWorkingDir + inputFolderName  + "/Results/";
    std::cout << "\nResults will be stored in: " << ResultsDir.c_str() << std::endl;
    int status = mkdir(ResultsDir.c_str(), 0777);
    std::string ResultsVelocity, ResultsStress, ResultsStrain, ResultsRateOfStrainI2,ResultsPressure;
    int digits = log10(numberOfSteps) + 1;

//*************************************************************************
//--------------------------------------------------------------------------    
               // Main Loop - iterates over number of steps

    for (unsigned i = 0; i <= numberOfSteps; i++) {

//-----------------------------------------------------------------------------
                           // Write Output
      if (i == writeCountOS * numberOfSubStepsOS) {
        // Write Velocity
        ResultsVelocity = ResultsDir + ResultStringFn("Velocity", i, numberOfSteps);
        std::ofstream outputFileVelocity(ResultsVelocity.c_str());
        particleCloudSoil.writeParticleCloudVelocityData(i, outputFileVelocity);

        // Write Stress
        ResultsStress = ResultsDir + ResultStringFn("Stress", i, numberOfSteps);
        std::ofstream outputFileStress(ResultsStress.c_str());
        particleCloudSoil.writeParticleCloudStressData(i, outputFileStress);

        // Write Strain
        ResultsStrain = ResultsDir + ResultStringFn("Strain", i, numberOfSteps);
        std::ofstream outputFileStrain(ResultsStrain.c_str());
        particleCloudSoil.writeParticleCloudStrainData(i, outputFileStrain);

        // Write Strain Rate
        ResultsRateOfStrainI2 = ResultsDir + ResultStringFn("RateOfStrainI2_", i, numberOfSteps);
        std::ofstream outputFileRateOfStrainI2(ResultsRateOfStrainI2.c_str());
        outputFileRateOfStrainI2.setf(std::ios::scientific);
        particleCloudSoil.writeParticleCloudRateOfStrainI2Data(i, outputFileRateOfStrainI2);

        // Display status on terminal
        std::cout << "\nStep(s): " << std::setw(digits) << std::right << i << " / " << std::setw(digits) << std::right << numberOfSteps << " runtime: " << std::setw(6) << std::right << (omp_get_wtime() - loop_start_time) << " / " << std::setw(6) << std::left << (omp_get_wtime() - beginning_of_time) << " s" << std::endl;
        loop_start_time = omp_get_wtime();

        // Close output files
        outputFileVelocity.close();
        outputFileStress.close();
        outputFileStrain.close();
        outputFileRateOfStrainI2.close();
        writeCountOS++;
      }

//------------------------------------------------------------------------------
                  // locate particles in mesh

      mesh.locateParticlesAndElements(particleCloudSoilPtr);
      // mesh.printParticlesInElement();
// locate nodes connected to each other
      mesh.addNodesToNodes();

//-----------------------------------------------------------------------------
         // calculate and store shape functions and gradient shape functions

      particleCloudSoil.iterateOverParticlesParallel(boost::bind(&SoilParticleType::evaluateShpfunAndGradShpfunHorMesh, _1));

//------------------------------------------------------------------------------
                        // calculate nodal mass

      particleCloudSoil.iterateOverParticles(boost::bind(&SoilParticleType::assignMassToNodes, _1));

//------------------------------------------------------------------------------
                     // calculate nodal velocity

      particleCloudSoil.iterateOverParticles(boost::bind(&SoilParticleType::assignMomentumToNodes, _1));
      mesh.iterateOverNodesOfSoilP(boost::bind(&NodeType::computeSoilVelocity, _1));

//------------------------------------------------------------------------------
              // calculate external forces at nodes

       // apply body forces to nodes
      particleCloudSoil.iterateOverParticles(boost::bind(&SoilParticleType::assignBodyForceToNodes, _1, G));

      // apply pressure forces to nodes
      particleCloudSoil.iterateOverParticles(boost::bind(&SoilParticleType::assignPressureToNodes, _1));

               // apply traction forces to nodes
      // particleCloudSoil.iterateOverParticles(boost::bind(&SoilParticleType::assignTractionsToNodes, _1, soilParticleSpacing_));

//------------------------------------------------------------------------------
                           // Implicit
              // Calculate and solve matrix to find the velocities at nodes

      solver.evaluateMatrix(meshPtr, dt);

//------------------------------------------------------------------------------
                           // Implicit
              // Compute and update velocity at nodes
      bool convergeStatus = solver.computeNodalVelocity();
      if (!convergeStatus){
          std::cout << "Unable to converge within the maximum number of iterations" << std::endl;
          break;
      }


//------------------------------------------------------------------------------
                          // Explicit

/*      
      // compute internal forces at nodes
      particleCloudSoil.iterateOverParticles(boost::bind(&SoilParticleType::computeInternalForceAtNodes, _1));
      
      // compute acceleration and update velocity at nodes
      mesh.iterateOverNodesOfSoilP(boost::bind(&NodeType::computeSoilAccelerationAndVelocity, _1, dt, boundaryMiu));
*/
//------------------------------------------------------------------------------
                   // Update the position of MPM points

      particleCloudSoil.iterateOverParticlesParallel(boost::bind(&SoilParticleType::updateSoilParticle, _1, dt));

//------------------------------------------------------------------------------
                                  // Explicit
/*
      // calculate nodal velocity
      particleCloudSoil.iterateOverParticles(boost::bind(&SoilParticleType::assignMomentumToNodes, _1));
      mesh.iterateOverNodesOfSoilP(boost::bind(&NodeType::computeSoilVelocity, _1));
     // calculate strain of soil particles
      particleCloudSoil.iterateOverParticlesParallel(boost::bind(&SoilParticleType::computeStrain, _1, dt));
      
      // calculate strain at the center of the elements
      particleCloudSoil.iterateOverParticlesParallel(boost::bind(&SoilParticleType::computeVolStrainIncCenter, _1, dt));
      
      // calculate strain of particles by B-Bar method
      particleCloudSoil.iterateOverParticlesParallel(boost::bind(&SoilParticleType::computeStrainBBar, _1, dt));
      // calculate stress of soil particles
      particleCloudSoil.iterateOverParticlesParallel(boost::bind(&SoilParticleType::computeStress, _1, particleCloudSoilPtr));
*/

//------------------------------------------------------------------------------
                   // Update phase averaged density of soil

      particleCloudSoil.iterateOverParticles(boost::bind(&SoilParticleType::updateSoilDensity, _1));

//------------------------------------------------------------------------------
                         // Initialise everything

      // intialise data inside mesh, elements, nodes
      mesh.initialise();
      solver.initialise();
    
      // clear pointers of nodes and elements
      particleCloudSoil.iterateOverParticles(boost::bind(&SoilParticleType::clearMeshDetails, _1));

    }

//------------------------------------------------------------------------------
//******************************************************************************
    
    // Close input streams
    inputMeshStream.close();
    inputSubMeshStream.close();
    constraintsStream.close();
    inputSoilParticleStream.close();
    initStressSoilPStream.close();
    
    // clear material container
    mc -> destroy();

//------------------------------------------------------------------------------
                  // Display status at the end of analysis

    time(&rawtime);
    std::cout << "\nAnalysis completed at: " << asctime(localtime(&rawtime)) << std::endl << "Total analysis runtime: " << omp_get_wtime() - beginning_of_time << " s" << std::endl;
    return 0;
}

//------------------------------------------------------------------------------
//******************************************************************************
//******************************************************************************

// String Function to Create Output FileNames (eg. Velocity0000*.vtk, Stress00*.vtk)
std::string ResultStringFn(std::string ResultStr, int itr, int NumberOfSteps) {
    std::stringstream ResultStringStr;
    std::string ResultString;
    ResultStringStr.str(std::string());
    ResultStringStr << ResultStr;
    ResultStringStr.fill('0');
    int digits = log10(NumberOfSteps)+1;
    ResultStringStr.width(digits);
    ResultStringStr << itr;
    ResultStringStr << ".vtk";
    ResultString = ResultStringStr.str().c_str();
    return ResultString;
}