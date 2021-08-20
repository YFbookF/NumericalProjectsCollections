```
int numFrequencies; // total number of frequencies
int T; // total number of distinct locations where PAT pressures need to be evaluated
       // T is determined by reading the header of the input file specifying microphone positions
double * pressureAbs; // output pressure buffer
double * microphonePos; // the 3xT 2D array holding 3D locations where PAT pressures need to be evaluated
std::vector<double*> patObjects; // vector of pointers to dipole data, one pointer per frequency
std::vector<double> kTable; // vector of wave numbers, k=2*pi/lambda, one per frequency
std::vector<int> numSources; // num sources per each frequency
int maxNumSources; // max num sources over all frequencies
double * buffer; // workspace
PerformanceCounter performanceCounter; // to time PAT evaluation
// note: these global variables could be wrapped in a class

// evaluates dipoles for T microphone positions from the array microphonePos,
// using data from patObjects, kTable, numSources, numFrequencies
// stores computed pressures into a numFrequencies x T matrix 'pressureAbs'
void EvaluateDipoles()
{
  // each .sources file contains the data for all the dipoles for that frequency
  // the .sources file is an 11 x (num sources) matrix
  // each dipole is described by 11 coefficients, double precision each
  // each dipole corresponds to one column of the .sources matrix
  // there will be one .sources and .k file for every frequency

  // structure of data for each individual dipole (11 entries):
  // first 3 coefficients: world-coordinate dipole position (x,y,z)
  // then, monopole term (complex number, first Re, then Im)
  // then m=0 dipole term (complex number, first Re, then Im)
  // then m=-1 dipole term (complex number, first Re, then Im)
  // then m=+1 dipole term (complex number, first Re, then Im)

  // note: the 0,-1,+1 order might seem counter-intuitive, but it
  // naturally generalizes to higher-order sources: 0,-1,+1,-2,+2,-3,+3, etc.

  const int ssize = 11;

  // over all frequencies:
  for(int frequencyIndex = 0; frequencyIndex < numFrequencies; frequencyIndex++)
  {
    printf("%d ",frequencyIndex+1);fflush(stdout);
    double k = kTable[frequencyIndex]; // get the wave number for this frequency
    double pressureRe, pressureIm;
    double * sourceData = patObjects[frequencyIndex]; // point to the sources for this frequency

    // over all microphone positions (i.e. over all time samples):
    for(int t=0; t<T; t++)
    {
      double * pos = &microphonePos[3*t]; // get microphone position at this time-step
      pressureRe = 0;
      pressureIm = 0;

      #define COMPLEX_MULTIPLY(aRe,aIm,bRe,bIm,oRe,oIm)\
        oRe = (aRe) * (bRe) - (aIm) * (bIm);\
        oIm = (aRe) * (bIm) + (aIm) * (bRe);

      #define COMPLEX_MULTIPLY_ADD(aRe,aIm,bRe,bIm,oRe,oIm)\
        oRe += (aRe) * (bRe) - (aIm) * (bIm);\
        oIm += (aRe) * (bIm) + (aIm) * (bRe);

      double * coef = sourceData;
      // over all dipole sources at this frequency:
      for(int source=0; source < numSources[frequencyIndex]; source++)
      {
        // relative position of microphone with respect to the source
        double x = pos[0] - coef[0];
        double y = pos[1] - coef[1];
        double z = pos[2] - coef[2];

        // auxiliary quantities
        double planarR2 = x*x + y*y;
        double planarR = sqrt(planarR2);
        double r = sqrt(planarR2 + z*z);
        double cosTheta = z / r;
        double sinTheta = planarR / r;
        double cosPhi = x / planarR;
        double sinPhi = y / planarR;

        double kr = k * r;

        double invKr = 1.0 / kr;
        double sinKr = sin(kr);
        double cosKr = cos(kr);

        // monopole term:

        double bufferRe = sinKr * invKr;
        double bufferIm = cosKr * invKr;

        COMPLEX_MULTIPLY_ADD(bufferRe,bufferIm,coef[3],coef[4],pressureRe,pressureIm);

        // dipole terms:

        double radialRe = invKr * (-cosKr + invKr * sinKr);
        double radialIm = invKr * ( sinKr + invKr * cosKr);

        // m = 0
        bufferRe = radialRe * cosTheta;
        bufferIm = radialIm * cosTheta;
        COMPLEX_MULTIPLY_ADD(bufferRe,bufferIm,coef[5],coef[6],pressureRe,pressureIm);

        double cosPhiSinTheta = cosPhi * sinTheta;
        double sinPhiSinTheta = sinPhi * sinTheta;

        // m = -1
        COMPLEX_MULTIPLY(radialRe,radialIm,cosPhiSinTheta,-sinPhiSinTheta,bufferRe,bufferIm);
        COMPLEX_MULTIPLY_ADD(bufferRe,bufferIm,coef[7],coef[8],pressureRe,pressureIm);

        // m = 1
        COMPLEX_MULTIPLY(radialRe,radialIm,-cosPhiSinTheta,-sinPhiSinTheta,bufferRe,bufferIm);
        COMPLEX_MULTIPLY_ADD(bufferRe,bufferIm,coef[9],coef[10],pressureRe,pressureIm);

        coef += ssize; // increment coef to point to the next source
      }

      // store |p| :
      pressureAbs[numFrequencies * t + frequencyIndex] = sqrt(pressureRe*pressureRe + pressureIm*pressureIm);

    }
  }
  printf("\n");
}
```

http://graphics.cs.cmu.edu/projects/pat/

# Precomputed Acoustic Transfer: Output-sensitive, accurate sound generation for geometrically complex vibration sources