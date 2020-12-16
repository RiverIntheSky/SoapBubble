# pragma once

# include "../include/header.cuh"
# include <opencv2/opencv.hpp>

#define CHECK_CUDA(func)						\
    {									\
	cudaError_t status = (func);					\
	if (status != cudaSuccess) {					\
	    printf("CUDA API failed at line %d with error: %s (%d)\n",	\
		   __LINE__, cudaGetErrorString(status), status);	\
	}								\
    }


#define CHECK_CUSPARSE(func)						\
    {									\
	cusparseStatus_t status = (func);				\
	if (status != CUSPARSE_STATUS_SUCCESS) {			\
	    printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
		   __LINE__, cusparseGetErrorString(status), status);	\
	}								\
    }


// TODO
#define CHECK_CUBLAS(func)						\
    {									\
 	cublasStatus_t status = (func);					\
	checkCudaErrors(status);					\
    }


using namespace cv;

class Bubble
{
private:
    fReal radius;               // radius of sphere. m
    fReal invRadius;		// inverted radius of sphere. m^-1
    fReal H;			// mean half film thickness. m
    fReal U;			// characteristic flow velocity. 1 m s^-1
    fReal c_m;			// bulk mean concentration. mol m^-3
    fReal Gamma_m;		// surface mean concentration. mol m^-2
    fReal T;			// room temperature. K
    fReal Ds;			// soap diffusivity. m^2 s^-1
    fReal gs;                   // nondimensional gravity
    fReal rm;			// A quantity in Lennard-Jones potential, the distance
                                // where the potential minimum is reached. Not used. m
    fReal epsilon;		// expansion parameter
    fReal sigma_r;		// accounts for the elasticity of the film.
    fReal M;			// M.
    fReal S;			// S.
    fReal re;			// reciprocal value of Reynolds constant
    fReal Cr;			// air friction constant
    size_t nTheta;              // number of grid cells in u direction
    size_t nPhi;                // number of grid cells in v direction
    fReal gridLen;              // grid spacing (square in uv plane)
    fReal invGridLen;		// inverted grid spacing
    
    /* practical condition: dt <= 5*dx / u_max */
    /* dt should be less than DT as well */
    fReal dt;                   // time step
    fReal DT;                   // frame rate @24 fps = 0.0147
    int frames;                 // number of frames to export

    std::string outputDir;	// folder destination output

    std::string thicknessImage;	// file path of thickness image initialization

    int device;			// which gpu device to use
    std::string AMGconfig;	// AMGX config file
    fReal blendCoeff;
    fReal gammaMax;		// max soap concentration,
				// corresponds to zero surface tension

public:
    Bubble(fReal radius = 0.05, fReal H = 0.0000005, fReal U = 1.0, fReal c_m = 0.5,
	   fReal Gamma_m = 0.000001, fReal T = 298.15, fReal Ds = 0.01,
	   fReal rm = 0.000000005,size_t nTheta = 128, fReal dt = 0.005,
	   fReal DT = 1.0 / 24.0, int frames = 1000,
	   std::string outputDir = "../output/test/",
	   std::string thicknessImage = "", int device = 0,
	   std::string AMGconfig = "", fReal blendCoeff = 0.5);
    ~Bubble();

    /* run the solver */
    void run();
};
