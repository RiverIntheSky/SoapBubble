# pragma once

# include "../include/KaminoHeader.cuh"
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

class Kamino
{
private:
    float radius;               // radius of sphere. m
    float invRadius;		// inverted radius of sphere. m^-1
    float H;			// characteristic film thickness. m
    float U;			// characteristic flow velocity. m s^-1
    float c_m;			// bulk mean concentration. mol m^-3
    float Gamma_m;		// surface mean concentration. mol m^-2
    float T;			// room temperature. K
    float Ds;			// soap diffusivity. m^2 s^-1
    float gs;                   // nondimensional gravity
    float rm;			// Van der Waals constant
    float epsilon;		// expansion parameter
    float sigma_r;		// accounts for the elasticity of the film.
    float M;			// M.
    float S;			// S.
    float re;			// reciprocal value of Reynolds constant
    float Cr;			// air friction constant
    size_t nTheta;              // number of grid cells in u direction
    size_t nPhi;                // number of grid cells in v direction
    float gridLen;              // grid spacing (square in uv plane)
    float invGridLen;		// inverted grid spacing
    
    /* practical condition: dt <= 5*dx / u_max */
    /* dt should be less than DT as well */
    float dt;                   // time step
    float DT;                   // frame rate @24 fps = 0.0147
    int frames;                 // number of frames to export
        
    /* velocity initialization */
    /* u = A + sin(B*phi / 2) * sin(C*theta / 2) */
    /* v = sin(D*phi / 2) * sin(E*theta / 2) */
    /* coefficients are for Fourier sums representing each of the above functions */
    // float A;
    // int B, C, D, E;
    
    std::string outputDir;	// folder destination output

    std::string thicknessImage;	// file path of thickness image initialization
    std::string solidImage;	// file path of SOLIDCELL image map
    std::string colorImage;     // file path of image defining particle color

    float particleDensity;	// how many particles in a grid cell
    int device;			// which gpu device to use
    std::string AMGconfig;	// AMGX config file
    float blendCoeff;

public:
    Kamino(float radius = 0.05, float H = 0.0000005, float U = 1.0, float c_m = 0.5,
	   float Gamma_m = 0.000001, float T = 298.15, float Ds = 0.01,
	   float rm = 0.000000005,size_t nTheta = 128, float dt = 0.005,
	   float DT = 1.0 / 24.0, int frames = 1000,
	   std::string outputDir = "../output/test/",
	   std::string thicknessImage = "", size_t particleDensity = 8, int device = 0,
	   std::string AMGconfig = "", float blendCoeff = 0.5f);
    ~Kamino();

    /* run the solver */
    void run();
};
