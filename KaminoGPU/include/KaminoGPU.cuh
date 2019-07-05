# pragma once

# include "KaminoSolver.cuh"
# include <opencv2/opencv.hpp>

using namespace cv;

class Kamino
{
private:
    fReal radius;               // radius of sphere. m
    fReal invRadius;		// inverted radius of sphere. m^-1
    fReal H;			// characteristic film thickness. m
    fReal U;			// characteristic flow velocity. m s^-1
    fReal c_m;			// bulk mean concentration. mol m^-3
    fReal Gamma_m;		// surface mean concentration. mol m^-2
    fReal T;			// room temperature. K
    fReal Ds;			// soap diffusivity. m^2 s^-1
    fReal gs;                   // nondimensional gravity
    fReal rm;			// Van der Waals constant
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
    float dt;                   // time step
    float DT;                   // frame rate @24 fps = 0.0147
    int frames;                 // number of frames to export
        
    /* velocity initialization */
    /* u = A + sin(B*phi / 2) * sin(C*theta / 2) */
    /* v = sin(D*phi / 2) * sin(E*theta / 2) */
    /* coefficients are for Fourier sums representing each of the above functions */
    fReal A;
    int B, C, D, E;
    
    std::string thicknessPath;	  // folder destination thickness output
    std::string velocityPath;     // folder destination velocity output

    std::string thicknessImage;	  // file path of thickness image initialization
    std::string solidImage;	  // file path of SOLIDCELL image map
    std::string colorImage;       // file path of image defining particle color

    fReal particleDensity;	  // how many particles in a grid cell

public:
    Kamino(fReal radius = 0.05, fReal H = 0.0000005, fReal U = 1.0, fReal c_m = 0.5,
	   fReal Gamma_m = 0.000001, fReal T = 298.15, fReal Ds = 0.01,
	   fReal rm = 0.000000005,size_t nTheta = 128, float dt = 0.005,
	   float DT = 1.0 / 24.0, int frames = 1000,
	   fReal A = 0.0, int B = 1, int C = 1, int D = 1, int E = 1,
	   std::string thicknessPath = "output/frame", std::string velocityPath = "output/vel",
	   std::string thicknessImage = "", size_t particleDensity = 8);
    ~Kamino();

    /* run the solver */
    void run();
};