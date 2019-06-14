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
    fReal sigma_a;		// surface tension of water. N m^-1 
    fReal R;			// gas constant. J mol^-1 K^-1
    fReal T;			// room temperature. K
    fReal rho;			// bulk fluid density. kg m^-3
    fReal mu;			// water dynamic viscosity. Pa s
    fReal Ds;			// soap diffusivity. m^2 s^-1
    fReal g;			// standard gravity. m s^-2
    fReal rm;			// Van der Waals constant
    fReal epsilon;		// expansion parameter
    fReal sigma_r;		// accounts for the elasticity of the film.
    fReal M;			// M.
    fReal S;			// S.
    fReal re;			// reciprocal value of Reynolds constant

    size_t nTheta;              // number of grid cells in u direction
    size_t nPhi;                // number of grid cells in v direction
    fReal gridLen;              // grid spacing (square in uv plane)
    fReal invGridLen;		// inverted grid spacing
    fReal particleDensity;      // number of particles per unit area on the flat sphere


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
    std::string particlePath;     // folder destination for simulation bgeo files

    std::string thicknessImage;	  // file path of thickness image initialization
    std::string solidImage;	  // file path of SOLIDCELL image map
    std::string colorImage;       // file path of image defining particle color


    vec3* colorMap;

public:
    Kamino(fReal radius = 0.05, fReal H = 0.0000005, fReal U = 1.0, fReal c_m = 0.5,
	   fReal Gamma_m = 0.000001, fReal sigma_a = 0.07275, fReal R = 8.3144598, fReal T = 298.15,
	   fReal rho = 997, fReal mu = 0.010005, fReal Ds = 0.01, fReal g = 9.8,
	   fReal rm = 0.000000005,size_t nTheta = 128, fReal particleDensity = 200.0,
	   float dt = 0.005, float DT = 1.0 / 24.0, int frames = 1000,
	   fReal A = 0.0, int B = 1, int C = 1, int D = 1, int E = 1,
	   std::string thicknessPath = "output/frame", std::string particlePath = "particles/frame",
	   std::string thicknessImage = "", std::string solidImage = "", std::string colorImage = "");
    ~Kamino();

    /* run the solver */
    void run();
};