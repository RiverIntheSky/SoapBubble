# pragma once

# include <string>
# include <map>
# include <iostream>
# include <vector>
# include <cmath>

# include "Partio.h"

// Using updated (v2) interfaces for CUBLAS and CUSPARSE
# include <cusparse.h>
# include <cublas_v2.h>

# include "cuda_runtime.h"
# include "helper_functions.h"
# include "device_launch_parameters.h"
# include "helper_cuda.h"
# include "helper_math.h"
# include "cufft.h"
# include "vectorUtil.cuh"

# define M_PI           3.14159265358979323846  /* pi */
# define M_2PI			6.28318530717958647692  /* 2pi */
# define M_hPI			1.57079632679489661923  /* pi / 2*/

# define centeredPhiOffset 0.0
# define centeredThetaOffset 0.5
# define vPhiPhiNorm M_2PI;
# define vPhiThetaNorm M_PI;
# define vThetaPhiNorm M_2PI;
# define vThetaThetaNorm (M_PI - 2 * gridLenGlobal)
# define pressurePhiNorm M_2PI
# define pressureThetaNorm M_PI
# define vPhiPhiOffset -0.5
# define vPhiThetaOffset 0.5
# define vThetaPhiOffset 0.0
# define vThetaThetaOffset 1.0


//# define getIndex(phi, theta) (theta * this->nPhi + phi)

# define DEBUGBUILD
# define greatCircle

// The solution to switch between double and float
// # define USEFLOAT

# ifdef USEFLOAT
typedef float fReal;
typedef float2 fReal2;
typedef float3 fReal3;
static __inline__ __host__ __device__ float2 make_fReal2(float a, float b) {
    return make_float2(a, b);
}


static __inline__ __host__ __device__ float3 make_fReal3(float a, float b, float c) {
    return make_float3(a, b, c);
}

# else
typedef double fReal;
typedef double2 fReal2;
typedef double3 fReal3;
static __inline__ __host__ __device__ double2 make_fReal2(double a, double b) {
    return make_double2(a, b);
}


static __inline__ __host__ __device__ double3 make_fReal3(double a, double b, double c) {
    return make_double3(a, b, c);
}


# endif
# define centeredOffset make_fReal2((fReal)0.5, (fReal)0.0) // (theta, phi)
# define vPhiOffset make_fReal2((fReal)0.5, (fReal)-0.5)
# define vThetaOffset make_fReal2((fReal)1.0, (fReal)0.0)

typedef cufftComplex ComplexFourier;

const size_t byte2Bits = 8;

const fReal uSolid = 0.0;
const fReal vSolid = 0.0;
const fReal R = 8.3144598;              // gas constant. J mol^-1 K^-1
const fReal sigma_a = 0.07275;		// surface tension of water. N m^-1
const fReal rho = 997;			// bulk fluid density. kg m^-3
const fReal mu = 0.00089;		// soap solution dynamic viscosity. Pa s
const fReal g = 9.8;			// standard gravity. m s^-2
const fReal rhoa = 1.184;		// air density. kg m^-3
const fReal nua = 1.562e-5;		// air kinematic viscosity. m^2 s^-1

enum gridType { FLUIDGRID, SOLIDGRID };

enum Coord { phi, theta };

# define WRITE_VELOCITY_DATA
// # define WRITE_PARTICLES
# define WRITE_THICKNESS_DATA
# define WRITE_CONCENTRATION_DATA
# define RUNGE_KUTTA
# define PERFORMANCE_BENCHMARK
# define TINYEXR_IMPLEMENTATION
# define sphere
# define gravity
// # define tiltedGravity

// if both gravity and tilted are defined, tilted has priority
# ifdef tiltedGravity
# undef gravity
# endif

# define evaporation -0.000000008
# define air
# define WRITE_TXT
# define BIMOCQ
// # define MULTISAMPLE