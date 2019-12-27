# include "KaminoSolver.cuh"
# include "KaminoGPU.cuh"
# include "KaminoTimer.cuh"
# include <boost/filesystem.hpp>
# include "utils.h"

# include <amgcl/make_solver.hpp>
# include <amgcl/profiler.hpp>
# include <amgcl/solver/bicgstab.hpp>
# include <amgcl/amg.hpp>
# include <amgcl/backend/cuda.hpp>
# include <amgcl/relaxation/cusparse_ilu0.hpp>
# include <amgcl/coarsening/smoothed_aggregation.hpp>
# include <amgcl/relaxation/spai0.hpp>
# include <amgcl/adapter/crs_tuple.hpp>

__constant__ fReal invGridLenGlobal;
static __constant__ size_t nPhiGlobal;
static __constant__ size_t nThetaGlobal;
static __constant__ fReal invRadiusGlobal;
static __constant__ fReal radiusGlobal;
static __constant__ fReal timeStepGlobal;
static __constant__ fReal currentTimeGlobal;
static __constant__ fReal gridLenGlobal;
static __constant__ fReal SGlobal;
static __constant__ fReal MGlobal;
static __constant__ fReal reGlobal;
static __constant__ fReal gGlobal;
static __constant__ fReal DsGlobal;
static __constant__ fReal CrGlobal;
static __constant__ fReal UGlobal;
static __constant__ fReal blend_coeff;
static __constant__ fReal evaporationRate;
static __constant__ fReal W1;
static __constant__ fReal W2;


# define eps 1e-5
# define MAX_BLOCK_SIZE 4096 /* TODO: deal with larger resolution */


namespace amgcl {
    profiler<> prof("v2");
}

typedef amgcl::backend::cuda<fReal> Backend;
typedef amgcl::make_solver<
  amgcl::amg<
    Backend,
	amgcl::coarsening::smoothed_aggregation,
	amgcl::relaxation::spai0
	>,
      amgcl::solver::bicgstab<Backend>
      > AMGCLSolver;

/**
 * query value at coordinate
 */
inline __device__ fReal& at(fReal* &array, int &thetaId, int &phiId) {
    return array[phiId + thetaId * nPhiGlobal];
}


inline __device__ fReal& at(fReal* &array, int2 &Id) {
    return array[Id.y + Id.x * nPhiGlobal];
}


/**
 * query value in pitched memory at coordinate
 */
inline __device__ fReal& at(fReal* array, int thetaId, int phiId, size_t pitch) {
    return array[phiId + thetaId * pitch];
}


inline __device__ fReal& at(fReal* array, int2 Id, size_t pitch) {
    return array[Id.y + Id.x * pitch];
}


/**
 * distance between two coordinates on the sphere (unit in grid)
 */
inline __device__ fReal dist(fReal2 Id1, fReal2 Id2) {
    Id1 *= gridLenGlobal;
    Id2 *= gridLenGlobal;
# ifdef USEFLOAT
    fReal3 Id13 = normalize(make_float3(cosf(Id1.y) * sinf(Id1.x),
					sinf(Id1.y) * sinf(Id1.x),
					cosf(Id1.x)));
    fReal3 Id23 = normalize(make_float3(cosf(Id2.y) * sinf(Id2.x),
					sinf(Id2.y) * sinf(Id2.x),
					cosf(Id2.x)));
# else
    fReal3 Id13 = normalize(make_double3(cos(Id1.y) * sin(Id1.x),
					sin(Id1.y) * sin(Id1.x),
					cos(Id1.x)));
    fReal3 Id23 = normalize(make_double3(cos(Id2.y) * sin(Id2.x),
					sin(Id2.y) * sin(Id2.x),
					cos(Id2.x)));
# endif
    return invGridLenGlobal * safe_acos(dot(Id13, Id23));
}


/**
 * return the maximal absolute value in array vel
 * usage: maxValKernel<<<gridSize, blockSize>>>(maxVal, vel);
 *        maxValKernel<<<1, blockSize>>>(maxVal, maxVal);
 */
__global__ void maxValKernel(fReal* maxVal, fReal* array) {
    __shared__ fReal maxValTile[MAX_BLOCK_SIZE];
	
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    maxValTile[tid] = fabsf(array[i]);
    __syncthreads();   
    //sequential addressing by reverse loop and thread-id based indexing
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
	if (tid < s) {
	    if (maxValTile[tid + s] > maxValTile[tid])
		maxValTile[tid] = maxValTile[tid + s];
	}
	__syncthreads();
    }
    
    if (tid == 0) {
	maxVal[blockIdx.x] = maxValTile[0];
    }
}


/**
 * map the index to the correct range (periodic boundary condition)
 * assume theta lies not too far away from the interval [0, nThetaGlobal],
 * otherwise is the step size too large;
 * thetaId = Id.x phiId = Id.y
 *
 * @param Id = (thetaCoord, phiCoord) / gridLen 
 */
__device__ bool validateCoord(fReal2& Id) {
    bool ret = false;

    if (Id.x >= nThetaGlobal) {
	Id.x = nPhiGlobal - Id.x;
	Id.y += nThetaGlobal;
    	ret = !ret;
    }
    if (Id.x < 0) {
    	Id.x = -Id.x;
    	Id.y += nThetaGlobal;
    	ret = !ret;
    }
    if (Id.x > nThetaGlobal || Id.x < 0)
	printf("Warning: step size too large! theta = %f\n", Id.x);
# ifdef USEFLOAT
    Id.y = fmodf(Id.y + nPhiGlobal, (fReal)nPhiGlobal);
# else
    Id.y = fmod(Id.y + nPhiGlobal, (fReal)nPhiGlobal);
# endif
    return ret;
}


__device__ void validateId(int2& Id) {
    Id.x = Id.x % nPhiGlobal;
    if (Id.x >= nThetaGlobal) {
	Id.x = nPhiGlobal - 1 - Id.x;
	Id.y += nThetaGlobal;
    }
    Id.y = Id.y % nPhiGlobal;
}


/**
 * linear interpolation
 */
__device__ fReal kaminoLerp(fReal from, fReal to, fReal alpha)
{
    return (1.0 - alpha) * from + alpha * to;
}


/**
 * bilinear interpolation
 */
__device__ fReal bilerp(fReal ll, fReal lr, fReal hl, fReal hr,
			fReal alphaPhi, fReal alphaTheta)
{
    return kaminoLerp(kaminoLerp(ll, lr, alphaPhi),
		      kaminoLerp(hl, hr, alphaPhi), alphaTheta);
}


/**
 * sample velocity in phi direction at position rawId
 * rawId is moved to velPhi coordinates to compensate MAC
 */
__device__ fReal sampleVPhi(fReal* input, fReal2 rawId, size_t pitch) {
    fReal2 Id = rawId - vPhiOffset;
    
    bool isFlippedPole = validateCoord(Id);

    int phiIndex = static_cast<int>(floorf(Id.y));
    int thetaIndex = static_cast<int>(floorf(Id.x));
    fReal alphaPhi = Id.y - static_cast<fReal>(phiIndex);
    fReal alphaTheta = Id.x - static_cast<fReal>(thetaIndex);
    
    if (thetaIndex == 0 && isFlippedPole) {
	size_t phiLower = (phiIndex) % nPhiGlobal;
	size_t phiHigher = (phiLower + 1) % nPhiGlobal;
	fReal higherBelt = -kaminoLerp(input[phiLower + pitch * thetaIndex],
				       input[phiHigher + pitch * thetaIndex], alphaPhi);

	phiLower = (phiIndex + nPhiGlobal / 2) % nPhiGlobal;
	phiHigher = (phiLower + 1) % nPhiGlobal;

	fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);
  
	fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
	return lerped;
    }
    
    if (isFlippedPole) {
	thetaIndex -= 1;
    }
    
    if (thetaIndex == nThetaGlobal - 1) {
	size_t phiLower = (phiIndex) % nPhiGlobal;
	size_t phiHigher = (phiLower + 1) % nPhiGlobal;
	fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);
	
	phiLower = (phiIndex + nPhiGlobal / 2) % nPhiGlobal;
	phiHigher = (phiLower + 1) % nPhiGlobal;

	fReal higherBelt = -kaminoLerp(input[phiLower + pitch * thetaIndex],
				       input[phiHigher + pitch * thetaIndex], alphaPhi);

	fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
	if (isFlippedPole)
	    lerped = -lerped;
	return lerped;
    }
  
    size_t phiLower = phiIndex % nPhiGlobal;
    size_t phiHigher = (phiLower + 1) % nPhiGlobal;
    size_t thetaLower = thetaIndex;
    size_t thetaHigher = thetaIndex + 1;

    fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaLower],
				 input[phiHigher + pitch * thetaLower], alphaPhi);
    fReal higherBelt = kaminoLerp(input[phiLower + pitch * thetaHigher],
				  input[phiHigher + pitch * thetaHigher], alphaPhi);

    fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
    if (isFlippedPole)
	lerped = -lerped;
    return lerped;
}
    

/**
 * sample velocity in theta direction at position rawId
 * rawId is moved to velTheta coordinates to compensate MAC
 */
__device__ fReal sampleVTheta(fReal* input, fReal2 rawId, size_t pitch) {
    fReal2 Id = rawId - vThetaOffset;
    
    bool isFlippedPole = validateCoord(Id);

    int phiIndex = static_cast<int>(floorf(Id.y));
    int thetaIndex = static_cast<int>(floorf(Id.x));
    fReal alphaPhi = Id.y - static_cast<fReal>(phiIndex);
    fReal alphaTheta = Id.x - static_cast<fReal>(thetaIndex);
    
    if (rawId.x < 0 && rawId.x > -1 || rawId.x > nThetaGlobal && rawId.x < nThetaGlobal + 1 ) {
	thetaIndex -= 1;
	alphaTheta += 1;
    } else if (rawId.x >= nThetaGlobal + 1 || rawId.x <= -1) {
    	thetaIndex -= 2;
    }

    if (thetaIndex == 0 && isFlippedPole && rawId.x > -1) {
    	size_t phiLower = phiIndex % nPhiGlobal;
    	size_t phiHigher = (phiLower + 1) % nPhiGlobal;
    	fReal higherBelt = -kaminoLerp(input[phiLower + pitch * thetaIndex],
    				       input[phiHigher + pitch * thetaIndex], alphaPhi);
	
    	phiLower = (phiLower + nPhiGlobal / 2) % nPhiGlobal;
    	phiHigher = (phiHigher + nPhiGlobal / 2) % nPhiGlobal;
    	fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
    				     input[phiHigher + pitch * thetaIndex], alphaPhi);

    	alphaTheta = 0.5 * alphaTheta;
    	fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
    	return lerped;
	
    }
    
    if (thetaIndex == nThetaGlobal - 2) {
	size_t phiLower = phiIndex % nPhiGlobal;
	size_t phiHigher = (phiLower + 1) % nPhiGlobal;
	fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);

	phiLower = (phiLower + nPhiGlobal / 2) % nPhiGlobal;
	phiHigher = (phiHigher + nPhiGlobal / 2) % nPhiGlobal;
	fReal higherBelt = -kaminoLerp(input[phiLower + pitch * thetaIndex],
				       input[phiHigher + pitch * thetaIndex], alphaPhi);

	alphaTheta = 0.5 * alphaTheta;
	fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
	if (isFlippedPole)
	    lerped = -lerped;
	return lerped;
    }
    
    size_t phiLower = phiIndex % nPhiGlobal;
    size_t phiHigher = (phiLower + 1) % nPhiGlobal;
    size_t thetaLower = thetaIndex;
    size_t thetaHigher = thetaIndex + 1;

    fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaLower],
				 input[phiHigher + pitch * thetaLower], alphaPhi);
    fReal higherBelt = kaminoLerp(input[phiLower + pitch * thetaHigher],
				  input[phiHigher + pitch * thetaHigher], alphaPhi);

    fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
    if (isFlippedPole)
	lerped = -lerped;
    return lerped;
}


/**
 * sample scalar at position rawId
 * rawId is moved to scalar coordinates to compensate MAC
 */
__device__ fReal sampleCentered(fReal* input, fReal2 rawId, size_t pitch) {
    fReal2 Id = rawId - centeredOffset;

    bool isFlippedPole = validateCoord(Id);

    int phiIndex = static_cast<int>(floorf(Id.y));
    int thetaIndex = static_cast<int>(floorf(Id.x));
    fReal alphaPhi = Id.y - static_cast<fReal>(phiIndex);
    fReal alphaTheta = Id.x - static_cast<fReal>(thetaIndex);

    if (thetaIndex == 0 && isFlippedPole) {
	size_t phiLower = phiIndex % nPhiGlobal;
	size_t phiHigher = (phiLower + 1) % nPhiGlobal;
	fReal higherBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				      input[phiHigher + pitch * thetaIndex], alphaPhi);

	phiLower = (phiLower + nPhiGlobal / 2) % nPhiGlobal;
	phiHigher = (phiHigher + nPhiGlobal / 2) % nPhiGlobal;
	fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);

	fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
	return lerped;
    }
    
    if (isFlippedPole) {
	thetaIndex -= 1;
    }
    
    if (thetaIndex == nThetaGlobal - 1) {
	size_t phiLower = phiIndex % nPhiGlobal;
	size_t phiHigher = (phiLower + 1) % nPhiGlobal;
	fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);

	phiLower = (phiLower + nPhiGlobal / 2) % nPhiGlobal;
	phiHigher = (phiHigher + nPhiGlobal / 2) % nPhiGlobal;
	fReal higherBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				      input[phiHigher + pitch * thetaIndex], alphaPhi);

	fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
	return lerped;
    }

    size_t phiLower = phiIndex % nPhiGlobal;
    size_t phiHigher = (phiLower + 1) % nPhiGlobal;
    size_t thetaLower = thetaIndex;
    size_t thetaHigher = thetaIndex + 1;

    fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaLower],
				 input[phiHigher + pitch * thetaLower], alphaPhi);
    fReal higherBelt = kaminoLerp(input[phiLower + pitch * thetaHigher],
				  input[phiHigher + pitch * thetaHigher], alphaPhi);

    fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
    return lerped;
}


/**
 * @return (velTheta, velPhi)
 */
inline __device__ fReal2 getVelocity(fReal* velPhi, fReal* velTheta, fReal2 &Id, size_t pitch){
    return make_fReal2(sampleVTheta(velTheta, Id, pitch),
		       sampleVPhi(velPhi, Id, pitch));
}

/**
 * Runge-Kutta 3rd Order Ralston
 * positive dt => trace backward;
 * negative dt => trace forward;
 */
inline __device__ fReal2 traceRK3(fReal* velTheta, fReal* velPhi, fReal dt,
				  fReal2& Id0, size_t pitch){
    double GridLenD = M_PI/ nThetaGlobal;
    double invGridD = nThetaGlobal / M_PI;
    double c0 = 2.0 / 9.0 * dt * invGridD,
	c1 = 3.0 / 9.0 * dt * invGridD,
	c2 = 4.0 / 9.0 * dt * invGridD;
    fReal2 vel0 = getVelocity(velPhi, velTheta, Id0, pitch);
    vel0.y /= fReal(sin(Id0.x * GridLenD));
    fReal2 Id1 = Id0 - 0.5 * dt * vel0 * invGridD;
    fReal2 vel1 = getVelocity(velPhi, velTheta, Id1, pitch);
    vel1.y /= fReal(sin(Id1.x * GridLenD));
    fReal2 Id2 = Id1 - 0.75 * dt * vel1 * invGridD;
    fReal2 vel2 = getVelocity(velPhi, velTheta, Id2, pitch);
    vel2.y /= fReal(sin(Id2.x * GridLenD));

    return Id0 - c0 * vel0 - c1 * vel1 - c2 * vel2;
}

inline __device__ fReal2 traceGreatCircle(fReal* velTheta, fReal* velPhi, fReal dt,
					  fReal2& gId, size_t pitch){
    fReal2 gCoord = gId * gridLenGlobal;

# ifdef USEFLOAT
    // Trigonometric functions
    fReal sinTheta = sinf(gCoord.x);
    fReal cosTheta = cosf(gCoord.x);
    fReal sinPhi = sinf(gCoord.y);
    fReal cosPhi = cosf(gCoord.y);
# else
    fReal sinTheta = sin(gCoord.x);
    fReal cosTheta = cos(gCoord.x);
    fReal sinPhi = sin(gCoord.y);
    fReal cosPhi = cos(gCoord.y);
# endif
    
    // Unit vector in theta and phi direction
    fReal3 eTheta = make_fReal3(cosTheta * cosPhi, cosTheta * sinPhi, -sinTheta);
    fReal3 ePhi = make_fReal3(-sinPhi, cosPhi, (fReal)0.0);

    // Sample the speed
    fReal guTheta = sampleVTheta(velTheta, gId, pitch);
    fReal guPhi = sampleVPhi(velPhi, gId, pitch);

    // Circle
    fReal3 u, u_, v_;
    fReal u_norm, deltaS;
    fReal3 w_ = make_fReal3(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);

    u = guTheta * eTheta + guPhi * ePhi;
    u_norm = length(u);
    if (u_norm > 0) {
    	u_ = normalize(u);
        v_ = cross(w_, u_);
    } else {
    	return gId;
    }

    // Traced halfway in phi-theta space
    deltaS = - 0.5 * u_norm * dt;
    
# ifdef USEFLOAT
    fReal3 midx = u_ * sinf(deltaS) + w_ * cosf(deltaS);
    fReal2 midCoord = make_fReal2(safe_acos(midx.z), atan2f(midx.y, midx.x));
# else
    fReal3 midx = u_ * sin(deltaS) + w_ * cos(deltaS);
    fReal2 midCoord = make_fReal2(safe_acos(midx.z), atan2(midx.y, midx.x));
# endif
    
    fReal2 midId = midCoord * invGridLenGlobal;

    fReal muTheta = sampleVTheta(velTheta, midId, pitch);
    fReal muPhi = sampleVPhi(velPhi, midId, pitch);
    
# ifdef USEFLOAT
    fReal3 mu = make_float3(muTheta * cosf(midCoord.x) * cosf(midCoord.y) - muPhi * sinf(midCoord.y),
    			    muTheta * cosf(midCoord.x) * sinf(midCoord.y) + muPhi * cosf(midCoord.y),
    			    -muTheta * sinf(midCoord.x));

    fReal3 uCircleMid_ = u_ * cosf(deltaS) - w_ * sinf(deltaS);
# else
    fReal3 mu = make_double3(muTheta * cos(midCoord.x) * cos(midCoord.y) - muPhi * sin(midCoord.y),
			     muTheta * cos(midCoord.x) * sin(midCoord.y) + muPhi * cos(midCoord.y),
			     -muTheta * sin(midCoord.x));

    fReal3 uCircleMid_ = u_ * cos(deltaS) - w_ * sin(deltaS);
# endif
    fReal3 vCircleMid_ = cross(midx, uCircleMid_);

    fReal mguTheta = dot(mu, vCircleMid_);
    fReal mguPhi = dot(mu, uCircleMid_);

    u = mguPhi * u_ + mguTheta * v_;
    u_norm = length(u);
    if (u_norm > 0) {
    	u_ = normalize(u);
        v_ = cross(w_, u_);
    } else {
    	return gId;
    }

    deltaS = -u_norm * dt;
    fReal3 px = u_ * sin(deltaS) + w_ * cos(deltaS);
    
# ifdef USEFLOAT
    return make_float2(safe_acos(px.z), atan2f(px.y, px.x)) * invGridLenGlobal;
# else
    return make_double2(safe_acos(px.z), atan2(px.y, px.x)) * invGridLenGlobal;
# endif
}


inline __device__ fReal2 lerpCoords(fReal2 from, fReal2 to, fReal alpha) {
    double gridLenD = M_PI / nThetaGlobal;
    double from_theta = from.x * gridLenD;
    double from_phi = from.y * gridLenD;
    double to_theta = to.x * gridLenD;
    double to_phi = to.y * gridLenD;
    double3 from3 = normalize(make_double3(cos(from_phi) * sin(from_theta),
					   sin(from_phi) * sin(from_theta),
					   cos(from_theta)));
    double3 to3 = normalize(make_double3(cos(to_phi) * sin(to_theta),
					 sin(to_phi) * sin(to_theta),
					 cos(to_theta)));
    double3 k = normalize(cross(from3, to3));
    if (isnan(k.x))
	return from;
    double span = safe_acos(dot(from3, to3));
    span *= alpha;
    double3 interpolated3 = from3 * cos(span) + cross(k, from3) * sin(span)
	+ k * dot(k, from3) * (1 - cos(span));

    double2 interpolated = make_double2(safe_acos(interpolated3.z),
					atan2(interpolated3.y,
					      interpolated3.x)) / gridLenD;
    return make_fReal2((fReal)interpolated.x, (fReal)interpolated.y);
}


__device__ fReal2 sampleMapping(fReal* map_t, fReal* map_p, fReal2& rawId){
    fReal2 Id = rawId - centeredOffset;
    bool isFlippedPole = validateCoord(Id);

    int phiIndex = static_cast<int>(floorf(Id.y));
    int thetaIndex = static_cast<int>(floorf(Id.x));
    fReal alphaPhi = Id.y - static_cast<fReal>(phiIndex);
    fReal alphaTheta = Id.x - static_cast<fReal>(thetaIndex);

    fReal2 ll, lr, hl, hr;
    if (isFlippedPole) {
	if (thetaIndex == 0) {
	    size_t phiLower = phiIndex % nPhiGlobal;
	    size_t phiHigher = (phiLower + 1) % nPhiGlobal;

	    hl.x = map_t[phiLower];
	    hl.y = map_p[phiLower];
	    hr.x = map_t[phiHigher];
	    hr.y = map_p[phiHigher];

	    phiLower = (phiLower + nPhiGlobal / 2) % nPhiGlobal;
	    phiHigher = (phiHigher + nPhiGlobal / 2) % nPhiGlobal;

	    ll.x = map_t[phiLower];
	    ll.y = map_p[phiLower];
	    lr.x = map_t[phiHigher];
	    lr.y = map_p[phiHigher];
	} else {
	    thetaIndex -= 1;
	}
    }

    if (thetaIndex == nThetaGlobal - 1) {
	size_t phiLower = phiIndex % nPhiGlobal;
	size_t phiHigher = (phiLower + 1) % nPhiGlobal;

	ll.x = map_t[phiLower + nPhiGlobal * thetaIndex];
	ll.y = map_p[phiLower + nPhiGlobal * thetaIndex];
	lr.x = map_t[phiHigher + nPhiGlobal * thetaIndex];
	lr.y = map_p[phiHigher + nPhiGlobal * thetaIndex];

	phiLower = (phiLower + nPhiGlobal / 2) % nPhiGlobal;
	phiHigher = (phiHigher + nPhiGlobal / 2) % nPhiGlobal;

	hl.x = map_t[phiLower + nPhiGlobal * thetaIndex];
	hl.y = map_p[phiLower + nPhiGlobal * thetaIndex];
	hr.x = map_t[phiHigher + nPhiGlobal * thetaIndex];
	hr.y = map_p[phiHigher + nPhiGlobal * thetaIndex];
    } else if (thetaIndex != 0 || (thetaIndex == 0 && !isFlippedPole)) {
	size_t phiLower = phiIndex % nPhiGlobal;
	size_t phiHigher = (phiLower + 1) % nPhiGlobal;
	size_t thetaLower = thetaIndex;
	size_t thetaHigher = thetaIndex + 1;

	ll.x = map_t[phiLower + nPhiGlobal * thetaLower];
	ll.y = map_p[phiLower + nPhiGlobal * thetaLower];
	lr.x = map_t[phiHigher + nPhiGlobal * thetaLower];
	lr.y = map_p[phiHigher + nPhiGlobal * thetaLower];
	hl.x = map_t[phiLower + nPhiGlobal * thetaHigher];
	hl.y = map_p[phiLower + nPhiGlobal * thetaHigher];
	hr.x = map_t[phiHigher + nPhiGlobal * thetaHigher];
	hr.y = map_p[phiHigher + nPhiGlobal * thetaHigher];
    }

    return lerpCoords(lerpCoords(ll, lr, alphaPhi),
		      lerpCoords(hl, hr, alphaPhi), alphaTheta);

}


__global__ void	updateMappingKernel(fReal* velTheta, fReal* velPhi, fReal dt,
				     fReal* map_t, fReal* map_p,
				     fReal* tmp_t, fReal* tmp_p, size_t pitch){
    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    fReal2 pos = make_fReal2((fReal)thetaId, (fReal)phiId) + centeredOffset;

# ifdef greatCircle
    fReal2 back_pos = traceGreatCircle(velTheta, velPhi, dt, pos, pitch);
# else
    fReal2 back_pos = traceRK3(velTheta, velPhi, dt, pos, pitch);
# endif

    fReal2 sampledId = sampleMapping(map_t, map_p, back_pos);

    validateCoord(sampledId);

    at(tmp_p, thetaId, phiId) = sampledId.y;
    at(tmp_t, thetaId, phiId) = sampledId.x;
}


/**
 * advect vetor using great cicle method
 */
__global__ void advectionVSpherePhiKernel
(fReal* velPhiOutput, fReal* velPhiInput, fReal* velThetaInput, size_t pitch)
{
    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;
    
    // Coord in phi space
    fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + vPhiOffset;
    fReal2 gCoord = gId * gridLenGlobal;

# ifdef USEFLOAT
    // Trigonometric functions
    fReal sinTheta = sinf(gCoord.x);
    fReal cosTheta = cosf(gCoord.x);
    fReal sinPhi = sinf(gCoord.y);
    fReal cosPhi = cosf(gCoord.y);
# else
    fReal sinTheta = sin(gCoord.x);
    fReal cosTheta = cos(gCoord.x);
    fReal sinPhi = sin(gCoord.y);
    fReal cosPhi = cos(gCoord.y);
# endif

    // Sample the speed
    fReal guTheta = sampleVTheta(velThetaInput, gId, pitch);
    fReal guPhi = velPhiInput[thetaId * pitch + phiId];

    // Unit vector in theta and phi direction
    fReal3 eTheta = make_fReal3(cosTheta * cosPhi, cosTheta * sinPhi, -sinTheta);
    fReal3 ePhi = make_fReal3(-sinPhi, cosPhi, 0.0);

    // Circle
    fReal3 u, u_, v_;
    fReal u_norm, deltaS;
    fReal3 w_ = make_fReal3(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
    
    u = guTheta * eTheta + guPhi * ePhi;
    u_norm = length(u);
    if (u_norm > 0) {
	u_ = normalize(u);
        v_ = cross(w_, u_);
    } else {
	velPhiOutput[thetaId * pitch + phiId] = 0.0;
	return;
    }
	
# ifdef RUNGE_KUTTA
    // Traced halfway in phi-theta space
    deltaS = - 0.5 * u_norm * timeStepGlobal;

# ifdef USEFLOAT    
    fReal3 midx = u_ * sinf(deltaS) + w_ * cosf(deltaS);
    fReal2 midCoord = make_fReal2(safe_acos(midx.z), atan2f(midx.y, midx.x));
# else
    fReal3 midx = u_ * sin(deltaS) + w_ * cos(deltaS);
    fReal2 midCoord = make_fReal2(safe_acos(midx.z), atan2(midx.y, midx.x));
# endif
    
    fReal2 midId = midCoord * invGridLenGlobal;

    fReal muTheta = sampleVTheta(velThetaInput, midId, pitch);
    fReal muPhi = sampleVPhi(velPhiInput, midId, pitch);
    
# ifdef USEFLOAT
    fReal3 mu = make_fReal3(muTheta * cosf(midCoord.x) * cosf(midCoord.y) - muPhi * sinf(midCoord.y),
			    muTheta * cosf(midCoord.x) * sinf(midCoord.y) + muPhi * cosf(midCoord.y),
			    -muTheta * sinf(midCoord.x));
    fReal3 uCircleMid_ = u_ * cosf(deltaS) - w_ * sinf(deltaS);
# else
    fReal3 mu = make_fReal3(muTheta * cos(midCoord.x) * cos(midCoord.y) - muPhi * sin(midCoord.y),
			    muTheta * cos(midCoord.x) * sin(midCoord.y) + muPhi * cos(midCoord.y),
			    -muTheta * sin(midCoord.x));
    fReal3 uCircleMid_ = u_ * cos(deltaS) - w_ * sin(deltaS);
# endif
    
    fReal3 vCircleMid_ = cross(midx, uCircleMid_);
    
    fReal mguTheta = dot(mu, vCircleMid_);
    fReal mguPhi = dot(mu, uCircleMid_);

    u = mguPhi * u_ + mguTheta * v_;
    u_norm = length(u);
    if (u_norm > 0) {
	u_ = normalize(u);
        v_ = cross(w_, u_);
    } else {
	at(velPhiOutput, thetaId, phiId, pitch) = 0.0;
	return;
    }

# endif
    deltaS = -u_norm * timeStepGlobal;

# ifdef USEFLOAT
    fReal3 px = u_ * sinf(deltaS) + w_ * cosf(deltaS);
    fReal2 pCoord = make_fReal2(safe_acos(px.z), atan2f(px.y, px.x));
# else
    fReal3 px = u_ * sin(deltaS) + w_ * cos(deltaS);
    fReal2 pCoord = make_fReal2(safe_acos(px.z), atan2(px.y, px.x));
# endif
    fReal2 pId = pCoord * invGridLenGlobal;

    fReal puTheta = sampleVTheta(velThetaInput, pId, pitch);
    fReal puPhi = sampleVPhi(velPhiInput, pId, pitch);
    
# ifdef USEFLOAT
    fReal3 pu = make_fReal3(puTheta * cosf(pCoord.x) * cosf(pCoord.y) - puPhi * sinf(pCoord.y),
			    puTheta * cosf(pCoord.x) * sinf(pCoord.y) + puPhi * cosf(pCoord.y),
			    -puTheta * sinf(pCoord.x));
	
    fReal3 uCircleP_ = u_ * cosf(deltaS) - w_ * sinf(deltaS);
# else
    fReal3 pu = make_fReal3(puTheta * cos(pCoord.x) * cos(pCoord.y) - puPhi * sin(pCoord.y),
			    puTheta * cos(pCoord.x) * sin(pCoord.y) + puPhi * cos(pCoord.y),
			    -puTheta * sin(pCoord.x));
	
    fReal3 uCircleP_ = u_ * cos(deltaS) - w_ * sin(deltaS);
# endif
    fReal3 vCircleP_ = cross(px, uCircleP_);

    puTheta = dot(pu, vCircleP_);
    puPhi = dot(pu, uCircleP_);

    pu = puPhi * u_ + puTheta * v_;
    at(velPhiOutput, thetaId, phiId, pitch) = dot(pu, ePhi);
}


/**
 * advect vetor using great cicle method
 */
// TODO: fReal to double??
__global__ void advectionVSphereThetaKernel
(fReal* velThetaOutput, fReal* velPhiInput, fReal* velThetaInput, size_t pitch)
{
    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;
    
    // Coord in theta space
    fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + vThetaOffset;
    fReal2 gCoord = gId * gridLenGlobal;

# ifdef USEFLOAT
    // Trigonometric functions
    fReal sinTheta = sinf(gCoord.x);
    fReal cosTheta = cosf(gCoord.x);
    fReal sinPhi = sinf(gCoord.y);
    fReal cosPhi = cosf(gCoord.y);
# else
    fReal sinTheta = sin(gCoord.x);
    fReal cosTheta = cos(gCoord.x);
    fReal sinPhi = sin(gCoord.y);
    fReal cosPhi = cos(gCoord.y);
# endif
    
    // Sample the speed
    fReal guTheta = velThetaInput[thetaId * pitch + phiId];
    fReal guPhi = sampleVPhi(velPhiInput, gId, pitch);

    // Unit vector in theta and phi direction
    fReal3 eTheta = make_fReal3(cosTheta * cosPhi, cosTheta * sinPhi, -sinTheta);
    fReal3 ePhi = make_fReal3(-sinPhi, cosPhi, 0.0);

    // Circle
    fReal3 u, u_, v_;
    fReal u_norm, deltaS;
    fReal3 w_ = make_fReal3(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
    
    u = guTheta * eTheta + guPhi * ePhi;
    u_norm = length(u);
    if (u_norm > 0) {
	u_ = normalize(u);
        v_ = cross(w_, u_);
    } else {
	velThetaOutput[thetaId * pitch + phiId] = 0.0;
	return;
    }

# ifdef RUNGE_KUTTA
    // Traced halfway in phi-theta space
    deltaS = - 0.5 * u_norm * timeStepGlobal;

# ifdef USEFLOAT   
    fReal3 midx = u_ * sinf(deltaS) + w_ * cosf(deltaS);
    fReal2 midCoord = make_fReal2(safe_acos(midx.z), atan2f(midx.y, midx.x));
# else
    fReal3 midx = u_ * sin(deltaS) + w_ * cos(deltaS);
    fReal2 midCoord = make_fReal2(safe_acos(midx.z), atan2(midx.y, midx.x));
# endif
    
    fReal2 midId = midCoord * invGridLenGlobal;

    fReal muTheta = sampleVTheta(velThetaInput, midId, pitch);
    fReal muPhi = sampleVPhi(velPhiInput, midId, pitch);
    
# ifdef USEFLOAT
    fReal3 mu = make_fReal3(muTheta * cosf(midCoord.x) * cosf(midCoord.y) - muPhi * sinf(midCoord.y),
			    muTheta * cosf(midCoord.x) * sinf(midCoord.y) + muPhi * cosf(midCoord.y),
			    -muTheta * sinf(midCoord.x));
    fReal3 uCircleMid_ = u_ * cosf(deltaS) - w_ * sinf(deltaS);
# else
    fReal3 mu = make_fReal3(muTheta * cos(midCoord.x) * cos(midCoord.y) - muPhi * sin(midCoord.y),
			    muTheta * cos(midCoord.x) * sin(midCoord.y) + muPhi * cos(midCoord.y),
			    -muTheta * sin(midCoord.x));
    fReal3 uCircleMid_ = u_ * cos(deltaS) - w_ * sin(deltaS);
# endif
    
    fReal3 vCircleMid_ = cross(midx, uCircleMid_);

    fReal mguTheta = dot(mu, vCircleMid_);
    fReal mguPhi = dot(mu, uCircleMid_);

    u = mguPhi * u_ + mguTheta * v_;
    u_norm = length(u);
    if (u_norm > 0) {
	u_ = normalize(u);
        v_ = cross(w_, u_);
    } else {
	at(velThetaOutput, thetaId, phiId, pitch) = 0.0;
	return;
    }
    
# endif
    deltaS = -u_norm * timeStepGlobal;
    
# ifdef USEFLOAT
    fReal3 px = u_ * sinf(deltaS) + w_ * cosf(deltaS);
    fReal2 pCoord = make_fReal2(safe_acos(px.z), atan2f(px.y, px.x));
# else
    fReal3 px = u_ * sin(deltaS) + w_ * cos(deltaS);
    fReal2 pCoord = make_fReal2(safe_acos(px.z), atan2(px.y, px.x));
# endif
    fReal2 pId = pCoord * invGridLenGlobal;
    
    fReal puTheta = sampleVTheta(velThetaInput, pId, pitch);
    fReal puPhi = sampleVPhi(velPhiInput, pId, pitch);
    
# ifdef USEFLOAT
    fReal3 pu = make_fReal3(puTheta * cosf(pCoord.x) * cosf(pCoord.y) - puPhi * sinf(pCoord.y),
			    puTheta * cosf(pCoord.x) * sinf(pCoord.y) + puPhi * cosf(pCoord.y),
			    -puTheta * sinf(pCoord.x));
	
    fReal3 uCircleP_ = u_ * cosf(deltaS) - w_ * sinf(deltaS);
# else
    fReal3 pu = make_fReal3(puTheta * cos(pCoord.x) * cos(pCoord.y) - puPhi * sin(pCoord.y),
			    puTheta * cos(pCoord.x) * sin(pCoord.y) + puPhi * cos(pCoord.y),
			    -puTheta * sin(pCoord.x));
	
    fReal3 uCircleP_ = u_ * cos(deltaS) - w_ * sin(deltaS);
# endif
    fReal3 vCircleP_ = cross(px, uCircleP_);

    puTheta = dot(pu, vCircleP_);
    puPhi = dot(pu, uCircleP_);

    pu = puPhi * u_ + puTheta * v_;
    at(velThetaOutput, thetaId, phiId, pitch) = dot(pu, eTheta);
}


/**
 * advect vectors on cartesian grid 
 * or test advection of vectors on sphere
 */
__global__ void advectionVPhiKernel
(fReal* attributeOutput, fReal* velPhi, fReal* velTheta, size_t pitch)
{
    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;
    
    // Coord in vel phi space
    fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + vPhiOffset;

    fReal2 traceId = traceRK3(velTheta, velPhi, timeStepGlobal, gId, pitch);

    attributeOutput[thetaId * pitch + phiId] = sampleVPhi(velPhi, traceId, pitch);
};


/**
 * advect vectors on cartesian grid 
 * or test advection of vectors on sphere
 */
__global__ void advectionVThetaKernel
(fReal* attributeOutput, fReal* velPhi, fReal* velTheta, size_t pitch)
{
    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;
    
    // Coord in vel theta space
    fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + vThetaOffset;

    fReal2 traceId = traceRK3(velTheta, velPhi, timeStepGlobal, gId, pitch);

    attributeOutput[thetaId * pitch + phiId] = sampleVTheta(velTheta, traceId, pitch);
}


/**
 * advect scalar
 */
__global__ void advectionCentered
(fReal* attributeOutput, fReal* attributeInput, fReal* velPhi, fReal* velTheta, size_t pitch)
{
    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;
    
    // Coord in scalar space
    fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + centeredOffset;

# ifdef greatCircle
    fReal2 traceId = traceGreatCircle(velTheta, velPhi, timeStepGlobal, gId, pitch);
# else
    fReal2 traceId = traceRK3(velTheta, velPhi, timeStepGlobal, gId, pitch);
# endif

    at(attributeOutput, thetaId, phiId, pitch)
	= sampleCentered(attributeInput, traceId, pitch);
}


/**
 * advect all scalars
 */
__global__ void advectionAllCentered
(fReal* thicknessOutput, fReal* thicknessInput, fReal* gammaOutput, fReal* gammaInput,
 fReal* velPhi, fReal* velTheta, size_t pitch)
{
    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;
    
    // Coord in scalar space
    fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + centeredOffset;

# ifdef greatCircle
    fReal2 traceId = traceGreatCircle(velTheta, velPhi, timeStepGlobal, gId, pitch);
# else
     fReal2 traceId = traceRK3(velTheta, velPhi, timeStepGlobal, gId, pitch);
# endif

    at(thicknessOutput, thetaId, phiId, pitch)
	= sampleCentered(thicknessInput, traceId, pitch);
    at(gammaOutput, thetaId, phiId, pitch)
	= sampleCentered(gammaInput, traceId, pitch);
    // if (thetaId < 5)
    // 	printf("thetaId %d %d traceId %f %f\n", thetaId, phiId, traceId.x, traceId.y);
}


__global__ void advectionCenteredBimocq
(fReal* thicknessOutput, fReal* thicknessInput, fReal* thicknessInit, fReal* thicknessDelta,
 fReal* thicknessInitLast, fReal* thicknessDeltaLast, fReal* velTheta, fReal* velPhi,
 fReal* bwd_t, fReal* bwd_p, fReal* bwd_tprev, fReal* bwd_pprev, size_t pitch) {
    fReal w[5] = {0.125, 0.125, 0.125, 0.125, 0.5};
    fReal2 dir[5] = {make_fReal2(-0.25,-0.25),
		     make_fReal2(0.25, -0.25),
		     make_fReal2(-0.25, 0.25),
		     make_fReal2( 0.25, 0.25),
		     make_fReal2(0., 0.)};

    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    // Coord in scalar space
    fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + centeredOffset;

    // if (thetaId < 0.0625 * fReal(nThetaGlobal) || thetaId > 0.9375 * fReal(nThetaGlobal)) {
    // 	fReal2 traceId = traceGreatCircle(velTheta, velPhi, timeStepGlobal, gId, pitch);
    // 	at(thicknessOutput, thetaId, phiId, pitch)
    // 	    = sampleCentered(thicknessInput, traceId, pitch);
    // } else {
	fReal thickness = 0.0;
	for (int i = 0; i < 5; i++) {
	    fReal2 posId = gId + dir[i];
	    fReal2 initPosId = sampleMapping(bwd_t, bwd_p, posId);
	    fReal2 lastPosId = sampleMapping(bwd_tprev, bwd_pprev, initPosId);
	    thickness += (1.0 - blend_coeff) * w[i] * (sampleCentered(thicknessInitLast, lastPosId, pitch) +
						       sampleCentered(thicknessDelta, initPosId, pitch) +
						       sampleCentered(thicknessDeltaLast, lastPosId, pitch)); 
	    thickness += blend_coeff * w[i] * (sampleCentered(thicknessInit, initPosId, pitch) +
					       sampleCentered(thicknessDelta, initPosId, pitch));
	}
	at(thicknessOutput, thetaId, phiId, pitch) = thickness;
    // }
}


// __global__ void advectionVThetaBimocq
// (fReal* velThetaOutput, fReal* velThetaInput, fReal* velThetaInit, fReal* velThetaDelta,
//  fReal* velPhi, fReal* bwd_t, fReal* bwd_p, size_t pitch) {
//     fReal w[5] = {0.125, 0.125, 0.125, 0.125, 0.5};
//     fReal2 dir[5] = {make_fReal2(-0.25,-0.25),
// 		     make_fReal2(0.25, -0.25),
// 		     make_fReal2(-0.25, 0.25),
// 		     make_fReal2( 0.25, 0.25),
// 		     make_fReal2(0., 0.)};

//     // Index
//     int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
//     int threadSequence = blockIdx.x % splitVal;
//     int phiId = threadIdx.x + threadSequence * blockDim.x;
//     int thetaId = blockIdx.x / splitVal;
//     if (phiId >= nPhiGlobal) return;

//     // Coord in velTheta space
//     fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + vThetaOffset;

//     if (thetaId < fReal(nThetaGlobal) / 16.f || thetaId > fReal(nThetaGlobal) * 15.f / 16.f) {
//     	fReal2 traceId = traceRK3(velThetaInput, velPhi, timeStepGlobal, gId, pitch);
//     	at(velThetaOutput, thetaId, phiId, pitch) = sampleVTheta(velThetaInput, traceId, pitch);
//     } else {
// 	fReal v = 0.0;
// 	for (int i = 0; i < 5; i++) {
// 	    fReal2 posId = gId + dir[i];
// 	    fReal2 initPosId = sampleMapping(bwd_t, bwd_p, posId);
// 	    v += w[i] * (sampleVTheta(velThetaInit, initPosId, pitch) +
// 			 sampleVTheta(velThetaDelta, initPosId, pitch));
// 	}
// 	at(velThetaOutput, thetaId, phiId, pitch) = v;
//     }
// } 


// __global__ void advectionVPhiBimocq
// (fReal* velPhiOutput, fReal* velPhiInput, fReal* velPhiInit, fReal* velPhiDelta,
//  fReal* velTheta, fReal* bwd_t, fReal* bwd_p, size_t pitch) {
//     fReal w[5] = {0.125, 0.125, 0.125, 0.125, 0.5};
//     fReal2 dir[5] = {make_fReal2(-0.25,-0.25),
// 		     make_fReal2(0.25, -0.25),
// 		     make_fReal2(-0.25, 0.25),
// 		     make_fReal2( 0.25, 0.25),
// 		     make_fReal2(0.0, 0.0)};

//     // Index
//     int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
//     int threadSequence = blockIdx.x % splitVal;
//     int phiId = threadIdx.x + threadSequence * blockDim.x;
//     int thetaId = blockIdx.x / splitVal;
//     if (phiId >= nPhiGlobal) return;

//     // Coord in uPhi space
//     fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + vPhiOffset;

//     if (thetaId < fReal(nThetaGlobal) / 16.f || thetaId > fReal(nThetaGlobal) * 15.f / 16.f) {
//     	fReal2 traceId = traceRK3(velTheta, velPhiInput, timeStepGlobal, gId, pitch);
//     	at(velPhiOutput, thetaId, phiId, pitch) = sampleVPhi(velPhiInput, traceId, pitch);
//     } else {
// 	fReal u = 0.0;
// 	for (int i = 0; i < 5; i++) {
// 	    fReal2 posId = gId + dir[i];
// 	    fReal2 initPosId = sampleMapping(bwd_t, bwd_p, posId);
// 	    u += w[i] * (sampleVPhi(velPhiInit, initPosId, pitch) +
// 			 sampleVPhi(velPhiDelta, initPosId, pitch));
// 	}
//         at(velPhiOutput, thetaId, phiId, pitch) = u;
//     }
// }


/**
 * return the maximal absolute value in array with nTheta rows and nPhi cols
 */
fReal KaminoSolver::maxAbs(fReal* array, size_t nTheta, size_t nPhi) {
    fReal *max, result;
    CHECK_CUDA(cudaMalloc(&max, MAX_BLOCK_SIZE * sizeof(fReal)));
    CHECK_CUDA(cudaMemset(max, 0, MAX_BLOCK_SIZE * sizeof(fReal)));

    dim3 gridLayout;
    dim3 blockLayout;
    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    if (gridLayout.x > MAX_BLOCK_SIZE) {
	gridLayout.x = MAX_BLOCK_SIZE;
	blockLayout.x = N / MAX_BLOCK_SIZE;
	// TODO: check whether blockLayout.x > deviceProp.maxThreadsDim[0]
    }
    maxValKernel<<<gridLayout, blockLayout>>>(max, array);
    CHECK_CUDA(cudaDeviceSynchronize());
    maxValKernel<<<1, blockLayout>>>(max, max);
    CHECK_CUDA(cudaMemcpy(&result, max, sizeof(fReal),
     			  cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(max));
    return result;
}


void KaminoSolver::updateForward(fReal dt, fReal* &fwd_t, fReal* &fwd_p) {
    bool substep = false;
    fReal T = 0.0;
    fReal dt_ = dt / radius; // scaled; assume U = 1
    
    dim3 gridLayout;
    dim3 blockLayout;
    determineLayout(gridLayout, blockLayout, nTheta, nPhi);

    if (substep) {
	fReal substep = std::min(cfldt, dt_); // cfl < 1 required
	while (T < dt_) {
	    if (T + substep > dt_) substep = dt_ - T;
	    updateMappingKernel<<<gridLayout, blockLayout>>>
		(velTheta->getGPUThisStep(), velPhi->getGPUThisStep(), -substep,
		 fwd_t, fwd_p, tmp_t, tmp_p, pitch);
	    T += substep;
	    CHECK_CUDA(cudaGetLastError());
	    CHECK_CUDA(cudaDeviceSynchronize());
	    std::swap(fwd_t, tmp_t);
	    std::swap(fwd_p, tmp_p);
	}
    } else {
	updateMappingKernel<<<gridLayout, blockLayout>>>
	    (velTheta->getGPUThisStep(), velPhi->getGPUThisStep(), -dt_,
	     fwd_t, fwd_p, tmp_t, tmp_p, pitch);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	std::swap(fwd_t, tmp_t);
	std::swap(fwd_p, tmp_p);
    }    
}


void KaminoSolver::updateBackward(fReal dt, fReal* &bwd_t, fReal* &bwd_p) {
    bool substep = false;
    fReal T = 0.0;
    fReal dt_ = dt / radius; // scaled; assume U = 1

    dim3 gridLayout;
    dim3 blockLayout;
    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    if (substep) {
	fReal substep = std::min(cfldt, dt_); // cfl < 1 required
	while (T < dt_) {
	    if (T + substep > dt_) substep = dt_ - T;
	    updateMappingKernel<<<gridLayout, blockLayout>>>
		(velTheta->getGPUThisStep(), velPhi->getGPUThisStep(), substep,
		 bwd_t, bwd_p, tmp_t, tmp_p, pitch);
	    T += substep;
	    CHECK_CUDA(cudaGetLastError());
	    CHECK_CUDA(cudaDeviceSynchronize());
	    std::swap(bwd_t, tmp_t);
	    std::swap(bwd_p, tmp_p);
	}
    } else {
	updateMappingKernel<<<gridLayout, blockLayout>>>
	    (velTheta->getGPUThisStep(), velPhi->getGPUThisStep(), dt_,
	     bwd_t, bwd_p, tmp_t, tmp_p, pitch);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	std::swap(bwd_t, tmp_t);
	std::swap(bwd_p, tmp_p);
    }
}


// TODO: delete
void KaminoSolver::updateCFL(){
    // values in padding are zero
    this->maxu = maxAbs(velPhi->getGPUThisStep(), velPhi->getNTheta(),
			velPhi->getThisStepPitchInElements());
    this->maxv = maxAbs(velTheta->getGPUThisStep(), velTheta->getNTheta(),
			velTheta->getThisStepPitchInElements());

    this->cfldt = gridLen / std::max(std::max(maxu, maxv), (fReal)eps);
    // std::cout << "this->timeStep " << this->timeStep;
    // std::cout << "max traveled distance " << std::max(maxu, maxv) * this->timeStep;
    // std::cout << "max traveled grids " << std::max(maxu, maxv) * this->timeStep / (radius * gridLen);
}


__global__ void estimateDistortionKernel(fReal* map1_t, fReal* map1_p,
					 fReal* map2_t, fReal* map2_p, fReal* result) {
    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    int2 Id = make_int2(thetaId, phiId);
    
    // Coord in scalar space
    fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + centeredOffset;

    // sample map2 using the entries of map 1
    fReal2 pos1 = make_fReal2(at(map1_t, Id), at(map1_p, Id));
    fReal2 pos2 = sampleMapping(map2_t, map2_p, pos1);

    at(result, Id) = dist(gId, pos2);
}


fReal KaminoSolver::estimateDistortion() {
    dim3 gridLayout;
    dim3 blockLayout;
    determineLayout(gridLayout, blockLayout, nTheta, nPhi);

    // forward then backward, result saved to tmp_t
    estimateDistortionKernel<<<gridLayout, blockLayout>>>
    	(forward_t, forward_p, backward_t, backward_p, tmp_t);
    // backward then forward, result saved to tmp_p
    estimateDistortionKernel<<<gridLayout, blockLayout>>>
    	(backward_t, backward_p, forward_t, forward_p, tmp_p);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    return max(maxAbs(tmp_t, nTheta, nPhi), maxAbs(tmp_p, nTheta, nPhi));
}


void KaminoSolver::advection()
{
    dim3 gridLayout;
    dim3 blockLayout;
    
    // Advect velTheta
    determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
    // if (useBimocq) {
    // 	advectionVThetaBimocq<<<gridLayout, blockLayout>>>
    // 	    (velTheta->getGPUNextStep(), velTheta->getGPUThisStep(), velTheta->getGPUInit(), velTheta->getGPUDelta(),
    // 	     velPhi->getGPUThisStep(), backward_t, backward_p, pitch);
    // } else {
    // # ifdef greatCircle
    advectionVSphereThetaKernel<<<gridLayout, blockLayout>>>
	(velTheta->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), velTheta->getNextStepPitchInElements());
    // # else
    // 	advectionVThetaKernel<<<gridLayout, blockLayout>>>
    // 	    (velTheta->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), velTheta->getNextStepPitchInElements());
    // # endif
    //}
    checkCudaErrors(cudaGetLastError());
    
    // Advect Phi
    determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());
    // if (useBimocq) {
    // 	advectionVPhiBimocq<<<gridLayout, blockLayout>>>
    // 	    (velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), velPhi->getGPUInit(), velPhi->getGPUDelta(),
    // 	     velTheta->getGPUThisStep(), backward_t, backward_p, pitch);
    // } else {
    // # ifdef greatCircle
    advectionVSpherePhiKernel<<<gridLayout, blockLayout>>>
	(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), velPhi->getNextStepPitchInElements());
    // # else
    // 	advectionVPhiKernel<<<gridLayout, blockLayout>>>
    // 	    (velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), velPhi->getNextStepPitchInElements());
    // # endif
    //}
    checkCudaErrors(cudaGetLastError());


    // Advect concentration
    determineLayout(gridLayout, blockLayout, surfConcentration->getNTheta(), surfConcentration->getNPhi());
# ifdef BIMOCQ
    advectionCenteredBimocq<<<gridLayout, blockLayout>>>
	(thickness->getGPUNextStep(), thickness->getGPUThisStep(), thickness->getGPUInit(),
	 thickness->getGPUDelta(), thickness->getGPUInitLast(),
	 thickness->getGPUDeltaLast(), velTheta->getGPUThisStep(), velPhi->getGPUThisStep(),
	 backward_t, backward_p, backward_tprev, backward_pprev, pitch);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    advectionCentered<<<gridLayout, blockLayout>>>
	(surfConcentration->getGPUNextStep(), surfConcentration->getGPUThisStep(),
	 velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), pitch);
# else
    advectionAllCentered<<<gridLayout, blockLayout>>>
	(thickness->getGPUNextStep(), thickness->getGPUThisStep(),
	 surfConcentration->getGPUNextStep(), surfConcentration->getGPUThisStep(),
	 velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), pitch);
# endif
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    thickness->swapGPUBuffer();	
	
    surfConcentration->swapGPUBuffer();
    swapVelocityBuffers();
}


// div(u) at cell center
__global__ void divergenceKernel
(fReal* div, fReal* velPhi, fReal* velTheta,
 size_t velPhiPitchInElements, size_t velThetaPitchInElements)
{
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    fReal thetaCoord = ((fReal)thetaId + centeredThetaOffset) * gridLenGlobal;

    fReal uEast = 0.0;
    fReal uWest = 0.0;
    fReal vNorth = 0.0;
    fReal vSouth = 0.0;

    fReal halfStep = 0.5 * gridLenGlobal;

    fReal thetaSouth = thetaCoord + halfStep;
    fReal thetaNorth = thetaCoord - halfStep;

    int phiIdWest = phiId;
    int phiIdEast = (phiIdWest + 1) % nPhiGlobal;

    uWest = velPhi[thetaId * velPhiPitchInElements + phiIdWest];
    uEast = velPhi[thetaId * velPhiPitchInElements + phiIdEast];
    
    if (thetaId != 0) {
	size_t thetaNorthIdx = thetaId - 1;
	vNorth = velTheta[thetaNorthIdx * velThetaPitchInElements + phiId];
    } 
    if (thetaId != nThetaGlobal - 1) {
	size_t thetaSouthIdx = thetaId;
	vSouth = velTheta[thetaSouthIdx * velThetaPitchInElements + phiId];
    }
    // otherwise sin(theta) = 0;

#ifdef sphere
    fReal invGridSine = 1.0 / sinf(thetaCoord);
    fReal sinNorth = sinf(thetaNorth);
    fReal sinSouth = sinf(thetaSouth);
    fReal factor = invGridSine * invGridLenGlobal;
    fReal termTheta = factor * (vSouth * sinSouth - vNorth * sinNorth);
#else
    fReal factor = invGridLenGlobal;
    fReal termTheta = factor * (vSouth  - vNorth);
# endif
# ifdef sphere
    fReal termPhi = invGridLenGlobal * (uEast - uWest);
# else
    fReal termPhi = factor * (uEast - uWest);
# endif

    fReal f = termTheta + termPhi;

    // if (thetaId == nThetaGlobal - 1 && phiId < 8) {
    // 	printf("phiId %d vNorth %f vSouth %f termTheta %f termPhi %f div %f\n", phiId,	vNorth , vSouth , termTheta , termPhi , f);
    // }
    div[thetaId * nPhiGlobal + phiId] = f;
}


// compute divergence using gamma
__global__ void divergenceKernel_fromGamma(fReal* div, fReal* gammaNext, fReal* gammaThis,
					   size_t pitch) {
    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    fReal gamma_a = gammaThis[thetaId * pitch + phiId];
    fReal gamma = gammaNext[thetaId * pitch + phiId];

    div[thetaId * nPhiGlobal + phiId] = (1 - gamma / gamma_a) / timeStepGlobal;
}


__global__ void concentrationLinearSystemKernel
(fReal* velPhi_a, fReal* velTheta_a, fReal* gamma_a, fReal* eta_a,
 fReal* W,  fReal* uair, fReal* vair,
 fReal* val, fReal* rhs, size_t pitch) {
    // TODO: pre-compute eta???
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    int idx = thetaId * nPhiGlobal + phiId;
    int idx5 = 5 * idx;

    fReal gamma = at(gamma_a, thetaId, phiId, pitch);
    fReal invDt = 1.0 / timeStepGlobal;

    fReal gTheta = ((fReal)thetaId + centeredThetaOffset) * gridLenGlobal;
    fReal gPhi = ((fReal)phiId + centeredPhiOffset) * gridLenGlobal;
    fReal halfStep = 0.5 * gridLenGlobal;
    fReal sinThetaSouth = sinf(gTheta + halfStep);
    fReal sinThetaNorth = sinf(gTheta - halfStep);
    fReal sinTheta = sinf(gTheta);
    fReal cscTheta = 1. / sinTheta;
    fReal cosTheta = cosf(gTheta);

    // neighboring values
    int phiIdEast = (phiId + 1) % nPhiGlobal;
    int phiIdWest = (phiId - 1 + nPhiGlobal) % nPhiGlobal;
    fReal eta = at(eta_a, thetaId, phiId, pitch);
    fReal etaWest = at(eta_a, thetaId, phiIdWest, pitch);
    fReal etaEast = at(eta_a, thetaId, phiIdEast, pitch);
    fReal etaNorth = 0.0;
    fReal etaSouth = 0.0;
    fReal WMid = at(W, thetaId, phiId, pitch);
    fReal WWest = at(W, thetaId, phiIdWest, pitch);
    fReal WEast = at(W, thetaId, phiIdEast, pitch);
    fReal WNorth = 0.0;
    fReal WSouth = 0.0;
    fReal uWest = at(velPhi_a, thetaId, phiId, pitch);
    fReal uEast = at(velPhi_a, thetaId, phiIdEast, pitch);
    fReal uNorth = 0.0;
    fReal uSouth = 0.0;
    fReal vWest = sampleVTheta(velTheta_a, make_fReal2(gTheta, gPhi - halfStep), pitch);
    fReal vEast = sampleVTheta(velTheta_a, make_fReal2(gTheta, gPhi + halfStep), pitch);
    fReal vNorth = 0.0;
    fReal vSouth = 0.0;
    fReal uAirWest = at(uair, thetaId, phiId);
    fReal uAirEast = at(uair, thetaId, phiIdEast);
    fReal vAirNorth = 0.0;
    fReal vAirSouth = 0.0;

    if (thetaId != 0) {
	int thetaNorthIdx = thetaId - 1;
	vAirNorth = at(vair, thetaNorthIdx, phiId);
	vNorth = at(velTheta_a, thetaNorthIdx, phiId, pitch);
	uNorth = sampleVPhi(velPhi_a, make_fReal2(gTheta - halfStep, gPhi), pitch);
    	etaNorth = at(eta_a, thetaNorthIdx, phiId, pitch);
	WNorth = at(W, thetaNorthIdx, phiId, pitch);
    } else {
    	int oppositePhiId = (phiId + nThetaGlobal) % nPhiGlobal;
    	etaNorth = eta_a[oppositePhiId];
	WNorth = W[oppositePhiId];
    }
    if (thetaId != nThetaGlobal - 1) {
	vAirSouth = at(vair, thetaId, phiId);
	vSouth = at(velTheta_a, thetaId, phiId, pitch);
	uSouth = sampleVPhi(velPhi_a, make_fReal2(gTheta + halfStep, gPhi), pitch);
    	etaSouth = at(eta_a, thetaId + 1, phiId, pitch);
	WSouth = at(W, thetaId + 1, phiId, pitch);
    } else {
    	int oppositePhiId = (phiId + nThetaGlobal) % nPhiGlobal;
    	etaSouth = at(eta_a, thetaId, oppositePhiId, pitch);
	WSouth = at(W, thetaId, oppositePhiId, pitch);
    }
    // at both poles sin(theta) = 0;

    // constant for this grid
    fReal CrDt = CrGlobal * timeStepGlobal; // Cr\Delta t
    fReal MDt = MGlobal * timeStepGlobal; // M\Delta t
    fReal s2 = invGridLenGlobal * invGridLenGlobal; // \Delta s^2
	
    // up
    fReal etaUp = (etaNorth + eta) / 2.0;
    val[idx5] = -s2 * sinThetaNorth * MDt / (etaUp + CrDt);

    // left
    fReal etaLeft = (etaWest + eta) / 2.0;
    val[idx5 + 1] = -s2 * cscTheta * MDt / (etaLeft + CrDt);

    // right
    fReal etaRight = (etaEast + eta) / 2.0;
    val[idx5 + 3] = -s2 * cscTheta * MDt / (etaRight + CrDt);
    
    // down
    fReal etaDown = (etaSouth + eta) / 2.0;
    val[idx5 + 4] = -s2 * sinThetaSouth * MDt / (etaDown + CrDt);

    // center
    val[idx5 + 2] = sinTheta / gamma * invDt
	- (val[idx5] + val[idx5 + 1] + val[idx5 + 3] + val[idx5 + 4]);
 
    // rhs
    // \sin\theta * div
    // u^*
    fReal sinThetaDiv = invGridLenGlobal *
	(uEast * etaRight / (etaRight + CrGlobal * timeStepGlobal) -
	 uWest * etaLeft / (etaLeft + CrGlobal * timeStepGlobal) +
	 (vSouth * sinThetaSouth * etaDown / (etaDown + CrGlobal * timeStepGlobal) -
	  vNorth * sinThetaNorth * etaUp / (etaUp + CrGlobal * timeStepGlobal)));

    //    van der Waals
    sinThetaDiv -= invGridLenGlobal * invGridLenGlobal *
    	(cscTheta * ((WEast - WMid) * etaRight / (invDt * etaRight + CrGlobal) -
    		     (WMid - WWest) * etaLeft/ (invDt * etaLeft + CrGlobal)) +
    	 sinThetaSouth * (WSouth - WMid) * etaDown / (invDt * etaDown + CrGlobal) -
    	 sinThetaNorth * (WMid - WNorth) * etaUp / (invDt * etaUp + CrGlobal));

# ifdef air
    sinThetaDiv += invGridLenGlobal * (uAirEast / (etaRight / CrGlobal * invDt + 1.0) -
				       uAirWest / (etaLeft / CrGlobal * invDt + 1.0) +
				       vAirSouth * sinThetaSouth / (etaDown / CrGlobal * invDt + 1.0) -
				       vAirNorth * sinThetaNorth / (etaUp / CrGlobal * invDt + 1.0));
# endif
    
# ifdef gravity
    sinThetaDiv += gGlobal * invGridLenGlobal *
    	(sinThetaSouth * sinThetaSouth / (invDt + CrGlobal / etaDown) -
    	 sinThetaNorth * sinThetaNorth / (invDt + CrGlobal / etaUp));
# else
# ifdef tiltedGravity
    // sinThetaDiv += 0.57735026919 * gGlobal * invGridLenGlobal *
    // 	((cosf(gPhi + halfStep) - sinf(gPhi + halfStep)) * etaRight
    // 	 / (invDt * etaRight + CrGlobal) -
    // 	 (cosf(gPhi - halfStep) - sinf(gPhi - halfStep)) * etaLeft
    // 	 / (invDt * etaLeft + CrGlobal) +
    // 	 (cosf(gTheta + halfStep) * (cosf(gPhi) + sinf(gPhi)) + sinThetaSouth)
    // 	 / (invDt * etaDown + CrGlobal) * sinThetaSouth * etaDown -
    // 	 (cosf(gTheta - halfStep) * (cosf(gPhi) + sinf(gPhi)) + sinThetaNorth)
    // 	 / (invDt * etaUp + CrGlobal) * sinThetaNorth * etaUp);

    sinThetaDiv += gGlobal * invGridLenGlobal *
     	 (sinThetaSouth * cosf(gTheta + halfStep) * sinf(gPhi) / (invDt + CrGlobal / etaDown) -
     	  sinThetaNorth * cosf(gTheta - halfStep) * sinf(gPhi) / (invDt + CrGlobal / etaUp) +
     	  cosf(gPhi + halfStep) / (invDt + CrGlobal / etaRight) -
     	  cosf(gPhi - halfStep) / (invDt + CrGlobal / etaLeft));

    // sinThetaDiv += gGlobal * invGridLenGlobal *
    // 	 (sinThetaSouth * cosf(gTheta + halfStep) * cosf(gPhi) / (invDt + CrGlobal / etaDown) -
    // 	  sinThetaNorth * cosf(gTheta - halfStep) * cosf(gPhi) / (invDt + CrGlobal / etaUp) -
    // 	  sinf(gPhi + halfStep) / (invDt + CrGlobal / etaRight) +
    // 	  sinf(gPhi - halfStep) / (invDt + CrGlobal / etaLeft));

# endif
# endif
    rhs[idx] = sinTheta * invDt - sinThetaDiv;
}


void KaminoSolver::AlgebraicMultiGridCG() {
    CHECK_CUDA(cudaMemcpy2D(d_x, nPhi * sizeof(fReal), surfConcentration->getGPUThisStep(),
			    surfConcentration->getThisStepPitchInElements() * sizeof(fReal),
			    nPhi * sizeof(fReal), nTheta,
			    cudaMemcpyDeviceToDevice));
    AMGX_vector_upload(b, N, 1, rhs);
    AMGX_vector_upload(x, N, 1, d_x);
    AMGX_matrix_upload_all(A, N, nz, 1, 1, row_ptr, col_ind, val, 0);
    AMGX_solver_setup(solver, A);
    AMGX_solver_solve(solver, b, x);
    AMGX_vector_download(x, d_x);
    CHECK_CUDA(cudaMemcpy2D(surfConcentration->getGPUNextStep(),
			    surfConcentration->getNextStepPitchInElements() * sizeof(fReal),
			    d_x, nPhi * sizeof(fReal), nPhi * sizeof(fReal), nTheta,
			    cudaMemcpyDeviceToDevice));
    int num_iter;
    AMGX_solver_get_iterations_number(solver, &num_iter);
    std::cout <<  "Total Iterations:  " << num_iter << std::endl;
    if (num_iter > 100)
    	setBroken(true);
}

void KaminoSolver::AMGCLSolve() {
    // copy data to CPU
    CHECK_CUDA(cudaMemcpy2D(xx.data(), nPhi * sizeof(fReal), surfConcentration->getGPUThisStep(),
			    surfConcentration->getThisStepPitchInElements() * sizeof(fReal),
			    nPhi * sizeof(fReal), nTheta,
			    cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(rhs_cpu.data(), rhs, sizeof(fReal) * N, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(val_cpu.data(), val, sizeof(fReal) * nz, cudaMemcpyDeviceToHost));

    // setup AMGCL
    AMGCLSolver::params sprm;
    sprm.solver.tol = 1e-6;
    Backend::params bprm;
    cusparseCreate(&bprm.cusparse_handle);

    AMGCLSolver solve(std::tie(N, ptr, col, val_cpu), sprm, bprm);

    auto f_b = Backend::copy_vector(rhs_cpu, bprm);
    auto x_b = Backend::copy_vector(xx, bprm);

    int    iters;
    double error;

    std::tie(iters, error) = solve(*f_b, *x_b);

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << error << std::endl;

    // copy data back to GPU
    CHECK_CUDA(cudaMemcpy2D(surfConcentration->getGPUNextStep(),
			    surfConcentration->getNextStepPitchInElements() * sizeof(fReal),
			    thrust::raw_pointer_cast(x_b->data()), nPhi * sizeof(fReal), nPhi * sizeof(fReal),
			    nTheta, cudaMemcpyHostToDevice));

}

__global__ void applyforcevelthetaKernel
(fReal* velThetaOutput, fReal* velThetaInput, fReal* velThetaDelta, fReal* velPhi,
 fReal* thickness, fReal* W, fReal* concentration, fReal* vair, size_t pitch) {
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    fReal gTheta = ((fReal)thetaId + vThetaThetaOffset) * gridLenGlobal;
    fReal gPhi = ((fReal)phiId + vThetaPhiOffset) * gridLenGlobal;

    int thetaSouthId = thetaId + 1;

    fReal v1 = at(velThetaInput, thetaId, phiId, pitch);
    fReal u = sampleVPhi(velPhi, make_fReal2(gTheta, gPhi), pitch);

    fReal GammaNorth = at(concentration, thetaId, phiId, pitch);
    fReal GammaSouth = at(concentration, thetaSouthId, phiId, pitch);
    fReal EtaNorth = at(thickness, thetaId, phiId, pitch);
    fReal EtaSouth = at(thickness, thetaSouthId, phiId, pitch);
    fReal WNorth = at(W, thetaId, phiId, pitch);
    fReal WSouth = at(W, thetaSouthId, phiId, pitch);

    // value at vTheta grid
    fReal invEta = 2. / (EtaNorth + EtaSouth);

    // pGpy = \frac{\partial\Gamma}{\partial\theta};
    fReal pGpy = invGridLenGlobal * (GammaSouth - GammaNorth);

    // elasticity
    fReal f1 = -MGlobal * invEta * pGpy;
    // air friction
    fReal vAir = at(vair, thetaId, phiId);
# if defined vair && defined sphere
    vAir = (gTheta < M_hPI) * 2 * (1 - smoothstep(0.0, 5.f, fabsf(currentTimeGlobal - 5.f)))
	* sinf(gTheta) * cosf(2 * gPhi) * radiusGlobal / UGlobal; // TODO: delete
# endif
    fReal f2 = CrGlobal * invEta * vAir;

    // gravity
    fReal f3 = 0.0;
# ifdef gravity
# ifdef sphere
    f3 = gGlobal * sinf(gTheta);
# else
    f3 = gGlobal;
# endif
# endif
# ifdef tiltedGravity
    //    f3 = 0.57735026919 * (cosf(gTheta)*(cosf(gPhi) + sinf(gPhi))+sinf(gTheta)) * gGlobal;
    f3 = cosf(gTheta) * sinf(gPhi) * gGlobal;
    // f3 = cosf(gTheta) * cosf(gPhi) * gGlobal;
# endif

    fReal f4 = 0.0;

    // van der Waals
    fReal f5 = invGridLenGlobal * (WNorth - WSouth);

    at(velThetaOutput, thetaId, phiId, pitch) = (v1 / timeStepGlobal + f1 + f2 + f3 + f4 + f5)
	/ (1./timeStepGlobal + CrGlobal * invEta);
    at(velThetaDelta, thetaId, phiId) = at(velThetaOutput, thetaId, phiId, pitch) - v1;
}


__global__ void applyforcevelphiKernel
(fReal* velPhiOutput, fReal* velPhiInput, fReal* velPhiDelta, fReal* velTheta,
 fReal* thickness, fReal* W, fReal* concentration, fReal* uair, size_t pitch) {
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

# ifdef sphere
    // Coord in phi-theta space
    fReal gPhi = ((fReal)phiId + vPhiPhiOffset) * gridLenGlobal;
    fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLenGlobal;
    fReal sinTheta = sinf(gTheta);
    fReal cscTheta = 1.0 / sinTheta;
# else
    fReal sinTheta = 1.0; // no effect
# endif

    int phiWestId = (phiId - 1 + nPhiGlobal) % nPhiGlobal;

    // values at centered grid
    fReal EtaWest = at(thickness, thetaId, phiWestId, pitch);
    fReal EtaEast = at(thickness, thetaId, phiId, pitch);
    fReal GammaWest = at(concentration, thetaId, phiWestId, pitch);
    fReal GammaEast = at(concentration, thetaId, phiId, pitch);
    fReal WWest = at(W, thetaId, phiWestId, pitch);
    fReal WEast = at(W, thetaId, phiId, pitch);
    
    fReal u1 = velPhiInput[thetaId * pitch + phiId];
    fReal v = sampleVTheta(velTheta, make_fReal2(gTheta, gPhi), pitch);
        
    // value at uPhi grid
    fReal invEta = 2. / (EtaWest + EtaEast);
    
    // pGpx = frac{1}{\sin\theta}\frac{\partial\Gamma}{\partial\phi};
    fReal pGpx = invGridLenGlobal * (GammaEast - GammaWest) * cscTheta;
    
    // elasticity
    fReal f1 = -MGlobal * invEta * pGpx;
    // air friction
    fReal uAir = at(uair, thetaId, phiId);
# if defined uair && defined sphere
    // uAir = 20.0 * (1 - smoothstep(0.0, 10.0, currentTimeGlobal)) * (M_hPI - gTheta)
    // 	* expf(-10 * powf(fabsf(gTheta - M_hPI), 2.0)) * radiusGlobal
    // 	* cosf(gPhi) / UGlobal;
    uAir = 20.0 * (1.0 - smoothstep(0.0, 10.0, currentTimeGlobal)) / UGlobal * sinTheta
	* (1.0 - smoothstep(M_PI * 0.0, M_PI * 0.45, fabsf(gTheta - M_hPI))) * radiusGlobal//  +
	// 20.0 * (1.0 - smoothstep(0.0, 1.0, fabsf(currentTimeGlobal - 9.f))) * (M_hPI - gTheta)
	// * expf(-10 * powf(fabsf(gTheta - M_hPI), 2.0)) * radiusGlobal
	// * cosf(gPhi - M_hPI) / UGlobal
	;
# endif
    fReal f2 = CrGlobal * invEta * uAir;
    fReal f3 = 0.0;
# ifdef tiltedGravity
    //    f3 = 0.57735026919 * (cosf(gPhi) - sinf(gPhi)) * gGlobal;
    f3 = cosf(gPhi) * gGlobal;
    // f3 = -sinf(gPhi) * gGlobal;
# endif

    fReal f4 = 0.0;

    // van der Waals
    fReal f5 = invGridLenGlobal * (WWest - WEast) * cscTheta;
    f5 = 0.0;

    fReal f = f1 + f2 + f3 + f4 + f5;

    at(velPhiOutput, thetaId, phiId, pitch) = (u1 / timeStepGlobal + (f1 + f2 + f3 + f4 + f5))
	/ (1./timeStepGlobal + CrGlobal * invEta);
    at(velPhiDelta, thetaId, phiId) = at(velPhiOutput, thetaId, phiId, pitch) - u1;
}


// Backward Euler
// __global__ void applyforcevelthetaKernel_viscous
// (fReal* velThetaOutput, fReal* velThetaInput, fReal* velPhi, fReal* thickness,
//  fReal* concentration, fReal* divCentered, size_t pitch) {
//     int splitVal = nPhiGlobal / blockDim.x;
//     int threadSequence = blockIdx.x % splitVal;
//     int phiId = threadIdx.x + threadSequence * blockDim.x;
//     int thetaId = blockIdx.x / splitVal;

//     // Coord in phi-theta space
//     fReal gPhi = ((fReal)phiId + vPhiPhiOffset) * gridLenGlobal;
//     fReal gTheta = ((fReal)thetaId + vThetaThetaOffset) * gridLenGlobal;

//     int thetaNorthId = thetaId;
//     int thetaSouthId = thetaId + 1;
//     int phiWestId = (phiId - 1 + nPhiGlobal) % nPhiGlobal;
//     int phiEastId = (phiId + 1) % nPhiGlobal;
    
//     // -   +  v0 +  -
//     // d0  u0   u1  d1
//     // v3  +  v1 +  v4
//     // d2  u2   u3  d3
//     // -   +  v2 +  -
//     //
//     // v1 is the current velTheta
//     fReal u0 = velPhi[thetaId * pitch + phiId];
//     fReal u1 = velPhi[thetaId * pitch + phiEastId];
//     fReal u2 = velPhi[thetaSouthId * pitch + phiId];
//     fReal u3 = velPhi[thetaSouthId * pitch + phiEastId];

//     // values at centered grid
//     fReal divNorth = divCentered[thetaId * nPhiGlobal + phiId];
//     fReal divSouth = divCentered[thetaSouthId * nPhiGlobal + phiId];
//     fReal GammaNorth = concentration[thetaId * pitch + phiId];
//     fReal GammaSouth = concentration[thetaSouthId * pitch + phiId];
//     fReal DeltaNorth = thickness[thetaId * pitch + phiId];
//     fReal DeltaSouth = thickness[thetaSouthId * pitch + phiId];

//     // values at vTheta grid
//     fReal div = 0.5 * (divNorth + divSouth);
//     fReal Delta = 0.5 * (DeltaNorth + DeltaSouth);
//     fReal invDelta = 1. / Delta;
//     fReal uPhi = 0.25 * (u0 + u1 + u2 + u3);

//     // pDpx = \frac{\partial\Delta}{\partial\phi}
//     fReal d0 = thickness[thetaId * pitch + phiWestId];
//     fReal d1 = thickness[thetaId * pitch + phiEastId];
//     fReal d2 = thickness[thetaSouthId * pitch + phiWestId];
//     fReal d3 = thickness[thetaSouthId * pitch + phiEastId];
//     fReal pDpx = 0.25 * invGridLenGlobal * (d1 + d3 - d0 - d2);
//     fReal pDpy = invGridLenGlobal * (DeltaSouth - DeltaNorth);

//     // pvpy = \frac{\partial u_theta}{\partial\theta}
//     fReal v0 = 0.0;
//     fReal v1 = velThetaInput[thetaId * pitch + phiId];
//     fReal v2 = 0.0;
//     fReal v3 = velThetaInput[thetaId * pitch + phiWestId];
//     fReal v4 = velThetaInput[thetaId * pitch + phiEastId];
//     if (thetaId != 0) {
// 	size_t thetaNorthId = thetaId - 1;
// 	v0 = velThetaInput[thetaNorthId * pitch + phiId];
//     } else {
// 	size_t oppositePhiId = (phiId + nPhiGlobal / 2) % nPhiGlobal;
// 	v0 = 0.5 * (velThetaInput[thetaId * pitch + phiId] -
// 		    velThetaInput[thetaId * pitch + oppositePhiId]);
//     }
//     if (thetaId != nThetaGlobal - 2) {
// 	v2 = velThetaInput[thetaSouthId * pitch + phiId];
//     } else {
// 	size_t oppositePhiId = (phiId + nPhiGlobal / 2) % nPhiGlobal;
// 	v2 = 0.5 * (velThetaInput[thetaId * pitch + phiId] -
// 		    velThetaInput[thetaId * pitch + oppositePhiId]);
//     }

//     fReal pvpy = 0.5 * invGridLenGlobal * (v2 - v0);
//     fReal pvpyNorth = invGridLenGlobal * (v1 - v0);
//     fReal pvpySouth = invGridLenGlobal * (v2 - v1);
	
//     // pvpx = \frac{\partial u_theta}{\partial\phi}    
//     fReal pvpx = 0.5 * invGridLenGlobal * (v4 - v3);
    
//     // pupy = \frac{\partial u_phi}{\partial\theta}
//     fReal pupy = 0.5 * invGridLenGlobal * (u2 + u3 - u0 - u1);

//     // pupx = \frac{\partial u_phi}{\partial\phi}
//     fReal pupx = 0.5 * invGridLenGlobal * (u1 + u3 - u0 - u2);

//     // pupxy = \frac{\partial^2u_\phi}{\partial\theta\partial\phi}
//     fReal pupxy = invGridLenGlobal * invGridLenGlobal * (u0 + u3 - u1 - u2);

//     // pvpxx = \frac{\partial^2u_\theta}{\partial\phi^2}    
//     fReal pvpxx = invGridLenGlobal * invGridLenGlobal * (v3 + v4 - 2 * v1);
    
//     // trigonometric function
//     fReal sinTheta = sinf(gTheta);
//     fReal cscTheta = 1. / sinTheta;
//     fReal cosTheta = cosf(gTheta);
//     fReal cotTheta = cosTheta * cscTheta;

//     // stress
//     // TODO: laplace term
//     fReal sigma11North = 0. +  2 * (pvpyNorth + divNorth);
//     fReal sigma11South = 0. +  2 * (pvpySouth + divSouth);
    
//     // fReal sigma11 = 0. +  2 * pvpy + 2 * div;
//     fReal sigma22 = -2 * (pvpy - 2 * div);
//     fReal sigma12 = cscTheta * pvpx + pupy - uPhi * cotTheta;

//     // psspy = \frac{\partial}{\partial\theta}(\sin\theta\sigma_{11})
//     fReal halfStep = 0.5 * gridLenGlobal;    
//     fReal thetaSouth = gTheta + halfStep;
//     fReal thetaNorth = gTheta - halfStep;    
//     fReal sinNorth = sinf(thetaNorth);
//     fReal sinSouth = sinf(thetaSouth);    
//     fReal psspy = invGridLenGlobal * (sigma11South * sinSouth - sigma11North * sinNorth);
    
//     // pspx = \frac{\partial\sigma_{12}}{\partial\phi}
//     fReal pspx = cscTheta * pvpxx + pupxy - cotTheta * pupx;

//     // pGpy = \frac{\partial\Gamma}{\partial\theta};
//     fReal pGpy = invGridLenGlobal * (GammaSouth - GammaNorth);

//     // force terms
//     fReal f1 = uPhi * uPhi * cotTheta;
//     fReal f2 = reGlobal * cscTheta * invDelta * pDpx * sigma12;
//     fReal f3 = -MGlobal * invDelta * pGpy;
//     fReal f4 = reGlobal * invDelta * pDpy * 2 * (div + pvpy);
//     fReal f5 = reGlobal * cscTheta * (psspy + pspx - cosTheta * sigma22);
    
// # ifdef gravity
//     fReal f7 = gGlobal * sinTheta;
// # else
//     fReal f7 = 0.0;
// # endif
//     fReal vAir = 0.0;
//     fReal f6 = CrGlobal * invDelta * (vAir - v1);
    
//     // output
//     fReal result = (v1 + timeStepGlobal * (f1 + f2 + f3 + f4 + f5 + CrGlobal * vAir + f7))
// 	/ (1.0 + CrGlobal * invDelta * timeStepGlobal);
//     // if (fabsf(result) < eps)
//     // 	result = 0.0;
//     velThetaOutput[thetaId * pitch + phiId] = result;
// }


// // Backward Euler
// __global__ void applyforcevelphiKernel_viscous
// (fReal* velPhiOutput, fReal* velTheta, fReal* velPhiInput, fReal* thickness,
//  fReal* concentration, fReal* divCentered, size_t pitch) {
//     int splitVal = nPhiGlobal / blockDim.x;
//     int threadSequence = blockIdx.x % splitVal;
//     int phiId = threadIdx.x + threadSequence * blockDim.x;
//     int thetaId = blockIdx.x / splitVal;

//     // Coord in phi-theta space
//     fReal gPhi = ((fReal)phiId + vPhiPhiOffset) * gridLenGlobal;
//     fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLenGlobal;

//     int phiWestId = (phiId - 1 + nPhiGlobal) % nPhiGlobal;
//     int thetaNorthId< = thetaId - 1;
//     int thetaSouthId = thetaId + 1;

//     // values at centered grid
//     fReal divWest = divCentered[thetaId * nPhiGlobal + phiWestId];
//     fReal divEast = divCentered[thetaId * nPhiGlobal + phiId];
//     fReal DeltaWest = thickness[thetaId * pitch + phiWestId];
//     fReal DeltaEast = thickness[thetaId * pitch + phiId];
//     fReal GammaWest = concentration[thetaId * pitch + phiWestId];
//     fReal GammaEast = concentration[thetaId * pitch + phiId];
    
//     // |  d0 u3 d2 |
//     // +  v0 +  v1 +
//     // u0    u1    u2
//     // +  v2 +  v3 + 
//     // |  d1 u4 d3 |
//     //
//     // u1 is the current velPhi
//     fReal v0 = 0.0;
//     fReal v1 = 0.0;
//     fReal v2 = 0.0;
//     fReal v3 = 0.0;
//     if (thetaId != 0) {
// 	v0 = velTheta[thetaNorthId * pitch + phiWestId];
// 	v1 = velTheta[thetaNorthId * pitch + phiId];
//     } else {
// 	v0 = 0.5 * (velTheta[thetaId * pitch + phiWestId] -
// 		    velTheta[thetaId * pitch + (phiWestId + nPhiGlobal / 2) % nPhiGlobal]);
// 	v1 = 0.5 * (velTheta[thetaId * pitch + phiId] -
// 		    velTheta[thetaId * pitch + (phiId + nPhiGlobal / 2) % nPhiGlobal]);
//     }
//     if (thetaId != nThetaGlobal - 1) {
// 	v2 = velTheta[thetaId * pitch + phiWestId];
// 	v3 = velTheta[thetaId * pitch + phiId];
//     } else {
// 	v2 = 0.5 * (velTheta[thetaNorthId * pitch + phiWestId] -
// 		    velTheta[thetaNorthId * pitch + (phiWestId + nPhiGlobal / 2) % nPhiGlobal]);
// 	v3 = 0.5 * (velTheta[thetaNorthId * pitch + phiId] -
// 		    velTheta[thetaNorthId * pitch + (phiId + nPhiGlobal / 2) % nPhiGlobal]);
//     }
    
//     // values at uPhi grid
//     fReal Delta = 0.5 * (DeltaWest + DeltaEast);
//     fReal invDelta = 1. / Delta;
//     fReal div = 0.5 * (divWest + divEast);
//     fReal vTheta = 0.25 * (v0 + v1 + v2 + v3);

//     // pvpx = \frac{\partial u_theta}{\partial\phi}
//     fReal pvpx = 0.5 * invGridLenGlobal * (v1 + v3 - v0 - v2);

//     // pvpy = \frac{\partial u_theta}{\partial\theta}
//     fReal pvpyWest = invGridLenGlobal * (v2 - v0);
//     fReal pvpyEast = invGridLenGlobal * (v3 - v1);
//     fReal pvpy = 0.5 * invGridLenGlobal * (v2 + v3 - v0 - v1);

//     // pupy = \frac{\partial u_phi}{\partial\theta}
//     fReal pupyNorth = 0.0;
//     fReal pupySouth = 0.0;
//     fReal u1 = velPhiInput[thetaId * pitch + phiId];
//     fReal u3 = 0.0;
//     fReal u4 = 0.0;
//     // actually pupyNorth != 0 at theta == 0, but pupyNorth appears only
//     // in sinNorth * pupyNorth, and sinNorth = 0 at theta == 0    
//     if (thetaId != 0) {
//         u3 = velPhiInput[thetaNorthId * pitch + phiId];
// 	pupyNorth = invGridLenGlobal * (u1 - u3);
//     } else {
// 	size_t oppositePhiId = (phiId + nPhiGlobal / 2) % nPhiGlobal;
// 	u3 = -velPhiInput[thetaId * pitch + oppositePhiId];
//     }
//     // actually pupySouth != 0 at theta == \pi, but pupySouth appears only
//     // in sinSouth * pupySouth, and sinSouth = 0 at theta == \pi   
//     if (thetaId != nThetaGlobal - 1) {
// 	u4 = velPhiInput[thetaSouthId * pitch + phiId];
// 	pupySouth = invGridLenGlobal * (u4 - u1);
//     } else {
// 	size_t oppositePhiId = (phiId + nPhiGlobal / 2) % nPhiGlobal;
// 	u4 = -velPhiInput[thetaId * pitch + oppositePhiId];
//     }
//     fReal pupy = 0.5 * invGridLenGlobal * (u4 - u3);

//     // pGpx = \frac{\partial\Gamma}{\partial\phi};
//     fReal pGpx = invGridLenGlobal * (GammaEast - GammaWest);

//     // trigonometric function
//     fReal sinTheta = sinf(gTheta);
//     fReal cscTheta = 1. / sinTheta;
//     fReal cosTheta = cosf(gTheta);
//     fReal cotTheta = cosTheta * cscTheta;
    
//     // stress
//     // TODO: laplace term
//     fReal sigma12 = cscTheta * pvpx + pupy - u1 * cotTheta;

//     // pDpx = \frac{\partial\Delta}{\partial\phi}
//     fReal pDpx = invGridLenGlobal * (DeltaEast - DeltaWest);

//     // pDpy = \frac{\partial\Delta}{\partial\theta}
//     // TODO: do we need to average the thickness value at the pole?
//     fReal pDpy = 0.0;
//     fReal d0 = 0.0;
//     fReal d1 = 0.0;
//     fReal d2 = 0.0;
//     fReal d3 = 0.0;
//     if (thetaId != 0) {
// 	d0 = thickness[thetaNorthId * pitch + phiWestId];
// 	d2 = thickness[thetaNorthId * pitch + phiId];
//     } else {
// 	d0 = thickness[thetaId * pitch + (phiWestId + nPhiGlobal / 2) % nPhiGlobal];
// 	d2 = thickness[thetaId * pitch + (phiId + nPhiGlobal / 2) % nPhiGlobal];
//     }
//     if (thetaId != nThetaGlobal - 1) {
// 	d1 = thickness[thetaSouthId * pitch + phiWestId];
// 	d3 = thickness[thetaSouthId * pitch + phiId];
//     } else {
// 	d1 = thickness[thetaId * pitch + (phiWestId + nPhiGlobal / 2) % nPhiGlobal];
// 	d3 = thickness[thetaId * pitch + (phiId + nPhiGlobal / 2) % nPhiGlobal];
//     }
//     pDpy = 0.25 * invGridLenGlobal * (d1 + d3 - d0 - d2);
    
//     // psspy = \frac{\partial}{\partial\theta}(\sin\theta\sigma_{12})
//     fReal halfStep = 0.5 * gridLenGlobal;    
//     fReal thetaSouth = gTheta + halfStep;
//     fReal thetaNorth = gTheta - halfStep;    
//     fReal sinNorth = sinf(thetaNorth);
//     fReal sinSouth = sinf(thetaSouth);
//     fReal cosNorth = cosf(thetaNorth);
//     fReal cosSouth = cosf(thetaSouth);
//     // TODO: uncertain about the definintion of u_\phi at both poles
//     fReal uNorth = 0.5 * (u3 + u1);
//     fReal uSouth = 0.5 * (u1 + u4);
//     fReal psspy = invGridLenGlobal * (invGridLenGlobal * (v0 + v3 - v1 - v2) +
// 							sinSouth * pupySouth - sinNorth * pupyNorth -
// 							cosSouth * uSouth + cosNorth * uNorth);
    
//     // pspx = \frac{\partial\sigma_{22}}{\partial\phi}
//     fReal sigma22West = 2 * (2 * divWest - pvpyWest);
//     fReal sigma22East = 2 * (2 * divEast - pvpyEast);
//     fReal pspx = invGridLenGlobal * (sigma22East - sigma22West);
    
//     // force terms
//     // fReal f1 = -vTheta * u1 * cotTheta;
//     fReal f2 = reGlobal * invDelta * pDpy * sigma12;
//     fReal f3 = -MGlobal * invDelta * cscTheta * pGpx;
//     fReal f4 = reGlobal * invDelta * cscTheta * pDpx * 2 * ( 2 * div - pvpy);
//     fReal f5 = reGlobal * cscTheta * (psspy + pspx + cosTheta * sigma12);

//     // fReal f7 = 0.0; 		// gravity
//     fReal uAir = 0.0;
// # ifdef uair
//     if (currentTimeGlobal < 5)
//     	uAir = 20.0 * (M_hPI - gTheta) * expf(-10 * powf(fabsf(gTheta - M_hPI), 2.0)) * radiusGlobal * cosf(gPhi) / UGlobal;
// # endif

//     fReal f6 = CrGlobal * invDelta * (uAir - u1);
    
//     // output
//     fReal result = (u1 + timeStepGlobal * (f2 + f3 + f4 + f5 + CrGlobal * uAir))
// 	/ (1.0 + (CrGlobal * invDelta + vTheta * cotTheta) * timeStepGlobal);
//     velPhiOutput[thetaId * pitch + phiId] = result;
// }


__global__ void applyforceThickness
(fReal* thicknessOutput, fReal* thicknessInput, fReal* thicknessDelta,
 fReal* div, size_t pitch)
{
    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    fReal eta = at(thicknessInput, thetaId, phiId, pitch);
    fReal f = at(div, thetaId, phiId);

    at(thicknessOutput, thetaId, phiId, pitch) = eta * (1 - timeStepGlobal * f);
# ifdef evaporation
    at(thicknessOutput, thetaId, phiId, pitch) += evaporationRate * timeStepGlobal;
# endif
    at(thicknessDelta, thetaId, phiId) = at(thicknessOutput, thetaId, phiId, pitch) - eta;
}

// Backward Euler
// __global__ void applyforceSurfConcentration
// (fReal* sConcentrationOutput, fReal* sConcentrationInput, fReal* div, size_t pitch)
// {
//     // Index
//     int splitVal = nPhiGlobal / blockDim.x;
//     int threadSequence = blockIdx.x % splitVal;
//     int phiId = threadIdx.x + threadSequence * blockDim.x;
//     int thetaId = blockIdx.x / splitVal;

//     fReal thetaCoord = ((fReal)thetaId + centeredThetaOffset) * gridLenGlobal;
    
//     fReal halfStep = 0.5 * gridLenGlobal;

//     fReal cscTheta = 1.0 / sinf(thetaCoord);
//     fReal sinThetaSouth = sinf(thetaCoord + halfStep);
//     fReal sinThetaNorth = sinf(thetaCoord - halfStep);

//     fReal gamma = sConcentrationInput[thetaId * pitch + phiId];
//     fReal gammaWest = sConcentrationInput[thetaId * pitch + (phiId - 1 + nPhiGlobal) % nPhiGlobal];
//     fReal gammaEast = sConcentrationInput[thetaId * pitch + (phiId + 1) % nPhiGlobal];
//     fReal gammaNorth = 0.0;
//     fReal gammaSouth = 0.0;
//     if (thetaId != 0) {
//     	gammaNorth = sConcentrationInput[(thetaId - 1) * pitch + phiId];
//     } else {
//     	size_t oppositePhiId = (phiId + nPhiGlobal / 2) % nPhiGlobal;
//     	gammaNorth = sConcentrationInput[thetaId * pitch + oppositePhiId];
//     }
//     if (thetaId != nThetaGlobal - 1) {
//     	gammaSouth = sConcentrationInput[(thetaId + 1) * pitch + phiId];
//     } else {
//     	size_t oppositePhiId = (phiId + nPhiGlobal / 2) % nPhiGlobal;
//     	gammaSouth = sConcentrationInput[thetaId * pitch + oppositePhiId];
//     }
// # ifdef sphere
//     fReal laplace = invGridLenGlobal * invGridLenGlobal * cscTheta *
// 	(sinThetaSouth * (gammaSouth - gamma) - sinThetaNorth * (gamma - gammaNorth) +
// 		    cscTheta * (gammaEast + gammaWest - 2 * gamma));
// # else
//     fReal laplace = invGridLenGlobal * invGridLenGlobal * 
//     	(gammaWest - 4*gamma + gammaEast + gammaNorth + gammaSouth);
// #endif
    
//     fReal f = div[thetaId * nPhiGlobal + phiId];
//     // fReal f2 = DsGlobal * laplace;
//     fReal f2 = 0.0;

//     sConcentrationOutput[thetaId * pitch + phiId] = max((gamma + f2 * timeStepGlobal) / (1 + timeStepGlobal * f), 0.0);
// }


/**
 * a = b - c
 * b and c are pitched memory
 */
__global__ void substractPitched(fReal* a, fReal* b, fReal* c, size_t pitch) {
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    at(a, thetaId, phiId) = at(b, thetaId, phiId, pitch) - at(c, thetaId, phiId, pitch);
}


__global__ void accumulateChangesThickness(fReal* thicknessDelta, fReal* thicknessDeltaTemp,
					   fReal* fwd_t, fReal* fwd_p,
					   size_t pitch) {
    fReal w[5] = {0.125, 0.125, 0.125, 0.125, 0.5};
    fReal2 dir[5] = {make_fReal2(-0.25,-0.25),
		     make_fReal2(0.25, -0.25),
		     make_fReal2(-0.25, 0.25),
		     make_fReal2( 0.25, 0.25),
		     make_fReal2(0.0, 0.0)};

    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;
    
    // Coord in scalar space
    fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + centeredOffset;

    for(int i = 0; i < 5; i++) {
	fReal2 posId = gId + dir[i];
	fReal2 initPosId = sampleMapping(fwd_t, fwd_p, posId);
	at(thicknessDelta, thetaId, phiId, pitch)
	    += w[i] * sampleCentered(thicknessDeltaTemp, initPosId, nPhiGlobal);
    }
}


__global__ void accumulateChangesVTheta(fReal* vThetaDelta, fReal* vThetaDeltaTemp,
					fReal* fwd_t, fReal* fwd_p,
					size_t pitch) {
    fReal w[5] = {0.125, 0.125, 0.125, 0.125, 0.5};
    fReal2 dir[5] = {make_fReal2(-0.25,-0.25),
		     make_fReal2(0.25, -0.25),
		     make_fReal2(-0.25, 0.25),
		     make_fReal2( 0.25, 0.25),
		     make_fReal2(0.0, 0.0)};

    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;
    
    // Coord in vTheta space
    fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + vThetaOffset;

    for(int i = 0; i < 5; i++) {
	fReal2 posId = gId + dir[i];
	fReal2 initPosId = sampleMapping(fwd_t, fwd_p, posId);
	
	at(vThetaDelta, thetaId, phiId, pitch)
	    += w[i] * sampleVTheta(vThetaDeltaTemp, initPosId, nPhiGlobal);
    }
}

__global__ void accumulateChangesVPhi(fReal* vPhiDelta, fReal* vPhiDeltaTemp,
				      fReal* fwd_t, fReal* fwd_p,
				      size_t pitch) {
    fReal w[5] = {0.125, 0.125, 0.125, 0.125, 0.5};
    fReal2 dir[5] = {make_fReal2(-0.25,-0.25),
		     make_fReal2(0.25, -0.25),
		     make_fReal2(-0.25, 0.25),
		     make_fReal2( 0.25, 0.25),
		     make_fReal2(0.0, 0.0)};

    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;
    
    // Coord in vPhi space
    fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + vPhiOffset;

    for(int i = 0; i < 5; i++) {
	fReal2 posId = gId + dir[i];
	fReal2 initPosId = sampleMapping(fwd_t, fwd_p, posId);
	
	at(vPhiDelta, thetaId, phiId, pitch)
	    += w[i] * sampleVPhi(vPhiDeltaTemp, initPosId, nPhiGlobal);
    }
}



__global__ void correctThickness1(fReal* thicknessCurr, fReal* thicknessError, fReal* thicknessDelta,
				  fReal* thicknessInit, fReal* fwd_t, fReal* fwd_p,
				  size_t pitch) {
    fReal w[5] = {0.125, 0.125, 0.125, 0.125, 0.5};
    fReal2 dir[5] = {make_fReal2(-0.25,-0.25),
		     make_fReal2(0.25, -0.25),
		     make_fReal2(-0.25, 0.25),
		     make_fReal2( 0.25, 0.25),
		     make_fReal2(0.0, 0.0)};

    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    fReal thickness = 0.0;
    // fReal gamma = 0.0;

    // Coord in scalar space
    fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + centeredOffset;
    for (int i = 0; i < 5; i++) {
	fReal2 posId = gId + dir[i];
	fReal2 initPosId = sampleMapping(fwd_t, fwd_p, posId);
	thickness += w[i] * sampleCentered(thicknessCurr, initPosId, pitch);
	// gamma += w[i] * sampleCentered(gammaCurr, initPosId, pitch);
    }
    at(thicknessError, thetaId, phiId) = (thickness - at(thicknessDelta, thetaId, phiId, pitch)
					  - at(thicknessInit, thetaId, phiId, pitch)) * 0.5;
    // at(gammaError, thetaId, phiId) = (gamma - at(gammaDelta, thetaId, phiId, pitch)
    // 					  - at(gammaInit, thetaId, phiId, pitch)) * 0.5;
}


__global__ void correctThickness2(fReal* thicknessOutput, fReal* thicknessInput,
				 fReal* thicknessError, fReal* bwd_t, fReal* bwd_p, size_t pitch) {
    fReal w[5] = {0.125, 0.125, 0.125, 0.125, 0.5};
    fReal2 dir[5] = {make_fReal2(-0.25,-0.25),
		     make_fReal2(0.25, -0.25),
		     make_fReal2(-0.25, 0.25),
		     make_fReal2( 0.25, 0.25),
		     make_fReal2(0.0, 0.0)};

    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    // Coord in scalar space
    fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + centeredOffset;
    
    for (int i = 0; i < 5; i++) {
	fReal2 posId = gId + dir[i];
	fReal2 sampleId = sampleMapping(bwd_t, bwd_p, posId);
	at(thicknessOutput, thetaId, phiId, pitch)
	    -= w[i] * sampleCentered(thicknessError, sampleId, nPhiGlobal);
	// at(gammaOutput, thetaId, phiId, pitch)
	//     -= w[i] * sampleCentered(gammaError, sampleId, nPhiGlobal);
    }

    // clamp local extrema
    int range[] = {-1, 0, 1};

    fReal minVal = at(thicknessInput, thetaId, phiId, pitch);
    fReal maxVal = minVal;
    for (int t : range) {
    	for (int p : range) {
    	    int2 sampleId = make_int2(thetaId + t, phiId + p);
    	    validateId(sampleId);
    	    fReal currentVal = at(thicknessInput, sampleId, pitch);
    	    minVal = fminf(minVal, currentVal);
    	    maxVal = fmaxf(maxVal, currentVal);
    	}
    }
    at(thicknessOutput, thetaId, phiId, pitch)
    	= clamp(at(thicknessOutput, thetaId, phiId, pitch), minVal, maxVal);
    at(thicknessOutput, thetaId, phiId, pitch) = fmaxf(0.0, at(thicknessOutput, thetaId, phiId, pitch));

    // minVal = at(gammaInput, thetaId, phiId, pitch);
    // maxVal = 0.0;
    // for (int t : range) {
    // 	for (int p : range) {
    // 	    int2 sampleId = make_int2(thetaId + t, phiId + p);
    // 	    validateId(sampleId);
    // 	    fReal currentVal = at(gammaInput, sampleId, pitch);
    // 	    minVal = fminf(minVal, currentVal);
    // 	    maxVal = fmaxf(maxVal, currentVal);
    // 	}
    // }
    // at(gammaOutput, thetaId, phiId, pitch)
    // 	= clamp(at(gammaOutnput, thetaId, phiId, pitch), minVal, maxVal);
}

__global__ void correctVTheta1(fReal* vThetaCurr, fReal* vThetaError, fReal* vThetaDelta,
			       fReal* vThetaInit, fReal* fwd_t, fReal* fwd_p,
			       size_t pitch) {
    fReal w[5] = {0.125, 0.125, 0.125, 0.125, 0.5};
    fReal2 dir[5] = {make_fReal2(-0.25,-0.25),
		     make_fReal2(0.25, -0.25),
		     make_fReal2(-0.25, 0.25),
		     make_fReal2( 0.25, 0.25),
		     make_fReal2(0.0, 0.0)};

    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    fReal vTheta = 0.0;

    // Coord in vTheta space
    fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + vThetaOffset;
    for (int i = 0; i < 5; i++) {
	fReal2 posId = gId + dir[i];
	fReal2 initPosId = sampleMapping(fwd_t, fwd_p, posId);
	vTheta += w[i] * sampleVTheta(vThetaCurr, initPosId, pitch);
    }
    at(vThetaError, thetaId, phiId) = (vTheta - at(vThetaDelta, thetaId, phiId, pitch)
				       - at(vThetaInit, thetaId, phiId, pitch)) * 0.5;
}


__global__ void correctVTheta2(fReal* vThetaOutput, fReal* vThetaInput,
			       fReal* vThetaError, fReal* bwd_t, fReal* bwd_p, size_t pitch) {
    fReal w[5] = {0.125, 0.125, 0.125, 0.125, 0.5};
    fReal2 dir[5] = {make_fReal2(-0.25,-0.25),
		     make_fReal2(0.25, -0.25),
		     make_fReal2(-0.25, 0.25),
		     make_fReal2( 0.25, 0.25),
		     make_fReal2(0.0, 0.0)};

    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    // Coord in scalar space
    fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + vThetaOffset;

    // sample
    for (int i = 0; i < 5; i++) {
	fReal2 posId = gId + dir[i];
	fReal2 sampleId = sampleMapping(bwd_t, bwd_p, posId);
	at(vThetaInput, thetaId, phiId, pitch)
	    -= w[i] * sampleVTheta(vThetaError, sampleId, nPhiGlobal);
    }

    // clamp local extrema
    int range[] = {-1, 0, 1};

    fReal minVal = at(vThetaOutput, thetaId, phiId, pitch);
    fReal maxVal = 0.0;
    for (int t : range) {
	for (int p : range) {
	    int2 sampleId = make_int2(thetaId + t, phiId + p);
	    validateId(sampleId);
	    fReal currentVal = at(vThetaOutput, sampleId, pitch);
	    minVal = fminf(minVal, currentVal);
	    maxVal = fmaxf(maxVal, currentVal);
	}
    }
    at(vThetaOutput, thetaId, phiId, pitch)
	= clamp(at(vThetaInput, thetaId, phiId, pitch), minVal, maxVal);
}

__global__ void correctVPhi1(fReal* vPhiCurr, fReal* vPhiError, fReal* vPhiDelta,
			     fReal* vPhiInit, fReal* fwd_t, fReal* fwd_p,
			     size_t pitch) {
    fReal w[5] = {0.125, 0.125, 0.125, 0.125, 0.5};
    fReal2 dir[5] = {make_fReal2(-0.25,-0.25),
		     make_fReal2(0.25, -0.25),
		     make_fReal2(-0.25, 0.25),
		     make_fReal2( 0.25, 0.25),
		     make_fReal2(0.0, 0.0)};

    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    fReal vPhi = 0.0;

    // Coord in vPhi space
    fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + vPhiOffset;
    for (int i = 0; i < 5; i++) {
	fReal2 posId = gId + dir[i];
	fReal2 initPosId = sampleMapping(fwd_t, fwd_p, posId);
	vPhi += w[i] * sampleVPhi(vPhiCurr, initPosId, pitch);
    }
    at(vPhiError, thetaId, phiId) = (vPhi - at(vPhiDelta, thetaId, phiId, pitch)
				     - at(vPhiInit, thetaId, phiId, pitch)) * 0.5;
}


__global__ void correctVPhi2(fReal* vPhiOutput, fReal* vPhiInput,
			     fReal* vPhiError, fReal* bwd_t, fReal* bwd_p, size_t pitch) {
    fReal w[5] = {0.125, 0.125, 0.125, 0.125, 0.5};
    fReal2 dir[5] = {make_fReal2(-0.25,-0.25),
		     make_fReal2(0.25, -0.25),
		     make_fReal2(-0.25, 0.25),
		     make_fReal2( 0.25, 0.25),
		     make_fReal2(0.0, 0.0)};

    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    // Coord in scalar space
    fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + vPhiOffset;

    // sample
    for (int i = 0; i < 5; i++) {
	fReal2 posId = gId + dir[i];
	fReal2 sampleId = sampleMapping(bwd_t, bwd_p, posId);
	at(vPhiInput, thetaId, phiId, pitch)
	    -= w[i] * sampleVPhi(vPhiError, sampleId, nPhiGlobal);
    }

    // clamp local extrema
    int range[] = {-1, 0, 1};

    fReal minVal = at(vPhiOutput, thetaId, phiId, pitch);
    fReal maxVal = 0.0;
    for (int t : range) {
	for (int p : range) {
	    int2 sampleId = make_int2(thetaId + t, phiId + p);
	    validateId(sampleId);
	    fReal currentVal = at(vPhiOutput, sampleId, pitch);
	    minVal = fminf(minVal, currentVal);
	    maxVal = fmaxf(maxVal, currentVal);
	}
    }
    at(vPhiOutput, thetaId, phiId, pitch)
	= clamp(at(vPhiInput, thetaId, phiId, pitch), minVal, maxVal);
}


__global__ void vanDerWaals(fReal* W, fReal* eta, size_t pitch) {
    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    fReal invEta = 1.0 / at(eta, thetaId, phiId, pitch);
    if (invEta <= 0.0) {
	at(W, thetaId, phiId, pitch) = 0.0;
    } else {
	at(W, thetaId, phiId, pitch) = W1 * powf(invEta, 4.0) - W2 * powf(invEta, 2.0);
    }
}


__global__ void airFlowU(fReal* uair) {
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    double gPhi = ((double)phiId + vPhiPhiOffset) * gridLenGlobal;
    double gTheta = ((double)thetaId + vPhiThetaOffset) * gridLenGlobal;

    // Trigonometric functions
    double sinTheta = sin(gTheta);
    double cosTheta = cos(gTheta);
    double sinPhi = sin(gPhi);
    double cosPhi = cos(gPhi);

    double alpha = smoothstep(0.0, 5.0, currentTimeGlobal) * M_hPI;
    double3 wind = make_double3(0., sin(alpha), cos(alpha));
    double3 position = make_double3(1., 0., 0.);
    double3 normal = normalize(cross(position, wind));
    double3 currentPosition = make_double3(sinTheta * cosPhi,
					   sinTheta * sinPhi,
					   cosTheta);

    double beta = safe_acos(dot(normal, currentPosition));

    double3 projected = normalize(currentPosition - dot(normal, currentPosition) * normal);

    double sinGamma = dot(projected, position);
    double3 projectedWind = cross(normal, projected);

    // Unit vector in theta and phi direction
    double3 ePhi = make_double3(-sinPhi, cosPhi, 0.);
    double3 Air = 20. * (1. - smoothstep(0.0, 10.0, currentTimeGlobal)) / UGlobal * sin(beta)
    	* (1. - smoothstep(M_PI * 0., M_PI * 0.45, fabs(beta - M_hPI))) * radiusGlobal
    	* projectedWind;
    at(uair, thetaId, phiId) = fReal(dot(Air, ePhi));
    if (isnan(at(uair, thetaId, phiId))) // normal == currentPosition
	at(uair, thetaId, phiId) = 0.0;
}


__global__ void airFlowV(fReal* vair) {
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    double gTheta = ((double)thetaId + vThetaThetaOffset) * gridLenGlobal;
    double gPhi = ((double)phiId + vThetaPhiOffset) * gridLenGlobal;

    // Trigonometric functions
    double sinTheta = sin(gTheta);
    double cosTheta = cos(gTheta);
    double sinPhi = sin(gPhi);
    double cosPhi = cos(gPhi);

    double alpha = smoothstep(0.0, 5.0, currentTimeGlobal) * M_hPI;
    double3 wind = make_double3(0., sin(alpha), cos(alpha));
    double3 position = make_double3(1., 0., 0.);
    double3 normal = normalize(cross(position, wind));
    double3 currentPosition = make_double3(sinTheta * cosPhi,
					   sinTheta * sinPhi,
		 			   cosTheta);
    double beta = safe_acos(dot(normal, currentPosition));

    double3 projected = normalize(currentPosition - dot(normal, currentPosition) * normal);

    double sinGamma = dot(projected, position);
    double3 projectedWind = cross(normal, projected);

    // Unit vector in theta and phi direction
    double3 eTheta = make_double3(cosTheta * cosPhi, cosTheta * sinPhi, -sinTheta);
    double3 Air = 20. * (1. - smoothstep(0.0, 10.0, currentTimeGlobal)) / UGlobal * sin(beta)
	* (1. - smoothstep(M_PI * 0., M_PI * 0.45, fabs(beta - M_hPI))) * radiusGlobal
	* projectedWind;
    at(vair, thetaId, phiId) = fReal(dot(Air, eTheta));
    if (isnan(at(vair, thetaId, phiId)))
        at(vair, thetaId, phiId) = 0.0;
}


void KaminoSolver::bodyforce() {
    dim3 gridLayout;
    dim3 blockLayout;

    bool inviscid = true;

# ifdef air
    // TODO: pass position and wind as parameter
    determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());
    airFlowU<<<gridLayout, blockLayout>>>(uair);
    determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
    airFlowV<<<gridLayout, blockLayout>>>(vair);
    checkCudaErrors(cudaGetLastError());
# endif

    determineLayout(gridLayout, blockLayout, nTheta, nPhi);

    // store van der waals forces in thickness->getGPUNextstep()
    vanDerWaals<<<gridLayout, blockLayout>>>
	(thickness->getGPUNextStep(), thickness->getGPUThisStep(), pitch);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    concentrationLinearSystemKernel<<<gridLayout, blockLayout>>>
    	(velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), surfConcentration->getGPUThisStep(),
    	 thickness->getGPUThisStep(), thickness->getGPUNextStep(),
    	 uair, vair,
    	 val, rhs, thickness->getThisStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

# ifdef PERFORMANCE_BENCHMARK
    KaminoTimer CGtimer;
    CGtimer.startTimer();
# endif
    AlgebraicMultiGridCG();
    // conjugateGradient();
    // AMGCLSolve();
# ifdef PERFORMANCE_BENCHMARK
    this->CGTime += CGtimer.stopTimer() * 0.001f;
# endif
    
    determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
    applyforcevelthetaKernel<<<gridLayout, blockLayout>>>
    	(velTheta->getGPUNextStep(), velTheta->getGPUThisStep(), tmp_t, velPhi->getGPUThisStep(),
    	 thickness->getGPUThisStep(), thickness->getGPUNextStep(),
	 surfConcentration->getGPUNextStep(), vair, pitch);
    checkCudaErrors(cudaGetLastError());

    // accumulateChangesVTheta<<<gridLayout, blockLayout>>>
    // 	(velTheta->getGPUDelta(), tmp_t, forward_t, forward_p, pitch);
    // checkCudaErrors(cudaGetLastError());

    determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());
    applyforcevelphiKernel<<<gridLayout, blockLayout>>>
    	(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), tmp_p, velTheta->getGPUThisStep(),
	 thickness->getGPUThisStep(), thickness->getGPUNextStep(),
	 surfConcentration->getGPUNextStep(), uair, pitch);
    checkCudaErrors(cudaGetLastError());

    // accumulateChangesVPhi<<<gridLayout, blockLayout>>>
    // 	(velPhi->getGPUDelta(), tmp_p, forward_t, forward_p, pitch);
    // checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    divergenceKernel_fromGamma<<<gridLayout, blockLayout>>>
    	(div, surfConcentration->getGPUNextStep(), surfConcentration->getGPUThisStep(),
    	 surfConcentration->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // write thickness difference to tmp_t, gamma difference to tmp_p
    determineLayout(gridLayout, blockLayout, thickness->getNTheta(), thickness->getNPhi());
    applyforceThickness<<<gridLayout, blockLayout>>>
	(thickness->getGPUNextStep(), thickness->getGPUThisStep(),
	 tmp_t, div, pitch);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    accumulateChangesThickness<<<gridLayout, blockLayout>>>
	(thickness->getGPUDelta(), tmp_t, forward_t, forward_p, pitch);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    thickness->swapGPUBuffer();
    surfConcentration->swapGPUBuffer();
    swapVelocityBuffers();
}


void KaminoSolver::reInitializeMapping() {
    dim3 gridLayout;
    dim3 blockLayout;

    bool errorCorrection = true;
    if (errorCorrection) {
	determineLayout(gridLayout, blockLayout, thickness->getNTheta(), thickness->getNPhi());
	CHECK_CUDA(cudaMemcpy(thickness->getGPUNextStep(), thickness->getGPUThisStep(),
			      thickness->getNTheta() * pitch * sizeof(fReal),
			      cudaMemcpyDeviceToDevice));

	correctThickness1<<<gridLayout, blockLayout>>>
	    (thickness->getGPUThisStep(), tmp_t, thickness->getGPUDelta(), thickness->getGPUInit(),
	     forward_t, forward_p, pitch);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	correctThickness2<<<gridLayout, blockLayout>>>
	    (thickness->getGPUThisStep(), thickness->getGPUNextStep(), tmp_t,
	     backward_t, backward_p, pitch);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
	// CHECK_CUDA(cudaMemcpy(velTheta->getGPUNextStep(), velTheta->getGPUThisStep(),
	// 		      velTheta->getNTheta() * pitch * sizeof(fReal),
	// 		      cudaMemcpyDeviceToDevice));

	// correctVTheta1<<<gridLayout, blockLayout>>>
	//     (velTheta->getGPUThisStep(), tmp_t, velTheta->getGPUDelta(), velTheta->getGPUInit(),
	//      forward_t, forward_p, pitch);
	// checkCudaErrors(cudaGetLastError());
	// checkCudaErrors(cudaDeviceSynchronize());

	// correctVTheta2<<<gridLayout, blockLayout>>>
	//     (velTheta->getGPUThisStep(), velTheta->getGPUNextStep(), tmp_t,
	//      backward_t, backward_p, pitch);
	// checkCudaErrors(cudaGetLastError());
	// checkCudaErrors(cudaDeviceSynchronize());

	// determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());
	// CHECK_CUDA(cudaMemcpy(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(),
	// 		      velPhi->getNTheta() * pitch * sizeof(fReal),
	// 		      cudaMemcpyDeviceToDevice));
      
	// correctVPhi1<<<gridLayout, blockLayout>>>
	//     (velPhi->getGPUThisStep(), tmp_t, velPhi->getGPUDelta(), velPhi->getGPUInit(),
	//      forward_t, forward_p, pitch);
	// checkCudaErrors(cudaGetLastError());
	// checkCudaErrors(cudaDeviceSynchronize());

	// correctVPhi2<<<gridLayout, blockLayout>>>
	//     (velPhi->getGPUThisStep(), velPhi->getGPUNextStep(), tmp_t,
	//      backward_t, backward_p, pitch);
	// checkCudaErrors(cudaGetLastError());
	// checkCudaErrors(cudaDeviceSynchronize());
    }

    std::swap(this->thickness->getGPUInitLast(), this->thickness->getGPUInit());
    std::swap(this->thickness->getGPUDeltaLast(), this->thickness->getGPUDelta());
    std::swap(backward_tprev, backward_t);
    std::swap(backward_pprev, backward_p);

    CHECK_CUDA(cudaMemcpy(this->thickness->getGPUInit(), this->thickness->getGPUThisStep(),
			  this->thickness->getThisStepPitchInElements() * this->thickness->getNTheta() *
			  sizeof(fReal), cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaMemset(this->thickness->getGPUDelta(), 0,
			  pitch * sizeof(fReal) * this->thickness->getNTheta()));
    // CHECK_CUDA(cudaMemset(this->velTheta->getGPUDelta(), 0,
    // 			  pitch * sizeof(fReal) * this->thickness->getNTheta()));
    // CHECK_CUDA(cudaMemset(this->velPhi->getGPUDelta(), 0,
    // 			  pitch * sizeof(fReal) * this->thickness->getNTheta()));

    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    initMapping<<<gridLayout, blockLayout>>>(forward_t, forward_p);
    initMapping<<<gridLayout, blockLayout>>>(backward_t, backward_p);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}


// assumes U = 1
Kamino::Kamino(fReal radius, fReal H, fReal U, fReal c_m, fReal Gamma_m,
	       fReal T, fReal Ds, fReal rm, size_t nTheta, 
	       fReal dt, fReal DT, int frames,
	       std::string outputDir, std::string thicknessImage,
	       size_t particleDensity, int device,
	       std::string AMGconfig, fReal blendCoeff):
    radius(radius), invRadius(1/radius), H(H), U(1.0), c_m(c_m), Gamma_m(Gamma_m), T(T),
    Ds(Ds/(U*radius)), gs(g*radius/(U*U)), rm(rm), epsilon(H/radius), sigma_r(R*T), M(Gamma_m*sigma_r/(3*rho*H*U*U)),
    S(sigma_a*epsilon/(2*mu*U)), re(mu/(U*radius*rho)), Cr(rhoa*sqrt(nua*radius/U)/(rho*H)),
    nTheta(nTheta), nPhi(2 * nTheta),
    gridLen(M_PI / nTheta), invGridLen(nTheta / M_PI), 
    dt(dt*U/radius), DT(DT), frames(frames), outputDir(outputDir),
    thicknessImage(thicknessImage), particleDensity(particleDensity), device(device),
    AMGconfig(AMGconfig), blendCoeff(blendCoeff)
{
    boost::filesystem::create_directories(outputDir);
    if (outputDir.back() != '/')
	this->outputDir.append("/");
    boost::filesystem::copy_file("../configKamino.txt", this->outputDir + "config.txt",
				 boost::filesystem::copy_option::overwrite_if_exists);
    boost::filesystem::copy_file("../include/KaminoHeader.cuh", this->outputDir + "KaminoHeader.cuh",
				 boost::filesystem::copy_option::overwrite_if_exists);

    std::cout << "Re^-1 " << re << std::endl;
    std::cout << "Cr " << Cr << std::endl;
    std::cout << "M " << M << std::endl;
    std::cout << "S " << S << std::endl;
    std::cout << "epsilon " << epsilon << std::endl;
    std::cout << "AMG config file: " << AMGconfig << std::endl;
}
Kamino::~Kamino()
{}

void Kamino::run()
{
    KaminoSolver solver(nPhi, nTheta, radius, dt*radius/U, H, device, AMGconfig, particleDensity);
    
    checkCudaErrors(cudaMemcpyToSymbol(nPhiGlobal, &(this->nPhi), sizeof(size_t)));
    checkCudaErrors(cudaMemcpyToSymbol(nThetaGlobal, &(this->nTheta), sizeof(size_t)));
    checkCudaErrors(cudaMemcpyToSymbol(radiusGlobal, &(this->radius), sizeof(fReal)));
    checkCudaErrors(cudaMemcpyToSymbol(invRadiusGlobal, &(this->invRadius), sizeof(fReal)));
    checkCudaErrors(cudaMemcpyToSymbol(timeStepGlobal, &(this->dt), sizeof(fReal)));
    checkCudaErrors(cudaMemcpyToSymbol(gridLenGlobal, &(this->gridLen), sizeof(fReal)));
    checkCudaErrors(cudaMemcpyToSymbol(invGridLenGlobal, &(this->invGridLen), sizeof(fReal)));
    checkCudaErrors(cudaMemcpyToSymbol(SGlobal, &(this->S), sizeof(fReal)));
    checkCudaErrors(cudaMemcpyToSymbol(MGlobal, &(this->M), sizeof(fReal)));
    checkCudaErrors(cudaMemcpyToSymbol(reGlobal, &(this->re), sizeof(fReal)));
    checkCudaErrors(cudaMemcpyToSymbol(gGlobal, &(this->gs), sizeof(fReal)));
    checkCudaErrors(cudaMemcpyToSymbol(DsGlobal, &(this->Ds), sizeof(fReal)));
    checkCudaErrors(cudaMemcpyToSymbol(CrGlobal, &(this->Cr), sizeof(fReal)));
    checkCudaErrors(cudaMemcpyToSymbol(UGlobal, &(this->U), sizeof(fReal)));
    checkCudaErrors(cudaMemcpyToSymbol(blend_coeff, &(this->blendCoeff), sizeof(fReal)));
# ifdef evaporation
    fReal eva = evaporation * radius / (H * U);
    checkCudaErrors(cudaMemcpyToSymbol(evaporationRate, &eva, sizeof(fReal)));
# endif
    fReal epsilon_ = 1.0;
    fReal W2_ = radius / (U * U) * pow(rm / H, 2) * epsilon_;
    fReal W1_ = 0.5 * W2_ * pow(rm / H, 2);
    checkCudaErrors(cudaMemcpyToSymbol(W1, &W1_, sizeof(fReal)));
    checkCudaErrors(cudaMemcpyToSymbol(W2, &W2_, sizeof(fReal)));

    solver.initThicknessfromPic(thicknessImage);

# ifdef WRITE_THICKNESS_DATA
    solver.write_thickness_img(outputDir, 0);
# endif  
# ifdef WRITE_VELOCITY_DATA
    solver.write_velocity_image(outputDir, 0);
# endif  
# ifdef WRITE_CONCENTRATION_DATA
    solver.write_concentration_image(outputDir, 0);
# endif

# ifdef PERFORMANCE_BENCHMARK
    KaminoTimer timer;
    timer.startTimer();
# endif

    fReal T = 0.0;              // simulation time
    int i = 1;
    fReal dt_ = dt * this->radius / this->U;
    for (; i < frames; i++) {
	checkCudaErrors(cudaMemcpyToSymbol(currentTimeGlobal, &T, sizeof(fReal)));
	std::cout << "current time " << T << std::endl;

	//     solver.adjustStepSize(dt, U, epsilon);

	checkCudaErrors(cudaMemcpyToSymbol(timeStepGlobal, &dt, sizeof(fReal)));
	std::cout << "current time step size is " << dt_ << " s" << std::endl;
	std::cout << "steps needed until next frame " << DT/dt_ << std::endl;
    
	while ((T + dt_) <= i*DT && !solver.isBroken()) {
	    solver.stepForward();
	    T += dt_;
	}
	if (T < i*DT && !solver.isBroken()) {
	    fReal tmp_dt = (i * DT - T) * this->U / this->radius;
	    checkCudaErrors(cudaMemcpyToSymbol(timeStepGlobal, &tmp_dt, sizeof(fReal)));
	    solver.stepForward(i * DT - T);
	}
	if (solver.isBroken()) {
	    std::cerr << "Film is broken." << std::endl;
	    break;
	}
	T = i*DT;
# ifdef BIMOCQ
	    fReal distortion = solver.estimateDistortion();
	    fReal q = solver.cfldt * distortion / dt;
	    std::cout << "max distortion " << distortion << std::endl;
// 	    std::cout << "q " << q << std::endl;
	    if (distortion > nTheta / 64.0) {
		//	if (q > .5) {
		solver.reInitializeMapping();
		std::cout << "mapping reinitialized" << std::endl;
	    }
# endif
	std::cout << "Frame " << i << " is ready" << std::endl;

# ifdef WRITE_THICKNESS_DATA
	solver.write_thickness_img(outputDir, i);
# endif  
# ifdef WRITE_VELOCITY_DATA
	solver.write_velocity_image(outputDir, i);
# endif  
# ifdef WRITE_CONCENTRATION_DATA
	solver.write_concentration_image(outputDir, i);
# endif



	

    }

# ifdef PERFORMANCE_BENCHMARK
    fReal gpu_time = timer.stopTimer();
# endif

    std::cout << "Time spent: " << gpu_time << "ms" << std::endl;
    std::cout << "Performance: " << 1000.0 * i / gpu_time << " frames per second" << std::endl;
}
