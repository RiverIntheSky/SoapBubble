# include "KaminoSolver.cuh"
# include "KaminoGPU.cuh"
# include "KaminoTimer.cuh"
# include <boost/filesystem.hpp>
# include "ceres/ceres.h"
# include "utils.h"

__constant__ float invGridLenGlobal;
static __constant__ size_t nPhiGlobal;
static __constant__ size_t nThetaGlobal;
static __constant__ float invRadiusGlobal;
static __constant__ float radiusGlobal;
static __constant__ float timeStepGlobal;
static __constant__ float currentTimeGlobal;
static __constant__ float gridLenGlobal;
static __constant__ float SGlobal;
static __constant__ float MGlobal;
static __constant__ float reGlobal;
static __constant__ float gGlobal;
static __constant__ float DsGlobal;
static __constant__ float CrGlobal;
static __constant__ float UGlobal;
static __constant__ float blend_coeff;


# define eps 1e-5f
# define MAX_BLOCK_SIZE 1024


/**
 * query value at coordinate
 */
inline __device__ float& at(float* &array, int &thetaId, int &phiId) {
    return array[phiId + thetaId * nPhiGlobal];
}


inline __device__ float& at(float* &array, int2 &Id) {
    return array[Id.y + Id.x * nPhiGlobal];
}


/**
 * query value in pitched memory at coordinate
 */
inline __device__ float& at(float* array, int thetaId, int phiId, size_t pitch) {
    return array[phiId + thetaId * pitch];
}


inline __device__ float& at(float* array, int2 Id, size_t pitch) {
    return array[Id.y + Id.x * pitch];
}


/**
 * distance between two coordinates on the sphere (unit in grid)
 */
inline __device__ float dist(float2 Id1, float2 Id2) {
    Id1 *= gridLenGlobal;
    Id2 *= gridLenGlobal;
    float3 Id13 = normalize(make_float3(cosf(Id1.y) * sinf(Id1.x),
					sinf(Id1.y) * sinf(Id1.x),
					cosf(Id1.x)));
    float3 Id23 = normalize(make_float3(cosf(Id2.y) * sinf(Id2.x),
					sinf(Id2.y) * sinf(Id2.x),
					cosf(Id2.x)));
    return invGridLenGlobal * safe_acosf(dot(Id13, Id23));
}


/**
 * return the maximal absolute value in array vel
 * usage: maxValKernel<<<gridSize, blockSize>>>(maxVal, vel);
 *        maxValKernel<<<1, blockSize>>>(maxVal, maxVal);
 */
__global__ void maxValKernel(float* maxVal, float* array) {
    __shared__ float maxValTile[MAX_BLOCK_SIZE];
	
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
__device__ bool validateCoord(float2& Id) {
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
    Id.y = fmodf(Id.y + nPhiGlobal, (float)nPhiGlobal);
    return ret;
}

bool KaminoSolver::validateCoord(double2& Id) {
    bool ret = false;

    if (Id.x >= nTheta) {
	Id.x = nPhi - Id.x;
	Id.y += nTheta;
    	ret = !ret;
    }
    if (Id.x < 0) {
    	Id.x = -Id.x;
    	Id.y += nTheta;
    	ret = !ret;
    }
    Id.y = fmod(Id.y + (double)nPhi, (double)nPhi);
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
__device__ float kaminoLerp(float from, float to, float alpha)
{
    return (1.0 - alpha) * from + alpha * to;
}


/**
 * bilinear interpolation
 */
__device__ float bilerp(float ll, float lr, float hl, float hr,
			float alphaPhi, float alphaTheta)
{
    return kaminoLerp(kaminoLerp(ll, lr, alphaPhi),
		      kaminoLerp(hl, hr, alphaPhi), alphaTheta);
}


/**
 * sample velocity in phi direction at position rawId
 * rawId is moved to velPhi coordinates to compensate MAC
 */
__device__ float sampleVPhi(float* input, float2 rawId, size_t pitch) {
    float2 Id = rawId - vPhiOffset;
    
    bool isFlippedPole = validateCoord(Id);

    int phiIndex = static_cast<int>(floorf(Id.y));
    int thetaIndex = static_cast<int>(floorf(Id.x));
    float alphaPhi = Id.y - static_cast<float>(phiIndex);
    float alphaTheta = Id.x - static_cast<float>(thetaIndex);
    
    if (thetaIndex == 0 && isFlippedPole) {
	size_t phiLower = (phiIndex) % nPhiGlobal;
	size_t phiHigher = (phiLower + 1) % nPhiGlobal;
	float higherBelt = -kaminoLerp(input[phiLower + pitch * thetaIndex],
				       input[phiHigher + pitch * thetaIndex], alphaPhi);

	phiLower = (phiIndex + nPhiGlobal / 2) % nPhiGlobal;
	phiHigher = (phiLower + 1) % nPhiGlobal;

	float lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);
  
	float lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
	return lerped;
    }
    
    if (isFlippedPole) {
	thetaIndex -= 1;
    }
    
    if (thetaIndex == nThetaGlobal - 1) {
	size_t phiLower = (phiIndex) % nPhiGlobal;
	size_t phiHigher = (phiLower + 1) % nPhiGlobal;
	float lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);
	
	phiLower = (phiIndex + nPhiGlobal / 2) % nPhiGlobal;
	phiHigher = (phiLower + 1) % nPhiGlobal;

	float higherBelt = -kaminoLerp(input[phiLower + pitch * thetaIndex],
				       input[phiHigher + pitch * thetaIndex], alphaPhi);

	float lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
	if (isFlippedPole)
	    lerped = -lerped;
	return lerped;
    }
  
    size_t phiLower = phiIndex % nPhiGlobal;
    size_t phiHigher = (phiLower + 1) % nPhiGlobal;
    size_t thetaLower = thetaIndex;
    size_t thetaHigher = thetaIndex + 1;

    float lowerBelt = kaminoLerp(input[phiLower + pitch * thetaLower],
				 input[phiHigher + pitch * thetaLower], alphaPhi);
    float higherBelt = kaminoLerp(input[phiLower + pitch * thetaHigher],
				  input[phiHigher + pitch * thetaHigher], alphaPhi);

    float lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
    if (isFlippedPole)
	lerped = -lerped;
    return lerped;
}
    

/**
 * sample velocity in theta direction at position rawId
 * rawId is moved to velTheta coordinates to compensate MAC
 */
__device__ float sampleVTheta(float* input, float2 rawId, size_t pitch) {
    float2 Id = rawId - vThetaOffset;
    
    bool isFlippedPole = validateCoord(Id);

    int phiIndex = static_cast<int>(floorf(Id.y));
    int thetaIndex = static_cast<int>(floorf(Id.x));
    float alphaPhi = Id.y - static_cast<float>(phiIndex);
    float alphaTheta = Id.x - static_cast<float>(thetaIndex);
    
    if (rawId.x < 0 && rawId.x > -1 || rawId.x > nThetaGlobal && rawId.x < nThetaGlobal + 1 ) {
	thetaIndex -= 1;
	alphaTheta += 1;
    } else if (rawId.x >= nThetaGlobal + 1 || rawId.x <= -1) {
    	thetaIndex -= 2;
    }

    if (thetaIndex == 0 && isFlippedPole && rawId.x > -1) {
    	size_t phiLower = phiIndex % nPhiGlobal;
    	size_t phiHigher = (phiLower + 1) % nPhiGlobal;
    	float higherBelt = -kaminoLerp(input[phiLower + pitch * thetaIndex],
    				       input[phiHigher + pitch * thetaIndex], alphaPhi);
	
    	phiLower = (phiLower + nPhiGlobal / 2) % nPhiGlobal;
    	phiHigher = (phiHigher + nPhiGlobal / 2) % nPhiGlobal;
    	float lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
    				     input[phiHigher + pitch * thetaIndex], alphaPhi);

    	alphaTheta = 0.5 * alphaTheta;
    	float lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
    	return lerped;
	
    }
    
    if (thetaIndex == nThetaGlobal - 2) {
	size_t phiLower = phiIndex % nPhiGlobal;
	size_t phiHigher = (phiLower + 1) % nPhiGlobal;
	float lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);

	phiLower = (phiLower + nPhiGlobal / 2) % nPhiGlobal;
	phiHigher = (phiHigher + nPhiGlobal / 2) % nPhiGlobal;
	float higherBelt = -kaminoLerp(input[phiLower + pitch * thetaIndex],
				       input[phiHigher + pitch * thetaIndex], alphaPhi);

	alphaTheta = 0.5 * alphaTheta;
	float lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
	if (isFlippedPole)
	    lerped = -lerped;
	return lerped;
    }
    
    size_t phiLower = phiIndex % nPhiGlobal;
    size_t phiHigher = (phiLower + 1) % nPhiGlobal;
    size_t thetaLower = thetaIndex;
    size_t thetaHigher = thetaIndex + 1;

    float lowerBelt = kaminoLerp(input[phiLower + pitch * thetaLower],
				 input[phiHigher + pitch * thetaLower], alphaPhi);
    float higherBelt = kaminoLerp(input[phiLower + pitch * thetaHigher],
				  input[phiHigher + pitch * thetaHigher], alphaPhi);

    float lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
    if (isFlippedPole)
	lerped = -lerped;
    return lerped;
}


/**
 * sample scalar at position rawId
 * rawId is moved to scalar coordinates to compensate MAC
 */
__device__ float sampleCentered(float* input, float2 rawId, size_t pitch) {
    float2 Id = rawId - centeredOffset;

    bool isFlippedPole = validateCoord(Id);

    int phiIndex = static_cast<int>(floorf(Id.y));
    int thetaIndex = static_cast<int>(floorf(Id.x));
    float alphaPhi = Id.y - static_cast<float>(phiIndex);
    float alphaTheta = Id.x - static_cast<float>(thetaIndex);

    if (thetaIndex == 0 && isFlippedPole) {
	size_t phiLower = phiIndex % nPhiGlobal;
	size_t phiHigher = (phiLower + 1) % nPhiGlobal;
	float higherBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				      input[phiHigher + pitch * thetaIndex], alphaPhi);

	phiLower = (phiLower + nPhiGlobal / 2) % nPhiGlobal;
	phiHigher = (phiHigher + nPhiGlobal / 2) % nPhiGlobal;
	float lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);

	float lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
	return lerped;
    }
    
    if (isFlippedPole) {
	thetaIndex -= 1;
    }
    
    if (thetaIndex == nThetaGlobal - 1) {
	size_t phiLower = phiIndex % nPhiGlobal;
	size_t phiHigher = (phiLower + 1) % nPhiGlobal;
	float lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);

	phiLower = (phiLower + nPhiGlobal / 2) % nPhiGlobal;
	phiHigher = (phiHigher + nPhiGlobal / 2) % nPhiGlobal;
	float higherBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				      input[phiHigher + pitch * thetaIndex], alphaPhi);

	float lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
	return lerped;
    }

    size_t phiLower = phiIndex % nPhiGlobal;
    size_t phiHigher = (phiLower + 1) % nPhiGlobal;
    size_t thetaLower = thetaIndex;
    size_t thetaHigher = thetaIndex + 1;

    float lowerBelt = kaminoLerp(input[phiLower + pitch * thetaLower],
				 input[phiHigher + pitch * thetaLower], alphaPhi);
    float higherBelt = kaminoLerp(input[phiLower + pitch * thetaHigher],
				  input[phiHigher + pitch * thetaHigher], alphaPhi);

    float lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
    return lerped;
}


/**
 * @return (velTheta, velPhi)
 */
inline __device__ float2 getVelocity(float* velPhi, float* velTheta, float2 &Id, size_t pitch){
    return make_float2(sampleVTheta(velTheta, Id, pitch),
		       sampleVPhi(velPhi, Id, pitch));
}


/**
 * Runge-Kutta 2nd Order
 * positive dt => trace backward;
 * negative dt => trace forward;
 */
inline __device__ float2 traceRK2(float* velTheta, float* velPhi, float dt,
				  float2& Id0, size_t pitch){

    float2 vel0 = getVelocity(velPhi, velTheta, Id0, pitch);
    float2 Id1 = Id0 - 0.5 * dt * vel0 * invGridLenGlobal;
    float2 vel1 = getVelocity(velPhi, velTheta, Id1, pitch);
    float2 Id2 = Id1 - dt * vel1 * invGridLenGlobal;

    return Id2;
}


/**
 * Runge-Kutta 3rd Order Ralston
 * positive dt => trace backward;
 * negative dt => trace forward;
 */
inline __device__ float2 traceRK3(float* velTheta, float* velPhi, float dt,
				  float2& Id0, size_t pitch){
    float c0 = 2.0 / 9.0 * dt * invGridLenGlobal,
	c1 = 3.0 / 9.0 * dt * invGridLenGlobal,
	c2 = 4.0 / 9.0 * dt * invGridLenGlobal;
    float2 vel0 = getVelocity(velPhi, velTheta, Id0, pitch);
    float2 Id1 = Id0 - 0.5 * dt * vel0 * invGridLenGlobal;
    float2 vel1 = getVelocity(velPhi, velTheta, Id1, pitch);
    float2 Id2 = Id1 - 0.75 * dt * vel1 * invGridLenGlobal;
    float2 vel2 = getVelocity(velPhi, velTheta, Id2, pitch);

    return Id0 - c0 * vel0 - c1 * vel1 - c2 * vel2;
}


// RK3 instead of real DMC for testing!!
// only temporarily
inline __device__ float2 DMC(float* velTheta, float* velPhi, float& dt, float2& pos,
			     size_t pitch){
    return traceRK3(velTheta, velPhi, dt, pos, pitch);
}


inline __device__ float2 lerpCoords(float2 from, float2 to, float alpha) {
    float2 from_ = from * gridLenGlobal;
    float2 to_ = to * gridLenGlobal;

    float3 from3 = normalize(make_float3(cosf(from_.y) * sinf(from_.x),
					 sinf(from_.y) * sinf(from_.x),
					 cosf(from_.x)));
    float3 to3 = normalize(make_float3(cosf(to_.y) * sinf(to_.x),
				       sinf(to_.y) * sinf(to_.x),
				       cosf(to_.x)));
    float3 k = normalize(cross(from3, to3));
    if (isnan(k.x))
	return from;
    float span = safe_acosf(dot(from3, to3));
    alpha *= span;
    float3 interpolated3 = from3 * cosf(alpha) + cross(k, from3) * sinf(alpha)
	+ k * dot(k, from3) * (1 - cosf(alpha));

    return invGridLenGlobal * make_float2(safe_acosf(interpolated3.z),
					  atan2f(interpolated3.y,
						 interpolated3.x));
}


__device__ float2 sampleMapping(float* map_t, float* map_p, float2& rawId){
    float2 Id = rawId - centeredOffset;
    bool isFlippedPole = validateCoord(Id);

    int phiIndex = static_cast<int>(floorf(Id.y));
    int thetaIndex = static_cast<int>(floorf(Id.x));
    float alphaPhi = Id.y - static_cast<float>(phiIndex);
    float alphaTheta = Id.x - static_cast<float>(thetaIndex);

    float2 ll, lr, hl, hr;
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


__global__ void	updateMappingKernel(float* velTheta, float* velPhi, float dt,
				     float* map_t, float* map_p,
				     float* tmp_t, float* tmp_p, size_t pitch){
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    float2 pos = make_float2((float)thetaId, (float)phiId) + centeredOffset;

    float2 back_pos = DMC(velTheta, velPhi, dt, pos, pitch);

    float2 sampledId = sampleMapping(map_t, map_p, back_pos);

    validateCoord(sampledId);

    tmp_p[thetaId * nPhiGlobal + phiId] = sampledId.y;
    tmp_t[thetaId * nPhiGlobal + phiId] = sampledId.x;
}


/**
 * advect vetor using great cicle method
 */
__global__ void advectionVSpherePhiKernel
(float* velPhiOutput, float* velPhiInput, float* velThetaInput, size_t pitch)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    
    // Coord in phi space
    float2 gId = make_float2((float)thetaId, (float)phiId) + vPhiOffset;
    float2 gCoord = gId * gridLenGlobal;

    // Trigonometric functions
    float sinTheta = sinf(gCoord.x);
    float cosTheta = cosf(gCoord.x);
    float sinPhi = sinf(gCoord.y);
    float cosPhi = cosf(gCoord.y);

    // Sample the speed
    float guTheta = sampleVTheta(velThetaInput, gId, pitch);
    float guPhi = velPhiInput[thetaId * pitch + phiId] * sinTheta;

    // Unit vector in theta and phi direction
    float3 eTheta = make_float3(cosTheta * cosPhi, cosTheta * sinPhi, -sinTheta);
    float3 ePhi = make_float3(-sinPhi, cosPhi, 0.f);

    // Circle
    float3 u, u_, v_;
    float u_norm, deltaS;
    float3 w_ = make_float3(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
    
    u = guTheta * eTheta + guPhi * ePhi;
    u_norm = length(u);
    if (u_norm > 0) {
	u_ = normalize(u);
        v_ = cross(w_, u_);
    } else {
	velPhiOutput[thetaId * pitch + phiId] = 0.f;
	return;
    }
	
# ifdef RUNGE_KUTTA
    // Traced halfway in phi-theta space
    deltaS = - 0.5 * u_norm * timeStepGlobal;
    float3 midx = u_ * sinf(deltaS) + w_ * cosf(deltaS);

    float2 midCoord = make_float2(safe_acosf(midx.z), atan2f(midx.y, midx.x));
    float2 midId = midCoord * invGridLenGlobal;

    float muTheta = sampleVTheta(velThetaInput, midId, pitch);
    float muPhi = sampleVPhi(velPhiInput, midId, pitch) * sinf(midCoord.x);

    float3 mu = make_float3(muTheta * cosf(midCoord.x) * cosf(midCoord.y) - muPhi * sinf(midCoord.y),
			    muTheta * cosf(midCoord.x) * sinf(midCoord.y) + muPhi * cosf(midCoord.y),
			    -muTheta * sinf(midCoord.x));

    float3 uCircleMid_ = u_ * cosf(deltaS) - w_ * sinf(deltaS);
    float3 vCircleMid_ = cross(midx, uCircleMid_);
    
    float mguTheta = dot(mu, vCircleMid_);
    float mguPhi = dot(mu, uCircleMid_);

    u = mguPhi * u_ + mguTheta * v_;
    u_norm = length(u);
    if (u_norm > 0) {
	u_ = normalize(u);
        v_ = cross(w_, u_);
    } else {
	velPhiOutput[thetaId * pitch + phiId] = 0.f;
	return;
    }

# endif
    deltaS = -u_norm * timeStepGlobal;
    float3 px = u_ * sinf(deltaS) + w_ * cosf(deltaS);

    float2 pCoord = make_float2(safe_acosf(px.z), atan2f(px.y, px.x));
    float2 pId = pCoord * invGridLenGlobal;

    float puTheta = sampleVTheta(velThetaInput, pId, pitch);
    float puPhi = sampleVPhi(velPhiInput, pId, pitch) * sinf(pCoord.x);

    float3 pu = make_float3(puTheta * cosf(pCoord.x) * cosf(pCoord.y) - puPhi * sinf(pCoord.y),
			    puTheta * cosf(pCoord.x) * sinf(pCoord.y) + puPhi * cosf(pCoord.y),
			    -puTheta * sinf(pCoord.x));
	
    float3 uCircleP_ = u_ * cosf(deltaS) - w_ * sinf(deltaS);
    float3 vCircleP_ = cross(px, uCircleP_);

    puTheta = dot(pu, vCircleP_);
    puPhi = dot(pu, uCircleP_);

    pu = puPhi * u_ + puTheta * v_;
    velPhiOutput[thetaId * pitch + phiId] = dot(pu, ePhi) / sinTheta;
}


/**
 * advect vetor using great cicle method
 */
__global__ void advectionVSphereThetaKernel
(float* velThetaOutput, float* velPhiInput, float* velThetaInput, size_t pitch)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    // Coord in theta space
    float2 gId = make_float2((float)thetaId, (float)phiId) + vThetaOffset;
    float2 gCoord = gId * gridLenGlobal;

    // Trigonometric functions
    float sinTheta = sinf(gCoord.x);
    float cosTheta = cosf(gCoord.x);
    float sinPhi = sinf(gCoord.y);
    float cosPhi = cosf(gCoord.y);

    // Sample the speed
    float guTheta = velThetaInput[thetaId * pitch + phiId];
    float guPhi = sampleVPhi(velPhiInput, gId, pitch) * sinTheta;

    // Unit vector in theta and phi direction
    float3 eTheta = make_float3(cosTheta * cosPhi, cosTheta * sinPhi, -sinTheta);
    float3 ePhi = make_float3(-sinPhi, cosPhi, 0.f);

    // Circle
    float3 u, u_, v_;
    float u_norm, deltaS;
    float3 w_ = make_float3(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
    
    u = guTheta * eTheta + guPhi * ePhi;
    u_norm = length(u);
    if (u_norm > 0) {
	u_ = normalize(u);
        v_ = cross(w_, u_);
    } else {
	velThetaOutput[thetaId * pitch + phiId] = 0.f;
	return;
    }

# ifdef RUNGE_KUTTA
    // Traced halfway in phi-theta space
    deltaS = - 0.5 * u_norm * timeStepGlobal;
    float3 midx = u_ * sinf(deltaS) + w_ * cosf(deltaS);

    float2 midCoord = make_float2(safe_acosf(midx.z), atan2f(midx.y, midx.x));
    float2 midId = midCoord * invGridLenGlobal;

    float muTheta = sampleVTheta(velThetaInput, midId, pitch);
    float muPhi = sampleVPhi(velPhiInput, midId, pitch) * sinf(midCoord.x);

    float3 mu = make_float3(muTheta * cosf(midCoord.x) * cosf(midCoord.y) - muPhi * sinf(midCoord.y),
			    muTheta * cosf(midCoord.x) * sinf(midCoord.y) + muPhi * cosf(midCoord.y),
			    -muTheta * sinf(midCoord.x));

    float3 uCircleMid_ = u_ * cosf(deltaS) - w_ * sinf(deltaS);
    float3 vCircleMid_ = cross(midx, uCircleMid_);

    float mguTheta = dot(mu, vCircleMid_);
    float mguPhi = dot(mu, uCircleMid_);

    u = mguPhi * u_ + mguTheta * v_;
    u_norm = length(u);
    if (u_norm > 0) {
	u_ = normalize(u);
        v_ = cross(w_, u_);
    } else {
	velThetaOutput[thetaId * pitch + phiId] = 0.f;
	return;
    }
    
# endif
    deltaS = -u_norm * timeStepGlobal;
    float3 px = u_ * sinf(deltaS) + w_ * cosf(deltaS);

    float2 pCoord = make_float2(safe_acosf(px.z), atan2f(px.y, px.x));
    float2 pId = pCoord * invGridLenGlobal;
    
    float puTheta = sampleVTheta(velThetaInput, pId, pitch);
    float puPhi = sampleVPhi(velPhiInput, pId, pitch) * sinf(pCoord.x);

    float3 pu = make_float3(puTheta * cosf(pCoord.x) * cosf(pCoord.y) - puPhi * sinf(pCoord.y),
			    puTheta * cosf(pCoord.x) * sinf(pCoord.y) + puPhi * cosf(pCoord.y),
			    -puTheta * sinf(pCoord.x));

    float3 uCircleP_ = u_ * cosf(deltaS) - w_ * sinf(deltaS);
    float3 vCircleP_ = cross(px, uCircleP_);

    puTheta = dot(pu, vCircleP_);
    puPhi = dot(pu, uCircleP_);

    pu = puPhi * u_ + puTheta * v_;
    velThetaOutput[thetaId * pitch + phiId] = dot(pu, eTheta);
}


/**
 * advect vectors on cartesian grid 
 * or test advection of vectors on sphere
 */
__global__ void advectionVPhiKernel
(float* attributeOutput, float* velPhi, float* velTheta, size_t pitch)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    
    // Coord in vel phi space
    float2 gId = make_float2((float)thetaId, (float)phiId) + vPhiOffset;
    
# ifdef RK3
    float2 traceId = traceRK3(velTheta, velPhi, timeStepGlobal, gId, pitch);
# else
    float2 traceId = traceRK2(velTheta, velPhi, timeStepGlobal, gId, pitch);
# endif

    attributeOutput[thetaId * pitch + phiId] = sampleVPhi(velPhi, traceId, pitch);
};


/**
 * advect vectors on cartesian grid 
 * or test advection of vectors on sphere
 */
__global__ void advectionVThetaKernel
(float* attributeOutput, float* velPhi, float* velTheta, size_t pitch)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    
    // Coord in vel theta space
    float2 gId = make_float2((float)thetaId, (float)phiId) + vThetaOffset;

# ifdef RK3
    float2 traceId = traceRK3(velTheta, velPhi, timeStepGlobal, gId, pitch);
# else
    float2 traceId = traceRK2(velTheta, velPhi, timeStepGlobal, gId, pitch);
# endif 

    attributeOutput[thetaId * pitch + phiId] = sampleVTheta(velTheta, traceId, pitch);
}


/**
 * advect scalar
 */
__global__ void advectionCentered
(float* attributeOutput, float* attributeInput, float* velPhi, float* velTheta, size_t pitch)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    
    // Coord in scalar space
    float2 gId = make_float2((float)thetaId, (float)phiId) + centeredOffset;

# ifdef RK3
    float2 traceId = traceRK3(velTheta, velPhi, timeStepGlobal, gId, pitch);
# else
    float2 traceId = traceRK2(velTheta, velPhi, timeStepGlobal, gId, pitch);
# endif
    
    at(attributeOutput, thetaId, phiId, pitch)
	= sampleCentered(attributeInput, traceId, pitch);
}


/**
 * advect all scalars
 */
__global__ void advectionAllCentered
(float* thicknessOutput, float* thicknessInput, float* gammaOutput, float* gammaInput,
 float* velPhi, float* velTheta, size_t pitch)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    
    // Coord in scalar space
    float2 gId = make_float2((float)thetaId, (float)phiId) + centeredOffset;

# ifdef RK3
    float2 traceId = traceRK3(velTheta, velPhi, timeStepGlobal, gId, pitch);
# else
    float2 traceId = traceRK2(velTheta, velPhi, timeStepGlobal, gId, pitch);
# endif

    at(thicknessOutput, thetaId, phiId, pitch)
	= sampleCentered(thicknessInput, traceId, pitch);
    at(gammaOutput, thetaId, phiId, pitch)
	= sampleCentered(gammaInput, traceId, pitch);
}

__global__ void advectionParticles(float* output, float* velPhi, float* velTheta, float* input, size_t pitch, size_t numOfParticles)
{
    int particleId = blockIdx.x * blockDim.x + threadIdx.x;

    if (particleId < numOfParticles) {
	float thetaId = input[2 * particleId];
	float phiId = input[2 * particleId + 1];
	
	// Coord in scalar space
	float2 gId = make_float2(thetaId, phiId);

# ifdef RK3
	float2 traceId = traceRK3(velTheta, velPhi, -timeStepGlobal, gId, pitch);
# else
	float2 traceId = traceRK2(velTheta, velPhi, -timeStepGlobal, gId, pitch);
# endif  
	output[2 * particleId] = traceId.x;
	output[2 * particleId + 1] = traceId.y;
    }
}

__global__ void advectionCenteredBimocq
(float* thicknessOutput, float* thicknessInput, float* thicknessInit, float* thicknessDelta,
 float* thicknessInitLast, float* thicknessDeltaLast, float* velTheta, float* velPhi,
 float* bwd_t, float* bwd_p, float* bwd_tprev, float* bwd_pprev, size_t pitch) {
    float w[5] = {0.125f, 0.125f, 0.125f, 0.125f, 0.5f};
    float2 dir[5] = {make_float2(-0.25f,-0.25f),
		     make_float2(0.25f, -0.25f),
		     make_float2(-0.25f, 0.25f),
		     make_float2( 0.25f, 0.25f),
		     make_float2(0.f, 0.f)};

    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    // Coord in scalar space
    float2 gId = make_float2((float)thetaId, (float)phiId) + centeredOffset;

    // if (thetaId < float(nThetaGlobal) / 16.f || thetaId > float(nThetaGlobal) * 15.f / 16.f) {
    // 	float2 traceId = traceRK3(velTheta, velPhi, timeStepGlobal, gId, pitch);
    // 	at(thicknessOutput, thetaId, phiId, pitch)
    // 	= sampleCentered(thicknessInput, traceId, pitch);
    // } else {
	float thickness = 0.f;
	//    float gamma = 0.f;
	for (int i = 0; i < 5; i++) {
	    float2 posId = gId + dir[i];
	    float2 initPosId = sampleMapping(bwd_t, bwd_p, posId);
	    float2 lastPosId = sampleMapping(bwd_tprev, bwd_pprev, initPosId);
	    thickness += (1.f - blend_coeff) * w[i] * (sampleCentered(thicknessInitLast, lastPosId, pitch) +
						       sampleCentered(thicknessDelta, initPosId, pitch) +
						       sampleCentered(thicknessDeltaLast, lastPosId, pitch)); 
	    thickness += blend_coeff * w[i] * (sampleCentered(thicknessInit, initPosId, pitch) +
					       sampleCentered(thicknessDelta, initPosId, pitch));
	    // gamma += w[i] * (sampleCentered(gammaInit, initPosId, pitch) +
	    //	 sampleCentered(gammaDelta, initPosId, pitch));
	}
	at(thicknessOutput, thetaId, phiId, pitch) = thickness;
	// gammaOutput[thetaId * pitch + phiId] = gamma;
	//    }
}


__global__ void advectionVThetaBimocq
(float* velThetaOutput, float* velThetaInput, float* velThetaInit, float* velThetaDelta,
 float* velPhi, float* bwd_t, float* bwd_p, size_t pitch) {
    float w[5] = {0.125f, 0.125f, 0.125f, 0.125f, 0.5f};
    float2 dir[5] = {make_float2(-0.25f,-0.25f),
		     make_float2(0.25f, -0.25f),
		     make_float2(-0.25f, 0.25f),
		     make_float2( 0.25f, 0.25f),
		     make_float2(0.f, 0.f)};

    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    // Coord in velTheta space
    float2 gId = make_float2((float)thetaId, (float)phiId) + vThetaOffset;

    if (thetaId < float(nThetaGlobal) / 16.f || thetaId > float(nThetaGlobal) * 15.f / 16.f) {
	float2 traceId = traceRK3(velThetaInput, velPhi, timeStepGlobal, gId, pitch);
    	at(velThetaOutput, thetaId, phiId, pitch) = sampleVTheta(velThetaInput, traceId, pitch);
    } else {
	float v = 0.f;
	for (int i = 0; i < 5; i++) {
	    float2 posId = gId + dir[i];
	    float2 initPosId = sampleMapping(bwd_t, bwd_p, posId);
	    v += w[i] * (sampleVTheta(velThetaInit, initPosId, pitch) +
			 sampleVTheta(velThetaDelta, initPosId, pitch));
	}
	at(velThetaOutput, thetaId, phiId, pitch) = v;
    }
} 


__global__ void advectionVPhiBimocq
(float* velPhiOutput, float* velPhiInput, float* velPhiInit, float* velPhiDelta,
 float* velTheta, float* bwd_t, float* bwd_p, size_t pitch) {
    float w[5] = {0.125f, 0.125f, 0.125f, 0.125f, 0.5f};
    float2 dir[5] = {make_float2(-0.25f,-0.25f),
		     make_float2(0.25f, -0.25f),
		     make_float2(-0.25f, 0.25f),
		     make_float2( 0.25f, 0.25f),
		     make_float2(0.f, 0.f)};

    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    // Coord in uPhi space
    float2 gId = make_float2((float)thetaId, (float)phiId) + vPhiOffset;

    if (thetaId < float(nThetaGlobal) / 16.f || thetaId > float(nThetaGlobal) * 15.f / 16.f) {
    	float2 traceId = traceRK3(velTheta, velPhiInput, timeStepGlobal, gId, pitch);
    	at(velPhiOutput, thetaId, phiId, pitch) = sampleVPhi(velPhiInput, traceId, pitch);
    } else {
	float u = 0.f;
	for (int i = 0; i < 5; i++) {
	    float2 posId = gId + dir[i];
	    float2 initPosId = sampleMapping(bwd_t, bwd_p, posId);
	    u += w[i] * (sampleVPhi(velPhiInit, initPosId, pitch) +
			 sampleVPhi(velPhiDelta, initPosId, pitch));
	}
        at(velPhiOutput, thetaId, phiId, pitch) = u;
    }
}


/**
 * return the maximal absolute value in array with nTheta rows and nPhi cols
 */
float KaminoSolver::maxAbs(float* array, size_t nTheta, size_t nPhi) {
    float *max, result;
    CHECK_CUDA(cudaMalloc(&max, MAX_BLOCK_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(max, 0, MAX_BLOCK_SIZE * sizeof(float)));

    dim3 gridLayout;
    dim3 blockLayout;
    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    maxValKernel<<<gridLayout, blockLayout>>>(max, array);
    CHECK_CUDA(cudaDeviceSynchronize());
    maxValKernel<<<1, blockLayout>>>(max, max);
    CHECK_CUDA(cudaMemcpy(&result, max, sizeof(float),
     			  cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(max));
    return result;
}


void KaminoSolver::updateForward(float dt, float* &fwd_t, float* &fwd_p) {
    float T = 0.0;
    float dt_ = dt / radius; // scaled; assume U = 1
    float substep = std::min(cfldt, dt_); // cfl < 1 required
    
    dim3 gridLayout;
    dim3 blockLayout;
    determineLayout(gridLayout, blockLayout, nTheta, nPhi);

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
}

void KaminoSolver::updateBackward(float dt, float* &bwd_t, float* &bwd_p) {
    float T = 0.0;
    float dt_ = dt / radius; // scaled; assume U = 1
    float substep = std::min(cfldt, dt_); // cfl < 1 required

    dim3 gridLayout;
    dim3 blockLayout;
    determineLayout(gridLayout, blockLayout, nTheta, nPhi);

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
}


void KaminoSolver::updateCFL(){
    this->maxu = maxAbs(velPhi->getGPUThisStep(), velPhi->getNTheta(),
			velPhi->getThisStepPitchInElements());
    this->maxv = maxAbs(velTheta->getGPUThisStep(), velTheta->getNTheta(),
			velTheta->getThisStepPitchInElements());

    this->cfldt = gridLen / std::max(std::max(maxu, maxv), eps);
}


__global__ void estimateDistortionKernel(float* map1_t, float* map1_p,
					 float* map2_t, float* map2_p, float* result) {
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    
    int2 Id = make_int2(thetaId, phiId);
    
    // Coord in scalar space
    float2 gId = make_float2((float)thetaId, (float)phiId) + centeredOffset;

    // sample map2 using the entries of map 1
    float2 pos1 = make_float2(at(map1_t, Id), at(map1_p, Id));
    float2 pos2 = sampleMapping(map2_t, map2_p, pos1);

    at(result, Id) = dist(gId, pos2);
}


float KaminoSolver::estimateDistortion() {
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
    std::cout << "max sampling backward " << maxAbs(tmp_t, nTheta, nPhi) << " max sampling forward " << maxAbs(tmp_p, nTheta, nPhi) << std::endl;
    return max(maxAbs(tmp_t, nTheta, nPhi), maxAbs(tmp_p, nTheta, nPhi));
}


void KaminoSolver::advection()
{
    bool useBimocq = true;

    dim3 gridLayout;
    dim3 blockLayout;
    
    // Advect Theta
    determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
    // if (useBimocq) {
    // 	advectionVThetaBimocq<<<gridLayout, blockLayout>>>
    // 	    (velTheta->getGPUNextStep(), velTheta->getGPUThisStep(), velTheta->getGPUInit(), velTheta->getGPUDelta(),
    // 	     velPhi->getGPUThisStep(), backward_t, backward_p, pitch);
    // } else {
# ifdef greatCircle
	advectionVSphereThetaKernel<<<gridLayout, blockLayout>>>
	    (velTheta->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), velTheta->getNextStepPitchInElements());
# else
	advectionVThetaKernel<<<gridLayout, blockLayout>>>
	    (velTheta->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), velTheta->getNextStepPitchInElements());
# endif
	//}
    checkCudaErrors(cudaGetLastError());
    
    // Advect Phi
    determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());
    // if (useBimocq) {
    // 	advectionVPhiBimocq<<<gridLayout, blockLayout>>>
    // 	    (velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), velPhi->getGPUInit(), velPhi->getGPUDelta(),
    // 	     velTheta->getGPUThisStep(), backward_t, backward_p, pitch);
    // } else {
# ifdef greatCircle
	advectionVSpherePhiKernel<<<gridLayout, blockLayout>>>
	    (velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), velPhi->getNextStepPitchInElements());
# else
	advectionVPhiKernel<<<gridLayout, blockLayout>>>
	    (velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), velPhi->getNextStepPitchInElements());
# endif
	//}
    checkCudaErrors(cudaGetLastError());

    if (particleDensity > 0) {
    	determineLayout(gridLayout, blockLayout, 1, particles->numOfParticles);
    	advectionParticles<<<gridLayout, blockLayout>>>
    	    (particles->coordGPUNextStep, velPhi->getGPUThisStep(), velTheta->getGPUThisStep(),
    	     particles->coordGPUThisStep, velPhi->getThisStepPitchInElements(), particles->numOfParticles);
    	checkCudaErrors(cudaGetLastError());

	determineLayout(gridLayout, blockLayout, surfConcentration->getNTheta(), surfConcentration->getNPhi());
	advectionCentered<<<gridLayout, blockLayout>>>
		(surfConcentration->getGPUNextStep(), surfConcentration->getGPUThisStep(),
		 velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), pitch);
	checkCudaErrors(cudaGetLastError());
    	checkCudaErrors(cudaDeviceSynchronize());
    } else {
	// Advect concentration
	determineLayout(gridLayout, blockLayout, surfConcentration->getNTheta(), surfConcentration->getNPhi());
	if (useBimocq) {
	    advectionCenteredBimocq<<<gridLayout, blockLayout>>>
		(thickness->getGPUNextStep(), thickness->getGPUThisStep(), thickness->getGPUInit(),
		 thickness->getGPUDelta(), thickness->getGPUInitLast(),
		 thickness->getGPUDeltaLast(), velTheta->getGPUThisStep(), velPhi->getGPUThisStep(),
		 backward_t, backward_p, backward_tprev, backward_pprev, pitch);
	    checkCudaErrors(cudaGetLastError());
 
	    advectionCentered<<<gridLayout, blockLayout>>>
		(surfConcentration->getGPUNextStep(), surfConcentration->getGPUThisStep(),
		 velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), pitch);

	} else {
	    advectionAllCentered<<<gridLayout, blockLayout>>>
		(thickness->getGPUNextStep(), thickness->getGPUThisStep(),
		 surfConcentration->getGPUNextStep(), surfConcentration->getGPUThisStep(),
		 velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), pitch);
	}
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// error correction
	if (useBimocq) {
	    // correctBimocq1<<<gridLayout, blockLayout>>>
	    //     (thickness->getGPUThisStep(), tmp_t, thickness->getGPUDelta(), thickness->getGPUInit(),
	    //      forward_t, forward_p, pitch);
	    // checkCudaErrors(cudaGetLastError());
	    // checkCudaErrors(cudaDeviceSynchronize());

	    // CHECK_CUDA(cudaMemcpy(thickness->getGPUNextStep(), thickness->getGPUThisStep(),
	    // 			  thickness->getNTheta() * pitch * sizeof(float),
	    // 			  cudaMemcpyDeviceToDevice));
	    // CHECK_CUDA(cudaMemcpy(surfConcentration->getGPUNextStep(),
	    // 		      surfConcentration->getGPUThisStep(),
	    // 		      surfConcentration->getNTheta() * pitch * sizeof(float),
	    // 		      cudaMemcpyDeviceToDevice));

	    // correctBimocq2<<<gridLayout, blockLayout>>>
	    //     (thickness->getGPUNextStep(), thickness->getGPUThisStep(), tmp_t,
	    //      backward_t, backward_p, pitch);
	    // checkCudaErrors(cudaGetLastError());
	    // checkCudaErrors(cudaDeviceSynchronize());
	}
    }
    
    if (particleDensity > 0) {
	particles->swapGPUBuffers();
    } else {
	thickness->swapGPUBuffer();	
    }
    surfConcentration->swapGPUBuffer();
    swapVelocityBuffers();
}


// div(u) at cell center
__global__ void divergenceKernel
(float* div, float* velPhi, float* velTheta,
 size_t velPhiPitchInElements, size_t velThetaPitchInElements)
{
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    float thetaCoord = ((float)thetaId + centeredThetaOffset) * gridLenGlobal;

    float uEast = 0.0;
    float uWest = 0.0;
    float vNorth = 0.0;
    float vSouth = 0.0;

    float halfStep = 0.5 * gridLenGlobal;

    float thetaSouth = thetaCoord + halfStep;
    float thetaNorth = thetaCoord - halfStep;

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
    float invGridSine = 1.0 / sinf(thetaCoord);
    float sinNorth = sinf(thetaNorth);
    float sinSouth = sinf(thetaSouth);
    float factor = invGridSine * invGridLenGlobal;
    float termTheta = factor * (vSouth * sinSouth - vNorth * sinNorth);
#else
    float factor = invGridLenGlobal;
    float termTheta = factor * (vSouth  - vNorth);
# endif
# ifdef sphere
    float termPhi = invGridLenGlobal * (uEast - uWest);
# else
    float termPhi = factor * (uEast - uWest);
# endif

    float f = termTheta + termPhi;

    // if (thetaId == nThetaGlobal - 1 && phiId < 8) {
    // 	printf("phiId %d vNorth %f vSouth %f termTheta %f termPhi %f div %f\n", phiId,	vNorth , vSouth , termTheta , termPhi , f);
    // }
    div[thetaId * nPhiGlobal + phiId] = f;
}


// compute divergence using gamma
__global__ void divergenceKernel_fromGamma(float* div, float* gammaNext, float* gammaThis,
					   size_t pitch) {
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    float gamma_a = gammaThis[thetaId * pitch + phiId];
    float gamma = gammaNext[thetaId * pitch + phiId];

    div[thetaId * nPhiGlobal + phiId] = (1 - gamma / gamma_a) / timeStepGlobal;
}


__global__ void concentrationLinearSystemKernel
(float* velPhi_a, float* velTheta_a, float* gamma_a, float* eta_a,
 float* val, float* rhs, size_t pitch) {
    // TODO: pre-compute eta???
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    int idx = thetaId * nPhiGlobal + phiId;
    int idx5 = 5 * idx;

    float gamma = at(gamma_a, thetaId, phiId, pitch);
    float invDt = 1.f / timeStepGlobal;

    float gTheta = ((float)thetaId + centeredThetaOffset) * gridLenGlobal;
    float gPhi = ((float)phiId + centeredPhiOffset) * gridLenGlobal;
    float halfStep = 0.5 * gridLenGlobal;
    float sinThetaSouth = sinf(gTheta + halfStep);
    float sinThetaNorth = sinf(gTheta - halfStep);
    float sinTheta = sinf(gTheta);
    float cscTheta = 1. / sinTheta;
    float cosTheta = cosf(gTheta);

    // neighboring values
    int phiIdEast = (phiId + 1) % nPhiGlobal;
    int phiIdWest = (phiId - 1 + nPhiGlobal) % nPhiGlobal;
    float eta = at(eta_a, thetaId, phiId, pitch);
    float etaWest = at(eta_a, thetaId, phiIdWest, pitch);
    float etaEast = at(eta_a, thetaId, phiIdEast, pitch);
    float etaNorth = 0.f;
    float etaSouth = 0.f;
    float uNorth = 0.f;
    float uSouth = 0.f;
    float uWest = 0.f;
    float uEast = 0.f;
    float vNorth = 0.f;
    float vSouth = 0.f;
    float vWest = sampleVTheta(velTheta_a, make_float2(gTheta, gPhi - halfStep), pitch);
    float vEast = sampleVTheta(velTheta_a, make_float2(gTheta, gPhi + halfStep), pitch);

    uWest = at(velPhi_a, thetaId, phiId, pitch);
    uEast = at(velPhi_a, thetaId, phiIdEast, pitch);

    if (thetaId != 0) {
	size_t thetaNorthIdx = thetaId - 1;
	vNorth = at(velTheta_a, thetaNorthIdx, phiId, pitch);
	uNorth = sampleVPhi(velPhi_a, make_float2(gTheta - halfStep, gPhi), pitch);
    	etaNorth = at(eta_a, thetaNorthIdx, phiId, pitch);
    } else {
    	int oppositePhiId = (phiId + nThetaGlobal) % nPhiGlobal;
    	etaNorth = eta_a[oppositePhiId];
    }
    if (thetaId != nThetaGlobal - 1) {
	vSouth = at(velTheta_a, thetaId, phiId, pitch);
	uSouth = sampleVPhi(velPhi_a, make_float2(gTheta + halfStep, gPhi), pitch);
    	etaSouth = at(eta_a, thetaId + 1, phiId, pitch);
    } else {
    	int oppositePhiId = (phiId + nThetaGlobal) % nPhiGlobal;
    	etaSouth = at(eta_a, thetaId, oppositePhiId, pitch);
    }
    // at both poles sin(theta) = 0;

    // constant for this grid
    float CrDt = CrGlobal * timeStepGlobal; // Cr\Delta t
    float MDt = MGlobal * timeStepGlobal; // M\Delta t
    float s2 = invGridLenGlobal * invGridLenGlobal; // \Delta s^2
	
    // up
    float etaUp = (etaNorth + eta) / 2.f;
    val[idx5] = -s2 * sinThetaNorth * MDt / (etaUp + CrDt);

    // left
    float etaLeft = (etaWest + eta) / 2.f;
    val[idx5 + 1] = -s2 * cscTheta * MDt / (etaLeft + CrDt);

    // right
    float etaRight = (etaEast + eta) / 2.f;
    val[idx5 + 3] = -s2 * cscTheta * MDt / (etaRight + CrDt);
    
    // down
    float etaDown = (etaSouth + eta) / 2.f;
    val[idx5 + 4] = -s2 * sinThetaSouth * MDt / (etaDown + CrDt);

    // center
    val[idx5 + 2] = sinTheta / (gamma * timeStepGlobal)
	- (val[idx5] + val[idx5 + 1] + val[idx5 + 3] + val[idx5 + 4]);

    // rhs
    // \sin\theta * div
    float sinThetaDiv = invGridLenGlobal *
	(sinTheta * (uEast / (1.f + CrGlobal * timeStepGlobal / etaRight) -
		     uWest / (1.f + CrGlobal * timeStepGlobal / etaLeft)) +
	 (vSouth * sinThetaSouth / (1.f + CrGlobal * timeStepGlobal / etaDown) -
	  vNorth * sinThetaNorth / (1.f + CrGlobal * timeStepGlobal / etaUp)));

# ifndef greatCircle
    sinThetaDiv += invGridLenGlobal *
	(cosTheta * (uWest * vWest / (invDt + CrGlobal / etaLeft) -
		     uEast * vEast / (invDt + CrGlobal / etaRight)) +
	 uSouth * uSouth * sinThetaSouth * sinThetaSouth * cosf(gTheta + halfStep) /
	 (invDt + CrGlobal / etaDown) -
	 uNorth * uNorth * sinThetaNorth * sinThetaNorth * cosf(gTheta - halfStep) /
	 (invDt + CrGlobal / etaUp));
# endif

# ifdef uair
    // sinThetaDiv += 20.f * (1 - smoothstep(0.f, 10.f, currentTimeGlobal)) * (M_hPI - gTheta)
    // 	* expf(-10 * powf(fabsf(gTheta - M_hPI), 2.f)) * radiusGlobal * cscTheta / UGlobal *
    // 	invGridLenGlobal * (cosf(gPhi + halfStep) / (etaRight / CrGlobal * invDt + 1.f) -
    // 			    cosf(gPhi - halfStep) / (etaLeft / CrGlobal * invDt + 1.f));
# endif
    // # ifdef vair
    //     diva += (gTheta < M_hPI) * 4 * (1 - smoothstep(0.f, 10.f, currentTimeGlobal)) * cosTheta
    // 	* cosf(2 * gPhi) * radiusGlobal / UGlobal;
    // # endif
    
# ifdef gravity
    // sinThetaDiv += gGlobal * invGridLenGlobal *
    // 	(sinThetaSouth * sinThetaSouth / (invDt + CrGlobal / etaDown) -
    // 	 sinThetaNorth * sinThetaNorth / (invDt + CrGlobal / etaUp));
    sinThetaDiv += 0.57735026919 * gGlobal * invGridLenGlobal *
    	((cosf(gPhi + halfStep) - sinf(gPhi + halfStep)) / (invDt + CrGlobal / etaRight) -
    	 (cosf(gPhi - halfStep) - sinf(gPhi - halfStep)) / (invDt + CrGlobal / etaLeft) +
    	 (cosf(gTheta + halfStep) * (cosf(gPhi) + sinf(gPhi)) + sinThetaSouth)
    	 / (invDt + CrGlobal / etaDown) * sinThetaSouth -
    	 (cosf(gTheta - halfStep) * (cosf(gPhi) + sinf(gPhi)) + sinThetaNorth)
    	 / (invDt + CrGlobal / etaUp) * sinThetaNorth);
# endif
    rhs[idx] = sinTheta * invDt - sinThetaDiv;
}


void KaminoSolver::conjugateGradient() {
    
    const int max_iter = 1000;
    int k = 0;
    cusparseSpMatDescr_t matA;
    void *dBuffer = 0;
    size_t bufferSize;
    const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

    CHECK_CUDA(cudaMemcpy2D(d_x, nPhi * sizeof(float), surfConcentration->getGPUThisStep(),
			    surfConcentration->getThisStepPitchInElements() * sizeof(float),
			    nPhi * sizeof(float), nTheta,
			    cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaMemcpy(d_r, rhs, N * sizeof(float),
			  cudaMemcpyDeviceToDevice));

    // r = b - Ax
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, N, N, nz, row_ptr, col_ind,
    				     val, CUSPARSE_INDEX_32I,
    				     CUSPARSE_INDEX_32I,
    				     CUSPARSE_INDEX_BASE_ZERO,
    				     CUDA_R_32F));

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, trans,
					   &minusone, matA, vecX, &one, vecR, CUDA_R_32F,
					   CUSPARSE_CSRMV_ALG1, &bufferSize));

    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, trans,
    				&minusone, matA, vecX, &one, vecR, CUDA_R_32F,
    				CUSPARSE_CSRMV_ALG1, dBuffer));
    
    //CHECK_CUSPARSE(cusparseScsrmv(cusparseHandle, trans, N, N, nz, &minusone, descrA, val,
    //				  row_ptr, col_ind, d_x, &one, d_r));
    
    CHECK_CUBLAS(cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));

    while (r1 / N > epsilon*epsilon && k < max_iter) {
	k++;
        if (k == 1) {
	    cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
	} else {
	    beta = r1/r0;
            cublasSscal(cublasHandle, N, &beta, d_p, 1);
            cublasSaxpy(cublasHandle, N, &one, d_r, 1, d_p, 1);
	}
	CHECK_CUSPARSE(cusparseScsrmv(cusparseHandle, trans, N, N, nz, &one, descrA, val,
				      row_ptr, col_ind, d_p, &zero, d_omega));

        cublasSdot(cublasHandle, N, d_p, 1, d_omega, 1, &dot);
        alpha = r1/dot;
        cublasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1);
        nalpha = -alpha;
        cublasSaxpy(cublasHandle, N, &nalpha, d_omega, 1, d_r, 1);
        r0 = r1;
        cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);	
    }
    printf("  iteration = %3d, residual = %e \n", k, sqrt(r1/N));
    
    CHECK_CUDA(cudaMemcpy2D(surfConcentration->getGPUNextStep(),
			    surfConcentration->getNextStepPitchInElements() * sizeof(float),
			    d_x, nPhi * sizeof(float), nPhi * sizeof(float), nTheta,
			    cudaMemcpyDeviceToDevice));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));    
    CHECK_CUDA(cudaFree(dBuffer));
}


void KaminoSolver::AlgebraicMultiGridCG() {
    CHECK_CUDA(cudaMemcpy2D(d_x, nPhi * sizeof(float), surfConcentration->getGPUThisStep(),
			    surfConcentration->getThisStepPitchInElements() * sizeof(float),
			    nPhi * sizeof(float), nTheta,
			    cudaMemcpyDeviceToDevice));

    AMGX_vector_upload(b, N, 1, rhs);
    AMGX_vector_upload(x, N, 1, d_x);
    AMGX_matrix_upload_all(A, N, nz, 1, 1, row_ptr, col_ind, val, 0);
    AMGX_solver_setup(solver, A);
    AMGX_solver_solve(solver, b, x);
    AMGX_vector_download(x, d_x);
    CHECK_CUDA(cudaMemcpy2D(surfConcentration->getGPUNextStep(),
			    surfConcentration->getNextStepPitchInElements() * sizeof(float),
			    d_x, nPhi * sizeof(float), nPhi * sizeof(float), nTheta,
			    cudaMemcpyDeviceToDevice));
    int num_iter;
    AMGX_solver_get_iterations_number(solver, &num_iter);
    std::cout <<  "Total Iterations:  " << num_iter << std::endl;
}

__global__ void applyforcevelthetaKernel
(float* velThetaOutput, float* velThetaInput, float* velThetaDelta,
 float* velPhi, float* thickness, float* concentration, size_t pitch) {
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

# ifdef sphere
    float gTheta = ((float)thetaId + vThetaThetaOffset) * gridLenGlobal;
    float gPhi = ((float)phiId + vThetaPhiOffset) * gridLenGlobal;
# endif

    int thetaSouthId = thetaId + 1;

    float v1 = velThetaInput[thetaId * pitch + phiId];
    float u = sampleVPhi(velPhi, make_float2(gTheta, gPhi), pitch);

    float GammaNorth = concentration[thetaId * pitch + phiId];
    float GammaSouth = concentration[thetaSouthId * pitch + phiId];
    float DeltaNorth = thickness[thetaId * pitch + phiId];
    float DeltaSouth = thickness[thetaSouthId * pitch + phiId];

    // value at vTheta grid
    float invDelta = 2. / (DeltaNorth + DeltaSouth);

    // pGpy = \frac{\partial\Gamma}{\partial\theta};
    float pGpy = invGridLenGlobal * (GammaSouth - GammaNorth);

    // elasticity
    float f1 = -MGlobal * invDelta * pGpy;
    // air friction
    float vAir = 0.f;
# if defined vair && defined sphere
    vAir = (gTheta < M_hPI) * 2 * (1 - smoothstep(0.f, 10.f, currentTimeGlobal))
	* sinf(gTheta) * cosf(2 * gPhi) * radiusGlobal / UGlobal;
# endif
    float f2 = CrGlobal * invDelta * vAir;
    // gravity
    float f3 = 0.f;
# ifdef gravity
# ifdef sphere
    // f3 = gGlobal * sinf(gTheta);
    f3 = 0.57735026919 * (cosf(gTheta)*(cosf(gPhi) + sinf(gPhi))+sinf(gTheta)) * gGlobal;
# else
    f3 = gGlobal;
# endif
# endif
# ifdef greatCircle
    float f4 = 0.f;
# else
    float f4 = u * u * sinf(gTheta) * cosf(gTheta);
# endif

    velThetaOutput[thetaId * pitch + phiId] = (v1 / timeStepGlobal + f1 + f2 + f3 + f4) / (1./timeStepGlobal + CrGlobal * invDelta);
    velThetaDelta[thetaId * pitch + phiId] = velThetaOutput[thetaId * pitch + phiId] - v1;
}


__global__ void applyforcevelphiKernel
(float* velPhiOutput, float* velPhiInput, float* velPhiDelta, float* velTheta,
 float* thickness, float* concentration, size_t pitch) {
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

# ifdef sphere
    // Coord in phi-theta space
    float gPhi = ((float)phiId + vPhiPhiOffset) * gridLenGlobal;
    float gTheta = ((float)thetaId + vPhiThetaOffset) * gridLenGlobal;
    float sinTheta = sinf(gTheta);
# else
    sinTheta = 1.f; // no effect
# endif

    int phiWestId = (phiId - 1 + nPhiGlobal) % nPhiGlobal;

    // values at centered grid
    float DeltaWest = thickness[thetaId * pitch + phiWestId];
    float DeltaEast = thickness[thetaId * pitch + phiId];
    float GammaWest = concentration[thetaId * pitch + phiWestId];
    float GammaEast = concentration[thetaId * pitch + phiId];
    
    float u1 = velPhiInput[thetaId * pitch + phiId];
    float v = sampleVTheta(velTheta, make_float2(gTheta, gPhi), pitch);
        
    // value at uPhi grid
    float invDelta = 2. / (DeltaWest + DeltaEast);
    
    // pGpx = frac{1}{\sin\theta}\frac{\partial\Gamma}{\partial\phi};
    float pGpx = invGridLenGlobal * (GammaEast - GammaWest) / sinTheta;
    
    // elasticity
    float f1 = -MGlobal * invDelta * pGpx;
    // air friction
    float uAir = 0.f;
# if defined uair && defined sphere
    // uAir = 20.f * (1 - smoothstep(0.f, 10.f, currentTimeGlobal)) * (M_hPI - gTheta)
    // 	* expf(-10 * powf(fabsf(gTheta - M_hPI), 2.f)) * radiusGlobal
    // 	* cosf(gPhi) / UGlobal;
    uAir = 2.f * (1.f - smoothstep(0.f, 10.f, currentTimeGlobal)) / UGlobal * sinTheta
	* (1.f - smoothstep(M_PI * 0.1875, M_PI * 0.3125, fabsf(gTheta - M_hPI))) * radiusGlobal;
# endif
    float f2 = CrGlobal * invDelta * uAir;
    float f3;
# ifdef gravity
    f3 = 0.57735026919 * (cosf(gPhi) - sinf(gPhi)) * gGlobal;
# endif
# ifdef greatCircle
    float f4 = 0.f;
# else
    float f4 = -u1 * v * cosf(gTheta);
# endif
        
    velPhiOutput[thetaId * pitch + phiId] = (u1 * sinTheta / timeStepGlobal + f1 + f2 + f3 + f4)
	/ (1./timeStepGlobal + CrGlobal * invDelta) / sinTheta;
    velPhiDelta[thetaId * pitch + phiId] = velPhiOutput[thetaId * pitch + phiId] - u1;
}


// Backward Euler
// __global__ void applyforcevelthetaKernel_viscous
// (float* velThetaOutput, float* velThetaInput, float* velPhi, float* thickness,
//  float* concentration, float* divCentered, size_t pitch) {
//     int splitVal = nPhiGlobal / blockDim.x;
//     int threadSequence = blockIdx.x % splitVal;
//     int phiId = threadIdx.x + threadSequence * blockDim.x;
//     int thetaId = blockIdx.x / splitVal;

//     // Coord in phi-theta space
//     float gPhi = ((float)phiId + vPhiPhiOffset) * gridLenGlobal;
//     float gTheta = ((float)thetaId + vThetaThetaOffset) * gridLenGlobal;

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
//     float u0 = velPhi[thetaId * pitch + phiId];
//     float u1 = velPhi[thetaId * pitch + phiEastId];
//     float u2 = velPhi[thetaSouthId * pitch + phiId];
//     float u3 = velPhi[thetaSouthId * pitch + phiEastId];

//     // values at centered grid
//     float divNorth = divCentered[thetaId * nPhiGlobal + phiId];
//     float divSouth = divCentered[thetaSouthId * nPhiGlobal + phiId];
//     float GammaNorth = concentration[thetaId * pitch + phiId];
//     float GammaSouth = concentration[thetaSouthId * pitch + phiId];
//     float DeltaNorth = thickness[thetaId * pitch + phiId];
//     float DeltaSouth = thickness[thetaSouthId * pitch + phiId];

//     // values at vTheta grid
//     float div = 0.5 * (divNorth + divSouth);
//     float Delta = 0.5 * (DeltaNorth + DeltaSouth);
//     float invDelta = 1. / Delta;
//     float uPhi = 0.25 * (u0 + u1 + u2 + u3);

//     // pDpx = \frac{\partial\Delta}{\partial\phi}
//     float d0 = thickness[thetaId * pitch + phiWestId];
//     float d1 = thickness[thetaId * pitch + phiEastId];
//     float d2 = thickness[thetaSouthId * pitch + phiWestId];
//     float d3 = thickness[thetaSouthId * pitch + phiEastId];
//     float pDpx = 0.25 * invGridLenGlobal * (d1 + d3 - d0 - d2);
//     float pDpy = invGridLenGlobal * (DeltaSouth - DeltaNorth);

//     // pvpy = \frac{\partial u_theta}{\partial\theta}
//     float v0 = 0.0;
//     float v1 = velThetaInput[thetaId * pitch + phiId];
//     float v2 = 0.0;
//     float v3 = velThetaInput[thetaId * pitch + phiWestId];
//     float v4 = velThetaInput[thetaId * pitch + phiEastId];
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

//     float pvpy = 0.5 * invGridLenGlobal * (v2 - v0);
//     float pvpyNorth = invGridLenGlobal * (v1 - v0);
//     float pvpySouth = invGridLenGlobal * (v2 - v1);
	
//     // pvpx = \frac{\partial u_theta}{\partial\phi}    
//     float pvpx = 0.5 * invGridLenGlobal * (v4 - v3);
    
//     // pupy = \frac{\partial u_phi}{\partial\theta}
//     float pupy = 0.5 * invGridLenGlobal * (u2 + u3 - u0 - u1);

//     // pupx = \frac{\partial u_phi}{\partial\phi}
//     float pupx = 0.5 * invGridLenGlobal * (u1 + u3 - u0 - u2);

//     // pupxy = \frac{\partial^2u_\phi}{\partial\theta\partial\phi}
//     float pupxy = invGridLenGlobal * invGridLenGlobal * (u0 + u3 - u1 - u2);

//     // pvpxx = \frac{\partial^2u_\theta}{\partial\phi^2}    
//     float pvpxx = invGridLenGlobal * invGridLenGlobal * (v3 + v4 - 2 * v1);
    
//     // trigonometric function
//     float sinTheta = sinf(gTheta);
//     float cscTheta = 1. / sinTheta;
//     float cosTheta = cosf(gTheta);
//     float cotTheta = cosTheta * cscTheta;

//     // stress
//     // TODO: laplace term
//     float sigma11North = 0. +  2 * (pvpyNorth + divNorth);
//     float sigma11South = 0. +  2 * (pvpySouth + divSouth);
    
//     // float sigma11 = 0. +  2 * pvpy + 2 * div;
//     float sigma22 = -2 * (pvpy - 2 * div);
//     float sigma12 = cscTheta * pvpx + pupy - uPhi * cotTheta;

//     // psspy = \frac{\partial}{\partial\theta}(\sin\theta\sigma_{11})
//     float halfStep = 0.5 * gridLenGlobal;    
//     float thetaSouth = gTheta + halfStep;
//     float thetaNorth = gTheta - halfStep;    
//     float sinNorth = sinf(thetaNorth);
//     float sinSouth = sinf(thetaSouth);    
//     float psspy = invGridLenGlobal * (sigma11South * sinSouth - sigma11North * sinNorth);
    
//     // pspx = \frac{\partial\sigma_{12}}{\partial\phi}
//     float pspx = cscTheta * pvpxx + pupxy - cotTheta * pupx;

//     // pGpy = \frac{\partial\Gamma}{\partial\theta};
//     float pGpy = invGridLenGlobal * (GammaSouth - GammaNorth);

//     // force terms
//     float f1 = uPhi * uPhi * cotTheta;
//     float f2 = reGlobal * cscTheta * invDelta * pDpx * sigma12;
//     float f3 = -MGlobal * invDelta * pGpy;
//     float f4 = reGlobal * invDelta * pDpy * 2 * (div + pvpy);
//     float f5 = reGlobal * cscTheta * (psspy + pspx - cosTheta * sigma22);
    
// # ifdef gravity
//     float f7 = gGlobal * sinTheta;
// # else
//     float f7 = 0.0;
// # endif
//     float vAir = 0.0;
//     float f6 = CrGlobal * invDelta * (vAir - v1);
    
//     // output
//     float result = (v1 + timeStepGlobal * (f1 + f2 + f3 + f4 + f5 + CrGlobal * vAir + f7))
// 	/ (1.0 + CrGlobal * invDelta * timeStepGlobal);
//     // if (fabsf(result) < eps)
//     // 	result = 0.f;
//     velThetaOutput[thetaId * pitch + phiId] = result;
// }


// // Backward Euler
// __global__ void applyforcevelphiKernel_viscous
// (float* velPhiOutput, float* velTheta, float* velPhiInput, float* thickness,
//  float* concentration, float* divCentered, size_t pitch) {
//     int splitVal = nPhiGlobal / blockDim.x;
//     int threadSequence = blockIdx.x % splitVal;
//     int phiId = threadIdx.x + threadSequence * blockDim.x;
//     int thetaId = blockIdx.x / splitVal;

//     // Coord in phi-theta space
//     float gPhi = ((float)phiId + vPhiPhiOffset) * gridLenGlobal;
//     float gTheta = ((float)thetaId + vPhiThetaOffset) * gridLenGlobal;

//     int phiWestId = (phiId - 1 + nPhiGlobal) % nPhiGlobal;
//     int thetaNorthId< = thetaId - 1;
//     int thetaSouthId = thetaId + 1;

//     // values at centered grid
//     float divWest = divCentered[thetaId * nPhiGlobal + phiWestId];
//     float divEast = divCentered[thetaId * nPhiGlobal + phiId];
//     float DeltaWest = thickness[thetaId * pitch + phiWestId];
//     float DeltaEast = thickness[thetaId * pitch + phiId];
//     float GammaWest = concentration[thetaId * pitch + phiWestId];
//     float GammaEast = concentration[thetaId * pitch + phiId];
    
//     // |  d0 u3 d2 |
//     // +  v0 +  v1 +
//     // u0    u1    u2
//     // +  v2 +  v3 + 
//     // |  d1 u4 d3 |
//     //
//     // u1 is the current velPhi
//     float v0 = 0.0;
//     float v1 = 0.0;
//     float v2 = 0.0;
//     float v3 = 0.0;
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
//     float Delta = 0.5 * (DeltaWest + DeltaEast);
//     float invDelta = 1. / Delta;
//     float div = 0.5 * (divWest + divEast);
//     float vTheta = 0.25 * (v0 + v1 + v2 + v3);

//     // pvpx = \frac{\partial u_theta}{\partial\phi}
//     float pvpx = 0.5 * invGridLenGlobal * (v1 + v3 - v0 - v2);

//     // pvpy = \frac{\partial u_theta}{\partial\theta}
//     float pvpyWest = invGridLenGlobal * (v2 - v0);
//     float pvpyEast = invGridLenGlobal * (v3 - v1);
//     float pvpy = 0.5 * invGridLenGlobal * (v2 + v3 - v0 - v1);

//     // pupy = \frac{\partial u_phi}{\partial\theta}
//     float pupyNorth = 0.0;
//     float pupySouth = 0.0;
//     float u1 = velPhiInput[thetaId * pitch + phiId];
//     float u3 = 0.0;
//     float u4 = 0.0;
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
//     float pupy = 0.5 * invGridLenGlobal * (u4 - u3);

//     // pGpx = \frac{\partial\Gamma}{\partial\phi};
//     float pGpx = invGridLenGlobal * (GammaEast - GammaWest);

//     // trigonometric function
//     float sinTheta = sinf(gTheta);
//     float cscTheta = 1. / sinTheta;
//     float cosTheta = cosf(gTheta);
//     float cotTheta = cosTheta * cscTheta;
    
//     // stress
//     // TODO: laplace term
//     float sigma12 = cscTheta * pvpx + pupy - u1 * cotTheta;

//     // pDpx = \frac{\partial\Delta}{\partial\phi}
//     float pDpx = invGridLenGlobal * (DeltaEast - DeltaWest);

//     // pDpy = \frac{\partial\Delta}{\partial\theta}
//     // TODO: do we need to average the thickness value at the pole?
//     float pDpy = 0.0;
//     float d0 = 0.0;
//     float d1 = 0.0;
//     float d2 = 0.0;
//     float d3 = 0.0;
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
//     float halfStep = 0.5 * gridLenGlobal;    
//     float thetaSouth = gTheta + halfStep;
//     float thetaNorth = gTheta - halfStep;    
//     float sinNorth = sinf(thetaNorth);
//     float sinSouth = sinf(thetaSouth);
//     float cosNorth = cosf(thetaNorth);
//     float cosSouth = cosf(thetaSouth);
//     // TODO: uncertain about the definintion of u_\phi at both poles
//     float uNorth = 0.5 * (u3 + u1);
//     float uSouth = 0.5 * (u1 + u4);
//     float psspy = invGridLenGlobal * (invGridLenGlobal * (v0 + v3 - v1 - v2) +
// 							sinSouth * pupySouth - sinNorth * pupyNorth -
// 							cosSouth * uSouth + cosNorth * uNorth);
    
//     // pspx = \frac{\partial\sigma_{22}}{\partial\phi}
//     float sigma22West = 2 * (2 * divWest - pvpyWest);
//     float sigma22East = 2 * (2 * divEast - pvpyEast);
//     float pspx = invGridLenGlobal * (sigma22East - sigma22West);
    
//     // force terms
//     // float f1 = -vTheta * u1 * cotTheta;
//     float f2 = reGlobal * invDelta * pDpy * sigma12;
//     float f3 = -MGlobal * invDelta * cscTheta * pGpx;
//     float f4 = reGlobal * invDelta * cscTheta * pDpx * 2 * ( 2 * div - pvpy);
//     float f5 = reGlobal * cscTheta * (psspy + pspx + cosTheta * sigma12);

//     // float f7 = 0.0; 		// gravity
//     float uAir = 0.0;
// # ifdef uair
//     if (currentTimeGlobal < 5)
//     	uAir = 20.f * (M_hPI - gTheta) * expf(-10 * powf(fabsf(gTheta - M_hPI), 2.f)) * radiusGlobal * cosf(gPhi) / UGlobal;
// # endif

//     float f6 = CrGlobal * invDelta * (uAir - u1);
    
//     // output
//     float result = (u1 + timeStepGlobal * (f2 + f3 + f4 + f5 + CrGlobal * uAir))
// 	/ (1.0 + (CrGlobal * invDelta + vTheta * cotTheta) * timeStepGlobal);
//     velPhiOutput[thetaId * pitch + phiId] = result;
// }


__global__ void resetThickness(float2* weight) {
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int idx = blockIdx.x * blockDim.x + phiId;

    weight[idx].x = 0.;
    weight[idx].y = 0.;
}

__global__ void applyforceThickness
(float* thicknessOutput, float* thicknessInput, float* thicknessDelta,
 float* div, size_t pitch)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    float eta = thicknessInput[thetaId * pitch + phiId];
    float f = div[thetaId * nPhiGlobal + phiId];

    thicknessOutput[thetaId * pitch + phiId] = eta * (1 - timeStepGlobal * f);
    thicknessDelta[thetaId * nPhiGlobal + phiId] = thicknessOutput[thetaId * pitch + phiId] - eta;
}


// Backward Euler
// __global__ void applyforceSurfConcentration
// (float* sConcentrationOutput, float* sConcentrationInput, float* div, size_t pitch)
// {
//     // Index
//     int splitVal = nPhiGlobal / blockDim.x;
//     int threadSequence = blockIdx.x % splitVal;
//     int phiId = threadIdx.x + threadSequence * blockDim.x;
//     int thetaId = blockIdx.x / splitVal;

//     float thetaCoord = ((float)thetaId + centeredThetaOffset) * gridLenGlobal;
    
//     float halfStep = 0.5 * gridLenGlobal;

//     float cscTheta = 1.f / sinf(thetaCoord);
//     float sinThetaSouth = sinf(thetaCoord + halfStep);
//     float sinThetaNorth = sinf(thetaCoord - halfStep);

//     float gamma = sConcentrationInput[thetaId * pitch + phiId];
//     float gammaWest = sConcentrationInput[thetaId * pitch + (phiId - 1 + nPhiGlobal) % nPhiGlobal];
//     float gammaEast = sConcentrationInput[thetaId * pitch + (phiId + 1) % nPhiGlobal];
//     float gammaNorth = 0.0;
//     float gammaSouth = 0.0;
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
//     float laplace = invGridLenGlobal * invGridLenGlobal * cscTheta *
// 	(sinThetaSouth * (gammaSouth - gamma) - sinThetaNorth * (gamma - gammaNorth) +
// 		    cscTheta * (gammaEast + gammaWest - 2 * gamma));
// # else
//     float laplace = invGridLenGlobal * invGridLenGlobal * 
//     	(gammaWest - 4*gamma + gammaEast + gammaNorth + gammaSouth);
// #endif
    
//     float f = div[thetaId * nPhiGlobal + phiId];
//     // float f2 = DsGlobal * laplace;
//     float f2 = 0.f;

//     sConcentrationOutput[thetaId * pitch + phiId] = max((gamma + f2 * timeStepGlobal) / (1 + timeStepGlobal * f), 0.f);
// }


/**
 * a = b - c
 * b and c are pitched memory
 */
__global__ void substractPitched(float* a, float* b, float* c, size_t pitch) {
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    at(a, thetaId, phiId) = at(b, thetaId, phiId, pitch) - at(c, thetaId, phiId, pitch);
}


__global__ void accumulateChangesThickness(float* thicknessDelta, float* thicknessDeltaTemp,
					   float* fwd_t, float* fwd_p,
					   size_t pitch) {
    float w[5] = {0.125f, 0.125f, 0.125f, 0.125f, 0.5f};
    float2 dir[5] = {make_float2(-0.25f,-0.25f),
		     make_float2(0.25f, -0.25f),
		     make_float2(-0.25f, 0.25f),
		     make_float2( 0.25f, 0.25f),
		     make_float2(0.f, 0.f)};

    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    
    // Coord in scalar space
    float2 gId = make_float2((float)thetaId, (float)phiId) + centeredOffset;

    for(int i = 0; i < 5; i++) {
	float2 posId = gId + dir[i];
	float2 initPosId = sampleMapping(fwd_t, fwd_p, posId);
	
	// at(gammaDelta, thetaId, phiId, pitch)
	//     += w[i] * sampleCentered(gammaDeltaTemp, initPosId, nPhiGlobal);
	at(thicknessDelta, thetaId, phiId, pitch)
	    += w[i] * sampleCentered(thicknessDeltaTemp, initPosId, nPhiGlobal);
    }
}


__global__ void accumulateChangesVTheta(float* vThetaDelta, float* vThetaDeltaTemp,
					float* fwd_t, float* fwd_p,
					size_t pitch) {
    float w[5] = {0.125f, 0.125f, 0.125f, 0.125f, 0.5f};
    float2 dir[5] = {make_float2(-0.25f,-0.25f),
		     make_float2(0.25f, -0.25f),
		     make_float2(-0.25f, 0.25f),
		     make_float2( 0.25f, 0.25f),
		     make_float2(0.f, 0.f)};

    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    
    // Coord in vTheta space
    float2 gId = make_float2((float)thetaId, (float)phiId) + vThetaOffset;

    for(int i = 0; i < 5; i++) {
	float2 posId = gId + dir[i];
	float2 initPosId = sampleMapping(fwd_t, fwd_p, posId);
	
	at(vThetaDelta, thetaId, phiId, pitch)
	    += w[i] * sampleVTheta(vThetaDeltaTemp, initPosId, nPhiGlobal);
    }
}

__global__ void accumulateChangesVPhi(float* vPhiDelta, float* vPhiDeltaTemp,
				      float* fwd_t, float* fwd_p,
				      size_t pitch) {
    float w[5] = {0.125f, 0.125f, 0.125f, 0.125f, 0.5f};
    float2 dir[5] = {make_float2(-0.25f,-0.25f),
		     make_float2(0.25f, -0.25f),
		     make_float2(-0.25f, 0.25f),
		     make_float2( 0.25f, 0.25f),
		     make_float2(0.f, 0.f)};

    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    
    // Coord in vPhi space
    float2 gId = make_float2((float)thetaId, (float)phiId) + vPhiOffset;

    for(int i = 0; i < 5; i++) {
	float2 posId = gId + dir[i];
	float2 initPosId = sampleMapping(fwd_t, fwd_p, posId);
	
	at(vPhiDelta, thetaId, phiId, pitch)
	    += w[i] * sampleVPhi(vPhiDeltaTemp, initPosId, nPhiGlobal);
    }
}


__global__ void applyforceParticles
(fReal* tempVal, fReal* value, fReal* coord, fReal* div, size_t numOfParticles) {
    int particleId = blockIdx.x * blockDim.x + threadIdx.x;

    if (particleId < numOfParticles) {
	fReal thetaId = coord[2 * particleId];
	fReal phiId = coord[2 * particleId + 1];

	float2 gId = make_float2(thetaId, phiId);
	    
	fReal f = sampleCentered(div, gId, nPhiGlobal);

	tempVal[particleId] = value[particleId] / (1 + timeStepGlobal * f);
    }
}


__global__ void mapParticlesToThickness
(fReal* particleCoord, fReal* particleVal, float2* weight, size_t numParticles)
{
    // Index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int particleId = index >> 1; // (index / 2)
    int partition = index & 1;	 // (index % 2)

    if (particleId < numParticles) {
	fReal gThetaId = particleCoord[2 * particleId];
	fReal gPhiId = particleCoord[2 * particleId + 1];

	fReal gTheta = gThetaId * gridLenGlobal;

	fReal sinTheta = sinf(gTheta);
	if (sinTheta < 1e-7f)
	    return;
	fReal gPhi = gPhiId * gridLenGlobal;

	size_t thetaId = static_cast<size_t>(floorf(gThetaId));

	fReal x1 = cosf(gPhi) * sinTheta; fReal y1 = sinf(gPhi) * sinTheta; fReal z1 = cosf(gTheta);

	fReal theta = (thetaId + 0.5) * gridLenGlobal;

	fReal phiRange = 0.5/sinTheta;
	int minPhiId = static_cast<int>(ceilf(gPhiId - phiRange));
	int maxPhiId = static_cast<int>(floorf(gPhiId + phiRange));

	fReal z2 = cosf(theta);
	fReal r = sinf(theta);
	fReal value = particleVal[particleId];

	int begin; int end;
	
	if (partition == 0) {
	    begin = minPhiId; end = static_cast<int>(gPhiId);
	} else {
	    begin = static_cast<int>(gPhiId); end = maxPhiId + 1;
	}
	    
	for (int phiId = begin; phiId < end; phiId++) {
	    fReal phi = phiId * gridLenGlobal;
	    fReal x2 = cosf(phi) * r; fReal y2 = sinf(phi) * r;

	    fReal dist2 = powf(fabsf(x1 - x2), 2.f) + powf(fabsf(y1 - y2), 2.f) + powf(fabsf(z1 - z2), 2.f);
	        
	    if (dist2 <= .25f) {
		fReal w = expf(-10*dist2);
		size_t normalizedPhiId = (phiId + nPhiGlobal) % nPhiGlobal;
		float2* currentWeight = weight + (thetaId * nPhiGlobal + normalizedPhiId);
		atomicAdd(&(currentWeight->x), w);
		atomicAdd(&(currentWeight->y), w * value);
	    }
	}
    }
}


__global__ void normalizeThickness
(fReal* thicknessOutput, fReal* thicknessInput, fReal* velPhi, fReal* velTheta,
 fReal* div, float2* weight, size_t pitch) {
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    float2* currentWeight = weight + (thetaId * nPhiGlobal + phiId);
    fReal w = currentWeight->x;
    fReal val = currentWeight->y;

    if (w > 0) {
	thicknessOutput[thetaId * pitch + phiId] = val / w;
    } else {
	//	printf("Warning: no particles contributed to grid thetaId %d, phiId %d\n", thetaId, phiId);
	// Coord in scalar space
	float2 gId = make_float2((float)thetaId, (float)phiId) + centeredOffset;
# ifdef RK3
	float2 traceId = traceRK3(velTheta, velPhi, timeStepGlobal, gId, pitch);
# else
	float2 traceId = traceRK2(velTheta, velPhi, timeStepGlobal, gId, pitch);
# endif

        fReal advectedVal = sampleCentered(thicknessInput, traceId, pitch);
	fReal f = div[thetaId * nPhiGlobal + phiId];
	thicknessOutput[thetaId * pitch + phiId] = advectedVal / (1 + timeStepGlobal * f);
    }
}


__global__ void correctThickness1(float* thicknessCurr, float* thicknessError, float* thicknessDelta,
				  float* thicknessInit, float* fwd_t, float* fwd_p,
				  size_t pitch) {
    float w[5] = {0.125f, 0.125f, 0.125f, 0.125f, 0.5f};
    float2 dir[5] = {make_float2(-0.25f,-0.25f),
		     make_float2(0.25f, -0.25f),
		     make_float2(-0.25f, 0.25f),
		     make_float2( 0.25f, 0.25f),
		     make_float2(0.f, 0.f)};

    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    float thickness = 0.f;
    // float gamma = 0.f;

    // Coord in scalar space
    float2 gId = make_float2((float)thetaId, (float)phiId) + centeredOffset;
    for (int i = 0; i < 5; i++) {
	float2 posId = gId + dir[i];
	float2 initPosId = sampleMapping(fwd_t, fwd_p, posId);
	thickness += w[i] * sampleCentered(thicknessCurr, initPosId, pitch);
	// gamma += w[i] * sampleCentered(gammaCurr, initPosId, pitch);
    }
    at(thicknessError, thetaId, phiId) = (thickness - at(thicknessDelta, thetaId, phiId, pitch)
					  - at(thicknessInit, thetaId, phiId, pitch)) * 0.5f;
    // at(gammaError, thetaId, phiId) = (gamma - at(gammaDelta, thetaId, phiId, pitch)
    // 					  - at(gammaInit, thetaId, phiId, pitch)) * 0.5f;
}


__global__ void correctThickness2(float* thicknessOutput, float* thicknessInput,
				 float* thicknessError, float* bwd_t, float* bwd_p, size_t pitch) {
    float w[5] = {0.125f, 0.125f, 0.125f, 0.125f, 0.5f};
    float2 dir[5] = {make_float2(-0.25f,-0.25f),
		     make_float2(0.25f, -0.25f),
		     make_float2(-0.25f, 0.25f),
		     make_float2( 0.25f, 0.25f),
		     make_float2(0.f, 0.f)};

    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    // Coord in scalar space
    float2 gId = make_float2((float)thetaId, (float)phiId) + centeredOffset;

    // sample
    // if (thetaId < float(nThetaGlobal) / 16.f || thetaId > float(nThetaGlobal) * 15.f / 16.f)
    // 	return;
    
    for (int i = 0; i < 5; i++) {
	float2 posId = gId + dir[i];
	float2 sampleId = sampleMapping(bwd_t, bwd_p, posId);
	at(thicknessOutput, thetaId, phiId, pitch)
	    -= w[i] * sampleCentered(thicknessError, sampleId, nPhiGlobal);
	// at(gammaOutput, thetaId, phiId, pitch)
	//     -= w[i] * sampleCentered(gammaError, sampleId, nPhiGlobal);
    }

    // clamp local extrema
    int range[] = {-1, 0, 1};

    float minVal = at(thicknessInput, thetaId, phiId, pitch);
    float maxVal = minVal;
    for (int t : range) {
	for (int p : range) {
	    int2 sampleId = make_int2(thetaId + t, phiId + p);
	    validateId(sampleId);
	    float currentVal = at(thicknessInput, sampleId, pitch);
	    minVal = fminf(minVal, currentVal);
	    maxVal = fmaxf(maxVal, currentVal);
	}
    }
    at(thicknessOutput, thetaId, phiId, pitch)
	= clamp(at(thicknessOutput, thetaId, phiId, pitch), minVal, maxVal);


    // minVal = at(gammaInput, thetaId, phiId, pitch);
    // maxVal = 0.f;
    // for (int t : range) {
    // 	for (int p : range) {
    // 	    int2 sampleId = make_int2(thetaId + t, phiId + p);
    // 	    validateId(sampleId);
    // 	    float currentVal = at(gammaInput, sampleId, pitch);
    // 	    minVal = fminf(minVal, currentVal);
    // 	    maxVal = fmaxf(maxVal, currentVal);
    // 	}
    // }
    // at(gammaOutput, thetaId, phiId, pitch)
    // 	= clamp(at(gammaOutnput, thetaId, phiId, pitch), minVal, maxVal);
}

__global__ void correctVTheta1(float* vThetaCurr, float* vThetaError, float* vThetaDelta,
			       float* vThetaInit, float* fwd_t, float* fwd_p,
			       size_t pitch) {
    float w[5] = {0.125f, 0.125f, 0.125f, 0.125f, 0.5f};
    float2 dir[5] = {make_float2(-0.25f,-0.25f),
		     make_float2(0.25f, -0.25f),
		     make_float2(-0.25f, 0.25f),
		     make_float2( 0.25f, 0.25f),
		     make_float2(0.f, 0.f)};

    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    float vTheta = 0.f;

    // Coord in vTheta space
    float2 gId = make_float2((float)thetaId, (float)phiId) + vThetaOffset;
    for (int i = 0; i < 5; i++) {
	float2 posId = gId + dir[i];
	float2 initPosId = sampleMapping(fwd_t, fwd_p, posId);
	vTheta += w[i] * sampleVTheta(vThetaCurr, initPosId, pitch);
    }
    at(vThetaError, thetaId, phiId) = (vTheta - at(vThetaDelta, thetaId, phiId, pitch)
				       - at(vThetaInit, thetaId, phiId, pitch)) * 0.5f;
}


__global__ void correctVTheta2(float* vThetaOutput, float* vThetaInput,
			       float* vThetaError, float* bwd_t, float* bwd_p, size_t pitch) {
    float w[5] = {0.125f, 0.125f, 0.125f, 0.125f, 0.5f};
    float2 dir[5] = {make_float2(-0.25f,-0.25f),
		     make_float2(0.25f, -0.25f),
		     make_float2(-0.25f, 0.25f),
		     make_float2( 0.25f, 0.25f),
		     make_float2(0.f, 0.f)};

    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    // Coord in scalar space
    float2 gId = make_float2((float)thetaId, (float)phiId) + vThetaOffset;

    // sample
    if (thetaId < float(nThetaGlobal) / 16.f || thetaId > float(nThetaGlobal) * 15.f / 16.f)
	return;
    
    for (int i = 0; i < 5; i++) {
	float2 posId = gId + dir[i];
	float2 sampleId = sampleMapping(bwd_t, bwd_p, posId);
	at(vThetaInput, thetaId, phiId, pitch)
	    -= w[i] * sampleVTheta(vThetaError, sampleId, nPhiGlobal);
    }

    // clamp local extrema
    int range[] = {-1, 0, 1};

    float minVal = at(vThetaOutput, thetaId, phiId, pitch);
    float maxVal = 0.f;
    for (int t : range) {
	for (int p : range) {
	    int2 sampleId = make_int2(thetaId + t, phiId + p);
	    validateId(sampleId);
	    float currentVal = at(vThetaOutput, sampleId, pitch);
	    minVal = fminf(minVal, currentVal);
	    maxVal = fmaxf(maxVal, currentVal);
	}
    }
    at(vThetaOutput, thetaId, phiId, pitch)
	= clamp(at(vThetaInput, thetaId, phiId, pitch), minVal, maxVal);
}

__global__ void correctVPhi1(float* vPhiCurr, float* vPhiError, float* vPhiDelta,
			     float* vPhiInit, float* fwd_t, float* fwd_p,
			     size_t pitch) {
    float w[5] = {0.125f, 0.125f, 0.125f, 0.125f, 0.5f};
    float2 dir[5] = {make_float2(-0.25f,-0.25f),
		     make_float2(0.25f, -0.25f),
		     make_float2(-0.25f, 0.25f),
		     make_float2( 0.25f, 0.25f),
		     make_float2(0.f, 0.f)};

    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    float vPhi = 0.f;

    // Coord in vPhi space
    float2 gId = make_float2((float)thetaId, (float)phiId) + vPhiOffset;
    for (int i = 0; i < 5; i++) {
	float2 posId = gId + dir[i];
	float2 initPosId = sampleMapping(fwd_t, fwd_p, posId);
	vPhi += w[i] * sampleVPhi(vPhiCurr, initPosId, pitch);
    }
    at(vPhiError, thetaId, phiId) = (vPhi - at(vPhiDelta, thetaId, phiId, pitch)
				     - at(vPhiInit, thetaId, phiId, pitch)) * 0.5f;
}


__global__ void correctVPhi2(float* vPhiOutput, float* vPhiInput,
			     float* vPhiError, float* bwd_t, float* bwd_p, size_t pitch) {
    float w[5] = {0.125f, 0.125f, 0.125f, 0.125f, 0.5f};
    float2 dir[5] = {make_float2(-0.25f,-0.25f),
		     make_float2(0.25f, -0.25f),
		     make_float2(-0.25f, 0.25f),
		     make_float2( 0.25f, 0.25f),
		     make_float2(0.f, 0.f)};

    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    // Coord in scalar space
    float2 gId = make_float2((float)thetaId, (float)phiId) + vPhiOffset;

    // sample
    if (thetaId < float(nThetaGlobal) / 16.f || thetaId > float(nThetaGlobal) * 15.f / 16.f)
	return;
    
    for (int i = 0; i < 5; i++) {
	float2 posId = gId + dir[i];
	float2 sampleId = sampleMapping(bwd_t, bwd_p, posId);
	at(vPhiInput, thetaId, phiId, pitch)
	    -= w[i] * sampleVPhi(vPhiError, sampleId, nPhiGlobal);
    }

    // clamp local extrema
    int range[] = {-1, 0, 1};

    float minVal = at(vPhiOutput, thetaId, phiId, pitch);
    float maxVal = 0.f;
    for (int t : range) {
	for (int p : range) {
	    int2 sampleId = make_int2(thetaId + t, phiId + p);
	    validateId(sampleId);
	    float currentVal = at(vPhiOutput, sampleId, pitch);
	    minVal = fminf(minVal, currentVal);
	    maxVal = fmaxf(maxVal, currentVal);
	}
    }
    at(vPhiOutput, thetaId, phiId, pitch)
	= clamp(at(vPhiInput, thetaId, phiId, pitch), minVal, maxVal);
}


void KaminoSolver::bodyforce() {
    dim3 gridLayout;
    dim3 blockLayout;

    bool inviscid = true;

    determineLayout(gridLayout, blockLayout, 1, N + 1);
    initLinearSystem<<<gridLayout, blockLayout>>>(row_ptr, col_ind);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    concentrationLinearSystemKernel<<<gridLayout, blockLayout>>>
    	(velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), surfConcentration->getGPUThisStep(),
	 thickness->getGPUThisStep(), val, rhs, thickness->getThisStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

# ifdef PERFORMANCE_BENCHMARK
    KaminoTimer CGtimer;
    CGtimer.startTimer();
# endif
    AlgebraicMultiGridCG();
    // conjugateGradient();
# ifdef PERFORMANCE_BENCHMARK
    this->CGTime += CGtimer.stopTimer() * 0.001f;
# endif
    
    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    divergenceKernel_fromGamma<<<gridLayout, blockLayout>>>
    	(div, surfConcentration->getGPUNextStep(), surfConcentration->getGPUThisStep(),
    	 surfConcentration->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    if (particleDensity > 0) {
    	determineLayout(gridLayout, blockLayout, 1, particles->numOfParticles);
    	applyforceParticles<<<gridLayout, blockLayout>>>
    	    (particles->tempVal, particles->value, particles->coordGPUThisStep, div, particles->numOfParticles);
    	checkCudaErrors(cudaGetLastError());

	// reset weight
    	determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    	resetThickness<<<gridLayout, blockLayout>>>(weight);
    	checkCudaErrors(cudaGetLastError());
    	checkCudaErrors(cudaDeviceSynchronize());

    	determineLayout(gridLayout, blockLayout, 2, particles->numOfParticles);
    	mapParticlesToThickness<<<gridLayout, blockLayout>>>
    	    (particles->coordGPUThisStep, particles->tempVal, weight, particles->numOfParticles);
    	checkCudaErrors(cudaGetLastError());
    	checkCudaErrors(cudaDeviceSynchronize());

	determineLayout(gridLayout, blockLayout, thickness->getNTheta(), thickness->getNPhi());
	normalizeThickness<<<gridLayout, blockLayout>>>
	    (thickness->getGPUNextStep(), thickness->getGPUThisStep(), velPhi->getGPUNextStep(),
    	     velTheta->getGPUNextStep(), div, weight, pitch);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());       
    } else {
	// write thickness difference to tmp_t, gamma difference to tmp_p
	determineLayout(gridLayout, blockLayout, thickness->getNTheta(), thickness->getNPhi());
	applyforceThickness<<<gridLayout, blockLayout>>>
	    (thickness->getGPUNextStep(), thickness->getGPUThisStep(),
	     tmp_t, div, thickness->getNextStepPitchInElements());
    
	// substractPitched<<<gridLayout, blockLayout>>>
	// 	(tmp_p, surfConcentration->getGPUNextStep(),
	// 	 surfConcentration->getGPUThisStep(), pitch);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	accumulateChangesThickness<<<gridLayout, blockLayout>>>
	    (thickness->getGPUDelta(), tmp_t, forward_t, forward_p, pitch);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
    }

    determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
    applyforcevelthetaKernel<<<gridLayout, blockLayout>>>
    	(velTheta->getGPUNextStep(), velTheta->getGPUThisStep(), tmp_t, velPhi->getGPUThisStep(),
    	 thickness->getGPUThisStep(), surfConcentration->getGPUNextStep(), velTheta->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    
    // accumulateChangesVTheta<<<gridLayout, blockLayout>>>
    // 	(velTheta->getGPUDelta(), tmp_t, forward_t, forward_p, pitch);
    // checkCudaErrors(cudaGetLastError());
	
    determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());
    applyforcevelphiKernel<<<gridLayout, blockLayout>>>
    	(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), tmp_p, velTheta->getGPUThisStep(),
	 thickness->getGPUThisStep(), surfConcentration->getGPUNextStep(), velPhi->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    
    // accumulateChangesVPhi<<<gridLayout, blockLayout>>>
    // 	(velPhi->getGPUDelta(), tmp_p, forward_t, forward_p, pitch);
    // checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // printGPUarraytoMATLAB<float>("dgamma.txt", surfConcentration->getGPUDelta(),
    // 				 nTheta, nPhi, pitch);
    // printGPUarraytoMATLAB<float>("gamma.txt", surfConcentration->getGPUNextStep(),
    // 				 nTheta, nPhi, pitch);

    // printGPUarraytoMATLAB<float>("deta.txt", thickness->getGPUDelta(),
    // 				 nTheta, nPhi, pitch);

    if (particleDensity > 0) std::swap(particles->tempVal, particles->value);
    thickness->swapGPUBuffer();
    surfConcentration->swapGPUBuffer();
    swapVelocityBuffers();
}


void KaminoSolver::reInitializeMapping() {
    dim3 gridLayout;
    dim3 blockLayout;
    determineLayout(gridLayout, blockLayout, thickness->getNTheta(), thickness->getNPhi());

    bool errorCorrection = true;
    if (errorCorrection) {
	CHECK_CUDA(cudaMemcpy(thickness->getGPUNextStep(), thickness->getGPUThisStep(),
			      thickness->getNTheta() * pitch * sizeof(float),
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
    }

    //  determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
    // correctVTheta1<<<gridLayout, blockLayout>>>
    // 	(velTheta->getGPUThisStep(), tmp_t, velTheta->getGPUDelta(), velTheta->getGPUInit(),
    // 	 forward_t, forward_p, pitch);
    // checkCudaErrors(cudaGetLastError());
    // checkCudaErrors(cudaDeviceSynchronize());

    // CHECK_CUDA(cudaMemcpy(velTheta->getGPUNextStep(), velTheta->getGPUThisStep(),
    // 			  velTheta->getNTheta() * pitch * sizeof(float),
    // 			  cudaMemcpyDeviceToDevice));
      
    // correctVTheta2<<<gridLayout, blockLayout>>>
    // 	(velTheta->getGPUNextStep(), velTheta->getGPUThisStep(), tmp_t,
    // 	 backward_t, backward_p, pitch);
    // checkCudaErrors(cudaGetLastError());
    // checkCudaErrors(cudaDeviceSynchronize());

    
    //  determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());
    // correctVPhi1<<<gridLayout, blockLayout>>>
    // 	(velPhi->getGPUThisStep(), tmp_t, velPhi->getGPUDelta(), velPhi->getGPUInit(),
    // 	 forward_t, forward_p, pitch);
    // checkCudaErrors(cudaGetLastError());
    // checkCudaErrors(cudaDeviceSynchronize());

    // CHECK_CUDA(cudaMemcpy(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(),
    // 			  velPhi->getNTheta() * pitch * sizeof(float),
    // 			  cudaMemcpyDeviceToDevice));
      
    // correctVPhi2<<<gridLayout, blockLayout>>>
    // 	(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), tmp_t,
    // 	 backward_t, backward_p, pitch);
    // checkCudaErrors(cudaGetLastError());
    // checkCudaErrors(cudaDeviceSynchronize());
    
    //swapVelocityBuffers();
    // thickness->swapGPUBuffer();

    std::swap(this->thickness->getGPUInitLast(), this->thickness->getGPUInit());
    std::swap(this->thickness->getGPUDeltaLast(), this->thickness->getGPUDelta());
    std::swap(backward_tprev, backward_t);
    std::swap(backward_pprev, backward_p);

    CHECK_CUDA(cudaMemcpy(this->thickness->getGPUInit(), this->thickness->getGPUThisStep(),
			  this->thickness->getThisStepPitchInElements() * this->thickness->getNTheta() *
			  sizeof(float), cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaMemset(this->thickness->getGPUDelta(), 0,
			  pitch * sizeof(float) * this->thickness->getNTheta()));

    // CHECK_CUDA(cudaMemcpy(this->velTheta->getGPUInit(), this->velTheta->getGPUThisStep(),
    // 			  this->velTheta->getThisStepPitchInElements() * this->velTheta->getNTheta() *
    // 			  sizeof(float), cudaMemcpyDeviceToDevice));

    // CHECK_CUDA(cudaMemset(this->velTheta->getGPUDelta(), 0,
    // 			  pitch * this->velTheta->getNTheta()));

    // CHECK_CUDA(cudaMemcpy(this->velPhi->getGPUInit(), this->velPhi->getGPUThisStep(),
    // 			  this->velPhi->getThisStepPitchInElements() * this->velPhi->getNTheta() *
    // 			  sizeof(float), cudaMemcpyDeviceToDevice));

    // CHECK_CUDA(cudaMemset(this->velPhi->getGPUDelta(), 0,
    // 			  pitch * this->velPhi->getNTheta()));

    initMapping<<<gridLayout, blockLayout>>>(forward_t, forward_p);
    initMapping<<<gridLayout, blockLayout>>>(backward_t, backward_p);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}


class CostFunctor {
public:
    CostFunctor(KaminoSolver* solver, double2 Id, double alphaTheta, double alphaPhi):
	solver(solver), Id(Id), alphaTheta(alphaTheta), alphaPhi(alphaPhi) {
	gridLen = static_cast<double>(solver->getGridLen());
    }

    template <typename T>
    bool operator()(const T* const llt, const T* const llp,
		    const T* const lrt, const T* const lrp,
		    const T* const hlt, const T* const hlp,
		    const T* const hrt, const T* const hrp,
		    T* residuals) const {
	std::vector<T> ll{*llt, *llp};
	std::vector<T> lr{*lrt, *lrp};
	std::vector<T> hl{*hlt, *hlp};
	std::vector<T> hr{*hrt, *hrp};
	// std::cout << "ll " << ll[0] << " " << ll[1] << "lr " << lr[0] << " " << lr[1] << "hl " << hl[0] << " " << hl[1] << "hr " << hr[0] << " " << hr[1] << std::endl;
	std::vector<T> sampledId = lerpCoords(lerpCoords(ll, lr, alphaPhi),
					      lerpCoords(hl, hr, alphaPhi),
					      alphaTheta);
	// std::cout << dist(sampledId, Id) << std::endl;
	residuals[0] = dist(sampledId, Id);
	return true;
    }
    KaminoSolver* solver;
    double2 Id;
    double alphaTheta;
    double alphaPhi;
    double gridLen;

    template <typename T>
    T dist(std::vector<T> Id1, double2 Id2) const {
    	Id1[0] *= gridLen;
	Id1[1] *= gridLen;
    	Id2 *= gridLen;
	std::vector<T> Id13{cos(Id1[1]) * sin(Id1[0]),
		            sin(Id1[1]) * sin(Id1[0]),
		            cos(Id1[0])};
	//	normalize(Id13);
	
	std::vector<double> Id23{cos(Id2.y) * sin(Id2.x),
		                 sin(Id2.y) * sin(Id2.x),
		                 cos(Id2.x)};
	//	normalize(Id23);
	// std::cout << "dot " << dot(Id13, Id23) <<std::endl;
    	// return safe_acos(dot(Id13, Id23)) / gridLen;
	T d2 = ceres::pow(Id13[0] - Id23[0], 2)
	    + ceres::pow(Id13[1] - Id23[1], 2)
	    + ceres::pow(Id13[2] - Id23[2], 2);
	// std::cout << "dist " << d2 << std::endl;
	return d2;
    }

    // double2 lerpCoords(double2 from, double2 to, double alpha) const {
    // 	double2 from_ = from * gridLen;
    // 	double2 to_ = to * gridLen;

    // 	double3 from3 = normalize(make_double3(cos(from_.y) * sin(from_.x),
    // 					       sin(from_.y) * sin(from_.x),
    // 					       cos(from_.x)));
    // 	double3 to3 = normalize(make_double3(cos(to_.y) * sin(to_.x),
    // 					     sin(to_.y) * sin(to_.x),
    // 					     cos(to_.x)));
    // 	double3 k = normalize(cross(from3, to3));
    // 	if (isnan(k.x))
    // 	    return from;
    // 	double span = safe_acos(dot(from3, to3));
    // 	alpha *= span;
    // 	double3 interpolated3 = from3 * cos(alpha) + cross(k, from3) * sin(alpha)
    // 	    + k * dot(k, from3) * (1 - cos(alpha));
    // 	return make_double2(safe_acos(interpolated3.z),
    // 			    atan2(interpolated3.y,
    // 				  interpolated3.x)) / gridLen;
    // }

    template <typename T>
    std::vector<T> lerpCoords(std::vector<T> from, std::vector<T> to, double alpha) const {
    	std::vector<T> from_(from.begin(), from.end());
    	from_[0] *= gridLen; from_[1] *= gridLen;
    	std::vector<T> to_(to.begin(), to.end());
    	to_[0] *= gridLen; to_[1] *= gridLen;
	// std::cout << "from_ " << from_[0] << " "<< from_[1]<<  " to " << to_[0]<< " " << to_[1]<<std::endl;
    	std::vector<T> from3{cos(from_[1]) * sin(from_[0]),
    			     sin(from_[1]) * sin(from_[0]),
    			     cos(from_[0])};
    	std::vector<T> to3{cos(to_[1]) * sin(to_[0]),
    			   sin(to_[1]) * sin(to_[0]),
    	  		   cos(to_[0])};

	std::vector<T> k = normalized(cross(from3, to3));
    	if (ceres::IsNaN(k[0]))
    	    return from;
    	T span = safe_acos(dot(from3, to3));
    	span *= alpha;
	std::vector<T> interpolated = cross(k, from3);
	T c0 = sin(span);
	T c1 = cos(span);
	T c2 = dot(k, from3) * (1.0 - c1);
	interpolated[0] = interpolated[0] * c0 + from3[0] * c1 + k[0] * c2;
	interpolated[1] = interpolated[1] * c0 + from3[1] * c1 + k[1] * c2;
	interpolated[2] = interpolated[2] * c0 + from3[2] * c1 + k[2] * c2;
	//std::cout << "from " << from3[0] << " "<< from3[1]<< " " << from3[2]<< " to " << to3[0]<< " " << to3[1]<< " " << to3[2]<< " inter " << interpolated[0]<< " " << interpolated[1]<< " " << interpolated[2] << std::endl;
	    
	std::vector<T> result{safe_acos(interpolated[2]) / gridLen,
		              atan2(interpolated[1], interpolated[0]) / gridLen};
    	return result;
    }
};

    // double2 lerpCoords(double2 from, double2 to, double alpha)  {
    // 	double gridLen =0.78539816339;
    // 	double2 from_ = from * gridLen;
    // 	double2 to_ = to * gridLen;

    // 	double3 from3 = normalize(make_double3(cos(from_.y) * sin(from_.x),
    // 					       sin(from_.y) * sin(from_.x),
    // 					       cos(from_.x)));
    // 	double3 to3 = normalize(make_double3(cos(to_.y) * sin(to_.x),
    // 					     sin(to_.y) * sin(to_.x),
    // 					     cos(to_.x)));
    // 	double3 k = normalize(cross(from3, to3));
    // 	if (isnan(k.x))
    // 	    return from;
    // 	double span = safe_acos(dot(from3, to3));
    // 	alpha *= span;
    // 	double3 interpolated3 = from3 * cos(alpha) + cross(k, from3) * sin(alpha)
    // 	    + k * dot(k, from3) * (1 - cos(alpha));
    // 	return make_double2(safe_acos(interpolated3.z),
    // 			    atan2(interpolated3.y,
    // 				  interpolated3.x)) / gridLen;
    // }


void KaminoSolver::correctMapping() {
    double x[2] = {0.0, 0.0};
    double y[2] = {0.0, 0.0};
    float theta = 0.5f;
    float phi = 0.f;

    // Build the problem.
    ceres::Problem problem;

    std::vector<float> backward_p_float(N), backward_t_float(N),
	forward_p_float(N), forward_t_float(N);
    std::vector<double> backward_p_double(N), backward_t_double(N),
	forward_p_double(N), forward_t_double(N);

    cudaMemcpy(backward_p_float.data(), backward_p, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(backward_t_float.data(), backward_t, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(forward_p_float.data(), forward_p, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(forward_t_float.data(), forward_t, N * sizeof(float), cudaMemcpyDeviceToHost);

    backward_p_double.assign(backward_p_float.begin(), backward_p_float.end());
    backward_t_double.assign(backward_t_float.begin(), backward_t_float.end());
    forward_p_double.assign(forward_p_float.begin(), forward_p_float.end());
    forward_t_double.assign(forward_t_float.begin(), forward_t_float.end());

    //    for (auto i = forward_t_double.begin(); i != forward_t_double.end(); i++)
    //	std::cout << *i << " ";
    // std::cout << std::endl;
    
    for (size_t t = 0; t < nTheta; t++) {
	for (size_t p = 0; p < nPhi; p++) {
	    double2 Id = make_double2((double)t, (double)p) + centeredOffsetd;
	    size_t index = p + nPhi * t;

	    double2 sampleId = make_double2(backward_t_double[index], backward_p_double[index]) - centeredOffsetd;
	    bool isFlippedPole = validateCoord(sampleId);

	    int phiIndex = static_cast<int>(floor(sampleId.y));
	    int thetaIndex = static_cast<int>(floor(sampleId.x));
	    double alphaPhi = sampleId.y - static_cast<double>(phiIndex);
	    double alphaTheta = sampleId.x - static_cast<double>(thetaIndex);

	    size_t ll, lr, hl, hr;

	    if (isFlippedPole) {
		if (thetaIndex == 0) {
		    size_t phiLower = phiIndex % nPhi;
		    size_t phiHigher = (phiLower + 1) % nPhi;

		    hl = phiLower;
		    hr = phiHigher;

		    phiLower = (phiLower + nTheta) % nPhi;
		    phiHigher = (phiHigher + nTheta) % nPhi;
		    
		    ll = phiLower;
		    lr = phiHigher;
		} else {
		    thetaIndex -= 1;
		}
	    }

	    if (thetaIndex == nTheta - 1) {
		size_t phiLower = phiIndex % nPhi;
		size_t phiHigher = (phiLower + 1) % nPhi;

		ll = phiLower + nPhi * thetaIndex;
		lr = phiHigher + nPhi * thetaIndex;

		phiLower = (phiLower + nTheta) % nPhi;
		phiHigher = (phiHigher + nTheta) % nPhi;

		hl = phiLower + nPhi * thetaIndex;
		hr = phiHigher + nPhi * thetaIndex;
	    } else if (thetaIndex != 0 || (thetaIndex == 0 && !isFlippedPole)) {
		size_t phiLower = phiIndex % nPhi;
		size_t phiHigher = (phiLower + 1) % nPhi;
		size_t thetaLower = thetaIndex;
		size_t thetaHigher = thetaIndex + 1;

		ll = phiLower + nPhi * thetaLower;
		lr = phiHigher + nPhi * thetaLower;
		hl = phiLower + nPhi * thetaHigher;
		hr = phiHigher + nPhi * thetaHigher;
	    }   
	    
	    ceres::CostFunction* cost_function
		= new ceres::AutoDiffCostFunction<CostFunctor, 1, 1, 1, 1, 1, 1, 1, 1, 1>
		(new CostFunctor(this, Id, alphaTheta, alphaPhi));
	    problem.AddResidualBlock(cost_function, NULL,
				     forward_t_double.data() + ll, forward_p_double.data() + ll,
				     forward_t_double.data() + lr, forward_p_double.data() + lr,
				     forward_t_double.data() + hl, forward_p_double.data() + hl,
				     forward_t_double.data() + hr, forward_p_double.data() + hr);
	}
    }

    
    // Run the solver!
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";

    // ceres::Problem problem2;    

    // for (size_t t = 0; t < nTheta; t++) {
    // 	for (size_t p = 0; p < nPhi; p++) {
    // 	    double2 Id = make_double2((double)t, (double)p) + centeredOffsetd;
    // 	    size_t index = p + nPhi * t;

    // 	    double2 sampleId = make_double2(forward_t_double[index], forward_p_double[index]) - centeredOffsetd;
    // 	    bool isFlippedPole = validateCoord(sampleId);

    // 	    int phiIndex = static_cast<int>(floor(sampleId.y));
    // 	    int thetaIndex = static_cast<int>(floor(sampleId.x));
    // 	    double alphaPhi = sampleId.y - static_cast<double>(phiIndex);
    // 	    double alphaTheta = sampleId.x - static_cast<double>(thetaIndex);

    // 	    size_t ll, lr, hl, hr;

    // 	    if (isFlippedPole) {
    // 		if (thetaIndex == 0) {
    // 		    size_t phiLower = phiIndex % nPhi;
    // 		    size_t phiHigher = (phiLower + 1) % nPhi;

    // 		    hl = phiLower;
    // 		    hr = phiHigher;

    // 		    phiLower = (phiLower + nTheta) % nPhi;
    // 		    phiHigher = (phiHigher + nTheta) % nPhi;
		    
    // 		    ll = phiLower;
    // 		    lr = phiHigher;
    // 		} else {
    // 		    thetaIndex -= 1;
    // 		}
    // 	    }

    // 	    if (thetaIndex == nTheta - 1) {
    // 		size_t phiLower = phiIndex % nPhi;
    // 		size_t phiHigher = (phiLower + 1) % nPhi;

    // 		ll = phiLower + nPhi * thetaIndex;
    // 		lr = phiHigher + nPhi * thetaIndex;

    // 		phiLower = (phiLower + nTheta) % nPhi;
    // 		phiHigher = (phiHigher + nTheta) % nPhi;

    // 		hl = phiLower + nPhi * thetaIndex;
    // 		hr = phiHigher + nPhi * thetaIndex;
    // 	    } else if (thetaIndex != 0 || (thetaIndex == 0 && !isFlippedPole)) {
    // 		size_t phiLower = phiIndex % nPhi;
    // 		size_t phiHigher = (phiLower + 1) % nPhi;
    // 		size_t thetaLower = thetaIndex;
    // 		size_t thetaHigher = thetaIndex + 1;

    // 		ll = phiLower + nPhi * thetaLower;
    // 		lr = phiHigher + nPhi * thetaLower;
    // 		hl = phiLower + nPhi * thetaHigher;
    // 		hr = phiHigher + nPhi * thetaHigher;
    // 	    }   
	    
    // 	    ceres::CostFunction* cost_function
    // 		= new ceres::AutoDiffCostFunction<CostFunctor, 1, 1, 1, 1, 1, 1, 1, 1, 1>
    // 		(new CostFunctor(this, Id, alphaTheta, alphaPhi));
    // 	    problem2.AddResidualBlock(cost_function, NULL,
    // 				     backward_t_double.data() + ll, backward_p_double.data() + ll,
    // 				     backward_t_double.data() + lr, backward_p_double.data() + lr,
    // 				     backward_t_double.data() + hl, backward_p_double.data() + hl,
    // 				     backward_t_double.data() + hr, backward_p_double.data() + hr);
    // 	}
    // }


    // ceres::Solver::Summary summary2;
    // Solve(options, &problem2, &summary2);

    // std::cout << summary2.BriefReport() << "\n";


    backward_p_float.assign(backward_p_double.begin(), backward_p_double.end());
    backward_t_float.assign(backward_t_double.begin(), backward_t_double.end());
    forward_p_float.assign(forward_p_double.begin(), forward_p_double.end());
    forward_t_float.assign(forward_t_double.begin(), forward_t_double.end());

    cudaMemcpy(backward_p, backward_p_float.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(backward_t, backward_t_float.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(forward_p, forward_p_float.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(forward_t, forward_t_float.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // for (auto i = forward_t_double.begin(); i != forward_t_double.end(); i++)
    // 	std::cout << *i << " ";
    // std::cout << std::endl;

    
}


// assumes U = 1
Kamino::Kamino(float radius, float H, float U, float c_m, float Gamma_m,
	       float T, float Ds, float rm, size_t nTheta, 
	       float dt, float DT, int frames,
	       std::string outputDir, std::string thicknessImage,
	       size_t particleDensity, int device,
	       std::string AMGconfig, float blendCoeff):
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
    std::cout << "Re^-1 " << re << std::endl;
    std::cout << "S " << S << std::endl;
    std::cout << "Cr " << Cr << std::endl;
    std::cout << "M " << M << std::endl;
    std::cout << "AMG config file: " << AMGconfig << std::endl;
}
Kamino::~Kamino()
{}

void Kamino::run()
{
    KaminoSolver solver(nPhi, nTheta, radius, dt*radius/U, H, device, AMGconfig, particleDensity);
    
    checkCudaErrors(cudaMemcpyToSymbol(nPhiGlobal, &(this->nPhi), sizeof(size_t)));
    checkCudaErrors(cudaMemcpyToSymbol(nThetaGlobal, &(this->nTheta), sizeof(size_t)));
    checkCudaErrors(cudaMemcpyToSymbol(radiusGlobal, &(this->radius), sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(invRadiusGlobal, &(this->invRadius), sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(timeStepGlobal, &(this->dt), sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(gridLenGlobal, &(this->gridLen), sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(invGridLenGlobal, &(this->invGridLen), sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(SGlobal, &(this->S), sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(MGlobal, &(this->M), sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(reGlobal, &(this->re), sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(gGlobal, &(this->gs), sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(DsGlobal, &(this->Ds), sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(CrGlobal, &(this->Cr), sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(UGlobal, &(this->U), sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(blend_coeff, &(this->blendCoeff), sizeof(float)));

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

    float T = 0.0;              // simulation time
    int i = 1;
    float dt_ = dt * this->radius / this->U;
    for (; i < frames; i++) {
	checkCudaErrors(cudaMemcpyToSymbol(currentTimeGlobal, &T, sizeof(float)));
	std::cout << "current time " << T << std::endl;

	//     solver.adjustStepSize(dt, U, epsilon);

	checkCudaErrors(cudaMemcpyToSymbol(timeStepGlobal, &dt, sizeof(float)));
	std::cout << "current time step size is " << dt_ << " s" << std::endl;
	std::cout << "steps needed until next frame " << DT/dt_ << std::endl;
    
	while ((T + dt_) <= i*DT && !solver.isBroken()) {
	    solver.stepForward();
	    T += dt_;
	}
	if (T < i*DT && !solver.isBroken()) {
	    float tmp_dt = (i * DT - T) * this->U / this->radius;
	    checkCudaErrors(cudaMemcpyToSymbol(timeStepGlobal, &tmp_dt, sizeof(float)));
	    solver.stepForward(i * DT - T);
	}
	if (solver.isBroken()) {
	    std::cerr << "Film is broken." << std::endl;
	    break;
	}
	T = i*DT;

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
    float gpu_time = timer.stopTimer();
# endif

    std::cout << "Time spent: " << gpu_time << "ms" << std::endl;
    std::cout << "Performance: " << 1000.0 * i / gpu_time << " frames per second" << std::endl;
}
