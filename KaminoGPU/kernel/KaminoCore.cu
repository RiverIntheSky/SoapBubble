# include "KaminoSolver.cuh"
# include "KaminoGPU.cuh"
# include "KaminoTimer.cuh"
#include <boost/filesystem.hpp>

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
    float3 Id13 = normalize(make_float3(cos(Id1.y) * sin(Id1.x),
					sin(Id1.y) * sin(Id1.x),
					cos(Id1.x)));
    float3 Id23 = normalize(make_float3(cos(Id2.y) * sin(Id2.x),
					sin(Id2.y) * sin(Id2.x),
					cos(Id2.x)));
    return invGridLenGlobal * acosf(clamp(dot(Id13, Id23), -1.0, 1.0));
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
    Id.y = fmod(Id.y + nPhiGlobal, (float)nPhiGlobal);
    return ret;
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
__device__ float sampleVPhi(float* input, float2& rawId, size_t pitch) {
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
__device__ float sampleVTheta(float* input, float2& rawId, size_t pitch) {
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
__device__ float sampleCentered(float* input, float2& rawId, size_t pitch) {
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
inline __device__ float2 traceRK2(float* velTheta, float* velPhi, float& dt,
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
inline __device__ float2 traceRK3(float* velTheta, float* velPhi, float& dt,
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

    float3 from3 = normalize(make_float3(cos(from_.y) * sin(from_.x),
					 sin(from_.y) * sin(from_.x),
					 cos(from_.x)));
    float3 to3 = normalize(make_float3(cos(to_.y) * sin(to_.x),
				       sin(to_.y) * sin(to_.x),
				       cos(to_.x)));
    float3 k = normalize(cross(from3, to3));
    if (isnan(k.x))
	return from;
    float span = acosf(clamp(dot(from3, to3), -1.0, 1.0));
    alpha *= span;
    float3 interpolated3 = from3 * cosf(alpha) + cross(k, from3) * sinf(alpha)
	+ k * dot(k, from3) * (1 - cosf(alpha));
    return invGridLenGlobal * make_float2(acosf(clamp(interpolated3.z, -1.0, 1.0)),
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

    float2 midCoord = make_float2(acosf(midx.z), atan2f(midx.y, midx.x));
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

    float2 pCoord = make_float2(acosf(px.z), atan2f(px.y, px.x));
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

    float2 midCoord = make_float2(acosf(midx.z), atan2f(midx.y, midx.x));
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

    float2 pCoord = make_float2(acosf(px.z), atan2f(px.y, px.x));
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
(fReal* attributeOutput, fReal* velPhi, fReal* velTheta, size_t pitch)
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
    
    float advectedAttribute = sampleVPhi(velPhi, traceId, pitch);

    attributeOutput[thetaId * pitch + phiId] = advectedAttribute;
};


/**
 * advect vectors on cartesian grid 
 * or test advection of vectors on sphere
 */
__global__ void advectionVThetaKernel
(fReal* attributeOutput, fReal* velPhi, fReal* velTheta, size_t pitch)
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

    float advectedAttribute = sampleVTheta(velTheta, traceId, pitch);

    attributeOutput[thetaId * pitch + phiId] = advectedAttribute;
}


__global__ void advectionCentered
(fReal* attributeOutput, fReal* attributeInput, fReal* velPhi, fReal* velTheta, size_t pitch)
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
    
    float advectedAttribute = sampleCentered(attributeInput, traceId, pitch);

    attributeOutput[thetaId * pitch + phiId] = advectedAttribute;
};


__global__ void advectionCenteredBimocq
(float* thicknessOutput, float* thicknessInit, float* thicknessDelta,
 float* bwd_t, float* bwd_p, size_t pitch) {
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

    float thickness = 0.f;
    float gamma = 0.f;
    for (int i = 0; i < 5; i++) {
	float2 posId = gId + dir[i];
	float2 initPosId = sampleMapping(bwd_t, bwd_p, posId);
	thickness += w[i] * (sampleCentered(thicknessInit, initPosId, pitch) +
			     sampleCentered(thicknessDelta, initPosId, pitch));
	// gamma += w[i] * (sampleCentered(gammaInit, initPosId, pitch) +
	// 		 sampleCentered(gammaDelta, initPosId, pitch));
    }
    thicknessOutput[thetaId * pitch + phiId] = thickness;
    //gammaOutput[thetaId * pitch + phiId] = gamma;
}


__global__ void correctBimocq1(float* thicknessCurr, float* thicknessError, float* thicknessDelta,
			       float* thicknessInit, float* fwd_t, float* fwd_p, size_t pitch) {
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

    // Coord in scalar space
    float2 gId = make_float2((float)thetaId, (float)phiId) + centeredOffset;
    for (int i = 0; i < 5; i++) {
	float2 posId = gId + dir[i];
	float2 initPosId = sampleMapping(fwd_t, fwd_p, posId);
	thickness += w[i] * sampleCentered(thicknessCurr, initPosId, pitch);
    }
    at(thicknessError, thetaId, phiId) = (thickness - at(thicknessDelta, thetaId, phiId, pitch)
					- at(thicknessInit, thetaId, phiId, pitch)) * 0.5f;
}


__global__ void correctBimocq2(float* thicknessOut, float* thicknessError,
			       float* bwd_t, float* bwd_p, size_t pitch) {
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

    // if (thetaId < nThetaGlobal/8 || thetaId > nThetaGlobal*7/8)
    // 	return;

    // Coord in scalar space
    float2 gId = make_float2((float)thetaId, (float)phiId) + centeredOffset;
    for (int i = 0; i < 5; i++) {
	float2 posId = gId + dir[i];
	float2 samplePosId = sampleMapping(bwd_t, bwd_p, posId);
	at(thicknessOut, thetaId, phiId, pitch) -= w[i] * sampleCentered(thicknessError, samplePosId, nPhiGlobal);
    }
}


// __global__ void advectionAllCentered
// (fReal* thicknessOutput, fReal* surfOutput, fReal* thicknessInput, fReal* surfInput, fReal* velPhi, fReal* velTheta, size_t nPitchInElements)
// {
//     // Index
//     int splitVal = nPhiGlobal / blockDim.x;
//     int threadSequence = blockIdx.x % splitVal;
//     int phiId = threadIdx.x + threadSequence * blockDim.x;
//     int thetaId = blockIdx.x / splitVal;
//     // Coord in phi-theta space
//     fReal gPhiId = (fReal)phiId + centeredPhiOffset;
//     fReal gThetaId = (fReal)thetaId + centeredThetaOffset;
//     fReal gTheta = gThetaId * gridLenGlobal;
    
//     // Sample the speed
//     fReal guTheta = sampleVTheta(velTheta, gPhiId, gThetaId, nPitchInElements);
//     fReal guPhi = 0.5 * (velPhi[thetaId * nPitchInElements + phiId] +
//     			 velPhi[thetaId * nPitchInElements + (phiId + 1) % nPhiGlobal]);

//     fReal cofTheta = timeStepGlobal * invGridLenGlobal;
// # ifdef sphere
//     fReal cofPhi = cofTheta / sinf(gTheta);
// # else
//     fReal cofPhi = cofTheta;
// # endif

//     fReal deltaPhi = guPhi * cofPhi;
//     fReal deltaTheta = guTheta * cofTheta;

// # ifdef RUNGE_KUTTA
//     // Traced halfway in phi-theta space
//     fReal midPhiId = gPhiId - 0.5 * deltaPhi;
//     fReal midThetaId = gThetaId - 0.5 * deltaTheta;
    
//     fReal muPhi = sampleVPhi(velPhi, midPhiId, midThetaId, nPitchInElements);
//     fReal muTheta = sampleVTheta(velTheta, midPhiId, midThetaId, nPitchInElements);

//     deltaPhi = muPhi * cofPhi;
//     deltaTheta = muTheta * cofTheta;
// # endif

//     fReal pPhiId = gPhiId - deltaPhi;
//     fReal pThetaId = gThetaId - deltaTheta;

//     fReal advectedThickness = sampleCentered(thicknessInput, pPhiId, pThetaId, nPitchInElements);
//     fReal advectedSurf = sampleCentered(surfInput, pPhiId, pThetaId, nPitchInElements);
    
//     thicknessOutput[thetaId * nPitchInElements + phiId] = advectedThickness;
//     surfOutput[thetaId * nPitchInElements + phiId] = advectedSurf;
// };


// __global__ void advectionParticles(fReal* output, fReal* velPhi, fReal* velTheta, fReal* input, size_t pitch, size_t numOfParticles)
// {
//     int particleId = blockIdx.x * blockDim.x + threadIdx.x;

//     if (particleId < numOfParticles) {
// 	fReal phiId = input[2 * particleId];
// 	fReal thetaId = input[2 * particleId + 1];
// 	fReal theta = thetaId * gridLenGlobal;
// 	fReal sinTheta = max(sinf(theta), eps);

// 	fReal uPhi = sampleVPhi(velPhi, phiId, thetaId, pitch);
// 	fReal uTheta = sampleVTheta(velTheta, phiId, thetaId, pitch);

// 	fReal cofTheta = timeStepGlobal * invGridLenGlobal;
// # ifdef sphere
// 	fReal cofPhi = cofTheta / sinTheta;
// # else
// 	fReal cofPhi = cofTheta;
// # endif
	
// 	fReal deltaPhi = uPhi * cofPhi;
// 	fReal deltaTheta = uTheta * cofTheta;

// # ifdef RUNGE_KUTTA
// 	// Traced halfway in phi-theta space
// 	fReal midPhiId = phiId + 0.5 * deltaPhi;
// 	fReal midThetaId = thetaId + 0.5 * deltaTheta;

// 	fReal muPhi = sampleVPhi(velPhi, midPhiId, midThetaId, pitch);
// 	fReal muTheta = sampleVTheta(velTheta, midPhiId, midThetaId, pitch);

// 	deltaPhi = muPhi * cofPhi;
// 	deltaTheta = muTheta * cofTheta;

// 	theta = midThetaId * gridLenGlobal;
// 	sinTheta = max(sinf(theta), eps);
// # endif

// 	fReal updatedThetaId = thetaId + deltaTheta;
// 	fReal updatedPhiId = phiId + deltaPhi;

// 	validateCoord(updatedPhiId, updatedThetaId, nPhiGlobal);
	
// 	output[2 * particleId] = updatedPhiId;
// 	output[2 * particleId + 1] = updatedThetaId;
//    }
// }


// __global__ void mapParticlesToThickness
// (fReal* particleCoord, fReal* particleVal, float2* weight, size_t numParticles)
// {
//     // Index
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     int particleId = index >> 1; // (index / 2)
//     int partition = index & 1;	 // (index % 2)

//     if (particleId < numParticles) {
// 	fReal gPhiId = particleCoord[2 * particleId];
// 	fReal gThetaId = particleCoord[2 * particleId + 1];

// 	fReal gTheta = gThetaId * gridLenGlobal;

// 	fReal sinTheta = max(sinf(gTheta), eps);

// 	fReal gPhi = gPhiId * gridLenGlobal;

// 	size_t thetaId = static_cast<size_t>(floorf(gThetaId));

// 	fReal x1 = cosf(gPhi) * sinTheta; fReal y1 = sinf(gPhi) * sinTheta; fReal z1 = cosf(gTheta);

// 	fReal theta = (thetaId + 0.5) * gridLenGlobal;

// 	fReal phiRange = .5f/sinTheta;
// 	int minPhiId = static_cast<int>(ceilf(gPhiId - phiRange));
// 	int maxPhiId = static_cast<int>(floorf(gPhiId + phiRange));

// 	fReal z2 = cosf(theta);
// 	fReal r = sinf(theta);
// 	fReal value = particleVal[particleId];

// 	int begin; int end;
	
// 	if (partition == 0) {
// 	    begin = minPhiId; end = static_cast<int>(gPhiId);
// 	} else {
// 	    begin = static_cast<int>(gPhiId); end = maxPhiId + 1;
// 	}
	    
// 	for (int phiId = begin; phiId < end; phiId++) {
// 	    fReal phi = phiId * gridLenGlobal;
// 	    fReal x2 = cosf(phi) * r; fReal y2 = sinf(phi) * r;

// 	    fReal dist2 = powf(fabsf(x1 - x2), 2.f) + powf(fabsf(y1 - y2), 2.f) + powf(fabsf(z1 - z2), 2.f);
	        
// 	    if (dist2 <= .25f) {
// 		fReal w = expf(-10*dist2);
// 		size_t normalizedPhiId = (phiId + nPhiGlobal) % nPhiGlobal;
// 		float2* currentWeight = weight + (thetaId * nPhiGlobal + normalizedPhiId);
// 		atomicAdd(&(currentWeight->x), w);
// 		atomicAdd(&(currentWeight->y), w * value);
// 	    }
// 	}
//     }
// }


// // advection
// __global__ void normalizeThickness_a
// (fReal* thicknessOutput, fReal* thicknessInput, fReal* velPhi, fReal* velTheta,
//  fReal* div, float2* weight, size_t pitch) {
//     // Index
//     int splitVal = nPhiGlobal / blockDim.x;
//     int threadSequence = blockIdx.x % splitVal;
//     int phiId = threadIdx.x + threadSequence * blockDim.x;
//     int thetaId = blockIdx.x / splitVal;

//     float2* currentWeight = weight + (thetaId * nPhiGlobal + phiId);
//     fReal w = currentWeight->x;
//     fReal val = currentWeight->y;

//     if (w > 0) {
// 	thicknessOutput[thetaId * pitch + phiId] = val / w;
//     } else {
// 	//	printf("Warning: no particles contributed to grid thetaId %d, phiId %d\n", thetaId, phiId);
// 	fReal gPhiId = (fReal)phiId + centeredPhiOffset;
// 	fReal gThetaId = (fReal)thetaId + centeredThetaOffset;
// 	fReal gTheta = gThetaId * gridLenGlobal;

// 	// Sample the speed
// 	fReal guPhi = sampleVPhi(velPhi, gPhiId, gThetaId, pitch);
// 	fReal guTheta = sampleVTheta(velTheta, gPhiId, gThetaId, pitch);

// 	fReal cofTheta = timeStepGlobal * invGridLenGlobal;
// # ifdef sphere
// 	fReal cofPhi = cofTheta / sinf(gTheta);
// # else
// 	fReal cofPhi = cofTheta;
// # endif

// 	fReal deltaPhi = guPhi * cofPhi;
// 	fReal deltaTheta = guTheta * cofTheta;
	
// # ifdef RUNGE_KUTTA
// 	// Traced halfway in phi-theta space
// 	fReal midPhiId = gPhiId - 0.5 * deltaPhi;
// 	fReal midThetaId = gThetaId - 0.5 * deltaTheta;
    
// 	fReal muPhi = sampleVPhi(velPhi, midPhiId, midThetaId, pitch);
// 	fReal muTheta = sampleVTheta(velTheta, midPhiId, midThetaId, pitch);

// 	deltaPhi = muPhi * cofPhi;
// 	deltaTheta = muTheta * cofTheta;
// # endif

// 	fReal pPhiId = gPhiId - deltaPhi;
// 	fReal pThetaId = gThetaId - deltaTheta;

//         fReal advectedVal = sampleCentered(thicknessInput, pPhiId, pThetaId, pitch);
// 	//	fReal f = div[thetaId * nPhiGlobal + phiId];
// 	thicknessOutput[thetaId * pitch + phiId] = advectedVal;// / (1 + timeStepGlobal * f);
//     }
// }


// // body force
// __global__ void normalizeThickness_f
// (fReal* thicknessOutput, fReal* thicknessInput,
//  fReal* div, float2* weight, size_t pitch) {
//     // Index
//     int splitVal = nPhiGlobal / blockDim.x;
//     int threadSequence = blockIdx.x % splitVal;
//     int phiId = threadIdx.x + threadSequence * blockDim.x;
//     int thetaId = blockIdx.x / splitVal;

//     float2* currentWeight = weight + (thetaId * nPhiGlobal + phiId);
//     fReal w = currentWeight->x;
//     fReal val = currentWeight->y;

//     if (w > 0) {
// 	thicknessOutput[thetaId * pitch + phiId] = val / w;
//     } else {
// 	fReal f = div[thetaId * nPhiGlobal + phiId];
// 	thicknessOutput[thetaId * pitch + phiId] =  thicknessInput[thetaId * pitch + phiId] * (1 - timeStepGlobal * f);
//     }
// }


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

    return max(maxAbs(tmp_t, nTheta, nPhi), maxAbs(tmp_p, nTheta, nPhi));
}


void KaminoSolver::advection()
{
    // Advect Phi
    dim3 gridLayout;
    dim3 blockLayout;
    determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());

# ifdef sphere
    advectionVSpherePhiKernel<<<gridLayout, blockLayout>>>
	(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), velPhi->getNextStepPitchInElements());
# else
    advectionVPhiKernel<<<gridLayout, blockLayout>>>
    	(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), velPhi->getNextStepPitchInElements());
# endif
    checkCudaErrors(cudaGetLastError());
    // checkCudaErrors(cudaDeviceSynchronize());

    // Advect Theta
    determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
# ifdef sphere
    advectionVSphereThetaKernel<<<gridLayout, blockLayout>>>
 	(velTheta->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), velTheta->getNextStepPitchInElements());
# else
    advectionVThetaKernel<<<gridLayout, blockLayout>>>
	(velTheta->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), velTheta->getNextStepPitchInElements());
# endif
    checkCudaErrors(cudaGetLastError());
    // checkCudaErrors(cudaDeviceSynchronize());
   

    // Advect concentration
    bool useBimocq = true;
    determineLayout(gridLayout, blockLayout, surfConcentration->getNTheta(), surfConcentration->getNPhi());
    if (useBimocq) {
	advectionCenteredBimocq<<<gridLayout, blockLayout>>>
	    (thickness->getGPUNextStep(), thickness->getGPUInit(), thickness->getGPUDelta(),
	     backward_t, backward_p, pitch);
    } else {
	advectionCentered<<<gridLayout, blockLayout>>>
	    (thickness->getGPUNextStep(), thickness->getGPUThisStep(),
	     velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), thickness->getNextStepPitchInElements());
    }
    checkCudaErrors(cudaGetLastError());

    advectionCentered<<<gridLayout, blockLayout>>>
	(surfConcentration->getGPUNextStep(), surfConcentration->getGPUThisStep(),
	 velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), surfConcentration->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    if (useBimocq) {
	correctBimocq1<<<gridLayout, blockLayout>>>
	    (thickness->getGPUNextStep(), tmp_t, thickness->getGPUDelta(), thickness->getGPUInit(),
	     forward_t, forward_p, pitch);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	correctBimocq2<<<gridLayout, blockLayout>>>
	    (thickness->getGPUNextStep(), tmp_t, backward_t, backward_p, pitch);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
    }

    //    checkCudaErrors(cudaDeviceSynchronize());
 
    // Advect thickness particles
    // if (particles->numOfParticles > 0) {
    // 	determineLayout(gridLayout, blockLayout, 1, particles->numOfParticles);
    // 	advectionParticles<<<gridLayout, blockLayout>>>
    // 	    (particles->coordGPUNextStep, velPhi->getGPUThisStep(), velTheta->getGPUThisStep(),
    // 	     particles->coordGPUThisStep, velPhi->getThisStepPitchInElements(), particles->numOfParticles);
    // 	checkCudaErrors(cudaGetLastError());
    // 	// checkCudaErrors(cudaDeviceSynchronize());

    // 	// reset weight
    // 	determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    // 	resetThickness<<<gridLayout, blockLayout>>>(weight);
    // 	checkCudaErrors(cudaGetLastError());
    // 	// checkCudaErrors(cudaDeviceSynchronize());

    // 	determineLayout(gridLayout, blockLayout, 2, particles->numOfParticles);
    // 	mapParticlesToThickness<<<gridLayout, blockLayout>>>
    // 	    (particles->coordGPUThisStep, particles->value,  weight, particles->numOfParticles);
    // 	checkCudaErrors(cudaGetLastError());
    // 	checkCudaErrors(cudaDeviceSynchronize());
    // }

    // average particle information
    // If numOfParticles == 0, choose semi-Lagrangian advection
    
    // determineLayout(gridLayout, blockLayout, thickness->getNTheta(), thickness->getNPhi());

    thickness->swapGPUBuffer();
    //    particles->swapGPUBuffers(); 
    surfConcentration->swapGPUBuffer();
    swapVelocityBuffers();
}


// div(u) at cell center
__global__ void divergenceKernel
(fReal* div, fReal* velPhi, fReal* velTheta,
 size_t velPhiPitchInElements, size_t velThetaPitchInElements)
{
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

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
__global__ void divergenceKernel_fromGamma(float* div, float* gammaNext, float* gammaThis,
					   size_t pitch) {
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    float gamma_a = gammaThis[thetaId * pitch + phiId];
    fReal gamma = gammaNext[thetaId * pitch + phiId];

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

    fReal gTheta = ((float)thetaId + centeredThetaOffset) * gridLenGlobal;
    fReal gPhi = ((float)phiId + centeredPhiOffset) * gridLenGlobal;
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
    fReal uEast = 0.0;
    fReal uWest = 0.0;
    fReal vNorth = 0.0;
    fReal vSouth = 0.0;

    uWest = at(velPhi_a, thetaId, phiId, pitch);
    uEast = at(velPhi_a, thetaId, phiIdEast, pitch);

    if (thetaId != 0) {
	size_t thetaNorthIdx = thetaId - 1;
	vNorth = at(velTheta_a, thetaNorthIdx, phiId, pitch);
    	etaNorth = at(eta_a, thetaNorthIdx, phiId, pitch);
    } else {
    	int oppositePhiId = (phiId + nThetaGlobal) % nPhiGlobal;
    	etaNorth = eta_a[oppositePhiId];
    }
    if (thetaId != nThetaGlobal - 1) {
	vSouth = at(velTheta_a, thetaId, phiId, pitch);
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
    float div = invGridLenGlobal *
	(uEast / (1.f + CrGlobal * timeStepGlobal / etaRight) -
	 uWest / (1.f + CrGlobal * timeStepGlobal / etaLeft) +
	 cscTheta * (vSouth * sinThetaSouth / (1.f + CrGlobal * timeStepGlobal / etaDown) -
		     vNorth * sinThetaNorth / (1.f + CrGlobal * timeStepGlobal / etaUp)));

    // # ifdef uair
    //     diva += 20.f * (1 - smoothstep(0.f, 10.f, currentTimeGlobal)) * (M_hPI - gTheta)
    // 	* expf(-10 * powf(fabsf(gTheta - M_hPI), 2.f)) * radiusGlobal
    // 	* sinf(gPhi) * cscTheta / UGlobal;
    // # endif
    // # ifdef vair
    //     diva += (gTheta < M_hPI) * 4 * (1 - smoothstep(0.f, 10.f, currentTimeGlobal)) * cosTheta
    // 	* cosf(2 * gPhi) * radiusGlobal / UGlobal;
    // # endif
    
# ifdef gravity
    // div += gGlobal * cscTheta * invGridLenGlobal *
    // 	(sinThetaSouth * sinThetaSouth / (invDt + CrGlobal / etaDown) -
    // 	 sinThetaNorth * sinThetaNorth / (invDt + CrGlobal / etaUp));
    div += 0.57735026919 * gGlobal * cscTheta * invGridLenGlobal *
    	((cosf(gPhi + halfStep) - sinf(gPhi + halfStep)) / (invDt + CrGlobal / etaRight) -
    	 (cosf(gPhi - halfStep) - sinf(gPhi - halfStep)) / (invDt + CrGlobal / etaLeft) +
    	 (cosf(gTheta + halfStep) * (cosf(gPhi) + sinf(gPhi)) + sinThetaSouth)
    	 / (invDt + CrGlobal / etaDown) * sinThetaSouth -
    	 (cosf(gTheta - halfStep) * (cosf(gPhi) + sinf(gPhi)) + sinThetaNorth)
    	 / (invDt + CrGlobal / etaUp) * sinThetaNorth);
# endif

    rhs[idx] = sinTheta * (invDt - div);
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

    //    printGPUarraytoMATLAB<float>("test/val.txt", val, N, 5, 5);
    //    printGPUarraytoMATLAB<float>("test/rhs.txt", d_r, N, 1, 1);
    // printGPUarraytoMATLAB<float>("test/x.txt", d_x, N, 1, 1);

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

__global__ void applyforcevelthetaKernel(fReal* velThetaOutput, fReal* velThetaInput, fReal* thickness, fReal* concentration, size_t pitch) {
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

# ifdef sphere
    fReal gTheta = ((fReal)thetaId + vThetaThetaOffset) * gridLenGlobal;
    fReal gPhi = ((fReal)phiId + vThetaPhiOffset) * gridLenGlobal;
# endif

    int thetaSouthId = thetaId + 1;

    float v1 = velThetaInput[thetaId * pitch + phiId];

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

    velThetaOutput[thetaId * pitch + phiId] = (v1 / timeStepGlobal + f1 + f2 + f3) / (1./timeStepGlobal + CrGlobal * invDelta);
}


__global__ void applyforcevelphiKernel
(fReal* velPhiOutput, fReal* velPhiInput, fReal* thickness,
 fReal* concentration, size_t pitch) {
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

# ifdef sphere
    // Coord in phi-theta space
    fReal gPhi = ((fReal)phiId + vPhiPhiOffset) * gridLenGlobal;
    fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLenGlobal;
    float sinTheta = sinf(gTheta);
# else
    sinTheta = 1.f; // no effect
# endif

    int phiWestId = (phiId - 1 + nPhiGlobal) % nPhiGlobal;

    // values at centered grid
    fReal DeltaWest = thickness[thetaId * pitch + phiWestId];
    fReal DeltaEast = thickness[thetaId * pitch + phiId];
    fReal GammaWest = concentration[thetaId * pitch + phiWestId];
    fReal GammaEast = concentration[thetaId * pitch + phiId];
    
    fReal u1 = velPhiInput[thetaId * pitch + phiId] * sinTheta;
        
    // value at uPhi grid
    fReal invDelta = 2. / (DeltaWest + DeltaEast);
    
    // pGpx = frac{1}{\sin\theta}\frac{\partial\Gamma}{\partial\phi};
    fReal pGpx = invGridLenGlobal * (GammaEast - GammaWest) / sinTheta;
    
    // elasticity
    float f1 = -MGlobal * invDelta * pGpx;
    // air friction
    float uAir = 0.f;
# if defined uair && defined sphere
    uAir = 20.f * (1 - smoothstep(0.f, 10.f, currentTimeGlobal)) * (M_hPI - gTheta)
	* expf(-10 * powf(fabsf(gTheta - M_hPI), 2.f)) * radiusGlobal
	* cosf(gPhi) / UGlobal;
# endif
    float f2 = CrGlobal * invDelta * uAir;

    float f3 = 0.57735026919 * (cosf(gPhi) - sinf(gPhi)) * gGlobal;
        
    velPhiOutput[thetaId * pitch + phiId] = (u1 / timeStepGlobal + f1 + f2 + f3) / (1./timeStepGlobal + CrGlobal * invDelta) / sinTheta;   
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
//     // 	result = 0.f;
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
//     	uAir = 20.f * (M_hPI - gTheta) * expf(-10 * powf(fabsf(gTheta - M_hPI), 2.f)) * radiusGlobal * cosf(gPhi) / UGlobal;
// # endif

//     fReal f6 = CrGlobal * invDelta * (uAir - u1);
    
//     // output
//     fReal result = (u1 + timeStepGlobal * (f2 + f3 + f4 + f5 + CrGlobal * uAir))
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

// __global__ void applyforceParticles
// (fReal* tempVal, fReal* value, fReal* coord, fReal* div, size_t numOfParticles) {
//     int particleId = blockIdx.x * blockDim.x + threadIdx.x;

//     if (particleId < numOfParticles) {
// 	fReal phiId = coord[2 * particleId];
// 	fReal thetaId = coord[2 * particleId + 1];

// 	fReal f = sampleCentered(div, phiId, thetaId, nPhiGlobal);

// 	tempVal[particleId] = value[particleId] * (1 - timeStepGlobal * f);
//     }
// }


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

//     fReal cscTheta = 1.f / sinf(thetaCoord);
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
//     fReal f2 = 0.f;

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

    a[phiId + nPhiGlobal * thetaId] = b[phiId + pitch * thetaId] -
	c[phiId + pitch * thetaId];
}


__global__ void accumulateChangesKernel(float* thicknessDelta, float* thicknessDeltaTemp,
					float* gammaDelta, float* gammaDeltaTemp,
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
	
	// gammaDelta[phiId + pitch * thetaId] += w[i] *
	//     sampleCentered(gammaDeltaTemp, initPosId, nPhiGlobal);
	thicknessDelta[phiId + pitch * thetaId] += w[i] *
	    sampleCentered(thicknessDeltaTemp, initPosId, nPhiGlobal);
    }
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

    determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
    applyforcevelthetaKernel<<<gridLayout, blockLayout>>>
    	(velTheta->getGPUNextStep(), velTheta->getGPUThisStep(), thickness->getGPUThisStep(), surfConcentration->getGPUNextStep(), velTheta->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());

    determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());
    applyforcevelphiKernel<<<gridLayout, blockLayout>>>
    	(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), thickness->getGPUThisStep(), surfConcentration->getGPUNextStep(), velPhi->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // div(u^{n+1})
    // determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    // divergenceKernel<<<gridLayout, blockLayout>>>
    // 	(div, velPhi->getGPUNextStep(), velTheta->getGPUNextStep(),
    // 	 velPhi->getNextStepPitchInElements(), velTheta->getNextStepPitchInElements());
    // checkCudaErrors(cudaGetLastError());
    // checkCudaErrors(cudaDeviceSynchronize());

    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    divergenceKernel_fromGamma<<<gridLayout, blockLayout>>>
    	(div, surfConcentration->getGPUNextStep(), surfConcentration->getGPUThisStep(),
    	 surfConcentration->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // printGPUarraytoMATLAB<float>("test/div_g.txt", div, nTheta, nPhi, nPhi);

    // if (particles->numOfParticles > 0) {
    // 	determineLayout(gridLayout, blockLayout, 1, particles->numOfParticles);
    // 	applyforceParticles<<<gridLayout, blockLayout>>>
    // 	    (particles->tempVal, particles->value, particles->coordGPUThisStep, div, particles->numOfParticles);
    // 	checkCudaErrors(cudaGetLastError());
    // 	checkCudaErrors(cudaDeviceSynchronize());

    // 	determineLayout(gridLayout, blockLayout, 2, particles->numOfParticles);
    // 	mapParticlesToThickness<<<gridLayout, blockLayout>>>
    // 	    (particles->coordGPUThisStep, particles->tempVal, weight, particles->numOfParticles);
    // 	checkCudaErrors(cudaGetLastError());
    // 	checkCudaErrors(cudaDeviceSynchronize());
    // }

    // write thickness difference to tmp_t, gamma difference to tmp_p
    determineLayout(gridLayout, blockLayout, thickness->getNTheta(), thickness->getNPhi());
    applyforceThickness<<<gridLayout, blockLayout>>>
    	(thickness->getGPUNextStep(), thickness->getGPUThisStep(),
	 tmp_t, div, thickness->getNextStepPitchInElements());
    
    substractPitched<<<gridLayout, blockLayout>>>
	(tmp_p, surfConcentration->getGPUNextStep(),
	 surfConcentration->getGPUThisStep(), pitch);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    accumulateChangesKernel<<<gridLayout, blockLayout>>>
	(thickness->getGPUDelta(), tmp_t, surfConcentration->getGPUDelta(), tmp_p,
	 forward_t, forward_p, pitch);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    printGPUarraytoMATLAB<float>("dgamma.txt", surfConcentration->getGPUDelta(),
				 nTheta, nPhi, pitch);
    printGPUarraytoMATLAB<float>("gamma.txt", surfConcentration->getGPUNextStep(),
				 nTheta, nPhi, pitch);

    printGPUarraytoMATLAB<float>("deta.txt", thickness->getGPUDelta(),
				 nTheta, nPhi, pitch);
     
    //    std::swap(particles->tempVal, particles->value);
    thickness->swapGPUBuffer();
    surfConcentration->swapGPUBuffer();
    swapVelocityBuffers();
}


// assumes U = 1
Kamino::Kamino(fReal radius, fReal H, fReal U, fReal c_m, fReal Gamma_m,
	       fReal T, fReal Ds, fReal rm, size_t nTheta, 
	       float dt, float DT, int frames,
	       std::string outputDir, std::string thicknessImage,
	       size_t particleDensity, int device,
	       std::string AMGconfig):
    radius(radius), invRadius(1/radius), H(H), U(1.0), c_m(c_m), Gamma_m(Gamma_m), T(T),
    Ds(Ds/(U*radius)), gs(g*radius/(U*U)), rm(rm), epsilon(H/radius), sigma_r(R*T), M(Gamma_m*sigma_r/(3*rho*H*U*U)),
    S(sigma_a*epsilon/(2*mu*U)), re(mu/(U*radius*rho)), Cr(rhoa*sqrt(nua*radius/U)/(rho*H)),
    nTheta(nTheta), nPhi(2 * nTheta),
    gridLen(M_PI / nTheta), invGridLen(nTheta / M_PI), 
    dt(dt*U/radius), DT(DT), frames(frames), outputDir(outputDir),
    thicknessImage(thicknessImage), particleDensity(particleDensity), device(device),
    AMGconfig(AMGconfig) 
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
    KaminoSolver solver(nPhi, nTheta, radius, dt*radius/U, H, device, AMGconfig);
    
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

    solver.initThicknessfromPic(thicknessImage, this->particleDensity);

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
	    float tmp_dt = (i * DT - T) * this->U / this->radius;
	    checkCudaErrors(cudaMemcpyToSymbol(timeStepGlobal, &tmp_dt, sizeof(fReal)));
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
