# include "solver.cuh"
# include "bubble.cuh"
# include "timer.cuh"
# include "utils.h"

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
static __constant__ fReal gammaMaxGlobal;

# define thresh 0.0
# define eps 1e-7
# define MAX_BLOCK_SIZE 8192 /* TODO: deal with larger resolution */

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
 */
__global__ void maxValKernel(fReal* maxVal, fReal* array) {
    __shared__ float maxValTile[MAX_BLOCK_SIZE];
	
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    maxValTile[tid] = fabsf(float(array[i]));
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
	maxVal[blockIdx.x] = fReal(maxValTile[0]);
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
__device__ fReal bubbleLerp(fReal from, fReal to, fReal alpha)
{
    if (isnan(from))
	return to;
    if (isnan(to))
	return from;
    return (1.0 - alpha) * from + alpha * to;
}


/**
 * bilinear interpolation
 */
__device__ fReal bilerp(fReal ll, fReal lr, fReal hl, fReal hr,
			fReal alphaPhi, fReal alphaTheta)
{
    return bubbleLerp(bubbleLerp(ll, lr, alphaPhi),
		      bubbleLerp(hl, hr, alphaPhi), alphaTheta);
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
	fReal higherBelt = -bubbleLerp(input[phiLower + pitch * thetaIndex],
				       input[phiHigher + pitch * thetaIndex], alphaPhi);

	phiLower = (phiIndex + nPhiGlobal / 2) % nPhiGlobal;
	phiHigher = (phiLower + 1) % nPhiGlobal;

	fReal lowerBelt = bubbleLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);
  
	fReal lerped = bubbleLerp(lowerBelt, higherBelt, alphaTheta);
	return lerped;
    }
    
    if (isFlippedPole) {
	thetaIndex -= 1;
    }
    
    if (thetaIndex == nThetaGlobal - 1) {
	size_t phiLower = (phiIndex) % nPhiGlobal;
	size_t phiHigher = (phiLower + 1) % nPhiGlobal;
	fReal lowerBelt = bubbleLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);
	
	phiLower = (phiIndex + nPhiGlobal / 2) % nPhiGlobal;
	phiHigher = (phiLower + 1) % nPhiGlobal;

	fReal higherBelt = -bubbleLerp(input[phiLower + pitch * thetaIndex],
				       input[phiHigher + pitch * thetaIndex], alphaPhi);

	fReal lerped = bubbleLerp(lowerBelt, higherBelt, alphaTheta);
	if (isFlippedPole)
	    lerped = -lerped;
	return lerped;
    }
  
    size_t phiLower = phiIndex % nPhiGlobal;
    size_t phiHigher = (phiLower + 1) % nPhiGlobal;
    size_t thetaLower = thetaIndex;
    size_t thetaHigher = thetaIndex + 1;

    fReal lowerBelt = bubbleLerp(input[phiLower + pitch * thetaLower],
				 input[phiHigher + pitch * thetaLower], alphaPhi);
    fReal higherBelt = bubbleLerp(input[phiLower + pitch * thetaHigher],
				  input[phiHigher + pitch * thetaHigher], alphaPhi);

    fReal lerped = bubbleLerp(lowerBelt, higherBelt, alphaTheta);
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
    	fReal higherBelt = -bubbleLerp(input[phiLower + pitch * thetaIndex],
    				       input[phiHigher + pitch * thetaIndex], alphaPhi);
	
    	phiLower = (phiLower + nPhiGlobal / 2) % nPhiGlobal;
    	phiHigher = (phiHigher + nPhiGlobal / 2) % nPhiGlobal;
    	fReal lowerBelt = bubbleLerp(input[phiLower + pitch * thetaIndex],
    				     input[phiHigher + pitch * thetaIndex], alphaPhi);

    	alphaTheta = 0.5 * alphaTheta;
    	fReal lerped = bubbleLerp(lowerBelt, higherBelt, alphaTheta);
    	return lerped;
	
    }
    
    if (thetaIndex == nThetaGlobal - 2) {
	size_t phiLower = phiIndex % nPhiGlobal;
	size_t phiHigher = (phiLower + 1) % nPhiGlobal;
	fReal lowerBelt = bubbleLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);

	phiLower = (phiLower + nPhiGlobal / 2) % nPhiGlobal;
	phiHigher = (phiHigher + nPhiGlobal / 2) % nPhiGlobal;
	fReal higherBelt = -bubbleLerp(input[phiLower + pitch * thetaIndex],
				       input[phiHigher + pitch * thetaIndex], alphaPhi);

	alphaTheta = 0.5 * alphaTheta;
	fReal lerped = bubbleLerp(lowerBelt, higherBelt, alphaTheta);
	if (isFlippedPole)
	    lerped = -lerped;
	return lerped;
    }
    
    size_t phiLower = phiIndex % nPhiGlobal;
    size_t phiHigher = (phiLower + 1) % nPhiGlobal;
    size_t thetaLower = thetaIndex;
    size_t thetaHigher = thetaIndex + 1;

    fReal lowerBelt = bubbleLerp(input[phiLower + pitch * thetaLower],
				 input[phiHigher + pitch * thetaLower], alphaPhi);
    fReal higherBelt = bubbleLerp(input[phiLower + pitch * thetaHigher],
				  input[phiHigher + pitch * thetaHigher], alphaPhi);

    fReal lerped = bubbleLerp(lowerBelt, higherBelt, alphaTheta);
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
	fReal higherBelt = bubbleLerp(input[phiLower + pitch * thetaIndex],
				      input[phiHigher + pitch * thetaIndex], alphaPhi);

	phiLower = (phiLower + nPhiGlobal / 2) % nPhiGlobal;
	phiHigher = (phiHigher + nPhiGlobal / 2) % nPhiGlobal;
	fReal lowerBelt = bubbleLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);

	fReal lerped = bubbleLerp(lowerBelt, higherBelt, alphaTheta);
	return lerped;
    }
    
    if (isFlippedPole) {
	thetaIndex -= 1;
    }
    
    if (thetaIndex == nThetaGlobal - 1) {
	size_t phiLower = phiIndex % nPhiGlobal;
	size_t phiHigher = (phiLower + 1) % nPhiGlobal;
	fReal lowerBelt = bubbleLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);

	phiLower = (phiLower + nPhiGlobal / 2) % nPhiGlobal;
	phiHigher = (phiHigher + nPhiGlobal / 2) % nPhiGlobal;
	fReal higherBelt = bubbleLerp(input[phiLower + pitch * thetaIndex],
				      input[phiHigher + pitch * thetaIndex], alphaPhi);

	fReal lerped = bubbleLerp(lowerBelt, higherBelt, alphaTheta);
	return lerped;
    }

    size_t phiLower = phiIndex % nPhiGlobal;
    size_t phiHigher = (phiLower + 1) % nPhiGlobal;
    size_t thetaLower = thetaIndex;
    size_t thetaHigher = thetaIndex + 1;

    fReal lowerBelt = bubbleLerp(input[phiLower + pitch * thetaLower],
				 input[phiHigher + pitch * thetaLower], alphaPhi);
    fReal higherBelt = bubbleLerp(input[phiLower + pitch * thetaHigher],
				  input[phiHigher + pitch * thetaHigher], alphaPhi);

    fReal lerped = bubbleLerp(lowerBelt, higherBelt, alphaTheta);
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

/**
 * Trace in great circle as discribed in the paper
 * positive dt => trace backward;
 * negative dt => trace forward;
 */
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


/**
 * \brief Spherical linear interpolation
 */
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
	at(velPhiOutput, thetaId, phiId, pitch) = u_norm; // 0 or NaN
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
	at(velPhiOutput, thetaId, phiId, pitch) = u_norm; // 0 or NaN
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
	at(velThetaOutput, thetaId, phiId, pitch) = u_norm; // 0 or NaN
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
	at(velThetaOutput, thetaId, phiId, pitch) = u_norm; // 0 or NaN
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
}


/**
 * using bimocq instead of the above two basic advection methods 
 */
__global__ void advectionCenteredBimocq
(fReal* attribOutput, fReal* attribInput, fReal* attribInit, fReal* attribDelta,
 fReal* attribInitLast, fReal* attribDeltaLast, fReal* velTheta, fReal* velPhi,
 fReal* bwd_t, fReal* bwd_p, fReal* bwd_tprev, fReal* bwd_pprev, size_t pitch) {

    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    // Coord in scalar space
    fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + centeredOffset;
    
    fReal attrib = 0.0;
    if (at(attribInput, thetaId, phiId, pitch) == thresh) {
    	at(attribOutput, thetaId, phiId, pitch) = thresh;
    	return;
    }

# ifdef MULTISAMPLE
    int samples = 5;
    fReal w[5] = {0.125, 0.125, 0.125, 0.125, 0.5};
    fReal2 dir[5] = {make_fReal2(-0.25,-0.25),
		     make_fReal2(0.25, -0.25),
		     make_fReal2(-0.25, 0.25),
		     make_fReal2( 0.25, 0.25),
		     make_fReal2(0., 0.)};
# else
    int samples = 1;
    fReal w[1] = {1.0};
    fReal2 dir[1] = {make_fReal2(0., 0.)};
# endif

    for (int i = 0; i < samples; i++) {
	fReal2 posId = gId + dir[i];
	fReal2 initPosId = sampleMapping(bwd_t, bwd_p, posId);
	fReal2 lastPosId = sampleMapping(bwd_tprev, bwd_pprev, initPosId);
	attrib += (1.0 - blend_coeff) * w[i] * (sampleCentered(attribInitLast, lastPosId, pitch) +
						   sampleCentered(attribDelta, initPosId, pitch) +
						   sampleCentered(attribDeltaLast, lastPosId, pitch));
	attrib += blend_coeff * w[i] * (sampleCentered(attribInit, initPosId, pitch) +
					   sampleCentered(attribDelta, initPosId, pitch));
    }
    at(attribOutput, thetaId, phiId, pitch) = fmax(thresh, attrib);
}


/**
 * return the maximal absolute value in array with nTheta rows and nPhi cols
 */
fReal Solver::maxAbs(fReal* array, size_t nTheta, size_t nPhi) {
    dim3 gridLayout;
    dim3 blockLayout;
    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    int layout = (gridLayout.x + blockLayout.x - 1) / blockLayout.x;
    fReal *max;
    std::vector<fReal> result(layout);

    checkCudaErrors(cudaMalloc(&max, MAX_BLOCK_SIZE * sizeof(fReal)));
    checkCudaErrors(cudaMemset(max, 0, MAX_BLOCK_SIZE * sizeof(fReal)));

    if (gridLayout.x > MAX_BLOCK_SIZE) {
	gridLayout.x = MAX_BLOCK_SIZE;
	blockLayout.x = N / MAX_BLOCK_SIZE;
	// TODO: check whether blockLayout.x > deviceProp.maxThreadsDim[0]
    }
    maxValKernel<<<gridLayout, blockLayout>>>(max, array);
    checkCudaErrors(cudaDeviceSynchronize());
    maxValKernel<<<layout, blockLayout>>>(max, max);
    checkCudaErrors(cudaMemcpy(&result[0], max, sizeof(fReal) * layout,
     			  cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(max));
    return *std::max_element(result.begin(), result.end());
}


/**
 * update bimocq forward mapping
 */
void Solver::updateForward(fReal dt, fReal* &fwd_t, fReal* &fwd_p) {
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
	    checkCudaErrors(cudaGetLastError());
	    checkCudaErrors(cudaDeviceSynchronize());
	    std::swap(fwd_t, tmp_t);
	    std::swap(fwd_p, tmp_p);
	}
    } else {
	updateMappingKernel<<<gridLayout, blockLayout>>>
	    (velTheta->getGPUThisStep(), velPhi->getGPUThisStep(), -dt_,
	     fwd_t, fwd_p, tmp_t, tmp_p, pitch);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	std::swap(fwd_t, tmp_t);
	std::swap(fwd_p, tmp_p);
    }    
}


/**
 * update bimocq backward mapping
 */
void Solver::updateBackward(fReal dt, fReal* &bwd_t, fReal* &bwd_p) {
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
	    checkCudaErrors(cudaGetLastError());
	    checkCudaErrors(cudaDeviceSynchronize());
	    std::swap(bwd_t, tmp_t);
	    std::swap(bwd_p, tmp_p);
	}
    } else {
	updateMappingKernel<<<gridLayout, blockLayout>>>
	    (velTheta->getGPUThisStep(), velPhi->getGPUThisStep(), dt_,
	     bwd_t, bwd_p, tmp_t, tmp_p, pitch);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	std::swap(bwd_t, tmp_t);
	std::swap(bwd_p, tmp_p);
    }
}


// TODO: delete
void Solver::updateCFL(){
    // values in padding are zero
    this->maxu = maxAbs(velPhi->getGPUThisStep(), velPhi->getNTheta(),
			velPhi->getThisStepPitchInElements());
    this->maxv = maxAbs(velTheta->getGPUThisStep(), velTheta->getNTheta(),
			velTheta->getThisStepPitchInElements());

    this->cfldt = gridLen / std::max(std::max(maxu, maxv), eps);
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


fReal Solver::estimateDistortion() {
    dim3 gridLayout;
    dim3 blockLayout;
    determineLayout(gridLayout, blockLayout, nTheta, nPhi);

    // forward then backward, result saved to tmp_t
    estimateDistortionKernel<<<gridLayout, blockLayout>>>
    	(forward_t, forward_p, backward_t, backward_p, tmp_t);
    // backward then forward, result saved to tmp_p
    estimateDistortionKernel<<<gridLayout, blockLayout>>>
    	(backward_t, backward_p, forward_t, forward_p, tmp_p);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    return max(maxAbs(tmp_t, nTheta, nPhi), maxAbs(tmp_p, nTheta, nPhi));
}


void Solver::advection()
{
    dim3 gridLayout;
    dim3 blockLayout;
    
    // Advect velTheta
    determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
    advectionVSphereThetaKernel<<<gridLayout, blockLayout>>>
	(velTheta->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), velTheta->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    
    // Advect velPhi
    determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());
    advectionVSpherePhiKernel<<<gridLayout, blockLayout>>>
	(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), velPhi->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());

    // Advect scalars
    determineLayout(gridLayout, blockLayout, concentration->getNTheta(), concentration->getNPhi());
# ifdef BIMOCQ
    advectionCenteredBimocq<<<gridLayout, blockLayout>>>
	(thickness->getGPUNextStep(), thickness->getGPUThisStep(), thickness->getGPUInit(),
	 thickness->getGPUDelta(), thickness->getGPUInitLast(),
	 thickness->getGPUDeltaLast(), velTheta->getGPUThisStep(), velPhi->getGPUThisStep(),
	 backward_t, backward_p, backward_tprev, backward_pprev, pitch);
    checkCudaErrors(cudaGetLastError());

    // do not apply bimocq on concentration for stability
    advectionCentered<<<gridLayout, blockLayout>>>
	(concentration->getGPUNextStep(), concentration->getGPUThisStep(),
	 velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), pitch);
# else
    advectionAllCentered<<<gridLayout, blockLayout>>>
	(thickness->getGPUNextStep(), thickness->getGPUThisStep(),
	 concentration->getGPUNextStep(), concentration->getGPUThisStep(),
	 velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), pitch);
# endif
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    thickness->swapGPUBuffer();	
    concentration->swapGPUBuffer();
    swapVelocityBuffers();
}


/**
 * div(u) at cell center
 */
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


    fReal invGridSine = 1.0 / sinf(thetaCoord);
    fReal sinNorth = sinf(thetaNorth);
    fReal sinSouth = sinf(thetaSouth);
    fReal factor = invGridSine * invGridLenGlobal;
    fReal termTheta = factor * (vSouth * sinSouth - vNorth * sinNorth);
    fReal termPhi = invGridLenGlobal * (uEast - uWest);

    fReal f = termTheta + termPhi;

    div[thetaId * nPhiGlobal + phiId] = f;
}


/**
 * according to eq (24b), we can also compute div(u) from gamma values, which is
 * simpler in our case
 */
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
    if (gamma <= 0.0 || gamma > gammaMaxGlobal - eps) {
	// film broken
	gammaNext[thetaId * pitch + phiId] = gammaMaxGlobal;
	at(div, thetaId, phiId) = 1.0 / timeStepGlobal;
    } else {
	gamma = fmin(gamma, 2 * gamma_a); // TODO: clamp??
	at(div, thetaId, phiId) = (1.0 - gamma / gamma_a) / timeStepGlobal;
    }
}


/**
 * build the linear system eq (26)
 */
__global__ void concentrationLinearSystemKernel
(fReal* velPhi_a, fReal* velTheta_a, fReal* gamma_a, fReal* eta_a,
 fReal* W,  fReal* uair, fReal* vair, fReal* fu, fReal* fv,
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
    fReal eta = at(eta_a, thetaId, phiId, pitch);
    if (gamma > (gammaMaxGlobal - eps) || eta == thresh) {
	// force gamma_next = gammaMaxGlobal
	val[idx5] = 0.0;	// up
	val[idx5 + 1] = 0.0;	// left
	val[idx5 + 3] = 0.0;    // right
	val[idx5 + 4] = 0.0;    // down
	val[idx5 + 2] = 1.0;    // center
	rhs[idx] = gammaMaxGlobal;
	at(gamma_a, thetaId, phiId, pitch) = gammaMaxGlobal;
	return;
    }
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
    fReal fuWest = at(fu, thetaId, phiId);
    fReal fuEast = at(fu, thetaId, phiIdEast);

    fReal vNorth = 0.0;
    fReal vSouth = 0.0;
    int oppositePhiId = nPhiGlobal - phiId;
    fReal uAirWest = 0.5 * (at(uair, thetaId, phiId) * currentTimeGlobal / 12.0
		      + at(uair, thetaId, oppositePhiId) * (1-currentTimeGlobal / 12.0));
    int oppositePhiIdEast = nPhiGlobal - phiIdEast;
    fReal uAirEast = 0.5 * (at(uair, thetaId, phiIdEast)  * currentTimeGlobal / 12.0
			    + at(uair, thetaId, oppositePhiIdEast) * (1-currentTimeGlobal / 12.0));
    fReal vAirNorth = 0.0;
    fReal vAirSouth = 0.0;
    fReal fvNorth = 0.0;
    fReal fvSouth = 0.0;

    if (thetaId != 0) {
	int thetaNorthIdx = thetaId - 1;
	vAirNorth = 0.5 * (at(vair, thetaNorthIdx, phiId) * currentTimeGlobal / 12.0
		     - at(vair, thetaNorthIdx, oppositePhiId)
		     * (1-currentTimeGlobal / 12.0 ));
	vNorth = at(velTheta_a, thetaNorthIdx, phiId, pitch);
	fvNorth = at(fv, thetaNorthIdx, phiId);
    	etaNorth = at(eta_a, thetaNorthIdx, phiId, pitch);
	WNorth = at(W, thetaNorthIdx, phiId, pitch);
    } else {
    	int oppositePhiId = (phiId + nThetaGlobal) % nPhiGlobal;
    	etaNorth = eta_a[oppositePhiId];
	WNorth = W[oppositePhiId];
    }
    if (thetaId != nThetaGlobal - 1) {
	vAirSouth = 0.5 * (at(vair, thetaId, phiId)* currentTimeGlobal / 12.0
		     - at(vair, thetaId, oppositePhiId)
			   * (1-currentTimeGlobal / 12.0 ));
	vSouth = at(velTheta_a, thetaId, phiId, pitch);
	fvSouth = at(fv, thetaId, phiId);
    	etaSouth = at(eta_a, thetaId + 1, phiId, pitch);
	WSouth = at(W, thetaId + 1, phiId, pitch);
    } else {
    	int oppositePhiId = (phiId + nThetaGlobal) % nPhiGlobal;
    	etaSouth = at(eta_a, thetaId, oppositePhiId, pitch);
	WSouth = at(W, thetaId, oppositePhiId, pitch);
    }
    if (isnan(vNorth))
	vNorth = 0.0;
    if (isnan(vSouth))
	vSouth = 0.0;
    if (isnan(uWest))
	uWest = 0.0;
    if (isnan(uEast))
	uEast = 0.0;
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
    val[idx5 + 2] = sinTheta * invDt / gamma
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

# ifdef airflow
    sinThetaDiv += invGridLenGlobal * (uAirEast / (etaRight / CrGlobal * invDt + 1.0) -
				       uAirWest / (etaLeft / CrGlobal * invDt + 1.0) +
				       vAirSouth * sinThetaSouth / (etaDown / CrGlobal * invDt + 1.0) -
				       vAirNorth * sinThetaNorth / (etaUp / CrGlobal * invDt + 1.0));
# endif
    
# ifdef gravity
# ifdef VISCOUS
        sinThetaDiv += invGridLenGlobal *
    	(sinThetaSouth * fvSouth * etaDown / (invDt * etaDown + CrGlobal) -
    	 sinThetaNorth * fvNorth * etaUp / (invDt * etaUp + CrGlobal) +
	 fuEast * etaRight / (invDt * etaRight + CrGlobal) -
    	 fuWest * etaLeft / (invDt * etaLeft + CrGlobal));
# else
    sinThetaDiv += gGlobal * invGridLenGlobal *
    	(sinThetaSouth * sinThetaSouth * etaDown / (invDt * etaDown + CrGlobal) -
    	 sinThetaNorth * sinThetaNorth * etaUp / (invDt * etaUp + CrGlobal));
# endif
# endif
    rhs[idx] = sinTheta * invDt - sinThetaDiv;
}


/**
 * solve the linear system using AmgX
 */
void Solver::AlgebraicMultiGridCG() {
    checkCudaErrors(cudaMemcpy2D(d_x, nPhi * sizeof(fReal), concentration->getGPUThisStep(),
				 pitch * sizeof(fReal),
				 nPhi * sizeof(fReal), nTheta,
				 cudaMemcpyDeviceToDevice));

    AMGX_vector_upload(b, N, 1, rhs);
    AMGX_vector_upload(x, N, 1, d_x);
    AMGX_matrix_upload_all(A, N, nz, 1, 1, row_ptr, col_ind, val, 0);
    AMGX_solver_setup(solver, A);
    AMGX_solver_solve(solver, b, x);
    AMGX_vector_download(x, d_x);
    checkCudaErrors(cudaMemcpy2D(concentration->getGPUNextStep(),
				 pitch * sizeof(fReal),
				 d_x, nPhi * sizeof(fReal), nPhi * sizeof(fReal), nTheta,
				 cudaMemcpyDeviceToDevice));
    int num_iter;
    AMGX_SOLVE_STATUS status;
    AMGX_solver_get_status(solver, &status);
    AMGX_solver_get_iterations_number(solver, &num_iter);
    std::cout <<  "Total Iterations:  " << num_iter << std::endl;
    if (status != AMGX_SOLVE_SUCCESS) {
	printGPUarraytoMATLAB("row_ptr.txt", row_ptr, N + 1, 1, 1);
	printGPUarraytoMATLAB("col_ind.txt", col_ind, N, 5, 5);
	printGPUarraytoMATLAB("val.txt", val, N, 5, 5);
	printGPUarraytoMATLAB("rhs.txt", rhs, N, 1, 1);
	printGPUarraytoMATLAB("x.txt", d_x, N, 1, 1);
	setBroken(true);
    }	
}


// This function is not used, only as a record for viscous terms
__global__ void computeforcevelthetaKernel
(fReal* fv, fReal* velTheta, fReal* velPhi, fReal* thickness, fReal* divCentered, size_t pitch) {
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    // Coord in phi-theta space
    fReal gTheta = ((fReal)thetaId + vThetaThetaOffset) * gridLenGlobal;
    fReal gPhi = ((fReal)phiId + vThetaPhiOffset) * gridLenGlobal;

    int thetaNorthId = thetaId;
    int thetaSouthId = thetaId + 1;
    int phiWestId = (phiId - 1 + nPhiGlobal) % nPhiGlobal;
    int phiEastId = (phiId + 1) % nPhiGlobal;
    
    // -   +  v0 +  -
    // d0  u0   u1  d1
    // v3  +  v1 +  v4
    // d2  u2   u3  d3
    // -   +  v2 +  -
    //
    // v1 is the current velTheta
    fReal u0 = velPhi[thetaId * pitch + phiId];
    fReal u1 = velPhi[thetaId * pitch + phiEastId];
    fReal u2 = velPhi[thetaSouthId * pitch + phiId];
    fReal u3 = velPhi[thetaSouthId * pitch + phiEastId];

    // values at centered grid
    fReal divNorth = divCentered[thetaId * nPhiGlobal + phiId];
    fReal divSouth = divCentered[thetaSouthId * nPhiGlobal + phiId];
    fReal EtaNorth = at(thickness, thetaId, phiId, pitch);
    fReal EtaSouth = at(thickness, thetaSouthId, phiId, pitch);

    // values at vTheta grid
    fReal invEta = 2. / (EtaNorth + EtaSouth);
    if (isinf(invEta)) {
	at(fv, thetaId, phiId, pitch) = CUDART_NAN;
	return;
    }

    fReal div = 0.5 * (divNorth + divSouth);
    fReal uPhi = 0.25 * (u0 + u1 + u2 + u3);

    // pDpx = \frac{\partial\Delta}{\partial\phi}
    fReal d0 = thickness[thetaId * pitch + phiWestId];
    fReal d1 = thickness[thetaId * pitch + phiEastId];
    fReal d2 = thickness[thetaSouthId * pitch + phiWestId];
    fReal d3 = thickness[thetaSouthId * pitch + phiEastId];
    fReal pDpx = 0.25 * invGridLenGlobal * (d1 + d3 - d0 - d2);
    fReal pDpy = invGridLenGlobal * (EtaSouth - EtaNorth);

    // pvpy = \frac{\partial u_theta}{\partial\theta}
    fReal v0 = 0.0;
    fReal v1 = velTheta[thetaId * pitch + phiId];
    fReal v2 = 0.0;
    fReal v3 = velTheta[thetaId * pitch + phiWestId];
    fReal v4 = velTheta[thetaId * pitch + phiEastId];
    if (thetaId != 0) {
	size_t thetaNorthId = thetaId - 1;
	v0 = velTheta[thetaNorthId * pitch + phiId];
    } else {
	size_t oppositePhiId = (phiId + nPhiGlobal / 2) % nPhiGlobal;
	v0 = 0.5 * (velTheta[thetaId * pitch + phiId] -
		    velTheta[thetaId * pitch + oppositePhiId]);
    }
    if (thetaId != nThetaGlobal - 2) {
	v2 = velTheta[thetaSouthId * pitch + phiId];
    } else {
	size_t oppositePhiId = (phiId + nPhiGlobal / 2) % nPhiGlobal;
	v2 = 0.5 * (velTheta[thetaId * pitch + phiId] -
		    velTheta[thetaId * pitch + oppositePhiId]);
    }

    fReal pvpy = 0.5 * invGridLenGlobal * (v2 - v0);
    fReal pvpyNorth = invGridLenGlobal * (v1 - v0);
    fReal pvpySouth = invGridLenGlobal * (v2 - v1);
	
    // pvpx = \frac{\partial u_theta}{\partial\phi}    
    fReal pvpx = 0.5 * invGridLenGlobal * (v4 - v3);
    
    // pupy = \frac{\partial u_phi}{\partial\theta}
    fReal pupy = 0.5 * invGridLenGlobal * (u2 + u3 - u0 - u1);

    // pupx = \frac{\partial u_phi}{\partial\phi}
    fReal pupx = 0.5 * invGridLenGlobal * (u1 + u3 - u0 - u2);

    // pupxy = \frac{\partial^2u_\phi}{\partial\theta\partial\phi}
    fReal pupxy = invGridLenGlobal * invGridLenGlobal * (u0 + u3 - u1 - u2);

    // pvpxx = \frac{\partial^2u_\theta}{\partial\phi^2}    
    fReal pvpxx = invGridLenGlobal * invGridLenGlobal * (v3 + v4 - 2 * v1);
    
    // trigonometric function
    fReal sinTheta = sin(gTheta);
    fReal cscTheta = 1. / sinTheta;
    fReal cosTheta = cos(gTheta);
    fReal cotTheta = cosTheta * cscTheta;

    // stress
    // TODO: laplace term
    fReal sigma11North = 0. +  2. * (pvpyNorth + divNorth);
    fReal sigma11South = 0. +  2. * (pvpySouth + divSouth);
    
    // fReal sigma11 = 0. +  2 * pvpy + 2 * div;
    fReal sigma22 = -2. * (pvpy - 2. * div);
    fReal sigma12 = cscTheta * pvpx + pupy - uPhi * cotTheta;

    // psspy = \frac{\partial}{\partial\theta}(\sin\theta\sigma_{11})
    fReal halfStep = 0.5 * gridLenGlobal;    
    fReal thetaSouth = gTheta + halfStep;
    fReal thetaNorth = gTheta - halfStep;    
    fReal sinNorth = sin(thetaNorth);
    fReal sinSouth = sin(thetaSouth);
    fReal psspy = invGridLenGlobal * (sigma11South * sinSouth - sigma11North * sinNorth);
    
    // pspx = \frac{\partial\sigma_{12}}{\partial\phi}
    fReal pspx = cscTheta * pvpxx + pupxy - cotTheta * pupx;

    // force terms
    fReal f2 = reGlobal * cscTheta * invEta * pDpx * sigma12;
    fReal f4 = reGlobal * invEta * pDpy * 2 * (div + pvpy);
    fReal f5 = reGlobal * cscTheta * (psspy + pspx - cosTheta * sigma22);


    
    fReal f7 = gGlobal * sinTheta;
    
    // output
    at(fv, thetaId, phiId) = clamp(f2 + f5 + f4, -0.1, 0.1)  + f7;
    if (sinTheta < 0.1) {
	at(fv, thetaId, phiId) = f7;
    }
}


// This function is not used, only as a record for viscous terms
__global__ void computeforcevelphiKernel_viscous
(fReal* fu, fReal* velPhi, fReal* velTheta, fReal* thickness, fReal* divCentered, size_t pitch) {
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    // Coord in phi-theta space
    fReal gPhi = ((fReal)phiId + vPhiPhiOffset) * gridLenGlobal;
    fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLenGlobal;

    int phiWestId = (phiId - 1 + nPhiGlobal) % nPhiGlobal;
    int thetaNorthId = thetaId - 1;
    int thetaSouthId = thetaId + 1;

    // values at centered grid
    fReal divWest = divCentered[thetaId * nPhiGlobal + phiWestId];
    fReal divEast = divCentered[thetaId * nPhiGlobal + phiId];
    fReal EtaWest = at(thickness, thetaId, phiWestId, pitch);
    fReal EtaEast = at(thickness, thetaId, phiId, pitch);

    // |  d0 u3 d2 |
    // +  v0 +  v1 +
    // u0    u1    u2
    // +  v2 +  v3 + 
    // |  d1 u4 d3 |
    //
    // u1 is the current velPhi
    fReal v0 = 0.0;
    fReal v1 = 0.0;
    fReal v2 = 0.0;
    fReal v3 = 0.0;
    if (thetaId != 0) {
	v0 = velTheta[thetaNorthId * pitch + phiWestId];
	v1 = velTheta[thetaNorthId * pitch + phiId];
    } else {
	v0 = 0.5 * (velTheta[thetaId * pitch + phiWestId] -
		    velTheta[thetaId * pitch + (phiWestId + nPhiGlobal / 2) % nPhiGlobal]);
	v1 = 0.5 * (velTheta[thetaId * pitch + phiId] -
		    velTheta[thetaId * pitch + (phiId + nPhiGlobal / 2) % nPhiGlobal]);
    }
    if (thetaId != nThetaGlobal - 1) {
	v2 = velTheta[thetaId * pitch + phiWestId];
	v3 = velTheta[thetaId * pitch + phiId];
    } else {
	v2 = 0.5 * (velTheta[thetaNorthId * pitch + phiWestId] -
		    velTheta[thetaNorthId * pitch + (phiWestId + nPhiGlobal / 2) % nPhiGlobal]);
	v3 = 0.5 * (velTheta[thetaNorthId * pitch + phiId] -
		    velTheta[thetaNorthId * pitch + (phiId + nPhiGlobal / 2) % nPhiGlobal]);
    }
    
    // values at uPhi grid
    fReal invEta = 2. / (EtaWest + EtaEast);
    if (isinf(invEta)) {
	at(fu, thetaId, phiId, pitch) = CUDART_NAN;
	return;
    }
    fReal div = 0.5 * (divWest + divEast);
    fReal vTheta = 0.25 * (v0 + v1 + v2 + v3);

    // pvpx = \frac{\partial u_theta}{\partial\phi}
    fReal pvpx = 0.5 * invGridLenGlobal * (v1 + v3 - v0 - v2);

    // pvpy = \frac{\partial u_theta}{\partial\theta}
    fReal pvpyWest = invGridLenGlobal * (v2 - v0);
    fReal pvpyEast = invGridLenGlobal * (v3 - v1);
    fReal pvpy = 0.5 * invGridLenGlobal * (v2 + v3 - v0 - v1);

    // pupy = \frac{\partial u_phi}{\partial\theta}
    fReal pupyNorth = 0.0;
    fReal pupySouth = 0.0;
    fReal u1 = at(velPhi, thetaId, phiId, pitch);
    fReal u3 = 0.0;
    fReal u4 = 0.0;
    // actually pupyNorth != 0 at theta == 0, but pupyNorth appears only
    // in sinNorth * pupyNorth, and sinNorth = 0 at theta == 0    
    if (thetaId != 0) {
        u3 = velPhi[thetaNorthId * pitch + phiId];
	pupyNorth = invGridLenGlobal * (u1 - u3);
    } else {
	size_t oppositePhiId = (phiId + nPhiGlobal / 2) % nPhiGlobal;
	u3 = -velPhi[thetaId * pitch + oppositePhiId];
    }
    // actually pupySouth != 0 at theta == \pi, but pupySouth appears only
    // in sinSouth * pupySouth, and sinSouth = 0 at theta == \pi   
    if (thetaId != nThetaGlobal - 1) {
	u4 = velPhi[thetaSouthId * pitch + phiId];
	pupySouth = invGridLenGlobal * (u4 - u1);
    } else {
	size_t oppositePhiId = (phiId + nPhiGlobal / 2) % nPhiGlobal;
	u4 = -velPhi[thetaId * pitch + oppositePhiId];
    }
    fReal pupy = 0.5 * invGridLenGlobal * (u4 - u3);

    // trigonometric function
    fReal sinTheta = sin(gTheta);
    fReal cscTheta = 1. / sinTheta;
    fReal cosTheta = cos(gTheta);
    fReal cotTheta = cosTheta * cscTheta;
    
    // stress
    // TODO: laplace term
    fReal sigma12 = cscTheta * pvpx + pupy - u1 * cotTheta;

    // pDpx = \frac{\partial\Delta}{\partial\phi}
    fReal pDpx = invGridLenGlobal * (EtaEast - EtaWest);

    // pDpy = \frac{\partial\Delta}{\partial\theta}
    // TODO: do we need to average the thickness value at the pole?
    fReal pDpy = 0.0;
    fReal d0 = 0.0;
    fReal d1 = 0.0;
    fReal d2 = 0.0;
    fReal d3 = 0.0;
    if (thetaId != 0) {
	d0 = thickness[thetaNorthId * pitch + phiWestId];
	d2 = thickness[thetaNorthId * pitch + phiId];
    } else {
	d0 = thickness[thetaId * pitch + (phiWestId + nPhiGlobal / 2) % nPhiGlobal];
	d2 = thickness[thetaId * pitch + (phiId + nPhiGlobal / 2) % nPhiGlobal];
    }
    if (thetaId != nThetaGlobal - 1) {
	d1 = thickness[thetaSouthId * pitch + phiWestId];
	d3 = thickness[thetaSouthId * pitch + phiId];
    } else {
	d1 = thickness[thetaId * pitch + (phiWestId + nPhiGlobal / 2) % nPhiGlobal];
	d3 = thickness[thetaId * pitch + (phiId + nPhiGlobal / 2) % nPhiGlobal];
    }
    pDpy = 0.25 * invGridLenGlobal * (d1 + d3 - d0 - d2);
    
    // psspy = \frac{\partial}{\partial\theta}(\sin\theta\sigma_{12})
    fReal halfStep = 0.5 * gridLenGlobal;    
    fReal thetaSouth = gTheta + halfStep;
    fReal thetaNorth = gTheta - halfStep;    
    fReal sinNorth = sinf(thetaNorth);
    fReal sinSouth = sinf(thetaSouth);
    fReal cosNorth = cosf(thetaNorth);
    fReal cosSouth = cosf(thetaSouth);
    // TODO: uncertain about the definintion of u_\phi at both poles
    fReal uNorth = 0.5 * (u3 + u1);
    fReal uSouth = 0.5 * (u1 + u4);
    if (thetaId == 0) 
	uNorth = 0.5 * (u1 - u3);
    if (thetaId == nThetaGlobal - 1)
	uSouth = 0.5 * (u1 - u4);
    fReal psspy = invGridLenGlobal * (invGridLenGlobal * (v0 + v3 - v1 - v2) +
							sinSouth * pupySouth - sinNorth * pupyNorth -
							cosSouth * uSouth + cosNorth * uNorth);
    
    // pspx = \frac{\partial\sigma_{22}}{\partial\phi}
    fReal sigma22West = 2 * (2 * divWest - pvpyWest);
    fReal sigma22East = 2 * (2 * divEast - pvpyEast);
    fReal pspx = invGridLenGlobal * (sigma22East - sigma22West);
    
    // force terms
    // fReal f1 = -vTheta * u1 * cotTheta;
    fReal f2 = reGlobal * invEta * pDpy * sigma12;
    fReal f4 = reGlobal * invEta * cscTheta * pDpx * 2 * ( 2 * div - pvpy);  
    fReal f5 = reGlobal * cscTheta * (psspy + pspx + cosTheta * sigma12);

    // output
    at(fu, thetaId, phiId) = clamp(f2 + f4 + f5, -0.1, 0.1);
    if (sinTheta < 0.1) {
	at(fu, thetaId, phiId) = 0.;
    }
}


__global__ void applyforcevelthetaKernel
(fReal* velThetaOutput, fReal* velThetaInput, fReal* velThetaDelta,
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
    
    fReal GammaNorth = at(concentration, thetaId, phiId, pitch);
    fReal GammaSouth = at(concentration, thetaSouthId, phiId, pitch);
    fReal EtaNorth = at(thickness, thetaId, phiId, pitch);
    fReal EtaSouth = at(thickness, thetaSouthId, phiId, pitch);
    fReal WNorth = at(W, thetaId, phiId, pitch);
    fReal WSouth = at(W, thetaSouthId, phiId, pitch);

    // value at vTheta grid
    fReal invEta = 2. / (EtaNorth + EtaSouth);

    if (isinf(invEta)) {
	at(velThetaOutput, thetaId, phiId, pitch) = CUDART_NAN;
	at(velThetaDelta, thetaId, phiId) = CUDART_NAN;
	return;
    }

    // pGpy = \frac{\partial\Gamma}{\partial\theta};
    fReal pGpy = invGridLenGlobal * (GammaSouth - GammaNorth);

    // elasticity
    fReal f1 = -MGlobal * invEta * pGpy;

    // air friction
    fReal vAir = at(vair, thetaId, phiId);
    // int oppositePhiId = nPhiGlobal - phiId;
    // fReal vAir = 0.5 * (at(vair, thetaId, phiId) * currentTimeGlobal / 12.0
    // 		  - at(vair, thetaId, oppositePhiId) * (1 - currentTimeGlobal / 12.0 ));
    fReal f2 = CrGlobal * invEta * vAir;

    // gravity
    fReal f3 = 0.0;
# ifdef gravity
    f3 = gGlobal * sin(gTheta);
# endif

    // van der Waals
    fReal f4 = invGridLenGlobal * (WNorth - WSouth);

    at(velThetaOutput, thetaId, phiId, pitch) = (v1 / timeStepGlobal + f1 + f2 + f3 + f4)
	/ (1./timeStepGlobal + CrGlobal * invEta);
    at(velThetaDelta, thetaId, phiId) = at(velThetaOutput, thetaId, phiId, pitch) - v1;
}


__global__ void applyforcevelphiKernel
(fReal* velPhiOutput, fReal* velPhiInput, fReal* velPhiDelta,
 fReal* thickness, fReal* W, fReal* concentration, fReal* uair, size_t pitch) {
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    // Coord in phi-theta space
    fReal gPhi = ((fReal)phiId + vPhiPhiOffset) * gridLenGlobal;
    fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLenGlobal;
    fReal sinTheta = sin(gTheta);
    fReal cscTheta = 1.0 / sinTheta;

    int phiWestId = (phiId - 1 + nPhiGlobal) % nPhiGlobal;

    // values at centered grid
    fReal EtaWest = at(thickness, thetaId, phiWestId, pitch);
    fReal EtaEast = at(thickness, thetaId, phiId, pitch);
    fReal GammaWest = at(concentration, thetaId, phiWestId, pitch);
    fReal GammaEast = at(concentration, thetaId, phiId, pitch);
    fReal WWest = at(W, thetaId, phiWestId, pitch);
    fReal WEast = at(W, thetaId, phiId, pitch);
    
    fReal u1 = velPhiInput[thetaId * pitch + phiId];
        
    // value at uPhi grid
    fReal invEta = 2. / (EtaWest + EtaEast);
    if (isinf(invEta)) {
	at(velPhiOutput, thetaId, phiId, pitch) = CUDART_NAN;
	at(velPhiDelta, thetaId, phiId) = CUDART_NAN;
	return;
    }

    
    // pGpx = frac{1}{\sin\theta}\frac{\partial\Gamma}{\partial\phi};
    fReal pGpx = invGridLenGlobal * (GammaEast - GammaWest) * cscTheta;
    
    // elasticity
    fReal f1 = -MGlobal * invEta * pGpx;
    // air friction
    fReal uAir = at(uair, thetaId, phiId);
    // int oppositePhiId = nPhiGlobal - phiId;
    // fReal uAir = 0.5 * (at(uair, thetaId, phiId)  * currentTimeGlobal / 12.0
    // 			+ at(uair, thetaId, oppositePhiId) * (1 - currentTimeGlobal / 12.0 ));

    fReal f2 = CrGlobal * invEta * uAir;
    fReal f3 = 0.0;

    // van der Waals
    fReal f4 = invGridLenGlobal * (WWest - WEast) * cscTheta;

    at(velPhiOutput, thetaId, phiId, pitch) = (u1 / timeStepGlobal + (f1 + f2 + f3 + f4))
	/ (1./timeStepGlobal + CrGlobal * invEta);
    at(velPhiDelta, thetaId, phiId) = at(velPhiOutput, thetaId, phiId, pitch) - u1;
}


// Backward Euler
// This function is not used, only as a record for viscous terms
__global__ void applyforcevelthetaKernel_viscous
(fReal* velThetaOutput, fReal* velThetaInput, fReal* thickness,
 fReal* concentration, fReal* vair, fReal* fv, size_t pitch) {
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    // Coord in phi-theta space
    fReal gTheta = ((fReal)thetaId + vThetaThetaOffset) * gridLenGlobal;
    fReal gPhi = ((fReal)phiId + vThetaPhiOffset) * gridLenGlobal;

    int thetaNorthId = thetaId;
    int thetaSouthId = thetaId + 1;

    // values at centered grid
    fReal GammaNorth = at(concentration, thetaId, phiId, pitch);
    fReal GammaSouth = at(concentration, thetaSouthId, phiId, pitch);
    fReal EtaNorth = at(thickness, thetaId, phiId, pitch);
    fReal EtaSouth = at(thickness, thetaSouthId, phiId, pitch);

    // values at vTheta grid
    fReal invEta = 2. / (EtaNorth + EtaSouth);
    if (isinf(invEta)) {
	at(velThetaOutput, thetaId, phiId, pitch) = CUDART_NAN;
	return;
    }

    fReal v1 = velThetaInput[thetaId * pitch + phiId];
    
    // pGpy = \frac{\partial\Gamma}{\partial\theta};
    fReal pGpy = invGridLenGlobal * (GammaSouth - GammaNorth);

    // force terms
    fReal f3 = -MGlobal * invEta * pGpy;

    int oppositePhiId = nPhiGlobal - phiId;
    fReal vAir = 0.5 * (at(vair, thetaId, phiId) * currentTimeGlobal / 12.0
		  - at(vair, thetaId, oppositePhiId) * (1-currentTimeGlobal / 12.0 ));
    fReal f6 = CrGlobal * invEta * vAir;
    
    // output
    at(velThetaOutput, thetaId, phiId, pitch) = (v1 / timeStepGlobal + f3 + f6 + at(fv, thetaId, phiId))
	/ (1. / timeStepGlobal + CrGlobal * invEta);
}


// This function is not used, only as a record for viscous terms
__global__ void applyforcevelphiKernel_viscous
(fReal* velPhiOutput, fReal* velPhiInput, fReal* thickness,
 fReal* concentration, fReal* uair, fReal* fu, size_t pitch) {
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    // Coord in phi-theta space
    fReal gPhi = ((fReal)phiId + vPhiPhiOffset) * gridLenGlobal;
    fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLenGlobal;

    int thetaNorthId = thetaId - 1;
    int thetaSouthId = thetaId + 1;
    int phiWestId = (phiId - 1 + nPhiGlobal) % nPhiGlobal;

    // values at centered grid
    fReal EtaWest = at(thickness, thetaId, phiWestId, pitch);
    fReal EtaEast = at(thickness, thetaId, phiId, pitch);
    fReal GammaWest = at(concentration, thetaId, phiWestId, pitch);
    fReal GammaEast = at(concentration, thetaId, phiId, pitch);
    
    // values at uPhi grid
    fReal invEta = 2. / (EtaWest + EtaEast);
    if (isinf(invEta)) {
	at(velPhiOutput, thetaId, phiId, pitch) = CUDART_NAN;
	return;
    }
    
    fReal u1 = at(velPhiInput, thetaId, phiId, pitch);
    
    // trigonometric function
    fReal cscTheta = 1. / sin(gTheta);
    
    // pGpx = \frac{\partial\Gamma}{\partial\phi};
    fReal pGpx = invGridLenGlobal * (GammaEast - GammaWest);
    // force terms
    fReal f3 = -MGlobal * invEta * cscTheta * pGpx;

    // fReal f7 = 0.0; 		// gravity
    int oppositePhiId = nPhiGlobal - phiId;
    fReal uAir = 0.5 * (at(uair, thetaId, phiId)  * currentTimeGlobal / 12.0
			+ at(uair, thetaId, oppositePhiId) * (1-currentTimeGlobal / 12.0 ));
    fReal f6 = CrGlobal * invEta * uAir;
    
    // output
    at(velPhiOutput, thetaId, phiId, pitch) = (u1 / timeStepGlobal + f3 + f6 + at(fu, thetaId, phiId))
	/ (1. / timeStepGlobal + CrGlobal * invEta);
}


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
    /* force minimal thickness */
    at(thicknessOutput, thetaId, phiId, pitch) = fmax(thresh, at(thicknessOutput, thetaId, phiId, pitch));
    /* entries for bimocq */
    at(thicknessDelta, thetaId, phiId) = at(thicknessOutput, thetaId, phiId, pitch) - eta;
}


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


__global__ void accumulateChangesCentered(fReal* attribDelta, fReal* attribDeltaTemp,
					   fReal* fwd_t, fReal* fwd_p,
					   size_t pitch) {
    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;
    
    // Coord in scalar space
    fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + centeredOffset;
    
# ifdef MULTISAMPLE
    int samples = 5;
    fReal w[5] = {0.125, 0.125, 0.125, 0.125, 0.5};
    fReal2 dir[5] = {make_fReal2(-0.25,-0.25),
		     make_fReal2(0.25, -0.25),
		     make_fReal2(-0.25, 0.25),
		     make_fReal2( 0.25, 0.25),
		     make_fReal2(0., 0.)};
# else
    int samples = 1;
    fReal w[1] = {1.0};
    fReal2 dir[1] = {make_fReal2(0., 0.)};
# endif

    for (int i = 0; i < samples; i++) {
	fReal2 posId = gId + dir[i];
	fReal2 initPosId = sampleMapping(fwd_t, fwd_p, posId);
	at(attribDelta, thetaId, phiId, pitch)
	    += w[i] * sampleCentered(attribDeltaTemp, initPosId, nPhiGlobal);
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


/** 
 * Error correction, see the bimocq paper
 */
__global__ void correctCentered1(fReal* thicknessCurr, fReal* thicknessError,
				 fReal* thicknessDelta, fReal* thicknessInit,
				 fReal* gammaCurr, fReal* gammaError,
				 fReal* gammaDelta, fReal* gammaInit,
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

    fReal thickness = 0.0;
    fReal gamma = 0.0;

    // Coord in scalar space
    fReal2 gId = make_fReal2((fReal)thetaId, (fReal)phiId) + centeredOffset;
    for (int i = 0; i < 5; i++) {
	fReal2 posId = gId + dir[i];
	fReal2 initPosId = sampleMapping(fwd_t, fwd_p, posId);
	thickness += w[i] * sampleCentered(thicknessCurr, initPosId, pitch);
        gamma += w[i] * sampleCentered(gammaCurr, initPosId, pitch);
    }
    at(thicknessError, thetaId, phiId) = (thickness - at(thicknessDelta, thetaId, phiId, pitch)
					  - at(thicknessInit, thetaId, phiId, pitch)) * 0.5;
    at(gammaError, thetaId, phiId) = (gamma - at(gammaDelta, thetaId, phiId, pitch)
				      - at(gammaInit, thetaId, phiId, pitch)) * 0.5;
}


__global__ void correctCentered2(fReal* thicknessOutput, fReal* thicknessInput,
				 fReal* thicknessError, fReal* gammaOutput,
				 fReal* gammaInput, fReal* gammaError,
				 fReal* bwd_t, fReal* bwd_p, size_t pitch) {
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
	 at(gammaOutput, thetaId, phiId, pitch)
	 -= w[i] * sampleCentered(gammaError, sampleId, nPhiGlobal);
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
    at(thicknessOutput, thetaId, phiId, pitch) = fmax(thresh, at(thicknessOutput, thetaId, phiId, pitch));

    minVal = at(gammaInput, thetaId, phiId, pitch);
    maxVal = minVal;
    for (int t : range) {
    	for (int p : range) {
    	    int2 sampleId = make_int2(thetaId + t, phiId + p);
    	    validateId(sampleId);
    	    fReal currentVal = at(gammaInput, sampleId, pitch);
    	    minVal = fminf(minVal, currentVal);
    	    maxVal = fmaxf(maxVal, currentVal);
    	}
    }
    at(gammaOutput, thetaId, phiId, pitch)
    	= clamp(at(gammaOutput, thetaId, phiId, pitch), minVal, maxVal);
    at(gammaOutput, thetaId, phiId, pitch) = fmax(thresh, at(gammaOutput, thetaId, phiId, pitch));
}


__global__ void vanDerWaals(fReal* W, fReal* eta, size_t pitch) {
    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    fReal Eta = at(eta, thetaId, phiId, pitch);
    fReal invEta = 1.0 / Eta;
    if (Eta > thresh) {
	at(W, thetaId, phiId, pitch) = W1 * powf(invEta, 4.0) - W2 * powf(invEta, 2.0);
    } else {
	at(W, thetaId, phiId, pitch) = 0.0;
    }
}


__global__ void airFlowU(fReal* uair, double* uair_init) {
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

    double angle = M_PI / 2.;
    double3 wind = normalize(make_double3(0., cos(angle), sin(angle)));
    double3 position = make_double3(1., 0, 0);
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
    // create a corridor of airflow with time variance, superposed upon the initial airflow
    double3 Air = 0.5 * sin(beta) / UGlobal * radiusGlobal * projectedWind * (1. - smoothstep(0.2, 0.4, fabs(beta - M_hPI))) * (3. - fabs(currentTimeGlobal - 3.));
    at(uair, thetaId, phiId) = at(uair_init, thetaId, phiId) + fReal(dot(Air, ePhi));
    if (isnan(at(uair, thetaId, phiId))) // normal == currentPosition
    	at(uair, thetaId, phiId) = 0.0;
}


__global__ void airFlowV(fReal* vair, double* vair_init) {
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

    double angle =  M_PI / 2.;
    double3 wind = normalize(make_double3(0., cos(angle), sin(angle)));
    double3 position = make_double3(1., 0, 0);
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
    // create a corridor of airflow with time variance, superposed upon the initial airflow
    double3 Air = 0.5 * sin(beta) / UGlobal * radiusGlobal * projectedWind * (1. - smoothstep(0.2, 0.4, fabs(beta - M_hPI))) * (3. - fabs(currentTimeGlobal - 3.));
    at(vair, thetaId, phiId) = at(vair_init, thetaId, phiId) + fReal(dot(Air, eTheta));
    
    if (isnan(at(vair, thetaId, phiId)))
        at(vair, thetaId, phiId) = 0.0;    
}


void Solver::bodyforce() {
    dim3 gridLayout;
    dim3 blockLayout;

    bool inviscid = true;

# ifdef airflow
    // TODO: pass position and wind as parameter
    determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());
    airFlowU<<<gridLayout, blockLayout>>>(uair, uair_init);
    determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
    airFlowV<<<gridLayout, blockLayout>>>(vair, vair_init);
    checkCudaErrors(cudaGetLastError());
# endif
 
    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    // store van der waals forces in thickness->getGPUNextstep()
    // TODO: deprecate
    vanDerWaals<<<gridLayout, blockLayout>>>
	(thickness->getGPUNextStep(), thickness->getGPUThisStep(), pitch);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

# ifdef VISCOUS
    divergenceKernel<<<gridLayout, blockLayout>>>
	(div, velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), pitch, pitch);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
    computeforcevelthetaKernel<<<gridLayout, blockLayout>>>
	(fv, velTheta->getGPUThisStep(), velPhi->getGPUThisStep(),
	 thickness->getGPUThisStep(), div, pitch);
    checkCudaErrors(cudaGetLastError());

    determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());;
    computeforcevelphiKernel_viscous<<<gridLayout, blockLayout>>>
	(fu, velPhi->getGPUThisStep(), velTheta->getGPUThisStep(),
	 thickness->getGPUThisStep(), div, pitch);
        checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
# endif

    concentrationLinearSystemKernel<<<gridLayout, blockLayout>>>
    	(velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), concentration->getGPUThisStep(),
    	 thickness->getGPUThisStep(), thickness->getGPUNextStep(),
    	 uair, vair, fu, fv,
    	 val, rhs, thickness->getThisStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

# ifdef PERFORMANCE_BENCHMARK
    Timer CGtimer;
    CGtimer.startTimer();
# endif
    AlgebraicMultiGridCG();
    // conjugateGradient();
# ifdef PERFORMANCE_BENCHMARK
    this->CGTime += CGtimer.stopTimer() * 0.001f;
# endif

    // WARNING: this function clamps gamma!!
    // must be called directly after CG solver!!
    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    divergenceKernel_fromGamma<<<gridLayout, blockLayout>>>
    	(div, concentration->getGPUNextStep(), concentration->getGPUThisStep(),
    	 concentration->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
# ifdef VISCOUS
    applyforcevelthetaKernel_viscous<<<gridLayout, blockLayout>>>
	(velTheta->getGPUNextStep(), velTheta->getGPUThisStep(), thickness->getGPUThisStep(),
	 concentration->getGPUNextStep(), vair, fv, pitch);
# else
    applyforcevelthetaKernel<<<gridLayout, blockLayout>>>
    	(velTheta->getGPUNextStep(), velTheta->getGPUThisStep(), tmp_t,
    	 thickness->getGPUThisStep(), thickness->getGPUNextStep(),
	 concentration->getGPUNextStep(), vair, pitch);
# endif
    checkCudaErrors(cudaGetLastError());
    
# ifdef VISCOUS
    applyforcevelphiKernel_viscous<<<gridLayout, blockLayout>>>
    	(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), thickness->getGPUThisStep(), 
    	 concentration->getGPUNextStep(), uair, fu, pitch);
# else
    determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());
    applyforcevelphiKernel<<<gridLayout, blockLayout>>>
    	(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), tmp_p,
	 thickness->getGPUThisStep(), thickness->getGPUNextStep(),
	 concentration->getGPUNextStep(), uair, pitch);
# endif   
    checkCudaErrors(cudaGetLastError());

    // write thickness difference to tmp_t, gamma difference to tmp_p
    determineLayout(gridLayout, blockLayout, thickness->getNTheta(), thickness->getNPhi());
    applyforceThickness<<<gridLayout, blockLayout>>>
	(thickness->getGPUNextStep(), thickness->getGPUThisStep(),
	 tmp_t, div, pitch);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    accumulateChangesCentered<<<gridLayout, blockLayout>>>
	(thickness->getGPUDelta(), tmp_t, forward_t, forward_p, pitch);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    substractPitched<<<gridLayout, blockLayout>>>
    	(tmp_p, concentration->getGPUNextStep(), concentration->getGPUThisStep(),
    	 pitch);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    accumulateChangesCentered<<<gridLayout, blockLayout>>>
    	(concentration->getGPUDelta(), tmp_p, forward_t, forward_p, pitch);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    thickness->swapGPUBuffer();
    concentration->swapGPUBuffer();
    swapVelocityBuffers();
}


void Solver::reInitializeMapping() {
    dim3 gridLayout;
    dim3 blockLayout;

    bool errorCorrection = true;
    if (errorCorrection) {
	determineLayout(gridLayout, blockLayout, thickness->getNTheta(), thickness->getNPhi());
	checkCudaErrors(cudaMemcpy(thickness->getGPUNextStep(), thickness->getGPUThisStep(),
			      thickness->getNTheta() * pitch * sizeof(fReal),
			      cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(concentration->getGPUNextStep(), concentration->getGPUThisStep(),
			      concentration->getNTheta() * pitch * sizeof(fReal),
			      cudaMemcpyDeviceToDevice));

	correctCentered1<<<gridLayout, blockLayout>>>
	    (thickness->getGPUThisStep(), tmp_t, thickness->getGPUDelta(), thickness->getGPUInit(),
	     concentration->getGPUThisStep(), tmp_p, concentration->getGPUDelta(), concentration->getGPUInit(),
	     forward_t, forward_p, pitch);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	correctCentered2<<<gridLayout, blockLayout>>>
	    (thickness->getGPUThisStep(), thickness->getGPUNextStep(), tmp_t,
	     concentration->getGPUThisStep(), concentration->getGPUNextStep(), tmp_p,
	     backward_t, backward_p, pitch);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
    }

    std::swap(this->thickness->getGPUInitLast(), this->thickness->getGPUInit());
    std::swap(this->thickness->getGPUDeltaLast(), this->thickness->getGPUDelta());
    std::swap(this->concentration->getGPUInitLast(), this->concentration->getGPUInit());
    std::swap(this->concentration->getGPUDeltaLast(), this->concentration->getGPUDelta());
    std::swap(backward_tprev, backward_t);
    std::swap(backward_pprev, backward_p);

    checkCudaErrors(cudaMemcpy(this->thickness->getGPUInit(), this->thickness->getGPUThisStep(),
			  this->thickness->getThisStepPitchInElements() * this->thickness->getNTheta() *
			  sizeof(fReal), cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemset(this->thickness->getGPUDelta(), 0,
			  pitch * sizeof(fReal) * this->thickness->getNTheta()));

    checkCudaErrors(cudaMemcpy(this->concentration->getGPUInit(), this->concentration->getGPUThisStep(),
    			  this->concentration->getThisStepPitchInElements() * this->concentration->getNTheta() *
    			  sizeof(fReal), cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemset(this->concentration->getGPUDelta(), 0,
    			  pitch * sizeof(fReal) * this->concentration->getNTheta()));

    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    initMapping<<<gridLayout, blockLayout>>>(forward_t, forward_p);
    initMapping<<<gridLayout, blockLayout>>>(backward_t, backward_p);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}


// assumes U = 1
Bubble::Bubble(fReal radius, fReal H, fReal U, fReal c_m, fReal Gamma_m,
	       fReal T, fReal Ds, fReal rm, size_t nTheta, 
	       fReal dt, fReal DT, int frames,
	       std::string outputDir, std::string thicknessImage,
	       int device, std::string AMGconfig, fReal blendCoeff):
    radius(radius), invRadius(1/radius), H(H), U(1.0), c_m(c_m), Gamma_m(Gamma_m), T(T),
    Ds(Ds/(U*radius)), gs(g*radius/(U*U)), rm(rm), epsilon(H/radius), sigma_r(R*T), M(Gamma_m*sigma_r/(3*rho*H*U*U)),
    S(sigma_a*epsilon/(2*nu*rho*U)), re(nu/(U*radius)), Cr(rhoa*sqrt(nua*radius/U)/(rho*H)),
    nTheta(nTheta), nPhi(2 * nTheta),
    gridLen(M_PI / nTheta), invGridLen(nTheta / M_PI), 
    dt(dt*U/radius), DT(DT), frames(frames), outputDir(outputDir),
    thicknessImage(thicknessImage), device(device),
    AMGconfig(AMGconfig), blendCoeff(blendCoeff), gammaMax(sigma_a / sigma_r / Gamma_m)
{
    // back up files for repeatable trials
    boost::filesystem::create_directories(outputDir);
    if (outputDir.back() != '/')
	this->outputDir.append("/");
    boost::filesystem::copy_file("soapBubble", this->outputDir + "soapBubble",
				 boost::filesystem::copy_option::overwrite_if_exists);
    boost::filesystem::copy_file("../config.txt", this->outputDir + "config.txt",
				 boost::filesystem::copy_option::overwrite_if_exists);

    std::string includeDir = this->outputDir + "include";
    std::string kernelDir = this->outputDir + "kernel";
    copyDir("../include", includeDir);
    copyDir("../kernel", kernelDir);

    std::cout << "Re^-1 " << re << std::endl;
    std::cout << "Cr " << Cr << std::endl;
    std::cout << "M " << M << std::endl;
    std::cout << "S " << S << std::endl;
    std::cout << "epsilon " << epsilon << std::endl;
    std::cout << "gammaMax " << gammaMax << std::endl;
    std::cout << "AMG config file: " << AMGconfig << std::endl;
}
Bubble::~Bubble()
{}

void Bubble::run()
{
    Solver solver(nPhi, nTheta, radius, dt*radius/U, H, device, AMGconfig);
    
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
    checkCudaErrors(cudaMemcpyToSymbol(gammaMaxGlobal, &(this->gammaMax), sizeof(fReal)));
# ifdef evaporation
    fReal eva = evaporation * radius / (H * U);
# else
    fReal eva = 0.;
# endif
    checkCudaErrors(cudaMemcpyToSymbol(evaporationRate, &eva, sizeof(fReal)));
    fReal epsilon_ = .0; // deactivate vdw?
    // Van der Waals forces. Not working properly
    fReal W2_ = radius / (U * U) * pow(rm / H, 2) * epsilon_;
    fReal W1_ = 0.5 * W2_ * pow(rm / H, 2);
    checkCudaErrors(cudaMemcpyToSymbol(W1, &W1_, sizeof(fReal)));
    checkCudaErrors(cudaMemcpyToSymbol(W2, &W2_, sizeof(fReal)));

    solver.initThicknessfromPic(thicknessImage);
    solver.initAirflowfromPic("../init/velocityfield_512.exr");

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
    Timer timer;
    timer.startTimer();
# endif

    fReal T = 0.0;              // simulation time
    int i = 1;
    fReal dt_ = dt * this->radius / this->U;
    for (; i < frames; i++) {
	checkCudaErrors(cudaMemcpyToSymbol(currentTimeGlobal, &T, sizeof(fReal)));
	std::cout << "current time " << T << std::endl;

	checkCudaErrors(cudaMemcpyToSymbol(timeStepGlobal, &dt, sizeof(fReal)));
	std::cout << "current time step size is " << dt_ << " s" << std::endl;
	std::cout << "steps needed until next frame " << DT/dt_ << std::endl;
    
	while ((T + dt_) <= i*DT && !solver.isBroken()) {
	    solver.stepForward();
	    T += dt_;
	}
	if ((T + eps) < i*DT && !solver.isBroken()) {
	    // TODO: is small interval imprecise?
	    fReal tmp_dt = (i * DT - T) * this->U / this->radius;
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
	std::cout << "--------------------------------------------------------" << std::endl;
    }

# ifdef PERFORMANCE_BENCHMARK
    fReal gpu_time = timer.stopTimer();
# endif

    std::cout << "Time spent: " << gpu_time << "ms" << std::endl;
    std::cout << "Performance: " << 1000.0 * i / gpu_time << " frames per second" << std::endl;
}
