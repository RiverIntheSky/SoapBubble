# include "KaminoSolver.cuh"
# include "../include/KaminoGPU.cuh"
# include "../include/KaminoTimer.cuh"

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


#define eps 1e-7f

__device__ bool validateCoord(fReal& phi, fReal& theta, size_t& nPhi) {
    bool ret = false;
    // assume theta lies not too far away from the interval [0, nThetaGlobal],
    // otherwise is the step size too large;
    size_t nTheta = nPhi / 2;

    if (theta >= nTheta) {
	theta = nPhi - theta;
	phi += nTheta;
    	ret = !ret;
    }
    if (theta < 0) {
    	theta = -theta;
    	phi += nTheta;
    	ret = !ret;
    }
    if (theta > nTheta || theta < 0)
	printf("Warning: step size too large! theta = %f\n", theta);
    phi = fmod(phi + nPhi, (fReal)nPhi);
    return ret;
}

__device__ fReal kaminoLerp(fReal from, fReal to, fReal alpha)
{
    return (1.0 - alpha) * from + alpha * to;
}

__device__ fReal sampleVPhi(fReal* input, fReal phiRawId, fReal thetaRawId, size_t pitch) {
    fReal phi = phiRawId - vPhiPhiOffset;
    fReal theta = thetaRawId - vPhiThetaOffset;
    // Phi and Theta are now shifted back to origin

    bool isFlippedPole = validateCoord(phi, theta, nPhiGlobal);

    int phiIndex = static_cast<int>(floorf(phi));
    int thetaIndex = static_cast<int>(floorf(theta));
    fReal alphaPhi = phi - static_cast<fReal>(phiIndex);
    fReal alphaTheta = theta - static_cast<fReal>(thetaIndex);
    
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
    

__device__ fReal sampleVTheta(fReal* input, fReal phiRawId, fReal thetaRawId, size_t pitch) {
    fReal phi = phiRawId - vThetaPhiOffset;
    fReal theta = thetaRawId - vThetaThetaOffset;
    // Phi and Theta are now shifted back to origin

    bool isFlippedPole = validateCoord(phi, theta, nPhiGlobal);

    int phiIndex = static_cast<int>(floorf(phi));
    int thetaIndex = static_cast<int>(floorf(theta));
    fReal alphaPhi = phi - static_cast<fReal>(phiIndex);
    fReal alphaTheta = theta - static_cast<fReal>(thetaIndex);
    
    if (thetaRawId < 0 && thetaRawId > -1 || thetaRawId > nThetaGlobal && thetaRawId < nThetaGlobal + 1 ) {
	thetaIndex -= 1;
	alphaTheta += 1;
    } else if (thetaRawId >= nThetaGlobal + 1 || thetaRawId <= -1) {
    	thetaIndex -= 2;
    }

    if (thetaIndex == 0 && isFlippedPole && thetaRawId > -1) {
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

__device__ fReal sampleCentered(fReal* input, fReal phiRawId, fReal thetaRawId, size_t pitch) {
    fReal phi = phiRawId - centeredPhiOffset;
    fReal theta = thetaRawId - centeredThetaOffset;
    // Phi and Theta are now shifted back to origin

    // if (thetaRawId > nThetaGlobal)
    // 	printf("theta %f\n", theta);
    // TODO change for particles
    bool isFlippedPole = validateCoord(phi, theta, pitch);

    int phiIndex = static_cast<int>(floorf(phi));
    int thetaIndex = static_cast<int>(floorf(theta));
    fReal alphaPhi = phi - static_cast<fReal>(phiIndex);
    fReal alphaTheta = theta - static_cast<fReal>(thetaIndex);

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


__global__ void advectionVSpherePhiKernel
(float* velPhiOutput, float* velPhiInput, float* velThetaInput, size_t pitch)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    
    // Coord in phi-theta space
    float gPhiId = (float)phiId + vPhiPhiOffset;
    float gThetaId = (float)thetaId + vPhiThetaOffset;
    float gTheta = gThetaId * gridLenGlobal;
    float gPhi = gPhiId * gridLenGlobal;

    // Trigonometric functions
    float sinTheta = sinf(gTheta);
    float cosTheta = cosf(gTheta);
    float sinPhi = sinf(gPhi);
    float cosPhi = cosf(gPhi);

    // Sample the speed
    float guTheta = sampleVTheta(velThetaInput, gPhiId, gThetaId, pitch);
    float guPhi = velPhiInput[thetaId * pitch + phiId];

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

    float midTheta = acosf(midx.z);
    float midPhi = atan2f(midx.y, midx.x);

    float midThetaId = midTheta * invGridLenGlobal;
    float midPhiId = midPhi * invGridLenGlobal;

    float muPhi = sampleVPhi(velPhiInput, midPhiId, midThetaId, pitch);
    float muTheta = sampleVTheta(velThetaInput, midPhiId, midThetaId, pitch);

    float3 mu = make_float3(muTheta * cosf(midTheta) * cosf(midPhi) - muPhi * sinf(midPhi),
			    muTheta * cosf(midTheta) * sinf(midPhi) + muPhi * cosf(midPhi),
			    -muTheta * sinf(midTheta));

    float3 uCircleMid_ = u_ * cosf(deltaS) - w_ * sinf(deltaS);
    float3 vCircleMid_ = cross(midx, uCircleMid_);

    float mguPhi = dot(mu, uCircleMid_);  
    float mguTheta = dot(mu, vCircleMid_);

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

    float pTheta = acosf(px.z);
    float pPhi = atan2f(px.y, px.x);

    float pThetaId = pTheta * invGridLenGlobal;
    float pPhiId = pPhi * invGridLenGlobal;

    float puPhi = sampleVPhi(velPhiInput, pPhiId, pThetaId, pitch);
    float puTheta = sampleVTheta(velThetaInput, pPhiId, pThetaId, pitch);

    float3 pu = make_float3(puTheta * cosf(pTheta) * cosf(pPhi) - puPhi * sinf(pPhi),
			    puTheta * cosf(pTheta) * sinf(pPhi) + puPhi * cosf(pPhi),
			    -puTheta * sinf(pTheta));
	
    float3 uCircleP_ = u_ * cosf(deltaS) - w_ * sinf(deltaS);
    float3 vCircleP_ = cross(px, uCircleP_);

    puPhi = dot(pu, uCircleP_);  
    puTheta = dot(pu, vCircleP_);

    pu = puPhi * u_ + puTheta * v_;
    velPhiOutput[thetaId * pitch + phiId] = dot(pu, ePhi);
}


__global__ void advectionVSphereThetaKernel
(float* velThetaOutput, float* velPhiInput, float* velThetaInput, size_t pitch)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    
    // Coord in phi-theta space
    float gPhiId = (float)phiId + vThetaPhiOffset;
    float gThetaId = (float)thetaId + vThetaThetaOffset;
    float gTheta = gThetaId * gridLenGlobal;
    float gPhi = gPhiId * gridLenGlobal;

    // Trigonometric functions
    float sinTheta = sinf(gTheta);
    float cosTheta = cosf(gTheta);
    float sinPhi = sinf(gPhi);
    float cosPhi = cosf(gPhi);

    // Sample the speed
    float guTheta = sampleVTheta(velThetaInput, gPhiId, gThetaId, pitch);
    float guPhi = velPhiInput[thetaId * pitch + phiId];

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

    float midTheta = acosf(midx.z);
    float midPhi = atan2f(midx.y, midx.x);

    float midThetaId = midTheta * invGridLenGlobal;
    float midPhiId = midPhi * invGridLenGlobal;

    float muPhi = sampleVPhi(velPhiInput, midPhiId, midThetaId, pitch);
    float muTheta = sampleVTheta(velThetaInput, midPhiId, midThetaId, pitch);

    float3 mu = make_float3(muTheta * cosf(midTheta) * cosf(midPhi) - muPhi * sinf(midPhi),
			    muTheta * cosf(midTheta) * sinf(midPhi) + muPhi * cosf(midPhi),
			    -muTheta * sinf(midTheta));

    float3 uCircleMid_ = u_ * cosf(deltaS) - w_ * sinf(deltaS);
    float3 vCircleMid_ = cross(midx, uCircleMid_);

    float mguPhi = dot(mu, uCircleMid_);  
    float mguTheta = dot(mu, vCircleMid_);

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

    float pTheta = acosf(px.z);
    float pPhi = atan2f(px.y, px.x);

    float pThetaId = pTheta * invGridLenGlobal;
    float pPhiId = pPhi * invGridLenGlobal;

    float puPhi = sampleVPhi(velPhiInput, pPhiId, pThetaId, pitch);
    float puTheta = sampleVTheta(velThetaInput, pPhiId, pThetaId, pitch);

    float3 pu = make_float3(puTheta * cosf(pTheta) * cosf(pPhi) - puPhi * sinf(pPhi),
			    puTheta * cosf(pTheta) * sinf(pPhi) + puPhi * cosf(pPhi),
			    -puTheta * sinf(pTheta));

    float3 uCircleP_ = u_ * cosf(deltaS) - w_ * sinf(deltaS);
    float3 vCircleP_ = cross(px, uCircleP_);

    puPhi = dot(pu, uCircleP_);  
    puTheta = dot(pu, vCircleP_);

    pu = puPhi * u_ + puTheta * v_;
    velThetaOutput[thetaId * pitch + phiId] = dot(pu, eTheta);
}


__global__ void advectionVPhiKernel
(fReal* attributeOutput, fReal* velPhi, fReal* velTheta, size_t pitch)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    
    // Coord in phi-theta space
    fReal gPhiId = (fReal)phiId + vPhiPhiOffset;
    fReal gThetaId = (fReal)thetaId + vPhiThetaOffset;
    fReal gTheta = gThetaId * gridLenGlobal;
	
    // Sample the speed
    fReal guTheta = sampleVTheta(velTheta, gPhiId, gThetaId, pitch);
    fReal guPhi = velPhi[thetaId * pitch + phiId];

    fReal cofTheta = timeStepGlobal * invGridLenGlobal;
// # ifdef sphere
// fReal cofPhi = cofTheta / sinf(gTheta);
// # else
    fReal cofPhi = cofTheta;
// # endif
    
    fReal deltaPhi = guPhi * cofPhi;
    fReal deltaTheta = guTheta * cofTheta;

# ifdef RUNGE_KUTTA
    // Traced halfway in phi-theta space
    fReal midPhiId = gPhiId - 0.5 * deltaPhi;
    fReal midThetaId = gThetaId - 0.5 * deltaTheta;

    fReal muPhi = sampleVPhi(velPhi, midPhiId, midThetaId, pitch);
    fReal muTheta = sampleVTheta(velTheta, midPhiId, midThetaId, pitch);

    // fReal averuPhi = 0.5 * (muPhi + guPhi);
    // fReal averuTheta = 0.5 * (muTheta + guTheta);

    deltaPhi = muPhi * cofPhi;
    deltaTheta = muTheta * cofTheta;
# endif

    fReal pPhiId = gPhiId - deltaPhi;
    fReal pThetaId = gThetaId - deltaTheta;

    fReal advectedVal = sampleVPhi(velPhi, pPhiId, pThetaId, pitch);

    attributeOutput[thetaId * pitch + phiId] = advectedVal;
};


__global__ void advectionVThetaKernel
(fReal* attributeOutput, fReal* velPhi, fReal* velTheta, size_t pitch)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    
    // Coord in phi-theta space
    fReal gPhiId = (fReal)phiId + vThetaPhiOffset;
    fReal gThetaId = (fReal)thetaId + vThetaThetaOffset;
    fReal gTheta = gThetaId * gridLenGlobal;

    // Sample the speed
    int thetaSouthId = thetaId + 1;
    int phiEastId = (phiId + 1) % nPhiGlobal;
    fReal u0 = velPhi[thetaId * pitch + phiId];
    fReal u1 = velPhi[thetaId * pitch + phiEastId];
    fReal u2 = velPhi[thetaSouthId * pitch + phiId];
    fReal u3 = velPhi[thetaSouthId * pitch + phiEastId];
    fReal guPhi = 0.25 * (u0 + u1 + u2 + u3);
    fReal guTheta = velTheta[thetaId * pitch + phiId];
    
    fReal cofTheta = timeStepGlobal * invGridLenGlobal;
// # ifdef sphere
//     fReal cofPhi = cofTheta / sinf(gTheta);
// # else
    fReal cofPhi = cofTheta;
// # endif
    
    fReal deltaPhi = guPhi * cofPhi;
    fReal deltaTheta = guTheta * cofTheta;

# ifdef RUNGE_KUTTA
    // Traced halfway in phi-theta space
    fReal midPhiId = gPhiId - 0.5 * deltaPhi;
    fReal midThetaId = gThetaId - 0.5 * deltaTheta;
    fReal muPhi = sampleVPhi(velPhi, midPhiId, midThetaId, pitch);
    fReal muTheta = sampleVTheta(velTheta, midPhiId, midThetaId, pitch);

    // fReal averuPhi = 0.5 * (muPhi + guPhi);
    // fReal averuTheta = 0.5 * (muTheta + guTheta);

    deltaPhi = muPhi * cofPhi;
    deltaTheta = muTheta * cofTheta;
# endif

    fReal pPhiId = gPhiId - deltaPhi;
    fReal pThetaId = gThetaId - deltaTheta;

    fReal advectedVal = sampleVTheta(velTheta, pPhiId, pThetaId, pitch);

    attributeOutput[thetaId * pitch + phiId] = advectedVal;
}


__global__ void advectionCentered
(fReal* attributeOutput, fReal* attributeInput, fReal* velPhi, fReal* velTheta, size_t nPitchInElements)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    // Coord in phi-theta space
    fReal gPhiId = (fReal)phiId + centeredPhiOffset;
    fReal gThetaId = (fReal)thetaId + centeredThetaOffset;
    fReal gTheta = gThetaId * gridLenGlobal;
    
    // Sample the speed
    // fReal guPhi = sampleVPhi(velPhi, gPhiId, gThetaId, nPitchInElements);
    fReal guTheta = sampleVTheta(velTheta, gPhiId, gThetaId, nPitchInElements);
    fReal guPhi = 0.5 * (velPhi[thetaId * nPitchInElements + phiId] +
    			 velPhi[thetaId * nPitchInElements + (phiId + 1) % nPhiGlobal]);

    fReal cofTheta = timeStepGlobal * invGridLenGlobal;
# ifdef sphere
    fReal cofPhi = cofTheta / sinf(gTheta);
# else
    fReal cofPhi = cofTheta;
# endif

    fReal deltaPhi = guPhi * cofPhi;
    fReal deltaTheta = guTheta * cofTheta;

# ifdef RUNGE_KUTTA
    // Traced halfway in phi-theta space
    fReal midPhiId = gPhiId - 0.5 * deltaPhi;
    fReal midThetaId = gThetaId - 0.5 * deltaTheta;
    
    fReal muPhi = sampleVPhi(velPhi, midPhiId, midThetaId, nPitchInElements);
    fReal muTheta = sampleVTheta(velTheta, midPhiId, midThetaId, nPitchInElements);

    deltaPhi = muPhi * cofPhi;
    deltaTheta = muTheta * cofTheta;
# endif

    fReal pPhiId = gPhiId - deltaPhi;
    fReal pThetaId = gThetaId - deltaTheta;

    fReal advectedAttribute = sampleCentered(attributeInput, pPhiId, pThetaId, nPitchInElements);
     
    attributeOutput[thetaId * nPitchInElements + phiId] = advectedAttribute;
};


__global__ void advectionAllCentered
(fReal* thicknessOutput, fReal* surfOutput, fReal* thicknessInput, fReal* surfInput, fReal* velPhi, fReal* velTheta, size_t nPitchInElements)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    // Coord in phi-theta space
    fReal gPhiId = (fReal)phiId + centeredPhiOffset;
    fReal gThetaId = (fReal)thetaId + centeredThetaOffset;
    fReal gTheta = gThetaId * gridLenGlobal;
    
    // Sample the speed
    fReal guTheta = sampleVTheta(velTheta, gPhiId, gThetaId, nPitchInElements);
    fReal guPhi = 0.5 * (velPhi[thetaId * nPitchInElements + phiId] +
    			 velPhi[thetaId * nPitchInElements + (phiId + 1) % nPhiGlobal]);

    fReal cofTheta = timeStepGlobal * invGridLenGlobal;
# ifdef sphere
    fReal cofPhi = cofTheta / sinf(gTheta);
# else
    fReal cofPhi = cofTheta;
# endif

    fReal deltaPhi = guPhi * cofPhi;
    fReal deltaTheta = guTheta * cofTheta;

# ifdef RUNGE_KUTTA
    // Traced halfway in phi-theta space
    fReal midPhiId = gPhiId - 0.5 * deltaPhi;
    fReal midThetaId = gThetaId - 0.5 * deltaTheta;
    
    fReal muPhi = sampleVPhi(velPhi, midPhiId, midThetaId, nPitchInElements);
    fReal muTheta = sampleVTheta(velTheta, midPhiId, midThetaId, nPitchInElements);

    deltaPhi = muPhi * cofPhi;
    deltaTheta = muTheta * cofTheta;
# endif

    fReal pPhiId = gPhiId - deltaPhi;
    fReal pThetaId = gThetaId - deltaTheta;

    fReal advectedThickness = sampleCentered(thicknessInput, pPhiId, pThetaId, nPitchInElements);
    fReal advectedSurf = sampleCentered(surfInput, pPhiId, pThetaId, nPitchInElements);
    
    thicknessOutput[thetaId * nPitchInElements + phiId] = advectedThickness;
    surfOutput[thetaId * nPitchInElements + phiId] = advectedSurf;
};


__global__ void advectionParticles(fReal* output, fReal* velPhi, fReal* velTheta, fReal* input, size_t pitch, size_t numOfParticles)
{
    int particleId = blockIdx.x * blockDim.x + threadIdx.x;

    if (particleId < numOfParticles) {
	fReal phiId = input[2 * particleId];
	fReal thetaId = input[2 * particleId + 1];
	fReal theta = thetaId * gridLenGlobal;
	fReal sinTheta = max(sinf(theta), eps);

	fReal uPhi = sampleVPhi(velPhi, phiId, thetaId, pitch);
	fReal uTheta = sampleVTheta(velTheta, phiId, thetaId, pitch);

	fReal cofTheta = timeStepGlobal * invGridLenGlobal;
# ifdef sphere
	fReal cofPhi = cofTheta / sinTheta;
# else
	fReal cofPhi = cofTheta;
# endif
	
	fReal deltaPhi = uPhi * cofPhi;
	fReal deltaTheta = uTheta * cofTheta;

# ifdef RUNGE_KUTTA
	// Traced halfway in phi-theta space
	fReal midPhiId = phiId + 0.5 * deltaPhi;
	fReal midThetaId = thetaId + 0.5 * deltaTheta;

	fReal muPhi = sampleVPhi(velPhi, midPhiId, midThetaId, pitch);
	fReal muTheta = sampleVTheta(velTheta, midPhiId, midThetaId, pitch);

	deltaPhi = muPhi * cofPhi;
	deltaTheta = muTheta * cofTheta;

	theta = midThetaId * gridLenGlobal;
	sinTheta = max(sinf(theta), eps);
# endif

	fReal updatedThetaId = thetaId + deltaTheta;
	fReal updatedPhiId = phiId + deltaPhi;

	validateCoord(updatedPhiId, updatedThetaId, nPhiGlobal);
	
	output[2 * particleId] = updatedPhiId;
	output[2 * particleId + 1] = updatedThetaId;
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
	fReal gPhiId = particleCoord[2 * particleId];
	fReal gThetaId = particleCoord[2 * particleId + 1];

	fReal gTheta = gThetaId * gridLenGlobal;

	fReal sinTheta = max(sinf(gTheta), eps);

	fReal gPhi = gPhiId * gridLenGlobal;

	size_t thetaId = static_cast<size_t>(floorf(gThetaId));

	fReal x1 = cosf(gPhi) * sinTheta; fReal y1 = sinf(gPhi) * sinTheta; fReal z1 = cosf(gTheta);

	fReal theta = (thetaId + 0.5) * gridLenGlobal;

	fReal phiRange = .5f/sinTheta;
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


// advection
__global__ void normalizeThickness_a
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
	fReal gPhiId = (fReal)phiId + centeredPhiOffset;
	fReal gThetaId = (fReal)thetaId + centeredThetaOffset;
	fReal gTheta = gThetaId * gridLenGlobal;

	// Sample the speed
	fReal guPhi = sampleVPhi(velPhi, gPhiId, gThetaId, pitch);
	fReal guTheta = sampleVTheta(velTheta, gPhiId, gThetaId, pitch);

	fReal cofTheta = timeStepGlobal * invGridLenGlobal;
# ifdef sphere
	fReal cofPhi = cofTheta / sinf(gTheta);
# else
	fReal cofPhi = cofTheta;
# endif

	fReal deltaPhi = guPhi * cofPhi;
	fReal deltaTheta = guTheta * cofTheta;
	
# ifdef RUNGE_KUTTA
	// Traced halfway in phi-theta space
	fReal midPhiId = gPhiId - 0.5 * deltaPhi;
	fReal midThetaId = gThetaId - 0.5 * deltaTheta;
    
	fReal muPhi = sampleVPhi(velPhi, midPhiId, midThetaId, pitch);
	fReal muTheta = sampleVTheta(velTheta, midPhiId, midThetaId, pitch);

	deltaPhi = muPhi * cofPhi;
	deltaTheta = muTheta * cofTheta;
# endif

	fReal pPhiId = gPhiId - deltaPhi;
	fReal pThetaId = gThetaId - deltaTheta;

        fReal advectedVal = sampleCentered(thicknessInput, pPhiId, pThetaId, pitch);
	//	fReal f = div[thetaId * nPhiGlobal + phiId];
	thicknessOutput[thetaId * pitch + phiId] = advectedVal;// / (1 + timeStepGlobal * f);
    }
}


// body force
__global__ void normalizeThickness_f
(fReal* thicknessOutput, fReal* thicknessInput,
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
	fReal f = div[thetaId * nPhiGlobal + phiId];
	thicknessOutput[thetaId * pitch + phiId] =  thicknessInput[thetaId * pitch + phiId] * (1 - timeStepGlobal * f);
    }
}


void KaminoSolver::advection(fReal& dt) {
    checkCudaErrors(cudaMemcpyToSymbol(timeStepGlobal, &dt, sizeof(fReal)));
    advection();
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
    checkCudaErrors(cudaDeviceSynchronize());

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
    //    checkCudaErrors(cudaDeviceSynchronize());
   

    // Advect concentration
    determineLayout(gridLayout, blockLayout, surfConcentration->getNTheta(), surfConcentration->getNPhi());
    advectionCentered<<<gridLayout, blockLayout>>>
	(surfConcentration->getGPUNextStep(), surfConcentration->getGPUThisStep(),
	 velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), surfConcentration->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    // checkCudaErrors(cudaDeviceSynchronize());
 
    // Advect thickness particles
    if (particles->numOfParticles > 0) {
	determineLayout(gridLayout, blockLayout, 1, particles->numOfParticles);
	advectionParticles<<<gridLayout, blockLayout>>>
	    (particles->coordGPUNextStep, velPhi->getGPUThisStep(), velTheta->getGPUThisStep(),
	     particles->coordGPUThisStep, velPhi->getThisStepPitchInElements(), particles->numOfParticles);
	checkCudaErrors(cudaGetLastError());
	// checkCudaErrors(cudaDeviceSynchronize());

	// reset weight
	determineLayout(gridLayout, blockLayout, nTheta, nPhi);
	resetThickness<<<gridLayout, blockLayout>>>(weight);
	checkCudaErrors(cudaGetLastError());
	// checkCudaErrors(cudaDeviceSynchronize());

	determineLayout(gridLayout, blockLayout, 2, particles->numOfParticles);
	mapParticlesToThickness<<<gridLayout, blockLayout>>>
	    (particles->coordGPUThisStep, particles->value,  weight, particles->numOfParticles);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
    }

    // average particle information
    // If numOfParticles == 0, choose semi-Lagrangian advection 
    determineLayout(gridLayout, blockLayout, thickness->getNTheta(), thickness->getNPhi());
    normalizeThickness_a<<<gridLayout, blockLayout>>>
    (thickness->getGPUNextStep(), thickness->getGPUThisStep(), velPhi->getGPUThisStep(),
    velTheta->getGPUThisStep(), div, weight, thickness->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    thickness->swapGPUBuffer();
    particles->swapGPUBuffers(); 
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
#endif
    fReal termPhi = factor * (uEast - uWest);

    fReal f = termTheta + termPhi;

    div[thetaId * nPhiGlobal + phiId] = f;
}


__global__ void concentrationLinearSystemKernel
(float* div_a, float* gamma_a, float* eta_a, float* val, float* rhs, size_t pitch) {
    // TODO: pre-compute eta???
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    int idx = thetaId * nPhiGlobal + phiId;
    int idx5 = 5 * idx;

    float gamma = gamma_a[thetaId * pitch + phiId];
    float div = div_a[idx];

# ifdef sphere
    fReal gTheta = ((float)thetaId + centeredThetaOffset) * gridLenGlobal;
    fReal gPhi = ((float)phiId + centeredPhiOffset) * gridLenGlobal;
    fReal halfStep = 0.5 * gridLenGlobal;
    fReal sinThetaSouth = sinf(gTheta + halfStep);
    fReal sinThetaNorth = sinf(gTheta - halfStep);
    fReal sinTheta = sinf(gTheta);
    fReal cscTheta = 1. / sinTheta;
# endif

    // neighboring eta values
    fReal eta = eta_a[thetaId * pitch + phiId];
    fReal etaWest = eta_a[thetaId * pitch + (phiId - 1 + nPhiGlobal) % nPhiGlobal];
    fReal etaEast = eta_a[thetaId * pitch + (phiId + 1) % nPhiGlobal];
    fReal etaNorth = 0.0;
    fReal etaSouth = 0.0;
    if (thetaId != 0) {
    	etaNorth = eta_a[(thetaId - 1) * pitch + phiId];
    } else {
    	int oppositePhiId = (phiId + nThetaGlobal) % nPhiGlobal;
    	etaNorth = eta_a[oppositePhiId];
    }
    if (thetaId != nThetaGlobal - 1) {
    	etaSouth = eta_a[(thetaId + 1) * pitch + phiId];
    } else {
    	int oppositePhiId = (phiId + nThetaGlobal) % nPhiGlobal;
    	etaSouth = eta_a[thetaId * pitch + oppositePhiId];
    }

    // constant for this grid
    float oDtCre = 1./timeStepGlobal + CrGlobal/eta; // \frac{1}{\Delta t} + Cr/eta
    float MDt = MGlobal * timeStepGlobal; // M\Delta t
    /* Ds is zero */
    // float oCrDteDs = (1 + CrGlobal * timeStepGlobal / eta) * DsGlobal; // (1+Cr\eta\Delta t)D_s
    float s2 = invGridLenGlobal * invGridLenGlobal; // \Delta s^2

    rhs[idx] = oDtCre - div;
    float diva = 0.f;
# ifdef sphere
# ifdef uair
    diva += 20.f * (1 - smoothstep(0.f, 10.f, currentTimeGlobal)) * (M_hPI - gTheta)
	* expf(-10 * powf(fabsf(gTheta - M_hPI), 2.f)) * radiusGlobal
	* sinf(gPhi) * cscTheta / UGlobal;
# endif
# ifdef vair
    diva += (gTheta < M_hPI) * 4 * (1 - smoothstep(0.f, 10.f, currentTimeGlobal)) * cosf(gTheta)
	* cosf(2 * gPhi) * radiusGlobal / UGlobal;
# endif
# endif
    rhs[idx] -= CrGlobal * timeStepGlobal / eta * diva;
# ifdef sphere
    rhs[idx] *= sinTheta;
# endif
	
    // up
    float etaxyminus_ = 2. / (etaNorth + eta);
# ifdef sphere
    val[idx5] = -s2 * sinThetaNorth * MDt * etaxyminus_;
# else
    val[idx5] = -s2 * MDt * etaxyminus_;
# endif

    // left
    float etaxminusy_ = 2. / (etaWest + eta);
# ifdef sphere
    val[idx5 + 1] = -s2 * cscTheta * MDt * etaxminusy_;
# else
    val[idx5 + 1] = -s2 * MDt * etaxminusy_;
# endif  

    // right
    float etaxplusy_ = 2. / (etaEast + eta);
# ifdef sphere
    val[idx5 + 3] = -s2 * cscTheta * MDt * etaxplusy_;
# else
    val[idx5 + 3] = -s2 * MDt * etaxplusy_;
# endif  
    
    // down
    float etaxyplus_ = 2. / (etaSouth + eta);
# ifdef sphere
    val[idx5 + 4] = -s2 * sinThetaSouth * MDt * etaxyplus_;
# else
    val[idx5 + 4] = -s2 * MDt * etaxyplus_;
# endif  

    // center
# ifdef sphere
    val[idx5 + 2] = oDtCre / gamma * sinTheta + MDt * s2 *
	(cscTheta * (etaxplusy_ + etaxminusy_)
	 + sinThetaNorth * etaxyminus_
	 + sinThetaSouth * etaxyplus_);
# else
    val[idx5 + 2] = oDtCre / gamma + s2 * MDt * 
	(etaxyminus_ + etaxminusy_ + etaxplusy_ + etaxyplus_);
# endif
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

    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, N, N, nz, row_ptr, col_ind,
    				     val, CUSPARSE_INDEX_32I,
    				     CUSPARSE_INDEX_32I,
    				     CUSPARSE_INDEX_BASE_ZERO,
    				     CUDA_R_32F));

    // printGPUarraytoMATLAB<float>("test/val.txt", val, N, 5, 5);
    // printGPUarraytoMATLAB<float>("test/rhs.txt", rhs, N, 1, 1);

    // r = b - Ax
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, trans,
					   &minusone, matA, vecX, &one, vecR, CUDA_R_32F,
					   CUSPARSE_CSRMV_ALG1, &bufferSize));

    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, trans,
    				&minusone, matA, vecX, &one, vecR, CUDA_R_32F,
    				CUSPARSE_CSRMV_ALG1, dBuffer));

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
	CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, trans,
				    &one, matA, vecP, &zero, vecO, CUDA_R_32F,
				    CUSPARSE_CSRMV_ALG1, dBuffer));

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
    f3 = gGlobal * sinf(gTheta);
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
# endif

    int phiWestId = (phiId - 1 + nPhiGlobal) % nPhiGlobal;

    // values at centered grid
    fReal DeltaWest = thickness[thetaId * pitch + phiWestId];
    fReal DeltaEast = thickness[thetaId * pitch + phiId];
    fReal GammaWest = concentration[thetaId * pitch + phiWestId];
    fReal GammaEast = concentration[thetaId * pitch + phiId];
    
    fReal u1 = velPhiInput[thetaId * pitch + phiId];
        
    // value at uPhi grid
    fReal invDelta = 2. / (DeltaWest + DeltaEast);
    
    // pGpx = \frac{\partial\Gamma}{\partial\phi};
    fReal pGpx = invGridLenGlobal * (GammaEast - GammaWest);    

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
        
    velPhiOutput[thetaId * pitch + phiId] = (u1 / timeStepGlobal + f1 + f2) / (1./timeStepGlobal + CrGlobal * invDelta);   
}


// Backward Euler
__global__ void applyforcevelthetaKernel_viscous
(fReal* velThetaOutput, fReal* velThetaInput, fReal* velPhi, fReal* thickness,
 fReal* concentration, fReal* divCentered, size_t pitch) {
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    // Coord in phi-theta space
    fReal gPhi = ((fReal)phiId + vPhiPhiOffset) * gridLenGlobal;
    fReal gTheta = ((fReal)thetaId + vThetaThetaOffset) * gridLenGlobal;

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
    fReal GammaNorth = concentration[thetaId * pitch + phiId];
    fReal GammaSouth = concentration[thetaSouthId * pitch + phiId];
    fReal DeltaNorth = thickness[thetaId * pitch + phiId];
    fReal DeltaSouth = thickness[thetaSouthId * pitch + phiId];

    // values at vTheta grid
    fReal div = 0.5 * (divNorth + divSouth);
    fReal Delta = 0.5 * (DeltaNorth + DeltaSouth);
    fReal invDelta = 1. / Delta;
    fReal uPhi = 0.25 * (u0 + u1 + u2 + u3);

    // pDpx = \frac{\partial\Delta}{\partial\phi}
    fReal d0 = thickness[thetaId * pitch + phiWestId];
    fReal d1 = thickness[thetaId * pitch + phiEastId];
    fReal d2 = thickness[thetaSouthId * pitch + phiWestId];
    fReal d3 = thickness[thetaSouthId * pitch + phiEastId];
    fReal pDpx = 0.25 * invGridLenGlobal * (d1 + d3 - d0 - d2);
    fReal pDpy = invGridLenGlobal * (DeltaSouth - DeltaNorth);

    // pvpy = \frac{\partial u_theta}{\partial\theta}
    fReal v0 = 0.0;
    fReal v1 = velThetaInput[thetaId * pitch + phiId];
    fReal v2 = 0.0;
    fReal v3 = velThetaInput[thetaId * pitch + phiWestId];
    fReal v4 = velThetaInput[thetaId * pitch + phiEastId];
    if (thetaId != 0) {
	size_t thetaNorthId = thetaId - 1;
	v0 = velThetaInput[thetaNorthId * pitch + phiId];
    } else {
	size_t oppositePhiId = (phiId + nPhiGlobal / 2) % nPhiGlobal;
	v0 = 0.5 * (velThetaInput[thetaId * pitch + phiId] -
		    velThetaInput[thetaId * pitch + oppositePhiId]);
    }
    if (thetaId != nThetaGlobal - 2) {
	v2 = velThetaInput[thetaSouthId * pitch + phiId];
    } else {
	size_t oppositePhiId = (phiId + nPhiGlobal / 2) % nPhiGlobal;
	v2 = 0.5 * (velThetaInput[thetaId * pitch + phiId] -
		    velThetaInput[thetaId * pitch + oppositePhiId]);
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
    fReal sinTheta = sinf(gTheta);
    fReal cscTheta = 1. / sinTheta;
    fReal cosTheta = cosf(gTheta);
    fReal cotTheta = cosTheta * cscTheta;

    // stress
    // TODO: laplace term
    fReal sigma11North = 0. +  2 * (pvpyNorth + divNorth);
    fReal sigma11South = 0. +  2 * (pvpySouth + divSouth);
    
    // fReal sigma11 = 0. +  2 * pvpy + 2 * div;
    fReal sigma22 = -2 * (pvpy - 2 * div);
    fReal sigma12 = cscTheta * pvpx + pupy - uPhi * cotTheta;

    // psspy = \frac{\partial}{\partial\theta}(\sin\theta\sigma_{11})
    fReal halfStep = 0.5 * gridLenGlobal;    
    fReal thetaSouth = gTheta + halfStep;
    fReal thetaNorth = gTheta - halfStep;    
    fReal sinNorth = sinf(thetaNorth);
    fReal sinSouth = sinf(thetaSouth);    
    fReal psspy = invGridLenGlobal * (sigma11South * sinSouth - sigma11North * sinNorth);
    
    // pspx = \frac{\partial\sigma_{12}}{\partial\phi}
    fReal pspx = cscTheta * pvpxx + pupxy - cotTheta * pupx;

    // pGpy = \frac{\partial\Gamma}{\partial\theta};
    fReal pGpy = invGridLenGlobal * (GammaSouth - GammaNorth);

    // force terms
    fReal f1 = uPhi * uPhi * cotTheta;
    fReal f2 = reGlobal * cscTheta * invDelta * pDpx * sigma12;
    fReal f3 = -MGlobal * invDelta * pGpy;
    fReal f4 = reGlobal * invDelta * pDpy * 2 * (div + pvpy);
    fReal f5 = reGlobal * cscTheta * (psspy + pspx - cosTheta * sigma22);
    
# ifdef gravity
    fReal f7 = gGlobal * sinTheta;
# else
    fReal f7 = 0.0;
# endif
    fReal vAir = 0.0;
    fReal f6 = CrGlobal * invDelta * (vAir - v1);
    
    // output
    fReal result = (v1 + timeStepGlobal * (f1 + f2 + f3 + f4 + f5 + CrGlobal * vAir + f7))
	/ (1.0 + CrGlobal * invDelta * timeStepGlobal);
    // if (fabsf(result) < eps)
    // 	result = 0.f;
    velThetaOutput[thetaId * pitch + phiId] = result;
}


// Backward Euler
__global__ void applyforcevelphiKernel_viscous
(fReal* velPhiOutput, fReal* velTheta, fReal* velPhiInput, fReal* thickness,
 fReal* concentration, fReal* divCentered, size_t pitch) {
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    // Coord in phi-theta space
    fReal gPhi = ((fReal)phiId + vPhiPhiOffset) * gridLenGlobal;
    fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLenGlobal;

    int phiWestId = (phiId - 1 + nPhiGlobal) % nPhiGlobal;
    int thetaNorthId = thetaId - 1;
    int thetaSouthId = thetaId + 1;

    // values at centered grid
    fReal divWest = divCentered[thetaId * nPhiGlobal + phiWestId];
    fReal divEast = divCentered[thetaId * nPhiGlobal + phiId];
    fReal DeltaWest = thickness[thetaId * pitch + phiWestId];
    fReal DeltaEast = thickness[thetaId * pitch + phiId];
    fReal GammaWest = concentration[thetaId * pitch + phiWestId];
    fReal GammaEast = concentration[thetaId * pitch + phiId];
    
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
    fReal Delta = 0.5 * (DeltaWest + DeltaEast);
    fReal invDelta = 1. / Delta;
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
    fReal u1 = velPhiInput[thetaId * pitch + phiId];
    fReal u3 = 0.0;
    fReal u4 = 0.0;
    // actually pupyNorth != 0 at theta == 0, but pupyNorth appears only
    // in sinNorth * pupyNorth, and sinNorth = 0 at theta == 0    
    if (thetaId != 0) {
        u3 = velPhiInput[thetaNorthId * pitch + phiId];
	pupyNorth = invGridLenGlobal * (u1 - u3);
    } else {
	size_t oppositePhiId = (phiId + nPhiGlobal / 2) % nPhiGlobal;
	u3 = -velPhiInput[thetaId * pitch + oppositePhiId];
    }
    // actually pupySouth != 0 at theta == \pi, but pupySouth appears only
    // in sinSouth * pupySouth, and sinSouth = 0 at theta == \pi   
    if (thetaId != nThetaGlobal - 1) {
	u4 = velPhiInput[thetaSouthId * pitch + phiId];
	pupySouth = invGridLenGlobal * (u4 - u1);
    } else {
	size_t oppositePhiId = (phiId + nPhiGlobal / 2) % nPhiGlobal;
	u4 = -velPhiInput[thetaId * pitch + oppositePhiId];
    }
    fReal pupy = 0.5 * invGridLenGlobal * (u4 - u3);

    // pGpx = \frac{\partial\Gamma}{\partial\phi};
    fReal pGpx = invGridLenGlobal * (GammaEast - GammaWest);

    // trigonometric function
    fReal sinTheta = sinf(gTheta);
    fReal cscTheta = 1. / sinTheta;
    fReal cosTheta = cosf(gTheta);
    fReal cotTheta = cosTheta * cscTheta;
    
    // stress
    // TODO: laplace term
    fReal sigma12 = cscTheta * pvpx + pupy - u1 * cotTheta;

    // pDpx = \frac{\partial\Delta}{\partial\phi}
    fReal pDpx = invGridLenGlobal * (DeltaEast - DeltaWest);

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
    fReal psspy = invGridLenGlobal * (invGridLenGlobal * (v0 + v3 - v1 - v2) +
							sinSouth * pupySouth - sinNorth * pupyNorth -
							cosSouth * uSouth + cosNorth * uNorth);
    
    // pspx = \frac{\partial\sigma_{22}}{\partial\phi}
    fReal sigma22West = 2 * (2 * divWest - pvpyWest);
    fReal sigma22East = 2 * (2 * divEast - pvpyEast);
    fReal pspx = invGridLenGlobal * (sigma22East - sigma22West);
    
    // force terms
    // fReal f1 = -vTheta * u1 * cotTheta;
    fReal f2 = reGlobal * invDelta * pDpy * sigma12;
    fReal f3 = -MGlobal * invDelta * cscTheta * pGpx;
    fReal f4 = reGlobal * invDelta * cscTheta * pDpx * 2 * ( 2 * div - pvpy);
    fReal f5 = reGlobal * cscTheta * (psspy + pspx + cosTheta * sigma12);

    // fReal f7 = 0.0; 		// gravity
    fReal uAir = 0.0;
# ifdef uair
    if (currentTimeGlobal < 5)
    	uAir = 20.f * (M_hPI - gTheta) * expf(-10 * powf(fabsf(gTheta - M_hPI), 2.f)) * radiusGlobal * cosf(gPhi) / UGlobal;
# endif

    fReal f6 = CrGlobal * invDelta * (uAir - u1);
    
    // output
    fReal result = (u1 + timeStepGlobal * (f2 + f3 + f4 + f5 + CrGlobal * uAir))
	/ (1.0 + (CrGlobal * invDelta + vTheta * cotTheta) * timeStepGlobal);
    velPhiOutput[thetaId * pitch + phiId] = result;
}


__global__ void resetThickness(float2* weight) {
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int idx = blockIdx.x * blockDim.x + phiId;

    weight[idx].x = 0.;
    weight[idx].y = 0.;
}

__global__ void applyforceThickness
(fReal* thicknessOutput, fReal* thicknessInput, fReal* div, size_t pitch)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    fReal eta = thicknessInput[thetaId * pitch + phiId];
    fReal f = div[thetaId * nPhiGlobal + phiId];

    thicknessOutput[thetaId * pitch + phiId] = eta * (1 - timeStepGlobal * f);
}

__global__ void applyforceParticles
(fReal* tempVal, fReal* value, fReal* coord, fReal* div, size_t numOfParticles) {
    int particleId = blockIdx.x * blockDim.x + threadIdx.x;

    if (particleId < numOfParticles) {
	fReal phiId = coord[2 * particleId];
	fReal thetaId = coord[2 * particleId + 1];

	fReal f = sampleCentered(div, phiId, thetaId, nPhiGlobal);

	tempVal[particleId] = value[particleId] * (1 - timeStepGlobal * f);
    }
}


// Backward Euler
__global__ void applyforceSurfConcentration
(fReal* sConcentrationOutput, fReal* sConcentrationInput, fReal* div, size_t pitch)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    fReal thetaCoord = ((fReal)thetaId + centeredThetaOffset) * gridLenGlobal;
    
    fReal halfStep = 0.5 * gridLenGlobal;

    fReal cscTheta = 1.f / sinf(thetaCoord);
    fReal sinThetaSouth = sinf(thetaCoord + halfStep);
    fReal sinThetaNorth = sinf(thetaCoord - halfStep);

    fReal gamma = sConcentrationInput[thetaId * pitch + phiId];
    fReal gammaWest = sConcentrationInput[thetaId * pitch + (phiId - 1 + nPhiGlobal) % nPhiGlobal];
    fReal gammaEast = sConcentrationInput[thetaId * pitch + (phiId + 1) % nPhiGlobal];
    fReal gammaNorth = 0.0;
    fReal gammaSouth = 0.0;
    if (thetaId != 0) {
    	gammaNorth = sConcentrationInput[(thetaId - 1) * pitch + phiId];
    } else {
    	size_t oppositePhiId = (phiId + nPhiGlobal / 2) % nPhiGlobal;
    	gammaNorth = sConcentrationInput[thetaId * pitch + oppositePhiId];
    }
    if (thetaId != nThetaGlobal - 1) {
    	gammaSouth = sConcentrationInput[(thetaId + 1) * pitch + phiId];
    } else {
    	size_t oppositePhiId = (phiId + nPhiGlobal / 2) % nPhiGlobal;
    	gammaSouth = sConcentrationInput[thetaId * pitch + oppositePhiId];
    }
# ifdef sphere
    fReal laplace = invGridLenGlobal * invGridLenGlobal * cscTheta *
	(sinThetaSouth * (gammaSouth - gamma) - sinThetaNorth * (gamma - gammaNorth) +
		    cscTheta * (gammaEast + gammaWest - 2 * gamma));
# else
    fReal laplace = invGridLenGlobal * invGridLenGlobal * 
    	(gammaWest - 4*gamma + gammaEast + gammaNorth + gammaSouth);
#endif
    
    fReal f = div[thetaId * nPhiGlobal + phiId];
    // fReal f2 = DsGlobal * laplace;
    fReal f2 = 0.f;

    sConcentrationOutput[thetaId * pitch + phiId] = max((gamma + f2 * timeStepGlobal) / (1 + timeStepGlobal * f), 0.f);
}


void KaminoSolver::bodyforce()
{
    dim3 gridLayout;
    dim3 blockLayout;

    bool inviscid = true;
    
    // div(u^n)
    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    divergenceKernel<<<gridLayout, blockLayout>>>
	(div, velPhi->getGPUThisStep(), velTheta->getGPUThisStep(),
	 velPhi->getThisStepPitchInElements(), velTheta->getThisStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    concentrationLinearSystemKernel<<<gridLayout, blockLayout>>>
    	(div, surfConcentration->getGPUThisStep(), thickness->getGPUThisStep(), val, rhs, thickness->getThisStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

# ifdef PERFORMANCE_BENCHMARK
    KaminoTimer CGtimer;
    CGtimer.startTimer();
# endif
    conjugateGradient();
# ifdef PERFORMANCE_BENCHMARK
    this->CGTime += CGtimer.stopTimer() * 0.001f;
# endif  

    applyforcevelthetaKernel<<<gridLayout, blockLayout>>>
    	(velTheta->getGPUNextStep(), velTheta->getGPUThisStep(), thickness->getGPUThisStep(), surfConcentration->getGPUNextStep(), velTheta->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());

    applyforcevelphiKernel<<<gridLayout, blockLayout>>>
    	(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), thickness->getGPUThisStep(), surfConcentration->getGPUNextStep(), velPhi->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // div(u^{n+1})
    divergenceKernel<<<gridLayout, blockLayout>>>
    	(div, velPhi->getGPUNextStep(), velTheta->getGPUNextStep(),
    	 velPhi->getNextStepPitchInElements(), velTheta->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    if (particles->numOfParticles > 0) {
    	determineLayout(gridLayout, blockLayout, 1, particles->numOfParticles);
    	applyforceParticles<<<gridLayout, blockLayout>>>
    	    (particles->tempVal, particles->value, particles->coordGPUThisStep, div, particles->numOfParticles);
    	checkCudaErrors(cudaGetLastError());
    	checkCudaErrors(cudaDeviceSynchronize());

    	determineLayout(gridLayout, blockLayout, 2, particles->numOfParticles);
    	mapParticlesToThickness<<<gridLayout, blockLayout>>>
    	    (particles->coordGPUThisStep, particles->tempVal,  weight, particles->numOfParticles);
    	checkCudaErrors(cudaGetLastError());
    	checkCudaErrors(cudaDeviceSynchronize());
    }
	
    determineLayout(gridLayout, blockLayout, thickness->getNTheta(), thickness->getNPhi());
    normalizeThickness_f<<<gridLayout, blockLayout>>>
    	(thickness->getGPUNextStep(), thickness->getGPUThisStep(), div, weight, thickness->getNextStepPitchInElements());

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());       
     
    std::swap(particles->tempVal, particles->value);
    thickness->swapGPUBuffer();
    surfConcentration->swapGPUBuffer();
    swapVelocityBuffers();
}


Kamino::Kamino(fReal radius, fReal H, fReal U, fReal c_m, fReal Gamma_m,
	       fReal T, fReal Ds, fReal rm, size_t nTheta, 
	       float dt, float DT, int frames, fReal A, int B, int C, int D, int E,
	       std::string thicknessPath, std::string velocityPath,
	       std::string thicknessImage, size_t particleDensity, int device):
    radius(radius), invRadius(1/radius), H(H), U(U), c_m(c_m), Gamma_m(Gamma_m), T(T),
    Ds(Ds/(U*radius)), gs(g*radius/(U*U)), rm(rm), epsilon(H/radius), sigma_r(R*T), M(Gamma_m*sigma_r/(3*rho*H*U*U)),
    S(sigma_a*epsilon/(2*mu*U)), re(mu/(U*radius*rho)), Cr(rhoa*sqrt(nua*radius/U)/(rho*H)),
    nTheta(nTheta), nPhi(2 * nTheta),
    gridLen(M_PI / nTheta), invGridLen(nTheta / M_PI), 
    dt(dt*U/radius), DT(DT), frames(frames),
    A(A), B(B), C(C), D(D), E(E),
    thicknessPath(thicknessPath), velocityPath(velocityPath),
    thicknessImage(thicknessImage), particleDensity(particleDensity), device(device)
{
    std::cout << "Re^-1 " << re << std::endl;
    std::cout << "S " << S << std::endl;
    std::cout << "Cr " << Cr << std::endl;
    std::cout << "M " << M << std::endl;
}
Kamino::~Kamino()
{}

void Kamino::run()
{
    KaminoSolver solver(nPhi, nTheta, radius, dt, A, B, C, D, E, H, device);
    
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
    solver.write_thickness_img(thicknessPath, 0);
# endif  
# ifdef WRITE_VELOCITY_DATA
    solver.write_velocity_image(velocityPath, 0);
    size_t split = velocityPath.find("/");
    std::string concentrationPath = velocityPath;
    concentrationPath.replace(concentrationPath.begin() + split + 1, concentrationPath.end(), "con");
    solver.write_concentration_image(concentrationPath, 0);
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
	    solver.stepForward(i*DT - T);
	}
	if (solver.isBroken()) {
	    std::cerr << "Film is broken." << std::endl;
	    break;
	}
	T = i*DT;

	std::cout << "Frame " << i << " is ready" << std::endl;
# ifdef WRITE_THICKNESS_DATA
	solver.write_thickness_img(thicknessPath, i);
# endif
# ifdef WRITE_VELOCITY_DATA
	solver.write_velocity_image(velocityPath, i);
	solver.write_concentration_image(concentrationPath, i);
# endif
    }

# ifdef PERFORMANCE_BENCHMARK
    float gpu_time = timer.stopTimer();
# endif

    std::cout << "Time spent: " << gpu_time << "ms" << std::endl;
    std::cout << "Performance: " << 1000.0 * i / gpu_time << " frames per second" << std::endl;
}
