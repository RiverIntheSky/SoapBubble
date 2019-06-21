# include "KaminoSolver.cuh"
# include "../include/KaminoGPU.cuh"
# include "../include/KaminoTimer.cuh"

static __constant__ size_t nPhiGlobal;
static __constant__ size_t nThetaGlobal;
static __constant__ fReal invRadiusGlobal;
static __constant__ fReal radiusGlobal;
static __constant__ fReal timeStepGlobal;
static __constant__ fReal gridLenGlobal;
static __constant__ fReal invGridLenGlobal;
static __constant__ fReal SGlobal;
static __constant__ fReal MGlobal;
static __constant__ fReal reGlobal;
static __constant__ fReal gGlobal;

__device__ bool validateCoord(fReal& phi, fReal& theta) {
    bool ret = false;
    theta = theta - static_cast<int>(floorf(theta / M_2PI)) * M_2PI;
    if (theta > M_PI) {
	theta = M_2PI - theta;
	phi += M_PI;
	ret = !ret;
    }
    if (theta < 0) {
	theta = -theta;
	phi += M_PI;
	ret = !ret;
    }
    phi = phi - static_cast<int>(floorf(phi / M_2PI)) * M_2PI;
    return ret;
}

__device__ fReal kaminoLerp(fReal from, fReal to, fReal alpha)
{
    return (1.0 - alpha) * from + alpha * to;
}

__device__ fReal sampleVPhi(fReal* input, fReal phiRaw, fReal thetaRaw, size_t pitch)
{
    fReal phi = phiRaw - gridLenGlobal * vPhiPhiOffset;
    fReal theta = thetaRaw - gridLenGlobal * vPhiThetaOffset;
    // Phi and Theta are now shifted back to origin

    fReal invGridLen = 1.0 / gridLenGlobal;
    bool isFlippedPole = validateCoord(phi, theta);
    fReal normedPhi = phi * invGridLen;
    fReal normedTheta = theta * invGridLen;

    int phiIndex = static_cast<int>(floorf(normedPhi));
    int thetaIndex = static_cast<int>(floorf(normedTheta));
    fReal alphaPhi = normedPhi - static_cast<fReal>(phiIndex);
    fReal alphaTheta = normedTheta - static_cast<fReal>(thetaIndex);

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
    if (thetaIndex == nThetaGlobal - 1)	{
	size_t phiLower = (phiIndex) % nPhiGlobal;
	size_t phiHigher = (phiLower + 1) % nPhiGlobal;
	fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);

	phiLower = (phiIndex + nPhiGlobal / 2) % nPhiGlobal;
	phiHigher = (phiLower + 1) % nPhiGlobal;

	fReal higherBelt = -kaminoLerp(input[phiLower + pitch * thetaIndex],
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

__device__ fReal sampleVTheta(fReal* input, fReal phiRaw, fReal thetaRaw, size_t pitch)
{
    fReal phi = phiRaw - gridLenGlobal * vThetaPhiOffset;
    fReal theta = thetaRaw - gridLenGlobal * vThetaThetaOffset;
    // Phi and Theta are now shifted back to origin

    fReal invGridLen = 1.0 / gridLenGlobal;
    bool isFlippedPole = validateCoord(phi, theta);
    fReal normedPhi = phi * invGridLen;
    fReal normedTheta = theta * invGridLen;

    int phiIndex = static_cast<int>(floorf(normedPhi));
    int thetaIndex = static_cast<int>(floorf(normedTheta));
    fReal alphaPhi = normedPhi - static_cast<fReal>(phiIndex);
    fReal alphaTheta = normedTheta - static_cast<fReal>(thetaIndex);

    if (thetaIndex == 0 && isFlippedPole) {
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

__device__ fReal sampleCentered(fReal* input, fReal phiRaw, fReal thetaRaw, size_t pitch)
{
    fReal phi = phiRaw - gridLenGlobal * centeredPhiOffset;
    fReal theta = thetaRaw - gridLenGlobal * centeredThetaOffset;
    // Phi and Theta are now shifted back to origin

    fReal invGridLen = 1.0 / gridLenGlobal;
    bool isFlippedPole = validateCoord(phi, theta);
    fReal normedPhi = phi * invGridLen;
    fReal normedTheta = theta * invGridLen;

    int phiIndex = static_cast<int>(floorf(normedPhi));
    int thetaIndex = static_cast<int>(floorf(normedTheta));
    fReal alphaPhi = normedPhi - static_cast<fReal>(phiIndex);
    fReal alphaTheta = normedTheta - static_cast<fReal>(thetaIndex);
    
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

__global__ void advectionVPhiKernel
(fReal* attributeOutput, fReal* velPhi, fReal* velTheta, size_t pitch)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    // Coord in phi-theta space
    fReal gPhi = ((fReal)phiId + vPhiPhiOffset) * gridLenGlobal;
    fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLenGlobal;
	
    // Sample the speed
    // fReal guTheta = sampleVTheta(velTheta, gPhi, gTheta, pitch);
    // fReal guPhi = sampleVPhi(velPhi, gPhi, gTheta, pitch);
    int thetaNorthId = thetaId - 1;
    int phiWestId = (phiId - 1 + nPhiGlobal) % nPhiGlobal;
    fReal v0 = 0.0;
    fReal v1 = 0.0;
    fReal v2 = 0.0;
    fReal v3 = 0.0;
    if (thetaId != 0) {
    	v0 = velTheta[thetaNorthId * pitch + phiWestId];
    	v1 = velTheta[thetaNorthId * pitch + phiId];
    } else {
    	v0 = velTheta[thetaId * pitch + (phiWestId + nPhiGlobal / 2) % nPhiGlobal];
    	v1 = velTheta[thetaId * pitch + (phiId + nPhiGlobal / 2) % nPhiGlobal];
    }
    if (thetaId != nThetaGlobal - 1) {
    	v2 = velTheta[thetaId * pitch + phiWestId];
    	v3 = velTheta[thetaId * pitch + phiId];
    } else {
    	v2 = velTheta[thetaNorthId * pitch + (phiWestId + nPhiGlobal / 2) % nPhiGlobal];
    	v3 = velTheta[thetaNorthId * pitch + (phiId + nPhiGlobal / 2) % nPhiGlobal];
    }
    fReal guTheta = 0.0;
    if (thetaId == 0) {	
    	guTheta = -0.125 * (v0 + v1) + 0.375 * (v2 + v3);
    } else if (thetaId == nThetaGlobal - 1) {
    	guTheta = 0.375 * (v0 + v1) - 0.125 * (v2 + v3);
    } else {
     	guTheta = 0.25 * (v0 + v1 + v2 + v3);
    }
    fReal guPhi = velPhi[thetaId * pitch + phiId];
       
    fReal cofTheta = timeStepGlobal * invRadiusGlobal;
    fReal cofPhi = cofTheta / sinf(gTheta);
    
    fReal deltaPhi = guPhi * cofPhi;
    fReal deltaTheta = guTheta * cofTheta;

# ifdef RUNGE_KUTTA
    // Traced halfway in phi-theta space
    fReal midPhi = gPhi - 0.5 * deltaPhi;
    fReal midTheta = gTheta - 0.5 * deltaTheta;

    fReal muPhi = sampleVPhi(velPhi, midPhi, midTheta, pitch);
    fReal muTheta = sampleVTheta(velTheta, midPhi, midTheta, pitch);

    fReal averuPhi = 0.5 * (muPhi + guPhi);
    fReal averuTheta = 0.5 * (muTheta + guTheta);

    deltaPhi = averuPhi * cofPhi;
    deltaTheta = averuTheta * cofTheta;
# endif

    fReal pPhi = gPhi - deltaPhi;
    fReal pTheta = gTheta - deltaTheta;

    fReal advectedVal = sampleVPhi(velPhi, pPhi, pTheta, pitch);

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
    fReal gPhi = ((fReal)phiId + vThetaPhiOffset) * gridLenGlobal;
    fReal gTheta = ((fReal)thetaId + vThetaThetaOffset) * gridLenGlobal;

    // Sample the speed
    // fReal guPhi = sampleVPhi(velPhi, gPhi, gTheta, pitch);
    // fReal guTheta = sampleVTheta(velTheta, gPhi, gTheta, pitch);  
    int thetaSouthId = thetaId + 1;
    int phiEastId = (phiId + 1) % nPhiGlobal;
    fReal u0 = velPhi[thetaId * pitch + phiId];
    fReal u1 = velPhi[thetaId * pitch + phiEastId];
    fReal u2 = velPhi[thetaSouthId * pitch + phiId];
    fReal u3 = velPhi[thetaSouthId * pitch + phiEastId];
    fReal guPhi = 0.25 * (u0 + u1 + u2 + u3);
    fReal guTheta = velTheta[thetaId * pitch + phiId];
    
    fReal cofTheta = timeStepGlobal * invRadiusGlobal;
    fReal cofPhi = cofTheta / sinf(gTheta);
    
    fReal deltaPhi = guPhi * cofPhi;
    fReal deltaTheta = guTheta * cofTheta;

# ifdef RUNGE_KUTTA
    // Traced halfway in phi-theta space
    fReal midPhi = gPhi - 0.5 * deltaPhi;
    fReal midTheta = gTheta - 0.5 * deltaTheta;
    fReal muPhi = sampleVPhi(velPhi, midPhi, midTheta, pitch);
    fReal muTheta = sampleVTheta(velTheta, midPhi, midTheta, pitch);

    fReal averuPhi = 0.5 * (muPhi + guPhi);
    fReal averuTheta = 0.5 * (muTheta + guTheta);

    deltaPhi = averuPhi * cofPhi;
    deltaTheta = averuTheta * cofTheta;
# endif

    fReal pPhi = gPhi - deltaPhi;
    fReal pTheta = gTheta - deltaTheta;

    fReal advectedVal = sampleVTheta(velTheta, pPhi, pTheta, pitch);

    attributeOutput[thetaId * pitch + phiId] = advectedVal;
}

__global__ void advectionCentered
(fReal* attributeOutput, fReal* velPhi, fReal* velTheta, fReal* attributeInput, size_t nPitchInElements)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    // Coord in phi-theta space
    fReal gPhi = ((fReal)phiId + centeredPhiOffset) * gridLenGlobal;
    fReal gTheta = ((fReal)thetaId + centeredThetaOffset) * gridLenGlobal;

    // Sample the speed
    fReal guPhi = sampleVPhi(velPhi, gPhi, gTheta, nPitchInElements);
    fReal guTheta = sampleVTheta(velTheta, gPhi, gTheta, nPitchInElements);

    fReal latRadius = radiusGlobal * sinf(gTheta);
    fReal cofPhi = timeStepGlobal / latRadius;
    fReal cofTheta = timeStepGlobal / radiusGlobal;

    fReal deltaPhi = guPhi * cofPhi;
    fReal deltaTheta = guTheta * cofTheta;

# ifdef RUNGE_KUTTA
    // Traced halfway in phi-theta space
    fReal midPhi = gPhi - 0.5 * deltaPhi;
    fReal midTheta = gTheta - 0.5 * deltaTheta;
    fReal muPhi = sampleVPhi(velPhi, midPhi, midTheta, nPitchInElements);
    fReal muTheta = sampleVTheta(velTheta, midPhi, midTheta, nPitchInElements);

    fReal averuPhi = 0.5 * (muPhi + guPhi);
    fReal averuTheta = 0.5 * (muTheta + guTheta);

    deltaPhi = averuPhi * cofPhi;
    deltaTheta = averuTheta * cofTheta;
# endif

    fReal pPhi = gPhi - deltaPhi;
    fReal pTheta = gTheta - deltaTheta;

    fReal advectedVal = sampleCentered(attributeInput, pPhi, pTheta, nPitchInElements);

    attributeOutput[thetaId * nPitchInElements + phiId] = advectedVal;
};

__global__ void advectionAllCentered
(fReal* thicknessOutput, fReal* bulkOutput, fReal* surfOutput, fReal* thicknessInput, fReal* bulkInput, fReal* surfInput, fReal* velPhi, fReal* velTheta, size_t nPitchInElements)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    // Coord in phi-theta space
    fReal gPhi = ((fReal)phiId + centeredPhiOffset) * gridLenGlobal;
    fReal gTheta = ((fReal)thetaId + centeredThetaOffset) * gridLenGlobal;

    // Sample the speed
    // fReal guPhi = sampleVPhi(velPhi, gPhi, gTheta, nPitchInElements);
    // fReal guTheta = sampleVTheta(velTheta, gPhi, gTheta, nPitchInElements);
    fReal guPhi = 0.5 * (velPhi[thetaId * nPitchInElements + phiId] +
    			 velPhi[thetaId * nPitchInElements + (phiId + 1) % nPhiGlobal]);
    fReal guTheta;
    if (thetaId == 0) {
    	guTheta = 0.75 * velTheta[phiId] -
    	    0.25 * velTheta[(phiId + nPhiGlobal / 2) % nPhiGlobal];
    } else if (thetaId == nThetaGlobal - 1) {
    	int thetaIdNorth = thetaId - 1;
    	guTheta = 0.75 * velTheta[thetaIdNorth * nPitchInElements + phiId] -
    	    0.25 * velTheta[thetaIdNorth * nPitchInElements + (phiId + nPhiGlobal / 2) % nPhiGlobal];
    } else {
    	guTheta = 0.5 * (velTheta[(thetaId - 1) * nPitchInElements + phiId] +
    			 velTheta[thetaId * nPitchInElements + phiId]);
    }

    fReal cofTheta = timeStepGlobal * invRadiusGlobal;
    fReal cofPhi = cofTheta / sinf(gTheta);

    fReal deltaPhi = guPhi * cofPhi;
    fReal deltaTheta = guTheta * cofTheta;

# ifdef RUNGE_KUTTA
    // Traced halfway in phi-theta space
    fReal midPhi = gPhi - 0.5 * deltaPhi;
    fReal midTheta = gTheta - 0.5 * deltaTheta;
    fReal muPhi = sampleVPhi(velPhi, midPhi, midTheta, nPitchInElements);
    fReal muTheta = sampleVTheta(velTheta, midPhi, midTheta, nPitchInElements);

    fReal averuPhi = 0.5 * (muPhi + guPhi);
    fReal averuTheta = 0.5 * (muTheta + guTheta);

    deltaPhi = averuPhi * cofPhi;
    deltaTheta = averuTheta * cofTheta;
# endif

    fReal pPhi = gPhi - deltaPhi;
    fReal pTheta = gTheta - deltaTheta;

    fReal advectedThickness = sampleCentered(thicknessInput, pPhi, pTheta, nPitchInElements);
    fReal advectedBulk = sampleCentered(bulkInput, pPhi, pTheta, nPitchInElements);
    fReal advectedSurf = sampleCentered(surfInput, pPhi, pTheta, nPitchInElements);

    thicknessOutput[thetaId * nPitchInElements + phiId] = advectedThickness;
    bulkOutput[thetaId * nPitchInElements + phiId] = advectedBulk;
    surfOutput[thetaId * nPitchInElements + phiId] = advectedSurf;
};

__global__ void advectionParticles(fReal* output, fReal* velPhi, fReal* velTheta, fReal* input, size_t nPitch)
{
    int particleId = blockIdx.x * blockDim.x + threadIdx.x;
    fReal posPhi = input[2 * particleId];
    fReal posTheta = input[2 * particleId + 1];

    fReal uPhi = sampleVPhi(velPhi, posPhi, posTheta, nPitch);
    fReal uTheta = sampleVTheta(velTheta, posPhi, posTheta, nPitch);

    fReal latRadius = radiusGlobal * sinf(posTheta);
    fReal cofPhi = timeStepGlobal / latRadius;
    fReal cofTheta = timeStepGlobal / radiusGlobal;

    fReal updatedTheta = posTheta + uTheta * cofTheta;
    fReal updatedPhi = posPhi;
    if (latRadius > 1e-7f)
	updatedPhi = posPhi + uPhi * cofPhi;
    validateCoord(updatedPhi, updatedTheta);

    output[2 * particleId] = updatedPhi;
    output[2 * particleId + 1] = updatedTheta;
}

void KaminoSolver::advection()
{
    ///kernel call goes here
    // Advect Phi
    dim3 gridLayout;
    dim3 blockLayout;
    determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());
    advectionVPhiKernel<<<gridLayout, blockLayout>>>
	(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), velPhi->getNextStepPitchInElements());    
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Advect Theta
    determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
    advectionVThetaKernel<<<gridLayout, blockLayout>>>
	(velTheta->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), velTheta->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Advect thickness and concentration
    determineLayout(gridLayout, blockLayout, thickness->getNTheta(), thickness->getNPhi());
    advectionAllCentered<<<gridLayout, blockLayout>>>
	(thickness->getGPUNextStep(), bulkConcentration->getGPUNextStep(), surfConcentration->getGPUNextStep(),
	 thickness->getGPUThisStep(), bulkConcentration->getGPUThisStep(), surfConcentration->getGPUThisStep(),
	 velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), thickness->getNextStepPitchInElements());	
    thickness->swapGPUBuffer();
    surfConcentration->swapGPUBuffer();
    // bulkConcentration->swapGPUBuffer(); // no bulk concentration yet
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    swapVelocityBuffers();
}

__device__ fReal _root3(fReal x)
{
    fReal s = 1.;
    while (x < 1.)
	{
	    x *= 8.;
	    s *= 0.5;
	}
    while (x > 8.)
	{
	    x *= 0.125;
	    s *= 2.;
	}
    fReal r = 1.5;
    r -= 1. / 3. * (r - x / (r * r));
    r -= 1. / 3. * (r - x / (r * r));
    r -= 1. / 3. * (r - x / (r * r));
    r -= 1. / 3. * (r - x / (r * r));
    r -= 1. / 3. * (r - x / (r * r));
    r -= 1. / 3. * (r - x / (r * r));
    return r * s;
}

__device__ fReal root3(double x)
{
    if (x > 0)
	return _root3(x);
    else if (x < 0)
	return -_root3(-x);
    else
	return 0.0;
}

#define eps 1e-12f

__device__ fReal solveCubic(fReal a, fReal b, fReal c)
{
    fReal a2 = a * a;
    fReal q = (a2 - 3 * b) / 9.0;
    //q = q >= 0.0 ? q : -q;
    fReal r = (a * (2.0 * a2 - 9.0 * b) + 27.0 * c) / 54.0;

    fReal r2 = r * r;
    fReal q3 = q * q * q;
    fReal A, B;
    if (r2 <= (q3 + eps))
	{
	    double t = r / sqrtf(q3);
	    if (t < -1)
		t = -1;
	    if (t > 1)
		t = 1;
	    t = acosf(t);
	    a /= 3.0;
	    q = -2.0 * sqrtf(q);
	    return q * cosf(t / 3.0) - a;
	}
    else
	{
	    A = -root3(fabsf(r) + sqrtf(r2 - q3));
	    if (r < 0)
		A = -A;

	    B = A == 0 ? 0 : B = q / A;

	    a /= 3.0;
	    return (A + B) - a;
	}
}

//nTheta by nPhi
__global__ void geometricFillKernel
(fReal* intermediateOutputPhi, fReal* intermediateOutputTheta, fReal* velPhiInput, fReal* velThetaInput,
 size_t nPitchInElements)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int gridPhiId = threadIdx.x + threadSequence * blockDim.x;
    int gridThetaId = blockIdx.x / splitVal;
    // Coord in phi-theta space
    fReal gPhi = ((fReal)gridPhiId + centeredPhiOffset) * gridLenGlobal;
    fReal gTheta = ((fReal)gridThetaId + centeredThetaOffset) * gridLenGlobal;
    // The factor

    size_t phiLeft = gridPhiId;
    size_t phiRight = (gridPhiId + 1) % nPhiGlobal;
    // u = velPhi
    fReal uPrev = 0.5 * (velPhiInput[phiLeft + nPitchInElements * gridThetaId]
			 + velPhiInput[phiRight + nPitchInElements * gridThetaId]);
    fReal guPhi = sampleVPhi(velPhiInput, gPhi, gTheta, nPitchInElements);

    // v = velTheta	
    fReal vPrev;
    if (gridThetaId == 0)
	{
	    size_t oppositePhiIdx = (gridPhiId + nPhiGlobal / 2) % nPhiGlobal;
	    vPrev = 0.75 * velThetaInput[gridPhiId]
		+ 0.25 * velThetaInput[oppositePhiIdx];
	}
    else if (gridThetaId == nThetaGlobal - 1)
	{
	    size_t oppositePhiIdx = (gridPhiId + nPhiGlobal / 2) % nPhiGlobal;
	    vPrev = 0.75 * velThetaInput[gridPhiId + nPitchInElements * (gridThetaId - 1)]
		+ 0.25 * velThetaInput[oppositePhiIdx + nPitchInElements * (gridThetaId - 1)];
	}
    else
	{
	    vPrev = 0.5 * (velThetaInput[gridPhiId + nPitchInElements * (gridThetaId - 1)]
			   + velThetaInput[gridPhiId + nPitchInElements * gridThetaId]);
	}

	

    fReal G = timeStepGlobal * cosf(gTheta) / (radiusGlobal * sinf(gTheta));
    fReal uNext;
    if (abs(G) > eps)
	{
	    fReal cof = G * G;
	    fReal A = 0.0;
	    fReal B = (G * vPrev + 1.0) / cof;
	    fReal C = -uPrev / cof;

	    uNext = solveCubic(A, B, C); 
	}
    else
	{
	    uNext = uPrev;
	}

    fReal vNext = vPrev + G * uNext * uNext;

    intermediateOutputPhi[gridPhiId + nPitchInElements * gridThetaId] = uNext;
    intermediateOutputTheta[gridPhiId + nPitchInElements * gridThetaId] = vNext;
}

//nTheta by nPhi
__global__ void assignPhiKernel(fReal* velPhiOutput, fReal* intermediateInputPhi,
				size_t nPitchInElements)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    size_t phigridLeft;
    if (phiId == 0)
	phigridLeft = nPhiGlobal - 1;
    else
	phigridLeft = phiId - 1;
    velPhiOutput[phiId + nPitchInElements * thetaId] =
	0.5 * (intermediateInputPhi[phigridLeft + nPitchInElements * thetaId]
	       + intermediateInputPhi[phiId + nPitchInElements * thetaId]);
}

//nTheta - 1 by nPhi
__global__ void assignThetaKernel(fReal* velThetaOutput, fReal*intermediateInputTheta,
				  size_t nPitchInElements)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    velThetaOutput[phiId + nPitchInElements * thetaId] =
	0.5 * (intermediateInputTheta[phiId + nPitchInElements * thetaId]
	       + intermediateInputTheta[phiId + nPitchInElements * (thetaId + 1)]);
}

void KaminoSolver::geometric()
{
    dim3 gridLayout;
    dim3 blockLayout;
    //intermediate: pressure.this as phi, next as theta

    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    // does this has nothing with pressure to do, but only a place-holder?
    geometricFillKernel<<<gridLayout, blockLayout>>> 
	(pressure->getGPUThisStep(), pressure->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(),
	 pressure->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());



    determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());
    assignPhiKernel<<<gridLayout, blockLayout>>>
	(velPhi->getGPUNextStep(), pressure->getGPUThisStep(), velPhi->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());



    determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
    assignThetaKernel<<<gridLayout, blockLayout>>>
	(velTheta->getGPUNextStep(), pressure->getGPUNextStep(), velTheta->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());



    swapVelocityBuffers();
}

__global__ void crKernel(fReal *d_a, fReal *d_b, fReal *d_c, fReal *d_d, fReal *d_x);

// div(u) at cell center
__global__ void comDivergenceKernel
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
    } else {
	size_t oppositePhiId = (phiId + nPhiGlobal / 2) % nPhiGlobal;
	vNorth = 0.5 * (velTheta[thetaId * velThetaPitchInElements + phiId] -
			velTheta[thetaId * velThetaPitchInElements + oppositePhiId]);
    }
    if (thetaId != nThetaGlobal - 1) {
	size_t thetaSouthIdx = thetaId;
	vSouth = velTheta[thetaSouthIdx * velThetaPitchInElements + phiId];
    } else {
	size_t oppositePhiId = (phiId + nPhiGlobal / 2) % nPhiGlobal;
	vSouth = 0.5 * (velTheta[thetaId * velThetaPitchInElements + phiId] -
			velTheta[thetaId * velThetaPitchInElements + oppositePhiId]);
    }

    fReal invGridSine = 1.0 / sinf(thetaCoord);
    fReal sinNorth = sinf(thetaNorth);
    fReal sinSouth = sinf(thetaSouth);
    fReal factor = invGridSine * invRadiusGlobal * invGridLenGlobal;
    fReal termTheta = factor * (vSouth * sinSouth - vNorth * sinNorth);
    fReal termPhi = factor * (uEast - uWest);

    fReal f = termTheta + termPhi;

    div[thetaId * nPhiGlobal + phiId] = f;
}

__global__ void applyforcevelthetaKernel
(fReal* velThetaOutput, fReal* velThetaInput, fReal* velPhi, fReal* thickness,
 fReal* concentration, fReal* divCentered, size_t pitch) {
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    // Coord in phi-theta space
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
    fReal Delta = 0.5 * (thickness[thetaId * pitch + phiId] +
			 thickness[(thetaId + 1) * pitch + phiId]);
    fReal invDelta = 1. / Delta;
    fReal uPhi = 0.25 * (u0 + u1 + u2 + u3);

    // pDpx = \frac{\partial\Delta}{\partial\phi}
    fReal d0 = thickness[thetaNorthId * pitch + phiWestId];
    fReal d1 = thickness[thetaNorthId * pitch + phiEastId];
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
	size_t thetaSouthId = thetaId + 1;
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
    fReal pupxy = 0.25 * invGridLenGlobal * invGridLenGlobal * (u0 + u3 - u1 - u2);

    // pvpxx = \frac{\partial^2u_\theta}{\partial\phi^2}    
    fReal pvpxx = invGridLenGlobal * invGridLenGlobal * (v3 + v4 - 2 * v1);
    
    // trigonometric function
    fReal sinTheta = sinf(gTheta);
    fReal cscTheta = 1. / sinTheta;
    fReal cosTheta = cosf(gTheta);
    fReal cotTheta = cosTheta * cscTheta;

    // stress
    // TODO: laplace term
    fReal sigma11North = 0. +  2 * (invRadiusGlobal * pvpyNorth + divNorth);
    fReal sigma11South = 0. +  2 * (invRadiusGlobal * pvpySouth + divSouth);
    
    // fReal sigma11 = 0. +  2 * invRadiusGlobal * pvpy + 2 * div;
    fReal sigma22 = -2 * invRadiusGlobal * pvpy + 4 * div;
    fReal sigma12 = invRadiusGlobal * (cscTheta * pvpx + pupy - uPhi * cotTheta);

    // psspy = \frac{\partial}{\partial\theta}(\sin\theta\sigma_{11})
    fReal halfStep = 0.5 * gridLenGlobal;    
    fReal thetaSouth = gTheta + halfStep;
    fReal thetaNorth = gTheta - halfStep;    
    fReal sinNorth = sinf(thetaNorth);
    fReal sinSouth = sinf(thetaSouth);    
    fReal psspy = invGridLenGlobal * (sigma11South * sinSouth - sigma11North * sinNorth);
    
    // pspx = \frac{\partial\sigma_{12}}{\partial\phi}
    fReal pspx = invRadiusGlobal * (cscTheta * pvpxx + pupxy - cotTheta * pupx);

    // pGpy = \frac{\partial\Gamma}{\partial\theta};
    fReal pGpy = invGridLenGlobal * (GammaSouth - GammaNorth);

    // force terms
    fReal f1 = uPhi * uPhi * cotTheta * invRadiusGlobal;
    fReal f2 = reGlobal * invRadiusGlobal * cscTheta * invDelta * pDpx * sigma12;
    fReal f3 = -2 * MGlobal * invDelta * invRadiusGlobal * pGpy;
    fReal f4 = reGlobal * invDelta * invRadiusGlobal * pDpy * 2 * (div + invRadiusGlobal * pvpy);
    fReal f5 = reGlobal * invRadiusGlobal * cscTheta * (psspy + pspx - cosTheta * sigma22);
    fReal f6 = gGlobal * sinTheta;
    // fReal f7 = 0.0; 		// Van der Waals
    
    // output
    velThetaOutput[thetaId * pitch + phiId] = velThetaInput[thetaId * pitch + phiId]
	+ timeStepGlobal * (f1 + f2 + f3 + f4 + f5 + f6);
}

__global__ void applyforcevelphiKernel
(fReal* velPhiOutput, fReal* velTheta, fReal* velPhiInput, fReal* thickness,
 fReal* concentration, fReal* divCentered, size_t pitch) {
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    // Coord in phi-theta space
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
    fReal sigma12 = invRadiusGlobal * (cscTheta * pvpx + pupy - u1 * cotTheta);

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
    fReal psspy = invRadiusGlobal * invGridLenGlobal * (invGridLenGlobal * (v0 + v3 - v1 - v2) +
							sinSouth * pupySouth - sinNorth * pupyNorth -
							cosSouth * uSouth + cosNorth * uNorth);
    
    // pspx = \frac{\partial\sigma_{22}}{\partial\phi}
    fReal sigma22West = 2 * (2 * divWest - invRadiusGlobal * pvpyWest);
    fReal sigma22East = 2 * (2 * divEast - invRadiusGlobal * pvpyEast);
    fReal pspx = invGridLenGlobal * (sigma22East - sigma22West);
    
    // force terms
    fReal f1 = -vTheta * u1 * cotTheta * invRadiusGlobal;
    fReal f2 = reGlobal * invRadiusGlobal * invDelta * pDpy * sigma12;
    fReal f3 = -2 * MGlobal * invDelta * invRadiusGlobal * cscTheta * pGpx;
    fReal f4 = reGlobal * invDelta * invRadiusGlobal * cscTheta * pDpx * 2 * ( 2 * div - invRadiusGlobal * pvpy);
    fReal f5 = reGlobal * invRadiusGlobal * cscTheta * (psspy + pspx + cosTheta * sigma12);
    // fReal f6 = 0.0; 		// Van der Waals
    
    // output
    velPhiOutput[thetaId * pitch + phiId] = velPhiInput[thetaId * pitch + phiId]
	+ timeStepGlobal * (f1 + f2 + f3 + f4 + f5);
}

__global__ void applyforceThickness
(fReal* thicknessOutput, fReal* thicknessInput, fReal* div, size_t nPitchInElements)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    fReal f = -div[thetaId * nPhiGlobal + phiId];
    thicknessOutput[thetaId * nPhiGlobal + phiId] = f * timeStepGlobal * thicknessInput[thetaId * nPhiGlobal + phiId] 
	+ thicknessInput[thetaId * nPhiGlobal + phiId];
}

__global__ void applyforceSurfConcentration
(fReal* sConcentrationOutput, fReal* sConcentrationInput, fReal* div, size_t nPitchInElements)
{
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    // TODO: add diffusivity
    fReal f = -div[thetaId * nPhiGlobal + phiId];
    sConcentrationOutput[thetaId * nPhiGlobal + phiId] = f * timeStepGlobal * sConcentrationInput[thetaId * nPhiGlobal + phiId]
	+ sConcentrationInput[thetaId * nPhiGlobal + phiId];
}

void KaminoSolver::bodyforce()
{
    dim3 gridLayout;
    dim3 blockLayout;
    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    comDivergenceKernel<<<gridLayout, blockLayout>>>
	(div, velPhi->getGPUThisStep(), velTheta->getGPUThisStep(),
	 velPhi->getThisStepPitchInElements(), velTheta->getThisStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    determineLayout(gridLayout, blockLayout, thickness->getNTheta(), thickness->getNPhi());    
    applyforceThickness<<<gridLayout, blockLayout>>>
    	(thickness->getGPUNextStep(), thickness->getGPUThisStep(), div, thickness->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    determineLayout(gridLayout, blockLayout, surfConcentration->getNTheta(), surfConcentration->getNPhi());
    applyforceSurfConcentration<<<gridLayout, blockLayout>>>
    	(surfConcentration->getGPUNextStep(), surfConcentration->getGPUThisStep(), div, surfConcentration->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
    applyforcevelthetaKernel<<<gridLayout, blockLayout>>>
	(velTheta->getGPUNextStep(), velTheta->getGPUThisStep(), velPhi->getGPUThisStep(),
	 thickness->getGPUThisStep(), surfConcentration->getGPUThisStep(), div,
	 velTheta->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());
    applyforcevelphiKernel<<<gridLayout, blockLayout>>>
    	(velPhi->getGPUNextStep(), velTheta->getGPUThisStep(), velPhi->getGPUThisStep(),
	 thickness->getGPUThisStep(), surfConcentration->getGPUThisStep(), div,
	 velPhi->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    thickness->swapGPUBuffer();
    surfConcentration->swapGPUBuffer();
    swapVelocityBuffers();
}

__global__ void fillDivergenceKernel
(ComplexFourier* outputF, fReal* velPhi, fReal* velTheta,
 size_t velPhiPitchInElements, size_t velThetaPitchInElements)
{
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int gridPhiId = threadIdx.x + threadSequence * blockDim.x;
    int gridThetaId = blockIdx.x / splitVal;
    //fReal gridPhiCoord = ((fReal)gridPhiId + centeredPhiOffset) * gridLen;
    fReal gridThetaCoord = ((fReal)gridThetaId + centeredThetaOffset) * gridLenGlobal;

    fReal uEast = 0.0;
    fReal uWest = 0.0;
    fReal vNorth = 0.0;
    fReal vSouth = 0.0;

    fReal halfStep = 0.5 * gridLenGlobal;

    fReal thetaSouth = gridThetaCoord + halfStep;
    fReal thetaNorth = gridThetaCoord - halfStep;

    int phiIdWest = gridPhiId;
    int phiIdEast = (phiIdWest + 1) % nPhiGlobal;

    uWest = velPhi[gridThetaId * velPhiPitchInElements + phiIdWest];
    uEast = velPhi[gridThetaId * velPhiPitchInElements + phiIdEast];

    if (gridThetaId != 0)
	{
	    int thetaNorthIdx = gridThetaId - 1;
	    vNorth = velTheta[thetaNorthIdx * velThetaPitchInElements + gridPhiId];
	}
    if (gridThetaId != nThetaGlobal - 1)
	{
	    int thetaSouthIdx = gridThetaId;
	    vSouth = velTheta[thetaSouthIdx * velThetaPitchInElements + gridPhiId];
	}

    fReal invGridSine = 1.0 / sinf(gridThetaCoord);
    fReal sinNorth = sinf(thetaNorth);
    fReal sinSouth = sinf(thetaSouth);
    fReal factor = invGridSine / gridLenGlobal;
    fReal termTheta = factor * (vSouth * sinSouth - vNorth * sinNorth);
    fReal termPhi = factor * (uEast - uWest);

    fReal div = termTheta + termPhi;

    ComplexFourier f;
    f.x = div;
    f.y = 0.0;
    outputF[gridThetaId * nPhiGlobal + gridPhiId] = f;
}

__global__ void shiftFKernel
(ComplexFourier* FFourierInput, fReal* FFourierShiftedReal, fReal* FFourierShiftedImag)
{
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int nIdx = threadIdx.x + threadSequence * blockDim.x;
    int thetaIdx = blockIdx.x / splitVal;

    int fftIndex = nPhiGlobal / 2 - nIdx;
    if (fftIndex < 0)
	fftIndex += nPhiGlobal;
    //FFourierShifted[thetaIdx * nPhi + phiIdx] = FFourierInput[thetaIdx * nPhi + fftIndex];
    fReal real = FFourierInput[thetaIdx * nPhiGlobal + fftIndex].x / (fReal)nPhiGlobal;
    fReal imag = FFourierInput[thetaIdx * nPhiGlobal + fftIndex].y / (fReal)nPhiGlobal;
    FFourierShiftedReal[nIdx * nThetaGlobal + thetaIdx] = real;
    FFourierShiftedImag[nIdx * nThetaGlobal + thetaIdx] = imag;
}

__global__ void copy2UFourier
(ComplexFourier* UFourierOutput, fReal* UFourierReal, fReal* UFourierImag)
{
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int nIdx = threadIdx.x + threadSequence * blockDim.x;
    int thetaIdx = blockIdx.x / splitVal;

    ComplexFourier u;
    u.x = UFourierReal[nIdx * nThetaGlobal + thetaIdx];
    u.y = UFourierImag[nIdx * nThetaGlobal + thetaIdx];
    UFourierOutput[thetaIdx * nPhiGlobal + nIdx] = u;
}

__global__ void cacheZeroComponents
(fReal* zeroComponentCache, ComplexFourier* input)
{
    int splitVal = nThetaGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int thetaIdx = threadIdx.x + threadSequence * blockDim.x;

    zeroComponentCache[thetaIdx] = input[thetaIdx * nPhiGlobal + nPhiGlobal / 2].x;
}

__global__ void shiftUKernel
(ComplexFourier* UFourierInput, fReal* pressure, fReal* zeroComponentCache,
 size_t nPressurePitchInElements)
{
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiIdx = threadIdx.x + threadSequence * blockDim.x;
    int thetaIdx = blockIdx.x / splitVal;

    int fftIndex = 0;
    fReal zeroComponent = zeroComponentCache[thetaIdx];
    if (phiIdx != 0)
	fftIndex = nPhiGlobal - phiIdx;
    fReal pressureVal;

    if (phiIdx % 2 == 0)
	pressureVal = UFourierInput[thetaIdx * nPhiGlobal + fftIndex].x - zeroComponent;
    else
	pressureVal = -UFourierInput[thetaIdx * nPhiGlobal + fftIndex].x - zeroComponent;

    pressure[thetaIdx * nPressurePitchInElements + phiIdx] = pressureVal;
}

__global__ void applyPressureTheta
(fReal* output, fReal* prev, fReal* pressure,
 size_t nPitchInElementsPressure, size_t nPitchInElementsVTheta)
{
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    int pressureThetaNorthId = thetaId;
    int pressureThetaSouthId = thetaId + 1;
    fReal pressureNorth = pressure[pressureThetaNorthId * nPitchInElementsPressure + phiId];
    fReal pressureSouth = pressure[pressureThetaSouthId * nPitchInElementsPressure + phiId];

    fReal deltaVTheta = (pressureSouth - pressureNorth) / (-gridLenGlobal);
    fReal previousVTheta = prev[thetaId * nPitchInElementsVTheta + phiId];
    output[thetaId * nPitchInElementsVTheta + phiId] = previousVTheta + deltaVTheta;
}

__global__ void applyPressurePhi
(fReal* output, fReal* prev, fReal* pressure,
 size_t nPitchInElementsPressure, size_t nPitchInElementsVPhi)
{
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    int pressurePhiWestId;
    if (phiId == 0)
	pressurePhiWestId = nPhiGlobal - 1;
    else
	pressurePhiWestId = phiId - 1;
    int pressurePhiEastId = phiId;

    fReal pressureWest = pressure[thetaId * nPitchInElementsPressure + pressurePhiWestId];
    fReal pressureEast = pressure[thetaId * nPitchInElementsPressure + pressurePhiEastId];

    fReal thetaBelt = (thetaId + centeredThetaOffset) * gridLenGlobal;
    fReal deltaVPhi = (pressureEast - pressureWest) / (-gridLenGlobal * sinf(thetaBelt));
    fReal previousVPhi = prev[thetaId * nPitchInElementsVPhi + phiId];
    output[thetaId * nPitchInElementsVPhi + phiId] = previousVPhi + deltaVPhi;
}

void KaminoSolver::projection()
{
    dim3 gridLayout;
    dim3 blockLayout;
    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    fillDivergenceKernel<<<gridLayout, blockLayout>>>
	(this->gpuFFourier, this->velPhi->getGPUThisStep(), this->velTheta->getGPUThisStep(),
	 this->velPhi->getThisStepPitchInElements(), this->velTheta->getThisStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());



    // Note that cuFFT inverse returns results are SigLen times larger
    checkCudaErrors((cudaError_t)cufftExecC2C(this->kaminoPlan,
					      this->gpuFFourier, this->gpuFFourier, CUFFT_INVERSE));
    checkCudaErrors(cudaGetLastError());



    // Siglen is nPhi
    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    shiftFKernel<<<gridLayout, blockLayout>>>
	(gpuFFourier, gpuFReal, gpuFImag);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // Now gpuFDivergence stores all the Fn



    gridLayout = dim3(nPhi);
    blockLayout = dim3(nTheta / 2);
    const unsigned sharedMemSize = nTheta * 5 * sizeof(fReal);
    crKernel<<<gridLayout, blockLayout, sharedMemSize>>>
	(this->gpuA, this->gpuB, this->gpuC, this->gpuFReal, this->gpuUReal);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    gridLayout = dim3(nPhi);
    blockLayout = dim3(nTheta / 2);
    crKernel<<<gridLayout, blockLayout, sharedMemSize>>>
	(this->gpuA, this->gpuB, this->gpuC, this->gpuFImag, this->gpuUImag);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());



    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    copy2UFourier<<<gridLayout, blockLayout>>>
	(this->gpuUFourier, this->gpuUReal, this->gpuUImag);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());



    determineLayout(gridLayout, blockLayout, 1, nTheta);
    cacheZeroComponents<<<gridLayout, blockLayout>>>
	(gpuFZeroComponent, gpuUFourier);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());



    checkCudaErrors((cudaError_t)cufftExecC2C(this->kaminoPlan,
					      this->gpuUFourier, this->gpuUFourier, CUFFT_FORWARD));
    checkCudaErrors(cudaGetLastError());



    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    shiftUKernel<<<gridLayout, blockLayout>>>
	(gpuUFourier, pressure->getGPUThisStep(), this->gpuFZeroComponent,
	 pressure->getThisStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //pressure->copyBackToCPU();

    determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
    applyPressureTheta<<<gridLayout, blockLayout>>>
	(velTheta->getGPUNextStep(), velTheta->getGPUThisStep(), pressure->getGPUThisStep(),
	 pressure->getThisStepPitchInElements(), velTheta->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());
    applyPressurePhi<<<gridLayout, blockLayout>>>
	(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), pressure->getGPUThisStep(),
	 pressure->getThisStepPitchInElements(), velPhi->getNextStepPitchInElements());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    swapVelocityBuffers();
}

Kamino::Kamino(fReal radius, fReal H, fReal U, fReal c_m, fReal Gamma_m,
	       fReal sigma_a, fReal R, fReal T, fReal rho, fReal mu,
	       fReal Ds, fReal g,  fReal rm, size_t nTheta, fReal particleDensity,
	       float dt, float DT, int frames, fReal A, int B, int C, int D, int E,
	       std::string thicknessPath, std::string particlePath,
	       std::string thicknessImage, std::string solidImage, std::string colorImage) :
    radius(radius), invRadius(1/radius), H(H), U(U), c_m(c_m), Gamma_m(Gamma_m), sigma_a(sigma_a), R(R), T(T),
    rho(rho), mu(mu), Ds(Ds), g(g/(U*U)), rm(rm), epsilon(H), sigma_r(R*T),
    M(Gamma_m*R*T/(3*rho*H*U*U)), S(sigma_a*H/(2*mu*U)), re(mu/(rho*U)),
    nTheta(nTheta), nPhi(2 * nTheta),
    gridLen(M_PI / nTheta), invGridLen(nTheta / M_PI), particleDensity(particleDensity),
    dt(dt), DT(DT), frames(frames),
    A(A), B(B), C(C), D(D), E(E),
    thicknessPath(thicknessPath), particlePath(particlePath),
    thicknessImage(thicknessImage), solidImage(solidImage), colorImage(colorImage)
{
    std::cout << "Re^-1 " << re << std::endl;
    std::cout << "S " << S << std::endl;
    std::cout << "inv radius " << invRadius << std::endl;
}

Kamino::~Kamino()
{}

void Kamino::run()
{
    KaminoSolver solver(nPhi, nTheta, radius, dt, A, B, C, D, E, H);
    solver.initThicknessfromPic(thicknessImage);
# ifdef WRITE_PARTICLES
    solver.initParticlesfromPic(colorImage, this->particleDensity);
# endif

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
    checkCudaErrors(cudaMemcpyToSymbol(gGlobal, &(this->g), sizeof(fReal)));

# ifdef WRITE_THICKNESS_DATA
    solver.write_thickness_img(thicknessPath, 0);
# endif  
# ifdef WRITE_VELOCITY_DATA
    solver.write_data_bgeo(gridPath, 0);
# endif
# ifdef WRITE_PARTICLES
    solver.write_particles_bgeo(particlePath, 0);
# endif

# ifdef PERFORMANCE_BENCHMARK
    KaminoTimer timer;
    timer.startTimer();
# endif

    float T = 0.0;              // simulation time
    for (int i = 1; i < frames; i++)
	{
	    while (T < i*DT && !solver.isBroken())
		{
		    solver.stepForward(dt);
		    T += dt/this->U;
		}
	    if (solver.isBroken()) {
		std::cerr << "Film is broken." << std::endl;
		break;
	    }
		
	    solver.stepForward(dt/this->U + i*DT - T);
	    T = i*DT;

	    std::cout << "Frame " << i << " is ready" << std::endl;
# ifdef WRITE_THICKNESS_DATA
	    solver.write_thickness_img(thicknessPath, i);
# endif	   
	}

# ifdef PERFORMANCE_BENCHMARK
    float gpu_time = timer.stopTimer();
# endif

    std::cout << "Time spent: " << gpu_time << "ms" << std::endl;
    std::cout << "Performance: " << 1000.0 * frames / gpu_time << " frames per second" << std::endl;
}