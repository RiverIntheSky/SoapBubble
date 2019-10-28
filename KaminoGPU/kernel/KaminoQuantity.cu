# include "../include/KaminoQuantity.cuh"

void KaminoQuantity::copyToGPU() {
    /* 
       Pitch : nPhi * sizeof(float)
       Width : nPhi * sizeof(float)
       Height: nTheta
    */
    checkCudaErrors(cudaMemcpy2D(gpuThisStep, thisStepPitch, cpuBuffer, 
				 nPhi * sizeof(float), nPhi * sizeof(float), nTheta, cudaMemcpyHostToDevice));
}


void KaminoQuantity::copyBackToCPU() {
    checkCudaErrors(cudaMemcpy2D((void*)this->cpuBuffer, nPhi * sizeof(float), (void*)this->gpuThisStep,
				 this->thisStepPitch, nPhi * sizeof(float), nTheta, cudaMemcpyDeviceToHost));
}


KaminoQuantity::KaminoQuantity(std::string attributeName, size_t nPhi, size_t nTheta,
			       float phiOffset, float thetaOffset)
    : nPhi(nPhi), nTheta(nTheta), gridLen(M_2PI / nPhi), invGridLen(1.0 / gridLen),
    attrName(attributeName), phiOffset(phiOffset), thetaOffset(thetaOffset) {
    cpuBuffer = new float[nPhi * nTheta];
    checkCudaErrors(cudaMallocPitch((void**)&gpuThisStep, &thisStepPitch, nPhi * sizeof(float), nTheta));
    checkCudaErrors(cudaMallocPitch((void**)&gpuNextStep, &nextStepPitch, nPhi * sizeof(float), nTheta));
}


ScalarQuantity::ScalarQuantity(std::string attributeName, size_t nPhi, size_t nTheta,
			       float phiOffset, float thetaOffset)
    : KaminoQuantity(attributeName, nPhi, nTheta, phiOffset, thetaOffset) {
    checkCudaErrors(cudaMallocPitch(&gpuInit, &thisStepPitch, nPhi * sizeof(float), nTheta));
    checkCudaErrors(cudaMallocPitch(&gpuDelta, &thisStepPitch, nPhi * sizeof(float), nTheta));
    checkCudaErrors(cudaMemset(gpuDelta, 0, thisStepPitch * nTheta));
}


KaminoQuantity::~KaminoQuantity() {
    delete[] cpuBuffer;

    checkCudaErrors(cudaFree(gpuThisStep));
    checkCudaErrors(cudaFree(gpuNextStep));
}


ScalarQuantity::~ScalarQuantity() {
    checkCudaErrors(cudaFree(gpuInit));
}


std::string KaminoQuantity::getName() {
    return this->attrName;
}


size_t KaminoQuantity::getNPhi() {
    return this->nPhi;
}


size_t KaminoQuantity::getNTheta() {
    return this->nTheta;
}


void KaminoQuantity::swapGPUBuffer() {
    float* tempPtr = this->gpuThisStep;
    this->gpuThisStep = this->gpuNextStep;
    this->gpuNextStep = tempPtr;
}


float KaminoQuantity::getCPUValueAt(size_t phi, size_t theta) {
    return this->accessCPUValueAt(phi, theta);
}


void KaminoQuantity::setCPUValueAt(size_t phi, size_t theta, float val) {
    this->accessCPUValueAt(phi, theta) = val;
}


float& KaminoQuantity::accessCPUValueAt(size_t phi, size_t theta) {
    assert(theta >= 0 && theta < nTheta && phi >= 0 && phi < nPhi);
    return this->cpuBuffer[theta * nPhi + phi];
}


float KaminoQuantity::getThetaOffset() {
    return this->thetaOffset;
}


float KaminoQuantity::getPhiOffset() {
    return this->phiOffset;
}


float* KaminoQuantity::getGPUThisStep() {
    return this->gpuThisStep;
}


float* KaminoQuantity::getGPUNextStep() {
    return this->gpuNextStep;
}


float* ScalarQuantity::getGPUInit() {
    return this->gpuInit;
}

float* ScalarQuantity::getGPUDelta() {
    return this->gpuDelta;
}


size_t KaminoQuantity::getThisStepPitchInElements() {
    return this->thisStepPitch / sizeof(float);
}


size_t KaminoQuantity::getNextStepPitchInElements() {
    return this->nextStepPitch / sizeof(float);
}