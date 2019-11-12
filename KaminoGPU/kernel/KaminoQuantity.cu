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
    checkCudaErrors(cudaMallocPitch(&gpuThisStep, &thisStepPitch, nPhi * sizeof(float), nTheta));
    checkCudaErrors(cudaMallocPitch(&gpuNextStep, &nextStepPitch, nPhi * sizeof(float), nTheta));
    checkCudaErrors(cudaMemset(gpuThisStep, 0, thisStepPitch * nTheta));
    checkCudaErrors(cudaMemset(gpuNextStep, 0, nextStepPitch * nTheta));
}


BimocqQuantity::BimocqQuantity(std::string attributeName, size_t nPhi, size_t nTheta,
			       float phiOffset, float thetaOffset)
    : KaminoQuantity(attributeName, nPhi, nTheta, phiOffset, thetaOffset) {
    checkCudaErrors(cudaMallocPitch(&gpuInit, &thisStepPitch, nPhi * sizeof(float), nTheta));
    checkCudaErrors(cudaMallocPitch(&gpuDelta, &thisStepPitch, nPhi * sizeof(float), nTheta));
    checkCudaErrors(cudaMallocPitch(&gpuInitLast, &thisStepPitch, nPhi * sizeof(float), nTheta));
    checkCudaErrors(cudaMallocPitch(&gpuDeltaLast, &thisStepPitch, nPhi * sizeof(float), nTheta));
    checkCudaErrors(cudaMemset(gpuInit, 0, thisStepPitch * nTheta));
    checkCudaErrors(cudaMemset(gpuDelta, 0, thisStepPitch * nTheta));
    checkCudaErrors(cudaMemset(gpuInitLast, 0, thisStepPitch * nTheta));
    checkCudaErrors(cudaMemset(gpuDeltaLast, 0, thisStepPitch * nTheta));
}


KaminoQuantity::~KaminoQuantity() {
    delete[] cpuBuffer;

    checkCudaErrors(cudaFree(gpuThisStep));
    checkCudaErrors(cudaFree(gpuNextStep));
}


BimocqQuantity::~BimocqQuantity() {
    checkCudaErrors(cudaFree(gpuInit));
    checkCudaErrors(cudaFree(gpuDelta));
    checkCudaErrors(cudaFree(gpuInitLast));
    checkCudaErrors(cudaFree(gpuDeltaLast));
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


float*& BimocqQuantity::getGPUInit() {
    return this->gpuInit;
}


float*& BimocqQuantity::getGPUDelta() {
    return this->gpuDelta;
}


float*& BimocqQuantity::getGPUInitLast() {
    return this->gpuInitLast;
}


float*& BimocqQuantity::getGPUDeltaLast() {
    return this->gpuDeltaLast;
}


size_t KaminoQuantity::getThisStepPitchInElements() {
    return this->thisStepPitch / sizeof(float);
}


size_t KaminoQuantity::getNextStepPitchInElements() {
    return this->nextStepPitch / sizeof(float);
}