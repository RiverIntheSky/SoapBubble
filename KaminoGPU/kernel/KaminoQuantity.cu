# include "../include/KaminoQuantity.cuh"

void KaminoQuantity::copyToGPU() {
    /* 
       Pitch : nPhi * sizeof(fReal)
       Width : nPhi * sizeof(fReal)
       Height: nTheta
    */
    checkCudaErrors(cudaMemcpy2D(gpuThisStep, thisStepPitch, cpuBuffer, 
				 nPhi * sizeof(fReal), nPhi * sizeof(fReal), nTheta, cudaMemcpyHostToDevice));
}


void KaminoQuantity::copyBackToCPU() {
    checkCudaErrors(cudaMemcpy2D((void*)this->cpuBuffer, nPhi * sizeof(fReal), (void*)this->gpuThisStep,
				 this->thisStepPitch, nPhi * sizeof(fReal), nTheta, cudaMemcpyDeviceToHost));
}


KaminoQuantity::KaminoQuantity(std::string attributeName, size_t nPhi, size_t nTheta,
			       fReal phiOffset, fReal thetaOffset)
    : nPhi(nPhi), nTheta(nTheta), gridLen(M_2PI / nPhi), invGridLen(1.0 / gridLen),
    attrName(attributeName), phiOffset(phiOffset), thetaOffset(thetaOffset) {
    cpuBuffer = new fReal[nPhi * nTheta];
    checkCudaErrors(cudaMallocPitch(&gpuThisStep, &thisStepPitch, nPhi * sizeof(fReal), nTheta));
    checkCudaErrors(cudaMallocPitch(&gpuNextStep, &nextStepPitch, nPhi * sizeof(fReal), nTheta));
    checkCudaErrors(cudaMemset(gpuThisStep, 0, thisStepPitch * nTheta));
    checkCudaErrors(cudaMemset(gpuNextStep, 0, nextStepPitch * nTheta));
}


BimocqQuantity::BimocqQuantity(std::string attributeName, size_t nPhi, size_t nTheta,
			       fReal phiOffset, fReal thetaOffset)
    : KaminoQuantity(attributeName, nPhi, nTheta, phiOffset, thetaOffset) {
    checkCudaErrors(cudaMallocPitch(&gpuInit, &thisStepPitch, nPhi * sizeof(fReal), nTheta));
    checkCudaErrors(cudaMallocPitch(&gpuDelta, &thisStepPitch, nPhi * sizeof(fReal), nTheta));
    checkCudaErrors(cudaMallocPitch(&gpuInitLast, &thisStepPitch, nPhi * sizeof(fReal), nTheta));
    checkCudaErrors(cudaMallocPitch(&gpuDeltaLast, &thisStepPitch, nPhi * sizeof(fReal), nTheta));
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
    fReal* tempPtr = this->gpuThisStep;
    this->gpuThisStep = this->gpuNextStep;
    this->gpuNextStep = tempPtr;
}


fReal KaminoQuantity::getCPUValueAt(size_t phi, size_t theta) {
    return this->accessCPUValueAt(phi, theta);
}


void KaminoQuantity::setCPUValueAt(size_t phi, size_t theta, fReal val) {
    this->accessCPUValueAt(phi, theta) = val;
}


fReal& KaminoQuantity::accessCPUValueAt(size_t phi, size_t theta) {
    assert(theta < nTheta && phi < nPhi);
    return this->cpuBuffer[theta * nPhi + phi];
}


fReal KaminoQuantity::getThetaOffset() {
    return this->thetaOffset;
}


fReal KaminoQuantity::getPhiOffset() {
    return this->phiOffset;
}


fReal* KaminoQuantity::getGPUThisStep() {
    return this->gpuThisStep;
}


fReal* KaminoQuantity::getGPUNextStep() {
    return this->gpuNextStep;
}


fReal*& BimocqQuantity::getGPUInit() {
    return this->gpuInit;
}


fReal*& BimocqQuantity::getGPUDelta() {
    return this->gpuDelta;
}


fReal*& BimocqQuantity::getGPUInitLast() {
    return this->gpuInitLast;
}


fReal*& BimocqQuantity::getGPUDeltaLast() {
    return this->gpuDeltaLast;
}


size_t KaminoQuantity::getThisStepPitchInElements() {
    return this->thisStepPitch / sizeof(fReal);
}


size_t KaminoQuantity::getNextStepPitchInElements() {
    return this->nextStepPitch / sizeof(fReal);
}