# include "../include/KaminoQuantity.cuh"

void KaminoQuantity::copyToGPU()
{
    /* 
       Pitch : nPhi * sizeof(fReal)
       Width : nPhi * sizeof(fReal)
       Height: nTheta
    */
    checkCudaErrors(cudaMemcpy2D(gpuThisStep, thisStepPitch, cpuBuffer, 
				 nPhi * sizeof(fReal), nPhi * sizeof(fReal), nTheta, cudaMemcpyHostToDevice));
}

void KaminoQuantity::copyBackToCPU()
{
    checkCudaErrors(cudaMemcpy2D((void*)this->cpuBuffer, nPhi * sizeof(fReal), (void*)this->gpuThisStep,
				 this->thisStepPitch, nPhi * sizeof(fReal), nTheta, cudaMemcpyDeviceToHost));
}

KaminoQuantity::KaminoQuantity(std::string attributeName, size_t nPhi, size_t nTheta,
			       fReal phiOffset, fReal thetaOffset)
    : nPhi(nPhi), nTheta(nTheta), gridLen(M_2PI / nPhi), invGridLen(1.0 / gridLen),
    attrName(attributeName), phiOffset(phiOffset), thetaOffset(thetaOffset)
{
    cpuBuffer = new fReal[nPhi * nTheta];
    checkCudaErrors(cudaMallocPitch((void**)&gpuThisStep, &thisStepPitch, nPhi * sizeof(fReal), nTheta));
    checkCudaErrors(cudaMallocPitch((void**)&gpuNextStep, &nextStepPitch, nPhi * sizeof(fReal), nTheta));
}

KaminoQuantity::~KaminoQuantity()
{
    delete[] cpuBuffer;

    checkCudaErrors(cudaFree(gpuThisStep));
    checkCudaErrors(cudaFree(gpuNextStep));
}

std::string KaminoQuantity::getName()
{
    return this->attrName;
}

size_t KaminoQuantity::getNPhi()
{
    return this->nPhi;
}

size_t KaminoQuantity::getNTheta()
{
    return this->nTheta;
}

void KaminoQuantity::swapGPUBuffer()
{
    fReal* tempPtr = this->gpuThisStep;
    this->gpuThisStep = this->gpuNextStep;
    this->gpuNextStep = tempPtr;
}

fReal KaminoQuantity::getCPUValueAt(size_t phi, size_t theta)
{
    return this->accessCPUValueAt(phi, theta);
}

void KaminoQuantity::setCPUValueAt(size_t phi, size_t theta, fReal val)
{
    this->accessCPUValueAt(phi, theta) = val;
}

fReal& KaminoQuantity::accessCPUValueAt(size_t phi, size_t theta)
{
    assert(theta >= 0 && theta < nTheta && phi >= 0 && phi < nPhi);
    return this->cpuBuffer[theta * nPhi + phi];
}

fReal KaminoQuantity::getThetaOffset()
{
    return this->thetaOffset;
}

fReal KaminoQuantity::getPhiOffset()
{
    return this->phiOffset;
}

fReal* KaminoQuantity::getGPUThisStep()
{
    return this->gpuThisStep;
}

fReal* KaminoQuantity::getGPUNextStep()
{
    return this->gpuNextStep;
}

size_t KaminoQuantity::getThisStepPitchInElements()
{
    return this->thisStepPitch / sizeof(fReal);
}

size_t KaminoQuantity::getNextStepPitchInElements()
{
    return this->nextStepPitch / sizeof(fReal);
}