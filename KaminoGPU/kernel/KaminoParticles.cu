# include "../include/KaminoParticles.cuh"

KaminoParticles::KaminoParticles(std::string path, size_t particleDensity, fReal gridLen, size_t nTheta)
    : particlePGrid(particlePGrid), nPhi(2 * nTheta), nTheta(nTheta)
{
    numOfParticles = nTheta * nPhi * particleDensity;

    checkCudaErrors(cudaMalloc(&coordGPUThisStep, sizeof(fReal) * numOfParticles * 2));
    checkCudaErrors(cudaMalloc(&coordGPUNextStep, sizeof(fReal) * numOfParticles * 2));
    coordCPUBuffer = new fReal[numOfParticles * 2];

    for (size_t i = 0; i < numOfParticles; i++) {
	fReal u1 = static_cast <fReal> (rand()) / static_cast <fReal> (RAND_MAX);
	fReal u2 = static_cast <fReal> (rand()) / static_cast <fReal> (RAND_MAX);
	fReal phiId = u1 * nPhi;
        fReal thetaId = acosf(1 - 2 * u2) * nTheta / M_PI;
	coordCPUBuffer[2 * i] = phiId;
	coordCPUBuffer[2 * i + 1] = thetaId;
    }

    copy2GPU();
    checkCudaErrors(cudaMalloc(&value, sizeof(fReal) * numOfParticles));
    checkCudaErrors(cudaMalloc(&tempVal, sizeof(fReal) * numOfParticles));
}

KaminoParticles::~KaminoParticles()
{
    delete[] coordCPUBuffer;
    delete[] value;
    delete[] tempVal;

    checkCudaErrors(cudaFree(coordGPUThisStep));
    checkCudaErrors(cudaFree(coordGPUNextStep));
}

void KaminoParticles::copy2GPU()
{
    if (numOfParticles != 0)
	{
	    checkCudaErrors(cudaMemcpy(this->coordGPUThisStep, this->coordCPUBuffer,
				       sizeof(fReal) * numOfParticles * 2, cudaMemcpyHostToDevice));
	}
}

void KaminoParticles::copyBack2CPU()
{
    if (numOfParticles != 0)
	{
	    checkCudaErrors(cudaMemcpy(this->coordCPUBuffer, this->coordGPUThisStep,
				       sizeof(fReal) * numOfParticles * 2, cudaMemcpyDeviceToHost));
	}
}

void KaminoParticles::swapGPUBuffers()
{
    fReal* temp = this->coordGPUThisStep;
    this->coordGPUThisStep = this->coordGPUNextStep;
    this->coordGPUNextStep = temp;
}