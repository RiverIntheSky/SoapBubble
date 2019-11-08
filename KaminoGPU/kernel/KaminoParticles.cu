# include "../include/KaminoParticles.cuh"

KaminoParticles::KaminoParticles(std::string path, size_t particleDensity, float gridLen, size_t nTheta)
    : particlePGrid(particlePGrid), nPhi(2 * nTheta), nTheta(nTheta)
{
    numOfParticles = nTheta * nPhi * particleDensity;

    checkCudaErrors(cudaMalloc(&coordGPUThisStep, sizeof(float) * numOfParticles * 2));
    checkCudaErrors(cudaMalloc(&coordGPUNextStep, sizeof(float) * numOfParticles * 2));
    coordCPUBuffer = new float[numOfParticles * 2];

    for (size_t i = 0; i < numOfParticles; i++) {
	float u1 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	float u2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	float phiId = u1 * nPhi;
	//float thetaId = u2 * nTheta;
        float thetaId = acosf(1 - 2 * u2) * nTheta / M_PI;
	coordCPUBuffer[2 * i] = thetaId;
	coordCPUBuffer[2 * i + 1] = phiId;
    }

    copy2GPU();
    checkCudaErrors(cudaMalloc(&value, sizeof(float) * numOfParticles));
    checkCudaErrors(cudaMalloc(&tempVal, sizeof(float) * numOfParticles));
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
				       sizeof(float) * numOfParticles * 2, cudaMemcpyHostToDevice));
	}
}

void KaminoParticles::copyBack2CPU()
{
    if (numOfParticles != 0)
	{
	    checkCudaErrors(cudaMemcpy(this->coordCPUBuffer, this->coordGPUThisStep,
				       sizeof(float) * numOfParticles * 2, cudaMemcpyDeviceToHost));
	}
}

void KaminoParticles::swapGPUBuffers()
{
    float* temp = this->coordGPUThisStep;
    this->coordGPUThisStep = this->coordGPUNextStep;
    this->coordGPUNextStep = temp;
}