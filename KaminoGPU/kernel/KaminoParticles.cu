# include "../include/KaminoParticles.cuh"

KaminoParticles::KaminoParticles(std::string path, size_t particleDensity, fReal gridLen, size_t nTheta)
    : particlePGrid(particlePGrid), nPhi(2 * nTheta), nTheta(nTheta)
{
    numOfParticles = nTheta * nPhi * particleDensity;
    // cv::Mat image_In, image_Out;
    // image_In = cv::imread(path, cv::IMREAD_COLOR);
    // if (!image_In.data)
    // 	{
    // 	    std::cerr << "No particle color image provided." << std::endl;
    // 	}
    // else
    // 	{
    // 	    cv::Mat image_Flipped;
    // 	    cv::flip(image_In, image_Flipped, 1);
    // 	    cv::Size size(nPhi, nTheta);
    // 	    cv::resize(image_Flipped, image_Out, size);
    // 	}

    // fReal linearDensity = sqrt(particleDensity);
    // fReal delta = M_PI / nTheta / linearDensity;
    // fReal halfDelta = delta / 2.0;

    // unsigned int numThetaParticles = linearDensity * nTheta;
    // unsigned int numPhiParticles = 2 * numThetaParticles;
    // numOfParticles = numThetaParticles * numPhiParticles;

    checkCudaErrors(cudaMalloc(&coordGPUThisStep, sizeof(fReal) * numOfParticles * 2));
    checkCudaErrors(cudaMalloc(&coordGPUNextStep, sizeof(fReal) * numOfParticles * 2));
    coordCPUBuffer = new fReal[numOfParticles * 2];

    for (size_t i = 0; i < numOfParticles; i++) {
	fReal u1 = static_cast <fReal> (rand()) / static_cast <fReal> (RAND_MAX);
	fReal u2 = static_cast <fReal> (rand()) / static_cast <fReal> (RAND_MAX);
	fReal phi = M_2PI * u1;
        fReal theta = acosf(1 - 2 * u2);
	coordCPUBuffer[2 * i] = phi;
	coordCPUBuffer[2 * i + 1] = theta;
    }
    // coordCPUBuffer = new fReal[numOfParticles * 2];
    // colorBGR = new fReal[numOfParticles * 3];

    // for (unsigned int i = 0; i < numPhiParticles; ++i)
    // 	{
    // 	    for (unsigned int j = 0; j < numThetaParticles; ++j)
    // 		{
    // 		    // distribute in phi and theta randomly
    // 		    // +/- is 50/50
    // 		    fReal signPhi = static_cast <fReal> (rand()) / static_cast <fReal> (RAND_MAX);
    // 		    signPhi = signPhi >= 0.5 ? 1.0 : -1.0;
    // 		    fReal signTheta = static_cast <fReal> (rand()) / static_cast <fReal> (RAND_MAX);
    // 		    signTheta = signTheta >= 0.5 ? 1.0 : -1.0;

    // 		    // get random value between 0 and halfDelta in +/- direction
    // 		    fReal randPhi = signPhi * halfDelta * static_cast <fReal> (rand()) / static_cast <fReal> (RAND_MAX);
    // 		    fReal randTheta = signTheta * halfDelta * static_cast <fReal> (rand()) / static_cast <fReal> (RAND_MAX);

    // 		    // assign positions (phi, theta)
    // 		    fReal phi = i * delta + randPhi;
    // 		    fReal theta = j * delta + randTheta;
    // 		    if (phi < 0.0)
    // 			phi = 0.0;
    // 		    if (theta < 0.0)
    // 			theta = 0.0;

    // 		    size_t x = std::floor(phi / gridLen);
    // 		    size_t y = std::floor(theta / gridLen);
			
    // 		    size_t index = i * numThetaParticles + j;
    // 		    // set particle position
    // 		    coordCPUBuffer[2 * index] = phi;
    // 		    coordCPUBuffer[2 * index + 1] = theta;
			
    // 		    if (image_In.data)
    // 			{
    // 			    // initialize velocities (0,0)
    // 			    cv::Point3_<uchar>* p = image_Out.ptr<cv::Point3_<uchar>>(y, x);
    // 			    // define particle color
    // 			    colorBGR[3 * index] = p->y / 255.0;
    // 			    colorBGR[3 * index + 1] = p->z / 255.0;
    // 			    colorBGR[3 * index + 2] = p->x / 255.0;
    // 			}
    // 		    else
    // 			{
    // 			    colorBGR[3 * index] = 0.0;
    // 			    colorBGR[3 * index + 1] = 0.0;
    // 			    colorBGR[3 * index + 2] = 0.0;
    // 			}
    // 		}
    // 	}

    copy2GPU();
    checkCudaErrors(cudaMalloc(&value, sizeof(fReal) * numOfParticles));
}

KaminoParticles::~KaminoParticles()
{
    delete[] coordCPUBuffer;
    delete[] value;

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