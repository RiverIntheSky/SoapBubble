# include "../include/KaminoSolver.cuh"
# include "../opencv_headers/opencv2/opencv.hpp"
# include "../include/KaminoTimer.cuh"
# include "../include/tinyexr.h"

// CONSTRUCTOR / DESTRUCTOR >>>>>>>>>>

const int fftRank = 1;
const int m = 3;
static __constant__ size_t nPhiGlobal;
static __constant__ size_t nThetaGlobal;
static __constant__ fReal gridLenGlobal;
static __constant__ fReal timeStepGlobal;

KaminoSolver::KaminoSolver(size_t nPhi, size_t nTheta, fReal radius, fReal frameDuration,
			   fReal A, int B, int C, int D, int E, fReal H) :
    nPhi(nPhi), nTheta(nTheta), radius(radius), invRadius(1.0/radius), gridLen(M_2PI / nPhi), invGridLen(1.0 / gridLen), frameDuration(frameDuration),
    timeStep(0.0), timeElapsed(0.0), advectionTime(0.0), geometricTime(0.0), projectionTime(0.0),
    A(A), B(B), C(C), D(D), E(E), H(H)
{
    /// FIXME: Should we detect and use device 0?
    /// Replace it later with functions from helper_cuda.h!
    checkCudaErrors(cudaSetDevice(0));

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
    this->nThreadxMax = deviceProp.maxThreadsDim[0];

    checkCudaErrors(cudaMalloc((void **)&gpuUFourier,
			       sizeof(ComplexFourier) * nPhi * nTheta));
    checkCudaErrors(cudaMalloc((void **)&gpuUReal,
			       sizeof(fReal) * nPhi * nTheta));
    checkCudaErrors(cudaMalloc((void **)&gpuUImag,
			       sizeof(fReal) * nPhi * nTheta));

    checkCudaErrors(cudaMalloc((void **)&gpuFFourier,
			       sizeof(ComplexFourier) * nPhi * nTheta));
    checkCudaErrors(cudaMalloc((void **)&gpuFReal,
			       sizeof(fReal) * nPhi * nTheta));
    checkCudaErrors(cudaMalloc((void **)&gpuFImag,
			       sizeof(fReal) * nPhi * nTheta));
    checkCudaErrors(cudaMalloc((void**)&gpuFZeroComponent,
			       sizeof(fReal) * nTheta));

    checkCudaErrors(cudaMalloc((void **)(&div),
			       sizeof(fReal) * nPhi * nTheta));

    checkCudaErrors(cudaMalloc((void **)(&gpuA),
			       sizeof(fReal) * nPhi * nTheta));
    checkCudaErrors(cudaMalloc((void **)(&gpuB),
			       sizeof(fReal) * nPhi * nTheta));
    checkCudaErrors(cudaMalloc((void **)(&gpuC),
			       sizeof(fReal) * nPhi * nTheta));
    precomputeABCCoef();

    this->velPhi = new KaminoQuantity("velPhi", nPhi, nTheta,
				      vPhiPhiOffset, vPhiThetaOffset);
    this->velTheta = new KaminoQuantity("velTheta", nPhi, nTheta - 1,
					vThetaPhiOffset, vThetaThetaOffset);
    this->pressure = new KaminoQuantity("p", nPhi, nTheta,
					centeredPhiOffset, centeredThetaOffset);
    this->density = new KaminoQuantity("density", nPhi, nTheta,
				       centeredPhiOffset, centeredThetaOffset);
    this->thickness = new KaminoQuantity("eta", nPhi, nTheta,
					 centeredPhiOffset, centeredThetaOffset);
    this->bulkConcentration = new KaminoQuantity("c", nPhi, nTheta,
						 centeredPhiOffset, centeredThetaOffset);
    this->surfConcentration = new KaminoQuantity("gamma", nPhi, nTheta,
						 centeredPhiOffset, centeredThetaOffset);
			   				   

    initWithConst(this->velPhi, 0.0);
    // initialize_velocity();
    initWithConst(this->velTheta, 0.0);
    // initWithConst(this->thickness, 1.0);
    initWithConst(this->bulkConcentration, 1.0);
    initWithConst(this->surfConcentration, 1.0);

    int sigLenArr[1];
    sigLenArr[0] = nPhi;
    checkCudaErrors((cudaError_t)cufftPlanMany(&kaminoPlan, fftRank, sigLenArr,
					       NULL, 1, nPhi,
					       NULL, 1, nPhi,
					       CUFFT_C2C, nTheta));
}

KaminoSolver::~KaminoSolver()
{
    checkCudaErrors(cudaFree(gpuUFourier));
    checkCudaErrors(cudaFree(gpuUReal));
    checkCudaErrors(cudaFree(gpuUImag));

    checkCudaErrors(cudaFree(gpuFFourier));
    checkCudaErrors(cudaFree(gpuFReal));
    checkCudaErrors(cudaFree(gpuFImag));
    checkCudaErrors(cudaFree(gpuFZeroComponent));

    checkCudaErrors(cudaFree(div));
		    
    checkCudaErrors(cudaFree(gpuA));
    checkCudaErrors(cudaFree(gpuB));
    checkCudaErrors(cudaFree(gpuC));

    delete this->velPhi;
    delete this->velTheta;
    delete this->pressure;
    delete this->density;
    delete this->thickness;
    delete this->bulkConcentration;
    delete this->surfConcentration;

# ifdef WRITE_PARTICLES
    delete this->particles;
# endif

    checkCudaErrors(cudaDeviceReset());

# ifdef PERFORMANCE_BENCHMARK
    float totalTimeUsed = this->advectionTime + this->bodyforceTime;
    std::cout << "Total time used for advection : " << this->advectionTime << std::endl;
    std::cout << "Total time used for body force : " << this->bodyforceTime << std::endl;
    std::cout << "Percentage of advection : " << advectionTime / totalTimeUsed * 100.0f << "%" << std::endl;
    std::cout << "Percentage of bodyforce : " << bodyforceTime / totalTimeUsed * 100.0f << "%" << std::endl;
# endif
}

void KaminoSolver::copyVelocity2GPU()
{
    velPhi->copyToGPU();
    velTheta->copyToGPU();
}

void KaminoSolver::copyDensity2GPU()
{
    density->copyToGPU();
}

__global__ void precomputeABCKernel
(fReal* A, fReal* B, fReal* C)
{
    int splitVal = nThetaGlobal / blockDim.x;
    int nIndex = blockIdx.x / splitVal;
    int threadSequence = blockIdx.x % splitVal;

    int i = threadIdx.x + threadSequence * blockDim.x;
    int n = nIndex - nPhiGlobal / 2;

    int index = nIndex * nThetaGlobal + i;
    fReal thetaI = (i + centeredThetaOffset) * gridLenGlobal;

    fReal cosThetaI = cosf(thetaI);
    fReal sinThetaI = sinf(thetaI);

    fReal valB = -2.0 / (gridLenGlobal * gridLenGlobal)
	- n * n / (sinThetaI * sinThetaI);
    fReal valA = 1.0 / (gridLenGlobal * gridLenGlobal)
	- cosThetaI / 2.0 / gridLenGlobal / sinThetaI;
    fReal valC = 1.0 / (gridLenGlobal * gridLenGlobal)
	+ cosThetaI / 2.0 / gridLenGlobal / sinThetaI;
    if (n != 0)
	{
	    if (i == 0)
		{
		    fReal coef = powf(-1.0, n);
		    valB += valA;
		    valA = 0.0;
		}
	    if (i == nThetaGlobal - 1)
		{
		    fReal coef = powf(-1.0, n);
		    valB += valC;
		    valC = 0.0;
		}
	}
    else
	{
	    valA = 0.0;
	    valB = 1.0;
	    valC = 0.0;
	}
    A[index] = valA;
    B[index] = valB;
    C[index] = valC;
}

void KaminoSolver::determineLayout(dim3& gridLayout, dim3& blockLayout,
				   size_t nTheta_row, size_t nPhi_col)
{
    if (nPhi_col <= this->nThreadxMax)
	{
	    gridLayout = dim3(nTheta_row);
	    blockLayout = dim3(nPhi_col);
	}
    else
	{
	    int splitVal = (nPhi_col + nThreadxMax - 1) / nThreadxMax;

	    gridLayout = dim3(nTheta_row * splitVal);
	    blockLayout = dim3(nThreadxMax);
	}
}

void KaminoSolver::precomputeABCCoef()
{
    checkCudaErrors(cudaMemcpyToSymbol(nPhiGlobal, &(this->nPhi), sizeof(size_t)));
    checkCudaErrors(cudaMemcpyToSymbol(nThetaGlobal, &(this->nTheta), sizeof(size_t)));
    checkCudaErrors(cudaMemcpyToSymbol(gridLenGlobal, &(this->gridLen), sizeof(fReal)));

    dim3 gridLayout;
    dim3 blockLayout;
    determineLayout(gridLayout, blockLayout, nPhi, nTheta);
    precomputeABCKernel<<<gridLayout, blockLayout>>>
	(this->gpuA, this->gpuB, this->gpuC);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void KaminoSolver::copyToCPU(KaminoQuantity* quantity, fReal* cpubuffer) {
    checkCudaErrors(cudaMemcpy2D((void*)cpubuffer, quantity->getNPhi() * sizeof(fReal),
				 (void*)quantity->getGPUThisStep(),
				 quantity->getThisStepPitchInElements() * sizeof(fReal),
				 quantity->getNPhi() * sizeof(fReal), quantity->getNTheta(),
				 cudaMemcpyDeviceToHost));
}

fReal KaminoSolver::maxAbsDifference(const fReal* A, const fReal* B, const size_t& size) {
    fReal res = 0.f;
    for (int i = 0; i < size; i++) {
	fReal diff = std::abs(A[i] - B[i]);
	if (diff > res)
	    res = diff;
    }
    return res;
}

void KaminoSolver::adjustStepSize(fReal& dt, const fReal& eps) {
    copyVelocityBack2CPU();
    thickness->copyBackToCPU();
    surfConcentration->copyBackToCPU();

    size_t sizePhiAndCentered = velPhi->getNPhi() * velPhi->getNTheta();
    size_t sizeTheta = velTheta->getNPhi() * velTheta->getNTheta();
    
    fReal* uSmall = new fReal[sizePhiAndCentered];
    fReal* vSmall = new fReal[sizeTheta];
    fReal* deltaSmall = new fReal[sizePhiAndCentered];
    fReal* gammaSmall = new fReal[sizePhiAndCentered];

    fReal* uLarge = new fReal[sizePhiAndCentered];
    fReal* vLarge = new fReal[sizeTheta];
    fReal* deltaLarge = new fReal[sizePhiAndCentered];
    fReal* gammaLarge = new fReal[sizePhiAndCentered];

    fReal optTimeStep = dt;
    int loop = 0;
    while (true) {

	// m small steps
	for (int i = 0; i < m; i++) {
	    stepForward(dt);
	}

	// store results in cpu;
	copyToCPU(velPhi, uSmall);
	copyToCPU(velTheta, vSmall);
	copyToCPU(thickness, deltaSmall);
	copyToCPU(surfConcentration, gammaSmall);
    
	// reload values before the small steps
	copyVelocity2GPU();
	thickness->copyToGPU();
	surfConcentration->copyToGPU();

	// a large step
	stepForward(dt * m);

	// store results in cpu;
	copyToCPU(velPhi, uLarge);
	copyToCPU(velTheta, vLarge);
	copyToCPU(thickness, deltaLarge);
	copyToCPU(surfConcentration, gammaLarge);
    
	// reload values before the large step
	copyVelocity2GPU();
	thickness->copyToGPU();
	surfConcentration->copyToGPU();

	fReal maxError = std::max(maxAbsDifference(uSmall, uLarge, sizePhiAndCentered),
				  std::max(maxAbsDifference(vSmall, vLarge, sizeTheta),
					   std::max(maxAbsDifference(deltaSmall, deltaLarge, sizePhiAndCentered),
						    maxAbsDifference(gammaSmall, gammaLarge, sizePhiAndCentered))));

	// optimal step size
	optTimeStep = dt * std::sqrt(eps*(m*m-1)/maxError);

	if ((optTimeStep > 2 * dt || dt > 2 * optTimeStep) && loop < 1) {
	    loop++;
	    dt = sqrt(dt * optTimeStep);	
	} else {
	    dt = optTimeStep;
	    break;
	}
    }
    
    delete[] uSmall;
    delete[] vSmall;
    delete[] deltaSmall;
    delete[] gammaSmall;
    delete[] uLarge;
    delete[] vLarge;
    delete[] deltaLarge;
    delete[] gammaLarge;
}

void KaminoSolver::setTimeStep(fReal timeStep) {
    this->timeStep = timeStep;
    checkCudaErrors(cudaMemcpyToSymbol(timeStepGlobal, &(this->timeStep), sizeof(fReal)));
}

void KaminoSolver::stepForward(fReal timeStep) {
    setTimeStep(timeStep);

# ifdef PERFORMANCE_BENCHMARK
    KaminoTimer timer;
    timer.startTimer();
# endif
    advection();
# ifdef PERFORMANCE_BENCHMARK
    this->advectionTime += timer.stopTimer() * 0.001f;
    timer.startTimer();
# endif
    bodyforce();
# ifdef PERFORMANCE_BENCHMARK
    this->bodyforceTime += timer.stopTimer() * 0.001f;
    timer.startTimer();
# endif

    this->timeElapsed += timeStep;
}

void KaminoSolver::swapVelocityBuffers()
{
    this->velPhi->swapGPUBuffer();
    this->velTheta->swapGPUBuffer();
}

void KaminoSolver::copyVelocityBack2CPU()
{
    this->velPhi->copyBackToCPU();
    this->velTheta->copyBackToCPU();
}

void KaminoSolver::copyDensityBack2CPU()
{
    this->density->copyBackToCPU();
}

void KaminoSolver::initWithConst(KaminoQuantity* attrib, fReal val)
{
    for (size_t i = 0; i < attrib->getNPhi(); ++i) {
	for (size_t j = 0; j < attrib->getNTheta(); ++j) {
		attrib->setCPUValueAt(i, j, val);
	    }
    }
    attrib->copyToGPU();
}

bool KaminoSolver::isBroken() {
    return this->broken;
}

void KaminoSolver::setBroken(bool broken) {
    this->broken = broken;
}

void KaminoSolver::initThicknessfromPic(std::string path)
{
    if (path == "")
	{
	    return;
	}
    cv::Mat image_In;
    image_In = cv::imread(path, cv::IMREAD_COLOR);
    std::cout << path << std::endl;
    if (!image_In.data)
	{
	    std::cerr << "No thickness image provided." << std::endl;
	    return;
	}

    cv::Mat image_Flipped;
    cv::flip(image_In, image_Flipped, 1);

    cv::Mat image_Resized;
    cv::Size size(nPhi, nTheta);
    cv::resize(image_Flipped, image_Resized, size);
    // cv::namedWindow( "window", cv::WINDOW_AUTOSIZE );
    // cv::imshow("window", image_Resized);
    // cv::waitKey(0);

    for (size_t i = 0; i < nPhi; ++i)
	{
	    for (size_t j = 0; j < nTheta; ++j)
		{
		    cv::Point3_<uchar>* p = image_Resized.ptr<cv::Point3_<uchar>>(j, i);
		    fReal B = p->x / 255.0; // B
		    fReal G = p->y / 255.0; // G
		    fReal R = p->z / 255.0; // R
		    this->thickness->setCPUValueAt(i, j, (B + G + R) / 3.0);
		}
	}

    this->thickness->copyToGPU();
}

void KaminoSolver::initParticlesfromPic(std::string path, size_t parPerGrid)
{
    this->particles = new KaminoParticles(path, parPerGrid, gridLen, nTheta);
}

void KaminoSolver::write_thickness_img(const std::string& s, const int frame)
{
    std::string file_string = std::to_string(frame);
    while (file_string.length() < 4) {
	file_string.insert(0, "0");
    }
    file_string.insert(0, s);
    file_string.append(".exr");
    
    const char *filename = file_string.c_str();

    thickness->copyBackToCPU();
    
    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 3;

    std::vector<float> images[3];
    images[0].resize(nPhi * nTheta);
    images[1].resize(nPhi * nTheta);
    images[2].resize(nPhi * nTheta);

    for (size_t j = 0; j < nTheta; ++j) {
	for (size_t i = 0; i < nPhi; ++i) {
	    fReal Delta = thickness->getCPUValueAt(i, j);
	    if (Delta < 0) {
		this->setBroken(true);
		return;
	    } else {
		Delta = Delta * this->H * 5e5; // Delta = 1 <==> thickness = 2000nm
		images[0][j*nPhi+i] = Delta;
		images[1][j*nPhi+i] = Delta;
		images[2][j*nPhi+i] = Delta;
	    }
	}
    }

    float* image_ptr[3];
    image_ptr[0] = &(images[2].at(0)); // B
    image_ptr[1] = &(images[1].at(0)); // G
    image_ptr[2] = &(images[0].at(0)); // R

    image.images = (unsigned char**)image_ptr;
    image.width = nPhi;
    image.height = nTheta;

    header.num_channels = 3;
    header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels); 
    // Must be (A)BGR order, since most of EXR viewers expect this channel order.
    strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
    strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
    strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';

    header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels); 
    header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
    for (int i = 0; i < header.num_channels; i++) {
      header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
      header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
    }

    const char* err = NULL; // or nullptr in C++11 or later.
    int ret = SaveEXRImageToFile(&image, &header, filename, &err);
    if (ret != TINYEXR_SUCCESS) {
      fprintf(stderr, "Save EXR err: %s\n", err);
      FreeEXRErrorMessage(err); // free's buffer for an error message
      return;
    }
    printf("Saved exr file. [ %s ] \n", filename);

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);

}

void KaminoSolver::write_data_bgeo(const std::string& s, const int frame)
{
    std::string file = s + std::to_string(frame) + ".bgeo";
    std::cout << "Writing to: " << file << std::endl;

    Partio::ParticlesDataMutable* parts = Partio::create();
    Partio::ParticleAttribute pH, vH, densityVal;
    pH = parts->addAttribute("position", Partio::VECTOR, 3);
    vH = parts->addAttribute("v", Partio::VECTOR, 3);
    densityVal = parts->addAttribute("density", Partio::FLOAT, 1);

    vec3 pos;
    vec3 vel;

    size_t iWest, iEast, jNorth, jSouth;
    fReal uWest, uEast, vNorth, vSouth;

    velPhi->copyBackToCPU();
    velTheta->copyBackToCPU();
    density->copyBackToCPU();

    for (size_t j = 0; j < nTheta; ++j)
	{
	    for (size_t i = 0; i < nPhi; ++i)
		{
		    iWest = i;
		    uWest = velPhi->getCPUValueAt(iWest, j);
		    i == (nPhi - 1) ? iEast = 0 : iEast = i + 1;
		    uEast = velPhi->getCPUValueAt(iEast, j);

		    if (j == 0)
			{
			    jNorth = jSouth = 0;
			}
		    else if (j == nTheta - 1)
			{
			    jNorth = jSouth = nTheta - 2;
			}
		    else
			{
			    jNorth = j - 1;
			    jSouth = j;
			}
		    vNorth = velTheta->getCPUValueAt(i, jNorth);
		    vSouth = velTheta->getCPUValueAt(i, jSouth);

		    fReal velocityPhi, velocityTheta;
		    velocityPhi = (uWest + uEast) / 2.0;
		    velocityTheta = (vNorth + vSouth) / 2.0;

		    pos = vec3((i + centeredPhiOffset) * gridLen, (j + centeredThetaOffset) * gridLen, 0.0);
		    vel = vec3(0.0, velocityTheta, velocityPhi);
		    mapVToSphere(pos, vel);
		    mapPToSphere(pos);

		    float densityValuefloat = density->getCPUValueAt(i, j);

		    int idx = parts->addParticle();
		    float* p = parts->dataWrite<float>(pH, idx);
		    float* v = parts->dataWrite<float>(vH, idx);
		    float* d = parts->dataWrite<float>(densityVal, idx);
			
		    for (int k = 0; k < 3; ++k) 
			{
			    p[k] = pos[k];
			    v[k] = vel[k];
			}
		    d[0] = densityValuefloat;
		}
	}

    Partio::write(file.c_str(), *parts);
    parts->release();
}

void KaminoSolver::write_particles_bgeo(const std::string& s, const int frame)
{
    std::string file = s + std::to_string(frame) + ".bgeo";
    std::cout << "Writing to: " << file << std::endl;

    Partio::ParticlesDataMutable* parts = Partio::create();
    Partio::ParticleAttribute pH, vH, colorVal;
    pH = parts->addAttribute("position", Partio::VECTOR, 3);
    vH = parts->addAttribute("v", Partio::VECTOR, 3);
    colorVal = parts->addAttribute("color", Partio::VECTOR, 3);

    vec3 pos;
    vec3 vel;
    vec3 col;

    this->particles->copyBack2CPU();

    for (size_t i = 0; i < particles->numOfParticles; ++i)
	{
	    pos = vec3(particles->coordCPUBuffer[2 * i],
		       particles->coordCPUBuffer[2 * i + 1], 0.0);
	    mapPToSphere(pos);

	    col = vec3(particles->colorBGR[3 * i + 1],
		       particles->colorBGR[3 * i + 2],
		       particles->colorBGR[3 * i + 3]);

	    int idx = parts->addParticle();
	    float* p = parts->dataWrite<float>(pH, idx);
	    float* v = parts->dataWrite<float>(vH, idx);
	    float* c = parts->dataWrite<float>(colorVal, idx);
	
	    for (int k = 0; k < 3; ++k)
		{
		    p[k] = pos[k];
		    v[k] = 0.0;
		    c[k] = col[k];
		}
	}

    Partio::write(file.c_str(), *parts);
    parts->release();
}

void KaminoSolver::mapPToSphere(vec3& pos) const
{
    float theta = pos[1];
    float phi = pos[0];
    pos[0] = radius * sin(theta) * cos(phi);
    pos[2] = radius * sin(theta) * sin(phi);
    pos[1] = radius * cos(theta);
}

void KaminoSolver::mapVToSphere(vec3& pos, vec3& vel) const
{
    float theta = pos[1];
    float phi = pos[0];

    float u_theta = vel[1];
    float u_phi = vel[2];

    vel[0] = cos(theta) * cos(phi) * u_theta - sin(phi) * u_phi;
    vel[2] = cos(theta) * sin(phi) * u_theta + cos(phi) * u_phi;
    vel[1] = -sin(theta) * u_theta;
}
