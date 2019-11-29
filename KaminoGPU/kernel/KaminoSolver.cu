# include "../include/KaminoSolver.cuh"
# include "../opencv_headers/opencv2/opencv.hpp"
# include "../include/KaminoTimer.cuh"
# include "../include/tinyexr.h"

// CONSTRUCTOR / DESTRUCTOR >>>>>>>>>>

const int fftRank = 1;
static const int m = 3;
static __constant__ size_t nPhiGlobal;
static __constant__ size_t nThetaGlobal;
static __constant__ float gridLenGlobal;
static __constant__ int Cols;
static __constant__ int Rows;
static __constant__ int Ratio;


__global__ void initParticleValues(float* particleVal, float* particleCoord, float* thickness, size_t pitch) {
    int particleId = blockIdx.x * blockDim.x + threadIdx.x;
    float thetaId = particleCoord[2 * particleId];
    float phiId = particleCoord[2 * particleId + 1];

    float2 sampleId = make_float2(thetaId, phiId);
    particleVal[particleId] = sampleCentered(thickness, sampleId, pitch);
}


__global__ void initMapping(float* map_theta, float* map_phi){
    // Index
    int splitVal = nPhiGlobal / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;

    map_theta[thetaId * nPhiGlobal + phiId] = (float)thetaId + centeredThetaOffset;
    map_phi[thetaId * nPhiGlobal + phiId] = (float)phiId + centeredPhiOffset;
}


__global__ void initLinearSystem(int* row_ptr, int* col_ind) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > nPhiGlobal * nThetaGlobal) return;
    int idx5 = 5 * idx;
    row_ptr[idx] = idx5;

    int n = nPhiGlobal * nThetaGlobal;
    
    if (idx < nPhiGlobal * nThetaGlobal) {
	// up
	if (idx < nPhiGlobal) { // first row
	    col_ind[idx5] = (idx + nThetaGlobal) % nPhiGlobal;
	} else {
	    col_ind[idx5] = idx - nPhiGlobal;
	}
	
	// left
	col_ind[idx5 + 1] = (idx % nPhiGlobal) == 0 ? idx + nPhiGlobal - 1 : idx - 1;
	
	// center
	col_ind[idx5 + 2] = idx;
	
	// right
	col_ind[idx5 + 3] = (idx % nPhiGlobal) == (nPhiGlobal - 1) ? idx - nPhiGlobal + 1 : idx + 1;
	
	// down
	if (idx >= (nThetaGlobal - 1) * nPhiGlobal + nThetaGlobal) {
	    // last half of the last row
	    col_ind[idx5 + 4] = idx - nThetaGlobal;
	} else if (idx >= (nThetaGlobal - 1) * nPhiGlobal) {
	    // first half of the last row
	    col_ind[idx5 + 4] = idx + nThetaGlobal;
	} else {
	    col_ind[idx5 + 4] = idx + nPhiGlobal;
	}
    }
}


KaminoSolver::KaminoSolver(size_t nPhi, size_t nTheta, float radius, float dt,
			   float H, int device, std::string AMGconfig, size_t particleDensity) :
    nPhi(nPhi), nTheta(nTheta), radius(radius), invRadius(1.0/radius), gridLen(M_2PI / nPhi),
    invGridLen(1.0 / gridLen), timeStep(dt), timeElapsed(0.0), advectionTime(0.0),
    bodyforceTime(0.0), CGTime(0.0), H(H), epsilon(H/radius), N(nPhi*nTheta), nz(5*N),
    particleDensity(particleDensity)
{
    /// FIXME: Should we detect and use device 0?
    /// Replace it later with functions from helper_cuda.h!
    checkCudaErrors(cudaSetDevice(device));

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device));
    this->nThreadxMax = std::min(deviceProp.maxThreadsDim[0], 128);

    checkCudaErrors(cudaMalloc((void **)(&div),
			       sizeof(float) * N));
    checkCudaErrors(cudaMalloc((void **)(&weight),
			       sizeof(float2) * N));
    checkCudaErrors(cudaMalloc((void **)(&row_ptr),
			       sizeof(int) * (N + 1)));
    checkCudaErrors(cudaMalloc((void **)(&col_ind),
			       sizeof(int) * nz));
    checkCudaErrors(cudaMalloc((void **)(&rhs),
			       sizeof(float) * N));
    checkCudaErrors(cudaMalloc((void **)(&val),
			       sizeof(float) * nz));

    checkCudaErrors(cudaMemcpyToSymbol(nPhiGlobal, &(this->nPhi), sizeof(size_t)));
    checkCudaErrors(cudaMemcpyToSymbol(nThetaGlobal, &(this->nTheta), sizeof(size_t)));

    this->velPhi = new BimocqQuantity("velPhi", nPhi, nTheta,
				      vPhiPhiOffset, vPhiThetaOffset);
    this->velTheta = new BimocqQuantity("velTheta", nPhi, nTheta - 1,
					vThetaPhiOffset, vThetaThetaOffset);
    this->thickness = new BimocqQuantity("eta", nPhi, nTheta,
					 centeredPhiOffset, centeredThetaOffset);
    this->surfConcentration = new BimocqQuantity("gamma", nPhi, nTheta,
						 centeredPhiOffset, centeredThetaOffset);
    this->pitch = surfConcentration->getThisStepPitchInElements();

    initWithConst(this->velPhi, 0.0);
    initWithConst(this->velTheta, 0.0);
    // initialize_velocity();
    initWithConst(this->thickness, 0.5);
    initWithConst(this->surfConcentration, 1.0);

    dim3 gridLayout;
    dim3 blockLayout;
			        
    determineLayout(gridLayout, blockLayout, 1, N + 1);
    initLinearSystem<<<gridLayout, blockLayout>>>(row_ptr, col_ind);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    // printGPUarraytoMATLAB<int>("test/row_ptr.txt", row_ptr, N + 1, 1, 1);
    // printGPUarraytoMATLAB<int>("test/col_ind.txt", col_ind, N, 5, 5);
    
    // cuSPARSE and cuBLAS
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    // Device memory management
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_r, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_p, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_omega, N * sizeof(float)));   
	
    // Create preconditioner and dense vectors
    // CHECK_CUSPARSE(cusparseCreateCsr(&matM, N, N, N, row_ptrm, col_indm,
    // 				     valm, CUSPARSE_INDEX_32I,
    // 				     CUSPARSE_INDEX_32I,
    // 				     CUSPARSE_INDEX_BASE_ZERO,
    // 				     CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, N, d_x, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecR, N, d_r, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecP, N, d_p, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecO, N, d_omega, CUDA_R_32F));

    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));

    /* Define the properties of the matrix */
    cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
    
    int sigLenArr[1];
    sigLenArr[0] = nPhi;
    checkCudaErrors((cudaError_t)cufftPlanMany(&kaminoPlan, fftRank, sigLenArr,
					       NULL, 1, nPhi,
					       NULL, 1, nPhi,
					       CUFFT_C2C, nTheta));


    /* AMGX */
    int devices[] = {device};
    const char *AMGconfigFile = AMGconfig.c_str();
    AMGX_initialize();
    AMGX_initialize_plugins();
    AMGX_config_create_from_file(&cfg, AMGconfigFile);
    AMGX_resources_create(&res, cfg, NULL, 1, devices);
    mode = AMGX_mode_dFFI;
    AMGX_matrix_create(&A, res, mode);
    AMGX_vector_create(&b, res, mode);
    AMGX_vector_create(&x, res, mode);
    AMGX_solver_create(&solver, res, mode, cfg);

    /* Bimocq mapping buffers */
    CHECK_CUDA(cudaMalloc(&forward_p, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&forward_t, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&backward_p, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&backward_t, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&backward_pprev, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&backward_tprev, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&tmp_p, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&tmp_t, N * sizeof(float)));
    
    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    initMapping<<<gridLayout, blockLayout>>>(forward_t, forward_p);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(backward_p, forward_p, N * sizeof(float),
			  cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(backward_pprev, forward_p, N * sizeof(float),
			  cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(backward_t, forward_t, N * sizeof(float),
			  cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(backward_tprev, forward_t, N * sizeof(float),
			  cudaMemcpyDeviceToDevice));
}


KaminoSolver::~KaminoSolver()
{
    CHECK_CUDA(cudaFree(div));
    CHECK_CUDA(cudaFree(weight));
    CHECK_CUDA(cudaFree(row_ptr));
    CHECK_CUDA(cudaFree(col_ind));
    CHECK_CUDA(cudaFree(rhs));
    CHECK_CUDA(cudaFree(val));

    // CHECK_CUSPARSE(cusparseDestroySpMat(matM));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecP));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecO));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));
    CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));

    CHECK_CUDA(cudaFree(d_r));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_p));
    CHECK_CUDA(cudaFree(d_omega));


    delete this->velPhi;
    delete this->velTheta;
    delete this->thickness;
    delete this->surfConcentration;

    /* AMGX */    
    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(b);
    AMGX_vector_destroy(x);
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(res);
    AMGX_config_destroy(cfg);
    AMGX_finalize_plugins();
    AMGX_finalize();

    /* Bimocq mapping buffers */
    CHECK_CUDA(cudaFree(forward_p));
    CHECK_CUDA(cudaFree(forward_t));
    CHECK_CUDA(cudaFree(backward_p));
    CHECK_CUDA(cudaFree(backward_t));
    CHECK_CUDA(cudaFree(backward_pprev));
    CHECK_CUDA(cudaFree(backward_tprev));
    CHECK_CUDA(cudaFree(tmp_p));
    CHECK_CUDA(cudaFree(tmp_t));
    
    if (particleDensity > 0) {
	delete this->particles;
    }

    CHECK_CUDA(cudaDeviceReset());

# ifdef PERFORMANCE_BENCHMARK
    float totalTimeUsed = this->advectionTime + this->bodyforceTime;
    std::cout << "Total time used for advection : " << this->advectionTime << std::endl;
    std::cout << "Total time used for body force : " << this->bodyforceTime << std::endl;
    std::cout << "Percentage of advection : " << advectionTime / totalTimeUsed * 100.0f << "%" << std::endl;
    std::cout << "Percentage of bodyforce : " << bodyforceTime / totalTimeUsed * 100.0f << "%" << std::endl;
    std::cout << "Percentage of CG / bodyforce : " << CGTime / bodyforceTime * 100.0f << "%" << std::endl;
    std::cout << "Elapsed time " << this->timeElapsed << std::endl;
# endif
}

void KaminoSolver::copyVelocity2GPU()
{
    velPhi->copyToGPU();
    velTheta->copyToGPU();
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


void KaminoSolver::copyToCPU(KaminoQuantity* quantity, float* cpubuffer) {
    checkCudaErrors(cudaMemcpy2D((void*)cpubuffer, quantity->getNPhi() * sizeof(float),
				 (void*)quantity->getGPUThisStep(),
				 quantity->getThisStepPitchInElements() * sizeof(float),
				 quantity->getNPhi() * sizeof(float), quantity->getNTheta(),
				 cudaMemcpyDeviceToHost));
}


float KaminoSolver::maxAbsDifference(const float* A, const float* B, const size_t& size) {
    float res = 0.f;
    for (int i = 0; i < size; i++) {
	float diff = std::abs(A[i] - B[i]);
	if (diff > res)
	    res = diff;
    }
    return res;
}


void KaminoSolver::adjustStepSize(float& dt, const float& U, const float& epsilon) {
    copyVelocityBack2CPU();
    thickness->copyBackToCPU();
    surfConcentration->copyBackToCPU();

    size_t sizePhiAndCentered = velPhi->getNPhi() * velPhi->getNTheta();
    size_t sizeTheta = velTheta->getNPhi() * velTheta->getNTheta();
    
    float* uSmall = new float[sizePhiAndCentered];
    float* vSmall = new float[sizeTheta];
    float* deltaSmall = new float[sizePhiAndCentered];
    float* gammaSmall = new float[sizePhiAndCentered];

    float* uLarge = new float[sizePhiAndCentered];
    float* vLarge = new float[sizeTheta];
    float* deltaLarge = new float[sizePhiAndCentered];
    float* gammaLarge = new float[sizePhiAndCentered];

    float optTimeStep = dt;
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

	float maxError = std::max(maxAbsDifference(uSmall, uLarge, sizePhiAndCentered),
				  std::max(maxAbsDifference(vSmall, vLarge, sizeTheta),
					   std::max(maxAbsDifference(deltaSmall, deltaLarge, sizePhiAndCentered),
						    maxAbsDifference(gammaSmall, gammaLarge, sizePhiAndCentered))));

	// optimal step size
	std::cout << "maxError " << maxError << std::endl;
	// std::cout << "epsilon " << epsilon << std::endl;
	// std::cout << "dt " << dt << std::endl;
	optTimeStep = dt * std::sqrt(epsilon* (m*m-1)/maxError);
	std::cout << "opt " << optTimeStep << std::endl;
		
	//optTimeStep = max(optTimeStep, 1e-7);

	if ((optTimeStep > 2 * dt || dt > 2 * optTimeStep) && loop < 2) {
	    loop++;
	    dt = optTimeStep;	
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


void KaminoSolver::stepForward(float dt) {
    float dt_ = this->timeStep;
    this->timeStep = dt;
    stepForward();
    this->timeStep = dt_;
}


void KaminoSolver::stepForward() {
# ifdef PERFORMANCE_BENCHMARK
    KaminoTimer timer;
    timer.startTimer();
# endif
    updateCFL();
    updateForward(this->timeStep, forward_t, forward_p);
    updateBackward(this->timeStep, backward_t, backward_p);

    advection();
# ifdef PERFORMANCE_BENCHMARK
    this->advectionTime += timer.stopTimer() * 0.001f;
    timer.startTimer();
# endif
    bodyforce();
# ifdef PERFORMANCE_BENCHMARK
    this->bodyforceTime += timer.stopTimer() * 0.001f;
# endif
    this->timeElapsed += this->timeStep;

    float distortion = estimateDistortion();
    std::cout << "max distortion " << distortion << std::endl;

    if (distortion > nTheta / 128.f)
    	reInitializeMapping();
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


void KaminoSolver::initWithConst(KaminoQuantity* attrib, float val)
{
    for (size_t i = 0; i < attrib->getNPhi(); ++i) {
	for (size_t j = 0; j < attrib->getNTheta(); ++j) {
		attrib->setCPUValueAt(i, j, val);
	    }
    }
    attrib->copyToGPU();
}


void KaminoSolver::initWithConst(BimocqQuantity* attrib, float val)
{
    for (size_t i = 0; i < attrib->getNPhi(); ++i) {
	for (size_t j = 0; j < attrib->getNTheta(); ++j) {
		attrib->setCPUValueAt(i, j, val);
	    }
    }
    attrib->copyToGPU();
    CHECK_CUDA(cudaMemcpy(attrib->getGPUInit(), attrib->getGPUThisStep(),
			  pitch * attrib->getNTheta() *
			  sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(attrib->getGPUInitLast(), attrib->getGPUThisStep(),
			  pitch * attrib->getNTheta() *
			  sizeof(float), cudaMemcpyDeviceToDevice));
}


bool KaminoSolver::isBroken() {
    return this->broken;
}


void KaminoSolver::setBroken(bool broken) {
    this->broken = broken;
}


void KaminoSolver::initThicknessfromPic(std::string path)
{
    if (path == "") {
	std::cout << "No thickness image provided, initialize with eta = 0.5" << std::endl;
    } else {
	cv::Mat image_In, image_Flipped;
	image_In = cv::imread(path, cv::IMREAD_UNCHANGED);
	if (!image_In.data) {
	    std::cout << "No thickness image provided, initialize with eta = 0.5" << std::endl;
	} else {
	    cv::Mat image_Resized;
	    cv::flip(image_In, image_Flipped, 1);
	    cv::Size size(nPhi, nTheta);
	    cv::resize(image_Flipped, image_Resized, size);

	    for (size_t i = 0; i < nPhi; ++i) {
		for (size_t j = 0; j < nTheta; ++j) {
		    cv::Point3_<float>* p = image_Resized.ptr<cv::Point3_<float>>(j, i);
		    float C = p->x; // Gray Scale
		    this->thickness->setCPUValueAt(i, j, C);
		}
	    }
	    this->thickness->copyToGPU();
    
	    this->rows = image_Flipped.rows;
	    this->cols = image_Flipped.cols;
   	}
    }

    CHECK_CUDA(cudaMemcpy(this->thickness->getGPUInit(), this->thickness->getGPUThisStep(),
			  this->thickness->getThisStepPitchInElements() * this->thickness->getNTheta() *
			  sizeof(float), cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaMemcpy(this->thickness->getGPUInitLast(), this->thickness->getGPUThisStep(),
			  this->thickness->getThisStepPitchInElements() * this->thickness->getNTheta() *
			  sizeof(float), cudaMemcpyDeviceToDevice));

    if (particleDensity > 0) {
	this->particles = new KaminoParticles(path, particleDensity, gridLen, nTheta);
    	dim3 gridLayout;
    	dim3 blockLayout;
    	determineLayout(gridLayout, blockLayout, 1, this->particles->numOfParticles);
    	initParticleValues<<<gridLayout, blockLayout>>>
    	    (this->particles->value, this->particles->coordGPUThisStep, this->thickness->getGPUThisStep(),
    	     this->thickness->getThisStepPitchInElements());
    	checkCudaErrors(cudaGetLastError());
    	checkCudaErrors(cudaDeviceSynchronize());	
    }
}


// void KaminoSolver::initParticlesfromPic(std::string path, size_t parPerGrid)
// {
//     this->particles = new KaminoParticles(path, parPerGrid, gridLen, nTheta);
// }


void KaminoSolver::write_image(const std::string& s, size_t width, size_t height, std::vector<float> *images) {
    const char *filename = s.c_str();
    
    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 3;

    float* image_ptr[3];
    image_ptr[0] = &(images[2].at(0)); // B
    image_ptr[1] = &(images[1].at(0)); // G
    image_ptr[2] = &(images[0].at(0)); // R

    image.images = (unsigned char**)image_ptr;
    image.width = width;
    //    image.height = nTheta;
    image.height = height;

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


void KaminoSolver::write_velocity_image(const std::string& s, const int frame) {
    std::string img_string = std::to_string(frame);
    while (img_string.length() < 4) {
	img_string.insert(0, "0");
    }
    img_string.insert(0, s + "vel");
    std::string u_string = img_string;
    std::string v_string = img_string;
    img_string.append(".exr");
    u_string.append("u.txt");
    v_string.append("v.txt");

    std::ofstream ofu(u_string);
    std::ofstream ofv(v_string);

    copyVelocityBack2CPU();
    std::vector<float> images[3];
    images[0].resize(nPhi * nTheta);
    images[1].resize(nPhi * nTheta);
    images[2].resize(nPhi * nTheta);

    float maxu = std::numeric_limits<float>::min();
    float maxv = std::numeric_limits<float>::min();
    float minu = std::numeric_limits<float>::max();
    float minv = std::numeric_limits<float>::max();
    size_t maxuthetaid = 0;
    size_t maxuphiid = 0;
    size_t minuthetaid = 0;
    size_t minuphiid = 0;
    size_t maxvthetaid = 0;
    size_t maxvphiid = 0;
    size_t minvthetaid = 0;
    size_t minvphiid = 0;

    for (size_t j = 0; j < nTheta; ++j) {
	for (size_t i = 0; i < nPhi; ++i) {
	    float uW = velPhi->getCPUValueAt(i, j);
	    // ofu << uW << " ";
	    float uE;
	    float vN;
	    float vS;
	    if (i != nPhi - 1) {
	        uE = velPhi->getCPUValueAt(i + 1, j);		
	    } else {
		uE = velPhi->getCPUValueAt(0, j);	
	    }
	    if (j != 0) {
		vN = velTheta->getCPUValueAt(i, j - 1);
	    } else {
		size_t oppositei = (i + nPhi/2) % nPhi;
		vN = 0.75 * velTheta->getCPUValueAt(i, j) -
		    0.25 * velTheta->getCPUValueAt(oppositei, j);
	    }
	    if (j != nTheta - 1) {
		vS = velTheta->getCPUValueAt(i, j);
		// ofv << vS << " ";
	    } else {
		size_t oppositei = (i + nPhi/2) % nPhi;
		vS = 0.75 * velTheta->getCPUValueAt(i, j - 1) -
		    0.25 * velTheta->getCPUValueAt(oppositei, j - 1);
	    }
	    float u = 0.5 * (uW + uE);
	    ofu << u << " ";
	    float v = 0.5 * (vN + vS);
	    ofv << v << " ";
	    if (u > maxu) {
		maxu = u;
		maxuthetaid = j;
		maxuphiid = i;
	    }
		
	    if (u < minu) {
		minu = u;
		minuthetaid = j;
		minuphiid = i;
	    }		
	    if (v > maxv) {
		maxv = v;
		maxvthetaid = j;
		maxvphiid = i;
	    }		
	    if (v < minv) {
		minv = v;
		minvthetaid = j;
		minvphiid = i;
	    }
		
	    // std::cout << "theta " << j << " phi " << i << " u " << u << " v " << v << std::endl;
	    images[0][j*nPhi+i] = u/2+0.5; // R
	    images[1][j*nPhi+i] = v/2+0.5; // G
	    images[2][j*nPhi+i] = 0.5; // B
	}
	ofu << std::endl;
	ofv << std::endl;
    }

    std::cout << "max u = " << maxu << " theta " << maxuthetaid << " phi " << maxuphiid << std::endl;
    std::cout << "min u = " << minu << " theta " << minuthetaid << " phi " << minuphiid << std::endl;
    std::cout << "max v = " << maxv << " theta " << maxvthetaid << " phi " << maxvphiid << std::endl;
    std::cout << "min v = " << minv << " theta " << minvthetaid << " phi " << minvphiid << std::endl;

    write_image(img_string, nPhi, nTheta, images);
}


void KaminoSolver::write_concentration_image(const std::string& s, const int frame) {
    std::string img_string = std::to_string(frame);
    while (img_string.length() < 4) {
	img_string.insert(0, "0");
    }
    img_string.insert(0, s + "con");
    std::string mat_string = img_string;
    img_string.append(".exr");
    mat_string.append(".txt");

    std::ofstream of(mat_string);

    surfConcentration->copyBackToCPU();
    std::vector<float> images[3];
    images[0].resize(nPhi * nTheta);
    images[1].resize(nPhi * nTheta);
    images[2].resize(nPhi * nTheta);

    for (size_t j = 0; j < nTheta; ++j) {
	for (size_t i = 0; i < nPhi; ++i) {
	    float con = surfConcentration->getCPUValueAt(i, j);
	  
	    images[0][j*nPhi+i] = (con - 0.9) / 0.2;
	    images[1][j*nPhi+i] = (con - 0.9) / 0.2;
	    images[2][j*nPhi+i] = (con - 0.9) / 0.2;
	    of << con << " ";
	}
	of << std::endl;
    }

    write_image(img_string, nPhi, nTheta, images);
} 


void KaminoSolver::write_thickness_img(const std::string& s, const int frame)
{
    std::string img_string = std::to_string(frame);
    while (img_string.length() < 4) {
	img_string.insert(0, "0");
    }
    img_string.insert(0, s + "frame");
    std::string mat_string = img_string;
    img_string.append(".exr");
    mat_string.append(".txt");

    std::ofstream of(mat_string);
    std::ofstream thick;
    thick.open("thickness.txt", std::ofstream::out | std::ofstream::app);

    //    if (frame != 0) {
    // if (false) {
    // 	dim3 gridLayout;
    // 	dim3 blockLayout;
    // 	determineLayout(gridLayout, blockLayout, rows, cols);
    // 	resetThickness<<<gridLayout, blockLayout>>>(weightFull);
    // 	checkCudaErrors(cudaGetLastError());
    // 	checkCudaErrors(cudaDeviceSynchronize());
	
    //     if (particles->numOfParticles > 0) {
    // 	    determineLayout(gridLayout, blockLayout, 2, particles->numOfParticles);
    // 	    upsampleParticles<<<gridLayout, blockLayout>>>
    // 		(particles->coordGPUThisStep, particles->value, weightFull, particles->numOfParticles);
    // 	    checkCudaErrors(cudaGetLastError());
    // 	    checkCudaErrors(cudaDeviceSynchronize());
    // 	}
	
    // 	determineLayout(gridLayout, blockLayout, rows, cols);
    // 	normalizeThickness<<<gridLayout, blockLayout>>>
    // 	    (thicknessFull, thickness->getGPUThisStep(), weightFull, thickness->getThisStepPitchInElements());
    // 	checkCudaErrors(cudaGetLastError());
    // 	checkCudaErrors(cudaDeviceSynchronize());
    // 	cudaMemcpy(thicknessFullCPU, thicknessFull, (cols * rows) * sizeof(float), cudaMemcpyDeviceToHost);

    // 	std::vector<float> images[3];
    // 	images[0].resize(cols * rows);
    // 	images[1].resize(cols * rows);
    // 	images[2].resize(cols * rows);

    // 	for (size_t j = 0; j < rows; ++j) {
    // 	    for (size_t i = 0; i < cols; ++i) {
    // 		float Delta = thicknessFullCPU[j * cols + i];
    // 		if (Delta < 0) { 
    // 		    this->setBroken(true);
    // 		    return;
    // 		} else {
    // 		    images[0][j * cols + i] = Delta * this->H * 5e5;
    // 		    images[1][j * cols + i] = Delta * this->H * 5e5;
    // 		    images[2][j * cols + i] = Delta * this->H * 5e5; // *4
    // 		    of << Delta * this->H * 2<< " ";
    // 		}
    // 	    }
    // 	    of << std::endl;
    // 	}

    // 	write_image(img_string, cols, rows, images);
    // } else 
    {
	thickness->copyBackToCPU();

	std::vector<float> images[3];
	images[0].resize(nPhi * nTheta);
	images[1].resize(nPhi * nTheta);
	images[2].resize(nPhi * nTheta);

	float minE = 1.f;
	float ratio = 2e5; // * 10

	for (size_t j = 0; j < nTheta; ++j) {
	    for (size_t i = 0; i < nPhi; ++i) {
		float Delta = thickness->getCPUValueAt(i, j);
		if (Delta < 0) {
		    this->setBroken(true);
		}    //		    return;
		//} else {
		if (Delta < minE)
		    minE = Delta;
		images[0][j*nPhi+i] = Delta * this->H * ratio;
		images[1][j*nPhi+i] = Delta * this->H * ratio;
		images[2][j*nPhi+i] = Delta * this->H * ratio; 
		of << Delta * this->H * 2 << " ";
		//}
	    }
	    of << std::endl;
	}
	thick << minE << std::endl;
	thick.close();

	write_image(img_string, nPhi, nTheta, images);
	std::cout << "min thickness " << minE << std::endl;
    }    
}


float KaminoSolver::getGridLen() {
    return this->gridLen;
}
