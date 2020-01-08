# include "../include/KaminoSolver.cuh"
# include "../opencv_headers/opencv2/opencv.hpp"
# include "../include/KaminoTimer.cuh"
# include "../include/tinyexr.h"

const int fftRank = 1;
static const int m = 3;
static __constant__ size_t nPhiGlobal;
static __constant__ size_t nThetaGlobal;


__global__ void initMapping(fReal* map_theta, fReal* map_phi){
    // Index
    int splitVal = (nPhiGlobal + blockDim.x - 1) / blockDim.x;
    int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
    int thetaId = blockIdx.x / splitVal;
    if (phiId >= nPhiGlobal) return;

    map_theta[thetaId * nPhiGlobal + phiId] = (fReal)thetaId + centeredThetaOffset;
    map_phi[thetaId * nPhiGlobal + phiId] = (fReal)phiId + centeredPhiOffset;
}


__global__ void initLinearSystem(int* row_ptr, int* col_ind) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > nPhiGlobal * nThetaGlobal) return;
    int idx5 = 5 * idx;
    row_ptr[idx] = idx5;
    
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


KaminoSolver::KaminoSolver(size_t nPhi, size_t nTheta, fReal radius, fReal dt,
			   fReal H, int device, std::string AMGconfig, size_t particleDensity) :
    nPhi(nPhi), nTheta(nTheta), radius(radius), invRadius(1.0/radius), gridLen(M_2PI / nPhi),
    invGridLen(1.0 / gridLen), timeStep(dt), timeElapsed(0.0), advectionTime(0.0),
    bodyforceTime(0.0), CGTime(0.0), H(H), epsilon(H/radius), N(nPhi*nTheta), nz(5*N),
    particleDensity(particleDensity), device(device)
{
    /// FIXME: Should we detect and use device 0?
    /// Replace it later with functions from helper_cuda.h!
    checkCudaErrors(cudaSetDevice(device));

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device));
    // otherwise no enough resource
    this->nThreadxMax = std::min(deviceProp.maxThreadsDim[0], 256);

    checkCudaErrors(cudaMalloc((void **)(&div),
			       sizeof(fReal) * N));
    checkCudaErrors(cudaMalloc((void **)(&weight),
			       sizeof(fReal2) * N));
    checkCudaErrors(cudaMalloc((void **)(&uair),
			       sizeof(fReal) * N));
    checkCudaErrors(cudaMalloc((void **)(&vair),
			       sizeof(fReal) * (N - nPhi)));
    checkCudaErrors(cudaMalloc((void **)(&row_ptr),
			       sizeof(int) * (N + 1)));
    checkCudaErrors(cudaMalloc((void **)(&col_ind),
			       sizeof(int) * nz));
    checkCudaErrors(cudaMalloc((void **)(&rhs),
			       sizeof(fReal) * N));
    checkCudaErrors(cudaMalloc((void **)(&val),
			       sizeof(fReal) * nz));

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

    /* AMGCL */
    ptr.resize(N + 1);
    col.resize(nz);
    val_cpu.resize(nz);
    rhs_cpu.resize(N);
    xx.resize(N);

    CHECK_CUDA(cudaMemcpy(ptr.data(), row_ptr, sizeof(int) * (N + 1), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(col.data(), col_ind, sizeof(int) * nz, cudaMemcpyDeviceToHost));

    // printGPUarraytoMATLAB<int>("test/row_ptr.txt", row_ptr, N + 1, 1, 1);
    // printGPUarraytoMATLAB<int>("test/col_ind.txt", col_ind, N, 5, 5);
    
    // Device memory management
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(fReal)));
    CHECK_CUDA(cudaMalloc(&d_r, N * sizeof(fReal)));
    
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
# ifdef USEFLOAT
    mode = AMGX_mode_dFFI;
# else
    mode = AMGX_mode_dDDI;
# endif
    AMGX_matrix_create(&A, res, mode);
    AMGX_vector_create(&b, res, mode);
    AMGX_vector_create(&x, res, mode);
    AMGX_solver_create(&solver, res, mode, cfg);

    /* Bimocq mapping buffers */
    CHECK_CUDA(cudaMalloc(&forward_p, N * sizeof(fReal)));
    CHECK_CUDA(cudaMalloc(&forward_t, N * sizeof(fReal)));
    CHECK_CUDA(cudaMalloc(&backward_p, N * sizeof(fReal)));
    CHECK_CUDA(cudaMalloc(&backward_t, N * sizeof(fReal)));
    CHECK_CUDA(cudaMalloc(&backward_pprev, N * sizeof(fReal)));
    CHECK_CUDA(cudaMalloc(&backward_tprev, N * sizeof(fReal)));
    CHECK_CUDA(cudaMalloc(&tmp_p, N * sizeof(fReal)));
    CHECK_CUDA(cudaMalloc(&tmp_t, N * sizeof(fReal)));
    
    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    initMapping<<<gridLayout, blockLayout>>>(forward_t, forward_p);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(backward_p, forward_p, N * sizeof(fReal),
			  cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(backward_pprev, forward_p, N * sizeof(fReal),
			  cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(backward_t, forward_t, N * sizeof(fReal),
			  cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(backward_tprev, forward_t, N * sizeof(fReal),
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

    CHECK_CUDA(cudaFree(d_r));
    CHECK_CUDA(cudaFree(d_x));

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
    fReal totalTimeUsed = this->advectionTime + this->bodyforceTime;
    std::cout << "Total time used for advection : " << this->advectionTime << std::endl;
    std::cout << "Total time used for body force : " << this->bodyforceTime << std::endl;
    std::cout << "Total time used for CG : " << CGTime << std::endl;
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


void KaminoSolver::adjustStepSize(fReal& dt, const fReal& U, const fReal& epsilon) {
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


void KaminoSolver::stepForward(fReal dt) {
    fReal dt_ = this->timeStep;
    this->timeStep = dt;
    stepForward();
    this->timeStep = dt_;
}


void KaminoSolver::stepForward() {
# ifdef PERFORMANCE_BENCHMARK
    KaminoTimer timer;
    timer.startTimer();
# endif
# ifdef BIMOCQ
    //    updateCFL();
    updateForward(this->timeStep, forward_t, forward_p);
    updateBackward(this->timeStep, backward_t, backward_p);
# endif

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
    count++;
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
    // check if film is broken
    setBroken(true);
    for (size_t i = 0; i < this->velPhi->getNPhi(); ++i) {
	for (size_t j = 0; j < this->velPhi->getNTheta(); ++j) {
	    fReal val = this->velPhi->getCPUValueAt(i, j);
	    if (!isnan(val)) {
		setBroken(false);
		goto finish;
	    }
	}
    }
    for (size_t i = 0; i < this->velTheta->getNPhi(); ++i) {
	for (size_t j = 0; j < this->velTheta->getNTheta(); ++j) {
	    fReal val = this->velTheta->getCPUValueAt(i, j);
	    if (!isnan(val)) {
		setBroken(false);
		goto finish;
	    }
	}
    }
    finish:
}


bool KaminoSolver::isBroken() {
    return this->broken;
}


void KaminoSolver::setBroken(bool broken) {
    this->broken = broken;
}


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

# ifdef WRITE_TXT
    std::string u_string = img_string;
    std::string v_string = img_string;
    u_string.append("u.txt");
    v_string.append("v.txt");

    std::ofstream ofu(u_string);
    std::ofstream ofv(v_string);
# endif

    img_string.append(".exr");

    copyVelocityBack2CPU();
    std::vector<float> images[3];
    images[0].resize(nPhi * nTheta);
    images[1].resize(nPhi * nTheta);
    images[2].resize(nPhi * nTheta);

    fReal maxu = std::numeric_limits<fReal>::min();
    fReal maxv = std::numeric_limits<fReal>::min();
    fReal minu = std::numeric_limits<fReal>::max();
    fReal minv = std::numeric_limits<fReal>::max();
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
	    fReal uW = velPhi->getCPUValueAt(i, j);
	    fReal uE;
	    fReal vN;
	    fReal vS;
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
	    fReal u = 0.5 * (uW + uE);
	    fReal v = 0.5 * (vN + vS);
# ifdef WRITE_TXT
	    ofu << u << " ";
	    ofv << v << " ";
# endif
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
	    images[0][j*nPhi+i] = float(u/2+0.5); // R
	    images[1][j*nPhi+i] = float(v/2+0.5); // G
	    images[2][j*nPhi+i] = float(0.5); // B
	}
# ifdef WRITE_TXT
	ofu << std::endl;
	ofv << std::endl;
# endif
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

# ifdef WRITE_TXT
    std::string mat_string = img_string;
    mat_string.append(".txt");
    std::ofstream of(mat_string);
# endif

    img_string.append(".exr");

    surfConcentration->copyBackToCPU();
    std::vector<float> images[3];
    images[0].resize(nPhi * nTheta);
    images[1].resize(nPhi * nTheta);
    images[2].resize(nPhi * nTheta);

    for (size_t j = 0; j < nTheta; ++j) {
	for (size_t i = 0; i < nPhi; ++i) {
	    fReal con = surfConcentration->getCPUValueAt(i, j);
	  
	    images[0][j*nPhi+i] = (con - 0.9) / 0.2;
	    images[1][j*nPhi+i] = (con - 0.9) / 0.2;
	    images[2][j*nPhi+i] = (con - 0.9) / 0.2;
# ifdef WRITE_TXT
	    of << con << " ";
# endif
	}
# ifdef WRITE_TXT
	of << std::endl;
# endif
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

# ifdef WRITE_TXT
    std::string mat_string = img_string;
    mat_string.append(".txt");

    std::ofstream of(mat_string);
# endif
    std::ofstream thick;
    thick.open("thickness1024_vdw.txt", std::ofstream::out | std::ofstream::app);

    img_string.append(".exr");
    thickness->copyBackToCPU();

    std::vector<float> images[3];
    images[0].resize(nPhi * nTheta);
    images[1].resize(nPhi * nTheta);
    images[2].resize(nPhi * nTheta);

    fReal minE = 1.0;
    fReal ratio = 2e5; // * 10
    this->setBroken(true);
    for (size_t j = 0; j < nTheta; ++j) {
	for (size_t i = 0; i < nPhi; ++i) {
	    fReal Delta = thickness->getCPUValueAt(i, j);
	    if (Delta > 0) {
		this->setBroken(false);
	    }    //		    return;
	    //} else {
	    if (Delta < minE)
		minE = Delta;
	    images[0][j*nPhi+i] = Delta * this->H * ratio;
	    images[1][j*nPhi+i] = Delta * this->H * ratio;
	    images[2][j*nPhi+i] = Delta * this->H * ratio;
# ifdef WRITE_TXT
	    of << Delta * this->H * 2 << " ";
# endif
	    //}
	}
# ifdef WRITE_TXT
	of << std::endl;
# endif
    }
    // # ifdef WRITE_TXT
    thick << minE << std::endl;
    thick.close();
    //# endif

    write_image(img_string, nPhi, nTheta, images);
    std::cout << "min thickness " << minE << std::endl;
}


fReal KaminoSolver::getGridLen() {
    return this->gridLen;
}
