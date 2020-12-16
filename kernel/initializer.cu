# include "../include/solver.cuh"

void Solver::initialize_velocity()
{
    std::cout << "Initializing velocity..." << std::endl;
    Quantity* u = this->velPhi;
    Quantity* v = this->velTheta;

    for (size_t j = 0; j < size_t(v->getNTheta()); ++j) {
    	for (size_t i = size_t(v->getNPhi() / 8); i < size_t(v->getNPhi() / 8 * 3); ++i) {
    	    v->setCPUValueAt(i, j, -M_PI/u->getNTheta() * radius);
    	}
    	for (size_t i = size_t(v->getNPhi() * 5 / 8); i < size_t(v->getNPhi() / 8 * 7); ++i) {
    	    v->setCPUValueAt(i, j, M_PI/u->getNTheta() * radius);
    	}
    }
    
    copyVelocity2GPU();
}


void Solver::initWithConst(Quantity* attrib, fReal val)
{
    for (size_t i = 0; i < attrib->getNPhi(); ++i) {
	for (size_t j = 0; j < attrib->getNTheta(); ++j) {
		attrib->setCPUValueAt(i, j, val);
	    }
    }
    attrib->copyToGPU();
}


void Solver::initWithConst(BimocqQuantity* attrib, fReal val)
{
    for (size_t i = 0; i < attrib->getNPhi(); ++i) {
	for (size_t j = 0; j < attrib->getNTheta(); ++j) {
		attrib->setCPUValueAt(i, j, val);
	    }
    }
    attrib->copyToGPU();
    checkCudaErrors(cudaMemcpy(attrib->getGPUInit(), attrib->getGPUThisStep(),
			  pitch * attrib->getNTheta() *
			  sizeof(fReal), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(attrib->getGPUInitLast(), attrib->getGPUThisStep(),
			  pitch * attrib->getNTheta() *
			  sizeof(fReal), cudaMemcpyDeviceToDevice));
}


void Solver::initThicknessfromPic(std::string path)
{
    if (path == "") {
	std::cout << "No thickness image provided, initialize with eta = 0.5" << std::endl;
    } else {
	cv::Mat image_In, image_Resized;
	image_In = cv::imread(path, cv::IMREAD_UNCHANGED);
	if (!image_In.data) {
	    std::cout << "No thickness image provided, initialize with eta = 0.5" << std::endl;
	} else {
	    cv::Size size(nPhi, nTheta);
	    cv::resize(image_In, image_Resized, size);

	    for (size_t i = 0; i < nPhi; ++i) {
	    	for (size_t j = 0; j < nTheta; ++j) {
	    	    cv::Point3_<float>* p = image_Resized.ptr<cv::Point3_<float>>(j, i);
	    	    fReal C = (fReal)p->x; // Gray Scale
		    // rescaling thickness
		    C = (C - 0.5) * 0.3 + 0.5;
		    this->thickness->setCPUValueAt(i, j, C);
	    	}
	    }
	    this->thickness->copyToGPU();
       	}
    }

    checkCudaErrors(cudaMemcpy(this->thickness->getGPUInit(), this->thickness->getGPUThisStep(),
			  this->thickness->getThisStepPitchInElements() * this->thickness->getNTheta() *
			  sizeof(fReal), cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemcpy(this->thickness->getGPUInitLast(), this->thickness->getGPUThisStep(),
			  this->thickness->getThisStepPitchInElements() * this->thickness->getNTheta() *
			  sizeof(fReal), cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemset(this->thickness->getGPUDelta(), 0,
			       pitch * sizeof(fReal) * this->thickness->getNTheta()));

    checkCudaErrors(cudaMemset(this->thickness->getGPUDeltaLast(), 0,
			       pitch * sizeof(fReal) * this->thickness->getNTheta()));

    dim3 gridLayout;
    dim3 blockLayout;
    determineLayout(gridLayout, blockLayout, nTheta, nPhi);
    initMapping<<<gridLayout, blockLayout>>>(forward_t, forward_p);
    initMapping<<<gridLayout, blockLayout>>>(backward_t, backward_p);
    initMapping<<<gridLayout, blockLayout>>>(backward_tprev, backward_pprev);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}


void Solver::initAirflowfromPic(std::string path)
{
    if (path == "") {
	std::cout << "No airflow image provided, initialize with zero" << std::endl;
    } else {
	cv::Mat image_In, image_Resized;
	image_In = cv::imread(path, cv::IMREAD_UNCHANGED);
	if (!image_In.data) {
	    std::cout << "No airflow image provided, initialize with zero" << std::endl;
	} else {
	    cv::Size size(nPhi, nTheta);
	    cv::resize(image_In, image_Resized, size);

	    fReal* uairCPU = new fReal[N];
	    fReal* vairCPU = new fReal[N - nPhi];
	    
	    for (size_t i = 0; i < nPhi; ++i) {
	    	for (size_t j = 0; j < nTheta; ++j) {
	    	    cv::Point3_<float>* p1 = image_Resized.ptr<cv::Point3_<float>>(j, i);
		    cv::Point3_<float>* p2 = image_Resized.ptr<cv::Point3_<float>>(j, (i + nPhi- 1) % nPhi);
		    uairCPU[j * nPhi + i] = 0.5 * (p1->y + p2->y) * radius;
		    if (j < nTheta - 1) {
			cv::Point3_<float>* p3 = image_Resized.ptr<cv::Point3_<float>>(j + 1, i);
			vairCPU[j * nPhi + i] = -0.5 * (p1->z + p3->z) * radius;
		    }
	    	}	
	    }

	    CHECK_CUDA(cudaMemcpy(uair_init, uairCPU, sizeof(fReal) * N,
				  cudaMemcpyHostToDevice));
	    CHECK_CUDA(cudaMemcpy(vair_init, vairCPU, sizeof(fReal) * (N - nPhi),
				  cudaMemcpyHostToDevice));
       	}
    }
}
