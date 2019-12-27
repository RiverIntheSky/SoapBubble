# include "../include/KaminoSolver.cuh"

void KaminoSolver::initialize_velocity()
{
    std::cout << "Initializing velocity..." << std::endl;
    KaminoQuantity* u = this->velPhi;
    KaminoQuantity* v = this->velTheta;

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


void KaminoSolver::initWithConst(KaminoQuantity* attrib, fReal val)
{
    for (size_t i = 0; i < attrib->getNPhi(); ++i) {
	for (size_t j = 0; j < attrib->getNTheta(); ++j) {
		attrib->setCPUValueAt(i, j, val);
	    }
    }
    attrib->copyToGPU();
}


void KaminoSolver::initWithConst(BimocqQuantity* attrib, fReal val)
{
    for (size_t i = 0; i < attrib->getNPhi(); ++i) {
	for (size_t j = 0; j < attrib->getNTheta(); ++j) {
		attrib->setCPUValueAt(i, j, val);
	    }
    }
    attrib->copyToGPU();
    CHECK_CUDA(cudaMemcpy(attrib->getGPUInit(), attrib->getGPUThisStep(),
			  pitch * attrib->getNTheta() *
			  sizeof(fReal), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(attrib->getGPUInitLast(), attrib->getGPUThisStep(),
			  pitch * attrib->getNTheta() *
			  sizeof(fReal), cudaMemcpyDeviceToDevice));
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
	    	    fReal C = (fReal)p->x; // Gray Scale
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
			  sizeof(fReal), cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaMemcpy(this->thickness->getGPUInitLast(), this->thickness->getGPUThisStep(),
			  this->thickness->getThisStepPitchInElements() * this->thickness->getNTheta() *
			  sizeof(fReal), cudaMemcpyDeviceToDevice));
}