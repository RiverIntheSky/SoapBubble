# pragma once

# include <amgx_c.h>
# include "../include/KaminoGPU.cuh"
# include "../include/KaminoQuantity.cuh"
# include "../include/KaminoParticles.cuh"
# include <iomanip>

extern __constant__ float invGridLenGlobal;

__device__ float kaminoLerp(float from, float to, float alpha);
__device__ float sampleCentered(float* input, float2 rawId, size_t pitch);
__global__ void resetThickness(float2* weight);
__global__ void initLinearSystem(int* row_ptr, int* col_ind);
__global__ void initMapping(float* map_theta, float* map_phi);

class KaminoSolver
{
private:
    // Handle for batched FFT
    cufftHandle kaminoPlan;

    KaminoParticles* particles;
    size_t particleDensity;

    // Buffer for U, the fouriered coefs
    // This pointer's for the pooled global memory (nTheta by nPhi)
    ComplexFourier* gpuUFourier;
    float* gpuUReal;
    float* gpuUImag;

    // Buffer for V, the fouriered coefs
    // This pointer's for the pooled global memory as well
    ComplexFourier* gpuFFourier;
    float* gpuFReal;
    float* gpuFImag;
    float* gpuFZeroComponent;
    
    // Buffer for elements that can be preallocated
    float* div;
    float2* weight;

    // original resolution
    float2* weightFull;
    float* thicknessFull;
    float* thicknessFullCPU;
    int cols;
    int rows;

    /// Precompute these!
    // nPhi by nTheta elements, but they should be retrieved by shared memories
    // in the TDM kernel we solve nTheta times with each time nPhi elements.
    float* gpuA;
    // Diagonal elements b (major diagonal);
    float* gpuB;
    // Diagonal elements c (upper);
    float* gpuC;
    void precomputeABCCoef();

    /* Grid dimensions */
    size_t nPhi;
    size_t nTheta;
    /* Cuda dimensions */
    size_t nThreadxMax;
    int device;
    /* Radius of sphere */
    float radius;
    /* Inverted radius of sphere */
    float invRadius;
    /* Grid size */
    float gridLen;
    /* Inverted grid size*/
    float invGridLen;
    /* Whether film is broken */
    bool broken = false;

    /* harmonic coefficients for velocity field initializaton */
    //    int B, C, D, E;

    /* average film thickness */
    float H;
    /* expansion parameter */
    float epsilon;

    /* cuSPARSE and cuBLAS*/
    int N;
    int nz;
    cublasHandle_t cublasHandle;
    cusparseHandle_t cusparseHandle;
    cusparseDnVecDescr_t vecR, vecX, vecP, vecO;
    /* Description of the A matrix*/
    cusparseMatDescr_t descrA;
    
    const float zero = 0.f;
    const float one = 1.f;
    const float minusone = -1.f;
    float r0, r1, beta, alpha, nalpha, dot;

    float *d_r, *d_p, *d_omega, *d_x;
    int *row_ptr, *col_ind;
    float *rhs, *val;

    cusparseSpMatDescr_t matM; /* pre-conditioner */

    BimocqQuantity* velTheta;
    BimocqQuantity* velPhi; // u_phi/(sin(theta)) is stored instead of u_phi
    BimocqQuantity* thickness;
    BimocqQuantity* surfConcentration;

    size_t pitch; // all the quantities have the same padding
    void copyVelocity2GPU();
    void copyVelocityBack2CPU();
    void copyDensity2GPU();
    void copyDensityBack2CPU();

    /* Something about time steps */
    float timeStep;
    float timeElapsed;

    float advectionTime;
    float bodyforceTime;
    float CGTime;

    /* CFL */
    float cfldt;
    float maxu;
    float maxv;

    /* AMGX */
    AMGX_Mode mode;
    AMGX_config_handle cfg;
    AMGX_resources_handle res;
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;
    //status handling
    AMGX_SOLVE_STATUS status;

    /* AMGCL */
    std::vector<int> ptr, col;
    std::vector<float> val_cpu, rhs_cpu, xx;

    /* Bimocq mapping buffers */
    float *forward_p, *forward_t,
	*backward_p, *backward_t,
	*backward_pprev, *backward_tprev,
	*tmp_p, *tmp_t;
    void updateCFL();

    /// Kernel calling from here
    void advection();
    void bodyforce();
    void conjugateGradient();
    void AlgebraicMultiGridCG();
    void AMGCLSolve();

    // Swap all these buffers of the attributes.
    void swapVelocityBuffers();

    /* distribute initial velocity values at grid points */
    void initialize_velocity();
	
    /* FBM noise function for velocity distribution */
    float FBM(const float x, const float y);
    /* 2D noise interpolation function for smooth FBM noise */
    float interpNoise2D(const float x, const float y) const;
    /* returns a pseudorandom number between -1 and 1 */
    float rand(const vec2 vecA) const;
    /* determine the layout of the grids and blocks */
    void determineLayout(dim3& gridLayout, dim3& blockLayout,
			 size_t nRow_theta, size_t nCol_phi);
public:
    KaminoSolver(size_t nPhi, size_t nTheta, float radius, float frameDuration,
		 float H, int device, std::string AMGconfig, size_t particleDensity);
    ~KaminoSolver();

    void initWithConst(KaminoQuantity* attrib, float val);
    void initWithConst(BimocqQuantity* attrib, float val);
    void initThicknessfromPic(std::string path);
    void initParticlesfromPic(std::string path, size_t parPergrid);

    void copyToCPU(KaminoQuantity* quantity, float* cpubuffer);
    float maxAbsDifference(const float* A, const float* B, const size_t& size);
    void adjustStepSize(float& timeStep, const float& U, const float& eps);
    void stepForward(float timeStep);
    void stepForward();
    float getGridLen();
    bool isBroken();
    void setBroken(bool broken);

    /* help functions */
    float maxAbs(float* array, size_t nTheta, size_t nPhi);
    void write_thickness_img(const std::string& s, const int frame);
    // void write_data_bgeo(const std::string& s, const int frame);
    void write_image(const std::string& s, size_t width, size_t height, std::vector<float> *images);
    void write_velocity_image(const std::string& s, const int frame);
    void write_concentration_image(const std::string& s, const int frame);
    template <typename T>
    void printGPUarray(std::string repr, T* vec, int len);
    template <typename T>
    void printGPUarraytoMATLAB(std::string filename, T* vec, int num_row, int num_col,
			       size_t pitch);

    /* Bimocq */
    void updateForward(float dt, float* &fwd_t, float* &fwd_p);
    void updateBackward(float dt, float* &bwd_t, float* &bwd_p);
    float estimateDistortion();
    void reInitializeMapping();
    void correctMapping();
    bool validateCoord(double2& Id);

    int count = 0;
};


template <typename T>
void KaminoSolver::printGPUarray(std::string repr, T* vec, int len) {
    T* cpuvec = new T[len];
    CHECK_CUDA(cudaMemcpy(cpuvec, vec, len * sizeof(T),
			  cudaMemcpyDeviceToHost));
    std::cout << repr << std::endl;
    for (int i = 0; i < len; i++)
	std::cout << cpuvec[i] << " ";
    std::cout << std::endl;

    delete[] cpuvec;
}


template <typename T>
void KaminoSolver::printGPUarraytoMATLAB(std::string filename, T* vec, int num_row,
					 int num_col, size_t pitch) {
    std::ofstream of(filename);
    int len = num_row * num_col;
    T* cpuvec = new T[len];

    CHECK_CUDA(cudaMemcpy2D(cpuvec, num_col * sizeof(T), vec,
			    pitch * sizeof(T),
			    num_col * sizeof(T), num_row,
			    cudaMemcpyDeviceToHost));

    for (int row = 0; row < num_row; row++) {
	for (int col = 0; col < num_col; col++) {
	    of << std::setprecision(10) << cpuvec[row * num_col + col] << " ";
	}
	of << "\n";
    }

    delete[] cpuvec;
}
