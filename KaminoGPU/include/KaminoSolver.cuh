# pragma once

# include <amgx_c.h>
# include "../include/KaminoGPU.cuh"
# include "../include/KaminoQuantity.cuh"
# include "../include/KaminoParticles.cuh"

extern __constant__ fReal invGridLenGlobal;

__device__ bool validateCoord(fReal& phi, fReal& theta, size_t& nPhi);
__device__ fReal kaminoLerp(fReal from, fReal to, fReal alpha);
__device__ fReal sampleCentered(fReal* input, fReal phiRawId, fReal thetaRawId, size_t pitch);
__global__ void resetThickness(float2* weight);
__global__ void initLinearSystem(int* row_ptr, int* col_ind);

class KaminoSolver
{
private:
    // Handle for batched FFT
    cufftHandle kaminoPlan;

    // KaminoParticles* particles;

    // Buffer for U, the fouriered coefs
    // This pointer's for the pooled global memory (nTheta by nPhi)
    ComplexFourier* gpuUFourier;
    fReal* gpuUReal;
    fReal* gpuUImag;

    // Buffer for V, the fouriered coefs
    // This pointer's for the pooled global memory as well
    ComplexFourier* gpuFFourier;
    fReal* gpuFReal;
    fReal* gpuFImag;
    fReal* gpuFZeroComponent;
    
    // Buffer for elements that can be preallocated
    fReal* div;
    float2* weight;

    // original resolution
    float2* weightFull;
    fReal* thicknessFull;
    fReal* thicknessFullCPU;
    int cols;
    int rows;

    /// Precompute these!
    // nPhi by nTheta elements, but they should be retrieved by shared memories
    // in the TDM kernel we solve nTheta times with each time nPhi elements.
    fReal* gpuA;
    // Diagonal elements b (major diagonal);
    fReal* gpuB;
    // Diagonal elements c (upper);
    fReal* gpuC;
    void precomputeABCCoef();

    /* Grid dimensions */
    size_t nPhi;
    size_t nTheta;
    /* Cuda dimensions */
    size_t nThreadxMax;
    /* Radius of sphere */
    fReal radius;
    /* Inverted radius of sphere */
    fReal invRadius;
    /* Grid size */
    fReal gridLen;
    /* Inverted grid size*/
    fReal invGridLen;
    /* Whether film is broken */
    bool broken = false;

    /* harmonic coefficients for velocity field initializaton */
    //    int B, C, D, E;

    /* average film thickness */
    fReal H;
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

    /* So that it remembers all these attributes within */
    //std::map<std::string, KaminoQuantity*> centeredAttr;
    //std::map<std::string, KaminoQuantity*> staggeredAttr;

    KaminoQuantity* velTheta;
    KaminoQuantity* velPhi;
    KaminoQuantity* thickness;
    KaminoQuantity* bulkConcentration;
    KaminoQuantity* surfConcentration;
    KaminoQuantity* pressure;
    KaminoQuantity* density;
    void copyVelocity2GPU();
    void copyVelocityBack2CPU();
    void copyDensity2GPU();
    void copyDensityBack2CPU();

    /* Something about time steps */
    fReal frameDuration;
    fReal timeStep;
    fReal timeElapsed;

    float advectionTime;
    float bodyforceTime;
    float CGTime;

    // AMGX
    AMGX_Mode mode;
    AMGX_config_handle cfg;
    AMGX_resources_handle res;
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;
    //status handling
    AMGX_SOLVE_STATUS status;

    /// Kernel calling from here
    void advection(fReal& timeStep);
    void advection();
    void bodyforce();
    void conjugateGradient();
    void AlgebraicMultiGridCG();

    // Swap all these buffers of the attributes.
    void swapVelocityBuffers();

    /* distribute initial velocity values at grid points */
    void initialize_velocity();
	
    /* FBM noise function for velocity distribution */
    fReal FBM(const fReal x, const fReal y);
    /* 2D noise interpolation function for smooth FBM noise */
    fReal interpNoise2D(const fReal x, const fReal y) const;
    /* returns a pseudorandom number between -1 and 1 */
    fReal rand(const vec2 vecA) const;
    /* determine the layout of the grids and blocks */
    void determineLayout(dim3& gridLayout, dim3& blockLayout,
			 size_t nRow_theta, size_t nCol_phi);
public:
    KaminoSolver(size_t nPhi, size_t nTheta, fReal radius, fReal frameDuration,
		 fReal H, int device, std::string AMGconfig);
    ~KaminoSolver();

    void initWithConst(KaminoQuantity* attrib, fReal val);
    void initThicknessfromPic(std::string path, size_t particleDensity);
    void initParticlesfromPic(std::string path, size_t parPergrid);

    void copyToCPU(KaminoQuantity* quantity, fReal* cpubuffer);
    fReal maxAbsDifference(const fReal* A, const fReal* B, const size_t& size);
    void adjustStepSize(fReal& timeStep, const fReal& U, const fReal& eps);
    void stepForward(fReal timeStep);
    void stepForward();
    bool isBroken();
    void setBroken(bool broken);

    /* help functions */
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

    KaminoParticles* particles;
};


template <typename T>
void KaminoSolver::printGPUarray(std::string repr, T* vec, int len) {
    T cpuvec[len];
    CHECK_CUDA(cudaMemcpy(cpuvec, vec, len * sizeof(T),
			  cudaMemcpyDeviceToHost));
    std::cout << repr << std::endl;
    for (int i = 0; i < len; i++)
	std::cout << cpuvec[i] << " ";
    std::cout << std::endl;
}


template <typename T>
void KaminoSolver::printGPUarraytoMATLAB(std::string filename, T* vec, int num_row,
					 int num_col, size_t pitch) {
    std::ofstream of(filename);
    int len = num_row * num_col;
    T cpuvec[len];

    CHECK_CUDA(cudaMemcpy2D(cpuvec, num_col * sizeof(T), vec,
			    pitch * sizeof(T),
			    num_col * sizeof(T), num_row,
			    cudaMemcpyDeviceToHost));

    for (int row = 0; row < num_row; row++) {
	for (int col = 0; col < num_col; col++) {
	    of << cpuvec[row * num_col + col] << " ";
	}
	of << "\n";
    }
}
