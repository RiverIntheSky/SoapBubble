# pragma once

# include <amgx_c.h>
# include "../include/KaminoGPU.cuh"
# include "../include/KaminoQuantity.cuh"
# include "../include/KaminoParticles.cuh"
# include <iomanip>

extern __constant__ fReal invGridLenGlobal;

__device__ fReal kaminoLerp(fReal from, fReal to, fReal alpha);
__device__ fReal sampleCentered(fReal* input, fReal2 rawId, size_t pitch);
__global__ void initLinearSystem(int* row_ptr, int* col_ind);
__global__ void initMapping(fReal* map_theta, fReal* map_phi);

class KaminoSolver
{
private:
    // Handle for batched FFT
    cufftHandle kaminoPlan;

    KaminoParticles* particles;
    size_t particleDensity;
  
    // Buffer for elements that can be preallocated
    fReal* div;
    fReal2* weight;

    // original resolution
    fReal2* weightFull;
    fReal* thicknessFull;
    fReal* thicknessFullCPU;
    int cols;
    int rows;

    /* Grid dimensions */
    size_t nPhi;
    size_t nTheta;
    /* Cuda dimensions */
    size_t nThreadxMax;
    int device;
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

    /* airflow */
    fReal *uair, *vair;

    /* average film thickness */
    fReal H;
    /* expansion parameter */
    fReal epsilon;

    /* cuSPARSE and cuBLAS*/
    int N;
    int nz;
    
    fReal *d_r, *d_p, *d_omega, *d_x;
    int *row_ptr, *col_ind;
    fReal *rhs, *val;

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
    fReal timeStep;
    fReal timeElapsed;

    fReal advectionTime;
    fReal bodyforceTime;
    fReal CGTime;

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
    std::vector<fReal> val_cpu, rhs_cpu, xx;

    /* Bimocq mapping buffers */
    fReal *forward_p, *forward_t,
	*backward_p, *backward_t,
	*backward_pprev, *backward_tprev,
	*tmp_p, *tmp_t;
    void updateCFL(); // TODO: delete

    /// Kernel calling from here
    void advection();
    void bodyforce();
    //    void conjugateGradient();
    void AlgebraicMultiGridCG();
    void AMGCLSolve();

    // Swap all these buffers of the attributes.
    void swapVelocityBuffers();

    /* distribute initial velocity values at grid points */
    void initialize_velocity();
	
    /* determine the layout of the grids and blocks */
    void determineLayout(dim3& gridLayout, dim3& blockLayout,
			 size_t nRow_theta, size_t nCol_phi);
public:
    KaminoSolver(size_t nPhi, size_t nTheta, fReal radius, fReal frameDuration,
		 fReal H, int device, std::string AMGconfig, size_t particleDensity);
    ~KaminoSolver();

    void initWithConst(KaminoQuantity* attrib, fReal val);
    void initWithConst(BimocqQuantity* attrib, fReal val);
    void initThicknessfromPic(std::string path);
    void initParticlesfromPic(std::string path, size_t parPergrid);

    void copyToCPU(KaminoQuantity* quantity, fReal* cpubuffer);
    fReal maxAbsDifference(const fReal* A, const fReal* B, const size_t& size);
    void adjustStepSize(fReal& timeStep, const fReal& U, const fReal& eps);
    void stepForward(fReal timeStep);
    void stepForward();
    fReal getGridLen();
    bool isBroken();
    void setBroken(bool broken);

    /* help functions */
    fReal maxAbs(fReal* array, size_t nTheta, size_t nPhi);
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
    void updateForward(fReal dt, fReal* &fwd_t, fReal* &fwd_p);
    void updateBackward(fReal dt, fReal* &bwd_t, fReal* &bwd_p);
    fReal estimateDistortion();
    void reInitializeMapping();
    bool validateCoord(double2& Id);

    int count = 0;

    // TODO: delete
    /* CFL */
    fReal cfldt;
    fReal maxu;
    fReal maxv;
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
