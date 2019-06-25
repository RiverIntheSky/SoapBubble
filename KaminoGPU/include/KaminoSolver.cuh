# pragma once

# include "../include/KaminoQuantity.cuh"
# include "../include/KaminoParticles.cuh"

static __constant__ size_t nPhiGlobal;
static __constant__ size_t nThetaGlobal;
static __constant__ fReal invRadiusGlobal;
static __constant__ fReal radiusGlobal;
static __constant__ fReal timeStepGlobal;
static __constant__ fReal gridLenGlobal;
static __constant__ fReal invGridLenGlobal;
static __constant__ fReal SGlobal;
static __constant__ fReal MGlobal;
static __constant__ fReal reGlobal;
static __constant__ fReal gGlobal;
static __constant__ fReal DsGlobal;

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

    // Buffer for elements that can be precomputed
    fReal* div;

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
    bool broken;

    /* harmonic coefficients for velocity field initializaton */
    fReal A;
    int B, C, D, E;
    fReal H;

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
    float geometricTime;
    float bodyforceTime;
    float projectionTime;

    /// Kernel calling from here
    void advection(fReal& timeStep);
    void geometric();
    void bodyforce();
    void projection();

    // Swap all these buffers of the attributes.
    void swapVelocityBuffers();

    /* distribute initial velocity values at grid points */
    void initialize_velocity();

    void mapPToSphere(vec3& pos) const;
    void mapVToSphere(vec3& pos, vec3& vel) const;
	
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
		 fReal A, int B, int C, int D, int E, fReal H);
    ~KaminoSolver();

    void initWithConst(KaminoQuantity* attrib, fReal val);
    void initThicknessfromPic(std::string path);
    void initParticlesfromPic(std::string path, size_t parPergrid);

    void copyToCPU(KaminoQuantity* quantity, fReal* cpubuffer);
    fReal maxAbsDifference(const fReal* A, const fReal* B, const size_t& size);
    void adjustStepSize(fReal& timeStep, const fReal& eps);
    void stepForward(fReal timeStep);
    bool isBroken();
    void setBroken(bool broken);

    void write_thickness_img(const std::string& s, const int frame);
    void write_data_bgeo(const std::string& s, const int frame);
    void write_particles_bgeo(const std::string& s, const int frame);

    KaminoParticles* particles;
};