# pragma once

# include "KaminoQuantity.h"

class KaminoSolver
{
private:
	// Buffer for U, the fouriered coefs
	// This pointer's for the pooled global memory (nTheta by nPhi)
	Complex* gpuUPool;
	// Buffer for V, the fouriered coefs
	// This pointer's for the pooled global memory as well
	Complex* gpuFPool;

	/// Precompute these!
	// nPhi by nTheta elements, but they should be retrieved by shared memories
	// in the TDM kernel we solve nTheta times with each time nPhi elements.
	fReal* gpuA;
	// Diagonal elements b (major diagonal);
	fReal* gpuB;
	// Diagonal elements c (upper);
	fReal* gpuC;
	void precomputeABCCoef();

	/* Grid types */
	gridType* cpuGridTypesBuffer;
	gridType* gpuGridTypes;
	void copyGridType2GPU();

	/* Grid dimensions */
	size_t nPhi;
	size_t nTheta;
	/* Radius of sphere */
	fReal radius;
	/* Grid size */
	fReal gridLen;
	/* Inverted grid size*/
	fReal invGridLen;

	/* harmonic coefficients for velocity field initializaton */
	fReal A;
	int B, C, D, E;

	/* So that it remembers all these attributes within */
	//std::map<std::string, KaminoQuantity*> centeredAttr;
	//std::map<std::string, KaminoQuantity*> staggeredAttr;
	
	KaminoQuantity* velTheta;
	KaminoQuantity* velPhi;
	void copyVelocity2GPU();
	table2D texVelTheta;//bind to u
	table2D texVelPhi;//bind to v
	table2D texPressure;//bind to UPool after it's shifted
	void defineTextureTable();
	void bindVelocity2Tex();
	void bindPressure2Tex();

	/* Something about time steps */
	fReal frameDuration;
	fReal timeStep;
	fReal timeElapsed;

	// Is it solid? or fluid? or even air?
	gridType getGridTypeAt(size_t x, size_t y);

	/// Kernel calling from here
	void advection();
	void geometric();
	void projection();
	void bodyForce();

	void fillDivergence();
	void transformDivergence();
	void invTransformPressure();

	// Swap all these buffers of the attributes.
	void swapAttrBuffers();

	/* distribute initial velocity values at grid points */
	void initialize_velocity();
	/* which grids are solid? */
	void initialize_boundary();

	/* sum of sine functions for velocity initialization */
	fReal fPhi(const fReal x);
	/* */
	fReal gTheta(const fReal y);
	/* */
	fReal lPhi(const fReal x);
	/* */
	fReal mTheta(const fReal y);
	/* FBM noise function for velocity distribution */
	fReal FBM(const fReal x, const fReal y);
	/* 2D noise interpolation function for smooth FBM noise */
	fReal interpNoise2D(const fReal x, const fReal y) const;

	/* Tri-diagonal matrix solver */
	void TDMSolve(fReal* a, fReal* b, fReal* c, fReal* d);

public:
	
	KaminoSolver(size_t nPhi, size_t nTheta, fReal radius, fReal gridLength, fReal frameDuration,
		fReal A, int B, int C, int D, int E);
	~KaminoSolver();

	void stepForward(fReal timeStep);

	//void addCenteredAttr(std::string name, fReal phiOffset = 0.0, fReal thetaOffset = 0.5);
	//void addStaggeredAttr(std::string name, fReal phiOffset, fReal thetaOffset);

	KaminoQuantity* getAttributeNamed(std::string name);
	KaminoQuantity* operator[](std::string name);

	gridType* getGridTypeHandle();
	void write_data_bgeo(const std::string& s, const int frame);
};