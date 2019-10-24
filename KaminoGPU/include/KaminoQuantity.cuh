# pragma once

# include "KaminoHeader.cuh"

class KaminoQuantity
{
protected:
    /* Name of the attribute */
    std::string attrName;

    /* Grid dimensions */
    size_t nPhi;
    size_t nTheta;

    /* Grid size */
    float gridLen;
    /* 1.0 / gridlen */
    float invGridLen;

    /* Staggered? */
    float phiOffset;
    float thetaOffset;

    /* Initial buffer at client side */
    float* cpuBuffer;
    /* Double pitch buffer at server side */
    float* gpuThisStep;
    size_t thisStepPitch;
    float* gpuNextStep;
    size_t nextStepPitch;

    //cudaChannelFormatDesc desc;
    /* Get index */
    //size_t getIndex(size_t phi, size_t theta);

public:
    /* Constructor */
    KaminoQuantity(std::string attributeName, size_t nPhi, size_t nTheta,
		   float phiOffset, float thetaOffset);
    /* Destructor */
    ~KaminoQuantity();

    /* Swap the GPU buffer */
    void swapGPUBuffer();
    /* Copy the CPU end part to GPU */
    void copyToGPU();
    /* Copy backwards */
    void copyBackToCPU();
	
    /* Get its name */
    std::string getName();
    /* Get phi dimension size */
    size_t getNPhi();
    /* Get theta dimension size */
    size_t getNTheta();
    /* Get the current step */
    float getCPUValueAt(size_t x, size_t y);
    /* Set the current step */
    void setCPUValueAt(size_t x, size_t y, float val);
    /* Access */
    float& accessCPUValueAt(size_t x, size_t y);
    /* Get the offset */
    float getPhiOffset();
    float getThetaOffset();
    float* getGPUThisStep();
    float* getGPUNextStep();

    size_t getThisStepPitchInElements();
    size_t getNextStepPitchInElements();

    static cudaChannelFormatDesc channelFormat;
};


class ScalarQuantity: public KaminoQuantity {
private:
    /* initial value */
    float* gpuInit;
public:
    /* Constructor */
    ScalarQuantity(std::string attributeName, size_t nPhi, size_t nTheta,
		   float phiOffset, float thetaOffset);
    /* Destructor */
    ~ScalarQuantity();

    float* getGPUInit();
};