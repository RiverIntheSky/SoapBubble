# pragma once

# include "header.cuh"

class Quantity
{
protected:
    /* Name of the attribute */
    std::string attrName;

    /* Grid dimensions */
    size_t nPhi;
    size_t nTheta;

    /* Grid size */
    fReal gridLen;
    /* 1.0 / gridlen */
    fReal invGridLen;

    /* Staggered? */
    fReal phiOffset;
    fReal thetaOffset;

    /* Initial buffer at client side */
    fReal* cpuBuffer;
    /* Double pitch buffer at server side */
    fReal* gpuThisStep;
    size_t thisStepPitch;
    fReal* gpuNextStep;
    size_t nextStepPitch;

    //cudaChannelFormatDesc desc;
    /* Get index */
    //size_t getIndex(size_t phi, size_t theta);

public:
    /* Constructor */
    Quantity(std::string attributeName, size_t nPhi, size_t nTheta,
		   fReal phiOffset, fReal thetaOffset);
    /* Destructor */
    ~Quantity();

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
    fReal getCPUValueAt(size_t x, size_t y);
    /* Set the current step */
    void setCPUValueAt(size_t x, size_t y, fReal val);
    /* Access */
    fReal& accessCPUValueAt(size_t x, size_t y);
    /* Get the offset */
    fReal getPhiOffset();
    fReal getThetaOffset();
    fReal* getGPUThisStep();
    fReal* getGPUNextStep();

    size_t getThisStepPitchInElements();
    size_t getNextStepPitchInElements();

    static cudaChannelFormatDesc channelFormat;
};


class BimocqQuantity: public Quantity {
private:
    /* initial value */
    fReal* gpuInit;
    /* accumulated changes */
    fReal* gpuDelta;
    /* last value before reinitialization */
    fReal* gpuInitLast;
    /* last accumulated changes */
    fReal* gpuDeltaLast;
public:
    /* Constructor */
    BimocqQuantity(std::string attributeName, size_t nPhi, size_t nTheta,
		   fReal phiOffset, fReal thetaOffset);
    /* Destructor */
    ~BimocqQuantity();

    fReal*& getGPUInit();
    fReal*& getGPUDelta();
    fReal*& getGPUInitLast();
    fReal*& getGPUDeltaLast();
};