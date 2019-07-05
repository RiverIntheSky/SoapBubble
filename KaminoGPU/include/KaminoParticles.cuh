# pragma once

# include "../include/KaminoHeader.cuh"
# include "../opencv_headers/opencv2/opencv.hpp"

class KaminoParticles
{
public:
    size_t nPhi;
    size_t nTheta;
    size_t particlePGrid;
    size_t numOfParticles;

    fReal* coordCPUBuffer;
    fReal* value;
    fReal* coordGPUThisStep;
    fReal* coordGPUNextStep;

    KaminoParticles(std::string path, size_t particleDensity, fReal gridLen, size_t nTheta);
    ~KaminoParticles();

    void copy2GPU();
    void copyBack2CPU();
    void swapGPUBuffers();
};