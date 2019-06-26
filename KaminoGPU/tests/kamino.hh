# pragma once
# include "KaminoHeader.cuh"

class KaminoTest {
public:
    size_t nTheta;
    size_t nPhi;
    float gridLen;
    float invGridLen;
    KaminoTest(size_t nTheta);
    ~KaminoTest();
    float sampleVPhi(float* input, float phiRaw, float thetaRaw, size_t pitch);
    float sampleVTheta(float* input, float phiRaw, float thetaRaw, size_t pitch);
    float sampleCentered(float* input, float phiRaw, float thetaRaw, size_t pitch);
    float kaminoLerp(float from, float to, float alpha);
    bool validateCoord(float& phi, float& theta);
};
