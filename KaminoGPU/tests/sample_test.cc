# include "KaminoHeader.cuh"
# include "kamino.hh"
# include "../include/KaminoGPU.cuh"
# include <gtest/gtest.h>

size_t nTheta = 8;
float eps = 1e-6;

TEST(SampleTest, SampleVPhiTest) {
    float* buffer;
    KaminoTest KaminoInstance(nTheta);

    size_t numRow = KaminoInstance.nTheta;
    size_t numColumn = KaminoInstance.nPhi;
    buffer = (float *) malloc(sizeof(float) * numRow * numColumn);
    for (size_t row = 0; row < numRow; row++) {
    	for (size_t column = 0; column < numColumn; column++) {
    	    buffer[column + row * numColumn] = sinf(column*M_2PI/numColumn+row*M_hPI/numRow);
	}
    }
    EXPECT_NEAR(0.41914452f, KaminoInstance.sampleVPhi(buffer, 6.25, 0.75, numColumn), eps);
    EXPECT_NEAR(-0.46378926f, KaminoInstance.sampleVPhi(buffer, 8.75, 0.25, numColumn), eps);
    EXPECT_NEAR(0.62454507f, KaminoInstance.sampleVPhi(buffer, 2.25, 7.75, numColumn), eps);
    EXPECT_NEAR(0.81549315f, KaminoInstance.sampleVPhi(buffer, 5.f, -0.25, numColumn), eps);
    EXPECT_NEAR(0.92387953f, KaminoInstance.sampleVPhi(buffer, 4.5, -0.5, numColumn), eps);
    EXPECT_NEAR(0.87767457f, KaminoInstance.sampleVPhi(buffer, 4.5, -1.f, numColumn), eps);
    EXPECT_NEAR(0.83146961f, KaminoInstance.sampleVPhi(buffer, 14.5, 8.f, numColumn), eps);
    EXPECT_NEAR(0.83146961f, KaminoInstance.sampleVPhi(buffer, 14.5, 8.5, numColumn), eps);
    EXPECT_NEAR(0.76928820f, KaminoInstance.sampleVPhi(buffer, 14.5, 9.f, numColumn), eps);
}

TEST(SampleTest, SampleVThetaTest) {
    float* buffer;
    KaminoTest KaminoInstance(nTheta);

    size_t numRow = KaminoInstance.nTheta - 1;
    size_t numColumn = KaminoInstance.nPhi;
    buffer = (float *) malloc(sizeof(float) * numRow * numColumn);
    for (size_t row = 0; row < numRow; row++) {
    	for (size_t column = 0; column < numColumn; column++) {
    	    buffer[column + row * numColumn] = sinf(M_PI*(row+1)/KaminoInstance.nTheta)*cosf(column*M_PI*3/numColumn);
	}
    }
    EXPECT_NEAR(-0.26257811f, KaminoInstance.sampleVTheta(buffer, 5.f, 0.25, numColumn), eps);
    EXPECT_NEAR(0.14350628f, KaminoInstance.sampleVTheta(buffer, 0.f, -0.25, numColumn), eps);
    EXPECT_NEAR(-0.21260752f, KaminoInstance.sampleVTheta(buffer, 1.f, -1.f, numColumn), eps);
    EXPECT_NEAR(-0.30272750f, KaminoInstance.sampleVTheta(buffer, 1.f, -1.5, numColumn), eps);
    EXPECT_NEAR(0.37639811f, KaminoInstance.sampleVTheta(buffer, 1.4, 1.6, numColumn), eps);
    EXPECT_NEAR(-0.45306372f, KaminoInstance.sampleVTheta(buffer, 15.f, 6.5, numColumn), eps);
    EXPECT_NEAR(-0.07232269f, KaminoInstance.sampleVTheta(buffer, 15.1, 7.8, numColumn), eps);
    EXPECT_NEAR(0.05951203f, KaminoInstance.sampleVTheta(buffer, 15.1, 8.4, numColumn), eps);
    EXPECT_NEAR(0.21260752f, KaminoInstance.sampleVTheta(buffer, 15.f, 9.f, numColumn), eps);
    EXPECT_NEAR(0.30272750f, KaminoInstance.sampleVTheta(buffer, 15.f, 9.5, numColumn), eps);
}

TEST(SampleTest, SampleCenteredTest) {
    float* buffer;
    KaminoTest KaminoInstance(nTheta);

    size_t numRow = KaminoInstance.nTheta;
    size_t numColumn = KaminoInstance.nPhi;
    buffer = (float *) malloc(sizeof(float) * numRow * numColumn);
    for (size_t row = 0; row < numRow; row++) {
    	for (size_t column = 0; column < numColumn; column++) {
    	    buffer[column + row * numColumn] = sinf(M_PI*(row+0.5)/numRow)*cosf(column*M_PI/numColumn);
	}
    }
    EXPECT_NEAR(0.88800058f, KaminoInstance.sampleCentered(buffer, 2.2, 3.7, numColumn), eps);
    EXPECT_NEAR(0.07327654f, KaminoInstance.sampleCentered(buffer, 3.3, 0.2, numColumn), eps);
    EXPECT_NEAR(-0.04678800f, KaminoInstance.sampleCentered(buffer, 3.7, -0.2, numColumn), eps);
    EXPECT_NEAR(-0.16221167f, KaminoInstance.sampleCentered(buffer, 5.f, -0.5, numColumn), eps);
    EXPECT_NEAR(-0.31207572f, KaminoInstance.sampleCentered(buffer, 5.f, -1.f, numColumn), eps);
    EXPECT_NEAR(-0.08515461f, KaminoInstance.sampleCentered(buffer, 15.2, 7.7, numColumn), eps);
    EXPECT_NEAR(-0.07664074f, KaminoInstance.sampleCentered(buffer, 15.f, 8.f, numColumn), eps);
    EXPECT_NEAR(0.03806023f, KaminoInstance.sampleCentered(buffer, 15.f, 8.5, numColumn), eps);
    EXPECT_NEAR(-0.03076035f, KaminoInstance.sampleCentered(buffer, 15.f, 8.2, numColumn), eps);
    EXPECT_NEAR(0.07322330f, KaminoInstance.sampleCentered(buffer, 15.f, 9.f, numColumn), eps);
}

