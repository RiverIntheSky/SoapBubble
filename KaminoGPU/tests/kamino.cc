# include "kamino.hh"

float KaminoTest::sampleVPhi(fReal* input, fReal phiRawId, fReal thetaRawId, size_t pitch)
{
    fReal phi = phiRawId - vPhiPhiOffset;
    fReal theta = thetaRawId - vPhiThetaOffset;
    // Phi and Theta are now shifted back to origin

    bool isFlippedPole = validateCoord(phi, theta);

    int phiIndex = static_cast<int>(floorf(phi));
    int thetaIndex = static_cast<int>(floorf(theta));
    fReal alphaPhi = phi - static_cast<fReal>(phiIndex);
    fReal alphaTheta = theta - static_cast<fReal>(thetaIndex);

    if (thetaIndex == 0 && isFlippedPole) {
	size_t phiLower = (phiIndex) % nPhi;
	size_t phiHigher = (phiLower + 1) % nPhi;
	fReal higherBelt = -kaminoLerp(input[phiLower + pitch * thetaIndex],
				       input[phiHigher + pitch * thetaIndex], alphaPhi);

	

	phiLower = (phiIndex + nPhi / 2) % nPhi;
	phiHigher = (phiLower + 1) % nPhi;

	fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);
	


	fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
	return lerped;
    }
    if (thetaIndex == nTheta - 1)	{
	size_t phiLower = (phiIndex) % nPhi;
	size_t phiHigher = (phiLower + 1) % nPhi;
	fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);

	phiLower = (phiIndex + nPhi / 2) % nPhi;
	phiHigher = (phiLower + 1) % nPhi;

	fReal higherBelt = -kaminoLerp(input[phiLower + pitch * thetaIndex],
				       input[phiHigher + pitch * thetaIndex], alphaPhi);

	fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
	return lerped;
    }
  
    size_t phiLower = phiIndex % nPhi;
    size_t phiHigher = (phiLower + 1) % nPhi;
    size_t thetaLower = thetaIndex;
    size_t thetaHigher = thetaIndex + 1;

    fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaLower],
				 input[phiHigher + pitch * thetaLower], alphaPhi);
    fReal higherBelt = kaminoLerp(input[phiLower + pitch * thetaHigher],
				  input[phiHigher + pitch * thetaHigher], alphaPhi);

    fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
    return lerped;
}

float KaminoTest::sampleVTheta(fReal* input, fReal phiRawId, fReal thetaRawId, size_t pitch)
{
    fReal phi = phiRawId - vThetaPhiOffset;
    fReal theta = thetaRawId - vThetaThetaOffset;
    // Phi and Theta are now shifted back to origin

    bool isFlippedPole = validateCoord(phi, theta);

    int phiIndex = static_cast<int>(floorf(phi));
    int thetaIndex = static_cast<int>(floorf(theta));
    fReal alphaPhi = phi - static_cast<fReal>(phiIndex);
    fReal alphaTheta = theta - static_cast<fReal>(thetaIndex);

    if (thetaRawId < 0 || thetaIndex == nTheta - 1) {
	thetaIndex -= 1;
	alphaTheta += 1;
    }
    if (thetaIndex == nTheta) {
	thetaIndex -= 2;
	phiIndex = (phiIndex + nPhi / 2) % nPhi;
	size_t phiLower = phiIndex % nPhi;
	size_t phiHigher = (phiLower + 1) % nPhi;
	return -kaminoLerp(input[phiLower + pitch * thetaIndex],
				 input[phiHigher + pitch * thetaIndex], alphaPhi);
    }
    
    if (thetaIndex == 0 && isFlippedPole) {
	size_t phiLower = phiIndex % nPhi;
	size_t phiHigher = (phiLower + 1) % nPhi;
	fReal higherBelt = -kaminoLerp(input[phiLower + pitch * thetaIndex],
				       input[phiHigher + pitch * thetaIndex], alphaPhi);
	
	phiLower = (phiLower + nPhi / 2) % nPhi;
	phiHigher = (phiHigher + nPhi / 2) % nPhi;
	fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);
	

	alphaTheta = 0.5 * alphaTheta;
	fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
	return lerped;
	
    }
    if (thetaIndex == nTheta - 2) {
	size_t phiLower = phiIndex % nPhi;
	size_t phiHigher = (phiLower + 1) % nPhi;
	fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);
	
	phiLower = (phiLower + nPhi / 2) % nPhi;
	phiHigher = (phiHigher + nPhi / 2) % nPhi;
	fReal higherBelt = -kaminoLerp(input[phiLower + pitch * thetaIndex],
				       input[phiHigher + pitch * thetaIndex], alphaPhi);
	
	alphaTheta = 0.5 * alphaTheta;
	fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
	return lerped;
    }
    
    size_t phiLower = phiIndex % nPhi;
    size_t phiHigher = (phiLower + 1) % nPhi;
    size_t thetaLower = thetaIndex;
    size_t thetaHigher = thetaIndex + 1;

    fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaLower],
				 input[phiHigher + pitch * thetaLower], alphaPhi);
    fReal higherBelt = kaminoLerp(input[phiLower + pitch * thetaHigher],
				  input[phiHigher + pitch * thetaHigher], alphaPhi);

    fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
    return lerped;
}

float KaminoTest::sampleCentered(fReal* input, fReal phiRawId, fReal thetaRawId, size_t pitch)
{
    fReal phi = phiRawId - centeredPhiOffset;
    fReal theta = thetaRawId - centeredThetaOffset;
    // Phi and Theta are now shifted back to origin

    bool isFlippedPole = validateCoord(phi, theta);

    int phiIndex = static_cast<int>(floorf(phi));
    int thetaIndex = static_cast<int>(floorf(theta));
    fReal alphaPhi = phi - static_cast<fReal>(phiIndex);
    fReal alphaTheta = theta - static_cast<fReal>(thetaIndex);
    
    if (thetaIndex == 0 && isFlippedPole) {
	size_t phiLower = phiIndex % nPhi;
	size_t phiHigher = (phiLower + 1) % nPhi;
	fReal higherBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				      input[phiHigher + pitch * thetaIndex], alphaPhi);

	phiLower = (phiLower + nPhi / 2) % nPhi;
	phiHigher = (phiHigher + nPhi / 2) % nPhi;
	fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);

	fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
	return lerped;
    }
    if (thetaIndex == nTheta - 1) {
	size_t phiLower = phiIndex % nPhi;
	size_t phiHigher = (phiLower + 1) % nPhi;
	fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				     input[phiHigher + pitch * thetaIndex], alphaPhi);

	phiLower = (phiLower + nPhi / 2) % nPhi;
	phiHigher = (phiHigher + nPhi / 2) % nPhi;
	fReal higherBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
				      input[phiHigher + pitch * thetaIndex], alphaPhi);

	fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
	return lerped;
    }

    size_t phiLower = phiIndex % nPhi;
    size_t phiHigher = (phiLower + 1) % nPhi;
    size_t thetaLower = thetaIndex;
    size_t thetaHigher = thetaIndex + 1;

    fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaLower],
				 input[phiHigher + pitch * thetaLower], alphaPhi);
    fReal higherBelt = kaminoLerp(input[phiLower + pitch * thetaHigher],
				  input[phiHigher + pitch * thetaHigher], alphaPhi);

    fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
    return lerped;
}


float KaminoTest::kaminoLerp(float from, float to, float alpha)
{
    return (1.0 - alpha) * from + alpha * to;
}


bool KaminoTest::validateCoord(fReal& phi, fReal& theta) {
    bool ret = false;
    if (theta > nTheta) {
	theta = nPhi - theta;
	phi += nTheta;
    	ret = !ret;
    }
    if (theta < 0) {
    	theta = -theta;
    	phi += nTheta;
    	ret = !ret;
    }
    phi = fmod(phi + nPhi, (fReal)nPhi);
    return ret;
}

KaminoTest::KaminoTest(size_t nTheta): nTheta(nTheta), nPhi(nTheta * 2), gridLen(M_PI/nTheta), invGridLen(nTheta / M_PI) {}
KaminoTest::~KaminoTest()
{}

