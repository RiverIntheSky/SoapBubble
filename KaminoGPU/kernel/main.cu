# include "../include/KaminoGPU.cuh"
# include <fstream>

int main(int argc, char** argv)
{
    if (argc == 2)
	{
	    std::string configFile = argv[1];
	    std::fstream fin;
	    fin.open(configFile, std::ios::in);
	    fReal r; fReal H; fReal U; fReal c_m;
	    fReal Gamma_m; fReal sigma_a; fReal R; fReal T;
	    fReal rho; fReal mu; fReal Ds; fReal g;
	    fReal rm; size_t nTheta; fReal particleDensity;
	    float dt; float DT; int frames;
	    float A; int B; int C; int D; int E;
	    std::string thicknessPath; std::string particlePath;
	    std::string thicknessImage; std::string solidImage; std::string colorImage;
	    std::string attrib;

	    fin >> attrib;  fin >> r;
	fin >> attrib;  fin >> H;
	fin >> attrib;  fin >> U;
	fin >> attrib;  fin >> c_m;
	fin >> attrib;  fin >> Gamma_m;
	fin >> attrib;  fin >> sigma_a;
	fin >> attrib;  fin >> R;
	fin >> attrib;  fin >> T;
	fin >> attrib;  fin >> rho;
	fin >> attrib;  fin >> mu;
	fin >> attrib;  fin >> Ds;
	fin >> attrib;  fin >> g;
	fin >> attrib;  fin >> rm;
	fin >> attrib;  fin >> nTheta;
	fin >> attrib;  fin >> particleDensity;
	fin >> attrib;  fin >> dt;
	fin >> attrib;  fin >> DT;
	fin >> attrib;  fin >> frames;
	fin >> attrib;  fin >> A;
	fin >> attrib;  fin >> B;
	fin >> attrib;  fin >> C;
	fin >> attrib;  fin >> D;
	fin >> attrib;  fin >> E;
	fin >> attrib; 	fin >> thicknessPath;
	fin >> attrib; 	fin >> particlePath;

	std::cout << nTheta << std::endl;

	fin >> attrib; 	fin >> thicknessImage;
	if (thicknessImage == "null")
	    {
		thicknessImage = "";
	    }
	fin >> solidImage;
	if (solidImage == "null")
	    {
		solidImage == "";
	    }
	fin >> colorImage;
	if (colorImage == "null")
	    {
		colorImage = "";
	    }

	Kamino KaminoInstance(r, H, U, c_m, Gamma_m, sigma_a, R, T, rho, mu, Ds, g, rm,
			      nTheta, particleDensity, dt, DT, frames, A, B, C, D, E,
			      thicknessPath, particlePath, thicknessImage, solidImage, colorImage);
	KaminoInstance.run();
	return 0;
	}
 else
     {
	 std::cout << "Please provide the path to configKamino.txt as an argument." << std::endl;
	 std::cout << "Usage example: ./kamino ./configKamino.txt" << std::endl;
	 std::cout << "Configuration file was missing, exiting." << std::endl;
	 return -1;
     }
}