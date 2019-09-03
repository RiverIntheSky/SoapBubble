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
	    fReal Gamma_m; fReal T; fReal Ds;
	    fReal rm; size_t nTheta;
	    float dt; float DT; int frames;
	    float A; int B; int C; int D; int E;
	    std::string thicknessPath; std::string velocityPath;
	    std::string thicknessImage; size_t particleDensity;
	    std::string attrib; int device;

	    fin >> attrib;  fin >> r;
	    fin >> attrib;  fin >> H;
	    fin >> attrib;  fin >> U;
	    fin >> attrib;  fin >> c_m;
	    fin >> attrib;  fin >> Gamma_m;
	    fin >> attrib;  fin >> T;
	    fin >> attrib;  fin >> Ds;
	    fin >> attrib;  fin >> rm;
	    fin >> attrib;  fin >> nTheta;
	    fin >> attrib;  fin >> dt;
	    fin >> attrib;  fin >> DT;
	    fin >> attrib;  fin >> frames;
	    fin >> attrib;  fin >> A;
	    fin >> attrib;  fin >> B;
	    fin >> attrib;  fin >> C;
	    fin >> attrib;  fin >> D;
	    fin >> attrib;  fin >> E;
	    fin >> attrib;  fin >> thicknessPath;
	    fin >> attrib;  fin >> velocityPath;
	    fin >> attrib;  fin >> thicknessImage;
	    fin >> attrib;  fin >> particleDensity;
	    fin >> attrib;  fin >> device;
	
	    if (thicknessImage == "null")
		{
		    thicknessImage = "";
		}

	    Kamino KaminoInstance(r, H, U, c_m, Gamma_m, T, Ds, rm,
				  nTheta, dt, DT, frames, A, B, C, D, E,
				  thicknessPath, velocityPath, thicknessImage, particleDensity, device);
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