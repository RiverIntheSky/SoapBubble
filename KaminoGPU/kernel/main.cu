# include "../include/KaminoGPU.cuh"
# include <fstream>

int main(int argc, char** argv)
{
    if (argc == 2)
	{
	    std::string configFile = argv[1];
	    std::fstream fin;
	    fin.open(configFile, std::ios::in);
	    float r; float H; float U; float c_m;
	    float Gamma_m; float T; float Ds;
	    float rm; size_t nTheta;
	    float dt; float DT; int frames;
	    std::string outputDir;
	    std::string thicknessImage; size_t particleDensity;
	    std::string attrib; int device;
	    std::string AMGconfig;

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
	    fin >> attrib;  fin >> outputDir;
	    fin >> attrib;  fin >> thicknessImage;
	    fin >> attrib;  fin >> AMGconfig;
	    fin >> attrib;  fin >> particleDensity;
	    fin >> attrib;  fin >> device;


	    if (thicknessImage == "null")
		{
		    thicknessImage = "";
		}

	    Kamino KaminoInstance(r, H, U, c_m, Gamma_m, T, Ds, rm, nTheta,
				  dt, DT, frames, outputDir, thicknessImage,
				  particleDensity, device, AMGconfig);
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