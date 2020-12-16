# include "../include/bubble.cuh"
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
	    fReal dt; fReal DT; int frames;
	    std::string outputDir;
	    std::string thicknessImage;
	    std::string attrib; int device; fReal blendCoeff;
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
	    fin >> attrib;  fin >> blendCoeff;
	    fin >> attrib;  fin >> device;

	    if (thicknessImage == "null")
		{
		    thicknessImage = "";
		}

	    Bubble bubble(r, H, U, c_m, Gamma_m, T, Ds, rm, nTheta,
			  dt, DT, frames, outputDir, thicknessImage,
			  device, AMGconfig, blendCoeff);
	    bubble.run();
	    return 0;
	}
    else
	{
	    std::cout << "Please provide the path to config.txt as an argument." << std::endl;
	    std::cout << "Usage example: ./soapBubble ../config.txt" << std::endl;
	    std::cout << "Configuration file was missing, exiting." << std::endl;
	    return -1;
	}
}
