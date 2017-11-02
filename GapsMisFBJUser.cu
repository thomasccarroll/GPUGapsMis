//GPUGapsMis
//Thomas Carroll - 2017

#include <iostream>
#include "GapsMisFBJUser.h"

GapsMisFBJUser::GapsMisFBJUser()
{
	printf("Please use constructor with Experiment List file as parameter\n");
	exit(-1);
}

GapsMisFBJUser::GapsMisFBJUser(string expFileName)
{
	printf("**********\nGapsMisFBJUser class (from home).\n**********\n\n\n");
	this -> experimentFileName = expFileName;
}

GapsMisFBJUser::~GapsMisFBJUser()
{
	printf("\nBye!\n");
}

int GapsMisFBJUser::parseExperimentFile()
{
	printf("GapsMisFBJUser::parseExperimentFile()\n");

	printf("\tParsing the file named: %s\n", this -> experimentFileName.c_str());
	using std::ifstream;
	ifstream in;
	in.open(this -> experimentFileName.c_str());

	if(!in.is_open())
	{
		std::cout << "Could not open the Experiment List File.\nExiting Program." << std::endl;
		exit(1);
	} else{
		cout << "Opened file." << endl;
	}

	string line;
	while(getline(in, line))
	{
		if(line[0] != '#')
		{
			int i = 0;
			int j = line.find(',');
			vector<string> tokens;
			while (j >= 0)
			{
				tokens.push_back(line.substr(i, j-i));
				i = ++j;
				j = line.find(',', j);
			}
			if(j < 0)
			{
				tokens.push_back(line.substr(i,line.length()));
			}

			if (tokens.size() != 9)
			{
				printf("There is a problem, the file is not in the correct format (not enough experiment arguments)\n");
				exit(-1);
			}
			try
			{
				experimentNames.push_back(tokens.at(0));
				textFileNames.push_back(tokens.at(1));
				patternFileNames.push_back(tokens.at(2));
				scoreFileNames.push_back(tokens.at(3));
				outFileNames.push_back(tokens.at(4));
				gapsVector.push_back(stoi(tokens.at(5).c_str()));
				openVector.push_back(stof(tokens.at(6).c_str()));
				extVector.push_back(stof(tokens.at(7).c_str()));
				approachVector.push_back(stoi(tokens.at(8).c_str()));
			}
			catch(...)
			{
				printf("There is a problem with the format\n");
				exit(-1);
			}
		}
	}
	printf("Parsed the file. There are %d experiments:\n\n", (int)experimentNames.size());
	in.close();
	return 0;
}

int GapsMisFBJUser::runExperiments()
{

	for(int e = 0; e < (int)experimentNames.size(); e++)
	{

		string expName = experimentNames.at(e);
		string textFileName = textFileNames.at(e);
		string patternFileName = patternFileNames.at(e);
		string scoreFileName = scoreFileNames.at(e);
		string outFileName = outFileNames.at(e);
		int gaps = gapsVector.at(e);
		float open = openVector.at(e);
		float ext = extVector.at(e);
		int approach = approachVector.at(e);
		string expheader;
		expheader += "Experiment" +
				to_string(e) + "\n\tName: " + expName.c_str() +
				"\n\tTexts: " + textFileName.c_str() +
				"\n\tPatterns: " + patternFileName.c_str() +
				"\n\tScore: " + scoreFileName.c_str() +
				"\n\tOutput: " + outFileName.c_str() +
				"\n\tGaps: " + to_string(gaps) +
				"\n\tOpen: " + to_string(open) +
				" Ext: " + to_string(ext) +
				" \n\tApproach: " + to_string(approach) + "\n";
		cout << expheader << endl;
		string outputString = expheader;
		int numRuns = 5;
		vector<double> total_time_vector;
		vector<double> host_time_vector;
		vector<double> gpu_time_vector;
		vector<double> backtrack_time_vector;
		unsigned long long int numCells = 0;
		for(int runNum = 0; runNum < numRuns; runNum++)
		{
			printf("Experiment %d Run number %d:\n", e, runNum);
			GapsMisFBJ* gfb = new GapsMisFBJ(textFileName, patternFileName, scoreFileName, gaps, open, ext, approach);
			gfb -> run();
			printf("Total time is %f\n", gfb->getTotalTime());
			total_time_vector.push_back(gfb->getTotalTime());
			backtrack_time_vector.push_back(gfb->getBacktrackTime());
			if(approach >= 3)
			{
				host_time_vector.push_back(gfb->getHostTime());
				gpu_time_vector.push_back(gfb->getGPUTime());
			}
			switch(gfb->_approach)
			{
			case 5:
			case 6:
			case 8:
				backtrack_time_vector.push_back(gfb->getBacktrackTime());
				break;
			default:
				break;

			};
			numCells = gfb -> getNumCells();
			delete gfb;
		}
		writeOutputToFile(outFileName,expheader+getOutput(approach, &total_time_vector, &host_time_vector, &gpu_time_vector, &backtrack_time_vector, numCells));

	}

	printf("\nEnd of all experiments.\n");
	return 0;

}

int GapsMisFBJUser::writeOutputToFile(string outputFile, string theOutput)
{
	ofstream theOutFile;
	theOutFile.open(outputFile, ios::out | ios::app );
	if(!theOutFile.is_open())
	{
		printf("Could not open the Output File of the experiment\nExiting.");
		exit (-1);
	}else
	{
		theOutFile << theOutput.c_str();
		theOutFile.close();
	}
	return 0;
}

int main(int argc, char** argv)
{
	printf("*GPUGapsMis*\n");
	GapsMisFBJUser* me = new GapsMisFBJUser("../experimentFiles/multiText-t.exp");
		cudaDeviceProp deviceProp;
		cudaSetDevice(1);
		cudaGetDeviceProperties(&deviceProp, 1);
		if (deviceProp.canMapHostMemory)
		{
			cudaSetDeviceFlags(cudaDeviceMapHost);
		}else{
			cout << "Can not map host memory" << endl;
			exit(-1);
		}
	me -> parseExperimentFile();
	me -> runExperiments();
	return 0;
}

double GapsMisFBJUser::getAverage(vector<double> *theVector)
{
	double ans = 0.0;
	for(int i = 0; i < theVector->size(); i ++)
	{
		ans += theVector->at(i);
	}

	ans = ans/theVector->size();
	return ans;
}

double GapsMisFBJUser::getStDev(vector<double> *theVector)
{
	double ans = 0.0;
	double mean = this->getAverage(theVector);
	for(int i = 0; i < theVector->size(); i ++)
	{
		double variance = theVector->at(i) - mean;
		ans += (variance * variance);
	}
	ans = ans/theVector->size();
	return ans;
}


string GapsMisFBJUser::getOutput(int approach, vector<double> *total_time, vector<double> *host_time, vector<double> *gpu_time, vector<double> *backtrack_time, int numCells)
{
	string output = "";
	string _headers[3] = {	"Total Time\tSD(total)\tGCUPS\tBacktrack",
							"Total Time\tSD(total)\tHost\tSD(host)\tGPU\tSD(GPU)\tBacktrack\tSD(Backtrack)\tGCUPS",
							"Total Time\tSD(total)\tHost\tSD(host)\tGPU\tSD(GPU)\tGCUPS"
							};

	switch(approach)
	{
	case 1:
	case 2:
		output += _headers[0] +"\n";
		break;
	case 3:
	case 4:
	case 5:
	case 6:
	case 8:
		output += _headers[2] + "\n";
		break;
	case 7:
	case 9:
		output += _headers[1] + "\n";
		break;
	default:
		break;
	};
	output += std::to_string(this->getAverage(total_time));
	output += "\t";
	output += std::to_string(this->getStDev(total_time));
	output += "\t";
	if(approach >= 3)
	{
		output += std::to_string(this->getAverage(host_time));
		output += "\t";
		output += std::to_string(this->getStDev(host_time));
		output += "\t";

		output += std::to_string(this->getAverage(gpu_time));
		output += "\t";
		output += std::to_string(this->getStDev(gpu_time));
		output += "\t";
	}

	printf("GCUPS: %f / %d = %f\n",(this->getAverage(total_time)/1000),numCells,
				((double)numCells/(this->getAverage(total_time)/1000))/1000000000
				);
	output += std::to_string(((double)numCells/(this->getAverage(total_time)/1000))/1000000000) + "\t";

	if(approach==2)
	{
		output += std::to_string(this->getAverage(backtrack_time));
		}

	return output;

}
