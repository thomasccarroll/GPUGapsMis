//GPUGapsMis
//Thomas Carroll - 2017

#ifndef GAPSMISFBUSER_H_
#define GAPSMISFBUSER_H_
#include <string>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "GapsMisFBJ.h"

using namespace std;
using std::string;
using std::vector;

class GapsMisFBJUser{
public:

	string experimentFileName;
	vector<string> experimentNames;
	vector<string> textFileNames;
	vector<string> patternFileNames;
	vector<string> scoreFileNames;
	vector<string> outFileNames;
	vector<int> gapsVector;
	vector<float> openVector;
	vector<float> extVector;
	vector<int> approachVector;

	GapsMisFBJUser();
	GapsMisFBJUser(string expFileName);
	~GapsMisFBJUser();

	int parseExperimentFile();
	int runExperiments();
	int writeOutputToFile(string outputFile, string theOutput);
	double getAverage(vector<double> *theVector);
	double getStDev(vector<double> *theVector);
	string getOutput(int approach, vector<double> *total_time, vector<double> *host_time, vector<double> *gpu_time, vector<double> *backtrack_time, int numCells);

};

#endif
