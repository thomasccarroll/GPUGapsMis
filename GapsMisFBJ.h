//GPUGapsMis
//Thomas Carroll - 2017

#ifndef GAPSMISFB_J_H_
#define GAPSMISFB_J_H_


#include <omp.h>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <list>
#include <limits>
#include <iterator>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <errno.h>
#include <chrono>
#include <stdio.h>
#include "Sequence.h"



using namespace std;
using std::string;
using std::vector;

#define DEBUG 0
#define DEBUG_GPU 0

using std::cout;
using std::endl;

class GapsMisFBJ{
public:

int* _score;
vector<Sequence*> _texts;
vector<Sequence*> _patterns;
string _alpha;

string _textsFile;
string _patternsFile;
string _scoreFile;
string _outFile;
string _headers[3] = {
						"Num Threads\t Total Latency", //CPU
						"Patt Length\tTotal Time\t Num Kernel Launches\t Prep Time\t Copy Time\t Kernel Time\t Average Kernel Time\t Host Backtrack Time\t Block Size", //Hybrid
						"Total Time\t Num Kernel Launches\t Prep Time\t Copy Time\t Kernel Time\t Average Kernel Time\t Block Size" //GPU
						};
int _maxTLength;
int _minTLength;
int _avgTLength;
int _maxPLength;
int _minPLength;
int _avgPLength;
int _aLength;
int _tAlign;
int _pAlign;
int _width;
int _widthAlign;
int _gaps;
int _approach;
float _open;
float _ext;
int _numTexts;
int _numPatterns;
int _numSeqPairs;
int _seqPairID;
int _textID;
int _pattID;
int _gpuBatchSize;
int _numBatches;
int _cpuBatchSize;
int _matrixSize;
int _threadBlockSize;
int _deviceMatrixSize;
size_t d_globalMemSize;
size_t  d_constMemSize;
size_t d_sharedMemSize;
size_t d_maxBlocks;
size_t d_alignVal;
float *d_A;
float *d_B;
int *d_H;
int *d_Bt;
int *h_pin_out_data;
int *d_pin_out_ptr;
int *h_H;
int *d_pin_H_ptr;
int *d_gaps;
int *d_texts;
int *h_texts;
int *d_patterns;
int  *h_patterns;
int *d_t_lengths;
vector<int> *h_t_lengths;
int *d_p_lengths;
vector<int>  *h_p_lengths;
float total_time;
float total_gpu_time;
float total_host_time;
float total_backtrack_time;
typedef std::chrono::high_resolution_clock Clock;
Clock theClock;
Clock::time_point total_start;
Clock::time_point total_stop;
Clock::time_point host_start;
Clock::time_point host_stop;
Clock::time_point gpu_start;
Clock::time_point gpu_stop;
Clock::time_point backtrack_start;
Clock::time_point backtrack_stop;
float milliseconds;

GapsMisFBJ();
GapsMisFBJ(string textsFile, string patternsFile, string scoreFile, int numGaps, float openPenalty, float extensionPenalty, int theSetting);
~GapsMisFBJ();
int processScoreMatrix(string fileName);
int processSequenceFile(string fileName, vector<Sequence*>& out);
string getHeaders();
string getOutputData();
int printScoreMatrix();
int printSequenceSet(vector<Sequence*>& theSet);
void map(string &seq, int *out);
int run();
int batchUp();
int batchUpConstant();
int batchUpConstantHShared();
int batchUpConstantHRow();
int batchUpConstantHRowPFVector();
int batchSingleText_dev();
int batchSingleText_host();
int batchMultiText_dev();
int batchMultiText_host();
int runCPU();
int runGPU_Simple_Align_Discard();
int runGPU_Multi_Align_Discard();
int runGPU_Simple_Align_Return();
int runGPU_Simple_Align_Keep();
int runGPU_Single_Host();
int runGPU_Single_Backtrack();
int runGPU_Multi_Backtrack();
int runGPU_Single_Sendback();
int runGPU_Multi_Sendback();
int runGPU_Multi_Host();
int runGPU_Multi_Device();
int batchUpPadded();
int bufferPaddedSequences(vector<Sequence> theSeqs, int* srcBuffer, int* lenBuffer);
int runBatch();
int runBatchConstant();
int runBatchConstantHShared();
int runBatchConstantHRow();
int runSingleTextBatch(int theTextNum, int theStartPattern, int theEndPattern, int paddingValue, bool devBackTrack);
int runMultiTextBatch(int batchSize, int t_offset, int p_offset, bool devBackTrack, int endJob);
int getIndex2d(int i, int j, int C);
int getHIndex(int i, int j, int C, int size, int job);
int getIndex3d(int i, int j, int k, int C, int R);
int getIndexOffset(int i, int j, int k, int C, int R, int job);
int getIndex(int i, int j, int C, int offset, int job);
int idx3d(int i, int j, int k, int C, int R);
int getScore(int i, int j);
void computeZero(int *mapT, int *mapX);
int alignment_max(vector<int> mapT, vector<int> mapX);
int cpu_alignment(int *text, int *pattern, int tLength, int pLength);
int batchUpCPU();
void print(int k);
void printAlphabet();
void experiment(string textFile, string patternFile, string scoreFile, string outputFile, int setting);
void loopExperiment(int setting, int attempts);
int backtrack(int start);
int backtrackFromDevice(int startJob, int endJob, int matrixSize, int paddingValue);
int writeOutput(string outFile);
int getLengthStats();
int getTimeStats();
string getOutput();
int getKernelLaunches();
double getTotalTime();
double getHostTime();
double getGPUTime();
double getBacktrackTime();
int getNumCells();
};

#endif
