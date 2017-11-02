//GPUGapsMis
//Thomas Carroll - 2017

#include "GapsMisFBJ.h"
#include "Sequence.h"
#include "kernels.h"

GapsMisFBJ::GapsMisFBJ()
{
	printf("New GapsMisFBJ has been created.\n");
}

GapsMisFBJ::GapsMisFBJ(	string textsFile,
						string patternsFile,
						string scoreFile,
						int numGaps,
						float openPenalty,
						float extensionPenalty,
						int theApproach)
{
	this->_textsFile = textsFile;
	this->_patternsFile = patternsFile;
	this->_scoreFile = scoreFile;
	this->_gaps = numGaps;
	this->_open = openPenalty;
	this ->_ext = extensionPenalty;
	this->_approach = theApproach;
	this->total_time= this->total_backtrack_time = this-> total_host_time = this->total_gpu_time = 0.0;
}

GapsMisFBJ::~GapsMisFBJ()
{
	printf("\n~~~~~~~~~~~~~~~~~~~~\nGapsMisFB Desturctor...\n");

	for(vector<Sequence*>::iterator it = this -> _texts.begin(); it != this -> _texts.end(); it++)
	{
		delete (*it);
	}

	for(vector<Sequence*>::iterator it = this -> _patterns.begin(); it != this -> _patterns.end(); it++)
	{
		delete (*it);
	}

	cudaFreeHost(this->h_texts);
	cudaFreeHost(this->h_patterns);
	cudaDeviceReset();
	printf("Clean up Complete.");
	std::cout << endl << "~~~~~~~~~~~~~~~~~~~~" << endl << "Bye from GapsMisFBJ!" << std::endl;

}

int GapsMisFBJ::processSequenceFile(string fileName, vector<Sequence*>& out)
{
	printf("Processing the sequence file: %s\n", fileName.c_str());
	using std::ifstream;
	ifstream in;

	in.open(fileName.c_str());

	if(!in.is_open())
	{
			printf("Could not open the file: %s, Exiting.\n", fileName.c_str());
			exit(-1);
	}

	string line;
	int numLines = 0;
	while(getline(in,line))
	{numLines ++;}
	try{
		out.reserve(numLines);
		printf("Reserved space for %d texts\n", numLines);
	}catch(...)
	{
		printf("Could Not reserve space for Sequence Vector.\n");
		exit(-1);
	}
	in.clear();

	in.seekg (0);
	int pos = in.tellg();
	int lineNum = 0;
	while(getline(in,line))
	{
		if(DEBUG)printf("Text number %d:\n", lineNum );
		lineNum ++;
		Sequence* seq = new Sequence();
		seq -> raw = line;
		seq -> length = ((seq -> raw.length())/sizeof(char));
		seq -> mapped = new int[seq -> length];
		if(seq -> mapped == NULL )
		{
				printf("Could not allocate mapped. Exiting.\n");
				exit(-1);
		}
		this -> map(seq -> raw, seq -> mapped);
		try{
				if(out.size() == 0)
				{
					out.insert(out.begin(),seq);
				}else
				{
					if(seq -> length >= (out.front()) ->length)
					{

						out.insert(out.begin(),seq);

					} else if(seq -> length <= (out.back()) ->length)
					{
						out.push_back(seq);
					}else
					{
						vector<Sequence*>::iterator it = out.begin();
						bool inserted = false;
						while( (it != out.end()) &&  !(inserted))
						{
							if(seq ->length >= (*it) ->length)
							{
								out.insert(it,seq);
								inserted = true;
							}
							it ++;
						}
					}
				}

		}catch(...)
		{
			printf("There has been a problem inserting!\n");
			exit(-1);
		}
	}
	return 0;
}

int GapsMisFBJ::processScoreMatrix(string fileName)
{
	printf("Processing Score Matrix and Alphabet from the file: %s ...\n", fileName.c_str());
	using std::ifstream;
	this->_aLength = 0;
	ifstream in;
	in.open(fileName.c_str());
	if(!in.is_open())
	{
		std::cout << "Could not open the Score Matrix File.\nExiting Program." << std::endl;
		exit(-1);
	}
	string line;
	int length;
	int lineNo= 0;
	bool alphabet = false;
	while(getline(in, line))
	{
		if(line[0] != '#' && !alphabet)
		{
			alphabet = true;
			line.erase(std::remove(line.begin(),line.end(),' '),line.end());
			length = line.length()/sizeof(char);
			this ->_alpha = line;
			this -> _aLength = length;
			this->_score = (int*)malloc(225*sizeof(int));
			lineNo = 0;
		}
		else if(lineNo < _aLength && alphabet)
		{
			line.erase(line.begin(),line.begin() + 1);
			std::stringstream ss;
			ss.str(line.c_str());
			int num;
			for (int i  = 0; i < _aLength; i ++)
			{
				ss >> num;
				int idx = getIndex2d(lineNo, i, length);
				_score[idx] = num;
			}
			lineNo ++;
		}
		else{
			lineNo ++;
		}
	}
	printf("Processing Finished.\n");
	for(int i = 0; i < this->_aLength; i ++)
	{
		for(int j = 0; j < this->_aLength; j ++)
		{
			printf("%d ",this->_score[i*_aLength + j]);
		}
		printf("\n");
	}
	return 0;
}
void GapsMisFBJ::map(string &seq, int *out)
{
	if(DEBUG)printf("Now Mapping...");
	int sLength = seq.length()/sizeof(char);
	int aLength = _alpha.length()/sizeof(char);
	if(out == NULL)
	{
			printf("Cannot map to a null pointer.");
			exit(1);
	}
	for (int i = 0; i < sLength; i ++)
	{
		bool found = false;
		for(int j = 0; j < aLength; j ++)
		{
			if (seq.at(i) == _alpha[j])
			{
				found = true; 
				out[i] = j;
			}
			if (found ) break;
		}
	}
	for(int i = 0; i < sLength; i++)
	{
		if(_alpha[out[i]] != seq.at(i))
		{
			printf("There is a problem with the mapping at position %d: \n", i);
			printf("Mapped\tAlphabet\tSequence\n");
			printf("%d\t%c\t%c\n", out[i], _alpha[out[i]], seq.at(i));
			exit(1);
		}
	}
	if(DEBUG)printf("Mapped OK\n");
}

int GapsMisFBJ::printSequenceSet(vector<Sequence*>& theSet)
{
	int count  = 0;
	for (vector<Sequence*>::iterator it = theSet.begin(); it != theSet.end(); it ++)
	{
		printf("Sequence number %d: \n", count);
		(*it) -> print();
		count ++;
	}
	return 0;
}

int GapsMisFBJ::getScore(int t, int x)
{
	int idx = getIndex2d(t,x,_aLength);
	return _score[idx];
}

int GapsMisFBJ::getIndex2d(int i, int j, int C)
{
	return (i * C) + j ;
}

int GapsMisFBJ::getIndex3d(int i, int j, int k, int C, int R)
{
	return (k * R * C) + i*C + j;
}

int GapsMisFBJ::getHIndex(int i, int j, int C, int size, int job)
{
	return (job * size) + (i * C) + j;
}

int GapsMisFBJ::cpu_alignment(int *text, int *pattern, int tLength, int pLength)
{

	int height = tLength + 1;
	int width = pLength + 1;
	int backTrackStart;
	float* G = (float*)malloc(this->_matrixSize * sizeof(float));
	int* H = (int*)malloc(this->_matrixSize  * sizeof(int));
	int* gap_pos, *gap_len, *where;
	if (this->_approach ==2)
	{
		gap_pos = (int*)malloc((this->_gaps +1) * sizeof(int));
		gap_len = (int*)malloc((this->_gaps+1) * sizeof(int));
		where = (int*)malloc((this->_gaps+1) * sizeof(int));
	}
	if(!G || !H || ((_approach == 2) && (!gap_pos || !gap_len || !where)))
	{
		printf("Fatal Error: Could not allocate Storage. Exiting.\n");
		exit(-2);
	}
	for (int i = 0; i < height; i ++)
	{
		for (int j = 0; j < width; j ++)
		{
			int idx = getIndex3d(i,j,0,width, height);
			if (i == 0 && j == 0)
			{
				G[idx]= 0.0;
			}
			else if (i != j)
			{
				G[idx] = 0.0- (float)(tLength + pLength) * this->_ext + this->_open;
			}else
			{
				int prevDiagIdx = getIndex3d(i-1, j-1, 0,width, height);
				G[idx] = G[prevDiagIdx] + (float)getScore(text[i-1],pattern[j-1]);
			}
			H[idx] = 0;
		}
	}
	try{
		for (int k = 1; k <= _gaps ; k ++)
		{
			vector<int> maxILoc(width, 0);
			int j = 0;
			for(int i = 0; i < height; i ++)
			{
				int idx = getIndex3d(i,j,k,width,height);
				G[idx] = -(float)i * this->_ext + this->_open;
				H[idx] =  i;
			}
			int i = 0;
			for (int j = 1; j < width; j ++)
			{
				int idx = getIndex3d(i,j,k,width,height);
				G[idx] = 0.0 -(float)j * this->_ext + this->_open;
				H[idx] = - j;
			}
			for(int i = 1; i < height; i ++)
			{
				int maxJLoc = 0;
				for(int j = 1; j < width; j ++)
				{
					int idx = getIndex3d(i, j, k, width, height);
					float u, v, w;
					int maxIdxI = getIndex3d(maxILoc[j],j,k-1, width, height);
					int maxIdxJ = getIndex3d(i,maxJLoc,k-1, width, height);
					int maxIdxPrev = getIndex3d(i,j,k-1, width, height);
					int prevDiagIdx = getIndex3d(i-1, j-1, k, width, height);
					bool newMaxI = false;
					bool newMaxJ = false;
					if(G[maxIdxPrev] >= (G[maxIdxI] - ((float)(i - maxILoc[j] -1) * this->_ext) + this->_open))
					{
						newMaxI = true;
					}
					if(G[maxIdxPrev] >= ( G[maxIdxJ] -  ((float)(j - maxJLoc -1) * this->_ext) + this->_open))
					{
						newMaxJ = true;
					}
					u = G[maxIdxJ] - (((float)(j-maxJLoc -1)*this->_ext) + this->_open);
					w = G[maxIdxI] - (((float)(i-maxILoc[j]-1)*this->_ext)+this->_open);
					v = G[prevDiagIdx] + (float)getScore(text[i-1],pattern[j-1]);
					float theMax = max(u,max(v,w));
					if(theMax == v)
					{
						G[idx] = v;
						H[idx] = 0;

					}else if (theMax == w)
					{
						G[idx] = w;
						H[idx] = i-maxILoc[j];
					}else
					{
						G[idx] = u;
						H[idx] = 0-(j-maxJLoc);
					}
					if(newMaxI)
					{
						maxILoc[j] = i;
					}
					if(newMaxJ)
					{
						maxJLoc = j;
					}
				}
			}
		}
	}
	catch(...)
	{
		printf("Error in the alignment phase.");
		exit(-1);
	}

	if(this->_approach == 2)
	{
		backtrack_start = Clock::now();
		if(DEBUG)printf("I am doing backtracking\n");
		try{
			backTrackStart = 0;
			int idx = getIndex3d(backTrackStart,width-1,_gaps,width,height);
			int val = G[idx];
			for(int i = 0; i < height; i ++)
			{
				idx = getIndex3d(i,width-1,_gaps,width,height);
				if(G[idx] >= val) backTrackStart = i;
			}

			int i = backTrackStart;
			int j = pLength;
			int b = 0;
			while ((i >= 0) && (j >= 0) && (b < _gaps))
			{
				int idx = getIndex3d(i,j,_gaps,width, height);
				if (H[idx] < 0)
				{
					gap_pos[b] = j;
					gap_len[b] = - H[idx];
					where[b] = 1;
					j = j + H[idx];
					b = b + 1;
				}
				else if (H[idx] > 0)
				{
					gap_pos[b] = i;
					gap_len[b] =  H[idx];
					where[b] = 2;
					i = i - H[idx];
					b = b + 1;
				}
				else if (H[idx] == 0)
				{
					i --;
					j --;
				}
				else
				{
					throw b;
				}

			}

		}catch(int e)
		{
				printf("Backtracking : Error at Gap %d",e);
				return 0;
		}
	delete(gap_pos);
	delete(gap_len);
	delete(where);
	backtrack_stop = Clock::now();
	auto diff = backtrack_stop - backtrack_start;
	milliseconds = chrono::duration <float, milli> (diff).count();
	this->total_backtrack_time += milliseconds;
	}
	delete(G);
	delete(H);

	return 0;
}

int GapsMisFBJ::runCPU()
{
	int myJobID = 0;
	host_start = Clock::now();

	for(int tNo = 0; tNo < this->_numTexts; tNo ++)
	{
		for(int pNo = 0; pNo < this->_numPatterns; pNo ++)
		{
			Sequence* myText = _texts.at(tNo);
			Sequence* myPattern = _patterns.at(pNo);
			int tLength = myText->length;
			int pLength = myPattern->length;
			int* myTextSrc = myText->mapped;
			int* myPatternSrc = myPattern->mapped;
			this->cpu_alignment(myTextSrc, myPatternSrc, tLength, pLength);
			myJobID ++;
		}
	}
	host_stop = Clock::now();
	auto diff = host_stop - host_start;
	milliseconds = chrono::duration <float, milli> (diff).count();
	this->total_host_time += milliseconds;

	return 0;
}

int GapsMisFBJ::runGPU_Simple_Align_Discard()
{
	gpu_start = Clock::now();
	copyConstantSymbols(this->_open, this->_ext, this->_gaps, this->_aLength, this->_maxTLength, this->_maxPLength, this->_pAlign, this->_maxTLength+1, this->_width, this->_widthAlign, this->_deviceMatrixSize, this->_score);
	gpu_stop = Clock::now();
	auto diff = gpu_stop - gpu_start;
	milliseconds = chrono::duration <float, milli> (diff).count();
	this->total_gpu_time += milliseconds;
	size_t textPinnedSize = this->_tAlign * sizeof(int);
	gpuErrchk(cudaMalloc((void**)&d_texts, textPinnedSize));
	size_t totalMem, remainingMem;
	gpuErrchk(cudaMemGetInfo(&remainingMem, &totalMem));
	size_t jobCost = 2*_deviceMatrixSize * sizeof(float) + _deviceMatrixSize * sizeof(int) + this->_pAlign * sizeof(int);
	int maxBatchSize = (int)floor(remainingMem/jobCost);
	int numBatchesNeeded = (int)ceil((double)this->_numPatterns/(double)maxBatchSize);
	int batchSize = 0;
	int startPattern = 0;
	size_t pattDeviceSize = maxBatchSize * this->_pAlign * sizeof(int);
	gpuErrchk(cudaMalloc((void**)&d_patterns, pattDeviceSize));
	for(int i = 0; i < this->_numTexts; i ++)
	{
		host_start = Clock::now();
		if(numBatchesNeeded == 1)
		{
			batchSize = this->_numPatterns;
		}else
		{
			batchSize = maxBatchSize;
		}
		gpu_start = Clock::now();
		gpuErrchk(cudaMemcpy(d_texts, &(h_texts[i * _tAlign]), _tAlign*sizeof(int), cudaMemcpyHostToDevice));
		gpu_stop = Clock::now();
		diff = gpu_stop - gpu_start;
		milliseconds = chrono::duration <float, milli> (diff).count();
		this->total_gpu_time += milliseconds;
		host_start = Clock::now();
		for(int batchNo = 0; batchNo < numBatchesNeeded; batchNo++)
		{
			host_stop = Clock::now();
			diff = host_stop - host_start;
			milliseconds = chrono::duration <float, milli> (diff).count();
			this->total_host_time += milliseconds;
			gpu_start = Clock::now();
			gpuErrchk(cudaMemcpy(d_patterns, &(h_patterns[startPattern *_pAlign]), batchSize * _pAlign * sizeof(int), cudaMemcpyHostToDevice ));

			gpuErrchk(cudaMalloc((void**)&d_A, batchSize*_deviceMatrixSize * sizeof(float)));
			gpuErrchk(cudaMalloc((void**)&d_B, batchSize*_deviceMatrixSize * sizeof(float)));
			gpuErrchk(cudaMalloc((void**)&d_H, batchSize*_deviceMatrixSize * sizeof(int)));

			int blockSize = 32* (int)ceil((float)_widthAlign/32.0f);
			size_t sharedSize = _pAlign * sizeof(int) //Pattern
								+ 2* (this->_widthAlign) * sizeof(int) //maxILoc and Max J Loc
								+ 2* (_widthAlign) * sizeof(float) //MaxIVal and Max J Val
								+ 4*(_widthAlign)*sizeof(float); //Row Caches
			SingleText<<<batchSize, blockSize, sharedSize>>>(
					d_texts, d_patterns,
					d_A, d_B);

			cudaDeviceSynchronize();
			cudaError_t  err = cudaGetLastError();
			if(err != cudaSuccess)
			{
				printf("[Single_Discard_N:%d Q:%d ] Error: Kernel\n%s",this->_maxTLength, this->_numTexts, cudaGetErrorString(err));
				exit (-1);
			}

			gpuErrchk(cudaFree(d_A));
			gpuErrchk(cudaFree(d_B));
			gpuErrchk(cudaFree(d_H));
			gpu_stop = Clock::now();
			diff = gpu_stop - gpu_start;
			milliseconds = chrono::duration <float, milli> (diff).count();
			this->total_gpu_time += milliseconds;
			host_start = Clock::now();
			startPattern += batchSize;

			if(startPattern + batchSize >= _numPatterns)
			{
				batchSize = _numPatterns - startPattern;
			}
		}

		startPattern = 0;
		host_stop = Clock::now();
		diff = host_stop - host_start;
		milliseconds = chrono::duration <float, milli> (diff).count();
		this->total_host_time += milliseconds;
	}
}


int GapsMisFBJ::runGPU_Single_Backtrack()
{
	if(DEBUG_GPU)printf("\n***\nGapsMisFBJ::runGPU_Single_Backtrack\n***\n");

	gpu_start = Clock::now();

	copyConstantSymbols(this->_open, this->_ext, this->_gaps, this->_aLength, this->_maxTLength, this->_maxPLength, this->_pAlign, this->_maxTLength+1, this->_width, this->_widthAlign, this->_deviceMatrixSize, this->_score);
	gpu_stop = Clock::now();
	auto diff = gpu_stop - gpu_start;
	milliseconds = chrono::duration <float, milli> (diff).count();
	this->total_gpu_time += milliseconds;
	size_t textPinnedSize = this->_tAlign * sizeof(int);
	gpuErrchk(cudaMalloc((void**)&d_texts, textPinnedSize));
	size_t totalMem, remainingMem;
	gpuErrchk(cudaMemGetInfo(&remainingMem, &totalMem));
	size_t jobCost = 2*_deviceMatrixSize * sizeof(float) + _deviceMatrixSize * sizeof(int) + this->_pAlign * sizeof(int) + 3*this->_gaps*sizeof(int);
	int maxBatchSize = (int)floor(remainingMem/jobCost);
	int numBatchesNeeded = (int)ceil((double)this->_numPatterns/(double)maxBatchSize);
	int batchSize = 0;
	int startPattern = 0;
	size_t pattDeviceSize = maxBatchSize * this->_pAlign * sizeof(int);
	gpuErrchk(cudaMalloc((void**)&d_patterns, pattDeviceSize));
	if(DEBUG_GPU)printf("Starting the Batching process\n");
	for(int i = 0; i < this->_numTexts; i ++)
	{
		host_start = Clock::now();
		if(numBatchesNeeded == 1)
		{
			batchSize = this->_numPatterns;
		}else
		{
			batchSize = maxBatchSize;
		}
		gpu_start = Clock::now();
		gpuErrchk(cudaMemcpy(d_texts, &(h_texts[i * _tAlign]), _tAlign*sizeof(int), cudaMemcpyHostToDevice));
		gpu_stop = Clock::now();
		diff = gpu_stop - gpu_start;
		milliseconds = chrono::duration <float, milli> (diff).count();
		this->total_gpu_time += milliseconds;
		host_start = Clock::now();
		for(int batchNo = 0; batchNo < numBatchesNeeded; batchNo++)
		{
			host_stop = Clock::now();
			diff = host_stop - host_start;
			milliseconds = chrono::duration <float, milli> (diff).count();
			this->total_host_time += milliseconds;
			gpu_start = Clock::now();
			gpuErrchk(cudaMemcpy(d_patterns, &(h_patterns[startPattern *_pAlign]), batchSize * _pAlign * sizeof(int), cudaMemcpyHostToDevice ));
			gpuErrchk(cudaMalloc((void**)&d_A, batchSize*_deviceMatrixSize * sizeof(float)));
			gpuErrchk(cudaMalloc((void**)&d_B, batchSize*_deviceMatrixSize * sizeof(float)));
			gpuErrchk(cudaMalloc((void**)&d_H, batchSize*_deviceMatrixSize * sizeof(int)));
			gpuErrchk(cudaMalloc((void**)&d_Bt, batchSize * 3 * _gaps * sizeof(int)));

			int blockSize = 32* (int)ceil((float)_widthAlign/32.0f);
			size_t sharedSize = (_pAlign * sizeof(int)) //Pattern
								+ (3* (this->_widthAlign) * sizeof(int)) //maxILoc and Max J Loc and hRow
								+ (2* (_widthAlign) * sizeof(float)) //MaxIVal and Max J Val
								+ (4*(_widthAlign)*sizeof(float)) //Row Caches
								+ (3*_gaps*sizeof(int)); //Backtracking
			SingleTextBacktrack<<<batchSize, blockSize, sharedSize>>>(
					d_texts, d_patterns,
					d_A, d_B, d_H, d_Bt);

			cudaDeviceSynchronize();
			cudaError_t  err = cudaGetLastError();
			if(err != cudaSuccess)
			{
				printf("[Single_backtrack_N:%d Q:%d ] Error: Kernel\n%s",this->_maxTLength, this->_numTexts, cudaGetErrorString(err));
				exit (-1);
			}
			gpuErrchk(cudaFree(d_A));
			gpuErrchk(cudaFree(d_B));
			gpuErrchk(cudaFree(d_H));
			gpuErrchk(cudaFree(d_Bt));
			gpu_stop = Clock::now();
			diff = gpu_stop - gpu_start;
			milliseconds = chrono::duration <float, milli> (diff).count();
			this->total_gpu_time += milliseconds;

			host_start = Clock::now();

			startPattern += batchSize;
			if(startPattern + batchSize >= _numPatterns)
			{
				batchSize = _numPatterns - startPattern;
			}
		}

		startPattern = 0;
		host_stop = Clock::now();
		diff = host_stop - host_start;
		milliseconds = chrono::duration <float, milli> (diff).count();
		this->total_host_time += milliseconds;
		if(DEBUG_GPU)printf("END of text %d\n", i);
	}
}

int GapsMisFBJ::runGPU_Single_Sendback()
{

	gpu_start = Clock::now();
	copyConstantSymbols(this->_open, this->_ext, this->_gaps, this->_aLength, this->_maxTLength, this->_maxPLength, this->_pAlign, this->_maxTLength+1, this->_width, this->_widthAlign, this->_deviceMatrixSize, this->_score);
	gpu_stop = Clock::now();
	auto diff = gpu_stop - gpu_start;
	milliseconds = chrono::duration <float, milli> (diff).count();
	this->total_gpu_time += milliseconds;
	size_t textPinnedSize = this->_tAlign * sizeof(int);
	gpuErrchk(cudaMalloc((void**)&d_texts, textPinnedSize));
	size_t totalMem, remainingMem;
	gpuErrchk(cudaMemGetInfo(&remainingMem, &totalMem));
	size_t jobCost = 2*_deviceMatrixSize * sizeof(float) + this->_pAlign * sizeof(int);
	int maxBatchSize = (int)floor(remainingMem/jobCost);
	int numBatchesNeeded = (int)ceil((double)this->_numPatterns/(double)maxBatchSize);
	int batchSize = 0;
	int startPattern = 0;
	size_t pattDeviceSize = maxBatchSize * this->_pAlign * sizeof(int);
	gpuErrchk(cudaMalloc((void**)&d_patterns, pattDeviceSize));
	for(int i = 0; i < this->_numTexts; i ++)
	{
		host_start = Clock::now();
		if(numBatchesNeeded == 1)
		{
			batchSize = this->_numPatterns;
		}else
		{
			batchSize = maxBatchSize;
		}

		gpu_start = Clock::now();
		gpuErrchk(cudaMemcpy(d_texts, &(h_texts[i * _tAlign]), _tAlign*sizeof(int), cudaMemcpyHostToDevice));
		gpu_stop = Clock::now();
		diff = gpu_stop - gpu_start;
		milliseconds = chrono::duration <float, milli> (diff).count();
		this->total_gpu_time += milliseconds;
		host_start = Clock::now();
		for(int batchNo = 0; batchNo < numBatchesNeeded; batchNo++)
		{
			host_stop = Clock::now();
			diff = host_stop - host_start;
			milliseconds = chrono::duration <float, milli> (diff).count();
			this->total_host_time += milliseconds;
			gpu_start = Clock::now();
			gpuErrchk(cudaMemcpy(d_patterns, &(h_patterns[startPattern *_pAlign]), batchSize * _pAlign * sizeof(int), cudaMemcpyHostToDevice ));
			gpuErrchk(cudaMalloc((void**)&d_A, batchSize*_deviceMatrixSize * sizeof(float)));
			gpuErrchk(cudaMalloc((void**)&d_B, batchSize*_deviceMatrixSize * sizeof(float)));
			gpuErrchk(cudaHostAlloc((void**)&h_H, batchSize*_deviceMatrixSize * sizeof(int), cudaHostAllocMapped));
			gpuErrchk(cudaHostGetDevicePointer(&d_H, h_H, 0));
			int blockSize = 32* (int)ceil((float)_widthAlign/32.0f);
			size_t sharedSize = _pAlign * sizeof(int) //Pattern
								+ 3* (this->_widthAlign) * sizeof(int) //maxILoc and Max J Loc and hRow
								+ 2* (_widthAlign) * sizeof(float) //MaxIVal and Max J Val
								+ 4*(_widthAlign)*sizeof(float);//Row Caches
			SingleTextSendback<<<batchSize, blockSize, sharedSize>>>(
					d_texts, d_patterns,
					d_A, d_B, d_H);

			cudaDeviceSynchronize();
			cudaError_t  err = cudaGetLastError();
			if(err != cudaSuccess)
			{
				printf("[Single_Sendback_N:%d Q:%d ] Error: Kernel\n%s",this->_maxTLength, this->_numTexts, cudaGetErrorString(err));
				exit (-1);

			}
			gpuErrchk(cudaFree(d_A));
			gpuErrchk(cudaFree(d_B));
			gpuErrchk(cudaFreeHost(h_H));
			gpu_stop = Clock::now();
			diff = gpu_stop - gpu_start;
			milliseconds = chrono::duration <float, milli> (diff).count();
			this->total_gpu_time += milliseconds;
			host_start = Clock::now();
			startPattern += batchSize;
			if(startPattern + batchSize >= _numPatterns)
			{
				batchSize = _numPatterns - startPattern;
			}

		}

		startPattern = 0;
		host_stop = Clock::now();
		diff = host_stop - host_start;
		milliseconds = chrono::duration <float, milli> (diff).count();
		this->total_host_time += milliseconds;
	}
}


 int GapsMisFBJ::runGPU_Multi_Align_Discard()
 {
	 	gpu_start = Clock::now();
	 	copyConstantSymbols(this->_open, this->_ext, this->_gaps, this->_aLength, this->_maxTLength, this->_tAlign,this->_maxPLength, this->_pAlign, this->_maxTLength+1, this->_width, this->_widthAlign, this->_deviceMatrixSize, this->_score, this->_numTexts, this->_numPatterns);
	 	gpu_stop = Clock::now();
	 	auto diff = gpu_stop - gpu_start;
	 	milliseconds = chrono::duration <float, milli> (diff).count();
	 	this->total_gpu_time += milliseconds;
		size_t textPinnedSize = this->_numTexts * this->_tAlign * sizeof(int);
		gpuErrchk(cudaMalloc(&d_texts, textPinnedSize));
		gpuErrchk(cudaMemcpy(d_texts, h_texts, _numTexts*_tAlign*sizeof(int), cudaMemcpyHostToDevice));
	 	size_t pattPinnedSize = this->_numPatterns * this->_pAlign * sizeof(int);
	 	gpuErrchk(cudaMalloc((void**)&d_patterns, pattPinnedSize));
		gpuErrchk(cudaMemcpy(d_patterns, h_patterns, pattPinnedSize, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();
	 	size_t totalMem = 0, remainingMem = 0;
	 	gpuErrchk(cudaMemGetInfo(&remainingMem, &totalMem));
	 	size_t jobCost = (2*_deviceMatrixSize * sizeof(float));
	 	int MaxBatchSize = (int)floor((remainingMem*0.8f)/jobCost)-1;

	 	bool alloc = false;
	 	while(!alloc & MaxBatchSize>0)
	 	{
	 		cudaError_t a = cudaMalloc((void**)&d_A, MaxBatchSize*_deviceMatrixSize * sizeof(float));
	 		cudaError_t b = cudaMalloc((void**)&d_B, MaxBatchSize*_deviceMatrixSize * sizeof(float));
	 		if(a != cudaErrorMemoryAllocation && b != cudaErrorMemoryAllocation)
	 		{
	 			alloc = true;
	 		}else{
	 			cudaFree(d_A);
	 			cudaFree(d_B);
	 			MaxBatchSize --;
	 		}
	 	}
	 	int numBatchesNeeded = ceil((double)_numSeqPairs/(double)MaxBatchSize);
	 	int batchSize = 0;
	 	int startSP = 0;
	 		batchSize = MaxBatchSize;
	 		if(numBatchesNeeded == 1)
	 		{
	 			batchSize = this->_numSeqPairs;
	 		}
	 		host_start = Clock::now();
	 		for(int batchNo = 0; batchNo < numBatchesNeeded; batchNo++)
	 		{
	 			gpu_start = Clock::now();
	 			int blockSize = 32* (int)ceil((float)_widthAlign/32.0f);
	 			size_t sharedSize = _pAlign * sizeof(int) //Pattern
	 								+ 2* (this->_widthAlign) * sizeof(int) //maxILoc and Max J Loc
	 								+ 2* (_widthAlign) * sizeof(float) //MaxIVal and Max J Val
	 								+ 4*(_widthAlign)*sizeof(float); //Row Caches
	 			MultiText<<<batchSize, blockSize, sharedSize>>>(
	 					d_texts, d_patterns,
	 					d_A, d_B, startSP);

	 			cudaDeviceSynchronize();
	 			cudaError_t  err = cudaGetLastError();
	 			if(err != cudaSuccess)
	 			{
	 				printf("[Multi_Discard_N:%d Q:%d ] Error: Kernel\n%s",this->_maxTLength, this->_numTexts, cudaGetErrorString(err));
	 				exit (-1);
	 			}
	 			gpu_stop = Clock::now();
	 			diff = gpu_stop - gpu_start;
	 			milliseconds = chrono::duration <float, milli> (diff).count();
	 			this->total_gpu_time += milliseconds;
	 			host_start = Clock::now();
	 			startSP += batchSize;
	 			if(startSP + batchSize >= _numSeqPairs)
	 			{
	 				batchSize = _numSeqPairs - startSP;
	 			}else
	 			{
	 				batchSize = MaxBatchSize;
	 			}
	 		}
	 		gpuErrchk(cudaFree(d_A));
	 		gpuErrchk(cudaFree(d_B));
	 		host_stop = Clock::now();
	 		diff = host_stop - host_start;
	 		milliseconds = chrono::duration <float, milli> (diff).count();
	 		this->total_host_time += milliseconds;
	 return 0;
 }

 int GapsMisFBJ::runGPU_Multi_Backtrack()
 {
	gpu_start = Clock::now();
	copyConstantSymbols(this->_open, this->_ext, this->_gaps, this->_aLength, this->_maxTLength, this->_tAlign,this->_maxPLength, this->_pAlign, this->_maxTLength+1, this->_width, this->_widthAlign, this->_deviceMatrixSize, this->_score, this->_numTexts, this->_numPatterns);
	gpu_stop = Clock::now();
	auto diff = gpu_stop - gpu_start;
	milliseconds = chrono::duration <float, milli> (diff).count();
	this->total_gpu_time += milliseconds;
	size_t textPinnedSize = this->_numTexts * this->_tAlign * sizeof(int);
	gpuErrchk(cudaMalloc(&d_texts, textPinnedSize));
	gpuErrchk(cudaMemcpy(d_texts, h_texts, _numTexts*_tAlign*sizeof(int), cudaMemcpyHostToDevice));
	size_t pattPinnedSize = this->_numPatterns * this->_pAlign * sizeof(int);
	gpuErrchk(cudaMalloc((void**)&d_patterns, pattPinnedSize));
	gpuErrchk(cudaMemcpy(d_patterns, h_patterns, pattPinnedSize, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	size_t totalMem = 0, remainingMem = 0;

	gpuErrchk(cudaMemGetInfo(&remainingMem, &totalMem));
	size_t jobCost = (2*_deviceMatrixSize * sizeof(float)) + (_deviceMatrixSize * sizeof(int)) + (3*_gaps*sizeof(int));
	int MaxBatchSize = (int)floor((remainingMem*0.8f)/jobCost)-1;

	bool alloc = false;
	while(!alloc && MaxBatchSize>0)
	{
		cudaError_t a = cudaMalloc((void**)&d_A, MaxBatchSize*_deviceMatrixSize * sizeof(float));
		cudaError_t b = cudaMalloc((void**)&d_B, MaxBatchSize*_deviceMatrixSize * sizeof(float));
		cudaError_t h = cudaMalloc((void**)&d_H, MaxBatchSize*_deviceMatrixSize * sizeof(int));
		cudaError_t bt = cudaMalloc((void**)&d_Bt, MaxBatchSize * 3 * _gaps * sizeof(int));
		if(a != cudaErrorMemoryAllocation && b != cudaErrorMemoryAllocation)
		{
			alloc = true;
		}else{
			cudaFree(d_A);
			cudaFree(d_B);
			cudaFree(d_H);
			cudaFree(d_Bt);
			MaxBatchSize --;
		}
	}

	int numBatchesNeeded = ceil((double)_numSeqPairs/(double)MaxBatchSize);
	 int batchSize = 0;
	int startSP = 0;
		batchSize = MaxBatchSize;
		if(numBatchesNeeded == 1)
		{
			batchSize = this->_numSeqPairs;
		}

		host_start = Clock::now();
		for(int batchNo = 0; batchNo < numBatchesNeeded; batchNo++)
		{

			gpu_start = Clock::now();
			int blockSize = 32* (int)ceil((float)_widthAlign/32.0f);
			size_t sharedSize = (_pAlign * sizeof(int)) //Pattern
										+ (3* (this->_widthAlign) * sizeof(int)) //maxILoc and Max J Loc and hRow
										+ (2* (_widthAlign) * sizeof(float)) //MaxIVal and Max J Val
										+ (4*(_widthAlign)*sizeof(float)) //Row Caches
										+ (3*_gaps*sizeof(int)); //Backtracking
			MultiTextBacktrack<<<batchSize, blockSize, sharedSize>>>(
					d_texts, d_patterns,
					d_A, d_B,d_H,  d_Bt,startSP);

			cudaDeviceSynchronize();
			cudaError_t  err = cudaGetLastError();
			if(err != cudaSuccess)
			{
				printf("[Multi_Discard_N:%d Q:%d ] Error: Kernel\n%s",this->_maxTLength, this->_numTexts, cudaGetErrorString(err));
				exit (-1);
			}
			gpu_stop = Clock::now();
			diff = gpu_stop - gpu_start;
			milliseconds = chrono::duration <float, milli> (diff).count();
			this->total_gpu_time += milliseconds;
			host_start = Clock::now();
			startSP += batchSize;
			if(startSP + batchSize >= _numSeqPairs)
			{
				batchSize = _numSeqPairs - startSP;
			}else
			{
				batchSize = MaxBatchSize;
			}
		}

		gpuErrchk(cudaFree(d_A));
		gpuErrchk(cudaFree(d_B));
		gpuErrchk(cudaFree(d_H));
		gpuErrchk(cudaFree(d_Bt));
		host_stop = Clock::now();
		diff = host_stop - host_start;
		milliseconds = chrono::duration <float, milli> (diff).count();
		this->total_host_time += milliseconds;
	 return 0;
 }


 int GapsMisFBJ::runGPU_Multi_Sendback()
 {
		gpu_start = Clock::now();
		copyConstantSymbols(this->_open, this->_ext, this->_gaps, this->_aLength, this->_maxTLength, this->_tAlign,this->_maxPLength, this->_pAlign, this->_maxTLength+1, this->_width, this->_widthAlign, this->_deviceMatrixSize, this->_score, this->_numTexts, this->_numPatterns);
		gpu_stop = Clock::now();
		auto diff = gpu_stop - gpu_start;
		milliseconds = chrono::duration <float, milli> (diff).count();
		this->total_gpu_time += milliseconds;
		size_t textPinnedSize = this->_numTexts * this->_tAlign * sizeof(int);
		gpuErrchk(cudaMalloc(&d_texts, textPinnedSize));
		gpuErrchk(cudaMemcpy(d_texts, h_texts, _numTexts*_tAlign*sizeof(int), cudaMemcpyHostToDevice));
		size_t pattPinnedSize = this->_numPatterns * this->_pAlign * sizeof(int);
		gpuErrchk(cudaMalloc((void**)&d_patterns, pattPinnedSize));
		gpuErrchk(cudaMemcpy(d_patterns, h_patterns, pattPinnedSize, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();
		size_t totalMem = 0, remainingMem = 0;
		gpuErrchk(cudaMemGetInfo(&remainingMem, &totalMem));
		size_t jobCost = (2*_deviceMatrixSize * sizeof(float)) + (_deviceMatrixSize * sizeof(int));
		int MaxBatchSize = (int)floor((remainingMem*0.8f)/jobCost)-1;
		bool alloc = false;
		while(!alloc && MaxBatchSize>0)
		{
			cudaError_t a = cudaMalloc((void**)&d_A, MaxBatchSize*_deviceMatrixSize * sizeof(float));
			cudaError_t b = cudaMalloc((void**)&d_B, MaxBatchSize*_deviceMatrixSize * sizeof(float));
			cudaError_t h = cudaHostAlloc(&d_H, MaxBatchSize*_deviceMatrixSize * sizeof(int), cudaHostAllocMapped);
			if(a != cudaErrorMemoryAllocation && b != cudaErrorMemoryAllocation)
			{
				alloc = true;
			}else{
				cudaFree(d_A);
				cudaFree(d_B);
				cudaFreeHost(d_H);
				MaxBatchSize --;
			}
		}

		int numBatchesNeeded = ceil((double)_numSeqPairs/(double)MaxBatchSize);
		int batchSize = 0;
		int startSP = 0;

			batchSize = MaxBatchSize;
			if(numBatchesNeeded == 1)
			{
				batchSize = this->_numSeqPairs;
			}

			host_start = Clock::now();
			for(int batchNo = 0; batchNo < numBatchesNeeded; batchNo++)
			{
				gpu_start = Clock::now();
				int blockSize = 32* (int)ceil((float)_widthAlign/32.0f);
				size_t sharedSize = (_pAlign * sizeof(int)) //Pattern
											+ (3* (this->_widthAlign) * sizeof(int)) //maxILoc and Max J Loc and hRow
											+ (2* (_widthAlign) * sizeof(float)) //MaxIVal and Max J Val
											+ (4*(_widthAlign)*sizeof(float)); //Row Caches
				MultiTextSendBack<<<batchSize, blockSize, sharedSize>>>(
						d_texts, d_patterns,
						d_A, d_B,d_H, startSP);

				cudaDeviceSynchronize();
				cudaError_t  err = cudaGetLastError();
				if(err != cudaSuccess)
				{
					printf("[Multi_Discard_N:%d Q:%d ] Error: Kernel\n%s",this->_maxTLength, this->_numTexts, cudaGetErrorString(err));
					exit (-1);
				}

				gpu_stop = Clock::now();
				diff = gpu_stop - gpu_start;
				milliseconds = chrono::duration <float, milli> (diff).count();
				this->total_gpu_time += milliseconds;
				host_start = Clock::now();
				startSP += batchSize;
				if(startSP + batchSize >= _numSeqPairs)
				{
					batchSize = _numSeqPairs - startSP;
				}else
				{
					batchSize = MaxBatchSize;
				}
			}
			gpuErrchk(cudaFree(d_A));
			gpuErrchk(cudaFree(d_B));
			gpuErrchk(cudaFreeHost(d_H));
			host_stop = Clock::now();
			diff = host_stop - host_start;
			milliseconds = chrono::duration <float, milli> (diff).count();
			this->total_host_time += milliseconds;
		 return 0;

 }


int GapsMisFBJ::run()
{

	this ->processScoreMatrix(this ->_scoreFile);
	vector<Sequence*>& textRef = this ->_texts;
	this ->processSequenceFile(this ->_textsFile,textRef);
	vector<Sequence*>& pattRef = this ->_patterns;
	this ->processSequenceFile(this ->_patternsFile, pattRef);
	this-> _numTexts = textRef.size();
	this -> _numPatterns = pattRef.size();
	this ->_numSeqPairs = this->_numPatterns * this ->_numTexts;
	this->_maxTLength = this->_texts.at(0)->length;
	this ->_maxPLength = this->_patterns.at(0)->length;
	this->_matrixSize = (this->_maxTLength +1) * (this->_maxPLength +1) * (this->_gaps + 1) ;
	this->_tAlign = (_maxTLength + 4 -1) - (_maxTLength + 4 -1) % 4;
	this ->_pAlign = (_maxPLength + 4 - 1) - (_maxPLength + 4 - 1) % 4;
	this ->_width = this->_maxPLength + 1;
	this ->_widthAlign = (this->_width + 4 - 1) - (this->_width + 4 - 1) % 4;
	gpuErrchk(cudaHostAlloc((void**)&h_texts, _numTexts * _tAlign * sizeof(int), cudaHostAllocMapped));
	if (h_texts)
	{
		memset(h_texts, -1, _numTexts * _tAlign * sizeof(int));

		for(int i = 0; i < _numTexts; i ++)
		{
			memcpy(h_texts+(i*_tAlign), this->_texts.at(i)->mapped, this->_texts.at(i)->length * sizeof(int));
		}
	}
	gpuErrchk(cudaHostAlloc((void**)&h_patterns, _numPatterns * _pAlign * sizeof(int), cudaHostAllocMapped));
		if (h_texts)
		{
			memset(h_patterns, -1, _numPatterns * _pAlign * sizeof(int));

			for(int i = 0; i < _numPatterns; i ++)
			{
				memcpy(h_patterns+(i*_pAlign), this->_patterns.at(i)->mapped, this->_patterns.at(i)->length * sizeof(int));
			}
		}
		this -> _deviceMatrixSize = (this->_maxTLength +1) * (this->_widthAlign);

	total_start = Clock::now();
	switch(this -> _approach){
		case 1:
		case 2:
			this->runCPU();
			break;
		case 3:
			this->runGPU_Simple_Align_Discard(); //GPU-S-A
			break;
		case 4:
			this->runGPU_Multi_Align_Discard();//GPU-M-A
			break;
		case 5:
			this->runGPU_Single_Backtrack();//GPU-S-B
			break;
		case 6:
			this->runGPU_Multi_Backtrack();//GPU-M-B
			break;
		case 7:
			this->runGPU_Single_Sendback();//GPU-S-H
			break;
		case 8:
			this->runGPU_Multi_Sendback();//GPU-M-H
			break;
		default:
			return -1;
	};
	total_stop = Clock::now();
	auto diff = total_stop - total_start;
	milliseconds = chrono::duration <float, milli> (diff).count();
	total_time += milliseconds;
	printf("Total Time: %f ms", total_time);
	return 0;
}

string GapsMisFBJ::getHeaders()
{
	string theHeaders = "ERROR";
		switch (this ->_approach )
		{
		case 1:
		case 2:
			return this->_headers[0].c_str();
		case 3:
		case 4:
			return this ->_headers[1].c_str();
		case 5:
		case 6:
		case 7:
		case 8:
		case 9:
			return this ->_headers[2].c_str();
		};

		return "ERROR - NO HEADERS";
}

double GapsMisFBJ::getTotalTime()
{
	return (double)this->total_time;
}

double GapsMisFBJ::getHostTime()
{
	if(this->_approach >= 3)
	{
		return (double)this->total_host_time;
	}

	return 0.0;
}

double GapsMisFBJ::getGPUTime()
{
	if(this->_approach >= 3)
	{
		return (double) (this->total_gpu_time);
	}

	return 0.0;
}

double GapsMisFBJ::getBacktrackTime()
{
	if (this->_approach == 5 || this->_approach == 6 || this->_approach == 8 || this->_approach == 2)
	{
		return (double) this->total_backtrack_time;
	}

	return 0.0;
}

int GapsMisFBJ::getNumCells()
{
	int ans = -1;
	switch(this->_approach)
	{
	case 1:
		ans = (this->_numSeqPairs * this->_matrixSize * (this->_gaps+1) *2);
		break;
	case 2:
		ans = (this->_numSeqPairs * this->_matrixSize * (this->_gaps+1)  * 2) + (3 * this->_gaps);
		break;
	case 3:
		ans = (this->_numSeqPairs * this->_matrixSize * (this->_gaps+1));
		break;
	case 4:
		ans = (this->_numSeqPairs * this->_matrixSize * (this->_gaps+1) * 2);
		break;
	case 5:
	case 6:
	case 7:
	case 8:
	case 9:
		ans = (this->_numSeqPairs * this->_matrixSize * (this->_gaps+1) * 2) + (3 * this->_gaps);
		break;
	default:
		break;

	};
	printf("Number of Cells is %d\n", ans);
	return ans;
}
