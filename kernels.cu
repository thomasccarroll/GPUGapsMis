//GPUGapsMis
//Thomas Carroll - 2017

#include "kernels.h"

__constant__ float OPEN; //Gap Opening Penalty
__constant__ float EXT; //Gap Extension Penalty
__constant__ int GAPS; //Number of Gaps Allowed
__constant__ int A_LEN; //Alphabet Length
__constant__ int T_LEN; //Text Length
__constant__ int X_LEN; //Pattern Length
__constant__ int T_OFFSET; //Text Offset
__constant__ int X_ALIGN; //Pattern Offset
__constant__ int HEIGHT; //Height of the Matrix
__constant__ int WIDTH; //Width of the Matrix
__constant__ int W_ALIGN; //Aligned to 4, width of matrix
__constant__ int MATRIX_SIZE;  //Matrix Size (a *single* matrix for ANY job is this size)
__constant__ int S[225]; //Score Matrix (15 nucleotide codes in the scoring matrices used)
__constant__ int Q; //Number of Texts in Batch
__constant__ int R; //Number of patterns in Batch

void copyConstantSymbols(float open, float ext, int gaps, int alen, int tlen, int xlen, int xAlign, int height, int width, int wAlign, int matrixSize, int*score)
{

		gpuErrchk(cudaMemcpyToSymbol(OPEN, &open, sizeof(float)));
		gpuErrchk(cudaMemcpyToSymbol(EXT, &ext, sizeof(float)));
		gpuErrchk(cudaMemcpyToSymbol(GAPS, &(gaps), sizeof(int)));
		gpuErrchk(cudaMemcpyToSymbol(A_LEN, &alen, sizeof(int)));
		gpuErrchk(cudaMemcpyToSymbol(T_LEN,&tlen, sizeof(int)));
		gpuErrchk(cudaMemcpyToSymbol(X_LEN,&xlen, sizeof(int)));
		gpuErrchk(cudaMemcpyToSymbol(X_ALIGN,&xlen, sizeof(int)));
		gpuErrchk(cudaMemcpyToSymbol(HEIGHT,&height, sizeof(int)));
		gpuErrchk(cudaMemcpyToSymbol(WIDTH,&width, sizeof(int)));
		gpuErrchk(cudaMemcpyToSymbol(W_ALIGN,&wAlign, sizeof(int)));
		gpuErrchk(cudaMemcpyToSymbol(MATRIX_SIZE, &matrixSize, sizeof(int)));
		gpuErrchk(cudaMemcpyToSymbol(S, score, 225*sizeof(int)));
}


void copyConstantSymbols(float open, float ext, int gaps, int alen, int tlen, int tAlign, int xlen, int xAlign, int height, int width, int wAlign, int matrixSize, int*score, int q, int r)
{

	copyConstantSymbols(open, ext, gaps, alen, tlen, xlen, xAlign, height, width, wAlign, matrixSize, score); //Copy some of the symbols which are same for Single Text kernel

	gpuErrchk(cudaMemcpyToSymbol(T_OFFSET, &tAlign, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol(Q, &q, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol(R, &r, sizeof(int)));
}


__device__ int getIndex(int i, int j, int C, int jobID)
{
	return jobID*MATRIX_SIZE + ((i * C) + j) ;
}


__device__ int getIndex(int i, int j, int C)
{
	return ((i * C) + j) ;
}

__device__ __host__ float getGapScore(int start, int finish, float o, float e)
{
	float ans = 0.0f - ((((float)finish - (float)start - 1.0f)*e) + o);
	return ans;
}

//Kernel for GPU-S-A
__global__ void SingleText(
					int* T, //the single Text
					int* X, //The many Patterns
					float* firstG, //Current and Previous G Matrices
					float* secondG //Current and Previous G Matrices
				)
{
	//Pointers to global memory G Matrices.
	float* currG, *prevG;

	//Pointers to shared memory row caches (swapped at each row iteration)
	float* currentRow, *previousRow, *prevGprevRow, *prevGcurrRow;

	//Holds the text character currently being considered
	int textChar;

	//Obtain thread specific variables, store in private memory
	int blockSize = blockDim.x;
	int jobID = blockIdx.x;
	int localThreadID = threadIdx.x;
	int workID = localThreadID + 1;


	///***** Shared Memory Declarations

			//Declare the Shared memory block, and split it up using pointers.
			extern __shared__ int shared[];

			//Holds the Pattern for this job
			int *patt = &shared[0];

			//Hold the Maximum I value location found in the matrix G(k-1) for each j value in G(k)
			int *maxILoc = &patt[X_ALIGN];

			//Holds the Maximum I value found in the matrix G(k-1), for each j value in G(k)
			float *maxIVal = (float*)&maxILoc[W_ALIGN];

			//Hold the Maximum J value location found in the matrix G(k-1) for each j value in G(k)
			int *maxJLoc = (int*)&maxIVal[W_ALIGN];

			//Holds the Maximum J value found in the matrix G(k-1), for each j value in G(k)
			float *maxJVal = (float*)&maxJLoc[W_ALIGN];

			//A,B,C,D are to hold row-caches (currentRow, previousRow, prevGcurrRow, prevGprevRow)
			float *A = (float*)&maxJVal[W_ALIGN];
			float *B = &A[W_ALIGN];
			float *C = &B[W_ALIGN];
			float *D = &C[W_ALIGN];

	///***** End Shared Memory Declarations

	//Point to row caches to start
	currentRow = A; previousRow = B; prevGcurrRow = C; prevGprevRow = D;

	//Number of tiles we need to break the row (calc for a pattern) into
	int numTiles = ceil((float)WIDTH/(float)blockSize) +1;


	//Copy the Pattern in to Shared Memory
	int numVecs = ceil((float)X_ALIGN/4.0f);
	int numVecTiles = ceil((float)numVecs/(float)blockSize);
	int gIdx = 0; //Where in Global memory does the thread write/read?
	int sIdx = 0; //Where in shared memory does the thread write/read?

	for(int t = 0; t < numVecTiles; t ++)
	{
		sIdx = (t * blockSize) + localThreadID;
		if(sIdx < numVecs){
			//Calculate index of Pattern
			gIdx = jobID * numVecs + localThreadID;
			reinterpret_cast<int4*>(patt)[sIdx] = reinterpret_cast<int4*>(X)[gIdx];
		}
	}
	//Wait until every thread has finished copying the pattern.
	__syncthreads();

	//Calculate G_0 as PreviousG
	prevG = firstG;
	currG = secondG;

	//Initiate Row 0 of matrix G[0][*][0] as prevRow
	//Step 1 - initialize cell in column 0 (but have everyone initialize a cell), only need work items to do this once, so don't use tiles
	int j =  workID;

	if (localThreadID ==0)
	{
		previousRow[0] = 0.0f;
	}

	//Step 2 - Every work item calculates for a pair (or multiple pairs of) of characters in the text and pattern, so we have to use tiles.
	//Row and Column 0 are comparing with the Empty String, so we don't need to obtain a charachter.
	for(int t = 0; t < numTiles; t ++)
	{
		j = (t * blockSize) + workID;
		if (j < WIDTH)
		{
			previousRow[j] = 0.0f-(((float)T_LEN*(float)X_LEN)*EXT + OPEN);
		}
	}
	__syncthreads();


	//Place Previous Row into Global memory
	numVecs = ceil((float)W_ALIGN/4.0f);
	int numMatVecs = HEIGHT * numVecs;
	numVecTiles = ceil((float)numVecs/(float)blockSize);
	gIdx = 0;
	sIdx = 0;

	for(int t = 0; t < numVecTiles; t ++)
	{
		sIdx = (t * blockSize) + localThreadID;
		if(sIdx < numVecs){
			gIdx = jobID * numMatVecs + sIdx;
			reinterpret_cast<float4*>(prevG)[gIdx] = reinterpret_cast<float4*>(previousRow)[sIdx];
		}
	}

	//Now initialize the remaining rows of G0 and H0
	for(int i = 1; i < HEIGHT ; i++)//Calculating for row i
	{
		//Obtain the text character for this row
		textChar = T[i-1];

		// Step 1 - initialize cell in column 0
		if (localThreadID == 0)
		{
			currentRow[0] = 0.0f- ((((float)T_LEN * (float)X_LEN) * EXT) + OPEN);
		}

		//Step 2 - calculate the remainder of cells in the Row
		for(int t = 0; t < numTiles; t ++)
		{
			//j corresponds the the character in the pattern for which this thread calculates for
			j = (t * blockSize) + workID;

			if (j < WIDTH)
			{

				//Calculate score from carrying on previous alignment
				currentRow[j] = previousRow[j-1] + (float)S[getIndex(textChar,patt[j-1],A_LEN)];

				if(i!=j)
				{
					//In G_0, non-diagonal cells are given maximum penalty,
					//So that they are not selected as the optimum cell to insert gap from in
					// G_1
					currentRow[j] = 0.0f - ((((float)T_LEN * (float)X_LEN) * EXT) + OPEN);
				}
			}
		}

		//Now place G_0[i][*] into Global memory
		for(int t = 0; t < numVecTiles; t ++)
		{
			sIdx = (t * blockSize) + localThreadID;
			if(sIdx < numVecs){
				gIdx = jobID * numMatVecs + (i * numVecs) + sIdx;
				reinterpret_cast<float4*>(prevG)[gIdx] = reinterpret_cast<float4*>(currentRow)[sIdx];
			}
		}

		//Swap row pointers
		float* temp = previousRow;
		previousRow = currentRow;
		currentRow = temp;
	}//End for (height)

	__syncthreads(); //Wait until all threads have stopped calculating for G_0

	//Now calculate for all allowable number of gaps in G_k
	for(int k = 1; k <= GAPS; k++)
	{
		//Step 1: Initialize the 0 row
		//Step 1a - initialize cell in column 0 (but have everyone initialize a cell)
		int j =  workID;

		if (localThreadID ==0)
		{
			previousRow[0] = 0.0f;
		}

		//Step 1b - Every work item calculates for a pair (or multiple) of characters in the text and pattern,
		//Row and Column 0 are comparing with the Empty String, so we don't need to obtain a charachter.
		for(int t = 0; t < numTiles; t ++)
		{
			j = (t * blockSize) + workID;
			if (j < WIDTH)
			{
				previousRow[j] = getGapScore(0,j,OPEN,EXT);
			}
		}
		__syncthreads();

		//Place Previous Row into Global memory
		for(int t = 0; t < numVecTiles; t ++)
		{
			sIdx = (t * blockSize) + localThreadID;
			if(sIdx < numVecs){
				gIdx = jobID * numMatVecs + (0 * numVecs) + sIdx;
				reinterpret_cast<float4*>(currG)[gIdx] = reinterpret_cast<float4*>(previousRow)[sIdx];
			}
		}

		//Step2: Calculate for the subsequent rows of G_k

		//Pull in the row of G_{k-1}[0][*] as first prevGprevRow
		for(int t = 0; t < numVecTiles; t ++)
		{
			sIdx = (t * blockSize) + localThreadID;
			if(sIdx < numVecs){
				gIdx = jobID * numMatVecs + (0 * numVecs) + sIdx;
				reinterpret_cast<float4*>(prevGprevRow)[sIdx] = reinterpret_cast<float4*>(prevG)[gIdx];
			}
		}
		__syncthreads();

		//Initialize MaxI
		for(int t = 0; t < numTiles; t ++)
		{
			int sIdx = (t*blockSize) + localThreadID;
			if(sIdx < WIDTH)
			{
				maxILoc[sIdx] = 0;
				maxIVal[sIdx] = prevGprevRow[sIdx];
			}
		}


		for(int i = 1; i < HEIGHT; i ++)
		{
			//Now calculate G_k[i][*]

			//Step 1: Pull in the Row caches
			for(int t = 0; t < numVecTiles; t ++)
			{
				sIdx = (t * blockSize) + localThreadID;
				if(sIdx < numVecs){
					gIdx = jobID * numMatVecs + (i * numVecs) + sIdx;
					reinterpret_cast<float4*>(prevGcurrRow)[sIdx] = reinterpret_cast<float4*>(prevG)[gIdx];
				}
			}
			__syncthreads();

			//Calculate the MaxJ array for this row

			//Initialize the maxJLoc and maxJVal
			for(int t = 0; t < numTiles; t++)
			{
				int sIdx = (t*blockSize) + localThreadID;
				if(sIdx < WIDTH)
				{
					maxJLoc[sIdx] = sIdx;
					maxJVal[sIdx] = prevGcurrRow[sIdx];
				}
			}
			__syncthreads();

			//Now calculate the maxJ with tree-based method
			int d = 1;

			while(d <= WIDTH)
			{

				for(int t = 0; t < numTiles; t++)
				{
					int sIdx = (t*blockSize) + localThreadID;
					if((sIdx < WIDTH) && (sIdx >= d))
					{


						int loc1 = maxJLoc[sIdx-d];
						int loc2 = maxJLoc[sIdx];

						float val1 = maxJVal[sIdx-d];
						float val2 = maxJVal[sIdx];

						float pen1 = getGapScore(loc1,sIdx,EXT,OPEN);
						float pen2 = getGapScore(loc2,sIdx,EXT,OPEN);

						float val = val2;
						int loc = loc2;
						if(val1 + pen1 >= val2 + pen2)
						{
							val = val1;
							loc = loc1;
						}

						maxJVal[sIdx] = val;
						maxJLoc[sIdx] = loc;

					}
				}
				d = d*2;
				__syncthreads();
			}

			//Initialize the 0 cell
			if(localThreadID == 0)
			{
				currentRow[0] = getGapScore(0,i,EXT,OPEN);
			}


			//Update max I
			for(int t = 0; t < numTiles; t++)
			{
				j = (t * blockSize + workID);
				if(j < WIDTH)
				{

					//Calculate the MaxI
					if((prevGprevRow[j] + getGapScore(i-1,i,EXT,OPEN)) > (maxIVal[j] + getGapScore(maxILoc[j],i,EXT,OPEN)))
					{
						maxIVal[j] = prevGprevRow[j];
						maxILoc[j] = i-1;
					}
				}
			}

			//Obtain text character required
			textChar = T[i-1];

			//Calculate cell
			for(int t = 0; t < numTiles; t++)
			{
				j = (t * blockSize + workID);
				if(j < WIDTH)
				{

					//Now calculate the Values to be compared.
					float carryOnDiag,gapInText,gapInPatt;

					carryOnDiag = previousRow[j-1] + S[getIndex(textChar,patt[j-1],A_LEN)];
					gapInPatt = maxIVal[j] + getGapScore(maxILoc[j],i,EXT,OPEN);
					gapInText = maxJVal[j-1] + getGapScore(maxJLoc[j-1],j,EXT,OPEN);

					currentRow[j] =  max(carryOnDiag,max(gapInText,gapInPatt));

				}
			}


			//Place CurrentRow into Global memory
			for(int t = 0; t < numVecTiles; t ++)
			{
				sIdx = (t * blockSize) + localThreadID;
				if(sIdx < numVecs){
					gIdx = jobID * numMatVecs + (i * numVecs) + sIdx;
					reinterpret_cast<float4*>(currG)[gIdx] = reinterpret_cast<float4*>(currentRow)[sIdx];
				}
			}
			__syncthreads();

			//Swap the prevGcurrRow and prevGprevRow pointers
			float* temp = prevGprevRow;
			prevGprevRow = prevGcurrRow;
			prevGcurrRow = temp;

			//Swap currRow and prevRow pointers
			temp = previousRow;
			previousRow = currentRow;
			currentRow = temp;

		} //End For (i = 1 -> HEIGHT)

		//Swap currG and prevG
		float* temp = prevG;
		prevG = currG;
		currG = temp;
	}//End For (K = 1->Gaps)

} //End of Kernel

//Kernel for GPU-S-B
__global__ void SingleTextBacktrack(
					int* T, //the single Text
					int* X, //The many Patterns
					float* firstG, //Current and Previous G Matrices
					float* secondG, //Current and Previous G Matrices
					int* H, //H matrices
					int* Bt //Backtracking data
				)
{
	float* currG, *prevG;

	float* currentRow, *previousRow, *prevGprevRow, *prevGcurrRow;

	int textChar;

	int blockSize = blockDim.x;
	int jobID = blockIdx.x;
	int localThreadID = threadIdx.x;
	int workID = localThreadID + 1;

	bool LAST = false;

			extern __shared__ int shared[];


			int *patt = &shared[0];
			int *maxILoc = &patt[X_ALIGN];
			float *maxIVal = (float*)&maxILoc[W_ALIGN];
			int *maxJLoc = (int*)&maxIVal[W_ALIGN];
			float *maxJVal = (float*)&maxJLoc[W_ALIGN];
			float *A = (float*)&maxJVal[W_ALIGN];
			float *B = &A[W_ALIGN];
			float *C = &B[W_ALIGN];
			float *D = &C[W_ALIGN];
			int *hRow = (int*)&D[W_ALIGN];
			int *backtrack = &hRow[W_ALIGN];

	currentRow = A; previousRow = B; prevGcurrRow = C; prevGprevRow = D;

	int numTiles = ceil((float)WIDTH/(float)blockSize) +1;

	int numVecs = ceil((float)X_ALIGN/4.0f);
	int numVecTiles = ceil((float)numVecs/(float)blockSize);
	int gIdx = 0;
	int sIdx = 0;

	for(int t = 0; t < numVecTiles; t ++)
	{
		sIdx = (t * blockSize) + localThreadID;
		if(sIdx < numVecs){
			gIdx = jobID * numVecs + localThreadID;
			reinterpret_cast<int4*>(patt)[sIdx] = reinterpret_cast<int4*>(X)[gIdx];
		}
	}
	__syncthreads();
	prevG = firstG;
	currG = secondG;
	int j =  workID;
	if (localThreadID ==0)
	{
		previousRow[0] = 0.0f;
	}
	for(int t = 0; t < numTiles; t ++)
	{
		j = (t * blockSize) + workID;
		if (j < WIDTH)
		{
			previousRow[j] = 0.0f-(((float)T_LEN*(float)X_LEN)*EXT + OPEN);
		}
	}
	__syncthreads();
	numVecs = ceil((float)W_ALIGN/4.0f);
	int numMatVecs = HEIGHT * numVecs;
	numVecTiles = ceil((float)numVecs/(float)blockSize);
	gIdx = 0;
	sIdx = 0;

	for(int t = 0; t < numVecTiles; t ++)
	{
		sIdx = (t * blockSize) + localThreadID;
		if(sIdx < numVecs){
			gIdx = jobID * numMatVecs + sIdx;
			reinterpret_cast<float4*>(prevG)[gIdx] = reinterpret_cast<float4*>(previousRow)[sIdx];
		}
	}
	for(int i = 1; i < HEIGHT ; i++)
	{

		textChar = T[i-1];
		if (localThreadID == 0)
		{
			currentRow[0] = 0.0f- ((((float)T_LEN * (float)X_LEN) * EXT) + OPEN);
		}
		for(int t = 0; t < numTiles; t ++)
		{
			j = (t * blockSize) + workID;

			if (j < WIDTH)
			{
				currentRow[j] = previousRow[j-1] + (float)S[getIndex(textChar,patt[j-1],A_LEN)];
				if(i!=j)
				{
					currentRow[j] = 0.0f - ((((float)T_LEN * (float)X_LEN) * EXT) + OPEN);
				}
			}
		}
		for(int t = 0; t < numVecTiles; t ++)
		{
			sIdx = (t * blockSize) + localThreadID;
			if(sIdx < numVecs){
				gIdx = jobID * numMatVecs + (i * numVecs) + sIdx;
				reinterpret_cast<float4*>(prevG)[gIdx] = reinterpret_cast<float4*>(currentRow)[sIdx];
			}
		}
		float* temp = previousRow;
		previousRow = currentRow;
		currentRow = temp;
	}
	__syncthreads();

	for(int k = 1; k <= GAPS; k++)
	{

		if (k == GAPS) LAST = true;

		int j =  workID;

		if (localThreadID ==0)
		{
			previousRow[0] = 0.0f;
			if(LAST) hRow[0] = 0;
		}
		for(int t = 0; t < numTiles; t ++)
		{
			j = (t * blockSize) + workID;
			if (j < WIDTH)
			{
				previousRow[j] = getGapScore(0,j,OPEN,EXT);
				if(LAST) hRow[j] = -j;
			}
		}
		__syncthreads();

		for(int t = 0; t < numVecTiles; t ++)
		{
			sIdx = (t * blockSize) + localThreadID;
			if(sIdx < numVecs){
				gIdx = jobID * numMatVecs + (0 * numVecs) + sIdx;
				reinterpret_cast<float4*>(currG)[gIdx] = reinterpret_cast<float4*>(previousRow)[sIdx];
				if(LAST)reinterpret_cast<int4*>(H)[gIdx] = reinterpret_cast<int4*>(hRow)[sIdx];
			}
		}
		for(int t = 0; t < numVecTiles; t ++)
		{
			sIdx = (t * blockSize) + localThreadID;
			if(sIdx < numVecs){
				gIdx = jobID * numMatVecs + (0 * numVecs) + sIdx;
				reinterpret_cast<float4*>(prevGprevRow)[sIdx] = reinterpret_cast<float4*>(prevG)[gIdx];
			}
		}
		__syncthreads();
		for(int t = 0; t < numTiles; t ++)
		{
			int sIdx = (t*blockSize) + localThreadID;
			if(sIdx < WIDTH)
			{
				maxILoc[sIdx] = 0;
				maxIVal[sIdx] = prevGprevRow[sIdx];
			}
		}


		for(int i = 1; i < HEIGHT; i ++)
		{
			for(int t = 0; t < numVecTiles; t ++)
			{
				sIdx = (t * blockSize) + localThreadID;
				if(sIdx < numVecs){
					gIdx = jobID * numMatVecs + (i * numVecs) + sIdx;
					reinterpret_cast<float4*>(prevGcurrRow)[sIdx] = reinterpret_cast<float4*>(prevG)[gIdx];
				}
			}
			__syncthreads();

			for(int t = 0; t < numTiles; t++)
			{
				int sIdx = (t*blockSize) + localThreadID;
				if(sIdx < WIDTH)
				{
					maxJLoc[sIdx] = sIdx;
					maxJVal[sIdx] = prevGcurrRow[sIdx];
				}
			}

			__syncthreads();

			int d = 1;

			while(d <= WIDTH)
			{

				for(int t = 0; t < numTiles; t++)
				{
					int sIdx = (t*blockSize) + localThreadID;
					if((sIdx < WIDTH) && (sIdx >= d))
					{


						int loc1 = maxJLoc[sIdx-d];
						int loc2 = maxJLoc[sIdx];

						float val1 = maxJVal[sIdx-d];
						float val2 = maxJVal[sIdx];

						float pen1 = getGapScore(loc1,sIdx,EXT,OPEN);
						float pen2 = getGapScore(loc2,sIdx,EXT,OPEN);

						float val = val2;
						int loc = loc2;
						if(val1 + pen1 >= val2 + pen2)
						{
							val = val1;
							loc = loc1;
						}

						maxJVal[sIdx] = val;
						maxJLoc[sIdx] = loc;

					}
				}
				d = d*2;
				__syncthreads();
			}

			if(localThreadID == 0)
			{
				currentRow[0] = getGapScore(0,i,EXT,OPEN);
				if(LAST) hRow[0] = i;
			}

			for(int t = 0; t < numTiles; t++)
			{
				j = (t * blockSize + workID);
				if(j < WIDTH)
				{

					if((prevGprevRow[j] + getGapScore(i-1,i,EXT,OPEN)) > (maxIVal[j] + getGapScore(maxILoc[j],i,EXT,OPEN)))
					{
						maxIVal[j] = prevGprevRow[j];
						maxILoc[j] = i-1;
					}
				}
			}

			textChar = T[i-1];

			for(int t = 0; t < numTiles; t++)
			{
				j = (t * blockSize + workID);
				if(j < WIDTH)
				{

					float carryOnDiag,gapInText,gapInPatt;

					carryOnDiag = previousRow[j-1] + S[getIndex(textChar,patt[j-1],A_LEN)];
					gapInPatt = maxIVal[j] + getGapScore(maxILoc[j],i,EXT,OPEN);
					gapInText = maxJVal[j-1] + getGapScore(maxJLoc[j-1],j,EXT,OPEN);


					float theCalc = max(carryOnDiag,max(gapInText,gapInPatt));

					if(LAST)
					{
						int h = 0;
						if(gapInText == theCalc)
						{
							h = 0 - (j-maxJLoc[j-1]);
						}
						if(gapInPatt == theCalc)
						{
							h = i-maxILoc[j];
						}
						hRow[j] = h;
					}
					currentRow[j] = theCalc;

				}
			}
			for(int t = 0; t < numVecTiles; t ++)
			{
				sIdx = (t * blockSize) + localThreadID;
				if(sIdx < numVecs){
					gIdx = jobID * numMatVecs + (i * numVecs) + sIdx;
					reinterpret_cast<float4*>(currG)[gIdx] = reinterpret_cast<float4*>(currentRow)[sIdx];
					if(LAST)reinterpret_cast<int4*>(H)[gIdx] = reinterpret_cast<int4*>(hRow)[sIdx];
				}
			}
			__syncthreads();

			float* temp = prevGprevRow;
			prevGprevRow = prevGcurrRow;
			prevGcurrRow = temp;
			temp = previousRow;
			previousRow = currentRow;
			currentRow = temp;

		}

		float* temp = prevG;
		prevG = currG;
		currG = temp;
	}
	__syncthreads();
	if (localThreadID == 0)
	{

		int i = HEIGHT-1;
		int j = WIDTH-1;
		int b = 0;


			while ((i >= 0) && (j >= 0) && (b <= GAPS))
			{
				int val = H[getIndex(i,j,W_ALIGN,jobID)];
				int posIdx = b*3;
				int lenIdx = b*3 + 1;
				int whereIdx = b*3 + 2;


				if (val < 0)
				{
					backtrack[posIdx] = j;
					backtrack[lenIdx] = 0 - val;
					backtrack[whereIdx] = 1;
					j = j + val;
					b = b + 1;
				}
				if (val > 0)
				{
					backtrack[posIdx] = i;
					backtrack[lenIdx] =  val;
					backtrack[whereIdx] = 2;
					i = i - val;
					b = b + 1;
				}
				if (val == 0)
				{
					i --;
					j --;
				}
			}

			for(int g = 0; g < GAPS; g++)
			{
				int sIdx = g*3;
				int gIdx = jobID * 3 * GAPS;
				Bt[gIdx] = backtrack[sIdx];
				Bt[gIdx+1] = backtrack[sIdx+1];
				Bt[gIdx+2] = backtrack[sIdx+2];
			}
		}

}

//Kernel for GPU-S-H
__global__ void SingleTextSendback(
					int* T,
					int* X,
					float* firstG,
					float* secondG,
					int* H
				)
{
	float* currG, *prevG;
	float* currentRow, *previousRow, *prevGprevRow, *prevGcurrRow;
	int textChar;
	int blockSize = blockDim.x;
	int jobID = blockIdx.x;
	int localThreadID = threadIdx.x;
	int workID = localThreadID + 1;

	bool LAST = false;
	extern __shared__ int shared[];
	int *patt = &shared[0];
	int *maxILoc = &patt[X_ALIGN];
	float *maxIVal = (float*)&maxILoc[W_ALIGN];
	int *maxJLoc = (int*)&maxIVal[W_ALIGN];
	float *maxJVal = (float*)&maxJLoc[W_ALIGN];
	float *A = (float*)&maxJVal[W_ALIGN];
	float *B = &A[W_ALIGN];
	float *C = &B[W_ALIGN];
	float *D = &C[W_ALIGN];
	int *hRow = (int*)&D[W_ALIGN];


	currentRow = A; previousRow = B; prevGcurrRow = C; prevGprevRow = D;
	int numTiles = ceil((float)WIDTH/(float)blockSize) +1;
	int numVecs = ceil((float)X_ALIGN/4.0f);
	int numVecTiles = ceil((float)numVecs/(float)blockSize);
	int gIdx = 0;
	int sIdx = 0;

	for(int t = 0; t < numVecTiles; t ++)
	{
		sIdx = (t * blockSize) + localThreadID;
		if(sIdx < numVecs){
			gIdx = jobID * numVecs + localThreadID;
			reinterpret_cast<int4*>(patt)[sIdx] = reinterpret_cast<int4*>(X)[gIdx];
		}
	}
	__syncthreads();

	prevG = firstG;
	currG = secondG;
	int j =  workID;

	if (localThreadID ==0)
	{
		previousRow[0] = 0.0f;
	}

	for(int t = 0; t < numTiles; t ++)
	{
		j = (t * blockSize) + workID;
		if (j < WIDTH)
		{
			previousRow[j] = 0.0f-(((float)T_LEN*(float)X_LEN)*EXT + OPEN);
		}
	}
	__syncthreads();
	numVecs = ceil((float)W_ALIGN/4.0f);
	int numMatVecs = HEIGHT * numVecs;
	numVecTiles = ceil((float)numVecs/(float)blockSize);
	gIdx = 0;
	sIdx = 0;

	for(int t = 0; t < numVecTiles; t ++)
	{
		sIdx = (t * blockSize) + localThreadID;
		if(sIdx < numVecs){
			gIdx = jobID * numMatVecs + sIdx;
			reinterpret_cast<float4*>(prevG)[gIdx] = reinterpret_cast<float4*>(previousRow)[sIdx];
		}
	}
	for(int i = 1; i < HEIGHT ; i++)
	{
		textChar = T[i-1];
		if (localThreadID == 0)
		{
			currentRow[0] = 0.0f- ((((float)T_LEN * (float)X_LEN) * EXT) + OPEN);
		}
		for(int t = 0; t < numTiles; t ++)
		{
			j = (t * blockSize) + workID;

			if (j < WIDTH)
			{
				currentRow[j] = previousRow[j-1] + (float)S[getIndex(textChar,patt[j-1],A_LEN)];
				if(i!=j)
				{
					currentRow[j] = 0.0f - ((((float)T_LEN * (float)X_LEN) * EXT) + OPEN);
				}
			}
		}
		for(int t = 0; t < numVecTiles; t ++)
		{
			sIdx = (t * blockSize) + localThreadID;
			if(sIdx < numVecs){
				gIdx = jobID * numMatVecs + (i * numVecs) + sIdx;
				reinterpret_cast<float4*>(prevG)[gIdx] = reinterpret_cast<float4*>(currentRow)[sIdx];
			}
		}

		float* temp = previousRow;
		previousRow = currentRow;
		currentRow = temp;
	}
	__syncthreads();
	for(int k = 1; k <= GAPS; k++)
	{
		if (k == GAPS) LAST = true;
		int j =  workID;
		if (localThreadID ==0)
		{
			previousRow[0] = 0.0f;
		}
		for(int t = 0; t < numTiles; t ++)
		{
			j = (t * blockSize) + workID;
			if (j < WIDTH)
			{
				previousRow[j] = getGapScore(0,j,OPEN,EXT);
			}
		}
		__syncthreads();
		for(int t = 0; t < numVecTiles; t ++)
		{
			sIdx = (t * blockSize) + localThreadID;
			if(sIdx < numVecs){
				gIdx = jobID * numMatVecs + (0 * numVecs) + sIdx;
				reinterpret_cast<float4*>(currG)[gIdx] = reinterpret_cast<float4*>(previousRow)[sIdx];
			}
		}
		for(int t = 0; t < numVecTiles; t ++)
		{
			sIdx = (t * blockSize) + localThreadID;
			if(sIdx < numVecs){
				gIdx = jobID * numMatVecs + (0 * numVecs) + sIdx;
				reinterpret_cast<float4*>(prevGprevRow)[sIdx] = reinterpret_cast<float4*>(prevG)[gIdx];
			}
		}
		__syncthreads();

		for(int t = 0; t < numTiles; t ++)
		{
			int sIdx = (t*blockSize) + localThreadID;
			if(sIdx < WIDTH)
			{
				maxILoc[sIdx] = 0;
				maxIVal[sIdx] = prevGprevRow[sIdx];
			}
		}


		for(int i = 1; i < HEIGHT; i ++)
		{
			for(int t = 0; t < numVecTiles; t ++)
			{
				sIdx = (t * blockSize) + localThreadID;
				if(sIdx < numVecs){
					gIdx = jobID * numMatVecs + (i * numVecs) + sIdx;
					reinterpret_cast<float4*>(prevGcurrRow)[sIdx] = reinterpret_cast<float4*>(prevG)[gIdx];
				}
			}
			__syncthreads();

			for(int t = 0; t < numTiles; t++)
			{
				int sIdx = (t*blockSize) + localThreadID;
				if(sIdx < WIDTH)
				{
					maxJLoc[sIdx] = sIdx;
					maxJVal[sIdx] = prevGcurrRow[sIdx];
				}
			}

			textChar = T[i-1];
			__syncthreads();

			int d = 1;

			while(d <= WIDTH)
			{

				for(int t = 0; t < numTiles; t++)
				{
					int sIdx = (t*blockSize) + localThreadID;
					if((sIdx < WIDTH) && (sIdx >= d))
					{


						int loc1 = maxJLoc[sIdx-d];
						int loc2 = maxJLoc[sIdx];

						float val1 = maxJVal[sIdx-d];
						float val2 = maxJVal[sIdx];

						float pen1 = getGapScore(loc1,sIdx,EXT,OPEN);
						float pen2 = getGapScore(loc2,sIdx,EXT,OPEN);

						float val = val2;
						int loc = loc2;
						if(val1 + pen1 >= val2 + pen2)
						{
							val = val1;
							loc = loc1;
						}

						maxJVal[sIdx] = val;
						maxJLoc[sIdx] = loc;

					}
				}
				d = d*2;
				__syncthreads();
			}
			if(localThreadID == 0)
			{
				currentRow[0] = getGapScore(0,i,EXT,OPEN);
			}


			for(int t = 0; t < numTiles; t++)
			{
				j = (t * blockSize + workID);
				if(j < WIDTH)
				{

					if((prevGprevRow[j] + getGapScore(i-1,i,EXT,OPEN)) > (maxIVal[j] + getGapScore(maxILoc[j],i,EXT,OPEN)))
					{
						maxIVal[j] = prevGprevRow[j];
						maxILoc[j] = i-1;
					}
				}
			}


			for(int t = 0; t < numTiles; t++)
			{
				j = (t * blockSize + workID);
				if(j < WIDTH)
				{
					float carryOnDiag,gapInText,gapInPatt;

					carryOnDiag = previousRow[j-1] + S[getIndex(textChar,patt[j-1],A_LEN)];
					gapInPatt = maxIVal[j] + getGapScore(maxILoc[j],i,EXT,OPEN);
					gapInText = maxJVal[j-1] + getGapScore(maxJLoc[j-1],j,EXT,OPEN);


					float theCalc = max(carryOnDiag,max(gapInText,gapInPatt));

					if(LAST)
					{
						int h = 0;
						if(gapInText == theCalc)
						{
							h = 0 - (j-maxJLoc[j-1]);
						}
						if(gapInPatt == theCalc)
						{
							h = i-maxILoc[j];
						}
						hRow[j] = h;
					}
					currentRow[j] = theCalc;

				}
			}
			for(int t = 0; t < numVecTiles; t ++)
			{
				sIdx = (t * blockSize) + localThreadID;
				if(sIdx < numVecs){
					gIdx = jobID * numMatVecs + (i * numVecs) + sIdx;
					reinterpret_cast<float4*>(currG)[gIdx] = reinterpret_cast<float4*>(currentRow)[sIdx];
					if(LAST)reinterpret_cast<int4*>(H)[gIdx] = reinterpret_cast<int4*>(hRow)[sIdx];
				}
			}
			__syncthreads();

			float* temp = prevGprevRow;
			prevGprevRow = prevGcurrRow;
			prevGcurrRow = temp;
			temp = previousRow;
			previousRow = currentRow;
			currentRow = temp;

		}

		float* temp = prevG;
		prevG = currG;
		currG = temp;
	}
}

//Kernel for GPU-M-A
__global__ void MultiText(
					int* T, //the many Texts (Global)
					int* X, //The many Patterns (Global)
					float* firstG,
					float* secondG, //Current and Previous G Matrices (Global)
					int startJob //The number of the start job
				)
{
		float* currG, *prevG;
		float* currentRow, *previousRow, *prevGprevRow, *prevGcurrRow;
		int textChar;
		int blockSize = blockDim.x;
		int jobID = blockIdx.x;
		int localThreadID = threadIdx.x;
		int workID = localThreadID + 1;
		int textID = ((startJob + jobID) / R);
		int pattID = (startJob + jobID) % R;

		extern __shared__ int shared[];
		int *patt = &shared[0];
		int *maxILoc = &patt[X_ALIGN];
		float *maxIVal = (float*)&maxILoc[W_ALIGN];
		int *maxJLoc = (int*)&maxIVal[W_ALIGN];
		float *maxJVal = (float*)&maxJLoc[W_ALIGN];
		float *A = (float*)&maxJVal[W_ALIGN];
		float *B = &A[W_ALIGN];
		float *C = &B[W_ALIGN];
		float *D = &C[W_ALIGN];

		currentRow = A; previousRow = B; prevGcurrRow = C; prevGprevRow = D;
		int numTiles = ceil((float)WIDTH/(float)blockSize) +1;

			int numVecs = ceil((float)X_ALIGN/4.0f);
			int numVecTiles = ceil((float)numVecs/(float)blockSize);
			int gIdx = 0;
			int sIdx = 0;

			for(int t = 0; t < numVecTiles; t ++)
			{
				sIdx = (t * blockSize) + localThreadID;
				if(sIdx < numVecs){
					gIdx = pattID * numVecs + localThreadID;
					reinterpret_cast<int4*>(patt)[sIdx] = reinterpret_cast<int4*>(X)[gIdx];
				}
			}

		__syncthreads();

		prevG = firstG;
		currG = secondG;
		int j =  workID;

		if (localThreadID ==0)
		{
			previousRow[0] = 0.0f;
		}
		for(int t = 0; t < numTiles; t ++)
		{
			j = (t * blockSize) + workID;
			if (j < WIDTH)
			{
				previousRow[j] = 0.0f-(((float)T_LEN*(float)X_LEN)*EXT + OPEN);
			}
		}
		__syncthreads();


		numVecs = ceil((float)W_ALIGN/4.0f);
		int numMatVecs = HEIGHT * numVecs;
		numVecTiles = ceil((float)numVecs/(float)blockSize);
		gIdx = 0;
		sIdx = 0;

		for(int t = 0; t < numVecTiles; t ++)
		{
			sIdx = (t * blockSize) + localThreadID;
			if(sIdx < numVecs){
				gIdx = jobID * numMatVecs + sIdx;
				reinterpret_cast<float4*>(prevG)[gIdx] = reinterpret_cast<float4*>(previousRow)[sIdx];
			}
		}

		for(int i = 1; i < HEIGHT ; i++)
		{
			textChar = T[textID * T_OFFSET + i-1];
			if (localThreadID == 0)
			{
				currentRow[0] = 0.0f- ((((float)T_LEN * (float)X_LEN) * EXT) + OPEN);
			}

			for(int t = 0; t < numTiles; t ++)
			{
				j = (t * blockSize) + workID;

				if (j < WIDTH)
				{
					currentRow[j] = previousRow[j-1] + (float)S[getIndex(textChar,patt[j-1],A_LEN)];
					if(i!=j)
					{
						currentRow[j] = 0.0f - ((((float)T_LEN * (float)X_LEN) * EXT) + OPEN);
					}
				}
			}
			for(int t = 0; t < numVecTiles; t ++)
			{
				sIdx = (t * blockSize) + localThreadID;
				if(sIdx < numVecs){
					gIdx = jobID * numMatVecs + (i * numVecs) + sIdx;
					reinterpret_cast<float4*>(prevG)[gIdx] = reinterpret_cast<float4*>(currentRow)[sIdx];
				}
			}

			float* temp = previousRow;
			previousRow = currentRow;
			currentRow = temp;
		}
		__syncthreads();
		for(int k = 1; k <= GAPS; k++)
		{
			int j =  workID;

			if (localThreadID ==0)
			{
				previousRow[0] = 0.0f;
			}

			for(int t = 0; t < numTiles; t ++)
			{
				j = (t * blockSize) + workID;
				if (j < WIDTH)
				{
					previousRow[j] = getGapScore(0,j,OPEN,EXT);
				}
			}
			__syncthreads();

			for(int t = 0; t < numVecTiles; t ++)
			{
				sIdx = (t * blockSize) + localThreadID;
				if(sIdx < numVecs){
					gIdx = jobID * numMatVecs + (0 * numVecs) + sIdx;
					reinterpret_cast<float4*>(currG)[gIdx] = reinterpret_cast<float4*>(previousRow)[sIdx];
				}
			}

			for(int t = 0; t < numVecTiles; t ++)
			{
				sIdx = (t * blockSize) + localThreadID;
				if(sIdx < numVecs){
					gIdx = jobID * numMatVecs + (0 * numVecs) + sIdx;
					reinterpret_cast<float4*>(prevGprevRow)[sIdx] = reinterpret_cast<float4*>(prevG)[gIdx];
				}
			}
			__syncthreads();

			for(int t = 0; t < numTiles; t ++)
			{
				int sIdx = (t*blockSize) + localThreadID;
				if(sIdx < WIDTH)
				{
					maxILoc[sIdx] = 0;
					maxIVal[sIdx] = prevGprevRow[sIdx];
				}
			}


			for(int i = 1; i < HEIGHT; i ++)
			{

				for(int t = 0; t < numVecTiles; t ++)
				{
					sIdx = (t * blockSize) + localThreadID;
					if(sIdx < numVecs){
						gIdx = jobID * numMatVecs + (i * numVecs) + sIdx;
						reinterpret_cast<float4*>(prevGcurrRow)[sIdx] = reinterpret_cast<float4*>(prevG)[gIdx];
					}
				}
				__syncthreads();

				for(int t = 0; t < numTiles; t++)
				{
					int sIdx = (t*blockSize) + localThreadID;
					if(sIdx < WIDTH)
					{
						maxJLoc[sIdx] = sIdx;
						maxJVal[sIdx] = prevGcurrRow[sIdx];
					}
				}

				textChar = T[textID * T_OFFSET + i-1];
				__syncthreads();

				int d = 1;

				while(d <= WIDTH)
				{

					for(int t = 0; t < numTiles; t++)
					{
						int sIdx = (t*blockSize) + localThreadID;
						if((sIdx < WIDTH) && (sIdx >= d))
						{


							int loc1 = maxJLoc[sIdx-d];
							int loc2 = maxJLoc[sIdx];

							float val1 = maxJVal[sIdx-d];
							float val2 = maxJVal[sIdx];

							float pen1 = getGapScore(loc1,sIdx,EXT,OPEN);
							float pen2 = getGapScore(loc2,sIdx,EXT,OPEN);

							float val = val2;
							int loc = loc2;
							if(val1 + pen1 >= val2 + pen2)
							{
								val = val1;
								loc = loc1;
							}

							maxJVal[sIdx] = val;
							maxJLoc[sIdx] = loc;

						}
					}
					d = d*2;
					__syncthreads();
				}

				if(localThreadID == 0)
				{
					currentRow[0] = getGapScore(0,i,EXT,OPEN);
				}


				for(int t = 0; t < numTiles; t++)
				{
					j = (t * blockSize + workID);
					if(j < WIDTH)
					{

						if((prevGprevRow[j] + getGapScore(i-1,i,EXT,OPEN)) > (maxIVal[j] + getGapScore(maxILoc[j],i,EXT,OPEN)))
						{
							maxIVal[j] = prevGprevRow[j];
							maxILoc[j] = i-1;
						}
					}
				}


				for(int t = 0; t < numTiles; t++)
				{
					j = (t * blockSize + workID);
					if(j < WIDTH)
					{
						float carryOnDiag,gapInText,gapInPatt;

						carryOnDiag = previousRow[j-1] + S[getIndex(textChar,patt[j-1],A_LEN)];
						gapInText = maxIVal[j] + getGapScore(maxILoc[j],i,EXT,OPEN);
						gapInPatt = maxJVal[j-1] + getGapScore(maxJLoc[j-1],j,EXT,OPEN);


						currentRow[j] = max(carryOnDiag,max(gapInText,gapInPatt));
					}
				}
				for(int t = 0; t < numVecTiles; t ++)
				{
					sIdx = (t * blockSize) + localThreadID;
					if(sIdx < numVecs){
						gIdx = jobID * numMatVecs + (i * numVecs) + sIdx;
						reinterpret_cast<float4*>(currG)[gIdx] = reinterpret_cast<float4*>(currentRow)[sIdx];
					}
				}
				__syncthreads();

				float* temp = prevGprevRow;
				prevGprevRow = prevGcurrRow;
				prevGcurrRow = temp;
				temp = previousRow;
				previousRow = currentRow;
				currentRow = temp;

			}
			float* temp = prevG;
			prevG = currG;
			currG = temp;
		}
	}

//Kernel for GPU-M-B
__global__ void MultiTextBacktrack(int* T, int* X, float* firstG, float* secondG, int* H, int* Bt, int startJob)
{
			float* currG, *prevG;
			float* currentRow, *previousRow, *prevGprevRow, *prevGcurrRow;
			int textChar;
			int blockSize = blockDim.x;
			int jobID = blockIdx.x;
			int localThreadID = threadIdx.x;
			int workID = localThreadID + 1;
			int textID = ((startJob + jobID) / R);
			int pattID = (startJob + jobID) % R;
			bool LAST = false;
			extern __shared__ int shared[];
			int *patt = &shared[0];
			int *maxILoc = &patt[X_ALIGN];
			float *maxIVal = (float*)&maxILoc[W_ALIGN];
			int *maxJLoc = (int*)&maxIVal[W_ALIGN];
			float *maxJVal = (float*)&maxJLoc[W_ALIGN];
			float *A = (float*)&maxJVal[W_ALIGN];
			float *B = &A[W_ALIGN];
			float *C = &B[W_ALIGN];
			float *D = &C[W_ALIGN];
			int *hRow = (int*)&D[W_ALIGN];
			int *backtrack = &hRow[W_ALIGN];
			currentRow = A; previousRow = B; prevGcurrRow = C; prevGprevRow = D;
			int numTiles = ceil((float)WIDTH/(float)blockSize) +1;
			int numVecs = ceil((float)X_ALIGN/4.0f);
			int numVecTiles = ceil((float)numVecs/(float)blockSize);
			int gIdx = 0;
			int sIdx = 0;
			for(int t = 0; t < numVecTiles; t ++)
			{
				sIdx = (t * blockSize) + localThreadID;
				if(sIdx < numVecs){
					gIdx = pattID * numVecs + localThreadID;
					reinterpret_cast<int4*>(patt)[sIdx] = reinterpret_cast<int4*>(X)[gIdx];
				}
			}
			__syncthreads();
			prevG = firstG;
			currG = secondG;
			int j =  workID;
			if (localThreadID ==0)
			{
				previousRow[0] = 0.0f;
			}
			for(int t = 0; t < numTiles; t ++)
			{
				j = (t * blockSize) + workID;
				if (j < WIDTH)
				{
					previousRow[j] = 0.0f-(((float)T_LEN*(float)X_LEN)*EXT + OPEN);
				}
			}
			__syncthreads();
			numVecs = ceil((float)W_ALIGN/4.0f);
			int numMatVecs = HEIGHT * numVecs;
			numVecTiles = ceil((float)numVecs/(float)blockSize);
			gIdx = 0;
			sIdx = 0;
			for(int t = 0; t < numVecTiles; t ++)
			{
				sIdx = (t * blockSize) + localThreadID;
				if(sIdx < numVecs){
					gIdx = jobID * numMatVecs + sIdx;
					reinterpret_cast<float4*>(prevG)[gIdx] = reinterpret_cast<float4*>(previousRow)[sIdx];
				}
			}
			for(int i = 1; i < HEIGHT ; i++)
			{
				textChar = T[textID * T_OFFSET + i-1];
				if (localThreadID == 0)
				{
					currentRow[0] = 0.0f- ((((float)T_LEN * (float)X_LEN) * EXT) + OPEN);
				}
				for(int t = 0; t < numTiles; t ++)
				{
					j = (t * blockSize) + workID;

					if (j < WIDTH)
					{
						currentRow[j] = previousRow[j-1] + (float)S[getIndex(textChar,patt[j-1],A_LEN)];
						if(i!=j)
						{
							currentRow[j] = 0.0f - ((((float)T_LEN * (float)X_LEN) * EXT) + OPEN);
						}
					}
				}
				for(int t = 0; t < numVecTiles; t ++)
				{
					sIdx = (t * blockSize) + localThreadID;
					if(sIdx < numVecs){
						gIdx = jobID * numMatVecs + (i * numVecs) + sIdx;
						reinterpret_cast<float4*>(prevG)[gIdx] = reinterpret_cast<float4*>(currentRow)[sIdx];
					}
				}
				float* temp = previousRow;
				previousRow = currentRow;
				currentRow = temp;
			}
			__syncthreads();
			for(int k = 1; k <= GAPS; k++)
			{
				if (k == GAPS) LAST = true;
				int j =  workID;
				if (localThreadID ==0)
				{
					previousRow[0] = 0.0f;
					if(LAST) hRow[0] = 0;
				}
				for(int t = 0; t < numTiles; t ++)
				{
					j = (t * blockSize) + workID;
					if (j < WIDTH)
					{
						previousRow[j] = getGapScore(0,j,OPEN,EXT);
						if(LAST) hRow[j] = -j;
					}
				}
				__syncthreads();
				for(int t = 0; t < numVecTiles; t ++)
				{
					sIdx = (t * blockSize) + localThreadID;
					if(sIdx < numVecs){
						gIdx = jobID * numMatVecs + (0 * numVecs) + sIdx;
						reinterpret_cast<float4*>(currG)[gIdx] = reinterpret_cast<float4*>(previousRow)[sIdx];
						if(LAST)reinterpret_cast<int4*>(H)[gIdx] = reinterpret_cast<int4*>(hRow)[sIdx];
					}
				}
				for(int t = 0; t < numVecTiles; t ++)
				{
					sIdx = (t * blockSize) + localThreadID;
					if(sIdx < numVecs){
						gIdx = jobID * numMatVecs + (0 * numVecs) + sIdx;
						reinterpret_cast<float4*>(prevGprevRow)[sIdx] = reinterpret_cast<float4*>(prevG)[gIdx];
					}
				}
				__syncthreads();
				for(int t = 0; t < numTiles; t ++)
				{
					int sIdx = (t*blockSize) + localThreadID;
					if(sIdx < WIDTH)
					{
						maxILoc[sIdx] = 0;
						maxIVal[sIdx] = prevGprevRow[sIdx];
					}
				}


				for(int i = 1; i < HEIGHT; i ++)
				{

					for(int t = 0; t < numVecTiles; t ++)
					{
						sIdx = (t * blockSize) + localThreadID;
						if(sIdx < numVecs){
							gIdx = jobID * numMatVecs + (i * numVecs) + sIdx;
							reinterpret_cast<float4*>(prevGcurrRow)[sIdx] = reinterpret_cast<float4*>(prevG)[gIdx];
						}
					}
					__syncthreads();
					for(int t = 0; t < numTiles; t++)
					{
						int sIdx = (t*blockSize) + localThreadID;
						if(sIdx < WIDTH)
						{
							maxJLoc[sIdx] = sIdx;
							maxJVal[sIdx] = prevGcurrRow[sIdx];
						}
					}
					textChar = T[i-1];
					__syncthreads();
					int d = 1;
					while(d <= WIDTH)
					{

						for(int t = 0; t < numTiles; t++)
						{
							int sIdx = (t*blockSize) + localThreadID;
							if((sIdx < WIDTH) && (sIdx >= d))
							{


								int loc1 = maxJLoc[sIdx-d];
								int loc2 = maxJLoc[sIdx];

								float val1 = maxJVal[sIdx-d];
								float val2 = maxJVal[sIdx];

								float pen1 = getGapScore(loc1,sIdx,EXT,OPEN);
								float pen2 = getGapScore(loc2,sIdx,EXT,OPEN);

								float val = val2;
								int loc = loc2;
								if(val1 + pen1 >= val2 + pen2)
								{
									val = val1;
									loc = loc1;
								}

								maxJVal[sIdx] = val;
								maxJLoc[sIdx] = loc;

							}
						}
						d = d*2;
						__syncthreads();
					}
					if(localThreadID == 0)
					{
						currentRow[0] = getGapScore(0,i,EXT,OPEN);
						if(LAST) hRow[0] = i;
					}


					for(int t = 0; t < numTiles; t++)
					{
						j = (t * blockSize + workID);
						if(j < WIDTH)
						{
							if((prevGprevRow[j] + getGapScore(i-1,i,EXT,OPEN)) > (maxIVal[j] + getGapScore(maxILoc[j],i,EXT,OPEN)))
							{
								maxIVal[j] = prevGprevRow[j];
								maxILoc[j] = i-1;
							}
						}
					}
					for(int t = 0; t < numTiles; t++)
					{
						j = (t * blockSize + workID);
						if(j < WIDTH)
						{
							float carryOnDiag,gapInText,gapInPatt;

							carryOnDiag = previousRow[j-1] + S[getIndex(textChar,patt[j-1],A_LEN)];
							gapInPatt = maxIVal[j] + getGapScore(maxILoc[j],i,EXT,OPEN);
							gapInText = maxJVal[j-1] + getGapScore(maxJLoc[j-1],j,EXT,OPEN);


							float theCalc = max(carryOnDiag,max(gapInText,gapInPatt));

							if(LAST)
							{
								int h = 0;
								if(gapInText == theCalc)
								{
									h = 0 - (j-maxJLoc[j-1]);
								}
								if(gapInPatt == theCalc)
								{
									h = i-maxILoc[j];
								}
								hRow[j] = h;
							}
							currentRow[j] = theCalc;

						}
					}
					for(int t = 0; t < numVecTiles; t ++)
					{
						sIdx = (t * blockSize) + localThreadID;
						if(sIdx < numVecs){
							gIdx = jobID * numMatVecs + (i * numVecs) + sIdx;
							reinterpret_cast<float4*>(currG)[gIdx] = reinterpret_cast<float4*>(currentRow)[sIdx];
							if(LAST)reinterpret_cast<int4*>(H)[gIdx] = reinterpret_cast<int4*>(hRow)[sIdx];
						}
					}
					__syncthreads();

					float* temp = prevGprevRow;
					prevGprevRow = prevGcurrRow;
					prevGcurrRow = temp;

					temp = previousRow;
					previousRow = currentRow;
					currentRow = temp;

				}

				//Swap currG and prevG
				float* temp = prevG;
				prevG = currG;
				currG = temp;
			}
			__syncthreads();

			if (localThreadID == 0)
			{

				int i = HEIGHT-1;
				int j = WIDTH-1;
				int b = 0;


					while ((i >= 0) && (j >= 0) && (b <= GAPS))
					{
						int val = H[getIndex(i,j,W_ALIGN,jobID)];
						int posIdx = b*3;
						int lenIdx = b*3 + 1;
						int whereIdx = b*3 + 2;


						if (val < 0)
						{
							backtrack[posIdx] = j;
							backtrack[lenIdx] = 0 - val;
							backtrack[whereIdx] = 1;
							j = j + val;
							b = b + 1;
						}
						if (val > 0)
						{
							backtrack[posIdx] = i;
							backtrack[lenIdx] =  val;
							backtrack[whereIdx] = 2;
							i = i - val;
							b = b + 1;
						}
						if (val == 0)
						{
							i --;
							j --;
						}
					}

					for(int g = 0; g < GAPS; g++)
					{
						int sIdx = g*3;
						int gIdx = jobID * 3 * GAPS;
						Bt[gIdx] = backtrack[sIdx];
						Bt[gIdx+1] = backtrack[sIdx+1];
						Bt[gIdx+2] = backtrack[sIdx+2];
					}
				}


}

//Kernel for GPU-M-H
__global__ void MultiTextSendBack(int* T, int* X, float* firstG, float* secondG, int* H, int startJob)
{
				float* currG, *prevG;
				float* currentRow, *previousRow, *prevGprevRow, *prevGcurrRow;
				int textChar;
				int blockSize = blockDim.x;
				int jobID = blockIdx.x;
				int localThreadID = threadIdx.x;
				int workID = localThreadID + 1;
				int textID = ((startJob + jobID) / R);
				int pattID = (startJob + jobID) % R;
				bool LAST = false;
				extern __shared__ int shared[];
				int *patt = &shared[0];
				int *maxILoc = &patt[X_ALIGN];
				float *maxIVal = (float*)&maxILoc[W_ALIGN];
				int *maxJLoc = (int*)&maxIVal[W_ALIGN];
				float *maxJVal = (float*)&maxJLoc[W_ALIGN];
				float *A = (float*)&maxJVal[W_ALIGN];
				float *B = &A[W_ALIGN];
				float *C = &B[W_ALIGN];
				float *D = &C[W_ALIGN];
				int *hRow = (int*)&D[W_ALIGN];
				currentRow = A; previousRow = B; prevGcurrRow = C; prevGprevRow = D;
				int numTiles = ceil((float)WIDTH/(float)blockSize) +1;
				int numVecs = ceil((float)X_ALIGN/4.0f);
				int numVecTiles = ceil((float)numVecs/(float)blockSize);
				int gIdx = 0;
				int sIdx = 0;
				for(int t = 0; t < numVecTiles; t ++)
				{
					sIdx = (t * blockSize) + localThreadID;
					if(sIdx < numVecs){
						gIdx = pattID * numVecs + localThreadID;
						reinterpret_cast<int4*>(patt)[sIdx] = reinterpret_cast<int4*>(X)[gIdx];
					}
				}
				__syncthreads();

				prevG = firstG;
				currG = secondG;
				int j =  workID;
				if (localThreadID ==0)
				{
					previousRow[0] = 0.0f;
				}

				for(int t = 0; t < numTiles; t ++)
				{
					j = (t * blockSize) + workID;
					if (j < WIDTH)
					{
						previousRow[j] = 0.0f-(((float)T_LEN*(float)X_LEN)*EXT + OPEN);
					}
				}
				__syncthreads();

				numVecs = ceil((float)W_ALIGN/4.0f);
				int numMatVecs = HEIGHT * numVecs;
				numVecTiles = ceil((float)numVecs/(float)blockSize);
				gIdx = 0;
				sIdx = 0;
				for(int t = 0; t < numVecTiles; t ++)
				{
					sIdx = (t * blockSize) + localThreadID;
					if(sIdx < numVecs){
						gIdx = jobID * numMatVecs + sIdx;
						reinterpret_cast<float4*>(prevG)[gIdx] = reinterpret_cast<float4*>(previousRow)[sIdx];
					}
				}
				for(int i = 1; i < HEIGHT ; i++)
				{
					textChar = T[textID * T_OFFSET + i-1];
					if (localThreadID == 0)
					{
						currentRow[0] = 0.0f- ((((float)T_LEN * (float)X_LEN) * EXT) + OPEN);
					}
					for(int t = 0; t < numTiles; t ++)
					{
						j = (t * blockSize) + workID;

						if (j < WIDTH)
						{
							currentRow[j] = previousRow[j-1] + (float)S[getIndex(textChar,patt[j-1],A_LEN)];
							if(i!=j)
							{
								currentRow[j] = 0.0f - ((((float)T_LEN * (float)X_LEN) * EXT) + OPEN);
							}
						}
					}
					for(int t = 0; t < numVecTiles; t ++)
					{
						sIdx = (t * blockSize) + localThreadID;
						if(sIdx < numVecs){
							gIdx = jobID * numMatVecs + (i * numVecs) + sIdx;
							reinterpret_cast<float4*>(prevG)[gIdx] = reinterpret_cast<float4*>(currentRow)[sIdx];
						}
					}
					float* temp = previousRow;
					previousRow = currentRow;
					currentRow = temp;
				}
				__syncthreads();

				for(int k = 1; k <= GAPS; k++)
				{
					if (k == GAPS) LAST = true;
					int j =  workID;

					if (localThreadID ==0)
					{
						previousRow[0] = 0.0f;
						if(LAST) hRow[0] = 0;
					}
					for(int t = 0; t < numTiles; t ++)
					{
						j = (t * blockSize) + workID;
						if (j < WIDTH)
						{
							previousRow[j] = getGapScore(0,j,OPEN,EXT);
							if(LAST) hRow[j] = -j;
						}
					}
					__syncthreads();

					for(int t = 0; t < numVecTiles; t ++)
					{
						sIdx = (t * blockSize) + localThreadID;
						if(sIdx < numVecs){
							gIdx = jobID * numMatVecs + (0 * numVecs) + sIdx;
							reinterpret_cast<float4*>(currG)[gIdx] = reinterpret_cast<float4*>(previousRow)[sIdx];
							if(LAST)reinterpret_cast<int4*>(H)[gIdx] = reinterpret_cast<int4*>(hRow)[sIdx];
						}
					}

					for(int t = 0; t < numVecTiles; t ++)
					{
						sIdx = (t * blockSize) + localThreadID;
						if(sIdx < numVecs){
							gIdx = jobID * numMatVecs + (0 * numVecs) + sIdx;
							reinterpret_cast<float4*>(prevGprevRow)[sIdx] = reinterpret_cast<float4*>(prevG)[gIdx];
						}
					}
					__syncthreads();
					for(int t = 0; t < numTiles; t ++)
					{
						int sIdx = (t*blockSize) + localThreadID;
						if(sIdx < WIDTH)
						{
							maxILoc[sIdx] = 0;
							maxIVal[sIdx] = prevGprevRow[sIdx];
						}
					}
					for(int i = 1; i < HEIGHT; i ++)
					{
						for(int t = 0; t < numVecTiles; t ++)
						{
							sIdx = (t * blockSize) + localThreadID;
							if(sIdx < numVecs){
								gIdx = jobID * numMatVecs + (i * numVecs) + sIdx;
								reinterpret_cast<float4*>(prevGcurrRow)[sIdx] = reinterpret_cast<float4*>(prevG)[gIdx];
							}
						}
						__syncthreads();

						for(int t = 0; t < numTiles; t++)
						{
							int sIdx = (t*blockSize) + localThreadID;
							if(sIdx < WIDTH)
							{
								maxJLoc[sIdx] = sIdx;
								maxJVal[sIdx] = prevGcurrRow[sIdx];
							}
						}

						textChar = T[i-1];
						__syncthreads();

						int d = 1;

						while(d <= WIDTH)
						{

							for(int t = 0; t < numTiles; t++)
							{
								int sIdx = (t*blockSize) + localThreadID;
								if((sIdx < WIDTH) && (sIdx >= d))
								{


									int loc1 = maxJLoc[sIdx-d];
									int loc2 = maxJLoc[sIdx];

									float val1 = maxJVal[sIdx-d];
									float val2 = maxJVal[sIdx];

									float pen1 = getGapScore(loc1,sIdx,EXT,OPEN);
									float pen2 = getGapScore(loc2,sIdx,EXT,OPEN);

									float val = val2;
									int loc = loc2;
									if(val1 + pen1 >= val2 + pen2)
									{
										val = val1;
										loc = loc1;
									}

									maxJVal[sIdx] = val;
									maxJLoc[sIdx] = loc;

								}
							}
							d = d*2;
							__syncthreads();
						}
						if(localThreadID == 0)
						{
							currentRow[0] = getGapScore(0,i,EXT,OPEN);
							if(LAST) hRow[0] = i;
						}


						for(int t = 0; t < numTiles; t++)
						{
							j = (t * blockSize + workID);
							if(j < WIDTH)
							{
								if((prevGprevRow[j] + getGapScore(i-1,i,EXT,OPEN)) > (maxIVal[j] + getGapScore(maxILoc[j],i,EXT,OPEN)))
								{
									maxIVal[j] = prevGprevRow[j];
									maxILoc[j] = i-1;
								}
							}
						}


						for(int t = 0; t < numTiles; t++)
						{
							j = (t * blockSize + workID);
							if(j < WIDTH)
							{

								float carryOnDiag,gapInText,gapInPatt;

								carryOnDiag = previousRow[j-1] + S[getIndex(textChar,patt[j-1],A_LEN)];
								gapInPatt = maxIVal[j] + getGapScore(maxILoc[j],i,EXT,OPEN);
								gapInText = maxJVal[j-1] + getGapScore(maxJLoc[j-1],j,EXT,OPEN);


								float theCalc = max(carryOnDiag,max(gapInText,gapInPatt));

								if(LAST)
								{
									int h = 0;
									if(gapInText == theCalc)
									{
										h = 0 - (j-maxJLoc[j-1]);
									}
									if(gapInPatt == theCalc)
									{
										h = i-maxILoc[j];
									}
									hRow[j] = h;
								}
								currentRow[j] = theCalc;

							}
						}
						for(int t = 0; t < numVecTiles; t ++)
						{
							sIdx = (t * blockSize) + localThreadID;
							if(sIdx < numVecs){
								gIdx = jobID * numMatVecs + (i * numVecs) + sIdx;
								reinterpret_cast<float4*>(currG)[gIdx] = reinterpret_cast<float4*>(currentRow)[sIdx];
								if(LAST)reinterpret_cast<int4*>(H)[gIdx] = reinterpret_cast<int4*>(hRow)[sIdx];
							}
						}
						__syncthreads();

						float* temp = prevGprevRow;
						prevGprevRow = prevGcurrRow;
						prevGcurrRow = temp;
						temp = previousRow;
						previousRow = currentRow;
						currentRow = temp;

					}

					float* temp = prevG;
					prevG = currG;
					currG = temp;
				}

}
