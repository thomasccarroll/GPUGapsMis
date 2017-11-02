//GPUGapsMis
//Thomas Carroll - 2017

#ifndef KERNELS_H
#define KERNELS_H

#include<stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__device__ int getIndex(int i, int j, int C, int offset);
__device__ int getIndex(int i, int j, int jobID);

//Kernels with Alignment only
__global__ void SingleText(int* T, int* X, float* firstG, float* secondG); //GPU-S-A
__global__ void MultiText(int* T, int* X, float* firstG, float* secondG, int startJob); //GPU-M-A

//Kernels with Backtrack on Device
__global__ void SingleTextBacktrack(int* T, int* X, float* firstG, float* secondG, int* H, int *B); //GPU-S-B
__global__ void MultiTextBacktrack(int* T, int* X, float* firstG, float* secondG, int* H, int* Bt, int startJob); //GPU-M-B

//Kernels with H Sendback, for backtrack on Host
__global__ void SingleTextSendback(int* T, int* X, float* firstG, float* secondG, int* H); //GPU-S-H
__global__ void MultiTextSendBack(int* T, int* X, float* firstG, float* secondG, int* H, int startJob); //GPU-M-H


void copyConstantSymbols(float open, float ext, int gaps, int alen, int tlen, int xlen, int xAlign, int height, int width, int wAlign, int matrixSize, int*score); //Copies the Constant Memory symbols to GPU for Single Text

void copyConstantSymbols(float open, float ext, int gaps, int alen, int tlen, int tAlign, int xlen, int xAlign, int height, int width, int wAlign, int matrixSize, int*score, int q, int r); //Copies the Constant Memory symbols to GPU for Multiple Text


#endif
