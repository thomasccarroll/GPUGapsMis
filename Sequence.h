//GPUGapsMis
//Thomas Carroll - 2017


#ifndef SEQUENCE_H_
#define SEQUENCE_H_


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <cstring>

using namespace std;
using std::string;
using std::cout;


//Debug is used for printing very verbose output, allowing debugging
#define DEBUG 0

using std::cout;
using std::endl;

class Sequence{
public:

	string raw;
	int *mapped;
	int length;
	
	
	//Methods
	Sequence();
	~Sequence();
	
	void print();
};
#endif
