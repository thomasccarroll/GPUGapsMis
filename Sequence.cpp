//GPUGapsMis
//Thomas Carroll - 2017

#include "Sequence.h"

Sequence::Sequence()
{}

Sequence::~Sequence()
{
		delete &mapped;
}

void Sequence::print()
{

		printf("Length: %d\n", this ->length);
		printf("Raw Sequence: %s\n", this ->raw.c_str());
		printf("Mapped Sequence: [");
		
		for(int i = 0; i < this ->length; i ++)
		{
				cout << this ->mapped[i] << " ";
		}
		cout << "]\n===\n";
}

