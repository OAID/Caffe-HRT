#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

#include <cblas.h>

extern "C" {
#include "testbed.h"
}

float * A;
float * B;
float * C;
int M;
int N;
int K;


void init_matrix(float * m, int size)
{
    int i;
    for(i=0;i<size;i+=2)
        m[i]=i;
}

void init_data(void)
{
    int sizeA;
    int sizeB;
    int sizeC;

    sizeA=M*K;
    sizeB=K*N;
    sizeC=M*N;

    A=(float *)malloc(sizeA*sizeof(float));
    B=(float *)malloc(sizeB*sizeof(float));
    C=(float *)malloc(sizeC*sizeof(float));

    init_matrix(A,sizeA);
    init_matrix(B,sizeB);
    init_matrix(C,sizeC);
}



void run_sgemm(void * dummy)
{
   int i;
   for(i=0;i<1;i++)
   {
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,
    1.0,A,K,B,N,0,C,N);
   }
   
}

int main(int argc, char * argv[])
{
  
     M=27;
     K=9;
     N=37632;

  
     init_data();
     init_testbed();

     run_test(16,1,run_sgemm,NULL);

     release_testbed();
     
     return 0;
}
