#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define THREAD 128

__global__ void dot(int N,float *x,float*y,float *ans);

int main(void){
    /*for CPU*/
    int i;
    int size = 1024;
    int block = (size + THREAD -1);//number of block

    float *x,*y,*ans;//(x,y)
    float z;
    
    cudaMallocHost((void **)&x,sizeof(float)*size);
    cudaMallocHost((void **)&y,sizeof(float)*size);
    cudaMallocHost((void **)&ans,sizeof(float)*block);
    
    
    /*fo GPU*/
    float *d_x,*d_y,*d_ans;
    cudaMalloc((void **)&d_x,sizeof(float)*size);
    cudaMalloc((void **)&d_y,sizeof(float)*size);
    cudaMalloc((void **)&d_ans,sizeof(float)*block);
    for(i=0;i<size;i++){
        x[i]=1.0;
        y[i]=1.0;
        }    
    
    /*Memory copy Host to Device*/
    
    cudaMemcpy(d_x,x,sizeof(float)*size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,y,sizeof(float)*size,cudaMemcpyHostToDevice);
    
    dot<<<block,THREAD>>>(size,d_x,d_y,d_ans);

    /*Memory copy Device to Host*/

    cudaMemcpy(ans,d_ans,sizeof(float)*block,cudaMemcpyDeviceToHost);
    z = 0.0;
    for(i=0;i<block;i++)
        z+=ans[i];
    //show answer
    printf("%f\n",z);
    
    /*CPU Memory free*/
    cudaFree(x);
    cudaFree(y);
    cudaFree(d_ans);


    /*GPU Memory free*/
   cudaFree(d_x);
   cudaFree(d_y);
   cudaFree(d_ans);
    return 0;  
  }


__global__ void dot(int N,float *x,float *y,float *ans){
    int i,j;
    __shared__ float tmp[THREAD];
    
    tmp[threadIdx.x]=0;
    j = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(j<N){
     tmp[threadIdx.x] += x[j] * y[j];
    }
    else {
     tmp[threadIdx.x] =0.0;    
    }
    for(i = THREAD/2;i>31;i=i/2){
        if(threadIdx.x<i){
            tmp[threadIdx.x] += tmp[threadIdx.x+i];
    __syncthreads();
    }
}
    if(threadIdx.x<16){
        tmp[threadIdx.x] += tmp[threadIdx.x + 16];
         __syncthreads();
        tmp[threadIdx.x] += tmp[threadIdx.x + 8];
         __syncthreads();
        tmp[threadIdx.x] += tmp[threadIdx.x + 4];
         __syncthreads();
        tmp[threadIdx.x] += tmp[threadIdx.x + 2];
         __syncthreads();
        tmp[threadIdx.x] += tmp[threadIdx.x + 1];
         __syncthreads();


        }
    if(threadIdx.x == 0){
        ans[blockIdx.x] = tmp[0];    
        }
    }

