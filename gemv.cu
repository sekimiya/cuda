#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>

#define THREAD 128

//texture<int2, 1, cudaReadModeElementType> yoshi;
	
__global__ void gemv(int n, double *adim, double *b, double *d_ans);

void cgemv(int n, double *adim, double *b, double *d_ans);

double gettime()
{
struct timeval tv;
gettimeofday(&tv, NULL);
return tv.tv_sec + (double)tv.tv_usec*1.0e-6;
}

int main(int argc, char **argv)
{
/* for CPU */
int i, j;
double *bdim, *c, *ans;
//double start, stop;
//double cpu_time, gpu_time;
int n = 1024;

bdim = (double *)malloc(sizeof(double) *n*n);
c = (double *)malloc(sizeof(double) *n);
ans = (double *)malloc(sizeof(double) *n);

/* for GPU */
double *d_bdim, *d_c, *d_ans;
cudaMalloc((void **)&d_bdim, sizeof(double)*n*n);
cudaMalloc((void **)&d_c, sizeof(double)*n);
cudaMalloc((void **)&d_ans, sizeof(double)*n);

for(i = 0; i < n; i++)
{
c[i] = 1.0;
for(j = 0; j < n; j++)
bdim[i*n+j] = 1.0;
}

/*start = gettime();
cgemv(n, bdim, c, ans);
stop = gettime();

cpu_time=stop - start;
*/
cudaMemcpy(d_bdim, bdim, sizeof(double)*n*n, cudaMemcpyHostToDevice);
cudaMemcpy(d_c, c, sizeof(double)*n, cudaMemcpyHostToDevice);

//cudaBindTexture(0, yoshi, d_c, sizeof(double)*n);

//start = gettime();
gemv<<<n, THREAD>>>(n, d_bdim, d_c, d_ans);
//stop = gettime();

//gpu_time=stop - start;

cudaMemcpy(ans, d_ans, sizeof(double)*n, cudaMemcpyDeviceToHost);

//printf("cpu_time : %.6f[sec]\n",cpu_time);
//printf("gpu_time : %.6f[sec]\n",gpu_time);
//printf("%f x\n", cpu_time / gpu_time);


for(i = 0; i < n; i++)
printf("%f\n", ans[i]);


free(bdim);
free(c);
free(ans);
cudaFree(d_bdim);
cudaFree(d_c);
cudaFree(d_ans);

return 0;
} 

__global__ void gemv(int n, double *adim, double *b, double *d_ans)
{
int i;
int div = n/THREAD;
//int2 fjt;
__shared__ double tmp[THREAD];

tmp[threadIdx.x] = 0.0;

for(i = 0; i < div; i++)
{
  /*fjt = tex1Dfetch(yoshi, i*THREAD+threadIdx.x); */
tmp[threadIdx.x] += adim[blockIdx.x*n+i*THREAD+threadIdx.x] * b[i * THREAD + threadIdx.x];
}
//fjt = tex1Dfetch(yoshi,div*THREAD+threadIdx.x);
if(threadIdx.x < n%THREAD)
tmp[threadIdx.x] += adim[blockIdx.x*n+THREAD*div+threadIdx.x] * b[THREAD * div + threadIdx.x];

__syncthreads();

for(i = THREAD / 2; i > 31; i = i / 2)
{
if(threadIdx.x < i)
tmp[threadIdx.x] += tmp[threadIdx.x + i];
__syncthreads();
}

if(threadIdx.x < 16)
{
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


if(threadIdx.x == 0)
d_ans[blockIdx.x] = tmp[0];

}

void cgemv(int n, double *adim, double *b, double *d_ans)
{
int i, j;

for(i = 0; i < n; i++)
for(j = 0; j < n; j++)
d_ans[i] = adim[i*n+j] * b[i];

}
