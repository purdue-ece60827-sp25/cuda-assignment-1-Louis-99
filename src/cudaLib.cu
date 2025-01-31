
#include "cudaLib.cuh"

#include "cpuLib.h"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < size) y[idx] += scale * x[idx];
}

int runGpuSaxpy(int vectorSize) {

	cudaError_t code;
	std::cout << "Hello GPU Saxpy!\n";

	float scale = 2.0f;

	float* x_cpu = (float*)malloc(sizeof(float) * vectorSize);
	float* y_cpu = (float*)malloc(sizeof(float) * vectorSize);
	float* y_res_cpu = (float*)malloc(sizeof(float) * vectorSize);
	if (!x_cpu || !y_cpu || !y_res_cpu) {
		fprintf(stderr, "Failed to allocate memory using malloc\n");
		return -1;
	}

	vectorInit(x_cpu, vectorSize);
	vectorInit(y_cpu, vectorSize);

	float* x_gpu = nullptr;
	code = cudaMalloc((void**)&x_gpu, sizeof(float) * vectorSize);
	gpuAssert(code, __FILE__, __LINE__, true);
	code = cudaMemcpy(x_gpu, x_cpu, sizeof(float) * vectorSize, cudaMemcpyHostToDevice);
	gpuAssert(code, __FILE__, __LINE__, true);

	float* y_gpu = nullptr;
	code = cudaMalloc((void**)&y_gpu, sizeof(float) * vectorSize);
	gpuAssert(code, __FILE__, __LINE__, true);
	code = cudaMemcpy(y_gpu, y_cpu, sizeof(float) * vectorSize, cudaMemcpyHostToDevice);
	gpuAssert(code, __FILE__, __LINE__, true);

	saxpy_gpu<<<ceil(vectorSize/256.0), 256>>>(x_gpu, y_gpu, scale, vectorSize);

	code = cudaMemcpy(y_res_cpu, y_gpu, sizeof(float) * vectorSize, cudaMemcpyDeviceToHost);
	gpuAssert(code, __FILE__, __LINE__, true);


	int errCnt = verifyVector(x_cpu, y_cpu, y_res_cpu, scale, vectorSize);
	printf("Found %d / %d erros\n", errCnt, vectorSize);

	code = cudaFree(x_gpu);
	gpuAssert(code, __FILE__, __LINE__, true);
	code = cudaFree(y_gpu);
	gpuAssert(code, __FILE__, __LINE__, true);

	free(x_cpu);
	free(y_cpu);
	free(y_res_cpu);

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= pSumSize) return;

	curandState_t rng;
	curand_init(clock64(), idx, 0, &rng);

	uint64_t sum = 0;
	for (uint64_t i = 0; i < sampleSize; i++) {
		float x = curand_uniform(&rng);
		float y = curand_uniform(&rng);
		sum += uint64_t(x*x + y*y < 1.0f);
	}

	pSums[idx] = sum;
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= (pSumSize + reduceSize - 1) / reduceSize) return;
	uint64_t total = 0;
	uint64_t start_pos = idx * reduceSize;
	for (uint64_t i = start_pos; i < start_pos + reduceSize && i < pSumSize; i++) {
		total += pSums[i];
	}
	totals[idx] = total;
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	auto tEnd= std::chrono::high_resolution_clock::now();
	std::cout << "Estimated Pi = " << approxPi << "\n";


	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	cudaError_t code;
	double approxPi = 0;

	uint64_t* sums_gpu = nullptr;
	code = cudaMalloc((void**)&sums_gpu, sizeof(uint64_t) * generateThreadCount);
	gpuAssert(code, __FILE__, __LINE__, true);

	uint64_t* totals_gpu = nullptr;
	code = cudaMalloc((void**)&totals_gpu, sizeof(uint64_t) * reduceThreadCount);
	gpuAssert(code, __FILE__, __LINE__, true);

	uint64_t *totals_cpu = (uint64_t*)malloc(sizeof(uint64_t) * reduceThreadCount);
	if (!totals_cpu) {
		fprintf(stderr, "Failed to allocate memory using malloc\n");
		exit(1);
	}

	generatePoints<<<ceil(generateThreadCount/256.0), 256>>>(
		sums_gpu, generateThreadCount, sampleSize
	);

	reduceCounts<<<ceil(reduceThreadCount/256.0), 256>>>(
		sums_gpu, totals_gpu, 
		generateThreadCount, reduceSize
	);

	code = cudaMemcpy(
		totals_cpu, totals_gpu, 
		sizeof(uint64_t) * reduceThreadCount, 
		cudaMemcpyDeviceToHost
	);
	gpuAssert(code, __FILE__, __LINE__, true);

	uint64_t total = 0;
	for (uint64_t i = 0; i < reduceThreadCount; i++) {
		total += totals_cpu[i];
	}

	approxPi = (4.0 * total) / (generateThreadCount * sampleSize);

	return approxPi;
}
