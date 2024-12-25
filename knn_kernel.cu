#include "knn_kernel.h"
#include "constants.h"
#include <cmath>
#include <float.h>

#define WARP_SIZE 32

__device__ double calculateDistance(const double* a, const double* b) {
    double distance = 0.0;
    for (int i = 0; i < DIMENSIONS; ++i) {
        double diff = a[i] - b[i];
        distance += diff * diff;
    }
    return sqrt(distance);
}

__global__ void computeKNN(const double* data, const double* queries,
    double* knnDistances, int* knnIndices,
    int numPoints, int numQueries, int k) {
    extern __shared__ double sharedMemory[];
    int tid = threadIdx.x;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    int queryIndex = blockIdx.x * blockDim.x / WARP_SIZE + warpId;

    if (queryIndex >= numQueries) return;

    const double* queryPoint = &queries[queryIndex * DIMENSIONS];
    double* warpDistances = &sharedMemory[warpId * k];
    int* warpIndices = (int*)&warpDistances[k];

    for (int i = laneId; i < k; i += WARP_SIZE) {
        warpDistances[i] = DBL_MAX;
        warpIndices[i] = -1;
    }

    __syncwarp();

    for (int dataIndex = laneId; dataIndex < numPoints; dataIndex += WARP_SIZE) {
        const double* dataPoint = &data[dataIndex * DIMENSIONS];
        double distance = calculateDistance(queryPoint, dataPoint);

        for (int i = 0; i < k; ++i) {
            if (distance < warpDistances[i]) {
                for (int j = k - 1; j > i; --j) {
                    warpDistances[j] = warpDistances[j - 1];
                    warpIndices[j] = warpIndices[j - 1];
                }
                warpDistances[i] = distance;
                warpIndices[i] = dataIndex;
                break;
            }
        }
    }

    __syncwarp();

    if (laneId < k) {
        knnDistances[queryIndex * k + laneId] = warpDistances[laneId];
        knnIndices[queryIndex * k + laneId] = warpIndices[laneId];
    }
}
