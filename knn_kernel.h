#pragma once
#ifndef KNN_KERNEL_H
#define KNN_KERNEL_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel function declaration
__global__ void computeKNN(const double* data, const double* queries,
    double* knnDistances, int* knnIndices,
    int numPoints, int numQueries, int k);

#endif // KNN_KERNEL_H
