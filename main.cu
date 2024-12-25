#include "data_utils.h"
#include "constants.h"
#include "knn_kernel.h"
#include "cuda_error_check.h"
#include <iostream>
#include <vector>

int main() {
    std::string dataFilename = "knn_data.txt";
    std::string queryFilename = "knn_query.txt";

    unsigned int dataSeed = 42;
    unsigned int querySeed = 4745;

    // Generate and save data
    generateData(dataFilename, dataSeed);

    // Load data from file
    std::vector<std::vector<double>> hostData;
    loadData(dataFilename, hostData);

    // Generate and save query points
    std::vector<std::vector<double>> hostQueries = generateQueryPoints(queryFilename, querySeed);

    // Flatten data for device
    std::vector<double> flatData(NUM_POINTS * DIMENSIONS);
    std::vector<double> flatQueries(NUM_QUERIES * DIMENSIONS);
    for (int i = 0; i < NUM_POINTS; ++i) {
        for (int j = 0; j < DIMENSIONS; ++j) {
            flatData[i * DIMENSIONS + j] = hostData[i][j];
        }
    }
    for (int i = 0; i < NUM_QUERIES; ++i) {
        for (int j = 0; j < DIMENSIONS; ++j) {
            flatQueries[i * DIMENSIONS + j] = hostQueries[i][j];
        }
    }

    // Allocate device memory
    double* deviceData, * deviceQueries, * deviceDistances;
    int* deviceIndices;
    CHECK_CUDA_ERROR(cudaMalloc(&deviceData, NUM_POINTS * DIMENSIONS * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&deviceQueries, NUM_QUERIES * DIMENSIONS * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&deviceDistances, NUM_QUERIES * K * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&deviceIndices, NUM_QUERIES * K * sizeof(int)));

    // Create CUDA events for timing
    cudaEvent_t totalStart, totalStop;
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&totalStart));
    CHECK_CUDA_ERROR(cudaEventCreate(&totalStop));
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Start total timing (data to GPU + compute + data back)
    CHECK_CUDA_ERROR(cudaEventRecord(totalStart, 0));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
    CHECK_CUDA_ERROR(cudaMemcpy(deviceData, flatData.data(), NUM_POINTS * DIMENSIONS * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(deviceQueries, flatQueries.data(), NUM_QUERIES * DIMENSIONS * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float dataToDeviceTime = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&dataToDeviceTime, start, stop));

    // Start recording kernel execution time
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));

    // Launch kernel
    int gridSize = (NUM_QUERIES + BLOCK_SIZE - 1) / BLOCK_SIZE; // 8 blocks
    // 将查询分配给一个线程
    size_t sharedMemorySize = BLOCK_SIZE / WARP_SIZE * K * sizeof(double) + BLOCK_SIZE / WARP_SIZE * K * sizeof(int);
    computeKNN << <gridSize, BLOCK_SIZE, sharedMemorySize >> > (
        deviceData, deviceQueries, deviceDistances, deviceIndices, NUM_POINTS, NUM_QUERIES, K);

    // Ensure kernel execution is complete
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float kernelTime = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, start, stop));

    // Copy results back to host
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
    std::vector<double> hostDistances(NUM_QUERIES * K);
    std::vector<int> hostIndices(NUM_QUERIES * K);
    CHECK_CUDA_ERROR(cudaMemcpy(hostDistances.data(), deviceDistances, NUM_QUERIES * K * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(hostIndices.data(), deviceIndices, NUM_QUERIES * K * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float dataToHostTime = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&dataToHostTime, start, stop));

    // Stop total timing
    CHECK_CUDA_ERROR(cudaEventRecord(totalStop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(totalStop));
    float totalTime = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&totalTime, totalStart, totalStop));

    // Display timings
    std::cout << "Data to Device Time: " << dataToDeviceTime << " ms\n";
    std::cout << "Kernel Execution Time: " << kernelTime << " ms\n";
    std::cout << "Data to Host Time: " << dataToHostTime << " ms\n";
    std::cout << "Total Time (Data to GPU + Compute + Data to Host): " << totalTime << " ms\n\n";

    // Display results
    for (int i = 0; i < 1; ++i) { // Display only first query results
        std::cout << "Query " << i << ":\n";
        for (int j = 0; j < 5; ++j) {
            std::cout << "Neighbor " << j << ": Index = " << hostIndices[i * K + j]
                << ", Distance = " << hostDistances[i * K + j] << "\n";
        }
    }

    // Free device memory
    cudaFree(deviceData);
    cudaFree(deviceQueries);
    cudaFree(deviceDistances);
    cudaFree(deviceIndices);

    // Destroy CUDA events
    cudaEventDestroy(totalStart);
    cudaEventDestroy(totalStop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
