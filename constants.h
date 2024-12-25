#pragma once
#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath>     // 提供 sin、cos、sqrt 等函数
#include <cfloat>    // 提供 DBL_MAX
#define M_PI 3.14159265358979323846 // 如果 M_PI 未定义

// 100000个点，每个点有50个维度
// 1000个查询点，每个查询点有50个维度
// 每个查询点找最近的50个点 knn

// 只输出最近的50个点
// 降低到0.5s以下
const int DIMENSIONS = 50;
const int NUM_POINTS = 100000; // 100k
const int NUM_QUERIES = 100000; // 1k - 10k - 100k
const int K = 50;
const int BLOCK_SIZE = 32; //(16), 32, 64, 128, 256, 512, 1024
const int WARP_SIZE = 32;

#endif // CONSTANTS_H
