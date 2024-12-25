#pragma once
#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath>     // �ṩ sin��cos��sqrt �Ⱥ���
#include <cfloat>    // �ṩ DBL_MAX
#define M_PI 3.14159265358979323846 // ��� M_PI δ����

// 100000���㣬ÿ������50��ά��
// 1000����ѯ�㣬ÿ����ѯ����50��ά��
// ÿ����ѯ���������50���� knn

// ֻ��������50����
// ���͵�0.5s����
const int DIMENSIONS = 50;
const int NUM_POINTS = 100000; // 100k
const int NUM_QUERIES = 100000; // 1k - 10k - 100k
const int K = 50;
const int BLOCK_SIZE = 32; //(16), 32, 64, 128, 256, 512, 1024
const int WARP_SIZE = 32;

#endif // CONSTANTS_H
