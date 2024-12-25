#pragma once
#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <string>
#include <vector>

// º¯ÊıÉùÃ÷
void generateData(const std::string& filename, unsigned int seed);
void loadData(const std::string& filename, std::vector<std::vector<double>>& data);
std::vector<std::vector<double>> generateQueryPoints(const std::string& filename, unsigned int seed);

#endif // DATA_UTILS_H
