#include "data_utils.h"
#include "constants.h"
#include <random>
#include <fstream>
#include <sstream>
#include <iostream>

/*
dataset: NUM_POINTS points£¬each point has DIMENSIONS
range: [0.0, 100.0] floating point numbers
format: csv, each line represents a point,
    the feature values of each line are separated by commas.
*/

// Generate dataset and save to file
void generateData(const std::string& filename, unsigned int seed) {
    // Initialize random number generator with the given seed
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 100.0);

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < NUM_POINTS; ++i) {
        for (int j = 0; j < DIMENSIONS; ++j) {
            file << dis(gen);
            if (j < DIMENSIONS - 1) file << ","; // Add comma for all but the last value
        }
        file << "\n"; // New line after each point
    }
    file.close();
    std::cout << "Dataset saved to: " << filename << std::endl;
}

// Load dataset from file
void loadData(const std::string& filename, std::vector<std::vector<double>>& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return;
    }

    std::string line;
    //std::cout << "Loading data from: " << filename << std::endl;
    while (std::getline(file, line)) {
        std::vector<double> point;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            point.push_back(std::stod(value)); // Convert string to double
        }
        data.push_back(point);
    }
    file.close();
}

// Generate query points and save to file
std::vector<std::vector<double>> generateQueryPoints(const std::string& filename, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(101.0, 200.0);

    std::vector<std::vector<double>> queryPoints(NUM_QUERIES, std::vector<double>(DIMENSIONS));
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return queryPoints;
    }

    for (int i = 0; i < NUM_QUERIES; ++i) {
        for (int j = 0; j < DIMENSIONS; ++j) {
            queryPoints[i][j] = dis(gen);
            file << queryPoints[i][j];
            if (j < DIMENSIONS - 1) file << ","; // Add comma for all but the last value
        }
        file << "\n"; // New line after each point
    }
    file.close();
    std::cout << "Query points saved to: " << filename << std::endl << std::endl;
    return queryPoints;
}
