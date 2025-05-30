#ifndef STATISTICS_H
#define STATISTICS_H

#include <vector>

struct Statistics {
    double totalRevenue = 0.0;
    int customersServed = 0;
    int customersLost = 0;
    double totalWaitTime = 0.0;
    double maxWaitTime = 0.0;
    double avgWaitTime = 0.0;
    double totalQueueLength = 0.0;
    int queueLengthSamples = 0;
    double avgQueueLength = 0.0;
    std::vector<double> hourlyRevenue;
    std::vector<int> hourlyCustomers;
    double baristaUtilization = 0.0;
    double machineUtilization = 0.0;

    double revenue = 0.0;
    double costs = 0.0;

    double profit = 0.0; 
    
    void recordWaitTime(double waitTime) {
        totalWaitTime += waitTime;
        if (waitTime > maxWaitTime) maxWaitTime = waitTime;
        avgWaitTime = totalWaitTime / customersServed;
    }
    
    void recordQueueLength(int length, double timeIncrement) {
        totalQueueLength += length * timeIncrement;
        queueLengthSamples++;
        avgQueueLength = totalQueueLength / queueLengthSamples;
    }
};

#endif