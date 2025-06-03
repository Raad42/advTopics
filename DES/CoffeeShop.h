#ifndef COFFEESHOP_H
#define COFFEESHOP_H

#include <vector>
#include <queue>
#include <random>
#include <map>
#include "Event.h"
#include "Customer.h"
#include "Barista.h"
#include "Statistics.h"

class CoffeeShop {
public:
    // Constructor
    CoffeeShop(double pricePerCup, int numBaristas, int numMachines, double operatingHours);

    // Simulation control
    void generateCustomers();
    void addCustomer(double arrivalTime, double serviceDuration,
                     CustomerType type = CustomerType::REGULAR,
                     int groupSize = 1, double patience = 30.0);
    void runSimulation();
    void printStatistics();
    double returnResults(); 

    // Accessor for statistics
    const Statistics& getStats() const { return stats; }

private:
    double pricePerCup;
    int totalBaristas;
    int totalMachines;
    double operatingHours;

    double currentTime = 0.0;
    double lastEventTime = 0.0;
    double peakHourMultiplier = 2;  

    double baristaWagePerHour = 15.0;
    double machineCostPerHour = 5.0;
    double rentPerDay = 100.0;
    double ingredientCostRegular = 1.0;
    double ingredientCostComplex = 1.5;
    double ingredientCostPerGroupCup = 1.0;
    double cupCost = 0.1;


    Statistics stats;
    std::vector<Barista> baristas;
    std::queue<Customer> waitingQueue;
    std::priority_queue<Event> eventQueue;
    std::vector<Customer> allCustomers;
    std::map<int, int> customerToBaristaMap;

    int availableBaristas;
    int availableMachines;

    // Random generators
    std::mt19937 rng;
    std::exponential_distribution<double> interarrivalDist;
    std::normal_distribution<double> regularServiceDist;
    std::normal_distribution<double> complexServiceDist;
    std::poisson_distribution<int> groupSizeDist;
    std::uniform_real_distribution<double> patienceDist;
    std::discrete_distribution<int> customerTypeDist;

    // Event handlers
    void handleArrival(const Event& e);
    void handleServiceStart(const Event& e);
    void handleServiceEnd(const Event& e);
    void tryServeNext();

    // Helpers
    void recordQueueStats();
    int findAvailableBarista();
    double getServiceTimeFactor(CustomerType type, BaristaSkill skill, int groupSize);
    double getHourlyMultiplier(double time);
    void updateHourlyStats();
};

#endif // COFFEESHOP_H


