#include "CoffeeShop.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>

CoffeeShop::CoffeeShop(double pricePerCup, int numBaristas, int numMachines, double operatingHours, double baristaWage)
    : pricePerCup(pricePerCup),
      totalBaristas(numBaristas),
      totalMachines(numMachines),
      operatingHours(operatingHours),
      availableBaristas(numBaristas),
      availableMachines(numMachines),
      baristaWage(baristaWage) {

    std::random_device rd;
    rng = std::mt19937(rd());

    double baseMeanInterarrival = 0.7;
    double priceSensitivity = 0.8;
    double priceBaseline = 2.5;

    double demandFactor = std::exp(-priceSensitivity * std::max(0.0, (pricePerCup - priceBaseline) / priceBaseline));
    double adjustedMeanInterarrival = baseMeanInterarrival / demandFactor;

    interarrivalDist   = std::exponential_distribution<double>(1.0 / adjustedMeanInterarrival);
    serviceTimeDist    = std::normal_distribution<double>(5.0, 0.5);
    patienceDist       = std::uniform_real_distribution<double>(5.0, 5.0);

    for (int i = 0; i < totalBaristas; ++i) {
        baristas.push_back({i, false, BaristaSkill::COMPETENT, 1.0, 0});
    }

    int numHours = static_cast<int>(std::ceil(operatingHours / 60.0));
    stats.hourlyRevenue.resize(numHours, 0.0);
    stats.hourlyCustomers.resize(numHours, 0);

    baseWage = 21.0;   // base wage for efficiency calculation
    efficiencyFactor = 0.4; // tuning parameter for efficiency gain
}

void CoffeeShop::generateCustomers() {
    double time = 0.0;

    if (pricePerCup <= 0.0) return;

    while (time < operatingHours) {
        double interval = interarrivalDist(rng) / getHourlyMultiplier(time);
        time += interval;
        if (time >= operatingHours) break;

        double serviceDuration = std::max(1.0, serviceTimeDist(rng));
        double patience = patienceDist(rng);
        addCustomer(time, serviceDuration, CustomerType::REGULAR, 1, patience);
    }
}

void CoffeeShop::addCustomer(double arrivalTime, double serviceDuration,
                             CustomerType type, int groupSize, double patience) {
    int id = allCustomers.size();
    allCustomers.push_back({id, arrivalTime, serviceDuration, type,
                            groupSize, patience, false});
    eventQueue.push({arrivalTime, EventType::ARRIVAL, id});
}

void CoffeeShop::runSimulation() {
    std::cout << "Starting coffee shop simulation for "
              << (operatingHours / 60.0) << " hours...\n";

    while (!eventQueue.empty() && currentTime <= operatingHours) {
        Event e = eventQueue.top();
        eventQueue.pop();
        lastEventTime = currentTime;
        currentTime = e.time;
        recordQueueStats();

        switch (e.type) {
          case EventType::ARRIVAL:       handleArrival(e);      break;
          case EventType::SERVICE_START: handleServiceStart(e); break;
          case EventType::SERVICE_END:   handleServiceEnd(e);   break;
        }
    }

    stats.baristaUtilization = 1.0 - (double)availableBaristas / totalBaristas;
    stats.machineUtilization = 1.0 - (double)availableMachines / totalMachines;
    updateHourlyStats();
    std::cout << "Simulation completed.\n";
}

void CoffeeShop::handleArrival(const Event& e) {
    Customer& c = allCustomers[e.customerId];
    if (currentTime >= operatingHours - 30.0) {
        c.abandoned = true; stats.customersLost++; return;
    }
    waitingQueue.push(c);
    tryServeNext();
}

void CoffeeShop::handleServiceStart(const Event& e) {
    Customer& c = allCustomers[e.customerId];
    double waitTime = currentTime - c.arrivalTime;
    stats.recordWaitTime(waitTime);

    if (waitTime > c.patience) {
        c.abandoned = true;
        stats.customersLost++;
        availableBaristas++;
        availableMachines++;
        tryServeNext();
        return;
    }

    int bId = customerToBaristaMap[c.id];
    Barista& b = baristas[bId];

    // Calculate barista efficiency based on wage (square root relationship)
    double wageDiff = std::max(0.0, baristaWage - baseWage);
    double baristaEfficiency = 1.0 + efficiencyFactor * std::sqrt(wageDiff);

    // Adjust service time by barista efficiency
    double actualTime = std::max(1.0, serviceTimeDist(rng) / baristaEfficiency);

    eventQueue.push({currentTime + actualTime, EventType::SERVICE_END, c.id});
}

void CoffeeShop::handleServiceEnd(const Event& e) {
    Customer& c = allCustomers[e.customerId];
    int bId = customerToBaristaMap[c.id];
    baristas[bId].isBusy = false;
    baristas[bId].customersServed++;
    availableBaristas++;
    availableMachines++;

    double revenue = pricePerCup;
    stats.totalRevenue += revenue;
    stats.customersServed++;

    int hr = static_cast<int>(currentTime / 60.0);
    if (hr < stats.hourlyRevenue.size()) {
        stats.hourlyRevenue[hr] += revenue;
        stats.hourlyCustomers[hr] += 1;
    }

    tryServeNext();
}

void CoffeeShop::tryServeNext() {
    while (!waitingQueue.empty() && availableBaristas > 0 && availableMachines > 0) {
        Customer c = waitingQueue.front();
        waitingQueue.pop();

        if (currentTime - c.arrivalTime > c.patience) {
            allCustomers[c.id].abandoned = true;
            stats.customersLost++;
            continue;
        }

        int bId = findAvailableBarista();
        if (bId < 0) break;

        baristas[bId].isBusy = true;
        customerToBaristaMap[c.id] = bId;
        availableBaristas--;
        availableMachines--;
        eventQueue.push({currentTime, EventType::SERVICE_START, c.id});
        break;
    }
}

int CoffeeShop::findAvailableBarista() {
    for (int i = 0; i < baristas.size(); ++i)
        if (!baristas[i].isBusy) return i;
    return -1;
}

void CoffeeShop::recordQueueStats() {
    if (currentTime > lastEventTime) {
        int q = waitingQueue.size();
        stats.recordQueueLength(q, currentTime - lastEventTime);
    }
}

double CoffeeShop::getHourlyMultiplier(double t) {
    double h = std::fmod(t / 60.0, 14.0);
    if (h >= 1.0 && h <= 3.0) return 1.8;
    else if (h >= 6.0 && h <= 8.0) return 1.6;
    else if (h >= 9.0 && h <= 11.0) return 0.8;
    else if (h >= 11.0 && h <= 14.0) return 1.2;
    return 1.0;
}

void CoffeeShop::updateHourlyStats() {
    double hours = operatingHours / 60.0;

    double baristaCost = totalBaristas * baristaWage * hours;
    double machineCost = totalMachines * machineCostPerHour * hours;
    double rentCost = rentPerDay;
    double ingredientCost = stats.customersServed * (ingredientCostRegular + cupCost);

    stats.costs = baristaCost + machineCost + rentCost + ingredientCost;
    stats.profit = stats.totalRevenue - stats.costs;
}

void CoffeeShop::printStatistics() {
    std::cout << "\n===== COFFEE SHOP SIMULATION RESULTS =====\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Operating Hours: " << (operatingHours / 60.0) << "\n";

    std::cout << "\nFINANCIAL STATISTICS:\n";
    std::cout << "Total Revenue: $" << stats.totalRevenue << "\n";
    std::cout << "Total Cost: $" << stats.costs << "\n";
    std::cout << "Profit: $" << stats.profit << "\n";
    std::cout << "Profit per Hour: $" << stats.profit / (operatingHours / 60.0) << "\n";

    std::cout << "\nCUSTOMER STATISTICS:\n";
    std::cout << "Served: " << stats.customersServed << "\n";
    std::cout << "Lost: " << stats.customersLost << "\n";
    double lossRate = (stats.customersLost / double(stats.customersServed + stats.customersLost)) * 100.0;
    std::cout << "Loss Rate: " << lossRate << "%\n";
    std::cout << "Avg Wait: " << stats.avgWaitTime << " min\n";
    std::cout << "Max Wait: " << stats.maxWaitTime << " min\n";
    std::cout << "Avg Queue: " << stats.avgQueueLength << "\n";

    std::cout << "\nRESOURCE UTILIZATION:\n";
    std::cout << "Baristas: " << (stats.baristaUtilization * 100.0) << "%\n";
    std::cout << "Machines: " << (stats.machineUtilization * 100.0) << "%\n";

    std::cout << "\n===== END OF REPORT =====\n";
}

double CoffeeShop::returnResults() {
    return stats.profit;
}
