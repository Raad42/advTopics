#include "CoffeeShop.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>

CoffeeShop::CoffeeShop(double pricePerCup, int numBaristas, int numMachines, double operatingHours)
    : pricePerCup(pricePerCup),
      totalBaristas(numBaristas),
      totalMachines(numMachines),
      operatingHours(operatingHours),
      availableBaristas(numBaristas),
      availableMachines(numMachines) {
    
    // Set up random number generator
    std::random_device rd;
    rng = std::mt19937(rd());
    
    // Base distributions
    interarrivalDist    = std::exponential_distribution<double>(1.0/0.5);  // mean 2 min
    regularServiceDist  = std::normal_distribution<double>(3.0, 1.0);    // mean 3±1
    complexServiceDist  = std::normal_distribution<double>(5.0, 1.5);    // mean 5±1.5
    groupSizeDist       = std::poisson_distribution<int>(2);             // mean 2 additional
    patienceDist        = std::uniform_real_distribution<double>(10.0, 45.0); // 10–45 min
    customerTypeDist    = std::discrete_distribution<int>({70,25,5});    // 70% reg,25% cmp,5% grp
    
    //Create baristas
    for (int i = 0; i < totalBaristas; ++i) {
        double roll = std::uniform_real_distribution<double>(0,1)(rng);

        BaristaSkill skill;
        double speedMod;

        //Made every employee same skill for now 
        skill = BaristaSkill::COMPETENT;
        speedMod = 1.0; 
       
        baristas.push_back({i,false,skill,speedMod,0});
    }
    
    // Prepare hourly stats
    int numHours = static_cast<int>(std::ceil(operatingHours / 60.0));
    stats.hourlyRevenue.resize(numHours, 0.0);
    stats.hourlyCustomers.resize(numHours, 0);
}


void CoffeeShop::generateCustomers() {
    double time = 0.0;
    
    if (pricePerCup <= 0.0){
        return;
    };
    
    while (time < operatingHours) {
        // draw base interval, then stretch by pricePerCup
        double rawInterval  = interarrivalDist(rng);
        double nextInterval = (rawInterval * pricePerCup) / getHourlyMultiplier(time);
        
        time += nextInterval;
        if (time >= operatingHours) break;
        
        // pick customer type
        int idx = customerTypeDist(rng);
        CustomerType type = static_cast<CustomerType>(idx);
        
        // service duration & group size
        double serviceDuration;
        int groupSize = 1;
        if (type == CustomerType::REGULAR) {
            serviceDuration = std::max(1.0, regularServiceDist(rng));
        }
        else if (type == CustomerType::COMPLEX) {
            serviceDuration = std::max(2.0, complexServiceDist(rng));
        }
        else { 
            groupSize = 1 + groupSizeDist(rng);
            serviceDuration = std::max(2.0,
                regularServiceDist(rng) * (1.0 + 0.5 * groupSize));
        }
        
        double patience = patienceDist(rng);
        addCustomer(time, serviceDuration, type, groupSize, patience);
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
              << (operatingHours/60.0) << " hours...\n";

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
    
    stats.baristaUtilization = 1.0 - (double)availableBaristas/totalBaristas;
    stats.machineUtilization = 1.0 - (double)availableMachines/totalMachines;
    updateHourlyStats();
    std::cout << "Simulation completed.\n";
}

void CoffeeShop::handleArrival(const Event& e) {
    Customer& c = allCustomers[e.customerId];
    //Do not accept new customers last 30 mins 
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
    double actualTime = getServiceTimeFactor(c.type, b.skill, c.groupSize)
                        * c.serviceDuration;
    eventQueue.push({currentTime + actualTime,
                     EventType::SERVICE_END, c.id});
}

double CoffeeShop::getServiceTimeFactor(CustomerType type,
                                       BaristaSkill skill,
                                       int groupSize) {
    double f=1.0;

    if (type==CustomerType::COMPLEX)         f*=1.2;
    std::uniform_real_distribution<double> v(0.9,1.1);
    return f * v(rng);
}

void CoffeeShop::handleServiceEnd(const Event& e) {
    Customer& c = allCustomers[e.customerId];
    int bId = customerToBaristaMap[c.id];
    baristas[bId].isBusy = false;
    baristas[bId].customersServed++;
    availableBaristas++;
    availableMachines++;
    
    double revenue = 0.0;
    switch (c.type) {
      case CustomerType::REGULAR: revenue = pricePerCup; break;
      case CustomerType::COMPLEX: revenue = pricePerCup * 1.5; break;
      case CustomerType::GROUP:   revenue = pricePerCup * c.groupSize; break;
    }
    stats.totalRevenue += revenue;
    stats.customersServed++;
    
    int hr = static_cast<int>(currentTime/60.0);
    if (hr < stats.hourlyRevenue.size()) {
        stats.hourlyRevenue[hr]   += revenue;
        stats.hourlyCustomers[hr] += 1;
    }
    
    tryServeNext();
}

void CoffeeShop::tryServeNext() {
    while (!waitingQueue.empty()
           && availableBaristas>0
           && availableMachines>0) {
        Customer c = waitingQueue.front();
        waitingQueue.pop();
        if (currentTime - c.arrivalTime > c.patience) {
            allCustomers[c.id].abandoned = true;
            stats.customersLost++;
            continue;
        }
        int bId = findAvailableBarista();
        if (bId<0) break;
        baristas[bId].isBusy = true;
        customerToBaristaMap[c.id] = bId;
        availableBaristas--;
        availableMachines--;
        eventQueue.push({currentTime,
                         EventType::SERVICE_START, c.id});
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
    double h = std::fmod(t/60.0, 14.0);
    if      (h>=1.0  && h<=3.0)  return peakHourMultiplier; //Morning
    else if (h>=6.0  && h<=8.0)  return peakHourMultiplier * 0.9; //Lunch
    else if (h>=9.0  && h<=11.0) return 0.8;
    else if (h>=11.0 && h<=14.0) return 1.2;
    return 1;
}

void CoffeeShop::updateHourlyStats() {
      // Operating time in hours
    double hours = operatingHours / 60.0;

    // Fixed Costs
    double baristaCost = totalBaristas * baristaWagePerHour * hours;
    double machineCost = totalMachines * machineCostPerHour * hours;
    double rentCost = rentPerDay;

    // Variable Costs
    double ingredientCost = 0.0;
    for (const auto& c : allCustomers) {
        if (!c.abandoned) {
            if (c.type == CustomerType::REGULAR)
                ingredientCost += ingredientCostRegular + cupCost;
            else if (c.type == CustomerType::COMPLEX)
                ingredientCost += ingredientCostComplex + cupCost;
            else // GROUP
                ingredientCost += (ingredientCostPerGroupCup + cupCost) * c.groupSize;
        }
    }

    stats.costs = baristaCost + machineCost + rentCost + ingredientCost;

    stats.profit = stats.totalRevenue - stats.costs; 

}

void CoffeeShop::printStatistics() {
    std::cout << "\n===== COFFEE SHOP SIMULATION RESULTS =====\n";
    std::cout << "Operating Hours: " << (operatingHours/60.0) << " hours\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nFINANCIAL STATISTICS:\n";
    std::cout << "Total Revenue: $" << stats.totalRevenue << "\n";
    std::cout << "Average Rev/Hour: $"
              << (stats.totalRevenue/(operatingHours/60.0)) << "\n\n";
    
    std::cout << "CUSTOMER STATISTICS:\n";
    std::cout << "Served: "  << stats.customersServed << "\n";
    std::cout << "Lost: "    << stats.customersLost  << "\n";
    double lossRate = (stats.customersLost /
                      double(stats.customersServed+stats.customersLost))*100;
    std::cout << "Loss Rate: " << lossRate << "%\n";
    std::cout << "Avg Wait: "  << stats.avgWaitTime << " min\n";
    std::cout << "Max Wait: "  << stats.maxWaitTime << " min\n";
    std::cout << "Avg Queue: " << stats.avgQueueLength << "\n\n";
    
    std::cout << "RESOURCE UTILIZATION:\n";
    std::cout << "Baristas: " << (stats.baristaUtilization*100) << "%\n";
    std::cout << "Machines: " << (stats.machineUtilization*100) << "%\n\n";
    
    std::cout << "BARISTA PERFORMANCE:\n";
    for (auto& b : baristas) {
        std::string lvl = (b.skill==BaristaSkill::NOVICE  ? "Novice"  :
                           b.skill==BaristaSkill::COMPETENT? "Competent":
                                                             "Expert");
        std::cout << "Barista #" << b.id << " (" << lvl << "): "
                  << b.customersServed << " served\n";
    }
    
    std::cout << "\nHOURLY BREAKDOWN:\n";
    std::cout << "Hour | Cust | Rev  | Rev/Cust\n";
    std::cout << "------------------------------\n";
    for (size_t i=0; i<stats.hourlyRevenue.size(); ++i) {
        double avg = stats.hourlyCustomers[i]
                   ? stats.hourlyRevenue[i]/stats.hourlyCustomers[i]
                   : 0.0;
        std::cout << std::setw(4) << i << " | "
                  << std::setw(4) << stats.hourlyCustomers[i] << " | $"
                  << std::setw(4) << stats.hourlyRevenue[i]   << " | $"
                  << std::setw(4) << avg << "\n";
    }
    std::cout << "\n===== END OF REPORT =====\n";

    // Operating time in hours
    double hours = operatingHours / 60.0;

    // Fixed Costs
    double baristaCost = totalBaristas * baristaWagePerHour * hours;
    double machineCost = totalMachines * machineCostPerHour * hours;
    double rentCost = rentPerDay;

    // Variable Costs
    double ingredientCost = 0.0;
    for (const auto& c : allCustomers) {
        if (!c.abandoned) {
            if (c.type == CustomerType::REGULAR)
                ingredientCost += ingredientCostRegular + cupCost;
            else if (c.type == CustomerType::COMPLEX)
                ingredientCost += ingredientCostComplex + cupCost;
            else // GROUP
                ingredientCost += (ingredientCostPerGroupCup + cupCost) * c.groupSize;
        }
    }

    double totalCost = baristaCost + machineCost + rentCost + ingredientCost;
    stats.profit = stats.totalRevenue - totalCost;

    // Print new info
    std::cout << "COST BREAKDOWN:\n";
    std::cout << "Barista Cost: $" << baristaCost << "\n";
    std::cout << "Machine Cost: $" << machineCost << "\n";
    std::cout << "Rent Cost: $" << rentCost << "\n";
    std::cout << "Ingredients & Supplies: $" << ingredientCost << "\n";
    std::cout << "Total Cost: $" << totalCost << "\n\n";

    std::cout << "NET PROFIT:\n";
    std::cout << "Profit: $" << stats.profit << "\n";
    std::cout << "Profit per Hour: $" << stats.profit / hours << "\n";

}

double CoffeeShop::returnResults(){
    return stats.profit;
}   
