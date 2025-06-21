#include "CoffeeShop.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./simulator <price> <barista_wage>\n";
        return 1;
    }

    double price = std::atof(argv[1]);
    double baristaWage = std::atof(argv[2]);

    int numBaristas = 2;       // You can change this default if you want
    int numMachines = 2;       // Similarly, adjust machines count if needed
    double operatingHours = 840.0;

    CoffeeShop shop(price, numBaristas, numMachines, operatingHours, baristaWage);
    shop.generateCustomers();  
    shop.runSimulation();

    double profit = shop.returnResults();

    std::cout << "Profit (captured): " << profit << std::endl;

    return 0;
}
