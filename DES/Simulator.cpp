#include "CoffeeShop.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./simulator <price> <baristas>\n";
        return 1;
    }

    double price = std::atof(argv[1]);
    int baristas = std::atoi(argv[2]);

    double operatingHours = 840.0;

    CoffeeShop shop(price, baristas, baristas, operatingHours);
    shop.generateCustomers();  
    shop.runSimulation();

    double profit = shop.returnResults();

    std::cout << "Profit (captured): " << profit << std::endl;

    return 0;
}
