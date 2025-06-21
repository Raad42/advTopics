#ifndef CUSTOMER_H
#define CUSTOMER_H

#include <string>

enum class CustomerType {
    REGULAR,     // Simple coffee order
    COMPLEX,     // Specialty drinks requiring more time
    GROUP,       // Multiple orders at once
};

struct Customer {
    int id;
    double arrivalTime;
    double serviceDuration;
    CustomerType type;
    int groupSize;   // For group orders
    double patience; // How long they're willing to wait before leaving
    bool abandoned;  // Whether they left without being served
};

#endif