#ifndef BARISTA_H
#define BARISTA_H

enum class BaristaSkill {
    NOVICE,     // Slower service time
    COMPETENT,  // Average service time
    EXPERT      // Faster service time
};

struct Barista {
    int id;
    bool isBusy = false;
    BaristaSkill skill;
    double speedModifier;  // Affects service time based on skill
    int customersServed;   // Keeps track of how many customers this barista served
};

#endif