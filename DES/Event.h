#ifndef EVENT_H
#define EVENT_H

#include <queue>

enum class EventType {
    ARRIVAL,
    SERVICE_START,
    SERVICE_END
};

struct Event {
    double time;
    EventType type;
    int customerId;

    bool operator<(const Event& other) const {
        return time > other.time; // for min-heap priority queue
    }
};


#endif