# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra

# Source files
SRCS = Simulator.cpp CoffeeShop.cpp

# Header files
HDRS = Event.h Customer.h Barista.h Statistics.h CoffeeShop.h

# Object files
OBJS = $(SRCS:.cpp=.o)

# Output binary
TARGET = simulator

# Build target
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Rule for object files - makes objects depend on both .cpp and all header files
%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)

# Run rule
run: $(TARGET)
	./$(TARGET)