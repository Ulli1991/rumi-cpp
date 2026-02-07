CXX = g++
CXXFLAGS = -O3 -march=native -std=c++17 -funroll-loops -fopenmp-simd -fno-trapping-math -fno-signed-zeros -Wall -Wextra -Iinitial_conditions
LDFLAGS = -lm

SRC = src/main.cpp src/particles.cpp src/forces.cpp src/integrator.cpp \
      src/timestep.cpp src/initial_conditions.cpp src/diagnostics.cpp \
      initial_conditions/sse.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = nbody

.PHONY: all clean debug

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

initial_conditions/%.o: initial_conditions/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

debug: CXXFLAGS = -O0 -g -std=c++17 -Wall -Wextra -fsanitize=address,undefined -Iinitial_conditions
debug: clean $(TARGET)

clean:
	rm -f $(TARGET) src/*.o initial_conditions/*.o
