
CXXFLAGS = -std=c++14 -g -Wall -Wextra -pthread -fPIC
CXXFLAGS += -O3 -ffast-math -flto
LIBS = #-lrpc
OBJECTS = self_play_client.o chomp_rules.o

all: self_play_client.so #fast_server self_play_client

self_play_client: $(OBJECTS) Makefile
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS) $(LIBS)

self_play_client.so: $(OBJECTS) Makefile
	$(CXX) $(CXXFLAGS) -shared -Wl,-soname,$@ -o $@ $(OBJECTS) $(LIBS)

.PHONY: clean
clean:
	rm -f *.o self_play_client self_play_client.so

