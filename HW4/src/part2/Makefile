CXX := mpicxx 
CXXFLAGS := -I./common -O3 -std=c++17 -Wall

ifeq (/usr/bin/g++-10,$(wildcard /usr/bin/g++-10*))
    CXX=g++-10
endif

all: myexp

myexp: main.cc cmetrix.cc
	$(CXX)  main.cc cmetrix.cc -o matmul

clean:
	rm -f *.o *.s myexp *~
