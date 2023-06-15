CXX := mpicxx 
CXXFLAGS := -I./common -O3 -std=c++17 -Wall


all: myexp

myexp: main.cc cmetrix.cc
	$(CXX) -O3 main.cc cmetrix.cc -o matmul

clean:
	rm -f *.o *.s myexp *~
