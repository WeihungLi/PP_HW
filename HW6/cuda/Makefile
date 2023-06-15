NVCC = nvcc
CXX=g++
CXXFLAGS=-I./common -Iobjs/ -O3 -std=c++17 -Wall -g -fPIC -lm

APP_NAME=res
CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' 
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3

default: $(APP_NAME)
clean:
	rm -f *.ppm *~ $(APP_NAME)

OBJS = main.o kernel.o bmpfuncs.o serialConv.o helper.o

$(APP_NAME): $(OBJS)
	$(NVCC) ${CUDA_LINK_FLAGS}-o $(APP_NAME) $@ $(OBJS) -lm
/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@
%.o: %.c
	$(CXX) $< $(CXXFLAGS) -c -o $@
main.o: CycleTimer.h kernel.h

kernel.o : kernel.cu kernel.h
	${NVCC} ${CUDA_COMPILE_FLAGS} -c kernel.cu -o $@
