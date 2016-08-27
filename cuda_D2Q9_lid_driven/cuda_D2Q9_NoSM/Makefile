## Compiler, tools and options
CC      =  nvcc
FC      =  nvcc
LINK    =  nvcc
CCFLAGS = -O3

## Libraries
LIBS =  -lm -lcublas

## Files
OBJECTS = D2Q9_LBGK.o collision_propagation_kernel.o exchange_kernel.o

TARGET  = d2q9

## Implicit rules
.SUFFIXES: .cu
.cu.o:
	$(CC) -c $(CCFLAGS) $<

## Build rules
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(LINK) -o $@ $(OBJECTS) $(CCFLAGS) $(LIBS)

clean:
	rm -f $(TARGET)
	rm -f *~ core
