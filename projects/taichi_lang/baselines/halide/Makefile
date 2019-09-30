saxpy: saxpy.cpp
	icc -mkl saxpy.cpp -o saxpy -O3 -std=c++14

halide_sl: halide_sl.cpp
	ti amal
	g++-5 halide_sl.cpp -I ~/repos/halide/include -I ~/repos/halide/tools -L ~/repos/halide/bin -lHalide `libpng-config --cflags --ldflags` -ljpeg -o halide_sl -O2 -std=c++14 -lX11 -lpthread -DTC_IMAGE_IO

run_halide_sl: halide_sl
	LD_LIBRARY_PATH=~/repos/halide/bin ./halide_sl

mpm88: mpm88.cpp
	g++ mpm88.cpp -std=c++14 -g -lX11 -lpthread -O3 -o mls-mpm
	./mls-mpm
