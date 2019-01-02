saxpy: saxpy.cpp
	icc -mkl saxpy.cpp -o saxpy -O3 -std=c++14


halide_sl: halide_sl.cpp
	g++-5 halide_sl.cpp -I ~/repos/halide/include -I ~/repos/halide/tools -L ~/repos/halide/bin -lHalide `libpng-config --cflags --ldflags` -ljpeg -o halide_sl -std=c++11 -O2

run_halide_sl: halide_sl
	LD_LIBRARY_PATH=~/repos/halide/bin ./halide_sl

