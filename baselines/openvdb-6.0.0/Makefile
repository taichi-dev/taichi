test: test.cpp
	 clang++-7 test.cpp -o test -g -Wall -std=c++14 -I. -ltbb -L. -lopenvdb -lboost_iostreams -lHalf	

run: test
	LD_LIBRARY_PATH=. ./test
