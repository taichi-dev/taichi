bw: bandwidth.cu
	nvcc bandwidth.cu -o bw -g -O2 -ccbin=/usr/bin/gcc-6 -lstdc++ -Xptxas -dlcm=ca -arch=compute_61 -code=sm_61,compute_61

fill: fill.cu
	nvcc fill.cu -o fill -g -O2 -ccbin=/usr/bin/gcc-6 -lstdc++ -Xptxas -dlcm=ca -arch=compute_61 -code=sm_61,compute_61

mutex: mutex.cu
	nvcc mutex.cu -o mutex -g -O2 -ccbin=/usr/bin/gcc-6 -lstdc++ -Xptxas -dlcm=ca -arch=compute_61 -code=sm_61,compute_61
