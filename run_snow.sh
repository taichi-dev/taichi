ti mpm_full scene=1 material=snow output=snow_unbounded ground_friction=0.4 frame_dt=0.001 dt_mul=0.5 E=4e4 group_size=1000 total_frames=200
cd examples
for i in {0..199}
do
	python3 renderer.py snow snow_unbounded $i
done
cd ..

ti mpm_full scene=1 material=snow output=snow_bounded bbox=true ground_friction=0.4 frame_dt=0.001 dt_mul=0.5 E=4e4 group_size=1000 total_frames=200
cd examples
for i in {0..199}
do
	python3 renderer.py snow snow_bounded $i
done
