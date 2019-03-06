/* Max Latitude - algorithm using thrust library on CUDA

Find the maximum Latitude among a lot of Latitudes

Obs:
A Max Latitude file must contain:
1. title: Latitude
2. N: Total number of coordinates
4. Latitudes of all points 

Example of file:
Latitude
5
-12.078973
-12.067450
-12.043947
-12.058031
-12.029678

Similarly for Max Longitude
To Min Latitude (or Min Longitude) only change to thrust::min_element

*/

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
using namespace std;

int main(void)
{
	int N;	// Number total of coordinates

	// Util pointers to host (h_) and device (_d)
	double *h_latitudes, h_max_lat;
	int h_max_pos;
	double *d_latitudes;

	// Total size in bytes
	const int TOTAL_SIZE_BYTES = N * sizeof(double);

	// Allocate CPU memory
	h_latitudes = (double*)malloc(TOTAL_SIZE_BYTES);

	// Allocate GPU memory
	cudaMalloc((void**)&d_latitudes, TOTAL_SIZE_BYTES);

	// Fill data in host
	readCoordinates(h_latitudes, N);

	// Transfer the coordinates' array from host to the device
	cudaMemcpy(d_latitudes, h_latitudes, TOTAL_SIZE_BYTES, cudaMemcpyHostToDevice);

	// device_ptr to use max_element
	thrust::device_ptr<double> d_lat_ptr = thrust::device_pointer_cast(d_latitudes);
	
	// Find the position (int) of the max latitude
	h_max_pos = thrust::max_element(d_lat_ptr, d_lat_ptr + N) - d_lat_ptr;

	// Find the max latitude
	h_max_lat = h_latitudes[h_max_pos];

	// Delete host memory
	free(h_latitudes);

	// Delete device memory
	cudaFree(d_latitudes);
	
	return 0;
}