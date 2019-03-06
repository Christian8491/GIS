/* Minimum Bounding Box on the Earth - algorithm using thrust library on CUDA

Find the extremas of a set of coordinates and get the Minimum Bounding Box

Obs:
A Minimum Bounding Box on the Earth file must contain:
1. title: Latitude,Longitude
2. N: Total number of coordinates
4. Coordinates of all points

Example of file:
Latitude, Longitude
5
-12.078973,-45.026742
-12.067450,-45.780002
-12.043947,-41.071173
-12.058031,-42.771933
-12.029678,-44.876330

This code was implemented by Christian Córdova Estrada
*/

#include <fstream>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

template<typename T>
struct Coordinate
{
	T latitude;
	T longitude;
	Coordinate(){}
	Coordinate(T lat_, T long_) { latitude = lat_; longitude = long_; }
};

int main(void)
{
	int N;		// Number of coordinates

	// Util pointers to host (h_) and device (_d)
	double *h_latitudes, *h_longitudes;
	double *d_latitudes, *d_longitudes;

	// To find extremas positions
	int h_min_lat_pos, h_min_lon_pos, h_max_lat_pos, h_max_lon_pos;

	// Total size in bytes
	const int TOTAL_SIZE_BYTES = N * sizeof(double);

	// Allocate CPU memory
	h_latitudes = (double*)malloc(TOTAL_SIZE_BYTES);
	h_longitudes = (double*)malloc(TOTAL_SIZE_BYTES);

	// Allocate GPU memory
	cudaMalloc((void**)&d_latitudes, TOTAL_SIZE_BYTES);
	cudaMalloc((void**)&d_longitudes, TOTAL_SIZE_BYTES);

	// Fill data with some funtion in host
	readCoordinates(h_latitudes, h_longitudes, N);
	
	// Transfer the coordinates' array from host to the device
	cudaMemcpy(d_latitudes, h_latitudes, TOTAL_SIZE_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_longitudes, h_longitudes, TOTAL_SIZE_BYTES, cudaMemcpyHostToDevice);

	// device_ptr to use max_element and min_element
	thrust::device_ptr<double> d_lat_ptr = thrust::device_pointer_cast(d_latitudes);
	thrust::device_ptr<double> d_lon_ptr = thrust::device_pointer_cast(d_longitudes);

	// Find the positions (int) of the max, min latitudes and longitudes
	h_min_lat_pos = thrust::min_element(d_lat_ptr, d_lat_ptr + N) - d_lat_ptr;
	h_max_lat_pos = thrust::max_element(d_lat_ptr, d_lat_ptr + N) - d_lat_ptr;
	h_min_lon_pos = thrust::min_element(d_lon_ptr, d_lon_ptr + N) - d_lon_ptr;
	h_max_lon_pos = thrust::max_element(d_lon_ptr, d_lon_ptr + N) - d_lon_ptr;

	// Find the extrema coordinates of the max, min latitudes and longitudes
	Coordinate<double> upper_left(h_latitudes[h_max_lat_pos], h_longitudes[h_min_lon_pos]);
	Coordinate<double> upper_right(h_latitudes[h_max_lat_pos], h_longitudes[h_max_lon_pos]);
	Coordinate<double> bottom_right(h_latitudes[h_min_lat_pos], h_longitudes[h_max_lon_pos]);
	Coordinate<double> bottom_left(h_latitudes[h_min_lat_pos], h_longitudes[h_min_lon_pos]);

	// Delete host memory
	free(h_latitudes), free(h_longitudes);

	// Delete device memory
	cudaFree(d_latitudes), cudaFree(d_longitudes);

	return 0;
}