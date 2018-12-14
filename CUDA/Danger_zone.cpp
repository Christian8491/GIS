/* DANGER ZONE from coordinates using CUDA

A danger zone file should contain:
1. Title: Latitude,Longitude
2. Total numbers of points N
4. N Coordinates

Example:
Latitude,Longitude
5
-12.054251,-77.099688
-12.078973,-77.093252
-12.067450,-77.078543
-12.043947,-77.098275
-12.058031,-77.073309

This code was implemented by Christian Córdova Estrada
*/

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

#define NUM_BOUNDS 20

string input_path = "../../../Datasets/danger_zone/input/dz_10.txt";	// put your path files here
__constant__ double DISTANCE = 20.0;	// in meters
__constant__ int ANGLE_OFFSET = 18;
__constant__ int TOTAL_DEGREES = 360;
__constant__ double DEG_TO_RAD = 0.01745329251994329;
__constant__ double EARTH_RADIUS = 6372797.560856;		// in meters
__constant__ double M_PI = 3.14159265358979323846;

/* This function find the size of coordinates from two files
/param N: N of coordinates */
template <typename T>
void find_N(T& N)
{
	ifstream file;
	file.open(input_path);

	string titleStr, NStr;

	if (file.is_open()) {
		getline(file, titleStr);
		getline(file, NStr);
		N = (T)atof(NStr.c_str());
		return;
	}
}

/* This function reads the coordinates from two files and fill data
/param lats: latitudes of the coordinates
/param lons: longitudes of the coordinates
/param N: Total Number of coordinates */
template <typename T>
void load_data(T*& lats, T*& lons, int N)
{
	ifstream file;
	file.open(input_path);

	string titleStr, NStr, latitStr, longitStr;

	if (file.is_open()) {
		getline(file, titleStr);
		getline(file, NStr);

		for (int i = 0; i < N; ++i) {
			getline(file, latitStr, ',');
			lats[i] = (T)atof(latitStr.c_str());
			getline(file, longitStr);
			lons[i] = (T)atof(longitStr.c_str());
		}
	}
}

/* Compute the latitud of the second point */
template <typename T>
__device__ T lat_destination_point(T lat, T bearing)
{
	return (DISTANCE / EARTH_RADIUS) / DEG_TO_RAD * cos(bearing);
}

/* Compute the longitud of the second point */
template <typename T>
__device__ T lon_destination_point(T lat, T bearing, T lat_2)
{
	T var = log(tan(M_PI / 4.0 + lat_2 / 2.0) / tan(M_PI / 4.0 + lat / 2.0));
	T q = abs(var) > 10e-12 ? (lat_2 - lat) / var : cos(lat);
	return (DISTANCE / EARTH_RADIUS) / DEG_TO_RAD * sin(bearing) / q;
}

/* Normalise the longitud to <-180, + 180> */
template <typename T>
__device__ T normalizing(T lon)
{
	return fmod(lon + 540.0, 360.0) - 180.0;
}

/* Kernel: find the coordinates for danger zone of each coordinate
/param lats: latitudes of the set of points
/param lons: longitudes of the set of points
/param lats_dz: latitudes of the danger zone
/param lons_dz: longitudes of the danger zone
/param n: Total N of worker threads (size of lats_dz) */
template <typename T>
__global__ void danger_zone(T* lats, T* lons, T* lats_dz, T* lons_dz, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		int bearing = (idx * ANGLE_OFFSET) % TOTAL_DEGREES;
		lats_dz[idx] = lats[idx / 20] + lat_destination_point(lats[idx / 20], bearing * DEG_TO_RAD);
		lons_dz[idx] = normalizing(lons[idx / 20] + lon_destination_point(lats[idx / 20], bearing * DEG_TO_RAD, lats_dz[idx]));
	}
}

int main(void)
{
	// Find the total of points from a file
	int N = 0;
	find_N(N);

	if (N == 0) {
		cerr << "Unable to open file or there isn't data" << endl;
		return 0;
	}
	
	// Utils pointers to host (h_) and device (d_)
	double *h_latitudes, *h_longitudes, *h_lat_buffer, *h_lon_buffer;
	double *d_latitudes, *d_longitudes, *d_lat_buffer, *d_lon_buffer;

	// Total size in bytes
	const int SIZE_POINTS_BYTES = N * sizeof(double);
	const int SIZE_BUFFER_BYTES = NUM_BOUNDS * N * sizeof(double);

	// Allocate CPU memory
	h_latitudes = (double*)malloc(SIZE_POINTS_BYTES);
	h_longitudes = (double*)malloc(SIZE_POINTS_BYTES);
	h_lat_buffer = (double*)malloc(SIZE_BUFFER_BYTES);
	h_lon_buffer = (double*)malloc(SIZE_BUFFER_BYTES);

	// Allocate GPU memory
	cudaMalloc((void**)&d_latitudes, SIZE_POINTS_BYTES);
	cudaMalloc((void**)&d_longitudes, SIZE_POINTS_BYTES);
	cudaMalloc((void**)&d_lat_buffer, SIZE_BUFFER_BYTES);
	cudaMalloc((void**)&d_lon_buffer, SIZE_BUFFER_BYTES);

	// Fill data in host
	load_data(h_latitudes, h_longitudes, N);

	// Transfer data from host to device
	cudaMemcpy(d_latitudes, h_latitudes, SIZE_POINTS_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_longitudes, h_longitudes, SIZE_POINTS_BYTES, cudaMemcpyHostToDevice);

	// Phase 1: Points of polygon 1 inside Polygon 2
	danger_zone<double> << < ceil(NUM_BOUNDS * N / 1024.0), 1024 >> >
		(d_latitudes, d_longitudes, d_lat_buffer, d_lon_buffer, NUM_BOUNDS * N);
	
	cudaMemcpy(h_lat_buffer, d_lat_buffer, SIZE_BUFFER_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_lon_buffer, d_lon_buffer, SIZE_BUFFER_BYTES, cudaMemcpyDeviceToHost);

	// Delete device memory
	cudaFree(d_latitudes), cudaFree(d_latitudes), cudaFree(d_lat_buffer), cudaFree(d_lon_buffer);
	
	return 0;
}