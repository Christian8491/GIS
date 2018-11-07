/* Point in polygon (PIP) algorithm using thrust library on CUDA

Obs 1:
A point in polygon file must contain:
1. title: Latitude, Longitude
2. Total number of points N (int)
3. The center(query) point (only one coordinate)
4. Coordinates of all points (size: N)

Example - quadrilateral polygon:
Latitude,Longitude
4
-12.076450,-77.085721
-12.071903,-77.126787
-12.078513,-77.104394
-12.078973,-77.093252
-12.078521,-77.071940

Obs 2:
Coordinates must form a continuous spherical polygon

   12°------11°	        8°--------7°
  /			  \		   /    	   \
 /			   10°----9°		    6°
1°								   /
 \								  /
  \		   		   				 /
   2°------3°----------4°------5°

Obs 3:
No point should contain the north pole or south pole

*/

#include <iostream>
#include <fstream>
#include <sstream>						// getline()
#include <thrust/device_vector.h>

using namespace std;

string input_path = "../../../Datasets/Point_in_polygon/input/poly_12.txt";	// put your path file here

__constant__ double center[2];		// It contains the latitude and longitude of the center

/* This function reads the size of points
@param N: size of point in the file */
template <typename T>
void find_N(T& N)
{
	ifstream coordinatesFile;
	coordinatesFile.open(input_path);

	string titleStr, NStr;
	if (coordinatesFile.is_open()) {
		getline(coordinatesFile, titleStr);

		getline(coordinatesFile, NStr);
		N = (T)atof(NStr.c_str());

		return;
	}
}

/* This function reads the coordinates from a file and fill to two host vectors
@param lats: vector in host of latitudes
@param lons: vector in host of longitudes */
template <typename T>
void readCoordinates(T*& latitudes, T*& longitudes, T*& queryPoint, int N)
{
	ifstream coordinatesFile;
	coordinatesFile.open(input_path);

	string latitStr, longitStr, titleStr, NStr;

	if (coordinatesFile.is_open()) {
		getline(coordinatesFile, titleStr);
		getline(coordinatesFile, NStr);

		// For the first point - query point
		getline(coordinatesFile, latitStr, ',');
		queryPoint[0] = (T)atof(latitStr.c_str());

		getline(coordinatesFile, longitStr);
		queryPoint[1] = (T)atof(longitStr.c_str());

		for (int i = 0; i < N; ++i) {
			getline(coordinatesFile, latitStr, ',');
			latitudes[i] = (T)atof(latitStr.c_str());
			getline(coordinatesFile, longitStr);
			longitudes[i] = (T)atof(longitStr.c_str());
		}
	}
}

/* Kernel for Pip
@param lats: vector of latitudes
@param lons: vector of longitudes
@param in_out: contains either 0 (not cut) or 1 (cut)
@param n: size of points */
template <typename T>
__global__ void point_in_polygon(T* lats, T* lons, int* in_out ,int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n) {
		if (center[0] > 0) {
			if (lats[idx] >= center[0]) {
				if (lons[idx] <= center[1] && center[1] < lons[(idx + 1)%n]) in_out[idx] = 1;
				else in_out[idx] = 0;
			}
		}
		else {
			if (lats[idx] <= center[0]) {
				if (lons[idx] <= center[1] && center[1] < lons[(idx + 1) % n]) in_out[idx] = 1;
				else in_out[idx] = 0;
			}
		}
	}
}


void printResult(int sum)
{
	if (sum % 2 == 0) cout << "point is outside the polygon \n" << endl;
	else cout << "Point is inside the polygon \n" << endl;
}


int main(void)
{
	// Find the total of points from a file
	int N;
	find_N(N);

	if (!N) {
		cerr << "Unable to open file or there isn't data" << endl;
		return 0;
	}

	// Size total of data
	double *h_latitudes, *h_longitudes, *h_queryPoint;
	double *d_latitudes, *d_longitudes;
	int* d_in_or_out;

	// Data for device
	const int TOTAL_SIZE_BYTES = N * sizeof(double);

	// Allocate CPU memory
	h_latitudes = (double*)malloc(TOTAL_SIZE_BYTES);
	h_longitudes = (double*)malloc(TOTAL_SIZE_BYTES);
	h_queryPoint = (double*)malloc(2 * sizeof(double));

	// Allocate GPU memory
	cudaMalloc((void**)&d_latitudes, TOTAL_SIZE_BYTES);
	cudaMalloc((void**)&d_longitudes, TOTAL_SIZE_BYTES);
	cudaMalloc((void**)&d_in_or_out, N * sizeof(int));
	//cudaMalloc((void**)&d_queryPoint, 2 * sizeof(double));

	// Fill data in host
	readCoordinates(h_latitudes, h_longitudes, h_queryPoint, N);

	// Transfer the array host to the array device
	cudaMemcpy(d_latitudes, h_latitudes, TOTAL_SIZE_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_longitudes, h_longitudes, TOTAL_SIZE_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(center, &h_queryPoint, 2 * sizeof(double));	// CONSTANT MEMORY

	// wrap raw pointer with a device_ptr 
	thrust::device_ptr<double> d_lat_ptr = thrust::device_pointer_cast(d_latitudes);
	thrust::device_ptr<double> d_lon_ptr = thrust::device_pointer_cast(d_longitudes);

	// Launch the kernel
	point_in_polygon<double> << < 1, N >> > (d_latitudes, d_longitudes, d_in_or_out, N);

	int* h_in_or_out = (int*)malloc(N * sizeof(int));
	cudaMemcpy(h_in_or_out, d_in_or_out, N * sizeof(int), cudaMemcpyDeviceToHost);

	// wrap raw pointer with a device_ptr and reduce to sum
	thrust::device_ptr<int> d_in_out_ptr = thrust::device_pointer_cast(d_in_or_out);
	int sum = thrust::reduce(d_in_out_ptr, d_in_out_ptr + N);

	printResult(sum);

	return 0;
}