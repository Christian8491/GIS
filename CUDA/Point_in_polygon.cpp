/* Point in polygon algorithm using thrust library on CUDA

Check if a coordinate point belongs or not to a specific polygon

A point in polygon file must contain:
1. title: Latitude,Longitude
2. Total number of points N (of the polygon)
3. The query point (only one point)
4. Coordinates of the spherical polygon (size: N)

Example - quadrilateral polygon:
Latitude,Longitude
4
-12.076450,-77.085721
-12.071903,-77.126787
-12.078513,-77.104394
-12.078973,-77.093252
-12.078521,-77.071940

This code was implemented by Christian Córdova Estrada
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <thrust/device_ptr.h>

using namespace std;

// It contains the latitude ([0]) and longitude ([1]) of the query point
__constant__ double query_point[2];		

/* To check its orientation between two points and the query point
/param p_x: longitud of first point
/param p_y: latitud of first point
/param q_x: longitud of second point
/param q_y: latitud of second point */
template <typename T>
__device__ T orientation(T p_x, T p_y, T q_x, T q_y)
{
	return ((q_y - p_y) * (query_point[1] - q_x) - (q_x - p_x) * (query_point[0] - q_y));
}

/* Kernel for Pip
/param lats: latitudes
/param lons: longitudes
/param in_out: it contains either 0 or 1
/param n: size of points */
template <typename T>
__global__ void point_in_polygon(T* lats, T* lons, int* in_out, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n) {
		int idx_ = (idx + 1) % n;
		if (lons[idx] <= query_point[1] && query_point[1] < lons[idx_]) {
			if (orientation(lons[idx], lats[idx], lons[idx_], lats[idx_]) < 0) in_out[idx] = 1;
		}
		else if (query_point[1] <= lons[idx] && lons[idx_] < query_point[1]) {
			if (orientation(lons[idx_], lats[idx_], lons[idx], lats[idx]) < 0) in_out[idx] = 1;
		}
		else in_out[idx] = 0;
	}
}

// Only print if the query point is inside the spherical polygon or not
void print_result(int sum)
{
	if (sum % 2 == 0) cout << "point is outside the polygon \n" << endl;
	else cout << "Point is inside the polygon \n" << endl;
}


int main(void)
{
	int N;		// Size of the polygon

	// Utils pointers to host (h_) and device (d_)
	double *h_latitudes, *h_longitudes, *h_query_point;
	double *d_latitudes, *d_longitudes;
	int* d_in_or_out;

	// Size total of data in bytes
	const int TOTAL_SIZE_BYTES = N * sizeof(double);

	// Allocate CPU memory
	h_latitudes = (double*)malloc(TOTAL_SIZE_BYTES);
	h_longitudes = (double*)malloc(TOTAL_SIZE_BYTES);
	h_query_point = (double*)malloc(2 * sizeof(double));

	// Allocate GPU memory
	cudaMalloc((void**)&d_latitudes, TOTAL_SIZE_BYTES);
	cudaMalloc((void**)&d_longitudes, TOTAL_SIZE_BYTES);
	cudaMalloc((void**)&d_in_or_out, N * sizeof(int));

	// Some function to fill data in host
	load_data(h_latitudes, h_longitudes, h_query_point, N);

	// Transfer data from host to device
	cudaMemcpy(d_latitudes, h_latitudes, TOTAL_SIZE_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_longitudes, h_longitudes, TOTAL_SIZE_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(query_point, &h_query_point[0], 2 * sizeof(double));

	// Launch the kernel with 1024 threads by block
	point_in_polygon<double> << < ceil(N / 1024.0), 1024 >> > (d_latitudes, d_longitudes, d_in_or_out, N);

	// wrap raw pointer with a device_ptr and reduce (add all values)
	thrust::device_ptr<int> d_in_out_ptr = thrust::device_pointer_cast(d_in_or_out);
	int sum = thrust::reduce(d_in_out_ptr, d_in_out_ptr + N);

	// Delete device memory
	cudaFree(d_latitudes), cudaFree(d_longitudes), cudaFree(d_in_or_out);

	// Delete host memory
	free(h_latitudes), free(h_longitudes), free(h_query_point);

	print_result(sum);

	return 0;
}