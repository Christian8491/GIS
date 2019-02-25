/* Nearest from a Radius (like K-nearest neighbor) algorithm using thrust library on CUDA 

Obs:
A Nearest from a Radius must contain:
1. title: Latitude, Longitude
2. N: Total number of reference points
3. Coordinates of the query point (only one coordinate)
4. Coordinates of all others points (size: N)

Example of file:
Latitude,Longitude
5
-12.054251,-77.099688
-12.078973,-77.093252
-12.067450,-77.078543
-12.043947,-77.098275
-12.058031,-77.073309
-12.029678,-77.082271

This code was implemented by Christian Córdova Estrada
*/

__constant__ double DEG_TO_RAD = 0.01745329251994329;
__constant__ double EARTH_RADIUS = 6372797.560856;		// in meters
__constant__ double radius = 20000;	// search inside this radius
__constant__ double d_query_point[2];	// Latitude and longitude of the query point

// To thrust::partition
struct correct_indexes
{
	__host__ __device__ bool operator()(const int &x)
	{
		return x != -1;
	}
};

// Using Haversine Formula
template <typename T>
__device__ T haversineFormula(T lat, T lon)
{
	T lat_variation = (lat - d_query_point[0]) * DEG_TO_RAD;
	T lon_variation = (lon - d_query_point[1]) * DEG_TO_RAD;
	T partial_1 = sin(lat_variation * 0.5);
	partial_1 *= partial_1;
	T partial_2 = sin(lon_variation * 0.5);
	partial_2 *= partial_2;
	T tmp = cos(d_query_point[0] * DEG_TO_RAD) * cos(lat * DEG_TO_RAD);
	return 2.0 * EARTH_RADIUS * asin(sqrt(partial_1 + tmp * partial_2));
}

// Kernel to find correct indexes
template <typename T>
__global__ void findIndexes(T* lats, T* lons, int* indexes_pos, int n)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < n) {
		if (haversineFormula(lats[idx], lons[idx]) <= radius) {
			indexes_pos[idx] = idx;
		} else {
			indexes_pos[idx] = -1;
		}
	}
}

int main(void)
{
	int N;    // Total of coordinates

	// Util pointers to host (h_) and device (_d)
	double *h_latitudes, *h_longitudes, *h_query_point;
	int *h_positions;
	double *d_latitudes, *d_longitudes;
	int *d_positions;

	// Total size in bytes
	const int TOTAL_SIZE_BYTES = N * sizeof(double);

	// Allocate CPU memory
	h_latitudes = (double*)malloc(TOTAL_SIZE_BYTES);
	h_longitudes = (double*)malloc(TOTAL_SIZE_BYTES);
	h_query_point = (double*)malloc(2 * sizeof(double));

	// Allocate GPU memory
	cudaMalloc((void**)&d_latitudes, TOTAL_SIZE_BYTES);
	cudaMalloc((void**)&d_longitudes, TOTAL_SIZE_BYTES);
	cudaMalloc((void**)&d_positions, N * sizeof(int));

	// Fill data in host
	readCoordinates(h_latitudes, h_longitudes, h_query_point, N);

	// Transfer the coordinates' array from host to the device
	cudaMemcpy(d_latitudes, h_latitudes, TOTAL_SIZE_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_longitudes, h_longitudes, TOTAL_SIZE_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_query_point, &h_query_point[0], 2 * sizeof(double));

	// Fill distances in device with the Haversine Formula
	findIndexes << < ceil(N / 1024.0), 1024 >> > (d_latitudes, d_longitudes, d_positions, N);

	// partition and transfer back to host
	// An util device_pointer and use partition (ordered like {1, 5, 8, 11, -1, -1, ..., -1})
	thrust::device_ptr<int> d_dist_ptr = thrust::device_pointer_cast(d_positions);
	int size_end_points = thrust::partition(d_dist_ptr, d_dist_ptr + N, correct_indexes());

	// Final size in bytes
	const int FINAL_SIZE_BYTES = size_end_points * sizeof(int);

	// Allocate CPU memory, end coordinates and transfer data from device to host
	h_positions = (int*)malloc(FINAL_SIZE_BYTES);
	cudaMemcpy(h_positions, thrust::raw_pointer_cast(d_dist_ptr), FINAL_SIZE_BYTES, cudaMemcpyDeviceToHost);

	// Delete host memory
	free(h_latitudes), free(h_longitudes), free(h_query_point);

	// Delete device memory
	cudaFree(d_latitudes), cudaFree(d_longitudes), cudaFree(d_positions);

	return 0;
}
