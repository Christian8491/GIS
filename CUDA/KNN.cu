/* KNN algorithm using thrust library on CUDA (version with Kernel)

Obs:
A knn file might contain:
1. title: Latitude, Longitude
2. N: Total number of reference points
3. K: Number for K nearest points
4. Coordinates of the query point (only one coordinate)
5. Coordinates of all reference points (size: N)

Example of file:
Latitude,Longitude
8
3
-12.054251,-77.099688
-12.078973,-77.093252
-12.067450,-77.078543
-12.043947,-77.098275
-12.058031,-77.073309
-12.029678,-77.082271
-12.068602,-77.075530
-12.036418,-77.096255
-12.060808,-77.126168
*/

#include <fstream>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>

using namespace std;

// put your path file here
string input_path = "../../../Datasets/Knn/input/knn_500k_750.txt";

__constant__ double DEG_TO_RAD = 0.01745329251994329;
__constant__ double EARTH_RADIUS = 6372797.560856;		// in meters

// It contains the latitude and longitude of the query point
__constant__ double query_point[2];

template <typename T>
void find_K_N(T& N, T& K)
{
	ifstream coordinatesFile;
	coordinatesFile.open(input_path);

	string titleStr, NStr, kStr;
	if (coordinatesFile.is_open()) {
		getline(coordinatesFile, titleStr);

		getline(coordinatesFile, NStr);
		N = (T)atof(NStr.c_str());

		getline(coordinatesFile, kStr);
		K = (T)atof(kStr.c_str());
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

	string latitStr, longitStr, titleStr, NStr, kStr;

	if (coordinatesFile.is_open()) {
		getline(coordinatesFile, titleStr);
		getline(coordinatesFile, NStr);
		getline(coordinatesFile, kStr);

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

// Find the K nearest coordinates
template <typename T>
void getKNearest(T*&  h_lats, T*&  h_lons, T*&  k_lats, T*&  k_lons, int*&  h_seq, int& k)
{
	for (size_t i = 0; i < k; ++i) {
		k_lats[i] = h_lats[h_seq[i]];
		k_lons[i] = h_lons[h_seq[i]];
	}
}

// With Haversine Formula
template <typename T>
__global__ void computeDistance(T* lats, T* lons, T* distances, int n)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < n) {
		T lat_variation = (lats[idx] - query_point[0]) * DEG_TO_RAD;
		T lon_variation = (lons[idx] - query_point[1]) * DEG_TO_RAD;
		T partial_1 = sin(lat_variation * 0.5);
		partial_1 *= partial_1;
		T partial_2 = sin(lon_variation * 0.5);
		partial_2 *= partial_2;
		T tmp = cos(lats[idx] * DEG_TO_RAD) * cos(query_point[0] * DEG_TO_RAD);
		distances[idx] = 2.0 * EARTH_RADIUS * asin(sqrt(partial_1 + tmp*partial_2));
	}
}

int main(void)
{
	// Find the total of points from a file
	int N, K;
	find_K_N(N, K);

	if (!N) {
		cerr << "Unable to open file" << endl;
		return 0;
	}

	// Util pointers to host (h_) and device (_d)
	double *h_latitudes, *h_longitudes, *h_query_point, *h_distances;
	int* h_sequence;
	double *d_latitudes, *d_longitudes, *d_distances;

	// Total size in bytes
	const int TOTAL_SIZE_BYTES = N * sizeof(double);

	// Allocate CPU memory
	h_latitudes = (double*)malloc(TOTAL_SIZE_BYTES);
	h_longitudes = (double*)malloc(TOTAL_SIZE_BYTES);
	h_distances = (double*)malloc(TOTAL_SIZE_BYTES);
	h_sequence = (int*)malloc(TOTAL_SIZE_BYTES);
	h_query_point = (double*)malloc(2 * sizeof(double));

	// Allocate GPU memory
	cudaMalloc((void**)&d_latitudes, TOTAL_SIZE_BYTES);
	cudaMalloc((void**)&d_longitudes, TOTAL_SIZE_BYTES);
	cudaMalloc((void**)&d_distances, TOTAL_SIZE_BYTES);

	// Fill data in host
	readCoordinates(h_latitudes, h_longitudes, h_query_point, N);

	// Transfer the array host to the array device
	cudaMemcpy(d_latitudes, h_latitudes, TOTAL_SIZE_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_longitudes, h_longitudes, TOTAL_SIZE_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(query_point, &h_query_point[0], 2 * sizeof(double));

	// Fill distances in device with the Haversine Formula
	computeDistance << < ceil(N/1024.0), 1024 >> > (d_latitudes, d_longitudes, d_distances, N);

	// wrap raw pointer with a device_ptr 
	thrust::device_ptr<double> d_dist_ptr = thrust::device_pointer_cast(d_distances);

	// Create a sequence. It work as iterator (indices to latitudes and longitudes)
	thrust::device_ptr<int> d_sequence_ptr = thrust::device_malloc<int>(N);
	thrust::sequence(d_sequence_ptr, d_sequence_ptr + N);

	// Sort by distances and the sequence vector must be reordered
	thrust::sort_by_key(d_dist_ptr, d_dist_ptr + N, d_sequence_ptr, thrust::less<int>());

	// Get only first K latitudes and longitudes
	double *h_nearest_lat, *h_nearest_lon;
	h_nearest_lat = (double*)malloc(K * sizeof(double));
	h_nearest_lon = (double*)malloc(K * sizeof(double));

	// Transfer the array host to the array device
	cudaMemcpy(h_sequence, thrust::raw_pointer_cast(d_sequence_ptr), K * sizeof(int), cudaMemcpyDeviceToHost);

	// Find the K nearest latitudes and longitudes
	getKNearest(h_latitudes, h_longitudes, h_nearest_lat, h_nearest_lon, h_sequence, K);

	// Delete host memory
	free(h_latitudes), free(h_longitudes), free(h_distances), free(h_query_point), free(h_sequence);

	// Delete device memory
	cudaFree(d_latitudes), cudaFree(d_longitudes), cudaFree(d_distances), thrust::device_free(d_sequence_ptr);

	return 0;
}