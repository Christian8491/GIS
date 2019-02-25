/* Perimeter of a spherical polygon using thrust library on CUDA

Obs:
A Perimeter of a spherical polygon file must contain:
1. title: Latitude,Longitude
2. N: Total number of points (polygon) with repetition (first = last)
4. Coordinates of the points

Example of file:
Latitude,Longitude
5
-12.054251,-77.099688
-12.078973,-77.093252
-12.067450,-77.078543
-12.043947,-77.098275
-12.054251,-77.099688

This code was implemented by Christian Córdova Estrada
*/

__constant__ double DEG_TO_RAD = 0.01745329251994329;
__constant__ double EARTH_RADIUS = 6372797.560856;		// in meters

// Using Haversine Formula
template <typename T>
__global__ void computePerimeter(T* lats, T* lons, T* distances, int n)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < n - 1) {
		T lat_variation = (lats[idx + 1] - lats[idx]) * DEG_TO_RAD;
		T lon_variation = (lons[idx + 1] - lons[idx]) * DEG_TO_RAD;
		T partial_1 = sin(lat_variation * 0.5);
		partial_1 *= partial_1;
		T partial_2 = sin(lon_variation * 0.5);
		partial_2 *= partial_2;
		T tmp = cos(lats[idx] * DEG_TO_RAD) * cos(lats[idx + 1] * DEG_TO_RAD);
		distances[idx] = 2.0 * EARTH_RADIUS * asin(sqrt(partial_1 + tmp * partial_2));
	}
}

int main(void)
{
	int N;    // polygon size (total of coordinates)
  
	// Util pointers to host (h_) and device (_d)
	double *h_latitudes, *h_longitudes, h_perimeter;
	double *d_latitudes, *d_longitudes, *d_distances;

	// Total size in bytes
	const int TOTAL_SIZE_BYTES = N * sizeof(double);

	// Allocate CPU memory
	h_latitudes = (double*)malloc(TOTAL_SIZE_BYTES);
	h_longitudes = (double*)malloc(TOTAL_SIZE_BYTES);

	// Allocate GPU memory
	cudaMalloc((void**)&d_latitudes, TOTAL_SIZE_BYTES);
	cudaMalloc((void**)&d_longitudes, TOTAL_SIZE_BYTES);
	cudaMalloc((void**)&d_distances, TOTAL_SIZE_BYTES);

	// A simple function to Fill data in host
	readCoordinates(h_latitudes, h_longitudes, N);

	// Transfer the coordinates' array from host to the device
	cudaMemcpy(d_latitudes, h_latitudes, TOTAL_SIZE_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_longitudes, h_longitudes, TOTAL_SIZE_BYTES, cudaMemcpyHostToDevice);

	// Fill distances in device with the Haversine Formula
	computePerimeter << < ceil(N / 1024.0), 1024 >> > (d_latitudes, d_longitudes, d_distances, N);

	// wrap raw pointer with a device_ptr and reduce (add all values)
	thrust::device_ptr<double> d_dist_ptr = thrust::device_pointer_cast(d_distances);
	h_perimeter = thrust::reduce(d_dist_ptr, d_dist_ptr + N);

	// Show the perimeter
	printf("Perimeter of the polygon in meters: %.8f\n", h_perimeter);

	// Delete host memory
	free(h_latitudes), free(h_longitudes);

	// Delete device memory
	cudaFree(d_latitudes), cudaFree(d_longitudes), cudaFree(d_distances);

	return 0;
}
