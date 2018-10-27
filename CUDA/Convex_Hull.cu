/* Based on - CudaChain: A Practical GPU-accelerated 2D Convex Hull Algorithm by Gang Mei.
   See this paper here: https://arxiv.org/ftp/arxiv/papers/1508/1508.05488.pdf 

   First step: It's necessary to load data from a file */

#include <iostream>
#include <fstream>
#include <sstream>						// getline()
#include <iomanip>						// setprecision()
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>				// min_element() & max_element()
#include <thrust/tuple.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include "device_launch_parameters.h"
#include <ctime>

using namespace std;

__device__ const int N_EXTREMES = 4;

template<typename T>
struct Coordinate
{
	T latitude;							// [-90.0, 90.0>
	T longitude;						// [-180.0, 180.0>
	Coordinate(){}
	Coordinate(T lat_, T long_) { latitude = lat_; longitude = long_; }
};

// function objects to thrust::partition
struct region_0
{
	__host__ __device__ bool operator()(const thrust::tuple<double, double, int> &x) { 
		return  thrust::get<2>(x) != 0;
	}
};

struct region_1
{
	__host__ __device__ bool operator()(const thrust::tuple<double, double, int> &x) { 
		return thrust::get<2>(x) == 1;
	}
};

struct region_2
{
	__host__ __device__ bool operator()(const thrust::tuple<double, double, int> &x){ 
		return thrust::get<2>(x) == 2;
	}
};

struct region_3
{
	__host__ __device__ bool operator()(const thrust::tuple<double, double, int> &x) { 
		return thrust::get<2>(x) == 3;
	}
};

struct my_greater
{
	__host__ __device__ bool operator()(const thrust::tuple<double, double, int> &x) {
		
	return thrust::get<0>(x) != thrust::get<1>(x);
	}
};

/* This kernel will save positions in diferents regions R0, R1, R2, R3 or R4
@param ext_lat: pointer to an array of extrema latitudes
@param ext_lon: pointer to an array of extrema longitudes
@param lat: pointer to an array of all latitudes
@param lon: pointer to an array of all longitudes
@param pos: empty pointer that will save positions */
template<typename T>
__global__ void distributePoints(T* ext_lat, T* ext_lon, T* lat, T* lon, int* pos, int n)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	// find P and Q extreme points of each region  (a curve is approximated to a straight)
	if (idx < n) {
		double extremsPQ[N_EXTREMES * 2];
		for (int i = 0; i < N_EXTREMES; i++) {
			extremsPQ[2 * i] = (ext_lat[(i + 1) % N_EXTREMES] - ext_lat[i]) / (ext_lon[(i + 1) % N_EXTREMES] - ext_lon[i]);
			extremsPQ[2 * i + 1] = ext_lat[i] - extremsPQ[2 * i] * ext_lon[i];
		}

		if (lat[idx] < extremsPQ[0] * lon[idx] + extremsPQ[1]) pos[idx] = 1;
		else if (lat[idx] < extremsPQ[2] * lon[idx] + extremsPQ[3]) pos[idx] = 2;
		else if (lat[idx] > extremsPQ[4] * lon[idx] + extremsPQ[5]) pos[idx] = 3;
		else if (lat[idx] > extremsPQ[6] * lon[idx] + extremsPQ[7]) pos[idx] = 4;
		else pos[idx] = 0;
	}
}

template<typename T>
__host__ void reducePointsR1(thrust::host_vector<T>& latsReduce, thrust::host_vector<T>& lonsReduce, 
	thrust::host_vector<T> lat, thrust::host_vector<T> lon, int first, int last)
{
	T lat_0 = lat[first];
	for (int i = first; i < last; ++i) {
		if (lat[i] <= lat_0) {
			latsReduce.push_back(lat[i]);
			lonsReduce.push_back(lon[i]);
			lat_0 = lat[i];
		}
	}
}

template<typename T>
__host__ void reducePointsR2(thrust::host_vector<T>& latsReduce, thrust::host_vector<T>& lonsReduce,
	thrust::host_vector<T> lat, thrust::host_vector<T> lon, int first, int last)
{
	T lon_0 = lon[first];
	for (int i = first; i < last; ++i) {
		if (lon[i] >= lon_0) {
			latsReduce.push_back(lat[i]);
			lonsReduce.push_back(lon[i]);
			lon_0 = lon[i];
		}
	}
}

template<typename T>
__host__ void reducePointsR3(thrust::host_vector<T>& latsReduce, thrust::host_vector<T>& lonsReduce,
	thrust::host_vector<T> lat, thrust::host_vector<T> lon, int first, int last)
{
	T lat_0 = lat[first];
	for (int i = first; i <= last; ++i) {
		if (lat[i] >= lat_0) {
			latsReduce.push_back(lat[i]);
			lonsReduce.push_back(lon[i]);
			lat_0 = lat[i];
		}
	}
}

template<typename T>
__host__ void reducePointsR4(thrust::host_vector<T>& latsReduce, thrust::host_vector<T>& lonsReduce,
	thrust::host_vector<T> lat, thrust::host_vector<T> lon, int first, int last)
{
	T lon_0 = lon[first];
	for (int i = first; i <= last; ++i) {
		if (lon[i] <= lon_0) {
			latsReduce.push_back(lat[i]);
			lonsReduce.push_back(lon[i]);
			lon_0 = lon[i];
		}
	}
}

/* This function reads the coordinates from a file and fill to two host vectors
@param lats: vector in host of latitudes
@param lons: vector in host of longitudes */
template <typename T>
__host__ void readCoordinates(thrust::host_vector<T>& lats, thrust::host_vector<T>& lons)
{
	ifstream coordinatesFile;
	coordinatesFile.open("../../../Datasets/Generate_Coordinates.txt");		// This path must be changed

	string latit, longit, title;

	if (coordinatesFile.is_open()) {
		getline(coordinatesFile, title);

		while (getline(coordinatesFile, latit, ',')) {
			lats.push_back((T)atof(latit.c_str()));
			getline(coordinatesFile, longit);
			lons.push_back((T)atof(longit.c_str()));
		}
	}
	else cerr << "Unable to open file" << endl;
}

/* Save extrema coordinates (four) into a path 
@param lats: vector in device of latitudes
@param lons: vector in device of longitudes */
template<typename T>
__host__ void saveExtremaPoints(thrust::device_vector<T>& lats, thrust::device_vector<T>& lons)
{
	thrust::host_vector<T> h_lat = lats, h_lon = lons;

	ofstream ExtremaPointsFile;
	ExtremaPointsFile.open("../../../Datasets/Extrema_4_points.txt");
	ExtremaPointsFile << "Latitude,Longitude\n";
	for (size_t i = 0; i < h_lat.size(); ++i) {
		ExtremaPointsFile << fixed << setprecision(6) << h_lat[i] << "," << h_lon[i];
		ExtremaPointsFile << "\n";
	}
}

__host__ thrust::host_vector<int> findBeginsOfRegions(thrust::device_vector<int>& d_positions)
{
	int N = 4;
	thrust::host_vector<int> h_positions = d_positions;
	thrust::host_vector<int> h_begins(N);

	int k = 0, value = 1, pos = 0;
	while (pos < N) {
		while (h_positions[k] == value) k++;
		h_begins[pos++] = k;
		value++;
	}
 	return h_begins;
}

int main(void)
{
	// Fill host latitud and longitud vectors
	thrust::host_vector<double> h_latitudes, h_longitudes;
	readCoordinates(h_latitudes, h_longitudes);

	// Fill device latitud and longitud vectors
	thrust::device_vector<double> d_latitudes = h_latitudes, d_longitudes = h_longitudes;

	// Find the extremes 4 (northest, southest, westhest and easthest)
	typedef thrust::device_vector<double>::iterator doubleIterator;
	doubleIterator minLat = thrust::min_element(d_latitudes.begin(), d_latitudes.end());
	doubleIterator maxLat = thrust::max_element(d_latitudes.begin(), d_latitudes.end());
	doubleIterator minLon = thrust::min_element(d_longitudes.begin(), d_longitudes.end());
	doubleIterator maxLon = thrust::max_element(d_longitudes.begin(), d_longitudes.end());
	
	// Find the extrema 4 coordinates
	thrust::device_vector<double> d_extreme_lat(4), d_extreme_lon(4);

	d_extreme_lat[0] = d_latitudes[minLon - d_longitudes.begin()]; 
	d_extreme_lat[1] = *minLat;
	d_extreme_lat[2] = d_latitudes[maxLon - d_longitudes.begin()];
	d_extreme_lat[3] = *maxLat;

	d_extreme_lon[0] = *minLon;
	d_extreme_lon[1] = d_longitudes[minLat - d_latitudes.begin()];
	d_extreme_lon[2] = *maxLon; 
	d_extreme_lon[3] = d_longitudes[maxLat - d_latitudes.begin()];
	
	// Indicate in which region must be each coordinate
	int n = h_latitudes.size();
	thrust::device_vector<int> d_positions(n);
	double* d_extreme_lat_ptr  = thrust::raw_pointer_cast(&d_extreme_lat[0]);
	double* d_extreme_lon_ptr = thrust::raw_pointer_cast(&d_extreme_lon[0]);
	double* d_lat_ptr = thrust::raw_pointer_cast(&d_latitudes[0]);
	double* d_lon_ptr = thrust::raw_pointer_cast(&d_longitudes[0]);
	int* d_position_ptr = thrust::raw_pointer_cast(&d_positions[0]);

	// Launch the kernel (for the moment only we use one block)
	distributePoints<double> << < 1, n >> > (d_extreme_lat_ptr, d_extreme_lon_ptr, d_lat_ptr, d_lon_ptr, d_position_ptr, n);

	// Copy back pointer of positions to device positions
	thrust::device_ptr<int> d_aux_pos(d_position_ptr);
	d_positions = thrust::device_vector<int>(d_aux_pos, d_aux_pos + n);

	// Create some useful zip iterators
	typedef thrust::device_vector<int>::iterator intIterator;
	typedef thrust::tuple<doubleIterator, doubleIterator, intIterator> tupleIterator;
	typedef thrust::zip_iterator<tupleIterator> zipIterator;

	zipIterator firstIt(thrust::make_tuple(d_latitudes.begin(), d_longitudes.begin(), d_positions.begin()));
	zipIterator lastIt(thrust::make_tuple(d_latitudes.end(), d_longitudes.end(), d_positions.end()));

	// Gather data [1,..,1, 2,..,2, 3,..,3, 4,..,4, 0,..,0] and find bound iterators
	zipIterator firstR0 = thrust::partition(firstIt, lastIt, region_0());
	zipIterator firstR2 = thrust::partition(firstIt, firstR0, region_1());
	zipIterator firstR3 = thrust::partition(firstR2, firstR0, region_2());
	zipIterator firstR4 = thrust::partition(firstR3, firstR0, region_3());

	// find positions where begin each region and next sort regions
	thrust::host_vector<int> h_begins_reg = findBeginsOfRegions(d_positions);
	thrust::sort_by_key(d_longitudes.begin(), d_longitudes.begin() + h_begins_reg[0], d_latitudes.begin(), thrust::less<double>());
	thrust::sort_by_key(d_latitudes.begin() + h_begins_reg[0], d_latitudes.begin() + h_begins_reg[1], d_longitudes.begin() + h_begins_reg[0], thrust::less<double>());
	thrust::sort_by_key(d_longitudes.begin() + h_begins_reg[1], d_longitudes.begin() + h_begins_reg[2], d_latitudes.begin() + h_begins_reg[1], thrust::greater<double>());
	thrust::sort_by_key(d_latitudes.begin() + h_begins_reg[2], d_latitudes.begin() + h_begins_reg[3], d_longitudes.begin() + h_begins_reg[2], thrust::greater<double>());
	
	// These vectors will contain  the ends latitudes and longitudes (simple polygon)
	thrust::host_vector<double> lat_finals, lon_finals;
	thrust::host_vector<double> lat_back = d_latitudes, lon_back = d_longitudes;

	// These functions discard some inner points and form a single polygon (SPA)
	reducePointsR1(lat_finals, lon_finals, lat_back, lon_back, 0, h_begins_reg[0]);
	reducePointsR2(lat_finals, lon_finals, lat_back, lon_back, h_begins_reg[0], h_begins_reg[1]);
	reducePointsR3(lat_finals, lon_finals, lat_back, lon_back, h_begins_reg[1], h_begins_reg[2]);
	reducePointsR4(lat_finals, lon_finals, lat_back, lon_back, h_begins_reg[2], h_begins_reg[3]);

	// Do Convex Hull on CPU because they are few points now



	return 0;
}