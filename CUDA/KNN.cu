/* KNN algorithm using thrust library on CUDA

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

#include <iostream>
#include <fstream>
#include <sstream>							// getline()
#include <iomanip>							// setprecision()
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <ctime>

using namespace std;

string input_path = "../../../Datasets/Knn/input/knn_64.txt";		// knn file, put your path here

__constant__ __device__ double DEG_TO_RAD = 0.017453292519943295769;
__constant__ __device__ double EARTH_RADIUS_IN_METERS = 6372797.560856;			// in meters

/* This function reads the coordinates from a file and fill to two host vectors
@param lats: vector in host of latitudes
@param lons: vector in host of longitudes */
template <typename T>
void readCoordinates(thrust::host_vector<T>& latitudes, thrust::host_vector<T>& longitudes, 
	thrust::host_vector<T>& queryPoint, int& k_size, int& n_size)
{
	ifstream coordinatesFile;
	coordinatesFile.open(input_path);

	string latitStr, longitStr, titleStr, NStr, kStr;
	
	if (coordinatesFile.is_open()) {
		getline(coordinatesFile, titleStr);

		getline(coordinatesFile, NStr);
		n_size = (int)atof(NStr.c_str());

		getline(coordinatesFile, kStr);
		k_size = (T)atof(kStr.c_str());

		// For the first point - query point
		getline(coordinatesFile, latitStr, ',');
		queryPoint[0] = (T)atof(latitStr.c_str());

		getline(coordinatesFile, longitStr);
		queryPoint[1] = (T)atof(longitStr.c_str());

		while (getline(coordinatesFile, latitStr, ',')) {
			latitudes.push_back((T)atof(latitStr.c_str()));
			getline(coordinatesFile, longitStr);
			longitudes.push_back((T)atof(longitStr.c_str()));
		}
	}
	else cerr << "Unable to open file" << endl;
}

template <typename T>
void getKNearest(thrust::host_vector<T>& h_lats, thrust::host_vector<T>& h_lons, 
	thrust::host_vector<T>& k_lats, thrust::host_vector<T>& k_lons, thrust::host_vector<T>& h_seq, int& k)
{
	for (size_t i = 0; i < k; ++i) {
		k_lats.push_back(h_lats[h_seq[i]]);
		k_lons.push_back(h_lons[h_seq[i]]);
	}
}

// Object function to Haversine
template <typename T>
struct haversine
{
	T queryLat, queryLon;
	
	haversine(){}
	haversine(const T lat_, const T lon_){ queryLat = lat_, queryLon = lon_; }

	__host__ __device__ T operator() (T lat, T lon)
	{
		T latitudeArc = (lat - queryLat) * DEG_TO_RAD;
		T longitudeArc = (lon - queryLon) * DEG_TO_RAD;
		T latitudeH = sin(latitudeArc * 0.5);
		latitudeH *= latitudeH;
		T lontitudeH = sin(longitudeArc * 0.5);
		lontitudeH *= lontitudeH;
		T tmp = cos(lat * DEG_TO_RAD) * cos(queryLat * DEG_TO_RAD);
		return EARTH_RADIUS_IN_METERS * 2.0 * asin(sqrt(latitudeH + tmp*lontitudeH));
	}
};

int main(void)
{
	// Find the total of points from a file
	clock_t begin = clock();

	// Create and fill host latitudes and longitudes
	thrust::host_vector<double> h_latitudes, h_longitudes, h_queryPoint(2), h_distances;
	int K, N;	// K nearest and total numbers
	readCoordinates(h_latitudes, h_longitudes, h_queryPoint, K, N);

	// Passing latitudes and longitudes to device
	thrust::device_vector<double> d_latitudes = h_latitudes, d_longitudes = h_longitudes;
	thrust::device_vector<double> d_distances(N);

	// Fill distances in device with the Haversine Formula
	haversine<double> functor_haversine(h_queryPoint[0], h_queryPoint[1]);
	thrust::transform(d_latitudes.begin(), d_latitudes.end(), d_longitudes.begin(), d_distances.begin(), functor_haversine);

	// Create a sequence. It work as iterator (indices to latitudes and longitudes)
	thrust::device_vector<int> d_sequence(N);
	thrust::sequence(d_sequence.begin(), d_sequence.end());

	// Sort by distances and the sequence vector must be reordered
	thrust::sort_by_key(d_distances.begin(), d_distances.end(), d_sequence.begin(), thrust::less<int>());

	// Get only first K latitudes and longitudes
	thrust::host_vector<double> h_sequence = d_sequence;
	thrust::host_vector<double> h_nearest_lat, h_nearest_lon;
	getKNearest(h_latitudes, h_longitudes, h_nearest_lat, h_nearest_lon, h_sequence, K);

	// Print Results
	cout << "N size: " << N << endl;
	cout << "K size: " << K << endl;
	cout << "\nK coordinates: " << endl;
	for (int i = 0; i < K; ++i) cout << fixed << setprecision(6) << h_nearest_lat[i] << ", " << h_nearest_lon[i] << endl;

	clock_t end = clock();
	cout << fixed << setprecision(6) << "\nElapsed time: " << double(end - begin) / CLOCKS_PER_SEC << endl;

	return 0;
}