/* This code generate random coordinates. Coordinates are into a "squared" bound */

#include <stdlib.h>									// rand
#include <vector>
#include <fstream>									// ofstream
#include <iomanip>									// setprecision
#include <string>									// to_string

using namespace std;

// An arbitrary center is selected (LIMA - PERÚ). Change this values for different data
#define LATITUD -12.001048
#define LONGITUD -76.007067
#define BOUND 0.25					// kilometers
#define TOTAL_POINTS 1024

template <typename T>
struct Coordinate
{
	T latitude;					// [-90.0, 90.0>
	T longitude;					// [-180.0, 180.0>
	Coordinate() {}
	Coordinate(T lat_, T long_) { latitude = lat_, longitude = long_; }
};

// Compute random coordinates
template<typename T>
void generateLatLong(T& lat_, T& lot_)
{
	T minLat = LATITUD - BOUND;
	T maxLat = LATITUD + BOUND;
	T minLon = LONGITUD - BOUND;
	T maxLon = LONGITUD + BOUND;

	float r1 = (T)rand() / (T)RAND_MAX;
	float r2 = (T)rand() / (T)RAND_MAX;

	lat_ = minLat + r1 * (maxLat - minLat);
	lot_ = minLon + r2 * (maxLon - minLon);
}

template<typename T>
void generateRandomCoordinates(vector<Coordinate<T>>& vectorOfPoints)
{
	while (vectorOfPoints.size() < TOTAL_POINTS) {
		T lat_, lon_;
		generateLatLong(lat_, lon_);
		vectorOfPoints.push_back(Coordinate<T>(lat_, lon_));
	}
}

// Save coordinates -- change the path
void SaveCoordinates(vector<Coordinate<double>>& points, string path)
{
	ofstream coordinatesFile;
	coordinatesFile.open(path);

	coordinatesFile << "Latitude,Longitude\n";
	for (size_t i = 0; i < points.size(); ++i) {
		coordinatesFile << fixed << setprecision(6) << points[i].latitude << "," << points[i].longitude;
		coordinatesFile << "\n";
	}
}

// Only to print in console
template<typename T>
void printCoordinates(vector<Coordinate<T>>& points)
{
	for (size_t i = 0; i < points.size(); ++i) {
		printf("latitud: %.6f \t longitud: %.6f \n", points[i].latitude, points[i].longitude);
	}
}

int main()
{
	// Change the path HERE
	string path = string("../../Datasets/Generate_Coordinates_") + to_string(TOTAL_POINTS) + string(".txt");
	vector<Coordinate<double>> vectorOfPoints;
	generateRandomCoordinates(vectorOfPoints);
	SaveCoordinates(vectorOfPoints, path);

	return 0;
}