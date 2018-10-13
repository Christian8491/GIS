/* This code generate a random vector of coordinates
   Coordinates are into a squared bound */

#include <stdlib.h>												// rand
#include <vector>
#include <fstream>												// ofstream
#include <iomanip>												// setprecision

using namespace std;

// An specific center is selected (LIMA - PERÚ)
#define LATITUD -12.049048
#define LONGITUD -77.097067
#define BOUND 0.03

#define TOTAL_POINTS 64

template <typename T>
struct Coordinate
{
	T latitude;				// [-90.0, 90.0>
	T longitude;			// [-180.0, 180.0>
	Coordinate() {}
	Coordinate(T lat_, T long_) { latitude = lat_, longitude = long_; }
};

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

template<typename T>
void printCoordinates(vector<Coordinate<T>>& vectorOfPoints)
{
	for (size_t i = 0; i < vectorOfPoints.size(); ++i) {
		printf("latitud: %.6f \t longitud: %.6f \n", vectorOfPoints[i].latitude, vectorOfPoints[i].longitude);
	}
}

/* Save coordinates -- change the path */
void SaveCoordinates(vector<Coordinate<double>>& vectorOfPoints)
{
	ofstream coordinatesFile;
	coordinatesFile.open("Write_your_path_here/Generate_Coordinates.txt");

	coordinatesFile << "Latitude,Longitude\n";
	for (size_t i = 0; i < vectorOfPoints.size(); ++i) {
		coordinatesFile << fixed << setprecision(6) << vectorOfPoints[i].latitude << "," << vectorOfPoints[i].longitude;
		coordinatesFile << "\n";
	}
}

int main()
{
	vector<Coordinate<double>> vectorOfPoints;
	generateRandomCoordinates(vectorOfPoints);
	SaveCoordinates(vectorOfPoints);

	return 0;
}