/* This code compute the distance between two random points (in meters) on Earth */

#include <iostream>
#include <cmath>
#include <stdio.h>
using namespace std;

#define DEG_TO_RAD 0.0174532925199432957
#define EARTH_RADIUS_METERS 6372797.560856

template <typename T>
struct Coordinate
{
    T latitude;
    T longitude;
    Coordinate(){}
    Coordinate(const T lat_, const T long_) { latitude = lat_; longitude = long_; }
};

template <typename T>
T distance(const Coordinate<T>& startPoint, const Coordinate<T>& endPoint) {
    T latitudeArc  = (startPoint.latitude - endPoint.latitude) * DEG_TO_RAD;
    T longitudeArc = (startPoint.longitude - endPoint.longitude) * DEG_TO_RAD;
    T latitudeH = sin(latitudeArc * 0.5);
    latitudeH *= latitudeH;
    T lontitudeH = sin(longitudeArc * 0.5);
    lontitudeH *= lontitudeH;
    T tmp = cos(startPoint.latitude * DEG_TO_RAD) * cos(endPoint.latitude * DEG_TO_RAD);
    return EARTH_RADIUS_METERS * 2.0 * asin(sqrt(latitudeH + tmp * lontitudeH));
}

int main()
{
    Coordinate<double> coordinat1(-12.054251,-77.099688), coordinat2(-12.035733,-77.098549);
    printf("Distance en meters: %.6f", distance(coordinat1, coordinat2));

    return 0;
}