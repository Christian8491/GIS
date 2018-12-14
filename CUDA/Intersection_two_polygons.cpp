/* INTERSECTION OF TWO POLYGONS using CUDA

A Intersection file should contain:
1. title: Latitude, Longitude
2. Total numbers of points for each polygon N1,N2
4. Coordinates of the first polygon (size: N1)
4. Coordinates of the second polygon (size: N2)

Example:
Latitude,Longitude
4,3
-12.054251,-77.099688
-12.078973,-77.093252
-12.067450,-77.078543
-12.043947,-77.098275
-12.058031,-77.073309
-12.029678,-77.082271
-12.068602,-77.075530

This code was implemented by Christian Córdova Estrada
*/

#include <iostream>
#include <thrust/device_vector.h>
#include <fstream>

using namespace std;

string input_path = "../../../Datasets/intersection/input/polys_1.txt";	// put your path files here

/* This function find the size of coordinates from two files
/param N1: N of coordinates of the polygon 1
/param N2: N of coordinates of the polygon 2 */
template <typename T>
void find_N1_N2(T& N1, T& N2)
{
	ifstream file;
	file.open(input_path);

	string titleStr, NStr;

	if (file.is_open()) {
		getline(file, titleStr);

		getline(file, NStr, ',');
		N1 = (T)atof(NStr.c_str());
		getline(file, NStr, ',');
		N2 = (T)atof(NStr.c_str());

		return;
	}
}

/* This function reads the coordinates from two files and fill data
/param lats_1: latitudes of the polygon 1
/param lons_1: longitudes of the polygon 1
/param lats_2: latitudes of the polygon 2
/param lons_2: longitudes of the polygon 2
/param N1: size of points for the polygon 1
/param N2: size of points for the polygon 2 */
template <typename T>
void load_data(T*& lats_1, T*& lons_1, T*& lats_2, T*& lons_2, int N1, int N2)
{
	ifstream file_1;
	file_1.open(input_path);

	string titleStr, NStr, latitStr, longitStr;

	if (file_1.is_open()) {
		getline(file_1, titleStr);
		getline(file_1, NStr);

		for (int i = 0; i < N1; ++i) {
			getline(file_1, latitStr, ',');
			lats_1[i] = (T)atof(latitStr.c_str());
			getline(file_1, longitStr);
			lons_1[i] = (T)atof(longitStr.c_str());
		}

		for (int i = 0; i < N2; ++i) {
			getline(file_1, latitStr, ',');
			lats_2[i] = (T)atof(latitStr.c_str());
			getline(file_1, longitStr);
			lons_2[i] = (T)atof(longitStr.c_str());
		}
	}
}

/* To check its orientation between two points and the query point
/param p_x: longitud of first point
/param p_y: latitud of first point
/param q_x: longitud of second point
/param q_y: latitud of second point */
template <typename T>
__device__ T orientation(T query_x, T query_y, T p_x, T p_y, T q_x, T q_y)
{
	return ((q_y - p_y) * (query_y - q_x) - (q_x - p_x) * (query_x - q_y));
}

/* Kernel: find the coordinates of Polygon 1 inside Polygon 2
/param lats_1: latitudes of the first polygon
/param lons_1: longitudes of the first polygon
/param lats_2: latitudes of the second polygon
/param lons_2: longitudes of the second polygon
/param in_out: it contains integer values (odd: inside, even: outside)
/param int_lat: latitudes of points inside
/param int_lon: longitudes of points inside
/param n1: size of coordinates from Polygon 1
/param n2: size of coordinates from Polygon 2 */
template <typename T>
__global__ void points_polygon_inside(T* lats_1, T* lons_1, T* lats_2, T* lons_2, int* in_out, T* int_lat, T* int_lon, int n1, int n2)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n1 - 1) {
		int_lat[idx] = int_lon[idx] = 0.0;
		in_out[idx] = 0;

		for (int i = 0; i < n2 - 1; i++) {
			if (lons_2[i] <= lons_1[idx] && lons_1[idx] < lons_2[i + 1]) {
				if (orientation(lats_1[idx], lons_1[idx], lons_2[i], lats_2[i], lons_2[i + 1], lats_2[i + 1]) < 0) {
					in_out[idx] += 1;
				}
			}
			else if (lons_1[idx] <= lons_2[i] && lons_2[i+1] < lons_1[idx]) {
				if (orientation(lats_1[idx], lons_1[idx], lons_2[i + 1], lats_2[i + 1], lons_2[i], lats_2[i]) < 0) {
					in_out[idx] += 1;
				}
			}
		}

		if (in_out[idx] % 2 != 0) {
			int_lat[idx] = lats_1[idx];
			int_lon[idx] = lons_1[idx];
		}
	}
}

/* Convert a 3D vector from a lat, lon coordinate */
__device__ double3 to_vector(double lat, double lon)
{
	double3 e;
	e.x = cos(lon) * cos(lat);
	e.y = sin(lon) * cos(lat);
	e.z = sin(lat);
	return e;
}

/* Compute the subtraction of two 3D vectors*/
__device__ double3 rest_vector(double3 p1, double3 p2)
{
	double3 r;
	r.x = p2.x - p1.x;
	r.y = p2.y - p1.y;
	r.z = p2.z - p1.z;
	return r;
}

/* Compute the cross product of two 3D vectors */
__device__ double3 cross_product(double3 e1, double3 e2)
{
	double3 e;
	e.x = e1.y * e2.z - e2.y * e1.z;
	e.y = e1.z * e2.x - e2.z * e1.x;
	e.z = e1.x * e2.y - e1.y * e2.x;
	return e;
}

/* Compute the norm of a 3D vector */
__device__ double norm(double3 e)
{
	double norm = sqrt(e.x * e.x + e.y * e.y + e.z * e.z);
	return norm;
}

/* Find a 3D vector from a initial p1 3D vector with a direction e */
__device__ double3 find_intersection(double3 p1, double3 e, double r)
{
	double3 p;
	p.x = p1.x + e.x * r;
	p.y = p1.y + e.y * r;
	p.z = p1.z + e.z * r;
	return p;
}

/* Kernel: find the coordinates of Polygon 2 inside Polygon 1*/
template <typename T>
__global__ void intersection_points(T* lats_1, T* lons_1, T* lats_2, T* lons_2, T* int_lat, T* int_lon, int n1, int n2)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n1) {
		int_lat[idx] = int_lon[idx] = 0.0;
		double3 p1 = to_vector(lats_1[idx], lons_1[idx]);
		double3 p2 = to_vector(lats_1[idx + 1], lons_1[idx + 1]);
		double3 p3, p4, e, h1, k1;
	
		for (int i = 0; i < n2 - 1; ++i) {
			p3 = to_vector(lats_2[i], lons_2[i]);
			p4 = to_vector(lats_2[i + 1], lons_2[i + 1]);

			e = rest_vector(p1, p2);

			h1 = cross_product(rest_vector(p3, p4), rest_vector(p1, p3));
			k1 = cross_product(rest_vector(p3, p4), rest_vector(p1, p2));

			double h = norm(h1), k = norm(k1);

			double3 final_p = find_intersection(p1, e, (h / k));

			double lat_ = asin(final_p.z / sqrt(final_p.x * final_p.x + final_p.y * final_p.y + final_p.z*final_p.z));
			double lon_ = atan2(final_p.y, final_p.x);

			bool on_curve_1 = ((fmin(p1.x, p2.x) < final_p.x) && (final_p.x < fmax(p1.x, p2.x))
				&& (fmin(p1.y, p2.y) < final_p.y) && (final_p.y < fmax(p1.y, p2.y)));

			bool on_curve_2 = ((fmin(p3.x, p4.x) < final_p.x) && (final_p.x < fmax(p3.x, p4.x))
				&& (fmin(p3.y, p4.y) < final_p.y) && (final_p.y < fmax(p3.y, p4.y)));

			if (on_curve_1 && on_curve_2) {
				int_lat[idx] = lat_;
				int_lon[idx] = lon_;
				i = n2 - 1;
			}
		}
	}
}

// Push final coordinates. O(n)
void push_coordinates(double* lats, double* lons, vector<double>& lats_final, vector<double>& lons_final, int N)
{
	for (int i = 0; i < N; ++i) {
		if (lats[i] != 0) {
			lats_final.push_back(lats[i]);
			lons_final.push_back(lons[i]);
		}
	}
}

int main(void)
{
	// Find the total of points from a file
	int N1, N2;
	find_N1_N2(N1, N2);

	if (!N1 || !N2) {
		cerr << "Unable to open file or there isn't data" << endl;
		return 0;
	}
	
	// Utils pointers to host (h_) and device (d_)
	double *h1_latitudes, *h1_longitudes, *h2_latitudes, *h2_longitudes, *h_inters_lat, *h_inters_lon;
	double *d1_latitudes, *d1_longitudes, *d2_latitudes, *d2_longitudes, *d_inters_lat, *d_inters_lon;
	int* d_in_out;

	// Total size in bytes of polygons
	int N;
	N1 > N2 ? N = N1 : N = N2;
	const int SIZE_POLYGON_1_BYTES = N1 * sizeof(double);
	const int SIZE_POLYGON_2_BYTES = N2 * sizeof(double);
	const int SIZE_POLYGON_BIGGEST_BYTES = N * sizeof(double);

	// Allocate CPU memory
	h1_latitudes = (double*)malloc(SIZE_POLYGON_1_BYTES);
	h1_longitudes = (double*)malloc(SIZE_POLYGON_1_BYTES);
	h2_latitudes = (double*)malloc(SIZE_POLYGON_2_BYTES);
	h2_longitudes = (double*)malloc(SIZE_POLYGON_2_BYTES);
	h_inters_lat = (double*)malloc(SIZE_POLYGON_BIGGEST_BYTES);
	h_inters_lon = (double*)malloc(SIZE_POLYGON_BIGGEST_BYTES);

	// Allocate GPU memory
	cudaMalloc((void**)&d1_latitudes, SIZE_POLYGON_1_BYTES);
	cudaMalloc((void**)&d1_longitudes, SIZE_POLYGON_1_BYTES);
	cudaMalloc((void**)&d2_latitudes, SIZE_POLYGON_2_BYTES);
	cudaMalloc((void**)&d2_longitudes, SIZE_POLYGON_2_BYTES);
	cudaMalloc((void**)&d_inters_lat, SIZE_POLYGON_BIGGEST_BYTES);
	cudaMalloc((void**)&d_inters_lon, SIZE_POLYGON_BIGGEST_BYTES);
	cudaMalloc((void**)&d_in_out, N * sizeof(int));

	// Fill data in host
	load_data(h1_latitudes, h1_longitudes, h2_latitudes, h2_longitudes, N1, N2);

	// Transfer data from host to device
	cudaMemcpy(d1_latitudes, h1_latitudes, SIZE_POLYGON_1_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d1_longitudes, h1_longitudes, SIZE_POLYGON_1_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d2_latitudes, h2_latitudes, SIZE_POLYGON_2_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d2_longitudes, h2_longitudes, SIZE_POLYGON_2_BYTES, cudaMemcpyHostToDevice);
	
	// Phase 1: Points of polygon 1 inside Polygon 2
	points_polygon_inside<double> << < ceil(N1 / 1024.0), 1024 >> > 
		(d1_latitudes, d1_longitudes, d2_latitudes, d2_longitudes, d_in_out, d_inters_lat, d_inters_lon, N1, N2);

	cudaMemcpy(h_inters_lat, d_inters_lat, SIZE_POLYGON_BIGGEST_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_inters_lon, d_inters_lon, SIZE_POLYGON_BIGGEST_BYTES, cudaMemcpyDeviceToHost);

	// Obtain the first group: points of Polygon 1 inside Polygon 2
	vector<double> lats_final, lons_final;
	push_coordinates(h_inters_lat, h_inters_lon, lats_final, lons_final, N);

	//  Phase 2: Points of polygon 2 inside Polygon 1
	points_polygon_inside<double> << < ceil(N2 / 1024.0), 1024 >> >
		(d2_latitudes, d2_longitudes, d1_latitudes, d1_longitudes, d_in_out, d_inters_lat, d_inters_lon, N1, N2);

	cudaMemcpy(h_inters_lat, d_inters_lat, SIZE_POLYGON_BIGGEST_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_inters_lon, d_inters_lon, SIZE_POLYGON_BIGGEST_BYTES, cudaMemcpyDeviceToHost);

	// Obtain the second group: points of Polygon 2 inside Polygon 1
	push_coordinates(h_inters_lat, h_inters_lon, lats_final, lons_final, N);
	
	//  Phase 3: Find the intersections between Polygon 1 and Polygon 2
	if (N1 < N2) {
		intersection_points<double> << < ceil(N2 / 1024.0), 1024 >> >
			(d2_latitudes, d2_longitudes, d1_latitudes, d1_longitudes, d_inters_lat, d_inters_lon, N2, N1);
	}
	else {
		intersection_points<double> << < ceil(N1 / 1024.0), 1024 >> >
			(d1_latitudes, d1_longitudes, d2_latitudes, d2_longitudes, d_inters_lat, d_inters_lon, N1, N2);
	}

	cudaMemcpy(h_inters_lat, d_inters_lat, SIZE_POLYGON_BIGGEST_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_inters_lon, d_inters_lon, SIZE_POLYGON_BIGGEST_BYTES, cudaMemcpyDeviceToHost);

	// Obtain the third group (and final): intersection of Polygon 1 and Polygon 2
	push_coordinates(h_inters_lat, h_inters_lon, lats_final, lons_final, N);
	
	// Delete device memory
	cudaFree(d1_latitudes), cudaFree(d2_latitudes), cudaFree(d1_longitudes), cudaFree(d2_longitudes);
	cudaFree(d_inters_lat), cudaFree(d_inters_lon), cudaFree(d_in_out);

	// Delete host memory
	free(h1_latitudes), free(h2_latitudes), free(h1_longitudes), free(h2_longitudes), free(h_inters_lat), free(h_inters_lon);
	
	return 0;
}