#ifndef KMEANS_H
#define KMEANS_H

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

using namespace std;

class KPoint
{
private:
	int id_point, id_cluster;
	vector<double> values;
	int total_values;
	string name;

public:
	KPoint(int id_point, vector<double>& values, string name);

	int getID();

	void setCluster(int id_cluster);

	int getCluster();

	double getValue(int index);

	int getTotalValues();

	void addValue(double value);

	string getName();
};

class Cluster
{
private:
	int id_cluster;
	vector<double> central_values;
	vector<KPoint> points;

public:
	Cluster(int id_cluster, KPoint point);

	void addPoint(KPoint point);

	bool removePoint(int id_point);

	double getCentralValue(int index);

	void setCentralValue(int index, double value);

	KPoint getPoint(int index);

	int getTotalPoints();

	int getID();

};

class KMeans
{
private:
	int K; // number of clusters
	int total_values, total_points, max_iterations;
	

	// return ID of nearest center (uses euclidean distance)
	int getIDNearestCenter(KPoint point);

public:
    vector<Cluster> clusters;
	KMeans(int K, int total_points, int total_values, int max_iterations);
	void run(vector<KPoint> & points);

};

#endif