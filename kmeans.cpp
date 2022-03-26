#include "./include/kmeans.h"

KPoint::KPoint(int id_point, vector<double>& values, string name = "")
{
		this->id_point = id_point;
		total_values = values.size();

		for(int i = 0; i < total_values; i++)
			this->values.push_back(values[i]);

		this->name = name;
		id_cluster = -1;
}

int KPoint::getID()
{
		return id_point;
}

void KPoint::setCluster(int id_cluster)
{
	this->id_cluster = id_cluster;
}

	int KPoint::getCluster()
	{
		return id_cluster;
	}

	double KPoint::getValue(int index)
	{
		return values[index];
	}

	int KPoint::getTotalValues()
	{
		return total_values;
	}

	void KPoint::addValue(double value)
	{
		values.push_back(value);
	}

	string KPoint::getName()
	{
		return name;
	}


Cluster::Cluster(int id_cluster, KPoint point)
{
		this->id_cluster = id_cluster;

		int total_values = point.getTotalValues();

		for(int i = 0; i < total_values; i++)
			central_values.push_back(point.getValue(i));

		points.push_back(point);
}

	void Cluster::addPoint(KPoint point)
	{
		points.push_back(point);
	}

	bool Cluster::removePoint(int id_point)
	{
		int total_points = points.size();

		for(int i = 0; i < total_points; i++)
		{
			if(points[i].getID() == id_point)
			{
				points.erase(points.begin() + i);
				return true;
			}
		}
		return false;
	}

	double Cluster::getCentralValue(int index)
	{
		return central_values[index];
	}

	void Cluster::setCentralValue(int index, double value)
	{
		central_values[index] = value;
	}

	KPoint Cluster::getPoint(int index)
	{
		return points[index];
	}

	int Cluster::getTotalPoints()
	{
		return points.size();
	}

	int Cluster::getID()
	{
		return id_cluster;
	}


int KMeans::getIDNearestCenter(KPoint point)
{
	double sum = 0.0, min_dist;
	int id_cluster_center = 0;

	for(int i = 0; i < total_values; i++)
	{
		sum += pow(clusters[0].getCentralValue(i) -
					point.getValue(i), 2.0);
	}

	min_dist = sqrt(sum);

	for(int i = 1; i < K; i++)
	{
			double dist;
			sum = 0.0;

			for(int j = 0; j < total_values; j++)
			{
				sum += pow(clusters[i].getCentralValue(j) -
						   point.getValue(j), 2.0);
			}

			dist = sqrt(sum);

			if(dist < min_dist)
			{
				min_dist = dist;
				id_cluster_center = i;
			}
	}

	return id_cluster_center;
}


KMeans::KMeans(int K, int total_points, int total_values, int max_iterations)
{
		this->K = K;
		this->total_points = total_points;
		this->total_values = total_values;
		this->max_iterations = max_iterations;
}

	void KMeans::run(vector<KPoint> & points)
	{
		if(K > total_points)
			return;

		vector<int> prohibited_indexes;

		// choose K distinct values for the centers of the clusters
		int index_point = rand() % total_points;
		prohibited_indexes.push_back(index_point);
		points[index_point].setCluster(0);
		Cluster cluster(0, points[index_point]);
		clusters.push_back(cluster);

		for(int i = 1; i < K; i++)
		{
			vector<double> distance(total_points, 1000000);
			double sumD = 0;
			for(int j = 0; j < total_points; j++)
			{
				for(int k = 0; k < i; k++)
				{
					double dis;
					for(int m = 0; m < total_values; m++)
					{
						dis += pow(clusters[k].getCentralValue(m) -
						   points[j].getValue(m), 2.0);
					}
					dis = sqrt(dis);

					if(dis < distance[j])
						distance[j] = dis;
				}
				sumD += distance[j];
			}
			
			
			double r = sumD * rand() / double(RAND_MAX - 1.);
        	int j = 0;
        	for(; r > 0; ++j){
            	r -= distance[j];
        	}
			if(j == total_points)
				j = j - 1;
			//cout << sumD << " " << total_points << " " << j <<endl;
			prohibited_indexes.push_back(j);
			points[j].setCluster(i);
			Cluster cluster(i, points[j]);
			clusters.push_back(cluster);
		}

		// for(int i = 0; i < K; i++)
		// {
		// 	while(true)
		// 	{
		// 		int index_point = rand() % total_points;

		// 		if(find(prohibited_indexes.begin(), prohibited_indexes.end(),
		// 				index_point) == prohibited_indexes.end())
		// 		{
		// 			prohibited_indexes.push_back(index_point);
		// 			points[index_point].setCluster(i);
		// 			Cluster cluster(i, points[index_point]);
		// 			clusters.push_back(cluster);
		// 			break;
		// 		}
		// 	}
		// }

		int iter = 1;

		while(true)
		{
			bool done = true;

			// associates each point to the nearest center
			for(int i = 0; i < total_points; i++)
			{
				int id_old_cluster = points[i].getCluster();
				int id_nearest_center = getIDNearestCenter(points[i]);

				if(id_old_cluster != id_nearest_center)
				{
					if(id_old_cluster != -1)
						clusters[id_old_cluster].removePoint(points[i].getID());

					points[i].setCluster(id_nearest_center);
					clusters[id_nearest_center].addPoint(points[i]);
					done = false;
				}
			}

			// recalculating the center of each cluster
			for(int i = 0; i < K; i++)
			{
				for(int j = 0; j < total_values; j++)
				{
					int total_points_cluster = clusters[i].getTotalPoints();
					double sum = 0.0;

					if(total_points_cluster > 0)
					{
						for(int p = 0; p < total_points_cluster; p++)
							sum += clusters[i].getPoint(p).getValue(j);
						clusters[i].setCentralValue(j, sum / total_points_cluster);
					}
				}
			}

			if(done == true || iter >= max_iterations)
			{
				//cout << "Break in iteration " << iter << "\n\n";
				break;
			}

			iter++;
		}

		// shows elements of clusters
		// for(int i = 0; i < K; i++)
		// {
		// 	int total_points_cluster =  clusters[i].getTotalPoints();

		// 	cout << "Cluster " << clusters[i].getID() + 1 << endl;
		// 	for(int j = 0; j < total_points_cluster; j++)
		// 	{
		// 		cout << "Point " << clusters[i].getPoint(j).getID() + 1 << ": ";
		// 		for(int p = 0; p < total_values; p++)
		// 			cout << clusters[i].getPoint(j).getValue(p) << " ";

		// 		string point_name = clusters[i].getPoint(j).getName();

		// 		if(point_name != "")
		// 			cout << "- " << point_name;

		// 		cout << endl;
		// 	}

		// 	cout << "Cluster values: ";

		// 	for(int j = 0; j < total_values; j++)
		// 		cout << clusters[i].getCentralValue(j) << " ";

		// 	cout << "\n\n";
		// }
	}
