#include <opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc_c.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include "include/Autocanny.h"
#include "include/kmeans.h"
#include <numeric>
#include <dirent.h>
#include <sys/time.h>

using namespace std;
using namespace cv;

//tower
typedef pair<int, int> Region; //用pair表示塔架区域的坐标（比如region_x表示横坐标的区域，firs表示左侧 second表示右侧 regionx_y同理）

//返回结构体
struct IDENT_INTERFACE
{
    int target;  //当前目标
    float angel; //角度
    int x;       //x坐标
    int y;       //y坐标
    int a;       //塔架长
    int b;       //塔架宽
    Mat image;   //debug图像
};


/***
 *  声明函数部分
 *  分为前置处理、塔检测和线检测三个部分
***/

//前置部分
pair<double, double> Line(double x1, double y1, double x2, double y2);
bool InOneLine(double k, double b , double k2, double b2, double K); // 判断相近直线
double Mean(vector<double> resultSet);
double Variance(vector<double> resultSet, double mean);
double StandardDev(double variance);
void Zscore(vector<double>& vec);
//设置图像大小
Mat ImageSize(Mat src, int k);
//通道分离
Mat ImageSplit(Mat srcImage);
//Canny边缘检测
Mat ImageCanny(Mat src);


/***
 *  角度检测与线检测部分
 *  霍夫变换->平行线->角度
 ***/

//第一步霍夫变换
vector<Vec4d> HoughFirstStep(Mat src);
//第二步霍夫变换
vector<Vec4d> HoughSecondStep(Mat src);
//获取平行线组
vector<Vec4d> GetParaLine(vector<Vec4d> lines);
//合并平行线中的同一条直线
vector<Vec4d> MergeParaLine(vector<Vec4d> para_line);
//聚类过程
double GetKmeansTheta(vector<Vec4d> para_line, vector<double> k_tan_bak, Mat& src);
//获取角度总函数
double GetTheta(Mat &src, Mat src_canny);
//第二步画线框函数
double DrawLine(Mat &src, Mat src_canny, Region region_x);

/***
 *  塔的边框检测部分
 ***/

//角点检测
Mat ImageHarris(Mat src);
//垂直积分投影
Region VerticalProjection(Mat srcImage, Mat src_harris);
//水平积分投影
Region HorizonProjection(Mat srcImage, Mat src_harris);
//塔架检测函数，返回值为一对正负1点对，first为横向返回值（向右为1，向左为-1），second为纵向返回值（向上为1，向下为-1）
Region tower_detection(Mat src, Mat src_re, Mat src_canny, Mat src_harris, Region region_x, Region region_y);
//第二步塔检测函数
Region tower_detection_two(Mat src, Mat src_re, Mat src_canny, Mat src_harris, Region region_x, Region region_y);

/***
 *  整合部分
 ***/

//第一步返回位置角度信息
IDENT_INTERFACE firstDetection(Mat src, bool debug);
//第二步
IDENT_INTERFACE secondDetection(Mat src, bool debug);
//第三步
IDENT_INTERFACE thirdDetection(Mat src, bool debug);

//返回函数
IDENT_INTERFACE Interface(Mat src, int targetLabel, bool debug);


//测试使用，旋转图片
void Rotate(const Mat &srcImage, Mat &destImage, double angle)
{
	Point2f center(srcImage.cols / 2, srcImage.rows / 2);//中心
	Mat M = getRotationMatrix2D(center, angle, 1);//计算旋转的仿射变换矩阵 
	warpAffine(srcImage, destImage, M, Size(srcImage.cols, srcImage.rows));//仿射变换  
	circle(destImage, center, 2, Scalar(255, 0, 0));
}



int main()
{
    VideoCapture cap;
    cap.open("usb-camera.mp4"); //打开视频,以上两句等价于VideoCapture cap("E://2.avi");
 
     //cap.open("http://www.laganiere.name/bike.avi");//也可以直接从网页中获取图片，前提是网页有视频，以及网速够快
    if(!cap.isOpened())//如果视频不能正常打开则返回    
    {
        printf("NO video");
        return 0;
    }
    Mat frame;
    while(1)
    {
        cap>>frame;//等价于cap.read(frame);
        if(frame.empty())//如果某帧为空则退出循环
             break;
        
        Mat src = frame.clone();
        Mat src_bak = frame.clone();

        IDENT_INTERFACE res = Interface(src, 2, 1);
        cout << res.target << " "<< res.x << " " << res.y << " " << res.a << " " << res.b << " " << res.angel << " " << endl;
    
        imshow("result", res.image);
        waitKey(20);//每帧延时20毫秒
    }
    cap.release();//释放资源
    // string path = "./pictures/2.jpg";//角度图片路径
    // Mat src = imread(path);
    // Mat src_bak = src.clone();

    // IDENT_INTERFACE res = Interface(src, 1, 1);
    // cout << res.target << " "<< res.x << " " << res.y << " " << res.a << " " << res.b << " " << res.angel << " " << endl;
    
    
    // vector<double> firstRes = firstDetection(src);
    // for(int i = 0; i < 3; i++)
    //     cout << firstRes[i] <<" ";
    // cout<< endl; 

    // Mat destImage;
    // Rotate(src, destImage, firstRes[2]);
    // imwrite("dst.png", destImage);

    // vector<double> secondRes = secondDetection(src_bak, 1);
    // for(int i = 0; i < 3; i++)
    //     cout << secondRes[i] <<" ";
    // cout << endl; 

    // vector<double> thirdRes = thirdDetection(src_bak, 1);
    // for(int i = 0; i < 3; i++)
    //     cout << thirdRes[i] <<" ";
    // cout << endl; 
}

pair<double, double> Line(double x1, double y1, double x2, double y2)
{
    float k;
    float b;
    if(x2 - x1 == 0)
    {
        k = (y2 - y1) / (x2 - x1 + 0.00001);
        b = x2;
    }
    else
    {
        k = (y2 - y1) / (x2 - x1);
        b = y2 - k * x2;
    }
    return make_pair(k, b);
}

bool InOneLine(double k, double b , double k2, double b2, double K)
{
    if(abs(k - k2) < 1)
    {
        double d = abs(b - b2) / sqrt(1 + K * K);
        if(d < 10)
            return true;
    }  
    return false;
}

double Mean(vector<double> resultSet)
{
    double sum = std::accumulate(std::begin(resultSet), std::end(resultSet), 0.0);
	double mean =  sum / resultSet.size(); //均值
    return mean;
}

double Variance(vector<double> resultSet, double mean)
{
    double accum  = 0.0;
	std::for_each (std::begin(resultSet), std::end(resultSet), [&](const double d) {
		accum  += (d-mean)*(d-mean);
	});

	double variance = accum/(resultSet.size()-1); //方差
    return variance;
}

double StandardDev(double variance)
{
	double stdev = sqrt(variance); //标准差
    return stdev;
}

void Zscore(vector<double>& vec)
{
    double mean = Mean(vec);
    double variance = Variance(vec, mean);
    double stdev = StandardDev(variance);
    for(int i = 0; i < vec.size(); i++)
    {
        vec[i] = (vec[i] - mean) / stdev;
    }
}

Mat ImageSize(Mat src, int k)//设置图像大小
{
	Mat dst;
    if(k == 1)
        resize(src, dst, Size(640, 480));
    else if(k == 2)
	    resize(src, dst, Size(1920, 1080));
	return dst;
}

Mat ImageSplit(Mat srcImage)//通道分离
{
	vector<Mat> channels;
	split(srcImage, channels);
	vector<Mat> mbgr(3);	//创建类型为Mat，数组长度为3的变量mbgr
	Mat hideChannel(srcImage.size(), CV_8UC1, Scalar(0));
	Mat imageRB(srcImage.size(), CV_8UC3);	
	mbgr[0] = channels[0];
	mbgr[1] = hideChannel;
	mbgr[2] = channels[2];
	merge(mbgr, imageRB);
	return imageRB;
}

Mat ImageCanny(Mat src)//边缘检测
{
	Mat dst;
	Mat src_gray;
    cvtColor(src, src_gray, WINDOW_AUTOSIZE);
	Mat src_Blur;
	GaussianBlur(src_gray, src_Blur, Size(5, 5), 2, 2); //高斯滤波
	double low = 50;
	double high = 200;
	//AdaptiveFindThreshold(src_Blur, low, high, 3);
    Canny(src_Blur, dst, low, high); //边缘检测
    //imshow("边缘图", dst);
	//imwrite("edge.png", dst);
	return dst;
}

//第一步霍夫变换
vector<Vec4d> HoughFirstStep(Mat src)
{
    //霍夫直线检测
    vector<Vec4d> line_data;
    //角度
    HoughLinesP(src, line_data, 1, CV_PI/180.0, 50, 50 ,5);
    //线
    //HoughLinesP(src, line_data, 1, CV_PI/180.0, 50, 50 ,10);
    //HoughLines(src_canny, line_data, 1, CV_PI/180.0, 300, W_min ,W_max);

    return line_data;
}

//第二步霍夫变换
vector<Vec4d> HoughSecondStep(Mat src)
{
    //霍夫直线检测
    vector<Vec4d> line_data;
    //角度
    //HoughLinesP(src, line_data, 1, CV_PI/180.0, 100, 80 ,5);
    //线
    HoughLinesP(src, line_data, 1, CV_PI/180.0, 50, 50 ,15);
    //HoughLines(src_canny, line_data, 1, CV_PI/180.0, 300, W_min ,W_max);

    //cout << line_data.size() << endl;
    return line_data;
}

//获取平行线组
vector<Vec4d> GetParaLine(vector<Vec4d> lines)
{
    vector<Vec4d> para_line;
    int n = -1; //平行线个数
    for (size_t i = 0; i < lines.size(); ++i)
    {
        Vec4d temp = lines[i];
        double k_i = atan2(temp[1] - temp[3], temp[0] - temp[2]) * 180 / CV_PI;
        for(int j = 0; j < lines.size(); ++j)
        {
            Vec4d temp_j = lines[j];
            double k_j = atan2(temp_j[1] - temp_j[3], temp_j[0] - temp_j[2]) * 180 / CV_PI;
            if(abs(k_i - k_j) < 1.0)
                n++;
        }
        if(n >= 4)
            para_line.push_back(temp);
        n = -1;
    }

    //cout << para_line.size() << endl;

    return para_line;
}

//合并平行线中的同一条直线
vector<Vec4d> MergeParaLine(vector<Vec4d> para_line)
{
    //判断是否在一条直线上
    vector<Vec4d> P_Line;
    P_Line.push_back(para_line[0]);
    for (int i = 1; i < para_line.size(); ++i)
    {
        Vec4d temp = para_line[i];
        pair<double, double> result = Line(temp[0], temp[1], temp[2], temp[3]);
        double k = atan2(temp[1] - temp[3], temp[0] - temp[2]) * 180 / CV_PI;
        double b = result.second;

        int flag = 0;

        for(int j = 0; j < P_Line.size(); ++j)
        {
            Vec4d temp_j = P_Line[j];
            pair<double, double> result_j = Line(temp_j[0], temp_j[1], temp_j[2], temp_j[3]);
            double k_j, b_j;
            k_j = atan2(temp_j[1] - temp_j[3], temp_j[0] - temp_j[2]) * 180 / CV_PI;
            b_j = result_j.second;
            if(InOneLine(k, b, k_j, b_j, result.first)) //如果两条直线相近
            {
                //删除
                flag = 1;
            }
        }
        if(flag == 0)
            P_Line.push_back(para_line[i]);
        flag = 0;
    }

    return P_Line;
}

//聚类过程
double GetKmeansTheta(vector<Vec4d> para_line, vector<double> k_tan_bak, Mat& src)
{
    int total_points = 0, total_values = 1, K , max_iterations = 50, has_name = 0;
	vector<KPoint> points;
	string point_name;

    if(k_tan_bak.size() / 8 < 2)
        K = 2;
    else if(k_tan_bak.size() / 8 > 6)
        K = 6;
    else
        K = k_tan_bak.size() / 8;
    
    for (int i = 0; i < para_line.size(); ++i)
    {
        Vec4d temp = para_line[i];
        //float k_temp = (temp[1] - temp[3]) / (temp[0] - temp[2] + 0.00001);
        vector<double> values;
            total_points += 1;
			values.push_back(k_tan_bak[i]);
			//values.push_back(abs(k_all[i][3]));
            //values.push_back(abs(k_all[i][1]));
            if(has_name)
            {
                point_name = "";
                KPoint p(i, values, point_name);
                points.push_back(p);
            }
            else
            {
                point_name = "";
                KPoint p(i, values,point_name);
                points.push_back(p);
            }

            //cout << k_all[i][2] << endl;
    }

    KMeans kmeans(K, total_points, total_values, max_iterations);
    kmeans.run(points);
    vector<Cluster> clusters = kmeans.clusters;
    int MAX_1 = 0, MAX_2 = 0;
    vector<double> stdev;
    for(int i = 0; i < K; ++i)
    {
            int total_points_cluster =  clusters[i].getTotalPoints();
            vector<double> theta;
            for(int p = 0; p < total_points_cluster; p++)
            {
                theta.push_back(clusters[i].getPoint(p).getValue(0));
            }
            double mean = Mean(theta);
            double var = Variance(theta, mean);
            double std = StandardDev(var);
            stdev.push_back(std);

            if(total_points_cluster > MAX_1)
            {
                MAX_1 = total_points_cluster;
                MAX_2 = i;
            }
    }
    // for(int i = 0; i < K; i++)
    //     cout << stdev[i] << endl;
    
    int total_points_cluster =  clusters[MAX_2].getTotalPoints();
    //cout << total_points_cluster << endl;
    //cout << "Cluster " << clusters[i].getID() + 1 << endl;
    for(int j = 0; j < total_points_cluster; ++j)
    {
        //cout << "Point " << clusters[i].getPoint(j).getID() + 1 << ": ";
                    
        Vec4f temp = para_line[clusters[MAX_2].getPoint(j).getID()];
        //if(abs(clusters[i].getPoint(j).getValue(0) - clusters[i].getCentralValue(0)) < 2)
            //line(src, Point(temp[0], temp[1]), Point(temp[2], temp[3]), Scalar(255, 0, 0), 2, CV_AA);     

            //for(int p = 0; p < total_values; p++)
            //cout << clusters[i].getPoint(j).getValue(p) << " ";
            //cout << k_tan_bak[clusters[i].getPoint(j).getID()] << " ";

            string point_name = clusters[MAX_2].getPoint(j).getName();

                    //if(point_name != "")
                        //cout << "- " << point_name;

                    //cout << endl;
    }

    return clusters[MAX_2].getCentralValue(0);
}

double GetTheta(Mat &src, Mat src_canny)
{
    //4k转化为1080p
    //resize(src, src, Size(src.cols/2, src.rows/2));
    if (!src.data){
        printf("cannot load image ...");
        return -1;
    }
    //imshow("src img", src);
    
    //灰度图到边缘检测
    //Mat src_canny = Gray2Canny(src);
    //cv::imwrite("edge.png", src_canny);
    
    //霍夫直线检测初步处理
    vector<Vec4d> line_data; 
    line_data = HoughFirstStep(src_canny);

    //平行线组
    vector<Vec4d> para_line;
    para_line = GetParaLine(line_data);

	if(para_line.size() == 0)
	{
		return 999;
	}
    //合并平行线
    para_line = MergeParaLine(para_line);
    //cout << para_line.size() << endl;

    //计算平行线斜率角度theta    
    vector<double> k_tan;
    for (int i = 0; i < para_line.size(); ++i)
	{
        Vec4f temp = para_line[i];
        k_tan.push_back(atan2(temp[1] - temp[3], temp[0] - temp[2]) * 180 / CV_PI);
        //cout << atan2(temp[1] - temp[3], temp[0] - temp[2]) * 180 / CV_PI << endl;
        //line(src, Point(temp[0], temp[1]), Point(temp[2], temp[3]), color, 3, CV_AA);
	}

    //数据归一化处理
    //vector<double> k_tan_bak = k_tan;
    //Zscore(k_tan);
    
    //获取聚类中心角度
    double theta = 0;
    if(para_line.size() > 1)
        theta = GetKmeansTheta(para_line, k_tan, src);
    else
        theta = k_tan[0];
    if(theta <= -90)
        theta =  (abs(theta) - 90) * -1;
    else if(theta >= 90)
        theta = theta - 90;

    //imshow("houghLinesP img", src);
    //resize(src, src, Size(src.cols/4, src.rows/4));
	//cv::imwrite(name, src);
    //waitKey(0);
    
    return theta;
}

double DrawLine(Mat &src, Mat src_canny, Region region_x)
{
    //霍夫直线检测初步处理
    vector<Vec4d> line_data; 
	line_data = HoughSecondStep(src_canny);

    //平行线组
    vector<Vec4d> para_line;
	para_line = GetParaLine(line_data);

	if(para_line.size() == 0)
	{
		return 999;
	}
    //合并平行线
    para_line = MergeParaLine(para_line);
    //cout << para_line.size() << endl;

    //保留90度直线  
    //vector<double> k_tan;
    vector<Vec4d> P_Line;
    for (int i = 0; i < para_line.size(); i++)
	{
        Vec4f temp = para_line[i];
        if(abs(abs(atan2(temp[1] - temp[3], temp[0] - temp[2]) * 180 / CV_PI) - 90) <= 10)
        {
            P_Line.push_back(temp);
        }     
	}

    //cout << P_Line.size() << endl;

    //没有90度左右的平行线
    if(P_Line.size() == 0)
        return 999;

    int Right = 0;
    double Min = -1;
    //region_x.first = region_x.first;
    //region_x.second = region_x.second;
    for (int i = 0; i < P_Line.size(); i++)
	{
        Vec4f temp = P_Line[i];
        //line(src, Point(temp[0], temp[1]), Point(temp[2], temp[3]), (0,0,255), 2, CV_AA);     
        pair<double, double> result = Line(temp[0], temp[1], temp[2], temp[3]);
        //上边界y = 0
        double x_top = -result.second / result.first;
        //下边界
        double x_bound = (src.rows - result.second) / result.first;

        //cout << i << " " <<x_top << " " << x_bound << " " << region_x.first << " " << region_x.second << endl;
        
        if(x_top > Min && x_top < region_x.second && x_top > region_x.first && x_bound < region_x.second && x_bound > region_x.first)
        //if(x_top > Min)
        {
            Min = x_top;
            Right = i; 
        }   
	}   
    //cout << Min << endl; 
    //cout << Right << endl;
    Rect rect(Min-15, 5, 30, src.rows);//左上坐标（x,y）和矩形的长(x)宽(y)
    cv::rectangle(src, rect, Scalar(0, 0, 255), 3, LINE_8,0);

    return Min;
}

Mat ImageHarris(Mat src)//角点检测
{
	Mat src_gray;
	cvtColor( src, src_gray, COLOR_BGR2GRAY);
	//GaussianBlur(src_gray, src_gray, Size(3, 3), 0, 0);
	//imshow("test", src_gray);
	Mat dst = Mat::zeros( src.size(), CV_32FC1 );
    cornerHarris( src_gray, dst, 2, 3, 0.04);
	Mat dst_norm, dst_norm_scaled;
    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );
	//imshow("test", dst_norm);
	for( int i = 0; i < dst_norm.rows ; i++ )
    {
        for( int j = 0; j < dst_norm.cols; j++ )
        {
            if( (int) dst_norm.at<float>(i,j) > 90 )
            {
                circle( dst_norm_scaled, Point(j,i), 1,  Scalar(0), 1, 8, 0 );
            }
        }
    }
	threshold(dst_norm_scaled, dst_norm_scaled, 0, 255, 1);
    //imshow("角点图", dst_norm_scaled);
	//imwrite("corner.png", dst_norm_scaled);
	return dst_norm_scaled;
}

Region VerticalProjection(Mat srcImage, Mat src_harris)//垂直积分投影
{
	Region region_x;//记录区域左右边界的横坐标
	//对边缘图进行投影
	if (srcImage.channels() > 1)
		cvtColor(srcImage, srcImage, COLOR_RGB2GRAY);
	Mat srcImageBin;
	threshold(srcImage, srcImageBin, 120, 255, THRESH_BINARY_INV);
	//imshow("二值图", srcImageBin);
	int *colswidth = new int[srcImage.cols];  //申请src.image.cols个int型的内存空间
	memset(colswidth, 0, srcImage.cols * 4);  //数组必须赋初值为零，否则出错。无法遍历数组。
	//  memset(colheight,0,src->width*4);  
	// CvScalar value; 
	int value;
	for (int i = 0; i < srcImage.cols; i++)
	for (int j = 0; j < srcImage.rows; j++)
	{
		//value=cvGet2D(src,j,i);
		value = srcImageBin.at<uchar>(j, i);
		if (value == 0)
		{
			colswidth[i]++; //统计每列的白色像素点  
		}
	}

	//对角点图进行垂直投影
	if (src_harris.channels() > 1)
		cvtColor(src_harris, src_harris, COLOR_RGB2GRAY);
	Mat src_harrisBin;
	threshold(src_harris, src_harrisBin, 120, 255, THRESH_BINARY_INV);
	int *cols_harris = new int[srcImage.cols];  //申请src.image.cols个int型的内存空间
	memset(cols_harris, 0, srcImage.cols * 4);  //数组必须赋初值为零，否则出错。无法遍历数组。
	
	for (int i = 0; i < srcImage.cols; i++)
	for (int j = 0; j < srcImage.rows; j++)
	{
		value = src_harrisBin.at<uchar>(j, i);
		if (value == 0)
		{
			cols_harris[i]++; //统计每列的白色像素点  
		}
	}

	int *colswidth_2 = new int[srcImage.cols];
	memset(colswidth_2, 0, srcImage.cols * 4);
	memcpy(colswidth_2, colswidth, sizeof(colswidth_2));

	int *cols_harris_2 = new int[srcImage.cols];
	memset(cols_harris_2, 0, srcImage.cols * 4);
	memcpy(cols_harris_2, cols_harris, sizeof(cols_harris_2));
	
	int ct; 
	//对相邻10列高度取平均值
	for (int i = 4; i < srcImage.cols - 6; i++)
	{
		ct = 0;
		for(int j = -4; j < 6 ; j++)
		{
			ct = ct + cols_harris[i + j] * 2;
		}
		cols_harris_2[i] = ct/10; 
	}
	//对边缘图和角点图取交集，如果该列角点数为0，将边缘数也设置为0
	for(int i = 0; i < srcImage.cols; i++)
	{
		if(cols_harris_2[i] == 0 )
		{
			colswidth[i] = 0;
		}
	}
	
	//将取交集后的边缘投影图平滑化处理
	for (int i = 4; i < srcImage.cols - 6; i++)
	{
		ct = 0;
		for(int j = -4; j < 6 ; j++)
		{
			ct = ct + colswidth[i + j];
		}
		colswidth_2[i] = ct/10; 
	}
	
	Mat histogramImage_2(srcImage.rows, srcImage.cols, CV_8UC1);
	for (int i = 0; i < srcImage.rows; i++)
	for (int j = 0; j < srcImage.cols; j++)
	{
		value = 255;  //背景设置 
		histogramImage_2.at<uchar>(i, j) = value;
	}
	for (int i = 0; i < srcImage.cols; i++)
	for (int j = 0; j < colswidth_2[i]; j++)
	{
		value = 0;  //直方图设置为黑色
		histogramImage_2.at<uchar>(srcImage.rows - 1 - j, i) = value;
	}
	//imwrite("垂直平均投影图.png", histogramImage_2);

	int window_size = 40;
	ct = 0;
	int average = 0;
	//计算平均条形高度（只计算高度不为0的列）
	for (int i = 0; i < srcImage.cols; i++)
	{
		if(colswidth_2[i] > 0)
		{
			ct++;
			average = average + colswidth_2[i];
		}
	}
	average = average/ct;
	int difference = 0;//相邻两区域间的落差
	int increase = 0;//最大升高落差（区域左侧边缘）
	int drop = 0;//最大下降落差（区域右侧边缘）
	vector<int>area;//分割小区域
	int area_size;//小区域个数
	area_size = srcImage.cols / window_size;
	for(int i = 0; i < area_size; i++)//计算每个小区域内的条形高度
	{
		ct = 0;
		for(int j = 0; j < window_size; j++)
		{
			ct = ct + colswidth_2[j + i * window_size];
		}
		area.push_back(ct);   
	}
	for (int i = 0; i < area.size() -  2; i++)//遍历小区域，寻找最大升高和下降落差，最大落差所在的小区域即为塔架边缘
	{
		difference = area[i + 1] - area[i];
		if(difference > increase)
		{
			increase = difference;
			region_x.first = window_size * (i + 1) - 1;
		}
		if(difference < drop)
		{
			drop = difference;
			region_x.second = window_size * (i + 2) - 1;
		}
	}
	//imwrite("VP.png", histogramImage_2);
	//cout<<region_x.second - region_x.first<<endl; 
	return region_x;
}

Region HorizonProjection(Mat srcImage, Mat src_harris)//水平积分投影
{
	Region region_y;
	//对边缘图进行水平投影
	if (srcImage.channels() > 1)
		cvtColor(srcImage, srcImage, COLOR_RGB2GRAY);
	Mat srcImageBin;
	threshold(srcImage, srcImageBin, 120, 255, THRESH_BINARY_INV);
	//imshow("二值图", srcImageBin);
	int *rowswidth = new int[srcImage.rows];  //申请src.image.rows个int型的内存空间
	memset(rowswidth, 0, srcImage.rows * 4);  //数组必须赋初值为零，否则出错。无法遍历数组。
	int value;
	for (int i = 0; i<srcImage.rows; i++)
	for (int j = 0; j<srcImage.cols; j++)
	{
		//value=cvGet2D(src,j,i);
		value = srcImageBin.at<uchar>(i, j);
		if (value == 0)
		{
			rowswidth[i]++; //统计每行的白色像素点  
		}
	}

	//对角点图进行水平投影
	if (srcImage.channels() > 1)
		cvtColor(src_harris, src_harris, COLOR_RGB2GRAY);
	Mat src_harrisBin;
	threshold(src_harris, src_harrisBin, 120, 255, THRESH_BINARY_INV);
	int *rows_harris = new int[srcImage.rows];  //申请src.image.rows个int型的内存空间
	memset(rows_harris, 0, srcImage.rows * 4);  //数组必须赋初值为零，否则出错。无法遍历数组。
	for (int i = 0; i<srcImage.rows; i++)
	for (int j = 0; j<srcImage.cols; j++)
	{
		value = src_harrisBin.at<uchar>(i, j);
		if (value == 0)
		{
			rows_harris[i]++; //统计每行的白色像素点  
		}
	}

	int ct; //记录相邻10列的直方图高度
	int *rowsharris_2 = new int[srcImage.cols];
	memset(rowsharris_2, 0, srcImage.cols * 4);
	memcpy(rowsharris_2, rowswidth, sizeof(rowsharris_2));
	//对相邻10列高度取平均值
	for (int i = 4; i < srcImage.rows - 6; i++)
	{
		ct = 0;
		for(int j = -4; j < 6 ; j++)
		{
			ct = ct + rows_harris[i + j] * 2;
		}
		rowsharris_2[i] = ct/10; 
	}
	//对边缘图和角点图取交集，如果该列角点数为0，将边缘数也设置为0
	for(int i = 0; i < srcImage.rows; i++)
	{
		if(rowsharris_2 == 0)
		{
			rowswidth = 0;
		}
	}
	
	int *rowswidth_2 = new int[srcImage.cols];
	memset(rowswidth_2, 0, srcImage.cols * 4);
	memcpy(rowswidth_2, rowswidth, sizeof(rowswidth_2));
	//平滑化处理
	for (int i = 4; i < srcImage.rows - 6; i++)
	{
		ct = 0;
		for(int j = -4; j < 6 ; j++)
		{
			ct = ct + rowswidth[i + j];
		}
		rowswidth_2[i] = ct/10; //对相邻10列高度取平均值
	}
	Mat histogramImage_2(srcImage.rows, srcImage.cols, CV_8UC1);
	for (int i = 0; i < srcImage.rows; i++)
	for (int j = 0; j < srcImage.cols; j++)
	{
		value = 255;  //背景设置 
		histogramImage_2.at<uchar>(i, j) = value;
	}
	for (int i = 0; i < srcImage.rows; i++)
	for (int j = 0; j < rowswidth_2[i]; j++)
	{
		value = 0;  //直方图设置为黑色
		histogramImage_2.at<uchar>(i, j) = value;
	}

	int window_size = 20;
	ct = 0;
	int average = 0;
	//计算平均条形高度（只计算高度不为0的列）
	for (int i = 0; i < srcImage.rows; i++)
	{
		if(rowswidth_2[i] > 0)
		{
			ct++;
			average = average + rowswidth_2[i];
		}
	}
	average = average/ct;
	int difference = 0, increase = 0, drop = 0;//相邻两区域间的落差
	vector<int>area;
	int area_size;
	area_size = srcImage.rows / window_size;
	for(int i = 0; i < area_size; i++)
	{
		ct = 0;
		for(int j = 0; j < window_size; j++)
		{
			ct = ct + rowswidth_2[j + i * window_size];
		}
		area.push_back(ct);   
	}
	for (int i = 0; i < area.size() -  2; i++)
	{
		difference = area[i + 1] - area[i];
		if(difference > increase)
		{
			increase = difference;
			region_y.first = window_size * i - 1;
		}
		if(difference < drop)
		{
			drop = difference;
			region_y.second = window_size * (i + 1) - 1;
		}
	}
	//imshow("水平平均投影图", histogramImage_2);
	//imwrite("HP.png", histogramImage_2);
	//cout<<region_y.second - region_y.first<<endl;
	return region_y;
}

Region tower_detection(Mat src, Mat src_re, Mat src_canny, Mat src_harris, Region region_x, Region region_y)//塔架检测函数，返回值为一对正负1点对，first为横向返回值（向右为1，向左为-1），second为纵向返回值（向上为1，向下为-1）
{
	//Mat src_re;//src为输入图像 src_re为绘制检测区域的图像
    //src = imread("/home/ppzsml/TT/001.jpg");
	// src = ImageSize(src);
	// src_re = src;
	// src = ImageSplit(src);
	
	//Mat src_canny;
	// Mat src_harris;
	// //src_canny = ImageCanny(src);
	// src_harris = ImageHarris(src);
	
	// Region region_x, region_y;//x为区域横坐标 y为区域纵坐标
	// region_x = VerticalProjection(src_canny, src_harris);
	// region_y = HorizonProjection(src_canny, src_harris);
	
	rectangle(src_re, Point(region_x.first, region_y.first), Point(region_x.second, region_y.second), Scalar( 0, 0, 255), 1, 8);//绘制塔架区域
	Region center;
	center.first = (region_x.first + region_x.second) / 2;
	center.second = (region_y.first + region_y.second) / 2;
	//cout<<center.first<<" "<<center.second<<endl;
	// circle( src_re, Point(center.first, center.second), 5,  Scalar(0, 0, 255), 1, 8, 0 );
	//Region res;//first为横向返回值（向右为1，向左为-1），second为纵向返回值（向上为1，向下为-1）
	// int width = 10; //中心区域宽度
	// if(center.first <= src.cols/2 - width)
	// {
	// 	res.first = -1;
	// }
	// else if(center.first <= src.cols/2 + width)
	// {
	// 	res.first = 0;
	// }
	// else
	// {
	// 	res.first = 1;
	// }
	// if(center.second <= src.rows - width)
	// {
	// 	res.second = 1;
	// }
	// else if(res.second <= src.rows + width)
	// {
	// 	res.second = 0;
	// }
	// else
	// {
	// 	res.second = -1;
	// }
	//cout<<res.first<<" "<<res.second<<endl;
	//imshow("塔架区域", src_re);
	//imwrite("Region.png", src_re);
    //waitKey(0);
	return center;
}

Region tower_detection_two(Mat src, Mat src_re, Mat src_canny, Mat src_harris, Region region_x, Region region_y)//塔架检测函数，返回值为一对正负1点对，first为横向返回值（向右为1，向左为-1），second为纵向返回值（向上为1，向下为-1）
{
	//Mat src_re;//src为输入图像 src_re为绘制检测区域的图像
    //src = imread("/home/ppzsml/TT/001.jpg");
	// src = ImageSize(src);
	// src_re = src;
	// src = ImageSplit(src);
	
	//Mat src_canny;
	// Mat src_harris;
	// //src_canny = ImageCanny(src);
	// src_harris = ImageHarris(src);
	
	// Region region_x, region_y;//x为区域横坐标 y为区域纵坐标
	// region_x = VerticalProjection(src_canny, src_harris);
	// region_y = HorizonProjection(src_canny, src_harris);
	
	rectangle(src_re, Point(region_x.first, region_y.first), Point(region_x.second, region_y.second), Scalar( 0, 0, 255), 1, 8);//绘制塔架区域
	Region center;
	center.first = (region_x.first + region_x.second) / 2;
	center.second = (region_y.first + region_y.second) / 2;
	//cout<<center.first<<" "<<center.second<<endl;
	// circle( src_re, Point(center.first, center.second), 5,  Scalar(0, 0, 255), 1, 8, 0 );
	// Region res;//first为横向返回值（向右为1，向左为-1），second为纵向返回值（向上为1，向下为-1）
	// int width = 10; //中心区域宽度
	// if(center.first <= src.cols/2 - width)
	// {
	// 	res.first = -1;
	// }
	// else if(center.first <= src.cols/2 + width)
	// {
	// 	res.first = 0;
	// }
	// else
	// {
	// 	res.first = 1;
	// }
	// if(center.second <= src.rows - width)
	// {
	// 	res.second = 1;
	// }
	// else if(res.second <= src.rows + width)
	// {
	// 	res.second = 0;
	// }
	// else
	// {
	// 	res.second = -1;
	// }
	//cout<<res.first<<" "<<res.second<<endl;
	//imshow("塔架区域", src_re);
	//imwrite("Region.png", src_re);
    //waitKey(0);
	return center;
}


//整合函数
IDENT_INTERFACE firstDetection(Mat src, bool debug)
{
    //存储结果
    IDENT_INTERFACE res;
	res.target = 1;
    int x = src.rows / 480; // 缩放比例

    //时间戳
	typedef std::chrono::high_resolution_clock Clock;

    //前置部分
	auto t1 = Clock::now();
	src = ImageSize(src, 1);
    res.image = src;
	Mat src_re = src;
	src = ImageSplit(src);
    Mat src_canny = ImageCanny(src);

    //Mat src_tower = ImageSize(src, 1);
    //Mat src_re_tower = ImageSize(src_re, 1);
    //Mat src_tower_canny = ImageSize(src_canny, 1);

    //塔检测 
	auto t2 = Clock::now();
    Mat src_harris;
	src_harris = ImageHarris(src_re);
	Region region_x, region_y;//x为区域横坐标 y为区域纵坐标
	region_x = VerticalProjection(src_canny, src_harris);
	region_y = HorizonProjection(src_canny, src_harris);

    Region t = tower_detection(src, src_re, src_canny, src_harris, region_x, region_y);
    
    res.x = t.first * x;
    res.y = t.second * x;
    res.a = abs(region_x.second - region_x.first) * x;
    res.b = abs(region_y.second - region_y.first) * x;

    //线检测
	auto t3 = Clock::now();
    double theta;
	theta = GetTheta(src_re, src_canny);
    res.angel = theta;
	if(theta == 999)
	{
    	//res.x = 999;
    	//res.y = 999;
		return res; 
	}
    auto t4 = Clock::now();

	cout << "步骤一前置时间为：" << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+9 <<'\n';
    cout << "步骤一塔检测时间为：" << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() / 1e+9 <<'\n';
    cout << "步骤一线检测时间为：" << std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() / 1e+9 <<'\n';
    cout << "步骤一总时间为：" << std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t1).count() / 1e+9 <<'\n';

    if(debug)
    {
        string text = "angel: " + to_string(theta) + " x: " + to_string(res.x) + " y: " + to_string(res.y);
        putText(src_re, text, Point(10,60), FONT_HERSHEY_SIMPLEX, 1 , (0,0,255), 2 , 8);
	    rectangle(src_re, Point(region_x.first, region_y.first), Point(region_x.second, region_y.second), Scalar( 0, 0, 255), 2, 8);//绘制塔架区域
        circle( src_re, Point(t.first, t.second), 5,  Scalar(0, 0, 255), 1, 8, 0 );
        res.image = src_re;
        //imwrite("first.png", src_re);
    }

    return res; 
}

//第二步
IDENT_INTERFACE secondDetection(Mat src, bool debug)
{
    //存储结果
    IDENT_INTERFACE res;
	res.target = 2;
    int x = src.rows / 480; // 缩放比例

    //时间戳
    typedef std::chrono::high_resolution_clock Clock;

    //前置部分
    auto t1 = Clock::now();

	src = ImageSize(src, 1);
    res.image = src;
	Mat src_re = src;
	src = ImageSplit(src);
    Mat src_canny = ImageCanny(src);
    //Mat src_tower = ImageSize(src, 1);
    Mat src_harris = ImageHarris(src);
    //Mat src_re_tower = ImageSize(src_re, 1);
    //Mat src_tower_canny = ImageSize(src_canny, 1);

    Region region_x, region_y;
	region_x = VerticalProjection(src_canny, src_harris);
	region_y = HorizonProjection(src_canny, src_harris);

    //塔检测 
    auto t2 = Clock::now();
    //Region t = tower_detection_two(src, src_re, src_canny, src_harris, region_x, region_y);

    //线
    auto t3 = Clock::now();
    double mid = DrawLine(src_re, src_canny, region_x);

    Region center;
    center.first = mid;
    center.second = (region_y.first + region_y.second) / 2;

    if(mid == 999)
    {
        res.x = 999;
        res.y = 999;
        res.angel = 999;
    }

    else
    {
        //cout<<center.first<<" "<<center.second<<endl;
        res.x = center.first * x;
        res.y = center.second * x;
        res.angel = 0;

    }

    auto t4 = Clock::now(); 
	    //cout<<center.first<<" "<<center.second<<endl;
    res.a = abs(region_x.second - region_x.first) * x;
    res.b = abs(region_y.second - region_y.first) * x;
    if(debug)
    {

        string text = "angel: " + to_string(res.angel) + " x: " + to_string(res.x) + " y: " + to_string(res.y);
        putText(src_re, text, Point(10,50), FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, 8);
        
        circle( src_re, Point(center.first, center.second), 5,  Scalar(0, 0, 255), 2, 8, 0 );
	    rectangle(src_re, Point(region_x.first, region_y.first), Point(region_x.second, region_y.second), Scalar( 0, 0, 255), 2, 8);//绘制塔架区域
        //imwrite("second.png", src_re);
        res.image = src_re;
    }


    cout << "步骤二前置时间为：" << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+9 <<'\n';
    cout << "步骤二塔检测时间为：" << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() / 1e+9 <<'\n';
    cout << "步骤二线检测时间为：" << std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() / 1e+9 <<'\n';
    cout << "步骤二总时间为：" << std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t1).count() / 1e+9 <<'\n';
    
    return res; 
}

//第三步
IDENT_INTERFACE thirdDetection(Mat src, bool debug)
{
    IDENT_INTERFACE res;
	res.target = 3;

    //时间戳
    typedef std::chrono::high_resolution_clock Clock;
    auto t1 = Clock::now();

    //前置
    src = ImageSize(src, 1);
    res.image = src;
	Mat src_re = src;
	src = ImageSplit(src);
    Mat src_canny = ImageCanny(src);

    //霍夫直线检测初步处理
    vector<Vec4d> line_data;
    line_data = HoughSecondStep(src_canny);

    //平行线组
    vector<Vec4d> para_line;
    para_line = GetParaLine(line_data);

	if(para_line.size() == 0)
	{
        //res.x = 999;
        //res.y = 999;
        res.angel = 999;
        return res;
	}
    //合并平行线
    para_line = MergeParaLine(para_line);
    //cout << para_line.size() << endl;

    Region region_x;
    region_x.first = (src.cols / 2 - 10);
    region_x.second = (src.cols / 2 + 10);

    //保留90度直线  
    //vector<double> k_tan;
    vector<Vec4d> P_Line;
    for (int i = 0; i < para_line.size(); i++)
	{
        Vec4f temp = para_line[i];
        if(abs(abs(atan2(temp[1] - temp[3], temp[0] - temp[2]) * 180 / CV_PI) - 90) <= 10)
        {
            P_Line.push_back(temp);
        }     
	}

    //cout << P_Line.size() << endl;

    //没有90度左右的平行线
    if(P_Line.size() == 0)
    {
        //res.x = 999;
        //res.y = 999;
        res.angel = 999;
        return res;
    }

    double theta;
    for (int i = 0; i < P_Line.size(); i++)
	{
        Vec4f temp = P_Line[i];
        //line(src, Point(temp[0], temp[1]), Point(temp[2], temp[3]), (0,0,255), 3, CV_AA);     
        pair<double, double> result = Line(temp[0], temp[1], temp[2], temp[3]);
        //上边界y = 0
        double x_top = -result.second / result.first;
        //下边界
        double x_bound = (src.rows - result.second) / result.first;

        //cout << i << " " <<x_top << " " << x_bound << " " << region_x.first << " " << region_x.second << endl;
        
        if(x_top < region_x.second && x_top > region_x.first && x_bound < region_x.second && x_bound > region_x.first)
        {
            theta = atan2(temp[1] - temp[3], temp[0] - temp[2]) * 180 / CV_PI;
            if(theta <= -90)
                theta =  (abs(theta) - 90) * -1;
            else if(theta >= 90)
                theta = theta - 90;
        }   
	}   

    auto t2 = Clock::now();
    cout << "步骤三时间为：" << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+9 <<'\n';
    
    //不用移动x，y
    res.x = 0;
    res.y = 0;
    res.angel = theta;

    if(debug)
    {
	    //cout<<center.first<<" "<<center.second<<endl;
        string text = "angel: " + to_string(res.angel) + " x: " + to_string(res.x) + " y: " + to_string(res.y);
        putText(src_re, text, Point(10,50), FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, 8);

        circle( src_re, Point(src.cols / 2, src.rows / 2), 5,  Scalar(0, 0, 255), 2, 8, 0 );
	    rectangle(src_re, Point(region_x.first, 10), Point(region_x.second, src.rows - 10), Scalar( 0, 0, 255), 2, 8);//绘制塔架区域
        res.image = src_re;
        //imwrite("third.png", src_re);
    }

    return res;
}

IDENT_INTERFACE Interface(Mat src, int targetLabel, bool debug)
{
    IDENT_INTERFACE res;
	
    switch (targetLabel)
    {
    case 1:
        res = firstDetection(src, debug);
        break;
    case 2:
        res = secondDetection(src, debug);
        break;
    case 3:
        res = thirdDetection(src, debug);
        break;
    }
    return res;
}
