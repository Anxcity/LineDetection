#include "./include/Autocanny.h"


void AdaptiveFindThreshold(const cv::Mat src, double& low, double& high, int aperture_size)
{
	const int cn = src.channels();
	cv::Mat dx_img(src.rows, src.cols, CV_16SC(cn));
	cv::Mat dy_img(src.rows, src.cols, CV_16SC(cn));

	cv::Sobel(src, dx_img, CV_16S, 1, 0, aperture_size, 1, 0, cv::BORDER_DEFAULT);
	cv::Sobel(src, dy_img, CV_16S, 0, 1, aperture_size, 1, 0, cv::BORDER_DEFAULT);

	const int width = dx_img.cols;
	const int height = dx_img.rows;

	// 计算边缘的强度, 并存于图像中
	float maxv = 0;
	cv::Mat img_dxy = cv::Mat(height, width, CV_32FC1);
	for (int i = 0; i < height; i++)
	{
		const short* _dx = (short*)(dx_img.data + dx_img.step*i);
		const short* _dy = (short*)(dy_img.data + dy_img.step*i);
		float* _image = (float *)(img_dxy.data + img_dxy.step*i);
		for (int j = 0; j < width; j++)
		{
			_image[j] = (float)(abs(_dx[j]) + abs(_dy[j]));
			maxv = maxv < _image[j] ? _image[j] : maxv;
			
			//if(i<height*0.25 && j<width*0.25)std::cout << _image[j] << ",";	
		}
		//if (i<height*0.25 )std::cout << std::endl;
	}
	//std::cout << std::endl << std::endl;
	//std::cout << img_dxy << std::endl;

	//std::cout << std::endl << std::endl;
	//std::cout << "maxv: " << maxv << std::endl;

	// 计算直方图
	int hist_size = maxv;// (int)(hist_size > maxv ? maxv : hist_size);//255
	//定义一个变量用来存储 单个维度 的数值的取值范围    
	float midRanges[] = { 0, maxv };//---------------------------从cvCalcHist换成calcHist函数，要记得改这个范围（不是0~256！！！）
	//确定每个维度的取值范围，就是横坐标的总数    
	const float *ranges[] = { midRanges };
	double  PercentOfPixelsNotEdges = 0.7;
	//range[1] = maxv;

	cv::Mat hist_dst;
	bool uniform = true;
	bool accumulate = false;
	//需要计算图像的哪个通道（bgr空间需要确定计算 b或g或r空间）    
	const int channels[1] = { 0 };
	calcHist(&img_dxy, 1, channels, cv::Mat(), hist_dst, 1, &hist_size, ranges, uniform, accumulate);

	int total = (int)(height * width * PercentOfPixelsNotEdges);
	float sum = 0;

	//std::cout << hist_dst.cols << ",,,,,,,,," << hist_dst.rows << std::endl;//1,,,,,,1156
	//std::cout << "hist_size" << hist_size << std::endl;//1,,,,,,255
	//std::cout << "hist_dst" << hist_dst << std::endl;


	//寻找最大梯度值
	float Hmax = 0.0f;
	float HmaxNum = 0.0f;
	float *h = (float*)hist_dst.data;
	for (int i=0; i < hist_size; i++)
	{
		if (h[i] > HmaxNum)
		{
			HmaxNum = h[i];
			Hmax = i;
		}
		//std::cout << i << ":" << h[i] << " ";
		//if ((float)hist_dst.data[i] > HmaxNum)
		//{
		//	HmaxNum = (float)hist_dst.data[i];
		//	Hmax = i;
		//}
	}
	//std::cout << "最大梯度值，数目：" << Hmax << "," << HmaxNum << std::endl;

	//计算像素最值梯度方差Emax
	float Emax = 0.0f;
	h = (float*)hist_dst.data;
	for (int i = 0; i < hist_size; i++)
	{
		float N = h[i];
		if (N > 0) {
			float temp = (i - Hmax)*(i - Hmax) / N;
			temp = temp > 1.0f ? 1.0f : temp;
			Emax += temp;
			//std::cout << temp << " ";
		}
	}
	//std::cout << "像素最值梯度方差：" << Emax << std::endl << std::endl;
	high = Hmax + Emax;

	// 计算高低门限
	//high = (i + 1) * maxv / hist_size;
	low = high * 0.4;

}
