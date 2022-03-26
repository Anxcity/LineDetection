#ifndef AUTO_CANNY_H
#define AUTO_CANNY_H
#include <opencv2/opencv.hpp>

void AdaptiveFindThreshold(const cv::Mat src, double& low, double& high, int aperture_size);

#endif