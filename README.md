# LineDetection
## 主要借口函数
* firstDetection(Mat src)->vector 接收一个Mat参数，返回一个队列
* secondDetection(Mat src, bool debug)->vector 接收一个Mat参数和一个debug参数，当debug为0时正常返回一个队列，debug为1时画出塔和线的边框