# LineDetection
## 主要接口函数
* IDENT_INTERFACE Interface(Mat src, int targetLabel, bool debug) 按照最新的接口文档接收3个参数，返回一个包含6个参数的结构体

## 一些更新
* 在速度与准确率之间找了一个平衡，目前处理时图片缩放到800*600
* 优化了塔架检测算法
* 优化了线检测算法