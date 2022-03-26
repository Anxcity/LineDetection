#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
//#define GAP 48
#define MIDDLE_TEXT 0
#define BOTTOM_TEXT 1
//#define CAPTION_TEXT 2

CDetection::~CDetection()
{
	if (location.size()!=0)
		location.clear();
	if (blockLocation.size()!=0)
		blockLocation.clear();
	if (blockNum.size()!=0)
		blockNum.clear();
}

void CDetection::DWT(IplImage *pImage, int nLayer) //��ά��ɢС���任
{
	// ִ������
   if (pImage)
   {
      if (pImage->nChannels == 1 &&
         pImage->depth == IPL_DEPTH_32F &&
         ((pImage->width >> nLayer) << nLayer) == pImage->width &&
         ((pImage->height >> nLayer) << nLayer) == pImage->height)
      {
         int     i, x, y, n;
         float   fValue   = 0;
         float   fRadius  = sqrt(2.0f);
         int     nWidth   = pImage->width;
         int     nHeight  = pImage->height;
         int     nHalfW   = nWidth / 2;
         int     nHalfH   = nHeight / 2;
         float **pData    = new float*[pImage->height];
         float  *pRow     = new float[pImage->width];
         float  *pColumn  = new float[pImage->height];
         for (i = 0; i < pImage->height; i++)
         {
            pData[i] = (float*) (pImage->imageData + pImage->widthStep * i);
         }
         // ���С���任
         for (n = 0; n < nLayer; n++, nWidth /= 2, nHeight /= 2, nHalfW /= 2, nHalfH /= 2)
         {
            // ˮƽ�任
            for (y = 0; y < nHeight; y++)
            {
               // ��ż����
               memcpy(pRow, pData[y], sizeof(float) * nWidth);
               for (i = 0; i < nHalfW; i++)
               {
                  x = i * 2;
                  pData[y][i] = pRow[x];
                  pData[y][nHalfW + i] = pRow[x + 1];
               }
               // ����С���任
               for (i = 0; i < nHalfW - 1; i++)
               {
                  fValue = (pData[y][i] + pData[y][i + 1]) / 2;
                  pData[y][nHalfW + i] -= fValue;
               }
               fValue = (pData[y][nHalfW - 1] + pData[y][nHalfW - 2]) / 2;
               pData[y][nWidth - 1] -= fValue;
               fValue = (pData[y][nHalfW] + pData[y][nHalfW + 1]) / 4;
               pData[y][0] += fValue;
               for (i = 1; i < nHalfW; i++)
               {
                  fValue = (pData[y][nHalfW + i] + pData[y][nHalfW + i - 1]) / 4;
                  pData[y][i] += fValue;
               }
               // Ƶ��ϵ��
               for (i = 0; i < nHalfW; i++)
               {
                  pData[y][i] *= fRadius;
                  pData[y][nHalfW + i] /= fRadius;
               }
            }
            // ��ֱ�任
            for (x = 0; x < nWidth; x++)
            {
               // ��ż����
               for (i = 0; i < nHalfH; i++)
               {
                  y = i * 2;
                  pColumn[i] = pData[y][x];
                  pColumn[nHalfH + i] = pData[y + 1][x];
               }
               for (i = 0; i < nHeight; i++)
               {
                  pData[i][x] = pColumn[i];
               }
               // ����С���任
               for (i = 0; i < nHalfH - 1; i++)
               {
                  fValue = (pData[i][x] + pData[i + 1][x]) / 2;
                  pData[nHalfH + i][x] -= fValue;
               }
               fValue = (pData[nHalfH - 1][x] + pData[nHalfH - 2][x]) / 2;
               pData[nHeight - 1][x] -= fValue;
               fValue = (pData[nHalfH][x] + pData[nHalfH + 1][x]) / 4;
               pData[0][x] += fValue;
               for (i = 1; i < nHalfH; i++)
               {
                  fValue = (pData[nHalfH + i][x] + pData[nHalfH + i - 1][x]) / 4;
                  pData[i][x] += fValue;
               }
               // Ƶ��ϵ��
               for (i = 0; i < nHalfH; i++)
               {
                  pData[i][x] *= fRadius;
                  pData[nHalfH + i][x] /= fRadius;
               }
            }
         }
         delete[] pData;
         delete[] pRow;
         delete[] pColumn;
      }
   }
}

//���ڱ�Եͼ����оֲ���ֵ��
IplImage* CDetection::Binary(IplImage *img, int size)
{
	int i,j;  //ѭ������
	int width = img->width;    //Դͼ�����                                
    int height = img->height;  //ԭͼ��߶�                                  
    int winH = size;   //�������ڸ߶�                                      
    int winW = size;   //�������ڿ���                                      
	CvRect rect;
    //��ͼ�񻺳���  
    IplImage*  temp = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
	//��ʼ����ͼ��Ϊԭʼͼ��
	cvCopy(img, temp, NULL);

	for (i = 0; i < height - winH + 1; i += winH)
	{
		for (j = 0; j < width - winW + 1; j += winW)
		{
			rect.x = j;
			rect.y = i;
			rect.width = winW;
			rect.height = winH;
			cvSetImageROI(temp, rect);
			cvThreshold(temp, temp, 0, 255, CV_THRESH_OTSU);
			cvResetImageROI(temp);
		}
	}

//	cvCopy(temp, img, NULL);
	return temp;
}

IplImage* CDetection::RowProjection(IplImage *img, std::vector<int> &hpNum) //ˮƽ����ͶӰ
{
	int sum = 0;
	uchar data;
	int height = img->height;
	int width = img->width;
	int i, j, k;
	IplImage* temp;
	temp = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			((uchar*)(temp->imageData + temp->widthStep * i))[j] = 0;
		}
	}

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			data = ((uchar*)(img->imageData + img->widthStep * i))[j];

			if (data)
			{
				sum++;
			}
		}

		hpNum.push_back(sum);


		for (k = 0; k < sum; k++)
		{
			((uchar*)(temp->imageData + temp->widthStep * i))[k] = 255;
		}

		sum = 0;
	}

	return temp;
}

IplImage* CDetection::ColProjection(IplImage* img, BORDER border, std::vector<int> &vpNum) //��ֱ����ͶӰ
{
	int sum = 0;
	uchar data;
	int height = img->height;
	int width = img->width;
	int i, j, k;
	IplImage *temp;
	CvSize size;
	
	size = cvSize(border.right - border.left + 1, border.bottom - border.top + 1);
	temp = cvCreateImage(size, IPL_DEPTH_8U, 1);

	for (i = 0; i < size.height; i++)
	{
		for (j = 0; j < size.width; j++)
		{
			((uchar*)(temp->imageData + temp->widthStep * i))[j] = 0;
		}
	}

	for (i = 0; i < width; i++)
	{
		for (j = border.top; j < border.bottom; j++)
		{
			data = ((uchar*)(img->imageData + img->widthStep * j))[i];
			if (data)
			{
				sum++;
			}
		}

		vpNum.push_back(sum);

		for (k = 0; k < sum; k++)
		{
			((uchar*)(temp->imageData + temp->widthStep * (border.bottom - border.top - k)))[i] = 255;
		}

		sum = 0;
	}

	return temp;
}

std::vector<int> CDetection::Filter(IplImage* img, std::vector<int>& hpNum)
{
	int i, j;
	int sum = 0;
	int avg = 0;
	int max = 0;
	std::vector<int> rowNO;
	uchar data;
	int height = img->height;
	int width = img->width;
	int minWidth = 32;

	for (i = 0; i < hpNum.size(); i++)
	{
		sum += hpNum[i];
	}

//	avg = sum / height; //�����������ֵ�ľ�ֵ

	rowNO.push_back(0);

	for (i = 0; i < hpNum.size(); i++) //ͳ������ֵ���ھ�ֵ���к�
	{
		if (hpNum[i] < minWidth)
		{
			rowNO.push_back(i);
		}

		if (i >= height * 3 / 4)
		{
			data = ((uchar*)(img->imageData + img->widthStep * i))[width * 3 / 4];
			if (data)
			{
				rowNO.push_back(i);
			}
		}
	}

	rowNO.push_back(img->height - 1);

	for (i = 0; i < rowNO.size(); i++) //С�ھ�ֵ��������ֵ����
	{
		for (j = 0; j < hpNum[rowNO[i]]; j++)
		{
			((uchar*)(img->imageData + img->widthStep * rowNO[i]))[j] = 0;
		}

		hpNum[rowNO[i]] = 0;
	}

	return rowNO;
}

void CDetection::RowLocation(IplImage* img, std::vector<int> rowNO, int baseline1, int baseline2, int minHeight) //ˮƽͶӰ�߽�ȷ��
{
	int i;
	BORDER border;
	int rowNum = 0;
	int num = 0;
	int height = img->height;
	int width = img->width;

	int line = 0;
//	int top, bottom;
//	int topBorder, bottomBorder;

	for (i = 0; i < rowNO.size() - 1; i++) //��¼��ѡ������λ�ã�ͳ�ƺ�ѡ��������Ŀ
	{
		if (rowNO[i + 1] < baseline1 || rowNO[i] > baseline2)
		{
			if (rowNO[i + 1] - rowNO[i] >= minHeight) //������С�иߵ��м�¼Ϊ��ѡ������
			{
				if (abs(rowNO[i + 1] - height) > 16 && rowNO[i] > 16)
				{
					border.top = rowNO[i] - 1;
					if (border.top < 0)
					{
						border.top = 0;
					}
					border.bottom = rowNO[i + 1] + 1;
					if (border.bottom > height - 1)
					{
						border.bottom = height - 1;
					}
					border.left = 0;
					border.right = width - 1;
					location.push_back(border);
					rowNum++;
				}				
			}
		}
	}
		
	rowNO.clear();
}


void CDetection::ColLocation(IplImage *pSrc, IplImage *img, std::vector<BORDER> &rowLocation) //��ֱͶӰ�߽�ȷ��
{
	int i, j;
	IplImage *temp;
	CString title = _T("");
	int titleNum = 1;
	int num = 0;
	int minWidth = 8;
	std::vector<int> vpNum;
	std::vector<int> colNO;
	BORDER border;
	int threshold;
	int height;
//	bool flag;

	for (i = 0; i < location.size(); i++)
	{
		temp = ColProjection(img, location[i], vpNum); //���ˮƽ�߽���д�ֱͶӰ
// 		IplConvKernel* element = NULL;
// 		element = cvCreateStructuringElementEx(5, 1, 0, 0, CV_SHAPE_RECT, NULL);
// 		cvDilate(temp, temp, element, 1);
// 		title.Format("��ֱͶӰ%d",titleNum++);
//   		cvvNamedWindow(title, 1);  
//  		cvvShowImage(title, temp);
		//cvNamedWindow("title", 1);  
		//cvShowImage("title", temp);
		//cvWaitKey();

		threshold = temp->height * 0.1;

		colNO.push_back(0);
		for (j = 0; j < vpNum.size(); j++)
		{
			if (vpNum[j] < threshold)
			{
				colNO.push_back(j);
			}
		}
		colNO.push_back(temp->width - 1);

		for (j = 0; j < colNO.size() - 1; j++)
		{
			if (colNO[j + 1] - colNO[j] > minWidth)
			{
				border.top = location[i].top;
				border.bottom = location[i].bottom;
				height = border.bottom - border.top + 1;
				border.left = colNO[j] - temp->height / 2;
				if (border.left < 0)
				{
					continue;
				}
				border.right = colNO[j + 1] + temp->height / 2;
				if (border.right > temp->width)
				{
					continue;
				}

// 				flag = Verify(img, border);
// 				if (flag == false)
// 				{
					blockLocation.push_back(border);
					num++;
// 				}						
			}
		}

		blockNum.push_back(num);

		vpNum.clear();
		colNO.clear();
		cvReleaseImage(&temp);
	}

	Localize(pSrc, img, blockNum, rowLocation);
}

//��ͨ�����
bool CDetection::Contour(IplImage* img)
{
	bool flag = false;
	IplImage* pContourImg = NULL;


	CvMemStorage * storage = cvCreateMemStorage(0);
	CvSeq* cont = 0;
	CvSeq* prev_seq = 0;
	CvSeq* first_seq = 0;
	int mode = CV_RETR_EXTERNAL;
	CvContourScanner scanner = 0;

	double sum = 0;
	double avg = 0;
	int num = 0;

	pContourImg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
	cvCopy(img, pContourImg, NULL);
	cvFindContours(pContourImg, storage, &cont, sizeof(CvContour), mode, CV_CHAIN_APPROX_SIMPLE);

	for (; cont; cont = cont->h_next) 
	{ 
		double contArea= fabs(cvContourArea(cont, CV_WHOLE_SEQ));//�ҳ����������������С
		sum += contArea;
		num++;
	}

	avg = sum / num;

	if (avg < 100)
	{
		flag = true;
	}
	cvReleaseImage(&pContourImg);
	cvReleaseMemStorage(&storage);
	return flag;
}

bool CDetection::Verify(IplImage *edge, BORDER border) //�ı�����֤����
{
	bool areaFlag = false; //����ж�
	bool aspectFlag = false; //���߱��ж�
	bool edgeDensityFlag = false; //��Ե�ܶ��ж�
	bool contourFlag = false; //��ͨ������ж�
	bool totalFlag = false; //�жϽ��
	double width = border.right - border.left + 1;
	double height = border.bottom - border.top + 1;
	double ratio = width / height;
	double area = width * height;
	int i, j;
	double edgeNum = 0; //�����б�Ե������
	uchar data;
//	CvRect rect;

// 	rect.x = border.left;
// 	rect.y = border.top;
// 	rect.width = border.right - border.left + 1;
// 	rect.height = border.bottom - border.top + 1;
// 	cvSetImageROI(edge, rect);
// 	contourFlag = Contour(edge);

	//���㵱ǰblock���еı�Ե������
	for (i = border.top; i < border.bottom; i++)
	{
		for (j = border.left; j < border.right; j++)
		{
			data = ((uchar*)(edge->imageData + edge->widthStep * i))[j];
			if (data)
			{
				edgeNum++;
			}
		}
	}

	//�����Ե���ܶȣ���Ե������/block���
	double edgeRatio = edgeNum / area;

	if (area < 800)
	{
		areaFlag = true;
	}

	if (ratio < 2)
	{
		aspectFlag = true;
	}

	if (edgeRatio < 0.1)
	{
		edgeDensityFlag = true;
	}

	if (aspectFlag || areaFlag || edgeDensityFlag)
	{
		totalFlag = true;
	}

	cvResetImageROI(edge);
	return totalFlag;
}

void CDetection::Morphology(IplImage *img) //��̬ѧ����
{
	IplConvKernel* element = NULL;
	element = cvCreateStructuringElementEx(3, 1, 0, 0, CV_SHAPE_RECT, NULL);
//	cvMorphologyEx(img, img, temp, element, CV_MOP_OPEN, 1);
	cvDilate(img, img, element, 1);
	cvReleaseStructuringElement(&element);
	
}

void CDetection::Draw(IplImage* image, BORDER border) //�ı����򻭿��ʾ
{
	int i;
	int left, right, top, bottom;

	left = border.left;
	right = border.right;
	top = border.top;
	bottom = border.bottom;

	for(i = left; i <= right; i++)
	{
		//�ϱ߽�
		((uchar*)(image->imageData + image->widthStep * top))[i * 3] = 0;
		((uchar*)(image->imageData + image->widthStep * top))[i * 3 + 1] = 255;
		((uchar*)(image->imageData + image->widthStep * top))[i * 3 + 2] = 0;
		//�±߽�
		((uchar*)(image->imageData + image->widthStep * bottom))[i * 3] = 0;
		((uchar*)(image->imageData + image->widthStep * bottom))[i * 3 + 1] = 255;
		((uchar*)(image->imageData + image->widthStep * bottom))[i * 3 + 2] = 0;

		((uchar*)(image->imageData + image->widthStep * (top + 1)))[i * 3] = 0;
		((uchar*)(image->imageData + image->widthStep * (top + 1)))[i * 3 + 1] = 255;
		((uchar*)(image->imageData + image->widthStep * (top + 1)))[i * 3 + 2] = 0;
		//�±߽�
		if ( bottom+1 <= image->height  )
		{
			((uchar*)(image->imageData + image->widthStep * (bottom + 1)))[i * 3] = 0;
			((uchar*)(image->imageData + image->widthStep * (bottom + 1)))[i * 3 + 1] = 255;
			((uchar*)(image->imageData + image->widthStep * (bottom + 1)))[i * 3 + 2] = 0;
		}

		((uchar*)(image->imageData + image->widthStep * (top + 2)))[i * 3] = 0;
		((uchar*)(image->imageData + image->widthStep * (top + 2)))[i * 3 + 1] = 255;
		((uchar*)(image->imageData + image->widthStep * (top + 2)))[i * 3 + 2] = 0;
		//�±߽�
		if ( bottom+2 <= image->height )
		{
			((uchar*)(image->imageData + image->widthStep * (bottom + 2)))[i * 3] = 0;
			((uchar*)(image->imageData + image->widthStep * (bottom + 2)))[i * 3 + 1] = 255;
			((uchar*)(image->imageData + image->widthStep * (bottom + 2)))[i * 3 + 2] = 0;
		}
	}

	for(i = top; i <= bottom; i++)
	{
		//��߽�
		((uchar*)(image->imageData + image->widthStep * i))[left * 3] = 0;
		((uchar*)(image->imageData + image->widthStep * i))[left * 3 + 1] = 255;
		((uchar*)(image->imageData + image->widthStep * i))[left * 3 + 2] = 0;
		//�ұ߽�
		((uchar*)(image->imageData + image->widthStep * i))[right * 3] = 0;
		((uchar*)(image->imageData + image->widthStep * i))[right * 3 + 1] = 255;
		((uchar*)(image->imageData + image->widthStep * i))[right * 3 + 2] = 0;

		((uchar*)(image->imageData + image->widthStep * i))[(left + 1) * 3] = 0;
		((uchar*)(image->imageData + image->widthStep * i))[(left + 1) * 3 + 1] = 255;
		((uchar*)(image->imageData + image->widthStep * i))[(left + 1) * 3 + 2] = 0;
		//�ұ߽�
		if ( right+1 <= image->width )
		{
			((uchar*)(image->imageData + image->widthStep * i))[(right + 1) * 3] = 0;
			((uchar*)(image->imageData + image->widthStep * i))[(right + 1) * 3 + 1] = 255;
			((uchar*)(image->imageData + image->widthStep * i))[(right + 1) * 3 + 2] = 0;
		}

		((uchar*)(image->imageData + image->widthStep * i))[(left + 2) * 3] = 0;
		((uchar*)(image->imageData + image->widthStep * i))[(left + 2) * 3 + 1] = 255;
		((uchar*)(image->imageData + image->widthStep * i))[(left + 2) * 3 + 2] = 0;
		//�ұ߽�
		if ( right+2 <= image->width )
		{
			((uchar*)(image->imageData + image->widthStep * i))[(right + 2) * 3] = 0;
			((uchar*)(image->imageData + image->widthStep * i))[(right + 2) * 3 + 1] = 255;
			((uchar*)(image->imageData + image->widthStep * i))[(right + 2) * 3 + 2] = 0;
		}
	}
}

void CDetection::Localize(IplImage* src, IplImage *edge, std::vector<int> blockNum, std::vector<BORDER> &rowLocation) //�ı�����λ
{
	int i;
	int current = 1;
	int num = 0; //ÿ���ı��������
	int left, right;
	BORDER border;
	bool flag;
	int gap=src->height * 0.1; 

	if (location.size() > 0 && blockLocation.size() > 0)
	{
		for (i = 0; i < location.size(); i++)
		{
			if (num>=blockLocation.size())
			{
				num=blockLocation.size()-1;
			}
			border.top = blockLocation[num].top;
			border.bottom = blockLocation[num].bottom;
			border.left = blockLocation[num].left;
			border.right = blockLocation[num].right;
			num = blockNum[i];

			while (current < num)
			{
				if (current>=blockLocation.size())
				{
					current=blockLocation.size()-1;
				}
				left = border.right;
				right = blockLocation[current].left;

				if (right >= left)
				{
					if (right - left < gap)
					{
						border.right = blockLocation[current].right;
						current++;
					}
					else
					{
						rowLocation.push_back(border);
						border.top = blockLocation[current].top;
						border.bottom = blockLocation[current].bottom;
						border.left = blockLocation[current].left;
						border.right = blockLocation[current].right;
						current++;
					}
				}
				else
				{
					border.right = blockLocation[current].right;
					current++;
				}
			} 
			
			flag = Verify(edge, border);
			if (flag == false)
			{
				rowLocation.push_back(border);
			}		
		}
	}	

 	for (i = 0; i < rowLocation.size(); i++)
 	{
 		Draw(src, rowLocation[i]);
 	}

/*
	cvNamedWindow("localize", 1); 
	cvShowImage("localize", src);
*/
}

int CDetection::Classify(IplImage *hProjectin, std::vector<int> hpNum)
{
	int topBoder = hProjectin->height / 5;
	int bottomBorder = hProjectin->height / 5 * 4;
	int imageType = -1;
	BORDER border;
	std::vector<int> background;
	std::vector<int> vBack;
 	std::vector<BORDER> middleRow;
	std::vector<int> rowDis;
	bool flag = false;
	int i;
	//ͳ���м�3/5������ͶӰ����Ϊ0����
	background.push_back(topBoder);
	for (i = topBoder; i < bottomBorder; i++)
	{
		if (hpNum[i] == 0)
		{
			background.push_back(i);
		}
	}
	background.push_back(bottomBorder);

	for (i = 0; i < background.size() - 1; i++)
	{
		if (background[i + 1] - background[i] > 16 && background[i + 1] - background[i] < 80)
		{
			border.top = background[i] - 1;
			border.bottom = background[i + 1] - 1;
			border.left = 0;
			border.right = hProjectin->width - 1;
			middleRow.push_back(border);
		}
	}

	if (middleRow.size() == 2)
	{
		int first, second;
		first = middleRow[0].bottom - middleRow[0].top;
		second = middleRow[1].bottom - middleRow[1].top;
		
		if (first >= 56 && second >= 56 && abs(first - second) < 10) 
		{
			flag = true;
		}		
	}

	if (middleRow.size() > 2)
	{
		for (i = 0; i < middleRow.size() - 1; i++) //�ж�������֮��ľ���
		{
			int dis = middleRow[i + 1].top - middleRow[i].bottom;
			if (dis > 8)
			{
				rowDis.push_back(dis);
			}			
		}

		if (rowDis.size()> 1)
		{
			for (i = 0; i < rowDis.size() - 1; i++)
			{
				if (abs(rowDis[i + 1] - rowDis[i]) < 5)
				{
					flag = true;
				}
				else
				{
					flag = false;
				}
			}
		}		
	}

	if (flag) //���ͼ���м�������г������в����м�������5������֮�ڣ����Ϊȫ������
	{
		imageType = MIDDLE_TEXT;
	}
	else
	{
		imageType = BOTTOM_TEXT;
	}


	background.clear();
	vBack.clear();
 	middleRow.clear();
	rowDis.clear();
// 	leftCol.clear();
// 	rightCol.clear();
// 	colDis.clear();
	return imageType;
}

IplImage* CDetection::GetEdge(IplImage *pSrc)
{
	IplImage *pV, *pH, *pD, *pAdd, *pVTemp, *pHTemp, *pDTemp, *temp, *pAddResize;
	IplImage *copySrc, *edge;
	CvSize newSize;
	int i, j;
	CvScalar s;

	// С���任����
	int nLayer = 1;
	// �����ɫͼ��
	copySrc = cvCreateImage(cvGetSize(pSrc), pSrc->depth, pSrc->nChannels);
	IplImage *src = cvCreateImage(cvGetSize(pSrc), IPL_DEPTH_8U, 1);
	cvCopy(pSrc, copySrc, NULL);
	cvCvtColor(pSrc, src, CV_RGB2GRAY);
	// ����С��ͼ���С
	CvSize size = cvGetSize(pSrc);
	if ((pSrc->width >> nLayer) << nLayer != pSrc->width)
	{
		size.width = ((pSrc->width >> nLayer) + 1) << nLayer;
	}
	if ((pSrc->height >> nLayer) << nLayer != pSrc->height)
	{
		size.height = ((pSrc->height >> nLayer) + 1) << nLayer;
	}
	// ����С��ͼ��
	IplImage *pWavelet = cvCreateImage(size, IPL_DEPTH_32F, 1);
	if (pWavelet)
	{
		// С��ͼ��ֵ
		cvSetImageROI(pWavelet, cvRect(0, 0, pSrc->width, pSrc->height));
		cvConvertScale(src, pWavelet, 1, 0);
		cvResetImageROI(pWavelet);
		// ��ɫͼ��С���任
		IplImage *pImage = cvCreateImage(cvGetSize(pWavelet), IPL_DEPTH_32F, 1);
		if (pImage)
		{
			cvSetImageCOI(pWavelet, 0);
			cvCopy(pWavelet, pImage, NULL);
			// ��ά��ɢС���任
			DWT(pImage, nLayer);
			// ��ά��ɢС���ָ�
			// IDWT(pImage, nLayer);
			cvCopy(pImage, pWavelet, NULL);
			cvSetImageCOI(pWavelet, 0);
		}
		cvReleaseImage(&pImage);
	// С���任ͼ��
	cvSetImageROI(pWavelet, cvRect(0, 0, pSrc->width, pSrc->height));
	cvConvertScale(pWavelet, src, 1, 0);
	cvResetImageROI(pWavelet); //���д����е���࣬���������������õı��ϰ��
	}
	cvReleaseImage(&pWavelet);

	// ��ʾͼ��pSrc
// 	cvNamedWindow("src", 1);
// 	cvShowImage( "src", src );

	cvSetImageROI(src, cvRect( src->width / 2, 0, src->width / 2, src->height / 2 )); //��ȡС��ˮƽ����
	pH = cvCloneImage(src);
//	cvNamedWindow("pH", 1); 
//	cvShowImage( "pH", pH );
	cvResetImageROI(src);

	cvSetImageROI(src, cvRect( 0, src->height / 2, src->width / 2, src->height / 2 )); //��ȡС����ֱ����
	pV = cvCloneImage(src);
//	cvNamedWindow("pV", 1); 
//	cvShowImage( "pV", pV ); 
	cvResetImageROI(src);

	cvSetImageROI(src, cvRect( src->width / 2, src->height / 2, src->width / 2, src->height / 2 )); //��ȡС���Խ��߷���
	pD = cvCloneImage(src);
//	cvNamedWindow("pD", 1); 
//	cvShowImage( "pD", pD ); 
	cvResetImageROI(src);

	pHTemp = cvCreateImage(cvGetSize(pH), pH->depth, pH->nChannels);
	pVTemp = cvCreateImage(cvGetSize(pH), pH->depth, pH->nChannels);
	pDTemp = cvCreateImage(cvGetSize(pH), pH->depth, pH->nChannels);
	temp = cvCreateImage(cvGetSize(pH), pH->depth, pH->nChannels);
	pAdd = cvCreateImage(cvGetSize(pH), pH->depth, pH->nChannels);

	cvMul(pH, pH, pHTemp, 1); //��������ͼ����кϲ�
	cvMul(pV, pV, pVTemp, 1);
	cvMul(pD, pD, pDTemp, 1);
	cvAdd(pHTemp, pVTemp, temp, NULL);
	cvAdd(temp, pDTemp, pAdd, NULL);

	//����
/*
	for (i = 0; i < pAdd->height; i++)
	{
		for (j = 0; j < pAdd->width; j++)
		{
			s = cvGet2D(pAdd, i, j);
			s.val[0] = cvSqrt(s.val[0]);
			cvSet2D(temp, i, j, s);
		}
	}*/

	
	
// 	cvNamedWindow("pAdd", 1); 
// 	cvShowImage( "pAdd", pAdd );
//	cvWaitKey(0);
	
	//ͼ��Ŵ�
	newSize.width = pAdd->width * 2;
	newSize.height = pAdd->height * 2;
	pAddResize = cvCreateImage(newSize, pAdd->depth, pAdd->nChannels);
	edge = cvCreateImage(cvGetSize(pAddResize), IPL_DEPTH_8U, 1);
	cvResize(pAdd, pAddResize, CV_INTER_LINEAR);
	cvThreshold(pAddResize, pAddResize, 0, 255, CV_THRESH_OTSU);
//  	cvNamedWindow("pAddResize", 1); 
//  	cvShowImage( "pAddResize", pAddResize );
	//��ֵ�˲�
	cvSmooth(pAddResize, pAddResize, CV_MEDIAN, 3, 0, 0, 0);
//  	cvNamedWindow("pAddResizeSmooth", 1); 
//  	cvShowImage( "pAddResizeSmooth", pAddResize );
	Morphology(pAddResize);
	cvCopy(pAddResize, edge, NULL);

	cvReleaseImage(&copySrc);
	cvReleaseImage(&src);
	cvReleaseImage(&pHTemp);
	cvReleaseImage(&pVTemp);
	cvReleaseImage(&pDTemp);
	cvReleaseImage(&pH);
	cvReleaseImage(&pV);
	cvReleaseImage(&pD);
	cvReleaseImage(&temp);
	cvReleaseImage(&pAdd);
	cvReleaseImage(&pAddResize);

	return edge;
}

int CDetection::DetectText(IplImage *pSrc, IplImage *result, std::vector<BORDER> &rowLocation) //����ı�������ں���
{
	IplImage *pRow, *pCol, *edge, *src;
	BORDER border;
	std::vector<int> hpNum; //ˮƽͶӰͼÿ�����ص�����
	std::vector<int> vpNum; //��ֱͶӰÿ�����ص�����
	std::vector<int> rowNO;
	int imageType = -1; //��Ƶͼ������
	int baseline1 = 0; 
	int baseline2 = 0;
	int borderline = 0;
	int height = pSrc->height;
	int width = pSrc->width;
	int minHeight;

	src = cvCreateImage(cvGetSize(pSrc), pSrc->depth, pSrc->nChannels);
	cvCopy(pSrc, src, NULL);
	border.left = 0;
	border.top = 0;
	border.right = src->width - 1;
	border.bottom = src->height - 1;

	edge = GetEdge(src);  //����С���任����ͼ���Ե
//  	cvNamedWindow("edge", 1); 
//  	cvShowImage( "edge", edge );
// 	cvWaitKey(0);


	pRow = RowProjection(edge, hpNum);  //ͼ��ˮƽͶӰ
	pCol = ColProjection(edge, border, vpNum);  //ͼ��ֱͶӰ
	//cvNamedWindow("pRow", 1); 
	//cvShowImage( "pRow", pRow );
	//cvNamedWindow("pCol", 1); 
	//cvShowImage( "pCol", pCol );
	rowNO = Filter(pRow, hpNum);
 //	cvNamedWindow("newRP", 1); 
 //	cvShowImage("newRP", pRow);
	
	if (type != 1)
	{
		imageType = Classify(pRow, hpNum);
	}
	else
	{
		imageType = type;
	}

	switch(imageType)
	{
	case MIDDLE_TEXT:
		baseline1 = 0;
		baseline2 = 0;
		minHeight = src->height * 0.04;
		RowLocation(pRow, rowNO, baseline1, baseline2, minHeight);
		ColLocation(src, edge, rowLocation);
		cvCopy(src, result, NULL);
		break;
	case BOTTOM_TEXT:
		type = 1;
		if (width > height)
		{
			baseline1 = 0;
			baseline2 = src->height / 3 * 2;
			minHeight = src->height * 0.04;
			RowLocation(pRow, rowNO, baseline1, baseline2, minHeight);
			ColLocation(src, edge, rowLocation);
			cvCopy(src, result, NULL);
		}
		else
		{
			baseline1 = src->height / 3;
			baseline2 = src->height / 3 * 2;
			minHeight = src->width * 0.04;
			RowLocation(pRow, rowNO, baseline1, baseline2, minHeight);
			//cvNamedWindow("test",1);
			//cvShowImage("test",edge);
			//cvWaitKey(0);
			ColLocation(src, edge, rowLocation);
			cvCopy(src, result, NULL);
		}	
		break;
	}

//	cvWaitKey(0);
	cvDestroyWindow("pSrc");
	cvDestroyWindow("src");
	cvDestroyWindow("pH");
	cvDestroyWindow("pV");
	cvDestroyWindow("pD");
	cvDestroyWindow("pAdd");
	cvDestroyWindow("pAddResize");
	cvDestroyWindow("pAddResizeSmooth");
	cvDestroyWindow("pRow");
	cvDestroyWindow("pCol");
	cvDestroyWindow("newRP");
	cvDestroyWindow("localize");
	cvDestroyWindow("edge");
	
//	cvReleaseImage(&pSrc);
	cvReleaseImage(&src);
	cvReleaseImage(&pRow);
	cvReleaseImage(&pCol);
	cvReleaseImage(&edge);
	hpNum.clear();
	rowNO.clear();
	vpNum.clear();

	return imageType;
}

void CDetection::Detection(const IplImage *image, std::vector<BORDER> &finalLocation)
{
	IplImage *pSrc = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
	cvCopy(image, pSrc, NULL);
	int imageType = -1;
	std::vector<BORDER> rowLocation;
	CvSize size;
	IplImage *transpose;
	IplImage *result2;
	std::vector<BORDER> colLocation;
//	BORDER border;
	int i; 
 	int min = pSrc->height;
	std::vector<BORDER>::iterator it;

	IplImage *result1 = cvCreateImage(cvGetSize(pSrc), pSrc->depth, pSrc->nChannels);
	//ˮƽ���
	CDetection cd;
	imageType = cd.DetectText(pSrc, result1, rowLocation);
	cd.~CDetection();
	if (imageType == 1)
	{
		//ȥ��������ˮƽ��
		IplImage *temp = cvCreateImage(cvGetSize(pSrc), pSrc->depth, pSrc->nChannels);
		cvCopy(pSrc, temp, NULL);
		int i, j, k;
		BORDER border;

		for (i = 0; i < rowLocation.size(); i++)
		{
			border.top = rowLocation[i].top;
			border.left = rowLocation[i].left;
			border.bottom = rowLocation[i].bottom;
			border.right = rowLocation[i].right;

			for (j = border.top; j < border.bottom; j++)
			{
				for (k = border.left; k < border.right; k++)
				{
					((uchar*)(temp->imageData + temp->widthStep * j))[k * 3 + 0] = 0;
					((uchar*)(temp->imageData + temp->widthStep * j))[k * 3 + 1] = 0;
					((uchar*)(temp->imageData + temp->widthStep * j))[k * 3 + 2] = 0;
				}
			}
		}

//	 	cvNamedWindow("temp", CV_WINDOW_AUTOSIZE);
//		cvShowImage("temp", temp);

//		cvWaitKey(0);
//		cvDestroyWindow("temp");
		//ͼ��ת��
		size.width = pSrc->height;
		size.height = pSrc->width;
		transpose = cvCreateImage(size, pSrc->depth, pSrc->nChannels);
		cvTranspose(temp, transpose);
		result2 = cvCreateImage(cvGetSize(transpose), transpose->depth , transpose->nChannels);
		//��ֱ���
		cd.DetectText(transpose, result2, colLocation);

		for (i = 0; i < rowLocation. size(); i++) //ˮƽ������ϵ�һ���߽���
		{
			int top = rowLocation[i].top;
			if (top < min)
			{
				min = top;
			}
			finalLocation.push_back(rowLocation[i]);
		}

		int numL, numR;
		std::vector<int> l, r;
		numR = numL = 0;
		for (i = 0; i < colLocation.size(); i++)
		{
			border.top = colLocation[i].left;
			if (border.top > pSrc->height / 2) //��ֱ�������ı�����ϱ߽����ˮƽ�������߽߱��ߣ�����
			{
				continue;
			}			
			
			border.bottom = colLocation[i].right;			
			if (border.bottom < pSrc->height / 3) //ͼ���Ϸ�1/3������ı��򣬹���
			{
				continue;
			}
			if (border.bottom > min)
			{
				border.bottom = min - 6;
			}

			border.left = colLocation[i].top;
			if (border.left > pSrc->width / 3 * 2)
			{
				numR++;
				r.push_back(border.left);
			}

			border.right = colLocation[i].bottom;
			if (border.right < pSrc->width / 3)
			{
				numL++;
				l.push_back(border.right);
			}

			finalLocation.push_back(border);
		}


		if (numL == 1)
		{
			for (it = finalLocation.begin(); it != finalLocation.end();)   
			{   
				if (l[0] == (*it).right) 
				{
					finalLocation.erase(it);  
					break;
				}
				else
					++it;
			}  
		}
		if (numR == 1)
			{			   
				for (it = finalLocation.begin(); it != finalLocation.end();)   
				{   
					if (r[0] == (*it).left) 
					{
						finalLocation.erase(it); 
						break;
					}
					else
						++it;
				}  
			}
	

		/*for (i = 0; i < finalLocation.size(); i++)
		{
			cd.Draw(pSrc, finalLocation[i]);
		}*/

// 		cvNamedWindow("final", CV_WINDOW_AUTOSIZE);
// 		cvShowImage("final", pSrc);

//		cvWaitKey(0);
//		cvDestroyWindow("final");
		
		cd.~CDetection();
		
		cvReleaseImage(&result2);
		cvReleaseImage(&transpose);
		cvReleaseImage(&temp);
		temp=NULL;
		result2 = NULL;
		transpose = NULL;
	}
	else
	{
		for (i = 0; i < rowLocation.size(); i++)
		{
			finalLocation.push_back(rowLocation[i]);
		}

	//	for (i = 0; i < finalLocation.size(); i++)
	//	{
	////		int top, bottom, left, right;
	////		top = finalLocation[i].top;
	////		bottom = finalLocation[i].bottom;
	////		left = finalLocation[i].left;
	////		right = finalLocation[i].right;

	//		//cd.Draw(pSrc, finalLocation[i]);
	//	}

		//cvNamedWindow("final", CV_WINDOW_AUTOSIZE);
		//cvShowImage("final", pSrc);

	}

//	cvWaitKey(1);
//	cvDestroyWindow("temp");
	cvReleaseImage(&result1);
	cvReleaseImage(&pSrc);
	
	result1 = NULL;
	pSrc = NULL;
	rowLocation.clear();
	colLocation.clear();
	//finalLocation.clear();
}

