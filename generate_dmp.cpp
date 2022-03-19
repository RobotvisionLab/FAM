#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <stdio.h>
#include <string>
#include <vector>

using namespace cv;
using namespace std;
using std::cout;
using std::endl;
using std::string;

vector<string> splitWithStl(const std::string &str, const std::string &pattern)
{
	std::vector<std::string> resVec;

	if ("" == str)
	{
		return resVec;
	}
	std::string strs = str + pattern;

	size_t pos = strs.find(pattern);
	size_t size = strs.size();

	while (pos != std::string::npos)
	{
		std::string x = strs.substr(0, pos);
		resVec.push_back(x);
		strs = strs.substr(pos + 1, size);
		pos = strs.find(pattern);
	}

	return resVec;
}

vector<string> img_name;
vector<Mat> read_images_in_folder(cv::String pattern)
{
	vector<cv::String> fn;
	glob(pattern, fn, false);

	vector<Mat> images;
	size_t count = fn.size();
	for (size_t i = 0; i < count; i++)
	{ 
		string name = splitWithStl(fn[i] , "/")[1];
		cout << name << endl;
		img_name.push_back(name);
		images.push_back(imread(fn[i], 0));
	}
	return images;
}


Mat distanceMap(Mat img)
{
	Mat mediaiImage;
	medianBlur(img, mediaiImage, 15);
	medianBlur(mediaiImage, mediaiImage, 15);
	medianBlur(mediaiImage, mediaiImage, 15);
	medianBlur(mediaiImage, mediaiImage, 15);

	Mat binaryImage;
	threshold(mediaiImage, binaryImage, 70, 255, THRESH_BINARY);


	Mat dmpImage;
	distanceTransform(binaryImage, dmpImage, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	Mat element;
	Mat dstImage;
	element = getStructuringElement(MORPH_RECT, Size(15, 15));
	morphologyEx(dmpImage, dstImage, MORPH_OPEN, element);
	cout << "====>dmp====>"<<endl;
	normalize(dstImage, dstImage, 0, 1, NORM_MINMAX);
	return dstImage*255;
}


int main()
{
	cv::String pattern = "img//*.png";
	vector<Mat> images = read_images_in_folder(pattern);
	cout << images.size() << endl;

	for (int i = 0; i < images.size(); i++)
	{
		cv::Mat src = images[i];
		cv::Mat img_dmp = distanceMap(src);
		string origin_name = img_name[i];
		string path = string("img_dmp/").append(origin_name);
		cv::imwrite(path, img_dmp);
	}

	return 0;
}