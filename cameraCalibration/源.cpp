#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>  
#include <sstream>

#define start 1
#define finish 10

using namespace std;
using namespace cv;

uchar toZero(int a)
{
	if (a <= 5)
		return 0;
	if (a > 250)
		return 255;
	else
		return 100;
}

//transform chessboardCorners(2D) to world position(3D) with z = 0
bool backProject(const Mat& boardRot64,
	const Mat& boardTrans64,
	const Mat& cameraMatrix,
	const vector<Point2f>& imgPt,
	vector<Point3f>& worldPt) {
	if (imgPt.size() == 0) {
		return false;
	}
	else
	{
		Mat imgPt_h = Mat::zeros(3, imgPt.size(), CV_32F);
		for (int h = 0; h<imgPt.size(); ++h) {
			imgPt_h.at<float>(0, h) = imgPt[h].x;
			imgPt_h.at<float>(1, h) = imgPt[h].y;
			imgPt_h.at<float>(2, h) = 1.0f;
		}
		
		Mat Kinv64 = cameraMatrix.inv();
		Mat Kinv, boardRot, boardTrans;
		Kinv64.convertTo(Kinv, CV_32F);
		boardRot64.convertTo(boardRot, CV_32F);
		boardTrans64.convertTo(boardTrans, CV_32F);

		Mat worldImgPt = Mat::zeros(3, imgPt.size(), CV_32F);
		Mat rot3x3;
		Rodrigues(boardRot, rot3x3);

		Mat transPlaneToCam = rot3x3.inv()*boardTrans;

		for (int i = 0; i<imgPt.size(); ++i) {
			Mat col = imgPt_h.col(i);
			Mat worldPtcam = Kinv*col;
			Mat worldPtPlane = rot3x3.inv()*(worldPtcam);

			float scale = transPlaneToCam.at<float>(2) / worldPtPlane.at<float>(2);
			Mat worldPtPlaneReproject = scale*worldPtPlane - transPlaneToCam;

			Point3f pt;
			pt.x = worldPtPlaneReproject.at<float>(0);
			pt.y = worldPtPlaneReproject.at<float>(1);
			pt.z = 0;
			worldPt.push_back(pt);
			/*cout << "x= " << pt.x << endl;
			cout << "y= " << pt.y << endl;
			cout << "z= " << pt.z << endl;*/
		}
	}
	return true;
}

int main()
{
	ifstream ifile_cameraPicture;
	ifstream ifile_projectorDMD;
	ifstream ifile_projecotrPicure;
	char camPicFilename[40];
	char proPicFilename[40];
	char proDMDFilename[40];
	Size camPicSize;
	camPicSize.width = 1920;
	camPicSize.height = 1080;

	vector<vector<Point3f>> object_points_projector;//save the projector ObjectPoint
	vector<vector<Point3f>> object_points_camera;//save the camera ObjectPoint
	vector<vector<Point2f>> image_points_seq_projector;//save the DMD corners
	vector<vector<Point2f>> image_points_seq_DMD;//save the DMD corners
	vector<vector<Point2f>> image_points_seq_camera;//save the camera corners
	

	//find chessBoardCorners in camera 
	for (int i = start; i <= finish; i++)
	{
		sprintf(camPicFilename, "C:\\developer\\4.3\\%d.0.jpg", i);
		vector<Point2f> tempPointSet2d_camera;

		ifile_cameraPicture.open(camPicFilename);
		long int cornerSum = 0;
		float cornerX = 0;
		float cornerY = 0;
		Mat camPicGray;
		
		Mat camPic = imread(camPicFilename);
		//flip(camPic, camPic, 1);//***
	/*	imshow("camPic", camPic);
		waitKey();*/
		resize(camPic, camPic,Size(camPic.size().width / 2, camPic.size().height / 2));
		
		std::vector<cv::Point2f> corners;
		CvSize patternSize;
		patternSize.height = 9;
		patternSize.width = 14;

		bool ret = findChessboardCorners(camPic, patternSize, corners);
		if (!ret)
		{
			return -1;
		}
		else {
			
			cvtColor(camPic, camPicGray,CV_RGB2GRAY);
			find4QuadCornerSubpix(camPicGray, corners, Size(11, 11));
			cv::drawChessboardCorners(camPic, patternSize, corners, ret);
			
			for (int j = 0; j <14*9; j++)
			{
				corners[j].x = corners[j].x * 2;
				corners[j].y = corners[j].y * 2;
				/*cout << "x[" << j << "]= " << corners[j].x << endl;
				cout << "y[" << j << "]= " << corners[j].y << endl;*/
			}
			image_points_seq_camera.push_back(corners);
		}
	}



    //calculate the world point of the chessboard
	Size square_size;
	square_size.width = 50;
	square_size.height = 50;

	int k, j, t;
	for (t = start; t <= finish; t++)
	{
		vector<Point3f> tempPointSet;
		for (k = 0; k<9; k++)
		{
			for (j = 0; j <14; j++)
			{
				Point3f realPoint;

				realPoint.x = j*square_size.width;
				realPoint.y = k*square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points_camera.push_back(tempPointSet);
	}


	//calibrate the camera
	Mat1d cameraMatrix;
	Mat distCoeffs;
	vector<Mat>rvecsMat;
	vector<Mat>tvecsMat;
	double rms = calibrateCamera(object_points_camera, image_points_seq_camera, camPicSize, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
	cout << "camera_rms" << endl << rms << endl;
	cout << "cameraMatrix" << endl << cameraMatrix << endl;
	cout << "distCoeffs" << endl << distCoeffs << endl;
	cout << "R" << endl << rvecsMat[0] << endl;
	/*while (getchar())
	{

	}*/



	//find chessboardCorners in DMD
	Size DMDpatternSize;

	for (int i = start; i <= finish; i++)
	{
		sprintf(proDMDFilename, "C:\\developer\\4.3\\DMD%d.jpg", i);
		//vector<Point2f> tempPointSet2d_DMD;

		ifile_projectorDMD.open(proDMDFilename);

		Mat proPicGray;

		Mat proPic = imread(proDMDFilename);
		//flip(proPic, proPic, 1);//***
		std::vector<cv::Point2f> DMDcorners;
		CvSize DMDpatternSize;
		DMDpatternSize.height = 6;
		DMDpatternSize.width = 9;
		cvtColor(proPic, proPicGray, CV_RGB2GRAY);
		int rowNumber = proPicGray.rows;
		int colNumber = proPicGray.cols;

		for (int i = 0; i < rowNumber; i++)
		{
			for (int j = 0; j < colNumber; j++)
			{
				proPicGray.at<uchar>(i, j) = toZero((int)proPicGray.at<uchar>(i, j));
			}
		}
		bool DMDret = findChessboardCorners(proPicGray, DMDpatternSize, DMDcorners);
		if (!DMDret)
		{
			return -1;
		}
		else {
			find4QuadCornerSubpix(proPicGray, DMDcorners, Size(11, 11));
			cv::drawChessboardCorners(proPic, DMDpatternSize, DMDcorners, DMDret);

			image_points_seq_DMD.push_back(DMDcorners);
		}
	}


	//calculate projector chessboard corner world position
	for (int q = start; q <= finish; q++)
	{
		sprintf(proPicFilename, "C:\\developer\\4.3\\Projector%d.jpg", q);
		vector<Point2f> tempPointSet2d_projector;

		ifile_projecotrPicure.open(proPicFilename);
		
		Mat proPicGray;

		Mat proPic = imread(proPicFilename);
		
		//flip(proPic, proPic, 1);//***
		//proPic = ~proPic;
		//resize(camPic, camPic, Size(camPic.size().width / 2, camPic.size().height / 2));

		/*Mat proPicUndis = proPic.clone();
		undistort(proPic, proPicUndis, cameraMatrix, distCoeffs);
		proPic = proPicUndis;*/


		std::vector<cv::Point2f> proCorners;
		CvSize proPatternSize;
		proPatternSize.height = 6;
		proPatternSize.width = 9;

		//cvtColor(camPic, camPic, CV_BGR2GRAY);
		bool proRet = findChessboardCorners(proPic, proPatternSize, proCorners);
		if (!proRet)
		{
			return -1;
		}
		else {

			cvtColor(proPic, proPicGray, CV_RGB2GRAY);
			find4QuadCornerSubpix(proPicGray, proCorners, Size(11, 11));
			cv::drawChessboardCorners(proPic, proPatternSize, proCorners, proRet);
			/*imshow("projector", proPic);
			moveWindow("projector", 0, 0);
			waitKey();*/
			image_points_seq_projector.push_back(proCorners);
			vector<Point3f>worldPTS;
			backProject(rvecsMat[q-1], tvecsMat[q-1], cameraMatrix, proCorners, worldPTS);
			//reverse(worldPTS.begin(),worldPTS.end());//***
			object_points_projector.push_back(worldPTS);
		}
	}
	//
	
	Size image_size = Size(1600, 1200);
	Size board_size = Size(6, 4);

	Mat cameraMatrix_projector = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 摄像机内参数矩阵 */
	Mat distCoeffs_camera = Mat(1, 5, CV_32FC1, Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
	Mat distCoeffs_projector = Mat(1, 5, CV_32FC1, Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */

	vector<Mat> proRvecsMat;
	vector<Mat> proTvecsMat;
	
	double ProjRms = calibrateCamera(object_points_projector, image_points_seq_DMD, image_size, cameraMatrix_projector, distCoeffs_projector, proRvecsMat, proTvecsMat, 0);

	cout << "projector rms = " << ProjRms << endl;
	cout << "projector matrix = " << cameraMatrix_projector << endl;
	cout << "projector distCoeffs = " << distCoeffs_projector << endl;

	//cout << object_points_projector[0] << endl;
	while (getchar())
	{

	}
}