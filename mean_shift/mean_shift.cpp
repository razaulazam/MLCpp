#include <vector>
#include <iostream>
#include <opencv2>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;

void meanShift(Mat matrix, Mat current) {
    double lambda = 0.3;
    int maxIter = 1;
    double tolerance = 0.001;
    int rows_mat = matrix.rows;
    int columns_mat = matrix.cols;
    Mat rowMat = Mat(1,columns_mat,CV_64F);
    Mat normMat = Mat(rows_mat,1,CV_64F);
    Mat var1 = Mat(1, columns_mat, CV_64F);
    Mat var = Mat(64, 3, CV_64F, Scalar(0.0));
    int inliers = 0;
    int i = 0;
    while(i < maxIter){
        for (int j = 0; j < rows_mat; j++){
            for (int k = 0; k < columns_mat; k++) {
                rowMat.at<double>(k) = matrix.at<double>(j, k);
            }
            normMat.at<double>(j) = norm(rowMat, current, NORM_L2);
        }
        Mat meanshift = Mat(1, columns_mat, CV_64F,Scalar(0.0));
        for (int p = 0; p < rows_mat; p++) {
            if (normMat.at<double>(p) < lambda) {
                matrix.row(p).copyTo(var1);
                meanshift = meanshift + var1;
                inliers++;
            }
        }
        if (inliers == 0)
            break;

        else {
            meanshift = meanshift / inliers;
        }

        meanshift = current;
        current = current + meanshift;
        if (norm(meanshift) < tolerance)
            break;
        i++;
    }
}
