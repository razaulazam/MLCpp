#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <map>
#include "opencv2/ml.hpp"
#include <time.h>
#include <time.h>
# define PI 3.14159265358979323846
# define e 2.71828

using namespace cv;
using namespace std;

#ifndef DENSITYTREE_H
#define DENSITYTREE_H

class DensityTree{
public:
    DensityTree();
    DensityTree(unsigned int D, unsigned int R, Mat X);
    Mat Training(Mat X, Mat &leftnode, Mat &rightnode);
    void train();
    Mat densityXY();
private:
    int ray = 0;
    int check = 1;
    Mat array[20];
    unsigned int D;
    unsigned int n_thresholds;
    Mat X;
};

double informationGain(Mat &X, Mat &nodeL, Mat &nodeR);




#endif /* DENSITYTREE_H */



