#include <time.h>

#include "DensityTree.hpp"

DensityTree::DensityTree(unsigned int D, unsigned int n_thresholds, Mat X){
    this->D = D;
    this->X = X;
    this->n_thresholds = n_thresholds;
}
void DensityTree::train()
{
    Mat nodeL, nodeR;
    int levels = D - 1, nodesOnLevel, counter = 0;
    array[0] = X;
    for (int levelCount = 0; levelCount < levels; levelCount++){
        nodesOnLevel = pow(2, levelCount);
        for (int count = 0; count < nodesOnLevel; count++){
            Mat res = Training(array[ray],nodeL,nodeR);
            ray++;
            counter++;
            array[counter] = nodeL;
            counter++;
            array[counter] = nodeR;
        }
    }
    ray = counter + 1;
    int leaves = pow(2, (D - 1));
    ray = ray - leaves;
}


Mat DensityTree::Training(Mat newX, Mat &nodeL,Mat &nodeR){
    double minVal1, maxVal1, minVal2, maxVal2;
    minMaxLoc(newX.col(0), &minVal1, &maxVal1, 0, 0);
    minMaxLoc(newX.col(1), &minVal2, &maxVal2, 0, 0);
    Mat randomX = Mat(50, 1, CV_64F);
    Mat randomY = Mat(50, 1, CV_64F);
    Mat nodes;
    
    RNG rng(getTickCount());
    for (int i = 0; i < 50; i++)
        randomX.at<double>(i, 0) = rng.uniform((maxVal1 ) , (minVal1 ) );
    for (int i = 0; i < 50; i++)
        randomY.at<double>(i, 0) = rng.uniform((maxVal2 ) , (minVal2 ) );
    int rowsX = newX.rows;
    Mat copyX = newX;
    Mat leftNode = Mat(rowsX, 3, CV_64F);
    Mat rightNode = Mat(rowsX, 3, CV_64F);
    Mat pLeftNode = Mat(rowsX, 3, CV_64F);
    Mat pRightNode = Mat(rowsX, 3, CV_64F);
    
    Mat part;
    double infoGain, pInfoGain = 0, currInfoGain;
    int countLnodes, countRnodes;
    for (int i = 0; i < 2; i++)
    {
        if (i == 0)
            part = randomX;
        else
            part = randomY;
        
        for (int iter = 0; iter < part.rows; iter++)
        {
            countLnodes = 0;
            countRnodes = 0;
            Mat currLeftNode = Mat(rowsX, 3, CV_64F);
            Mat currRightNode = Mat(rowsX, 3, CV_64F);
            
            for (int count = 0; count < rowsX; count++)
            {
                if (copyX.at<double>(count, i) > part.at<double>(iter, 0))
                {
                    currRightNode.at<double>(countRnodes, 0) = copyX.at<double>(count, 0);
                    currRightNode.at<double>(countRnodes, 1) = copyX.at<double>(count, 1);
                    if (check == 1)
                        currRightNode.at<double>(countRnodes, 2) = (double)count;
                    else
                        currRightNode.at<double>(countRnodes, 2) = copyX.at<double>(count, 2);
                    countRnodes++;
                }
                
                else
                {
                    currLeftNode.at<double>(countLnodes, 0) = copyX.at<double>(count, 0);
                    currLeftNode.at<double>(countLnodes, 1) = copyX.at<double>(count, 1);
                    if (check == 1)
                        currLeftNode.at<double>(countLnodes, 2) = (double)count;
                    else
                        currLeftNode.at<double>(countLnodes, 2) = copyX.at<double>(count, 2);
                    countLnodes++;
                }
            }
            Mat cLeftNode(currLeftNode, Rect(0, 0, currLeftNode.cols, countLnodes));
            Mat cRightNode(currRightNode, Rect(0, 0, currRightNode.cols, countRnodes));
            
            cRightNode.col(2) = (currRightNode.col(2) + 0);
            cLeftNode.col(2) = (currLeftNode.col(2) + 0);
            
            Mat t1 = Mat(cRightNode.rows, 2, CV_64F);
            Mat t2 = Mat(cLeftNode.rows, 2, CV_64F);
            t1.col(0) = (cRightNode.col(0) + 0);
            t1.col(1) = (cRightNode.col(1) + 0);
            t2.col(0) = (cLeftNode.col(0) + 0);
            t2.col(1) = (cLeftNode.col(1) + 0);
            currInfoGain = informationGain(copyX, t1, t2);
            
            if (currInfoGain > pInfoGain){
                infoGain = currInfoGain;
                pInfoGain = currInfoGain;
                
                leftNode = cLeftNode;
                rightNode = cRightNode;
            }
            else{
                infoGain = pInfoGain;
            }
        }
    }
    nodeL = leftNode;
    nodeR = rightNode;
    check++;
    
    nodes.push_back(nodeL);
    nodes.push_back(nodeR);
    return nodes;
}

Mat DensityTree::densityXY(){
    static int y = 1;
    y++;
    int limit = pow(2,D) - 1;
    int leaves = pow(2,(D - 1));
    ray = limit  - leaves;
    Mat tempData, temp, emp;
    Mat res = Mat::zeros(X.rows, 2, CV_64F);
    for (; ray < limit; ray++){
        Ptr<cv::ml::EM> emModel = cv::ml::EM::create();
        emModel->setClustersNumber(1);
        Mat resTemp = Mat::zeros(array[ray].rows, 2, CV_64F);
        resTemp.col(0) = (array[ray].col(0) + 0);
        resTemp.col(1) = (array[ray].col(1) + 0);
        
        resTemp.convertTo(tempData, CV_32F);
        emModel->train(tempData, 0, emp);
        Mat data = Mat(1, 2, CV_64F);
        Mat data2 = Mat(1, 2, CV_64F);
        for (int i = 0; i < array[ray].rows; i++){
            data.at<double>(0, 0) = array[ray].at<double>(i, 0);
            data2.at<double>(0, 0) = array[ray].at<double>(i, 1);
            double pos = array[ray].at<double>(i, 2);
            
            for (int j = 0; j < array[ray].rows; j++){
                data.at<double>(0, 1) = array[ray].at<double>(j, 1);
                Vec2d probability = emModel->predict2(data, temp);
                float a = probability.val[0];
                float val = (float)exp(a);
                res.at<double>(pos, 0) = res.at<double>(pos, 0) + val;
                data2.at<double>(0, 1) = array[ray].at<double>(j, 0);
                Vec2d probability2 = emModel->predict2(data2, temp);
                float a2 = probability2.val[0];
                float val2 = (float)exp(a2);
                res.at<double>(pos, 1) = res.at<double>(pos, 1) + val2;
            }
        }
    }
    
    return res;
}



double informationGain(Mat &X, Mat &nodeL, Mat &nodeR){
    Mat cov, mu;
    cv::calcCovarMatrix(X, cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS);
    double detX = determinant(cov);
    double entropyX = 0.5* log(pow(2 * PI* 2.71828, 2) *detX);
    
    
    Mat covL, muL;
    cv::calcCovarMatrix(nodeL, covL, muL, CV_COVAR_NORMAL | CV_COVAR_ROWS);
    double detL = determinant(cov);
    double entropyL = 0.5* log(pow(2 * PI* 2.71828, 2) *detL);
    
    Mat covR, muR;
    cv::calcCovarMatrix(nodeR, covR, muR, CV_COVAR_NORMAL | CV_COVAR_ROWS);
    double detR = determinant(cov);
    double entropyR = 0.5* log(pow(2 * PI* 2.71828, 2) *detR);
    
    int rowsX = X.rows;
    int rowsL = nodeL.rows;
    int rowsR = nodeR.rows;
    
    double scaledL = (rowsL / rowsX) * entropyL;
    double scaledR = (rowsR / rowsX) * entropyR;
    double res = entropyX - (scaledL + scaledR);
    return res;
    
}

