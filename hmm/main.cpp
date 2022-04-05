#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <map>

using namespace cv;
using namespace std;

double observationProbabilityForward(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount);
double bestStateSequence(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount, unsigned int* bestStates);
double rd() { return (double)rand() / (double)RAND_MAX; } // we suggest to use this to create uniform random values between 0 and 1

// Observation probability -> Forward algorithm
double observationProbabilityForward(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount){
    int numStates = A.rows;
    Mat forwardMtx = Mat(numStates,observationCount,CV_64F,Scalar(0.));
    double prob = 0.0;
    
    for(int i = 0; i < observationCount; i++){
        for(int j = 0; j < numStates; j++){
            if(i == 0){
                forwardMtx.at<double>(j,i) = P.at<double>(0,j)*B.at<double>(j,observations[i]);
            }
            else{
                double sum = 0.0;
                for(int k = 0; k < numStates; k++){
                    sum = sum + (forwardMtx.at<double>(k,i-1)*A.at<double>(k,j));
                }
                forwardMtx.at<double>(j,i) = sum*B.at<double>(j,observations[i]);
            }
        }
    }
    
    for(int i = 0; i < numStates; i++){
        prob = prob + forwardMtx.at<double>(i,(observationCount-1));
    }
    
    return prob;
}

// best state sequence and observation probability using this state sequence -> Viterbi algorithm
double bestStateSequence(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount, unsigned int* bestStates){
    
    int numStates = A.rows;
    Mat forwardMtx = Mat(numStates,observationCount,CV_64F,Scalar(0.));
    Mat backwardMtx = Mat::zeros(numStates,observationCount,CV_32S);
    int key = 0;
    int index = 0;
    double prob = 0.0;
    multimap<double,int> valMap;
    multimap<double,int>::iterator it;
    
    for(int i = 0; i < observationCount; i++){
        for(int j = 0; j < numStates; j++){
            if(i == 0){
                forwardMtx.at<double>(j,i) = P.at<double>(0,j)*B.at<double>(j,observations[i]);
                backwardMtx.at<int>(j,i) = 0;
                continue;
            }
            else{
                for(int k = 0; k < numStates; k++){
                    auto val = forwardMtx.at<double>(k,i-1)*A.at<double>(k,j);
                    valMap.insert(pair<double,int>(val,key));
                    key++;
                }
                it = valMap.begin();
                for(int k = 0; k < (key-1); k++){
                    it++;
                }
                auto value = it->first;
                index = it->second;
                forwardMtx.at<double>(j,i) = value*B.at<double>(j,observations[i]);
                for(int k = 0; k < i; k++){
                    backwardMtx.at<int>(j,k) = backwardMtx.at<int>(index,k);
                }
                backwardMtx.at<int>(j,i) = j;
            }
            valMap.clear();
            key = 0;
        }
    }
    
    index = 0;
    for(int i = 0; i < numStates; i++){
        auto val = forwardMtx.at<double>(i,(observationCount-1));
        valMap.insert(pair<double,int>(val,key));
        key++;
    }
    it = valMap.begin();
    for(int k = 0; k < (key-1); k++){
        it++;
    }
    index = it->second;
    
    for(int i = 0; i < observationCount; i++){
        bestStates[i] = backwardMtx.at<int>(index,i);
    }
    
    for (int i = 0; i < observationCount; ++i){
        if(i == 0){
            prob = forwardMtx.at<double>(bestStates[i],observations[i]);
        }
        else{
            prob = prob*A.at<double>(bestStates[i-1],bestStates[i])*B.at<double>(bestStates[i], observations[i]);
        }
    }
    
    valMap.clear();
    return prob;
}




