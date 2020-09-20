#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <map>

using namespace cv;
using namespace std;

void generateRandomObservations(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount);
double observationProbabilityForward(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount);
double bestStateSequence(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount, unsigned int* bestStates);
double rd() { return (double)rand() / (double)RAND_MAX; } // we suggest to use this to create uniform random values between 0 and 1

int main(int argc, char** argv){
    // keep this to produce our results
    srand(42);
    // Four states, 3 symbols
    Mat A =  Mat(4, 4, CV_64F);
    Mat B =  Mat(4, 3, CV_64F);
    Mat P = Mat(1, 4, CV_64F);
    A.at<double>(0,0) = 0.5;
    A.at<double>(0,1) = 0.2;
    A.at<double>(0,2) = 0.3;
    A.at<double>(0,3) = 0.0;
    
    A.at<double>(1,0) = 0.2;
    A.at<double>(1,1) = 0.4;
    A.at<double>(1,2) = 0.1;
    A.at<double>(1,3) = 0.3;
    
    A.at<double>(2,0) = 0.7;
    A.at<double>(2,1) = 0.1;
    A.at<double>(2,2) = 0.1;
    A.at<double>(2,3) = 0.1;
    
    A.at<double>(3,0) = 0.0;
    A.at<double>(3,1) = 0.1;
    A.at<double>(3,2) = 0.8;
    A.at<double>(3,3) = 0.1;
    
    P.at<double>(0,0) = 0.7;
    P.at<double>(0,1) = 0.2;
    P.at<double>(0,2) = 0.1;
    P.at<double>(0,3) = 0.0;
    
    B.at<double>(0,0) = 0.6;
    B.at<double>(0,1) = 0.2;
    B.at<double>(0,2) = 0.2;
    
    B.at<double>(1,0) = 0.4;
    B.at<double>(1,1) = 0.4;
    B.at<double>(1,2) = 0.2;
    
    B.at<double>(2,0) = 0.3;
    B.at<double>(2,1) = 0.3;
    B.at<double>(2,2) = 0.4;
    
    B.at<double>(3,0) = 0.1;
    B.at<double>(3,1) = 0.2;
    B.at<double>(3,2) = 0.7;
    
    // Length = 2;
    
    unsigned int cnt = 2;
    unsigned int obs1[cnt];
    unsigned int bestStates1[cnt];
    generateRandomObservations(A,B,P,obs1,cnt);
    obs1[0] = 0;
    obs1[1] = 2;
    cout << "Observation Sequence: ";
    for( int i = 0; i < cnt; i++){
        cout << obs1[i] << " ";
    }
    cout << endl;
    double prob_all = observationProbabilityForward(A,B,P,obs1,cnt);
    double prob_best = bestStateSequence(A, B, P, obs1, cnt, bestStates1);
    cout << "Probability: " << prob_all << endl;
    cout << "Best Sequence: ";
    for( int i = 0; i < cnt; i++){
        cout << bestStates1[i] << " ";
    }
    cout << "Probability: " << prob_best << endl;
    cout << "Best Prob. / Total Prob: " << prob_best / prob_all << endl << endl;
   // Length = 10
    
    cnt = 10;
    unsigned int obs2[cnt];
    unsigned int bestStates2[cnt];
    generateRandomObservations(A,B,P,obs2,cnt);
    obs2[0] = 0;
    obs2[1] = 0;
    obs2[2] = 0;
    obs2[3] = 2;
    obs2[4] = 0;
    obs2[5] = 1;
    obs2[6] = 1;
    obs2[7] = 1;
    obs2[8] = 2;
    obs2[9] = 2;
    
    cout << "Observation Sequence: ";
    for( int i = 0; i < cnt; i++){
        cout << obs2[i] << " ";
    }
    cout << endl;
    prob_all = observationProbabilityForward(A,B,P,obs2,cnt);
    prob_best = bestStateSequence(A, B, P, obs2, cnt, bestStates2);
    cout << "Probability: " << prob_all << endl;
    cout << "Best Sequence: ";
    for( int i = 0; i < cnt; i++){
        cout << bestStates2[i] << " ";
    }
    cout << "Probability: " << prob_best << endl;
    cout << "Best Prob. / Total Prob: " << prob_best / prob_all << endl << endl;
    
     return 0;
}

//Generating random observations by random sampling as learned in the lecture
void generateRandomObservations(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount){
    int numStates = A.rows;
    int numSymbols = B.cols;
    int state;
    
    double r = rd(); //random number
    double s = 0.0; // probability counter
    // Starting state
    // iterate over all states and set the starting state
    for (int i = 0; i < numStates; i++){
        if(r >= s && r < s + P.at<double>(0,i)){
            state = i;
            break;
        }
        s = s + P.at<double>(0,i);
    }
    
    for(int t = 0; t < observationCount; t++){
        // emit symbol
        r = rd();
        s = 0.0;
        for(int i = 0; i < numSymbols; i++)
        {
            if(r >= s && r < s + B.at<double>(state,i))
            {
                observations[t] = i;
                break;
            }
            s = s + B.at<double>(state,i);
        }
        // switch to next state
        if(t < observationCount-1)
        {
            r = rd();
            s = 0.0;
            for (int i = 0; i < numStates; i++)
            {
                if(r >= s && r < s + A.at<double>(state,i))
                {
                    state = i;
                    break;
                }
                s = s + A.at<double>(state,i);
            }
        }
    }
}

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




