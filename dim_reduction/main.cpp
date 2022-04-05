#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <map>

using namespace cv;
using namespace std;

Mat reducePCA(Mat &dataMatrix, unsigned int dim)
{
    int num_rows = dataMatrix.rows;
    int num_columns = dataMatrix.cols;
    Mat support = Mat(num_rows,1,CV_64F,Scalar(1.0));
    Mat means = Mat(1,num_columns,CV_64F);
    Mat covariance_matrix = Mat(num_columns,num_columns,CV_64F,Scalar(0.0));
    Mat eigen_values = Mat(1,num_columns,CV_64F);
    Mat eigen_vectors = Mat(num_columns,num_columns,CV_64F);
    Mat intermediate = Mat(num_rows,1,CV_64F);
    Mat norms_val = Mat(1,num_columns,CV_64F);
    for (int j = 0; j < num_columns; j++){
        for (int i = 0; i < num_rows; i++){
            intermediate.at<double>(i) = dataMatrix.at<double>(i,j); //computing the L2 Norm
        }
        norms_val.at<double>(j) = norm(intermediate,NORM_L2);
    }
    for (int j = 0; j < num_columns; j++){
        for (int i = 0; i < num_rows; i++){
            dataMatrix.at<double>(i,j) = dataMatrix.at<double>(i,j)/norms_val.at<double>(j);//Normalization
        }
    }
    reduce(dataMatrix, means, 0, CV_REDUCE_AVG);
    dataMatrix = dataMatrix - support*means; 
    
    Mat dataMatrix_transposed = dataMatrix.t();
    covariance_matrix = dataMatrix_transposed*dataMatrix; //covariance matrix
    covariance_matrix = covariance_matrix/(num_rows);
    eigen(covariance_matrix, eigen_values, eigen_vectors); //eigen values and vectors
    int eigen_columns = eigen_vectors.cols;
    Mat required = Mat(dim,eigen_columns,CV_64F);
    for (int i = 0; i < dim; i++){
        for (int j = 0; j < eigen_columns; j++){
            required.at<double>(i,j) = eigen_vectors.at<double>(i,j); 
        }
    }
    Mat required_transpose = required.t();
    dataMatrix = dataMatrix*required_transpose; // 2 dimensional data
    return dataMatrix;
}

Mat reduceIsomap(Mat &dataMatrix, unsigned int dim)
{
    int num_rows = dataMatrix.rows;
    int num_columns = dataMatrix.cols;
    int key = 0;
    double not_neighbours = 10000.0;
    double val = 0.0;
    Mat distances = Mat(num_rows,num_rows,CV_64F,Scalar(0.0000));
    Mat short_distances = Mat(num_rows,num_rows,CV_64F,Scalar(0.0000));
    Mat intermediate = Mat(1,num_columns,CV_64F,Scalar(0.0000));
    Mat norms = Mat(num_rows,num_rows,CV_64F,Scalar(0.0000));
    Mat intermediate_extra = Mat(1,num_columns,CV_64F,Scalar(0.0000));
    Mat identity_mat = Mat(num_rows,num_rows,CV_64F,Scalar(0.0));
    Mat ones = Mat(num_rows,num_rows,CV_64F,Scalar(1.0));
    Mat centering = Mat(num_rows,num_rows,CV_64F,Scalar(0.0));
    Mat tau = Mat(num_rows,num_rows,CV_64F,Scalar(0.0));
    Mat eigen_values = Mat(1,num_rows,CV_64F,Scalar(0.0000));
    Mat eigen_vectors = Mat(num_rows,num_rows,CV_64F,Scalar(0.0000));
    Mat required = Mat(dim,num_rows,CV_64F,Scalar(0.0000));
    
    for (int i = 0; i < num_rows; i++)
    { 
        dataMatrix.row(i).copyTo(intermediate);
        for (int j = 0; j < num_rows; j++){
            dataMatrix.row(j).copyTo(intermediate_extra);
            norms.at<double>(j,i) = norm(intermediate,intermediate_extra,NORM_L2);
        }
    }
    distances = norms.t();
    
    map<double,int>distanceMap;  
    for (int i = 0; i < num_rows; i++)
    { 
        for (int j = 0; j < num_rows; j++){
            val = distances.at<double>(i,j);
            distanceMap.insert(pair<double,int>(val,key));
            distances.at<double>(i,j) = not_neighbours;
            key++;
        }
        
        //Sorting process
        map<double,int>::iterator it = distanceMap.begin();
        for (int k = 0; k < 10; k++){ 
            distances.at<double>(i,it->second) = it->first;
            it++;
        }
        distanceMap.clear();
        key = 0;
    }
    distances.copyTo(short_distances);
    for (int k = 0; k < num_rows; k++)
    { 
        for (int i = 0; i < num_rows; i++)
        {
            for (int j = 0; j < num_rows; j++){
                if (short_distances.at<double>(i,k) + short_distances.at<double>(k,j) < short_distances.at<double>(i,j)){
                    short_distances.at<double>(i,j) = short_distances.at<double>(i,k) + short_distances.at<double>(k,j);
                }
            }
        }
    }
    
    for (int i = 0; i < num_rows; i++){
        identity_mat.at<double>(i,i) = 1.0;
    }
    ones = ones/num_rows;  
    centering = identity_mat - ones;
    for (int i = 0; i < num_rows; i++){ 
        for (int j = 0; j < num_rows; j++){
            short_distances.at<double>(i,j) = short_distances.at<double>(i,j)*short_distances.at<double>(i,j);
        }
    }
    tau = centering*short_distances*centering;
    tau = -tau/2.0;
    eigen(tau, eigen_values, eigen_vectors);
    for (int i = 0; i < dim; i++){
        for (int j = 0; j < num_rows; j++){
            required.at<double>(i,j) = eigen_vectors.at<double>(i,j);
        }
    }
    required = required.t();
    return required;
}






