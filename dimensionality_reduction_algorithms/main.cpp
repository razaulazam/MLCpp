#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <map>

using namespace cv;
using namespace std;

// functions for drawing
void Draw3DManifold(Mat &dataMatrix, char const * name, int nSamplesI, int nSamplesJ);
void Draw2DManifold(Mat &dataMatrix, char const * name, int nSamplesI, int nSamplesJ);

// functions for dimensionality reduction
// first parameter: features saved in rows
// second parameter: desired # of dimensions ( here: 2)
Mat reducePCA(Mat &dataMatrix, unsigned int dim);
Mat reduceIsomap(Mat &dataMatrix, unsigned int dim);
Mat reduceLLE(Mat &dataMatrix, unsigned int dim);

int main(int argc, char** argv){
    // generate Data Matrix
    unsigned int nSamplesI = 10;
    unsigned int nSamplesJ = 10;
    Mat dataMatrix =  Mat(nSamplesI*nSamplesJ, 3, CV_64F);
    // noise in the data
    double noiseScaling = 10.0;
    
    for (int i = 0; i < nSamplesI; i++)
    {
        for (int j = 0; j < nSamplesJ; j++)
        {
            dataMatrix.at<double>(i*nSamplesJ+j,0) = (i/(double)nSamplesI * 2.0 * 3.14 + 3.14) * cos(i/(double)nSamplesI * 2.0 * 3.14) + (rand() % 100)/noiseScaling;
            dataMatrix.at<double>(i*nSamplesJ+j,1) = (i/(double)nSamplesI * 2.0 * 3.14 + 3.14) * sin(i/(double)nSamplesI * 2.0 * 3.14) + (rand() % 100)/noiseScaling;
            dataMatrix.at<double>(i*nSamplesJ+j,2) = 10.0*j/(double)nSamplesJ + (rand() % 100)/noiseScaling;
        }
    }
    
    // Draw 3D Manifold
    Draw3DManifold(dataMatrix, "3D Points",nSamplesI,nSamplesJ);
    
    // PCA
    //Mat dataPCA = reducePCA(dataMatrix,2);
    //Draw2DManifold(dataPCA,"PCA",nSamplesI,nSamplesJ);
    
    // Isomap
    Mat dataIsomap = reduceIsomap(dataMatrix,2);
    Draw2DManifold(dataIsomap,"ISOMAP",nSamplesI,nSamplesJ);
    waitKey(0); //waits until keypress
    return 0;
}

Mat reducePCA(Mat &dataMatrix, unsigned int dim){
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
    dataMatrix = dataMatrix - support*means; //centered data at origin (matrix)
    
    Mat dataMatrix_transposed = dataMatrix.t();
    covariance_matrix = dataMatrix_transposed*dataMatrix; //covariance matrix
    covariance_matrix = covariance_matrix/(num_rows);
    eigen(covariance_matrix, eigen_values, eigen_vectors); //eigen values and vectors
    int eigen_columns = eigen_vectors.cols;
    Mat required = Mat(dim,eigen_columns,CV_64F);
    for (int i = 0; i < dim; i++){
        for (int j = 0; j < eigen_columns; j++){
            required.at<double>(i,j) = eigen_vectors.at<double>(i,j); //First two eigen vectors
        }
    }
    Mat required_transpose = required.t();
    dataMatrix = dataMatrix*required_transpose; //2 dimensional data
    return dataMatrix;
}

Mat reduceIsomap(Mat &dataMatrix, unsigned int dim){
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
    
    for (int i = 0; i < num_rows; i++){ //For making the distance matrix (distance of every point to all other points)
        dataMatrix.row(i).copyTo(intermediate);
        for (int j = 0; j < num_rows; j++){
            dataMatrix.row(j).copyTo(intermediate_extra);
            norms.at<double>(j,i) = norm(intermediate,intermediate_extra,NORM_L2);
        }
    }
    distances = norms.t();
    
    map<double,int>distanceMap;  //For sorting the distance matrix and picking up 10 nearest neighbours
    for (int i = 0; i < num_rows; i++){ //Distance map -> distance values
        for (int j = 0; j < num_rows; j++){
            val = distances.at<double>(i,j);
            distanceMap.insert(pair<double,int>(val,key));
            distances.at<double>(i,j) = not_neighbours;
            key++;
        }
        map<double,int>::iterator it = distanceMap.begin(); //Sorting process
        for (int k = 0; k < 10; k++){ //Picking up nearest neighbours
            distances.at<double>(i,it->second) = it->first;
            it++;
        }
        distanceMap.clear();
        key = 0;
    }
    distances.copyTo(short_distances);
    for (int k = 0; k < num_rows; k++){ //Floyd Warshall Algorithm
        for (int i = 0; i < num_rows; i++){
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
    ones = ones/num_rows;  //D = (centeringmatrix*adjacency^2*centeringmatrix)/2; centeringmatrix = Identitymatrix - (ones/n)
    centering = identity_mat - ones;
    for (int i = 0; i < num_rows; i++){ //Taking the square of the adjacency matrix (D)
        for (int j = 0; j < num_rows; j++){
            short_distances.at<double>(i,j) = short_distances.at<double>(i,j)*short_distances.at<double>(i,j);
        }
    }
    tau = centering*short_distances*centering;
    tau = -tau/2.0;
    eigen(tau, eigen_values, eigen_vectors);
    for (int i = 0; i < dim; i++){
        for (int j = 0; j < num_rows; j++){
            required.at<double>(i,j) = eigen_vectors.at<double>(i,j); //First two eigen vectors
        }
    }
    required = required.t();
    return required;
}

void Draw3DManifold(Mat &dataMatrix, char const * name, int nSamplesI, int nSamplesJ)
{
    Mat origImage = Mat(1000,1000,CV_8UC3);
    origImage.setTo(0.0);
    for (int i = 0; i < nSamplesI; i++)
    {
        for (int j = 0; j < nSamplesJ; j++)
        {
            Point p1;
            p1.x = dataMatrix.at<double>(i*nSamplesJ+j,0)*50.0 +500.0 - dataMatrix.at<double>(i*nSamplesJ+j,2)*10;
            p1.y = dataMatrix.at<double>(i*nSamplesJ+j,1) *50.0 + 500.0 - dataMatrix.at<double>(i*nSamplesJ+j,2)*10;
           // circle(origImage,p1,3,Scalar( 255, 255, 255 ));
            
            Point p2;
            if(i < nSamplesI-1)
            {
                p2.x = dataMatrix.at<double>((i+1)*nSamplesJ+j,0)*50.0 +500.0 - dataMatrix.at<double>((i+1)*nSamplesJ+(j),2)*10;
                p2.y = dataMatrix.at<double>((i+1)*nSamplesJ+j,1) *50.0 + 500.0 - dataMatrix.at<double>((i+1)*nSamplesJ+(j),2)*10;
                
                line( origImage, p1, p2, Scalar( 255, 255, 255 ), 1, 8 );
            }
            if(j < nSamplesJ-1)
            {
                p2.x = dataMatrix.at<double>((i)*nSamplesJ+j+1,0)*50.0 +500.0 - dataMatrix.at<double>((i)*nSamplesJ+(j+1),2)*10;
                p2.y = dataMatrix.at<double>((i)*nSamplesJ+j+1,1) *50.0 + 500.0 - dataMatrix.at<double>((i)*nSamplesJ+(j+1),2)*10;
                
                line( origImage, p1, p2, Scalar( 255, 255, 255 ), 1, 8 );
            }
        }
    }
    
    
    namedWindow( name, WINDOW_AUTOSIZE );
    imshow( name, origImage );
}

void Draw2DManifold(Mat &dataMatrix, char const * name, int nSamplesI, int nSamplesJ)
{
    Mat origImage = Mat(1000,1000,CV_8UC3);
    origImage.setTo(0.0);
    for (int i = 0; i < nSamplesI; i++)
    {
        for (int j = 0; j < nSamplesJ; j++)
        {
            Point p1;
            p1.x = dataMatrix.at<double>(i*nSamplesJ+j,0)*1000.0 +500.0;
            p1.y = dataMatrix.at<double>(i*nSamplesJ+j,1) *1000.0 + 500.0;
            circle(origImage,p1,3,Scalar( 255, 255, 255 ));
            
            Point p2;
            if(i < nSamplesI-1)
            {
                p2.x = dataMatrix.at<double>((i+1)*nSamplesJ+j,0)*1000.0 +500.0;
                p2.y = dataMatrix.at<double>((i+1)*nSamplesJ+j,1) *1000.0 + 500.0;
                line( origImage, p1, p2, Scalar( 255, 255, 255 ), 1, 8 );
            }
            if(j < nSamplesJ-1)
            {
                p2.x = dataMatrix.at<double>((i)*nSamplesJ+j+1,0)*1000.0 +500.0;
                p2.y = dataMatrix.at<double>((i)*nSamplesJ+j+1,1) *1000.0 + 500.0;
                line( origImage, p1, p2, Scalar( 255, 255, 255 ), 1, 8 );
            }
            
        }
    }
    
    
    namedWindow( name, WINDOW_AUTOSIZE );
    imshow( name, origImage );
    imwrite( (String(name) + ".png").c_str(),origImage);
}


