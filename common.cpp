#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <ceres/ceres.h>
//#include <Eigen/Core>
//#include <Eigen/Dense>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <string>
#include <fstream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/QR>
#include <cmath>


#include "common.h"
#include "rotation.h"
#include "random.h"
#include "SnavelyReprojectionError.h"
#include "glog/logging.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

#define image_count 19
#define w_sq 1.5               //mm
#define point_num 81         //12*9=108   9*9=81
using namespace Eigen;
using namespace std;
using namespace cv;


typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;

template<typename T>
void FscanfOrDie(FILE *fptr, const char *format, T *value) {
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1)
        std::cerr << "Invalid UW data file. ";
}

void PerturbPoint3(const double sigma, double *point) {
    for (int i = 0; i < 3; ++i)
        point[i] += RandNormal() * sigma;
}

double Median(std::vector<double> *data) {
    int n = data->size();
    std::vector<double>::iterator mid_point = data->begin() + n / 2;
    std::nth_element(data->begin(), mid_point, data->end());
    return *mid_point;
}


 bool CamProjectionWithDistortion(const double* camera, const double* point, const double* inparameter, double* predictions) {
    // Rodrigues' formula
    double p[3];
    AngleAxisRotatePoint(camera, point, p);//旋转向量--->旋转矩阵
    // camera[3,4,5] are the translation
    p[0] += camera[3];    //Xc
    p[1] += camera[4];//P = R*X+t     R是旋转矩阵   Yc
    p[2] += camera[5];    //Zc        
//         std::cout<<"图像坐标系下的Z轴坐标："从<<p[0]<<"\n";
        // Compute the center fo distortion

//         T xp = p[0]/p[2];//p = P/P.z   归一化坐标 
//         T yp = p[1]/p[2];                                           // 注释是远心相机的配置

        // Apply second and fourth order radial distortion
    const double& l1 = inparameter[4];  //jing xiang jibian
    const double& l2 = inparameter[5];  //bolengjing jibiannnnnnnnnnnnnnnn 
    const double& l3 = inparameter[8];  //bolengjing jibian 
    const double& p1 = inparameter[6];//qie xiang jibian 
    const double& p2 = inparameter[7];//qie xiang jibian 
    const double& focalx = inparameter[0];
    const double& focaly = inparameter[1];
    const double& cx = inparameter[2];
    const double& cy = inparameter[3];
    //         T du = p[0] - cx;
    //         T dv = p[1] - cy;

    double r2 = p[0] * p[0] + p[1] * p[1];  //  xp * xp + yp * yp;               //远心相机的配置  du==p[0]   dv == p[1]

    double distortion = 1.0 + r2 * (l1 + r2 * (l2 + l3 * r2));//add the k3
//         T distortion =  T(1.0) + r2 * ( l1 + r2 * l2);
    double xd = p[0] * distortion;//+ T(2.0) * p1 * p[0] * p[1] + p2 * (r2 +T(2.0) * p[0] * p[0]);//远心相机的配置  xp * distortion + T(2.0) * p1 * xp * yp + p2 * (r2 +T(2) * xp * xp);du
    double yd = p[1] * distortion;//+ p1 * (r2 + T(2.0) * p[1] * p[1]) + T(2.0) * p2 * p[0] *p[1];//注释是远心相机的配置yp * distortion + p1 * (r2 + T(2.0) * yp * yp) + T(2) * p2 * xp *yp;

    ////modify code according matlab code
//         T xd = l1 * p[0] * r2 + p1 * p[0] * p[0] + p2 * p[0] * p[1] + l2 * r2;
//         T yd = l1 * p[1] * r2 + p2 * p[1] * p[1] + p1 * p[0] * p[1] + l3 * r2;


    predictions[0] = focalx * xd + cx;
    predictions[1] = focaly * yd + cy;
    //std::cout << "predictions[0] = " << predictions[0] << "   predictions[1] = " << predictions[1] << std::endl;

    return true;
}


BALProblem::BALProblem(const std::string &filename, bool use_quaternions) {

    int w, h;
    w = 9;
    h = 9;
    int num_point = w * h;
    Size pattern_size = Size(w, h);
    Mat srcImage, grayImage;
    string picname, path, imname, txtcorner_u,txtcorner_v;
    double m;
    path = filename;

    //初始化需要优化所需参数//
    num_cameras_ = image_count;
    num_points_ = point_num;
    num_observations_ = num_cameras_ * num_points_;

    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];


    observations_ = new double[2 * num_observations_];

    constpoint_ = new double[3 * num_points_];//////////add the constpoint///////////

    num_parameters_ = 6 * num_cameras_ + 3 * num_points_ + 9;///////////////////////
    parameters_ = new double[num_parameters_];

    reprojectu_ = new double[num_points_];////////////////////////////////////////////////////////////////////
    reprojectv_ = new double[num_points_];
    reprojectu_ = { 0 };
    reprojectv_ = { 0 };
    //初始化需要优化所需参数//

    ///world points
    vector<Point2f> object_points;
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            Point2f realPoint;
            realPoint.x = (j - 4) * w_sq;
            realPoint.y = (i - 4) * w_sq;
            object_points.push_back(realPoint);
        }
    }

    Matrix<double, Dynamic, Dynamic> M;
    M.setZero(point_num * 2, point_num * 2);  //108*2=216                         
    for (int p = 0, k = 0; k < 2 * num_point; k += 2) {
        M(k, 0) = object_points[p].x;
        M(k, 1) = object_points[p].y;
        M(k, 2) = 1;
        M(k + 1.0, 3) = object_points[p].x;
        M(k + 1.0, 4) = object_points[p].y;
        M(k + 1.0, 5) = 1;
        p++;
    }
    float u0;
    float v0;
    vector<Matrix<double, 3, 3>> homographies(image_count, Matrix<double, 3, 3>(3, 3));
    vector<Matrix<double, 3, 3>> k_mats(image_count, Matrix<double, 3, 3>(3, 3));
    vector<Matrix<double, point_num * 2, 1>> corner(image_count, Matrix<double, point_num * 2, 1>(2 * point_num, 1));

    ///detect point each image
    for (int i = 0; i < image_count; i++) {//image_count
        picname = path + to_string(i+1) + ".bmp";
        srcImage = imread(picname);
        imname = "img" + to_string(i);
       
        if (srcImage.empty())
            cout << "there is no " << to_string(i) << "'s picture imreaded" << endl; //是否加载成功
        if (!srcImage.data)
            cout << "NO data in picture " << to_string(i) << endl;//判断是否有数据

        u0 = srcImage.cols / 2;
        v0 = srcImage.rows / 2;
        vector<Point2f> corners;
        Mat gray;
        cvtColor(srcImage, gray, COLOR_BGR2GRAY);
        int rows = srcImage.rows;
        int cols = srcImage.cols;

        //------points detector
        SimpleBlobDetector::Params params;//change the params config according the color of blob
         //params.minThreshold = 0;
         //params.maxThreshold = 150;
         //params.filterByArea = true;
         //params.minArea = 50;
         //params.maxArea = 10e6;
         //params.minDistBetweenBlobs = 10;
        params.filterByColor = true;
        params.blobColor = 255;////////仿真时修改过
        Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
        bool found = findCirclesGrid(gray, pattern_size, corners, CALIB_CB_SYMMETRIC_GRID, detector);
        //drawChessboardCorners(srcImage, pattern_size, corners, found);
        //imshow(imname,srcImage);
        //waitKey(100);
        //if(i==1) imwrite("1.jpg", srcImage);
        cout << "image_" << to_string(i) << "_:" << found << endl;



        ///////----------自适应特征点检测算法(从txt文件中读取角点)--------------
        //读取数据文件
        //txtcorner_u = path + to_string(i + 1) + "_u.txt";
        //txtcorner_v = path + to_string(i + 1) + "_v.txt";
        //ifstream in_u(txtcorner_u, ios::in);
        //ifstream in_v(txtcorner_v, ios::in);
        //if (!in_u.is_open()|| !in_v.is_open())
        //{
        //    cout << "open error!" << endl;
        //    exit(0);
        //}
        ////将数据文件数据存入数组
        //int pu = 0;
        //vector<double> v(w * h);
        //vector<double> u(w * h);
        //while (!in_u.eof() && pu < w * h)
        //{
        //    in_u >> u[pu];
        //    pu++;
        //}
        //int pv = 0;
        //while (!in_v.eof() && pv < w * h)
        //{
        //    in_v >> v[pv];
        //    pv++;
        //}

        //for (int j = 0, flag = 0; flag < point_num; flag++) {
        //    corner[i](j, 0) = u[flag];
        //    corner[i](j + 1, 0) = v[flag];
        //    j = j + 2;
        //}
        ///////----------自适应特征点检测算法(从txt文件中读取角点)--------------

        //原版
        for (int j = 0, flag = 0; flag < point_num; flag++) {
            corner[i](j, 0) = corners[flag].x;
            corner[i](j + 1, 0) = corners[flag].y;
            j = j + 2;
        }


        //初始化需要优化所需参数    开始
        for (int j = 0; j < point_num; j++) {
            camera_index_[i * point_num + j] = i;
            point_index_[i * point_num + j] = j;
        }



        for (int j = 0; j < 2 * point_num; j++) {
            observations_[i * point_num * 2 + j] = corner[i](j, 0);
        }
        //初始化需要优化所需参数    结束



        Matrix<double, 2 * point_num, 1> h = M.colPivHouseholderQr().solve(corner[i]); //以前216=108*2      2 * point_num
        homographies[i](0, 0) = h(0, 0);
        homographies[i](0, 1) = h(1, 0);
        homographies[i](0, 2) = h(2, 0);
        homographies[i](1, 0) = h(3, 0);
        homographies[i](1, 1) = h(4, 0);
        homographies[i](1, 2) = h(5, 0);
        homographies[i](2, 0) = 0;
        homographies[i](2, 1) = 0;
        homographies[i](2, 2) = 1;
        double c1 = h(0, 0) * h(0, 0) + h(1, 0) * h(1, 0) + h(3, 0) * h(3, 0) + h(4, 0) * h(4, 0);
        double c2 = (h(0, 0) * h(4, 0) - h(1, 0) * h(3, 0)) * (h(0, 0) * h(4, 0) - h(1, 0) * h(3, 0));
        Mat coef = (Mat_<double>(5, 1) << c2, 0, -c1, 0, 1);
        Mat roots;
        solvePoly(coef, roots);
        vector<double> coff = { roots.at<double>(0,0),roots.at<double>(1,0),roots.at<double>(2,0),roots.at<double>(3,0) };
        m = *max_element(coff.begin(), coff.end());
        k_mats[i] << m, 0, u0, 0, m, v0, 0, 0, 1;      //internal matrix each picture
    }

    //compute globle internal matrix
    Matrix<double, image_count, image_count> G;//image_count   image_count
    G.setOnes(image_count, image_count);
    Matrix<double, image_count, 1> W;//image_count
    W.setZero(image_count, 1);
    for (int k = 0; k < image_count; k++) {
        Matrix<double, 3, 3> H = homographies[k];
        G(k, 1) = -(H(1, 0) * H(1, 0)) - (H(1, 1) * H(1, 1));
        G(k, 2) = -(H(0, 0) * H(0, 0)) - (H(0, 1) * H(0, 1));
        G(k, 3) = 2 * (H(0, 0) * H(1, 0) + H(0, 1) * H(1, 1));
        W(k, 0) = -((H(0, 0) * H(1, 1) - H(0, 1) * H(1, 0)) * (H(0, 0) * H(1, 1) - H(0, 1) * H(1, 0)));
    }

    Matrix<double, image_count, 1> L = G.colPivHouseholderQr().solve(W);//colPivHouseholderQr()   image_count

    double alpha = sqrt((L(1, 0) * L(2, 0) - L(3, 0) * L(3, 0)) / L(2, 0));
    double beta = sqrt(L(2, 0));
    double gama = sqrt(L(1, 0) - (L(0, 0) / L(2, 0)));
    //     cout << "fx =  " << alpha << "\n" << "fy =  " << beta << endl;

    Matrix<double, 3, 3> K_lsp;
    K_lsp << alpha, 0, u0, 0, beta, v0, 0, 0, 1;
    cout << "Internal Matrix = \n" << K_lsp << endl;

    // compute external matrix
    Vector3d T;//Vector3d--eigen
    vector<double> R_vec;
    Mat R;
    double even_error_u = 0;
    double even_error_v = 0;

    vector<vector<double>> Rs;
    vector<Vector3d> Ts;

    for (int k = 0; k < image_count; k++) {
        Matrix<double, 3, 3> H = homographies[k];
        Matrix<double, 3, 3> K = k_mats[k];
        Matrix<double, 3, 3> E = K.colPivHouseholderQr().solve(H);
        double tx = (H(0, 2) - u0) / m;
        double ty = (H(1, 2) - v0) / m;
        double r13 = sqrt(1 - E(0, 0) * E(0, 0) - E(1, 0) * E(1, 0));
        double r23 = sqrt(1 - E(0, 1) * E(0, 1) - E(1, 1) * E(1, 1));
        Vector3d r1(E(0, 0), E(0, 1), r13);
        Vector3d r2(E(1, 0), E(1, 1), r23);
        Vector3d r3 = r1.cross(r2);
        T << tx, ty, 0;
        Matrix3d Rm;
        Rm << r1, r2, r3;
        //cout << "***********file__" << to_string(k + 1) << "__*************" << endl;

        R = (Mat_<double>(3, 3) << E(0, 0), E(0, 1), r13,//关于旋转矩阵的r13和r23的放置位置，有两种方式
                                   E(1, 0), E(1, 1), r23,
                                   r3(0, 0), r3(1, 0), r3(2, 0));
        //R = (Mat_<double>(3, 3) << E(0, 0), E(0, 1), r3(0, 0),
        //                           E(1, 0), E(1, 1), r3(1, 0),
        //                           r13, r23, r3(2, 0));

        Rodrigues(R, R_vec);//opencv--Mat vector<double>

        Rs.push_back(R_vec);
        Ts.push_back(T);

        //初始化优化所需参数
        for (int i = 0; i < 3; i++) {
            parameters_[k * 6 + i] = R_vec[i];
            parameters_[k * 6 + 3 + i] = T[i];
        }
        //初始化优化所需参数


        vector<double> e_u(point_num);
        vector<double> e_v(point_num);
        double pic_error_u = 0;
        double pic_error_v = 0;
    }

    //初始化优化所需参数世界坐标
    for (int i = 0; i < point_num; i++) {
        parameters_[6 * image_count + i * 3] = object_points[i].x;
        parameters_[6 * image_count + i * 3 + 1] = object_points[i].y;
        parameters_[6 * image_count + i * 3 + 2] = 0;
    }

    parameters_[6 * image_count + 3 * point_num] = alpha;
    parameters_[6 * image_count + 3 * point_num + 1] = beta;
    parameters_[6 * image_count + 3 * point_num + 2] = u0;
    parameters_[6 * image_count + 3 * point_num + 3] = v0;
    parameters_[6 * image_count + 3 * point_num + 4] = 0;//以下是畸变参数  k1
    parameters_[6 * image_count + 3 * point_num + 5] = 0;//                k2
    parameters_[6 * image_count + 3 * point_num + 6] = 0;//                p1
    parameters_[6 * image_count + 3 * point_num + 7] = 0;//                p2
    parameters_[6 * image_count + 3 * point_num + 8] = 0;//                k3

    destroyAllWindows();




    ////////////////////////////上：远心标定//////////////////////////下：优化函数初始化////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    //FILE *fptr = fopen(filename.c_str(), "r");

    //if (fptr == NULL) {
    //    std::cerr << "Error: unable to open file " << filename;
    //    return;
    //};
    
    /////// This wil die horribly on invalid files. Them's the breaks.
    //FscanfOrDie(fptr, "%d", &num_cameras_);
    //FscanfOrDie(fptr, "%d", &num_points_);
    //FscanfOrDie(fptr, "%d", &num_observations_);

    //std::cout << "Header: " << num_cameras_
    //          << " " << num_points_
    //          << " " << num_observations_<<'\n';

    //point_index_ = new int[num_observations_];
    //camera_index_ = new int[num_observations_];


    //observations_ = new double[2 * num_observations_];

    //constpoint_ = new double[3 * num_points_];//////////add the constpoint///////////

    //num_parameters_ = 6 * num_cameras_ + 3 * num_points_+ 9;///////////////////////
    //parameters_ = new double[num_parameters_];
    //
    //reprojectu_ = new double[num_points_] ;////////////////////////////////////////////////////////////////////
    //reprojectv_ = new double[num_points_] ;
    //reprojectu_ = {0};
    //reprojectv_ = {0};

    //观测的数据0 1 345.52334 2354.43534
    //for (int i = 0; i < num_observations_; ++i) {
    //    FscanfOrDie(fptr, "%d", camera_index_ + i);
    //    FscanfOrDie(fptr, "%d", point_index_ + i);
    //    for (int j = 0; j < 2; ++j) {
    //        FscanfOrDie(fptr, "%lf", observations_ + 2 * i + j);
    //    }
    //}

    //每个位姿的旋转向量，平移向量，世界坐标，fx,fy,u0,v0,0,0,0,0,0
    //for (int i = 0; i < num_parameters_; ++i) {
    //    FscanfOrDie(fptr, "%lf", parameters_ + i);  
    //}
    
    /////////////////////拷贝一个constpoint数组////////世界坐标///
    memcpy(constpoint_, parameters_ + 6 * num_cameras_, 3 * num_points_ * sizeof(double));

    //fclose(fptr);


    /// ///////如果运用了四元数////////////////////
    use_quaternions_ = use_quaternions;
    if (use_quaternions) {
        // Switch the angle-axis rotations to quaternions.
        num_parameters_ = 10 * num_cameras_ + 3 * num_points_;
        double *quaternion_parameters = new double[num_parameters_];
        double *original_cursor = parameters_;
        double *quaternion_cursor = quaternion_parameters;
        for (int i = 0; i < num_cameras_; ++i) {
            AngleAxisToQuaternion(original_cursor, quaternion_cursor);
            quaternion_cursor += 4;
            original_cursor += 3;
            for (int j = 4; j < 10; ++j) {
                *quaternion_cursor++ = *original_cursor++;
            }
        }
        // Copy the rest of the points.
        for (int i = 0; i < 3 * num_points_; ++i) {
            *quaternion_cursor++ = *original_cursor++;
        }
        // Swap in the quaternion parameters.
        delete[]parameters_;
        parameters_ = quaternion_parameters;
    }
}


void BALProblem::WriteToFile(const std::string &filename) const {
    FILE *fptr = fopen(filename.c_str(), "w");
    
//     std::string filenameR = filename;
//     replace(filenameR.begin(),filenameR.end(),'L','R');
//     FILE *fptrR = fopen(filenameR.c_str(), "w");

    if (fptr == NULL ) {//|| fptrR == NULL
        std::cerr << "Error: unable to open file " << filename;
        return;
    }

     fprintf(fptr, "%d %d %d %d\n", num_cameras_, num_cameras_, num_points_, num_observations_);
 
     for (int i = 0; i < num_observations_; ++i) {
         fprintf(fptr, "%d %d", camera_index_[i], point_index_[i]);
         for (int j = 0; j < 2; ++j) {
             fprintf(fptr, " %.6f", observations_[2 * i + j]);
         }
         fprintf(fptr, "\n");
     }

    for (int i = 0; i < num_cameras(); ++i) {
        double angleaxis[10];
        
        if (use_quaternions_) {
            //OutPut in angle-axis format.
            QuaternionToAngleAxis(parameters_ + 10 * i, angleaxis);
            memcpy(angleaxis + 3, parameters_ + 10 * i + 4, 6 * sizeof(double));
        } else {
            memcpy(angleaxis, parameters_ + 6 * i, 6 * sizeof(double));
//             memcpy(angleaxisR, parametersR_ + 6 * i, 6 * sizeof(double));
        }
//         fprintf(fptr,"camera%d: ",i);
        for (int j = 0; j < 3; ++j) {
            fprintf(fptr, "%.5g  ", angleaxis[j]);
            //fprintf(fptrR, "%.5g  ", angleaxisR[j]);
        }
        fprintf(fptr, "\n");
        //fprintf(fptrR, "\n");
    }
    ///这个for循环写入平移向量
    for(int i = 0;i < num_cameras(); ++i){
        double angleaxis[10];
        //double angleaxisR[10];
        if (use_quaternions_) {
            //OutPut in angle-axis format.
            QuaternionToAngleAxis(parameters_ + 10 * i, angleaxis);
            memcpy(angleaxis + 3, parameters_ + 10 * i + 4, 6 * sizeof(double));
        } else {
            memcpy(angleaxis, parameters_ + 6 * i, 6 * sizeof(double));
//             memcpy(angleaxisR, parametersR_ + 6 * i, 6 * sizeof(double));
        }
//         fprintf(fptr,"camera%d: ",i);
        for (int j = 3; j < 6; ++j) {
            fprintf(fptr, "%.5g  ", angleaxis[j]);
            //fprintf(fptrR, "%.5g  ", angleaxisR[j]);
        }
        fprintf(fptr, "\n");
        //fprintf(fptrR, "\n");
    }
    

    const double *points = parameters_ + camera_block_size() * num_cameras_;
    for (int i = 0; i < num_points(); ++i) {
        const double *point = points + i * point_block_size();
//         fprintf(fptr,"point%d: ",i);
        for (int j = 0; j < point_block_size(); ++j) {
            fprintf(fptr, "%.5g  ", point[j]);
            //fprintf(fptrR, "%.5g  ", point[j]);
        }
        fprintf(fptr, "\n");
        //fprintf(fptrR, "\n");
    }
    //写入内参
    const double *inparameter = parameters_+camera_block_size()*num_cameras_+point_block_size()*num_points_;
//     const double *inparameterR = parametersR_ + camera_block_size()*num_cameras_;
    for(int i = 0;i < 9;i++){/////根据内参的优化个数修改打印出来的数量//////
        fprintf(fptr,"%.12f %d %d\n",inparameter[i],0,0);
//         fprintf(fptrR,"%f %d %d\n",inparameterR[i],0,0);
    }
    fclose(fptr);
    //fclose(fptrR);
    
    
}


void BALProblem::ReprojectError(BALProblem& bal_problem) {
    const int point_block_size = bal_problem.point_block_size();//3
    const int camera_block_size = bal_problem.camera_block_size();//6
    double* inparameters = bal_problem.mutable_inparameters();
    double* points = bal_problem.mutable_points();
    double* cameras = bal_problem.mutable_cameras();
    const double* observations = bal_problem.observations();//数组observation_

    double errorx = 0;
    double errory = 0;

    vector<double> obserrorx(bal_problem.num_observations(),0), obserrory(bal_problem.num_observations(),0);
    //////////平均误差
    for (int i = 0; i < bal_problem.num_observations(); ++i) {

        double* camera = cameras + camera_block_size * bal_problem.camera_index()[i];
        double* point = points + point_block_size * bal_problem.point_index()[i];
        double* inparameter = inparameters;
 
        double predictions[2];
        CamProjectionWithDistortion(camera, point, inparameter, predictions);
       
        errorx = errorx + abs(predictions[0] - observations[2 * i + 0]);
        errory = errory + abs(predictions[1] - observations[2 * i + 1]);

        obserrorx[i] = predictions[0] - observations[2 * i + 0];
        obserrory[i] = predictions[1] - observations[2 * i + 1];
        

    }
    cout << "average error:   u: " << errorx / bal_problem.num_observations() << endl;
    cout << "                 v: " << errory / bal_problem.num_observations() << endl;


    //////////标准差
    double tmpx = 0;
    double tmpy = 0;

    for (int i = 0; i < bal_problem.num_observations(); i++) {

        tmpx = tmpx + pow(abs(obserrorx[i]) - errorx / bal_problem.num_observations(), 2);
        tmpy = tmpy + pow(abs(obserrory[i]) - errory / bal_problem.num_observations(), 2);
    }
    
    double stderrorx = sqrt(tmpx / bal_problem.num_observations());
    double stderrory = sqrt(tmpy / bal_problem.num_observations());
    cout << "standard deviations:" << endl;
    cout << "                       x:  " << stderrorx << endl;
    cout << "                       y:  " << stderrory << endl;
    cout << "changdu  = " << obserrorx.size() << endl;
    //画图 重投影误差图
    plt::figure(1);
    for (int i = 0; i < bal_problem.num_cameras(); i++) {
        int flag = i * bal_problem.num_points();
        vector<double> ex(bal_problem.num_points());
        vector<double> ey(bal_problem.num_points());
        ex.assign(&obserrorx[flag], &obserrorx[flag] + bal_problem.num_points());
        ey.assign(&obserrory[flag], &obserrory[flag] + bal_problem.num_points());
        switch (i)
        {
        case 0:
            plt::scatter(ex, ey, 5, { {"color","red"},{"marker","<"}, {"label","1"} });
            continue;
        case 1:
            plt::scatter(ex, ey, 5, { {"color","red"},{"marker","o"}, {"label","2"} });
            continue;
        case 2:
            plt::scatter(ex, ey, 5, { {"color","red"},{"marker","8"}, {"label","3"} });
            continue;
        case 3:
            plt::scatter(ex, ey, 5, { {"color","red"},{"marker","P"}, {"label","4"} });
            continue;
        case 4:
            plt::scatter(ex, ey, 5, { {"color","red"},{"marker","H"}, {"label","5"} }); 
            continue;
        case 5:
            plt::scatter(ex, ey, 5, { {"color","red"},{"marker","+"}, {"label","6"} });
            continue;
        case 6:
            plt::scatter(ex, ey, 5, { {"color","blue"},{"marker","s"}, {"label","7"} });
            continue;
        case 7:
            plt::scatter(ex, ey, 5, { {"color","green"},{"marker","d"}, {"label","8"} });
            continue;
        case 8:
            plt::scatter(ex, ey, 5, { {"color","blue"},{"marker","x"}, {"label","9"} });
            continue;
        case 9:
            plt::scatter(ex, ey, 5, { {"color","aqua"},{"marker","x"}, {"label","10"} });
            continue;
        case 10:
            plt::scatter(ex, ey, 5, { {"color","aliceblue"},{"marker","x"}, {"label","11"} });
            continue;
        case 11:
            plt::scatter(ex, ey, 5, { {"color","cyan"},{"marker","x"}, {"label","12"} });
            continue;
        case 12:
            plt::scatter(ex, ey, 5, { {"color","fuchsia"},{"marker","x"}, {"label","13"} });
            continue;
        case 13:
            plt::scatter(ex, ey, 5, { {"color","gold"},{"marker","x"}, {"label","14"} });
            continue;
        case 14:
            plt::scatter(ex, ey, 5, { {"color","lavender"},{"marker","x"}, {"label","15"} });
            continue;
        case 15:
            plt::scatter(ex, ey, 5, { {"color","lightcoral"},{"marker","x"}, {"label","16"} });
            continue;
        case 16:
            plt::scatter(ex, ey, 5, { {"color","tan"},{"marker","x"}, {"label","17"} });
            continue;
        case 17:
            plt::scatter(ex, ey, 5, { {"color","thistle"},{"marker","x"}, {"label","18"} });
            continue;
        case 18:
            plt::scatter(ex, ey, 5, { {"color","violet"},{"marker","x"}, {"label","19"} });
            continue;
        case 19:
            plt::scatter(ex, ey, 5, { {"color","plum"},{"marker","x"}, {"label","20"} });
            continue;
        case 20:
            plt::scatter(ex, ey, 5, { {"color","rosybrown"},{"marker","x"}, {"label","21"} });
            continue;
        case 21:
            plt::scatter(ex, ey, 5, { {"color","seagreen"},{"marker","x"}, {"label","22"} });
            continue;
        case 22:
            plt::scatter(ex, ey, 5, { {"color","silver"},{"marker","x"}, {"label","23"} });
            continue;
        default:
            plt::scatter(ex, ey, 5, { {"color","oldlace"},{"marker","x"}, {"label","24"} });
            continue;
            
        }
        
    }
    plt::grid(1);
    plt::xlabel("x(pixel)");
    plt::ylabel("y(pixel)");
    plt::xlim(-0.4, 0.4);
    plt::ylim(-0.4, 0.4);
    plt::legend("upper right");
    plt::title("Reprojection error(in pixel)");
    plt::show();


    //画图 世界坐标图
    //Matrix<double, 9, 9> X, Y, Z;
    //for (int i = 0; i < int(sqrt(point_num)); i++) {
    //    for (int j = 0; j < int(sqrt(point_num)); j++) {
    //        int index = 3 * (i * sqrt(point_num) + j);
    //        X(i, j) = points[index];
    //        Y(i, j) = points[index + 1];
    //        Z(i, j) = points[index + 2];

    //    }
    //}
    //
    //plt::plot_surface(X, Y, Z);
    //plt::show();

}


void BALProblem::Set_threshold_for_reproject(BALProblem& bal_problem) {
    const int point_block_size = bal_problem.point_block_size();//3        
    const int camera_block_size = bal_problem.camera_block_size();//6
    double* inparameters = bal_problem.mutable_inparameters();
    double* points = bal_problem.mutable_points();

    double* cameras = bal_problem.mutable_cameras();
    const double* observations = bal_problem.observations();//数组observation_

    int count = 0;
    for (int i = 0; i < bal_problem.num_observations(); ++i) {

        double errorx = 0;
        double errory = 0;
        
        double* point = points + point_block_size * bal_problem.point_index()[i];
        double* inparameter = inparameters;
        //需要修改的参数camera_index_,point_index_和 observation
        double* camera = cameras + camera_block_size * bal_problem.camera_index()[i];

        double predictions[2];
        CamProjectionWithDistortion(camera, point, inparameter, predictions);

        errorx = predictions[0] - observations[2 * i + 0];
        errory = predictions[1] - observations[2 * i + 1];
        if (errorx < 1 && errory < 1) {
            observations_[2 * count + 0] = observations[2 * i + 0];
            observations_[2 * count + 1] = observations[2 * i + 1];
            camera_index_[count] = camera_index_[i];
            point_index_[count] = point_index_[i];
            ++count;
        }
    }
    num_observations_ = count;


}

// Write the problem to a PLY file for inspection in Meshlab or CloudCompare
void BALProblem::WriteToPLYFile(const std::string &filename) const {
    std::ofstream of(filename.c_str());

    of << "ply"
       << '\n' << "format ascii 1.0"
       << '\n' << "element vertex " << num_cameras_ + num_points_
       << '\n' << "property float x"
       << '\n' << "property float y"
       << '\n' << "property float z"
       << '\n' << "property uchar red"
       << '\n' << "property uchar green"
       << '\n' << "property uchar blue"
       << '\n' << "end_header" << std::endl;

    // Export extrinsic data (i.e. camera centers) as green points.
    double angle_axis[3];
    double center[3];
    for (int i = 0; i < num_cameras(); ++i) {
        const double *camera = cameras() + camera_block_size() * i;
        CameraToAngelAxisAndCenter(camera, angle_axis, center);
        of << center[0] << ' ' << center[1] << ' ' << center[2]<<' '
           << "0 255 0" << '\n';
    }

    // Export the structure (i.e. 3D Points) as white points.
    const double *points = parameters_ + camera_block_size() * num_cameras_;
    for (int i = 0; i < num_points(); ++i) {
        const double *point = points + i * point_block_size();
        for (int j = 0; j < point_block_size(); ++j) {
            of << point[j] << ' ';
        }
        of << "255 255 255\n";
    }
    of.close();
}

void BALProblem::CameraToAngelAxisAndCenter(const double *camera,
                                            double *angle_axis,
                                            double *center) const {
    VectorRef angle_axis_ref(angle_axis, 3);
    if (use_quaternions_) {
        QuaternionToAngleAxis(camera, angle_axis);
    } else {
        angle_axis_ref = ConstVectorRef(camera, 3);
    }

    // c = -R't
    Eigen::VectorXd inverse_rotation = -angle_axis_ref;
    AngleAxisRotatePoint(inverse_rotation.data(),
                         camera + camera_block_size() - 6,
                         center);
    VectorRef(center, 3) *= -1.0;
}

void BALProblem::AngleAxisAndCenterToCamera(const double *angle_axis,
                                            const double *center,
                                            double *camera) const {
    ConstVectorRef angle_axis_ref(angle_axis, 3);
    if (use_quaternions_) {
        AngleAxisToQuaternion(angle_axis, camera);
    } else {
        VectorRef(camera, 3) = angle_axis_ref;
    }

    // t = -R * c
    AngleAxisRotatePoint(angle_axis, center, camera + camera_block_size() - 6);
    VectorRef(camera + camera_block_size() - 6, 3) *= -1.0;
}

void BALProblem::Normalize() {
    // Compute the marginal median of the geometry
    std::vector<double> tmp(num_points_);
    Eigen::Vector3d median;
    double *points = mutable_points();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < num_points_; ++j) {
            tmp[j] = points[3 * j + i];
        }
        median(i) = Median(&tmp);
    }

    for (int i = 0; i < num_points_; ++i) {
        VectorRef point(points + 3 * i, 3);
        tmp[i] = (point - median).lpNorm<1>();
    }

    const double median_absolute_deviation = Median(&tmp);

    // Scale so that the median absolute deviation of the resulting
    // reconstruction is 100

    const double scale = 100.0 / median_absolute_deviation;

    // X = scale * (X - median)
    for (int i = 0; i < num_points_; ++i) {
        VectorRef point(points + 3 * i, 3);
        point = scale * (point - median);
    }

    double *cameras = mutable_cameras();
    double angle_axis[3];
    double center[3];
    for (int i = 0; i < num_cameras_; ++i) {
        double *camera = cameras + camera_block_size() * i;
        CameraToAngelAxisAndCenter(camera, angle_axis, center);
        // center = scale * (center - median)
        VectorRef(center, 3) = scale * (VectorRef(center, 3) - median);
        AngleAxisAndCenterToCamera(angle_axis, center, camera);
    }
}

void BALProblem::Perturb(const double rotation_sigma,
                         const double translation_sigma,
                         const double point_sigma) {
    assert(point_sigma >= 0.0);
    assert(rotation_sigma >= 0.0);
    assert(translation_sigma >= 0.0);

    double *points = mutable_points();
    if (point_sigma > 0) {
        for (int i = 0; i < num_points_; ++i) {
            PerturbPoint3(point_sigma, points + 3 * i);
        }
    }

    for (int i = 0; i < num_cameras_; ++i) {
        double *camera = mutable_cameras() + camera_block_size() * i;

        double angle_axis[3];
        double center[3];
        // Perturb in the rotation of the camera in the angle-axis
        // representation
        CameraToAngelAxisAndCenter(camera, angle_axis, center);
        if (rotation_sigma > 0.0) {
            PerturbPoint3(rotation_sigma, angle_axis);
        }
        AngleAxisAndCenterToCamera(angle_axis, center, camera);

        if (translation_sigma > 0.0)
            PerturbPoint3(translation_sigma, camera + camera_block_size() - 6);
    }
}
