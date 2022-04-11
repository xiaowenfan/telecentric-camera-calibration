// telecalibrate.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#define _CRT_SECURE_NO_WARNINGS
#include <ceres/ceres.h>
#include "common.h"
#include "SnavelyReprojectionError.h"
#include "glog/logging.h"
#include <iostream>  
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>                 //
#include <string>                                 //
#include <fstream>                                 //
#include <eigen3/Eigen/Core>                         //                         2022.1.14
#include <eigen3/Eigen/Dense>                       //                          修改了图片路径,w,h,point_num
#include <vector>                                 //                    修改了common.cpp中的图片输入格式‘.png’和斑点检测参数
#include <eigen3/Eigen/QR>                         //         在common.cpp中添加了直接输入特征点图像坐标函数，不用Opencv识别
#include <Python.h>                                 //
#include <chrono>                                 //
#include <thread>                                 //
#include <ctime>                                 //
//#include "matplotlibcpp.h"
//#include <cmath>
//namespace plt = matplotlibcpp;
//分支修改再上传


#define image_count 23
#define w_sq 1.5//mm
#define point_num 81 //12*9=108
using namespace Eigen;
using namespace std;
using namespace cv;
using namespace std::chrono;


template<typename T>
static inline bool CamProjectionWithDistortion(const T* camera, const T* point, const T* inparameter, T* predictions) {
    // Rodrigues' formula
    T p[3];
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
    const T& l1 = inparameter[4];  //jing xiang jibian
    const T& l2 = inparameter[5];  //bolengjing jibiannnnnnnnnnnnnnnn 
    const T& l3 = inparameter[8];  //bolengjing jibian 
    const T& p1 = inparameter[6];//qie xiang jibian/现修改为薄棱镜畸变参数 x
    const T& p2 = inparameter[7];//qie xiang jibian /现修改为薄棱镜畸变参数 y
    const T& focalx = inparameter[0];
    const T& focaly = inparameter[1];
    const T& cx = inparameter[2];
    const T& cy = inparameter[3];
    //         T du = p[0] - cx;
    //         T dv = p[1] - cy;

    T r2 = p[0] * p[0] + p[1] * p[1];  //  xp * xp + yp * yp;               //远心相机的配置  du==p[0]   dv == p[1]

    T distortion = T(1.0) + r2 * (l1 + r2 * (l2 + l3 * r2));//add the k3
//         T distortion =  T(1.0) + r2 * ( l1 + r2 * l2);
    T xd = p[0] * distortion;// +T(2.0) * p1 * p[0] * p[1] + p2 * (r2 + T(2.0) * p[0] * p[0]);// +p[0] * (p1 * p[0] - p2 * p[1]);//远心相机的配置  xp * distortion + T(2.0) * p1 * xp * yp + p2 * (r2 +T(2) * xp * xp);du
    T yd = p[1] * distortion;// +p1 * (r2 + T(2.0) * p[1] * p[1]) + T(2.0) * p2 * p[0] * p[1];// +p[1] * (p1 * p[0] - p2 * p[1]);//远心相机的配置  yp * distortion + p1 * (r2 + T(2.0) * yp * yp) + T(2) * p2 * xp *yp;

    ////modify code according matlab code
//         T xd = l1 * p[0] * r2 + p1 * p[0] * p[0] + p2 * p[0] * p[1] + l2 * r2;
//         T yd = l1 * p[1] * r2 + p2 * p[1] * p[1] + p1 * p[0] * p[1] + l3 * r2;


    predictions[0] = focalx * xd + cx;
    predictions[1] = focaly * yd + cy;
    //std::cout << "predictions[0] = " << predictions[0] << "   predictions[1] = " << predictions[1] << std::endl;

    return true;
}



void SolveBA(BALProblem& bal_problem) {
    
    const int point_block_size = bal_problem.point_block_size();//3
    const int camera_block_size = bal_problem.camera_block_size();//6
    const int inparameter_block_size = bal_problem.inparameter_block_size();//9
    double* points = bal_problem.mutable_points();
    double* cameras = bal_problem.mutable_cameras();
    double* inparameters = bal_problem.mutable_inparameters();


//     double *reprojectionerrors_u = bal_problem.mutable_reprojectionerrors_u();
//     double *reprojectionerrors_v = bal_problem.mutable_reprojectionerrors_v();
    // Observations is 2 * num_observations long array observations
    // [u_1, u_2, ... u_n], where each u_i is two dimensional, the x
    // and y position of the observation.
    const double* observations = bal_problem.observations();//数组observation_
    const double* constpoints = bal_problem.constpoint();



    ceres::Problem problem;

    for (int i = 0; i < bal_problem.num_observations(); ++i) {

        // Each observation corresponds to a pair of a camera and a point
        // which are identified by camera_index()[i] and point_index()[i]
        // respectively.
        double* camera = cameras + camera_block_size * bal_problem.camera_index()[i];
        double* point = points + point_block_size * bal_problem.point_index()[i];
        double* inparameter = inparameters;

        // Each Residual block takes a point and a camera as input
        // and outputs a 2 dimensional Residual
        ceres::CostFunction* cost_function;

        //在这里计算一次重投影误差，误差大的就舍去，不做优化，在画图过程中不显示误差大于阈值的点
        //double predictions[2];
        //CamProjectionWithDistortion(camera, point, inparameter, predictions);
        //double errorx = predictions[0] - observations[2 * i + 0];
        //double errory = predictions[1] - observations[2 * i + 1];
        //double error = (errorx + errory) / 2;

        if(true){
            cost_function = SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1], constpoints[3 * bal_problem.point_index()[i] + 0], constpoints[3 * bal_problem.point_index()[i] + 1], constpoints[3 * bal_problem.point_index()[i] + 2]);
            // If enabled use Huber's loss function.
            ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);

         //         double *reprojectionerror_u = reprojectionerrors_u;
         //         double *reprojectionerror_v = reprojectionerrors_v;

            problem.AddResidualBlock(cost_function, loss_function, camera, point, inparameter);//loss_function
        }
    }

    // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
        << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl;

    std::cout << "Solving ceres BA ... " << endl;
    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;//最小化器类型
    options.gradient_tolerance = 1e-10;
    options.function_tolerance = 1e-10;
    options.max_num_consecutive_invalid_steps = 5;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.initial_trust_region_radius = 1e4;
    options.max_trust_region_radius = 1e16;
    options.min_trust_region_radius = 1e-32;
    options.max_num_iterations = 200;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;//SPARSE_SCHUR
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

}

//已经写入构造函数
void estimate_params(const std::string& root) {
    int w, h;
    w = 9;
    h = 9;
    int num_point = w * h;
    Size pattern_size = Size(w, h);
    Mat srcImage, grayImage;
    string filename, path, imname;
    double m;
    path = root;

    ///world points
    vector<Point2f> object_points;
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            Point2f realPoint;
            realPoint.x = j * w_sq;
            realPoint.y = i * w_sq;
            object_points.push_back(realPoint);
        }
    }

    Matrix<double, Dynamic, Dynamic> M;
    M.setZero(162, 162);  //108*2=216                         
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
    vector<Matrix<double, 2 * point_num, 1>> corner(image_count, Matrix<double, 2 * point_num, 1>(2 * point_num, 1));

    ///detect point each image
    for (int i = 0; i < image_count; i++) {//image_count
        filename = path + "cb0_" + to_string(i) + ".jpg";
        srcImage = imread(filename);
        imname = "img" + to_string(i);
        u0 = srcImage.cols / 2;
        v0 = srcImage.rows / 2;
        vector<Point2f> corners;
        Mat gray;
        cvtColor(srcImage, gray, COLOR_BGR2GRAY);
        int rows = srcImage.rows;
        int cols = srcImage.cols;

        //points detector
        SimpleBlobDetector::Params params;//change the params config according the color of blob
//         params.minThreshold = 0;
//         params.maxThreshold = 150;
//         params.filterByArea = true;
//         params.minArea = 50;
//         params.maxArea = 10e6;
//         params.minDistBetweenBlobs = 10;
        params.filterByColor = true;
        params.blobColor = 0;
        Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
        bool found = findCirclesGrid(gray, pattern_size, corners, CALIB_CB_SYMMETRIC_GRID, detector);
        //drawChessboardCorners(srcImage, pattern_size, corners, found);
        //imshow(imname,srcImage);
        //waitKey(100);
        //if(i==1) imwrite("1.jpg", srcImage);
        cout << "image_" << to_string(i) << "_:" << found << endl;

        //load imgpoints
        //cout << "corners = \n" << corners << endl;
        //cout << "corners(0,0) = " << corners[80].x << endl;
        for (int j = 0,flag = 0; flag < point_num; flag++) {
            corner[i](j, 0) = corners[flag].x;
            corner[i](j + 1, 0) = corners[flag].y;
            j = j + 2;
        }
        //cout << "修改后 ： \n" << corner[i] << endl;

        Matrix<double, 162, 1> h = M.colPivHouseholderQr().solve(corner[i]); //以前216=108*2
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
    Matrix<double, image_count, image_count> G;
    G.setOnes(image_count, image_count);
    Matrix<double, image_count, 1> W;
    W.setZero(image_count, 1);
    for (int k = 0; k < image_count; k++) {
        Matrix<double, 3, 3> H = homographies[k];
        G(k, 1) = -(H(1, 0) * H(1, 0)) - (H(1, 1) * H(1, 1));
        G(k, 2) = -(H(0, 0) * H(0, 0)) - (H(0, 1) * H(0, 1));
        G(k, 3) = 2 * (H(0, 0) * H(1, 0) + H(0, 1) * H(1, 1));
        W(k, 0) = -((H(0, 0) * H(1, 1) - H(0, 1) * H(1, 0)) * (H(0, 0) * H(1, 1) - H(0, 1) * H(1, 0)));
    }

    Matrix<double, image_count, 1> L = G.colPivHouseholderQr().solve(W);//colPivHouseholderQr()

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
    //修改过的GITHUB
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
        cout << "***********file__" << to_string(k + 1) << "__*************" << endl;
        R = (Mat_<double>(3, 3) << E(0, 0), E(0, 1), r13,
            E(1, 0), E(1, 1), r23,
            r3(0, 0), r3(1, 0), r3(2, 0));
   
        Rodrigues(R, R_vec);//opencv--Mat vector<double>

        Rs.push_back(R_vec);
        Ts.push_back(T);

        vector<double> e_u(point_num);
        vector<double> e_v(point_num);
        double pic_error_u = 0;
        double pic_error_v = 0;
    }
    destroyAllWindows();



    ////////write params to file////////////////////////////////////////////
    string filepath = path + "prefinal.txt";
    FILE* fptr = fopen(filepath.c_str(), "w");
    if (fptr == NULL) {
        std::cerr << "Error: unable to open file " << filepath;
        return;
    }
    //image_num-point_num-image*point_num
    fprintf(fptr, "%d %d %d\n", image_count, point_num, image_count * point_num);
    //observation data
    for (int i = 0; i < image_count; i++)
        for (int flag = 0, j = 0; flag < point_num; flag++, j += 2)
            fprintf(fptr, "%d %d %.6f %.6f\n", i, flag, corner[i](j, 0), corner[i](j + 1.0, 0));

    //     //external data each picture
    //     for(int i = 0; i < Rs.size(); i++){
    //         for(int j = 0; j < Rs[i].size(); j++)
    //             fprintf(fptr, "%.6f\n", Rs[i][j]);//Rs[i][j]
    //         for(int j = 0; j < Rs[i].size(); j++)
    //             fprintf(fptr, "%.6f\n", Ts[i][j]);
    //     }

    for (int i = 0; i < Rs.size(); i++) {
        for (int j = 0; j < Rs[i].size(); j++) {
            fprintf(fptr, "%.6f ", Rs[i][j]);
        }
        fprintf(fptr, "\n");
    }

    for (int i = 0; i < Rs.size(); i++) {
        for (int j = 0; j < Rs[i].size(); j++) {
            fprintf(fptr, "%.6f ", Ts[i][j]);
        }
        fprintf(fptr, "\n");
    }

    //     //world points
    for (int i = 0; i < point_num; i++)
    {
        fprintf(fptr, "%.6f ", object_points[i].y);//如果是输出paramsfile.txt就先输出y,后输出x
        fprintf(fptr, "%.6f ", object_points[i].x);
        fprintf(fptr, "%d\n", 0);
    }
    //     //internal param of camera-----fx,fy,u0,v0,k1,k2,p1,p2,k3
    fprintf(fptr, "%.6f\n", alpha);
    fprintf(fptr, "%.6f\n", beta);
    fprintf(fptr, "%.6f\n", u0);
    fprintf(fptr, "%.6f\n", v0);
    for (int i = 0; i < 5; i++)//initial distortion as 0
        fprintf(fptr, "%d\n", 0);

    fclose(fptr);
    ////////write params to file////////////////////////////////////////////
}




int main(int argc, char** argv)
{
    //////////时间戳配置
    // Step one: 定义一个clock
    typedef system_clock sys_clk_t;
    // Step two: 分别获取两个时刻的时间
    typedef system_clock::time_point time_point_t;
    typedef duration<int, std::ratio<1, 1000>> mili_sec_t;


    google::InitGoogleLogging(argv[0]);
    string rootpath = "D:/xwf/日常杂事/20220318图片/tu/telecalibrate0327/";//图片存储位置
    /////////estimate_params(rootpath);

    /////////string path_txt = rootpath + "paramsfile.txt";
    BALProblem bal_problem(rootpath, 0);//0--don`t use quaternions
    //bal_problem.WriteToFile("C:/Users/Administrator/Documents/WeChat Files/wxid_oe4fhjaukstx22/FileStorage/File/2022-02/圆点图片/圆点图片/Initial.txt");
    bal_problem.ReprojectError(bal_problem);
    //去掉误差大的点
    //bal_problem.Set_threshold_for_reproject(bal_problem);
    

    time_point_t start = sys_clk_t::now();    //运行前
    SolveBA(bal_problem);
    time_point_t end = sys_clk_t::now();    //运行后

    cout << "time_cost = " << (time_point_cast<mili_sec_t>(end) - time_point_cast<mili_sec_t>(start)).count() << endl;
    cout << "......Optimized......." << endl;
    bal_problem.ReprojectError(bal_problem);
    //bal_problem.WriteToFile("D:/xwf/日常杂事/20220318图片/tu/telecalibrate0327/Optimized.txt");//优化后参数保存路径
    
    //////////////////////////////////////////////////////////////////////////////////////////////////

    //Mat img = imread("D:/xwf/MicroMeasurementProject/TargetPictureCaptured/0908-vc-blackcircle/1.bmp");
    //if (img.empty())
    //{
    //    cout << "请确认图像文件名称是否正确" << endl;
    //    return -1;
    //}
    //imshow("原图", img);
    //Mat gray;
    //cvtColor(img, gray, COLOR_BGR2GRAY);
    //vector<Vec3f> circles;
    //double dp = 1;
    //double minDist = 10;
    //double param1 = 100;
    //double param2 = 100;
    //int min_radius = 10;
    //int max_radius = 150;
    //HoughCircles(gray, circles, HOUGH_GRADIENT, dp, minDist, param1, param2, min_radius, max_radius);


    //for (int i = 0; i < circles.size(); i++) {
    //    cout << "第" << i + 1 << "个点(" << circles[i][0] << "," << circles[i][1] << "),半径 = " << circles[i][2] << endl;
    //    Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    //    int radius = cvRound(circles[i][2]);
    //    circle(img, center, 3, Scalar(0, 255, 0), -1, 8, 0);
    //    circle(img, center, radius, Scalar(0, 0, 255), 3, 8, 0);
    //}

    //imshow("圆检测结果", img);
    //waitKey(0);

    return 0;
}


// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
