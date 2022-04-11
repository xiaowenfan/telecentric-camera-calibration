#define _CRT_SECURE_NO_WARNINGS
#include <ceres/ceres.h>
#include "common.h"
#include "SnavelyReprojectionError.h"
#include "glog/logging.h"
#include <iostream>  
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>                 
#include <string>                                 
#include <fstream>                                 
#include <eigen3/Eigen/Core>                        
#include <eigen3/Eigen/Dense>                      
#include <vector>                                
#include <eigen3/Eigen/QR>                     
#include <Python.h>                                 
#include <chrono>                                 
#include <thread>                                 
#include <ctime>                                 


/*
*！！！！！ 根据实验需要调整的相关参数！！！！！
*-------------- 在common.cpp中修改--------------
* image_count----标靶图像数量
* w_sq----标靶中特征点之间的物理距离
* point_num----标靶上特征点数量
* w,h----w表示特征点阵的长，h表示特征点阵的宽
* .bmp----读取的图片格式
*/
using namespace Eigen;
using namespace std;
using namespace cv;
using namespace std::chrono;

template<typename T>
static inline bool CamProjectionWithDistortion(const T* camera, const T* point, const T* inparameter, T* predictions) {
    T p[3];
    // 罗德里格斯公式
    AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5]是平移向量
    p[0] += camera[3];    //Xc
    p[1] += camera[4];    //P = R*X+t     R是旋转矩阵   Yc
    p[2] += camera[5];    //Zc     

        // Apply second and fourth order radial distortion
    const T& l1 = inparameter[4];  //径向畸变---k1
    const T& l2 = inparameter[5];  //径向畸变---k2
    const T& l3 = inparameter[8];  //径向畸变---k3
    const T& p1 = inparameter[6];//切向畸变---p1
    const T& p2 = inparameter[7];//切向畸变---p2
    const T& focalx = inparameter[0];
    const T& focaly = inparameter[1];
    const T& cx = inparameter[2];
    const T& cy = inparameter[3];

    T r2 = p[0] * p[0] + p[1] * p[1];  //  xp * xp + yp * yp; 

    //双远心镜头只考虑径向畸变，畸变系数：k1,k2,k3
    T distortion = T(1.0) + r2 * (l1 + r2 * (l2 + l3 * r2));
    T xd = p[0] * distortion;
    T yd = p[1] * distortion;

    predictions[0] = focalx * xd + cx;
    predictions[1] = focaly * yd + cy;

    return true;
}

void SolveBA(BALProblem& bal_problem) {
    
    const int point_block_size = bal_problem.point_block_size();//3
    const int camera_block_size = bal_problem.camera_block_size();//6
    const int inparameter_block_size = bal_problem.inparameter_block_size();//9
    double* points = bal_problem.mutable_points();
    double* cameras = bal_problem.mutable_cameras();
    double* inparameters = bal_problem.mutable_inparameters();
    const double* observations = bal_problem.observations();//数组observation_
    const double* constpoints = bal_problem.constpoint();

    ceres::Problem problem;
    for (int i = 0; i < bal_problem.num_observations(); ++i) {

        double* camera = cameras + camera_block_size * bal_problem.camera_index()[i];
        double* point = points + point_block_size * bal_problem.point_index()[i];
        double* inparameter = inparameters;

        ceres::CostFunction* cost_function;
        
         cost_function = SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1], constpoints[3 * bal_problem.point_index()[i] + 0], constpoints[3 * bal_problem.point_index()[i] + 1], constpoints[3 * bal_problem.point_index()[i] + 2]);
         ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);
         problem.AddResidualBlock(cost_function, loss_function, camera, point, inparameter);//loss_function
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

int main(int argc, char** argv)
{
    //时间戳配置
    // Step one: 定义一个clock
    typedef system_clock sys_clk_t;
    // Step two: 分别获取两个时刻的时间
    typedef system_clock::time_point time_point_t;
    typedef duration<int, std::ratio<1, 1000>> mili_sec_t;
    google::InitGoogleLogging(argv[0]);

    //图片存储位置
    string rootpath = "D:/xwf/日常杂事/20220318图片/tu/telecalibrate0327/";

    BALProblem bal_problem(rootpath);//构造函数，功能：初值估计

    //写入初值估计的文件
    //string init_txt = rootpath + "Initial2.txt";
    //bal_problem.WriteToFile(init_txt);

    //显示初值估计的重投影误差
    bal_problem.ReprojectError(bal_problem);

    //去掉误差大的点
    //int threshold = 1;//误差阈值
    //bal_problem.Set_threshold_for_reproject(bal_problem,threshold);
    
    //计算优化时间
    time_point_t start = sys_clk_t::now();    
    SolveBA(bal_problem);//优化
    time_point_t end = sys_clk_t::now();   
    cout << "time_cost = " << (time_point_cast<mili_sec_t>(end) - time_point_cast<mili_sec_t>(start)).count() << endl;
    cout << "......Optimized......." << endl;

    //显示优化后重投影误差
    bal_problem.ReprojectError(bal_problem);

    //写入优化后参数
    //string opti_txt = rootpath + "Optimized2.txt";
    //bal_problem.WriteToFile(opti_txt);
    
    return 0;
}
