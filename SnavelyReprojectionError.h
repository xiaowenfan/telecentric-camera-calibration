#ifndef SnavelyReprojection_H
#define SnavelyReprojection_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"

class SnavelyReprojectionError {
public:
    SnavelyReprojectionError(double observation_x, double observation_y, double constpoint_x, double constpoint_y, double constpoint_z): observed_x(observation_x), observed_y(observation_y), conpoint_x(constpoint_x), conpoint_y(constpoint_y), conpoint_z(constpoint_z){}

    template<typename T>
    bool operator()(const T* const camera,//3个旋转向量+3个平移向量
                    const T* const point,//xyz三个方向
                    const T* const inparameter,//9个内参，fx,fy,ux,uy,k1,k2,p1,p2,k3
                    T* residuals) const {
        // camera[0,1,2] are the angle-axis rotation
        T predictions[2];
        
        CamProjectionWithDistortion(camera, point,inparameter, predictions);
          
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);
        residuals[2] = point[0] * conpoint_y - point[1] * conpoint_x;
        residuals[3] = point[0] - conpoint_x;
        residuals[4] = point[1] - conpoint_y;
        residuals[5] = point[2] - conpoint_z;

        return true;
    }
    
    // camera : 9 dims array
    // [0-2] : angle-axis rotation
    // [3-5] : translateion
    // [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
    // point : 3D location.
    // predictions : 2D predictions with center of the image plane.
    template<typename T>
    static inline bool CamProjectionWithDistortion(const T *camera, const T *point,const T *inparameter, T *predictions) {
        // Rodrigues' formula
        T p[3];
        AngleAxisRotatePoint(camera, point, p);//旋转向量--->旋转矩阵
        // camera[3,4,5] are the translation
        p[0] += camera[3];    //Xc
        p[1] += camera[4];//P = R*X+t     R是旋转矩阵   Yc
        p[2] += camera[5];    //Zc        
        //std::cout<<"图像坐标系下的Z轴坐标："<<p[2]<<"   ";
        // Compute the center fo distortion
        
//         T xp = p[0]/p[2];//p = P/P.z   归一化坐标 
//         T yp = p[1]/p[2];                                           // 注释是远心相机的配置

        // Apply second and fourth order radial distortion
        const T &l1 = inparameter[4];  //jing xiang jibian
        const T &l2 = inparameter[5];  //bolengjing jibiannnnnnnnnnnnnnnn 
        const T &l3 = inparameter[8];  //bolengjing jibian 
        const T &p1 = inparameter[6];//qie xiang jibian/现修改为薄棱镜畸变参数 x
        const T &p2 = inparameter[7];//qie xiang jibian /现修改为薄棱镜畸变参数 y
        const T &focalx = inparameter[0];
        const T &focaly = inparameter[1];
        const T &cx = inparameter[2];
        const T &cy = inparameter[3];
//         T du = p[0] - cx;
//         T dv = p[1] - cy;
        
        T r2 = p[0]*p[0] + p[1]*p[1];  //  xp * xp + yp * yp;               //远心相机的配置  du==p[0]   dv == p[1]
        
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

    static ceres::CostFunction *Create(const double observed_x, const double observed_y,const double conpoint_x,const double conpoint_y,const double conpoint_z) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 6, 6, 3, 9>(//9个内参，3表示一个世界点有xyz三个值，6表示一个相机有3个旋转向量+3个平移向量，第一个6表示有6个residuals
            new SnavelyReprojectionError(observed_x, observed_y , conpoint_x, conpoint_y, conpoint_z)));
    }///////////////////////////////////

private:
    double observed_x;
    double observed_y;
    double conpoint_x;
    double conpoint_y;
    double conpoint_z;
    
};

#endif // SnavelyReprojection.h

