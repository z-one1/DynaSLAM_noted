/**
* This file is part of DynaSLAM.
*
* Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/bertabescos/DynaSLAM>.
*
*/

#include "MaskNet.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <dirent.h>
#include <errno.h>

namespace DynaSLAM
{
/* 代码运行计时 */
#define U_SEGSt(a)\
    gettimeofday(&tvsv,0);\ // 系统调用：获取当前时间，'0'表示不使用时区
    a = tvsv.tv_sec + tvsv.tv_usec/1000000.0 // 获取当前系统时间，秒单位
struct timeval tvsv;
double t1sv, t2sv,t0sv,t3sv;
void tic_initsv(){U_SEGSt(t0sv);}
void toc_finalsv(double &time){U_SEGSt(t3sv); time =  (t3sv- t0sv)/1;}
void ticsv(){U_SEGSt(t1sv);}
void tocsv(){U_SEGSt(t2sv);}
// std::cout << (t2sv - t1sv)/1 << std::endl;}

SegmentDynObject::SegmentDynObject(){
    /* 在C++中嵌入Python */
    std::cout << "Importing Mask R-CNN Settings..." << std::endl;
    ImportSettings();
    std::string x;
    setenv("PYTHONPATH", this->py_path.c_str(), 1); // 设置环境变量
    x = getenv("PYTHONPATH");
    Py_Initialize(); // 初始化python解释器
    this->cvt = new NDArrayConverter(); // NumPy -> cv::Mat
    this->py_module = PyImport_ImportModule(this->module_name.c_str());
    assert(this->py_module != NULL);
    this->py_class = PyObject_GetAttrString(this->py_module, this->class_name.c_str());
    assert(this->py_class != NULL);
    this->net = PyInstance_New(this->py_class, NULL, NULL);
    assert(this->net != NULL);
    std::cout << "Creating net instance..." << std::endl;
    cv::Mat image  = cv::Mat::zeros(480,640,CV_8UC3); //Be careful with size!!
    std::cout << "Loading net parameters..." << std::endl;
    GetSegmentation(image);
}

SegmentDynObject::~SegmentDynObject(){
    delete this->py_module;
    delete this->py_class;
    delete this->net;
    delete this->cvt;
}

cv::Mat SegmentDynObject::GetSegmentation(cv::Mat &image,std::string dir, std::string name){
    cv::Mat seg = cv::imread(dir+"/"+name,CV_LOAD_IMAGE_UNCHANGED);
    // 还没生成mask，满足seg.empty()
    if(seg.empty()){
        PyObject* py_image = cvt->toNDArray(image.clone()); // cv::Mat -> Np.array
        assert(py_image != NULL);
        //* 相当于执行python语句： GetDynSeg(py_image) 得到分割结果
        PyObject* py_mask_image = PyObject_CallMethod(this->net, const_cast<char*>(this->get_dyn_seg.c_str()),"(O)",py_image);
        seg = cvt->toMat(py_mask_image).clone(); // Np.array -> cv::Mat
        seg.cv::Mat::convertTo(seg,CV_8U);//0 background y 1 foreground
        //* 检查图像储存路径，储存图像
        if(dir.compare("no_save")!=0){
            DIR* _dir = opendir(dir.c_str());
            if (_dir) {closedir(_dir);}
            else if (ENOENT == errno)
            {
                const int check = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                if (check == -1) {
                    std::string str = dir;
                    str.replace(str.end() - 6, str.end(), "");
                    mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                }
            }
            cv::imwrite(dir+"/"+name,seg);
        }
    }
    return seg;
}

void SegmentDynObject::ImportSettings(){
    std::string strSettingsFile = "./Examples/RGB-D/MaskSettings.yaml";
    cv::FileStorage fs(strSettingsFile.c_str(), cv::FileStorage::READ);
    fs["py_path"] >> this->py_path;
    fs["module_name"] >> this->module_name;
    fs["class_name"] >> this->class_name;
    fs["get_dyn_seg"] >> this->get_dyn_seg;

    // std::cout << "    py_path: "<< this->py_path << std::endl;
    // std::cout << "    module_name: "<< this->module_name << std::endl;
    // std::cout << "    class_name: "<< this->class_name << std::endl;
    // std::cout << "    get_dyn_seg: "<< this->get_dyn_seg << std::endl;
}


}






















