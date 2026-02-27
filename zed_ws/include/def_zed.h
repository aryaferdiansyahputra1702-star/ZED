#ifndef DEF_ZED_H
#define DEF_ZED_H
#include "msgpack.hpp"

struct bbox_t {
  unsigned int x, y, w, h;  // (x,y) - top-left corner, (w, h) - width & height of bounded box
  float x_3d, y_3d, z_3d;  // center of object (in Meters) if ZED 3D Camera is used
};

struct objPose{
    objPose():x(0),y(0),z(0),isDetected(false){}
    objPose(bbox_t bbx){
    if(std::isfinite(bbx.x_3d)){
        x = bbx.x_3d;
        y = bbx.y_3d;
        z = bbx.z_3d;
    }
    }
    double x, y, z;
    bool isDetected;
    MSGPACK_DEFINE(x,y, z,isDetected);

    
};

struct dataZed{
  objPose Bola, Robot;
  MSGPACK_DEFINE(Bola, Robot);

};


#endif