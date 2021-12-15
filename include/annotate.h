#ifndef ANN_H
#define ANN_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "img_processing.h"

namespace annotate{
    cv::Mat add_side_panel(cv::Mat & frame_in);
    void annotate_lanes(std::vector<ImgProcessing::LaneLine> lanes, cv::Mat & frame, cv::Scalar color = cv::Scalar(255,255,128));
};

#endif

// void Graphics::drawTrafficObjects(){
//     // reset images
//     _images.at(1) = _images.at(0).clone();
//     _images.at(2) = _images.at(0).clone();

//     // create overlay from all traffic objects
//     for (auto it : _trafficObjects)    {
//         double posx, posy;
//         it->getPosition(posx, posy);

//         if (it->getType() == ObjectType::objectIntersection) {
//             // cast object type from TrafficObject to Intersection
//             std::shared_ptr<Intersection> intersection = std::dynamic_pointer_cast<Intersection>(it);

//             // draw a simple traffic light: black box defined by 2 edge points and  2 circles
//             cv::Point pt_upperRight(posx - 28, posy - 60);
//             cv::Point pt_lowerLeft(posx + 28, posy + 60);
//             cv::rectangle(_images.at(1), pt_lowerLeft, pt_upperRight, cv::Scalar(0, 0, 0), -1);  // line type -1 ==> filled 
//             // note: colors in BGR system
//             cv::Scalar g_color, r_color;
//             if(intersection->trafficLightIsGreen()){
//                 g_color = cv::Scalar(0, 255, 0);     // bright green
//                 r_color = cv::Scalar(100, 100, 150); // greyish red
//             } else {
//                 g_color = cv::Scalar(100, 150, 100); // greyish green
//                 r_color = cv::Scalar(0, 0, 255);     // bright red
//             }
//             cv::circle(_images.at(1), cv::Point2d(posx, posy-30), 25, r_color, -1);  // upper circle
//             cv::circle(_images.at(1), cv::Point2d(posx, posy+30), 25, g_color, -1);  // lower circle
       
//         } else if (it->getType() == ObjectType::objectVehicle) {
//             cv::RNG rng(it->getID());
//             int b = rng.uniform(0, 255);
//             int g = rng.uniform(50, 200);  // avoid same color as traffic light (thus not 0-255)
//             int r = sqrt(255*255 - g*g - r*r); // ensure that length of color vector is always 255
//             cv::Scalar vehicleColor = cv::Scalar(b,g,r);
//             cv::circle(_images.at(1), cv::Point2d(posx, posy), 50, vehicleColor, -1);
//         }
//     }

//     float opacity = 0.85;
//     cv::addWeighted(_images.at(1), opacity, _images.at(0), 1.0 - opacity, 0, _images.at(2));

//     // display background and overlay image
//     cv::imshow(_windowName, _images.at(2));
//     cv::waitKey(33);
// }
