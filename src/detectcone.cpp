/**
* Copyright (C) 2017 Chalmers Revere
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
* USA.
*/
#include "detectcone.hpp"

DetectCone::DetectCone(std::map<std::string, std::string> commandlineArguments, cluon::OD4Session& od4) :
  m_od4(od4)
, m_lastLidarData()
, m_lastCameraData()
, m_pointMatched()
, m_diffVec()
, m_finalPointCloud()
, m_threshold()
, m_timeDiffMilliseconds()
, m_lastTimeStamp()
, m_coneCollector()
, m_lastObjectId()
, m_newFrame(true)
, m_coneMutex()
, m_recievedFirstImg(false)
, m_img()
, m_slidingWindow()
, m_efficientSlidingWindow()
, m_lidarIsWorking(false)
, m_checkLiarMilliseconds()
, m_count()
, m_patchSize()
, m_width()
, m_height()
{
  m_diffVec = 0;
  m_pointMatched = Eigen::MatrixXd::Zero(4,1);
  m_lastCameraData = Eigen::MatrixXd::Zero(4,1);
  m_lastLidarData = Eigen::MatrixXd::Zero(4,1);
  m_coneCollector = Eigen::MatrixXd::Zero(4,200);
  m_lastObjectId = 0;

  setUp(commandlineArguments);
}

DetectCone::~DetectCone()
{
  m_img.release();
  cv::destroyAllWindows();
}

void DetectCone::setUp(std::map<std::string, std::string> commandlineArguments)
{
  m_threshold = std::stod(commandlineArguments["threshold"]);
  m_timeDiffMilliseconds = std::stol(commandlineArguments["timeDiffMilliseconds"]);;
  m_checkLiarMilliseconds = std::stol(commandlineArguments["checkLidarMilliseconds"]);
  m_senderStamp = std::stoul(commandlineArguments["senderStamp"]);
  m_attentionSenderStamp = std::stoul(commandlineArguments["attentionSenderStamp"]);

  slidingWindow("slidingWindow");
}

void DetectCone::nextContainer(cluon::data::Envelope data)
{
  bool correctSenderStamp = static_cast<uint32_t>(data.sampleTimeStamp().microseconds()) == m_attentionSenderStamp;
  if (data.dataType() == opendlv::logic::perception::ObjectDirection::ID() && correctSenderStamp) {
    std::cout << "Recieved Direction" << std::endl;
    m_lastTimeStamp = data.sampleTimeStamp();
    auto coneDirection = cluon::extractMessage<opendlv::logic::perception::ObjectDirection>(std::move(data));
    uint32_t objectId = coneDirection.objectId();
    bool newFrameDist = false;
    {
      std::unique_lock<std::mutex> lockCone(m_coneMutex);
      m_coneCollector(0,objectId) = -coneDirection.azimuthAngle();  //Negative for conversion from car to LIDAR frame
      m_coneCollector(1,objectId) = coneDirection.zenithAngle();
      m_lastObjectId = (m_lastObjectId<objectId)?(objectId):(m_lastObjectId);
      newFrameDist = m_newFrame;
      m_newFrame = false;
    }
    //Check last timestamp if they are from same message
    //std::cout << "Message Recieved " << std::endl;
    if (newFrameDist){
       std::thread coneCollector(&DetectCone::initializeCollection, this);
       coneCollector.detach();
       initializeCollection();
    }
  }

  else if(data.dataType() == opendlv::logic::perception::ObjectDistance::ID() && correctSenderStamp){
    std::cout << "Recieved Distance" << std::endl;
    m_lastTimeStamp = data.sampleTimeStamp();;
    auto coneDistance = cluon::extractMessage<opendlv::logic::perception::ObjectDistance>(std::move(data));
    uint32_t objectId = coneDistance.objectId();
    bool newFrameDist = false;
    {
      std::unique_lock<std::mutex> lockCone(m_coneMutex);
      m_coneCollector(2,objectId) = coneDistance.distance();
      m_coneCollector(3,objectId) = 0;
      m_lastObjectId = (m_lastObjectId<objectId)?(objectId):(m_lastObjectId);
      newFrameDist = m_newFrame;
      m_newFrame = false;
    }
    //Check last timestamp if they are from same message
    //std::cout << "Message Recieved " << std::endl;
    if (newFrameDist){
       std::thread coneCollector(&DetectCone::initializeCollection, this);
       coneCollector.detach();
       initializeCollection();
    }
  }
}

void DetectCone::blockMatching(cv::Mat& disp, cv::Mat imgL, cv::Mat imgR){
  cv::Mat grayL, grayR, dispL, dispR;

  cv::cvtColor(imgL, grayL, 6);
  cv::cvtColor(imgR, grayR, 6);

  cv::Ptr<cv::StereoBM> sbmL = cv::StereoBM::create(); 
  sbmL->setBlockSize(21);
  sbmL->setNumDisparities(32);
  sbmL->compute(grayL, grayR, dispL);

  // auto wls_filter = cv::ximgproc::createDisparityWLSFilter(sbmL);
  // cv::Ptr<cv::StereoMatcher> sbmR = cv::ximgproc::createRightMatcher(sbmL);
  // sbmR->compute(grayR, grayL, dispR);
  // wls_filter->setLambda(8000);
  // wls_filter->setSigmaColor(0.8);
  // wls_filter->filter(dispL, imgL, dispL, dispR);
  disp = dispL/16;

  // cv::Mat disp8;
  // cv::normalize(dispL, disp, 0, 255, 32, CV_8U);
  // cv::namedWindow("disp", cv::WINDOW_AUTOSIZE);
  // cv::imshow("disp", imgL+imgR);
  // cv::waitKey(10);
}

void DetectCone::reconstruction(cv::Mat img, cv::Mat& Q, cv::Mat& disp, cv::Mat& rectified, cv::Mat& XYZ){
  // cv::Mat mtxLeft = (cv::Mat_<double>(3, 3) <<
  //   350.6847, 0, 332.4661,
  //   0, 350.0606, 163.7461,
  //   0, 0, 1);
  // cv::Mat distLeft = (cv::Mat_<double>(5, 1) << -0.1674, 0.0158, 0.0057, 0, 0);
  // cv::Mat mtxRight = (cv::Mat_<double>(3, 3) <<
  //   351.9498, 0, 329.4456,
  //   0, 351.0426, 179.0179,
  //   0, 0, 1);
  // cv::Mat distRight = (cv::Mat_<double>(5, 1) << -0.1700, 0.0185, 0.0048, 0, 0);
  // cv::Mat R = (cv::Mat_<double>(3, 3) <<
  //   0.9997, 0.0015, 0.0215,
  //   -0.0015, 1, -0.00008,
  //   -0.0215, 0.00004, 0.9997);
  // cv::Mat T = (cv::Mat_<double>(3, 1) << -0.1191807, 0.0001532, 0.0011225);
  // cv::Size stdSize = cv::Size(m_width, m_height);

  //official
  cv::Mat mtxLeft = (cv::Mat_<double>(3, 3) <<
    349.891, 0, 334.352,
    0, 349.891, 187.937,
    0, 0, 1);
  cv::Mat distLeft = (cv::Mat_<double>(5, 1) << -0.173042, 0.0258831, 0, 0, 0);
  cv::Mat mtxRight = (cv::Mat_<double>(3, 3) <<
    350.112, 0, 345.88,
    0, 350.112, 189.891,
    0, 0, 1);
  cv::Mat distRight = (cv::Mat_<double>(5, 1) << -0.174209, 0.026726, 0, 0, 0);
  cv::Mat rodrigues = (cv::Mat_<double>(3, 1) << -0.0132397, 0.021005, -0.00121284);
  cv::Mat R;
  cv::Rodrigues(rodrigues, R);
  cv::Mat T = (cv::Mat_<double>(3, 1) << -0.12, 0, 0);
  cv::Size stdSize = cv::Size(m_width, m_height);

  int width = img.cols;
  int height = img.rows;
  cv::Mat imgL(img, cv::Rect(0, 0, width/2, height));
  cv::Mat imgR(img, cv::Rect(width/2, 0, width/2, height));

  // cv::resize(imgL, imgL, stdSize);
  // cv::resize(imgR, imgR, stdSize);

  //std::cout << imgR.size() <<std::endl;

  cv::Mat R1, R2, P1, P2;
  cv::Rect validRoI[2];
  cv::stereoRectify(mtxLeft, distLeft, mtxRight, distRight, stdSize, R, T, R1, R2, P1, P2, Q,
    cv::CALIB_ZERO_DISPARITY, 0.0, stdSize,& validRoI[0],& validRoI[1]);

  cv::Mat rmap[2][2];
  cv::initUndistortRectifyMap(mtxLeft, distLeft, R1, P1, stdSize, CV_16SC2, rmap[0][0], rmap[0][1]);
  cv::initUndistortRectifyMap(mtxRight, distRight, R2, P2, stdSize, CV_16SC2, rmap[1][0], rmap[1][1]);
  cv::remap(imgL, imgL, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
  cv::remap(imgR, imgR, rmap[1][0], rmap[1][1], cv::INTER_LINEAR);

  // //check whether the camera is facing forward
  // cv::Mat rectify = imgL+imgR;
  // cv::line(rectify, cv::Point(336,0), cv::Point(336,378),cv::Scalar(0,0,0),1);
  // cv::namedWindow("rectified", cv::WINDOW_NORMAL);
  // cv::imshow("rectified", rectify);
  // cv::waitKey(0);

  // cv::imwrite("tmp/imgL.png", imgL);
  // cv::imwrite("tmp/imgR.png", imgR);
  // return;

  blockMatching(disp, imgL, imgR);

  // cv::namedWindow("disp", cv::WINDOW_NORMAL);
  // cv::imshow("disp", imgL+imgR);
  // cv::waitKey(0);
  // cv::waitKey(0);

  rectified = imgL;

  cv::reprojectImageTo3D(disp, XYZ, Q);
}

void DetectCone::convertImage(cv::Mat img, int w, int h, tiny_dnn::vec_t& data){
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(w, h));
  data.resize(w * h * 3);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
       data[c * w * h + y * w + x] =
         float(resized.at<cv::Vec3b>(y, x)[c] / 255.0);
      }
    }
  }
}

void DetectCone::slidingWindow(const std::string& dictionary) {
  using conv    = tiny_dnn::convolutional_layer;
  using fc      = tiny_dnn::fully_connected_layer;
  using tanh    = tiny_dnn::tanh_layer;
  using relu    = tiny_dnn::relu_layer;
  using softmax = tiny_dnn::softmax_layer;

  tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

  m_slidingWindow << conv(64, 64, 4, 3, 16, tiny_dnn::padding::valid, true, 2, 2, backend_type) << tanh()                                                   
     << conv(31, 31, 3, 16, 16, tiny_dnn::padding::valid, true, 2, 2, backend_type) << tanh() 
     // << dropout(15*15*16, 0.25)
     << conv(15, 15, 3, 16, 32, tiny_dnn::padding::valid, true, 2, 2, backend_type) << tanh() 
     << conv(7, 7, 3, 32, 32, tiny_dnn::padding::valid, true, 2, 2, backend_type) << tanh() 
     // << dropout(3*3*32, 0.25)                     
     << fc(3 * 3 * 32, 128, true, backend_type) << relu()  
     << fc(128, 5, true, backend_type) << softmax(5); 

  // load nets
  std::ifstream ifs(dictionary.c_str());
  ifs >> m_slidingWindow;
}

std::vector <cv::Point> DetectCone::imRegionalMax(cv::Mat input, int nLocMax, double threshold, int minDistBtwLocMax)
{
    cv::Mat scratch = input.clone();
    // std::cout<<scratch<<std::endl;
    // cv::GaussianBlur(scratch, scratch, cv::Size(3,3), 0, 0);
    std::vector <cv::Point> locations(0);
    locations.reserve(nLocMax); // Reserve place for fast access
    for (int i = 0; i < nLocMax; i++) {
        cv::Point location;
        double maxVal;
        cv::minMaxLoc(scratch, NULL,& maxVal, NULL,& location);
        if (maxVal > threshold) {
            int col = location.x;
            int row = location.y;
            locations.push_back(cv::Point(col, row));
            int r0 = (row-minDistBtwLocMax > -1 ? row-minDistBtwLocMax : 0);
            int r1 = (row+minDistBtwLocMax < scratch.rows ? row+minDistBtwLocMax : scratch.rows-1);
            int c0 = (col-minDistBtwLocMax > -1 ? col-minDistBtwLocMax : 0);
            int c1 = (col+minDistBtwLocMax < scratch.cols ? col+minDistBtwLocMax : scratch.cols-1);
            for (int r = r0; r <= r1; r++) {
                for (int c = c0; c <= c1; c++) {
                    if (sqrt((r-row)*(r-row)+(c-col)*(c-col)) <= minDistBtwLocMax) {
                      scratch.at<double>(r,c) = 0.0;
                    }
                }
            }
        } else {
            break;
        }
    }
    return locations;
}


float DetectCone::median(std::vector<float> vec) {
  int size = vec.size();
  float tvecan;
  if (size % 2 == 0) { // even
    tvecan = (vec[vec.size() / 2 - 1] + vec[vec.size() / 2]) / 2;
  }

  else //odd
    tvecan = vec[vec.size() / 2];
  return tvecan;
}

float DetectCone::mean(std::vector<float> vec) {
  float result = 0;
  size_t size = vec.size();
  for(size_t i = 0; i < size; i++){
    result += vec[i];
  }
  result /= size;
  return result;
}

void DetectCone::gather_points(//初始化
  cv::Mat source,
  std::vector<float> vecQuery,
  std::vector<int>& vecIndex,
  std::vector<float>& vecDist
  )
{  
  double radius = 1;
  unsigned int max_neighbours = 100;
  cv::flann::KDTreeIndexParams indexParams(2);
  cv::flann::Index kdtree(source, indexParams); //此部分建立kd-tree索引同上例，故不做详细叙述
  cv::flann::SearchParams params(1024);//设置knnSearch搜索参数
  kdtree.radiusSearch(vecQuery, vecIndex, vecDist, radius, max_neighbours, params);
}

void DetectCone::filterKeypoints(std::vector<cv::Point3f>& point3Ds){
  std::vector<Pt> data;
  std::vector<cv::Point2f> data_tmp;
  
  for(size_t i = 0; i < point3Ds.size(); i++){
    if(point3Ds[i].y > 0.5 && point3Ds[i].y < 2){
      cv::Point2d pt(point3Ds[i].x, point3Ds[i].z);
      data_tmp.push_back(pt);
      data.push_back(Pt(pt,-1));
    }
  }
  point3Ds.clear();

  if(data.size() == 0)
    return;

  cv::Mat source = cv::Mat(data_tmp).reshape(1);
 
  int resultSize = 1000;
  float resultResize = 50;
  cv::RNG rng(time(0));
  cv::Mat result = cv::Mat::zeros(resultSize, resultSize, CV_8UC3);
  cv::Point2f point2D;
  int groupId = 0;
  

  for(size_t j = 0; j < data.size()-1; j++)
  {   
    if(data[j].group == -1){
      std::vector<float> vecQuery;//存放 查询点 的容器（本例都是vector类型）
      vecQuery.push_back(data[j].pt.x);
      vecQuery.push_back(data[j].pt.y);
      std::vector<int> vecIndex;
      std::vector<float> vecDist;

      gather_points(source, vecQuery, vecIndex, vecDist);//kd tree finish; find the points in the circle with point center vecQuery and radius, return index in vecIndex
      int num = 0;
      for(size_t i = 0; i < vecIndex.size(); i++){
        if(vecIndex[i]!=0)
          num++;
      }
      for (size_t i = 1; i < vecIndex.size(); i++){
        if (vecIndex[i] == 0 && vecIndex[i+1] != 0){
          num++;
        }
      }
      if (num == 0){
        if (data[j].group == -1){ 
          data[j].group = groupId++;
          point2D = data[j].pt;
          // std::cout<<j<<" type 1"<<" "<<data[j].pt.x<<","<<data[j].pt.y<<" group "<<data[j].group<<std::endl;
        }
      }
      else{   
        std::vector<Pt> groupAll;
        std::vector<int> filteredIndex;
        std::vector<float> centerPointX;
        std::vector<float> centerPointY;
        for (int v = 0; v < num; v++){
          groupAll.push_back(data[vecIndex[v]]);
          filteredIndex.push_back(vecIndex[v]);
        }
      
        int noGroup = 0;
        for(size_t i = 0; i < groupAll.size(); i++){
          if(groupAll[i].group == -1)
            noGroup++;
        }

        if (noGroup > 0){
          for (size_t k = 0; k < filteredIndex.size(); k++)
          { 
            if (data[filteredIndex[k]].group == -1)
            { 
              data[filteredIndex[k]].group = groupId;
              centerPointX.push_back(data[vecIndex[k]].pt.x);
              centerPointY.push_back(data[vecIndex[k]].pt.y);

              int X1 = int(data[filteredIndex[k]].pt.x*resultResize+resultSize/2);
              int Y1 = int(data[filteredIndex[k]].pt.y*resultResize);
              // std::cout<<k<<" type 2"<<" "<<data[vecIndex[k]].pt.x<<","<<data[vecIndex[k]].pt.y<<" group "<< data[vecIndex[k]].group<<std::endl;
              cv::circle(result, cv::Point(X1,Y1), 5, cv::Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)), -1);
            }
          }
          groupId++;
          point2D.x = mean(centerPointX);
          point2D.y = mean(centerPointY);
        }
        else{
          data[j].group = data[vecIndex[0]].group;
          point2D = data[j].pt;
          // std::cout<<j<<" type 2"<<" "<<data[j].pt.x<<","<<data[j].pt.y<<" group "<<data[j].group<<std::endl;
        }
      }
      point3Ds.push_back(cv::Point3f(point2D.x, 1, point2D.y));

      int X1 = int(point2D.x*resultResize+resultSize/2);
      int Y1 = int(point2D.y*resultResize);
      cv::circle(result, cv::Point(X1,Y1), 8, cv::Scalar(0, 255, 255), -1);
    }
  }

  // for (int p = 0; p < data.size(); p++){
  //   std::cout<<p<<" "<<data[p].pt.x<<" "<<data[p].pt.y<<" group "<<data[p].group<<std::endl;
  // }

  // for (int r = 0; r < point3Ds.size(); r++){
  //   std::cout<<"NO."<<r<<" "<<point3Ds[r].x<<","<<point3Ds[r].z<<std::endl;
  // }

  // cv::flip(result, result, 0);
  // cv::namedWindow("result", cv::WINDOW_NORMAL);
  // cv::imshow("result", result);
  // cv::waitKey(0);
}

// void DetectCone::xyz2xy(cv::Mat Q, cv::Point3f xyz, cv::Point2f& xy, int& radius){
//   double X = xyz.x;
//   double Y = xyz.y;
//   double Z = xyz.z;
//   double Cx = -Q.at<double>(0,3);
//   double Cy = -Q.at<double>(1,3);
//   double f = Q.at<double>(2,3);
//   double a = Q.at<double>(3,2);
//   double b = Q.at<double>(3,3);
//   double d = (f - Z * b ) / ( Z * a);
//   xy.x = X * ( d * a + b ) + Cx;
//   xy.y = Y * ( d * a + b ) + Cy;
//   radius = int(0.4 * ( d * a + b ));
// }

// void DetectCone::forwardDetectionORB(cv::Mat img){
//   //Given RoI by SIFT detector and detected by CNN
//   double threshold = 0.1;

//   std::vector<tiny_dnn::tensor_t> inputs;
//   std::vector<int> verifiedIndex;
//   std::vector<cv::Point> candidates;
//   // std::vector<int> outputs;

//   cv::Mat Q, disp, XYZ, imgRoI, imgSource;
//   reconstruction(img, Q, disp, img, XYZ);
//   img.copyTo(imgSource);

//   // int rowT = 160;
//   // int rowB = 290;
//   int rowT = 190;
//   int rowB = 320;
//   // int rowT = 180;
//   // int rowB = 376;
//   imgRoI = img.rowRange(rowT, rowB);

//   cv::Ptr<cv::ORB> detector = cv::ORB::create();
//   std::vector<cv::KeyPoint> keypoints;
//   detector->detect(imgRoI, keypoints);

//   // cv::Mat Match;
//   // cv::drawKeypoints(gray, keypoints, Match);
//   // cv::namedWindow("Match", cv::WINDOW_NORMAL);
//   // cv::imshow("Match", Match);
//   // cv::waitKey(0);

//   // cv::resize(img, img, cv::Size(m_width/2, m_height/2));
//   // cv::Mat probMap = cv::Mat::zeros(m_height/2, m_width/2, CV_64F);
//   // cv::Mat indexMap = cv::Mat::zeros(m_height/2, m_width/2, CV_32S);
//   cv::Mat probMap = cv::Mat::zeros(m_height, m_width, CV_64F);
//   cv::Mat indexMap = cv::Mat::zeros(m_height, m_width, CV_32S);

//   std::vector<cv::Point3f> point3Ds;
//   cv::Point2f point2D;
//   for(size_t i = 0; i < keypoints.size(); i++){
//     cv::Point position(keypoints[i].pt.x, keypoints[i].pt.y+rowT);
//     cv::Point3f point3D = XYZ.at<cv::Point3f>(position);
//     if(point3D.y>0.8 && point3D.y<1.1){
//       point3Ds.push_back(point3D);
//     }
//     // std::cout << cv::Point3f(XYZ.at<cv::Point3f>(position)) << std::endl;
//   }
//   filterKeypoints(point3Ds);
//   for(size_t i = 0; i < point3Ds.size(); i++){
//     int radius;
//     xyz2xy(Q, point3Ds[i], point2D, radius);
//     int x = point2D.x;
//     int y = point2D.y;

//     // float_t ratio = depth2resizeRate(point3Ds[i].x, point3Ds[i].z);
//     // int length = ratio * 25;
//     // int radius = (length-1)/2;
//     // radius = 12;

//     cv::Rect roi;
//     roi.x = std::max(x - radius, 0);
//     roi.y = std::max(y - radius, 0);
//     roi.m_width = std::min(x + radius, img.cols) - roi.x;
//     roi.m_height = std::min(y + radius, img.rows) - roi.y;

//     //cv::circle(img, cv::Point (x,y), radius, cv::Scalar (0,0,0));
//     // // cv::circle(disp, cv::Point (x,y), 3, 0, CV_FILLED);
//     // cv::namedWindow("roi", cv::WINDOW_NORMAL);
//     // cv::imshow("roi", img_hsv);
//     // cv::waitKey(0);
//     //cv::destroyAllWindows();

//     if (0 > roi.x || 0 > roi.m_width || roi.x + roi.m_width > img.cols || 0 > roi.y || 0 > roi.m_height || roi.y + roi.m_height > img.rows){
//       std::cout << "Wrong roi!" << std::endl;
//       // outputs.push_back(-1);
//     }
//     else{
//       auto patchImg = img(roi);
//       // cv::namedWindow("roi", cv::WINDOW_NORMAL);
//       // cv::imshow("roi", patchImg);
//       // cv::waitKey(0);
//       tiny_dnn::vec_t data;
//       convertImage(patchImg, m_patchSize, m_patchSize, data);
//       inputs.push_back({data});
//       // outputs.push_back(0);
//       verifiedIndex.push_back(i);
//       candidates.push_back(cv::Point(x,y));
//     }
//   }
  
//   int index, index2;
//   std::string filename, savePath;
//   index = imgPath.find_last_of('/');
//   filename = imgPath.substr(index+1);
//   index2 = filename.find_last_of('.');
//   std::ofstream savefile;
//   savePath = imgPath.substr(0,index-7)+"/results/"+filename.substr(0,index2)+".csv";
//   savefile.open(savePath);

//   int resultm_width = m_height;
//   int resultm_height = m_height;
//   double resultResize = 20;
//   cv::Mat result = cv::Mat::zeros(resultm_height, resultm_width, CV_8UC3);
//   std::string labels[] = {"background", "blue", "yellow", "orange", "big orange"};

//   if(inputs.size()>0){
//     auto prob = m_slidingWindow.predict(inputs);
//     for(size_t i = 0; i < inputs.size(); i++){
//       size_t maxIndex = 0;
//       double maxProb = prob[i][0][0];
//       for(size_t j = 1; j < 5; j++){
//         if(prob[i][0][j] > maxProb){
//           maxIndex = j;
//           maxProb = prob[i][0][j];
//         }
//       }
//       // outputs[verifiedIndex[i]] = maxIndex;
//       int x = candidates[i].x;
//       int y = candidates[i].y;
//       probMap.at<double>(y,x) = maxProb;
//       indexMap.at<int>(y,x) = maxIndex;
//     }
//     std::vector <cv::Point> cones = imRegionalMax(probMap, 10, threshold, 10);

//     for(size_t i = 0; i < cones.size(); i++){
//       int x = cones[i].x;
//       int y = cones[i].y;
//       double maxProb = probMap.at<double>(y,x);
//       int maxIndex = indexMap.at<int>(y,x);
//       cv::Point position(x, y);
//       cv::Point3f point3D = XYZ.at<cv::Point3f>(position);
//       std::string labelName = labels[maxIndex];
//       // float_t ratio = depth2resizeRate(point3D.x, point3D.z);
//       // int length = ratio * 25;
//       // int radius = (length-1)/2;
//       int radius;
//       cv::Point2f position_tmp;
//       xyz2xy(Q, point3D, position_tmp, radius);

//       if(radius>0){
//         if (labelName == "background"){
//           std::cout << "No cone detected" << std::endl;
//           cv::circle(img, position, radius, cv::Scalar (0,0,0));
//         } 
//         else{
//           if (labelName == "blue")
//             cv::circle(img, position, radius, cv::Scalar (175,238,238));
//           else if (labelName == "yellow")
//             cv::circle(img, position, radius, cv::Scalar (0,255,255));
//           else if (labelName == "orange")
//             cv::circle(img, position, radius, cv::Scalar (0,165,255));
//           else if (labelName == "big orange")
//             cv::circle(img, position, radius, cv::Scalar (0,0,255));

//           int xt = int(point3D.x * float(resultResize) + resultm_width/2);
//           int yt = int(point3D.z * float(resultResize));
//           if (xt >= 0 && xt <= resultm_width && yt >= 0 && yt <= resultm_height){
//             if (labelName == "blue")
//               cv::circle(result, cv::Point (xt,yt), 5, cv::Scalar (255,0,0), -1);
//             else if (labelName == "yellow")
//               cv::circle(result, cv::Point (xt,yt), 5, cv::Scalar (0,255,255), -1);
//             else if (labelName == "orange")
//               cv::circle(result, cv::Point (xt,yt), 5, cv::Scalar (0,165,255), -1);
//             else if (labelName == "big orange")
//               cv::circle(result, cv::Point (xt,yt), 10, cv::Scalar (0,0,255), -1);
//           }

//           std::cout << position << " " << labelName << " " << point3D << " " << maxProb << std::endl;
//           savefile << std::to_string(position.x)+","+std::to_string(position.y)+","+labelName+","+std::to_string(point3D.x)+","+std::to_string(point3D.y)+","+std::to_string(point3D.z)+"\n";
//         }
//       }
//     }
//   }
      

//   for(size_t i = 0; i < keypoints.size(); i++){
//     cv::circle(img, cv::Point(keypoints[i].pt.x,keypoints[i].pt.y+rowT), 2, cv::Scalar (255,255,255), -1);
//   }

//   cv::line(img, cv::Point(0,rowT), cv::Point(m_width,rowT), cv::Scalar(0,0,255), 2);
//   cv::line(img, cv::Point(0,rowB), cv::Point(m_width,rowB), cv::Scalar(0,0,255), 2);

//   // int resultm_width = 672;
//   // int resultm_height = 600;
//   // double resultResize = 30;
//   // cv::Mat result[2] = cv::Mat::zeros(resultm_height, resultm_width, CV_8UC3), coResult;
//   // std::string labels[] = {"background", "blue", "yellow", "orange", "big orange"};
//   // if(inputs.size()>0){
//   //   auto prob = m_slidingWindow.predict(inputs);
//   //   for(size_t i = 0; i < inputs.size(); i++){
//   //     size_t maxIndex = 1;
//   //     double maxProb = prob[i][0][1];
//   //     for(size_t j = 2; j < 5; j++){
//   //       if(prob[i][0][j] > maxProb){
//   //         maxIndex = j;
//   //         maxProb = prob[i][0][j];
//   //       }
//   //     }
//   //     // outputs[verifiedIndex[i]] = maxIndex;
//   //     int x = candidates[i].x;
//   //     int y = candidates[i].y;
//   //     cv::Point position(x*2, y*2+180);
//   //     cv::Point3f point3D = XYZ.at<cv::Point3f>(position);
//   //     std::string labelName = labels[maxIndex];     

//   //     if (labelName == "background"){
//   //       std::cout << "No cone detected" << std::endl;
//   //       cv::circle(img, position, 2, cv::Scalar (0,0,0), -1);
//   //     } 
//   //     else{
//   //       // std::cout << "Find one " << labelName << " cone"<< std::endl;
//   //       if (labelName == "blue")
//   //         cv::circle(img, position, 2, cv::Scalar (175,238,238), -1);
//   //       else if (labelName == "yellow")
//   //         cv::circle(img, position, 2, cv::Scalar (0,255,255), -1);
//   //       else if (labelName == "orange")
//   //         cv::circle(img, position, 2, cv::Scalar (0,165,255), -1);
//   //       else if (labelName == "big orange")
//   //         cv::circle(img, position, 4, cv::Scalar (0,0,255), -1);

//   //       int xt = int(point3D.x * float(resultResize) + resultm_width/2);
//   //       int yt = int((point3D.z-1.872f) * float(resultResize));
//   //       if (xt >= 0 && xt <= resultm_width && yt >= 0 && yt <= resultm_height){
//   //         if (labelName == "blue")
//   //           cv::circle(result[0], cv::Point (xt,yt), 5, cv::Scalar (255,0,0), -1);
//   //         else if (labelName == "yellow")
//   //           cv::circle(result[0], cv::Point (xt,yt), 5, cv::Scalar (0,255,255), -1);
//   //         else if (labelName == "orange")
//   //           cv::circle(result[0], cv::Point (xt,yt), 5, cv::Scalar (0,165,255), -1);
//   //         else if (labelName == "big orange")
//   //           cv::circle(result[0], cv::Point (xt,yt), 10, cv::Scalar (0,0,255), -1);
//   //       }

//   //       std::cout << position << " " << labelName << " " << point3D << std::endl;
//   //       savefile << std::to_string(position.x)+","+std::to_string(position.y)+","+labelName+","+std::to_string(point3D.x)+","+std::to_string(point3D.y)+","+std::to_string(point3D.z)+"\n";
//   //     }
//   //   }
//   // }

//   // int resultm_width = m_height;
//   // int resultm_height = m_height;
//   // double resultResize = 20;
//   // cv::Mat result = cv::Mat::zeros(resultm_height, resultm_width, CV_8UC3);
//   // for(size_t i = 0; i < m_finalPointCloud.cols(); i++){
//   //   savefile << std::to_string(m_finalPointCloud(0,i))+","+std::to_string(m_finalPointCloud(1,i))+","+std::to_string(m_finalPointCloud(2,i))+"\n";
//   //   int x = int(m_finalPointCloud(0,i) * resultResize + resultm_width/2);
//   //   int y = int(m_finalPointCloud(1,i) * resultResize);
//   //   if (x >= 0 && x <= resultm_width && y >= 0 && y <= resultm_height){
//   //     cv::circle(result[0], cv::Point (x,y), 5, cv::Scalar (255, 255, 255), -1);
//   //   }
//   // }

//   // cv::circle(result[0], cv::Point (int(resultm_width/2),0), 5, cv::Scalar (0, 0, 255), -1);
//   cv::flip(result, result, 0);
//   // img.copyTo(result[1].rowRange(resultm_height-376,resultm_height));
//   // cv::hconcat(result[1], result[0], coResult);

//   // cv::Mat coResult;
//   // cv::hconcat(img, result, coResult);
//   // cv::namedWindow("disp", cv::WINDOW_NORMAL);
//   // cv::imshow("disp", coResult);
//   // cv::waitKey(0);
//   // cv::waitKey(0);

//   cv::imwrite(imgPath.substr(0,index-7)+"/results/"+filename.substr(0,index2)+".png", img);

//   // savePath = imgPath.substr(0,index-7)+"/results/"+filename.substr(0,index2)+".png";
//   // cv::imwrite(savePath, img);

//   // savePath = imgPath.substr(0,index-7)+"/disp_filtered/"+filename.substr(0,index2)+".png";
//   // std::cout<<savePath<<std::endl;
//   // cv::imwrite(savePath, disp);

//   cv::namedWindow("img", cv::WINDOW_NORMAL);
//   cv::setWindowProperty("img", cv::WND_PROP_FULLSCREEN , cv::WINDOW_FULLSCREEN ); 
//   cv::imshow("img", img);
//   // cv::namedWindow("disp", cv::WINDOW_NORMAL);
//   // cv::imshow("disp", disp);
//   cv::waitKey(30);
//   // cv::destroyAllWindows();

//   // for(size_t i = 0; i < pts.size(); i++)
//   //   std::cout << i << ": " << outputs[i] << std::endl;
// }

// void DetectCone::backwardDetection(cv::Mat img, std::vector<cv::Point3f> pts, std::vector<int>& outputs){
//   //Given RoI in 3D world, project back to the camera frame and then detect
//   float_t threshold = 0.7;
//   cv::Mat disp, Q, rectified, XYZ;
//   reconstruction(img, Q, disp, rectified, XYZ);
//   std::vector<tiny_dnn::tensor_t> inputs;
//   std::vector<int> verifiedIndex;
//   std::vector<cv::Vec3i> porperty;
//   outputs.clear();

//   for(size_t i = 0; i < pts.size(); i++){
//     cv::Point2f point2D;
//     xyz2xy(Q, pts[i], point2D);

//     int x = point2D.x;
//     int y = point2D.y;

//     // std::cout << "Camera region center: " << x << ", " << y << std::endl;
//     float_t ratio = depth2resizeRate(pts[i].x, pts[i].z);
//     if (ratio > 0) {
//       int length = ratio * 25;
//       int radius = (length-1)/2;
//       // std::cout << "radius: " << radius << std::endl;

//       cv::Rect roi;
//       roi.x = std::max(x - radius, 0);
//       roi.y = std::max(y - radius, 0);
//       roi.width = std::min(x + radius, rectified.cols) - roi.x;
//       roi.height = std::min(y + radius, rectified.rows) - roi.y;

//       //cv::circle(img, cv::Point (x,y), radius, cv::Scalar (0,0,0));
//       // // cv::circle(disp, cv::Point (x,y), 3, 0, CV_FILLED);
//       //cv::namedWindow("roi", cv::WINDOW_NORMAL);
//       //cv::imshow("roi", img);
//       //cv::waitKey(0);
//       //cv::destroyAllWindows();
//       if (0 > roi.x || 0 > roi.width || roi.x + roi.width > rectified.cols || 0 > roi.y || 0 > roi.height || roi.y + roi.height > rectified.rows){
//         std::cout << "Wrong roi!" << std::endl;
//         outputs.push_back(-1);
//       }
//       else{
//         auto patchImg = rectified(roi);
//         tiny_dnn::vec_t data;
//         convertImage(patchImg, 25, 25, data);
//         inputs.push_back({data});
//         outputs.push_back(0);
//         verifiedIndex.push_back(i);
//         porperty.push_back(cv::Vec3i(x,y,radius));
//       }
//     }
//   }
  
//   if(inputs.size()>0){
//     auto prob = m_slidingWindow.predict(inputs);
//     for(size_t i = 0; i < inputs.size(); i++){
//       size_t maxIndex = 0;
//       float_t maxProb = prob[i][0][0];
//       for(size_t j = 1; j < 5; j++){
//         if(prob[i][0][j] > maxProb){
//           maxIndex = j;
//           maxProb = prob[i][0][j];
//         }
//       }
//       outputs[verifiedIndex[i]] = maxIndex;
//       int x = int(porperty[i][0]);
//       int y = int(porperty[i][1]);
//       float_t radius = porperty[i][2];

//       std::string labels[] = {"blue", "yellow", "orange", "big orange"};
//       if (maxIndex == 0 || maxProb < threshold){
//         std::cout << "No cone detected" << std::endl;
//         cv::circle(rectified, cv::Point (x,y), radius, cv::Scalar (0,0,0));
//       } 
//       else{
//         std::cout << "Find one " << labels[maxIndex-1] << " cone"<< std::endl;
//         if (labels[maxIndex-1] == "blue")
//           cv::circle(rectified, cv::Point (x,y), radius, cv::Scalar (255,0,0));
//         else if (labels[maxIndex-1] == "yellow")
//           cv::circle(rectified, cv::Point (x,y), radius, cv::Scalar (0,255,255));
//         else if (labels[maxIndex-1] == "orange")
//           cv::circle(rectified, cv::Point (x,y), radius, cv::Scalar (0,165,255));
//         else if (labels[maxIndex-1] == "big orange")
//           cv::circle(rectified, cv::Point (x,y), radius*2, cv::Scalar (0,0,255));
//       }
//     }
//   }

//   // cv::namedWindow("disp", cv::WINDOW_NORMAL);
//   // // cv::setWindowProperty("result", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
//   // cv::imshow("disp", rectified);
//   // cv::waitKey(0);
//   // cv::destroyAllWindows();

//   // for(size_t i = 0; i < pts.size(); i++)
//   //   std::cout << i << ": " << outputs[i] << std::endl;
// }

Eigen::MatrixXd DetectCone::Spherical2Cartesian(double azimuth, double zenimuth, double distance)
{
  //double xyDistance = distance * cos(azimuth * static_cast<double>(DEG2RAD));
  double xData = distance * cos(zenimuth * static_cast<double>(DEG2RAD))*sin(azimuth * static_cast<double>(DEG2RAD));
  double yData = distance * cos(zenimuth * static_cast<double>(DEG2RAD))*cos(azimuth * static_cast<double>(DEG2RAD));
  double zData = distance * sin(zenimuth * static_cast<double>(DEG2RAD));
  Eigen::MatrixXd recievedPoint = Eigen::MatrixXd::Zero(4,1);
  recievedPoint << xData,
                   yData,
                   zData,
                    0;
  return recievedPoint;
}

void DetectCone::Cartesian2Spherical(double x, double y, double z, opendlv::logic::sensation::Point& pointInSpherical)
{
  double distance = sqrt(x*x+y*y+z*z);
  double azimuthAngle = atan2(x,y)*static_cast<double>(RAD2DEG);
  double zenithAngle = atan2(z,sqrt(x*x+y*y))*static_cast<double>(RAD2DEG);
  pointInSpherical.distance(float(distance));
  pointInSpherical.azimuthAngle(float(azimuthAngle));
  pointInSpherical.zenithAngle(float(zenithAngle));
}


void DetectCone::initializeCollection(){
  //std::this_thread::sleep_for(std::chrono::duration 1s); //std::chrono::milliseconds(m_timeDiffMilliseconds)

  bool sleep = true;
  auto start = std::chrono::system_clock::now();

  while(sleep)
  {
    auto now = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start);
    if ( elapsed.count() > m_timeDiffMilliseconds*1000 )
        sleep = false;
  }


  Eigen::MatrixXd extractedCones;
  {
    std::unique_lock<std::mutex> lockCone(m_coneMutex);
    std::cout << "FRAME IN LOCK: " << m_newFrame << std::endl;
    extractedCones = m_coneCollector.leftCols(m_lastObjectId+1);
    m_newFrame = true;
    m_lastObjectId = 0;
    m_coneCollector = Eigen::MatrixXd::Zero(4,200);
  }
  //Initialize for next collection
  //std::cout << "Collection done " << extractedCones.cols() << std::endl;
  if(extractedCones.cols() > 0){
    //std::cout << "Extracted Cones " << std::endl;
    //std::cout << extractedCones << std::endl;
    //std::cout << "Extracted Cones " << std::endl;
    //std::cout << extractedCones << std::endl;
    if(m_recievedFirstImg){
      SendCollectedCones(extractedCones);
    }
  }
}

void DetectCone::SendCollectedCones(Eigen::MatrixXd lidarCones)
{
  //Convert to cartesian
  Eigen::MatrixXd cone;
  for(int p = 0; p < lidarCones.cols(); p++){
    cone = Spherical2Cartesian(lidarCones(0,p), lidarCones(1,p), lidarCones(2,p));
    lidarCones.col(p) = cone;
  }
  //std::cout << "lidarCones " << std::endl;
  //std::cout << lidarCones << std::endl;
  m_finalPointCloud = lidarCones;
  double yShift = 0;//1872mm
  double zShift = 0;
  std::vector<cv::Point3f> pts;
  std::vector<int> outputs;

  // int m_width = 640;
  // int m_height = 1000;
  // double resultResize = 50;
  // cv::Mat result[2] = cv::Mat::zeros(m_height, m_width, CV_8UC3), coResult;

  for (int i = 0; i < m_finalPointCloud.cols(); i++){
    pts.push_back(cv::Point3d(m_finalPointCloud(0,i), -zShift-m_finalPointCloud(2,i), yShift+m_finalPointCloud(1,i)));
  //   int x = int(m_finalPointCloud(0,i) * resultResize + m_width/2);
  //   int y = int(m_finalPointCloud(1,i) * resultResize);
  //   if (x >= 0 && x <= m_width && y >= 0 && y <= m_height){
  //     cv::circle(result[0], cv::Point (x,y), 5, cv::Scalar (255, 255, 255), -1);
  //   }
  }

  // cv::circle(result[0], cv::Point (int(m_width/2),0), 5, cv::Scalar (0, 0, 255), -1);
  // cv::flip(result[0], result[0], 0);
  // cv::flip(result[0], result[0], 1);
  // cv::Mat rectified = m_img.colRange(0,1280);
  // cv::resize(rectified, rectified, cv::Size(640, 360));
  // // rectified.convertTo(rectified, CV_8UC3);
  // rectified.copyTo(result[1].rowRange(320,680));
  // // result[1].rowRange(320,680) = rectified;
  // cv::hconcat(result[0], result[1], coResult);

  // cv::imwrite("results/"+std::to_string(m_count++)+".png", coResult);


  // forwardDetectionRoI(m_img, m_slidingWindow);
  // backwardDetection(m_img, pts, outputs);
  // for (int i = 0; i < m_finalPointCloud.cols(); i++){
  //   m_finalPointCloud(3,i) = outputs[i];
  // }

  //std::cout << "matched: " << std::endl;
  //std::cout << m_finalPointCloud << std::endl;
  SendMatchedContainer(m_finalPointCloud);
}

void DetectCone::SendMatchedContainer(Eigen::MatrixXd cones)
{
  opendlv::logic::perception::Object object;
  object.objectId(cones.cols());
  std::chrono::system_clock::time_point tp = std::chrono::system_clock::now();
  cluon::data::TimeStamp sampleTime = cluon::time::convert(tp);
  m_od4.send(object,sampleTime,m_senderStamp);

  for(int n = 0; n < cones.cols(); n++){

    opendlv::logic::sensation::Point conePoint;
    Cartesian2Spherical(cones(0,n), cones(1,n), cones(2,n), conePoint);

    opendlv::logic::perception::ObjectDirection coneDirection;
    coneDirection.objectId(n);
    coneDirection.azimuthAngle(-conePoint.azimuthAngle());  //Negative to convert to car frame from LIDAR
    coneDirection.zenithAngle(conePoint.zenithAngle());
    m_od4.send(coneDirection,sampleTime,m_senderStamp);

    opendlv::logic::perception::ObjectDistance coneDistance;
    coneDistance.objectId(n);
    coneDistance.distance(conePoint.distance());
    m_od4.send(coneDistance,sampleTime,m_senderStamp);

    opendlv::logic::perception::ObjectType coneType;
    coneType.objectId(n);
    coneType.type(uint32_t(cones(3,n)));
    m_od4.send(coneType,sampleTime,m_senderStamp);
  }
}