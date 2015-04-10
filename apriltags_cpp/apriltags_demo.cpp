// C++ standard library files
#include <iostream>
#include <cstring>
#include <cmath>
#include <vector>
#include <list>
#include <ctime>
// OpenCV library files
#include <opencv2/opencv.hpp>
// Eigen library files
#include <Eigen/Core>
// AprilTags library files
#include <apriltags/TagDetector.h>
#include <apriltags/TagDetection.h>
#include <apriltags/Tag16h5.h>
#include <apriltags/Tag25h7.h>
#include <apriltags/Tag25h9.h>
#include <apriltags/Tag36h9.h>
#include <apriltags/Tag36h11.h>

class AprilTagDetector
{
public:
  AprilTagDetector() :
    m_tagCode(AprilTags::tagCodes36h11),
    m_debug(true),
    m_rectify(false),
    m_detector(new AprilTags::TagDetector(AprilTags::tagCodes36h11))
  {
    init_undistortMap();
  }

  ~AprilTagDetector() {}

  void init_undistortMap()
  {
    m_cameraMatrix = cv::Mat::zeros(3,3,CV_32FC1);
    m_cameraMatrix.at<float>(0,0) = 1097;
    m_cameraMatrix.at<float>(1,1) = m_cameraMatrix.at<float>(0,0);
    m_cameraMatrix.at<float>(0,2) = 643.5;
    m_cameraMatrix.at<float>(1,2) = 481.5;
    m_cameraMatrix.at<float>(2,2) = 1;
    std::cout << m_cameraMatrix << std::endl;

    m_distCoeff = cv::Mat::zeros(1,5, CV_32FC1);
    m_distCoeff.at<float>(0,0) = -0.2458;
    m_distCoeff.at<float>(0,1) = 0.2232;
    m_distCoeff.at<float>(0,2) = -0.0029;
    m_distCoeff.at<float>(0,3) = -0.00046;
    m_distCoeff.at<float>(0,4) = 0.000000;

    cv::initUndistortRectifyMap(m_cameraMatrix, m_distCoeff, cv::Mat(), m_cameraMatrix, cv::Size(1288, 964), CV_32FC1, m_map1, m_map2);
  }

  double standardRad(double t)
  {
    if (t >= 0.) {
      t = std::fmod(t+M_PI, 2*M_PI) - M_PI;
    } else {
      t = std::fmod(t-M_PI, -2*M_PI) + M_PI;
    }
    return t;
  }

  void wRo_to_euler(const Eigen::Matrix3d& wRo, double& yaw, double& pitch, double& roll)
  {
    yaw = standardRad(std::atan2(wRo(1,0), wRo(0,0)));
    double c = std::cos(yaw);
    double s = std::sin(yaw);
    pitch = standardRad(std::atan2(-wRo(2,0), wRo(0,0)*c + wRo(1,0)*s));
    roll  = standardRad(std::atan2(wRo(0,2)*s - wRo(1,2)*c, -wRo(0,1)*s + wRo(1,1)*c));
  }

  void setImage(cv::Mat& image)
  {
    m_image = image.clone();
  }

  void setTagFamily(std::string code)
  {
    if (code == "16h5")
    {
      m_tagCode = AprilTags::tagCodes16h5;
    } else if (code == "25h7")
    {
      m_tagCode = AprilTags::tagCodes25h7;
    } else if (code == "25h9")
    {
      m_tagCode = AprilTags::tagCodes25h9;
    } else if (code == "36h9")
    {
      m_tagCode = AprilTags::tagCodes36h9;
    } else if (code == "36h11")
    {
      m_tagCode = AprilTags::tagCodes36h11;
    } else
    {
      std::cout << "Invalid tag family specified!" << std::endl;
      exit(1);
    }
    m_detector = new AprilTags::TagDetector(m_tagCode);
  }

  void processImage()
  {
    cv::Mat image_gray;
    if(m_rectify)
    {
      cv::Mat rectImage;
      cv::remap(m_image, rectImage, m_map1, m_map2, CV_INTER_LINEAR);
      cv::cvtColor(rectImage, image_gray, CV_BGR2GRAY);
    }
    else
    {
      cv::cvtColor(m_image, image_gray, CV_BGR2GRAY);
    }
    m_detections = m_detector->extractTags(image_gray);
  }

  void printDetections()
  {
    if(m_detections.empty())
    {
      std::cout << "  No tags found!" << std::endl;
    }
    for(std::vector<AprilTags::TagDetection>::iterator it = m_detections.begin(); it != m_detections.end(); ++it)
    {
      std::cout << "  Id: " << (*it).id << " (Hamming: " << (*it).hammingDistance << ")";

      // recovering the relative pose of a tag:

      // NOTE: for this to be accurate, it is necessary to use the
      // actual camera parameters here as well as the actual tag size
      // (m_fx, m_fy, m_px, m_py, m_tagSize)

      Eigen::Vector3d translation;
      Eigen::Matrix3d rotation;
      (*it).getRelativeTranslationRotation(0.15, m_cameraMatrix.at<float>(0,0), m_cameraMatrix.at<float>(1,1), m_cameraMatrix.at<float>(0,2),
                                           m_cameraMatrix.at<float>(1,2), translation, rotation);
      Eigen::Matrix3d F;
      F << 1,  0,  0,
           0, -1,  0,
           0,  0,  1;
      Eigen::Matrix3d fixed_rot = F*rotation;
      double yaw, pitch, roll;
      wRo_to_euler(fixed_rot, yaw, pitch, roll);

      fprintf(stdout, " distance=%8.4fm, x=%8.4f,y=%8.4f,z=%8.4f,yaw=%8.4f,pitch=%8.4f,roll=%8.4f\n",
               translation.norm(), translation(0), translation(1), translation(2), yaw, pitch, roll);
    }
  }

  void drawDetections()
  {
    for(std::vector<AprilTags::TagDetection>::iterator it = m_detections.begin(); it != m_detections.end(); ++it)
    {
      (*it).draw(m_image);
    }
    if(m_debug)
    {
      cv::imshow("Detected Apriltags", m_image);
    }
  }

private:
  AprilTags::TagDetector* m_detector;
  AprilTags::TagCodes m_tagCode;
  std::vector<AprilTags::TagDetection> m_detections;
  cv::Mat m_cameraMatrix;
  cv::Mat m_distCoeff;
  cv::Mat m_map1, m_map2;
  cv::Mat m_image;
  bool m_debug, m_rectify;
};

int main(int argc, char** argv)
{
  cv::VideoCapture capture(CV_CAP_OPENNI);
  if(!capture.isOpened())
  {
    std::cerr << "ERROR: Can't find video device!" << std::endl;
    exit(1);
  }
  capture.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_SXGA_30HZ);
  capture.grab();

  AprilTagDetector detector;
  float tic, toc, fps;
  // capture loop
  char key = 0;
  while(key != 'q')
  {
    tic = cv::getTickCount();
    // Get the image
    if(capture.grab())
    {
      cv::Mat rawImage;
      capture.retrieve(rawImage, CV_CAP_OPENNI_BGR_IMAGE);
      // convert to rgb

      detector.setTagFamily("36h11");
      detector.setImage(rawImage);
      detector.processImage();
      detector.printDetections();
      detector.drawDetections();

      toc = cv::getTickCount();
      fps = cv::getTickFrequency()/(toc - tic);
      fprintf(stdout, "fps: %8.2f \n", fps);

      key = cv::waitKey(1);
    }
  }
//  capture.stop();
  return EXIT_SUCCESS;
}
