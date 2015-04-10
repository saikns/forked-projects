#ifndef TAGDETECTOR_H
#define TAGDETECTOR_H

#include <vector>

#include <opencv2/opencv.hpp>

#include <apriltags/TagDetection.h>
#include <apriltags/TagFamily.h>
#include <apriltags/FloatImage.h>

namespace AprilTags {

class TagDetector {
public:
	
	const TagFamily thisTagFamily;

	//! Constructor
  // note: TagFamily is instantiated here from TagCodes
	TagDetector(const TagCodes& tagCodes) : thisTagFamily(tagCodes) {}
	
	std::vector<TagDetection> extractTags(const cv::Mat& image);
	
};

} // namespace

#endif
