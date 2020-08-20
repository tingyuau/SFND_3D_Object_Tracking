
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0;
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // calculate euclidean distance for all keypoint matches
    std::vector<double> distances;

    for (auto it = kptMatches.begin(); it != kptMatches.end(); ++it)
    {
        if (boundingBox.roi.contains(kptsCurr[(*it).trainIdx].pt))
        {
            double dist = cv::norm(kptsPrev[(*it).queryIdx].pt - kptsCurr[(*it).trainIdx].pt);
            distances.push_back(dist);
        }
    }

    // calculate mean and sd for filtering outliers
    double sum;
    for (auto it = distances.begin(); it != distances.end(); ++it)
    {
        sum += *it;
    }
    double mean = sum/distances.size();

    double variance;
    for (auto it = distances.begin(); it != distances.end(); ++it)
    {
        variance += std::pow(*it - mean, 2);
    }
    variance = variance/distances.size();
    double stdDeviation = std::sqrt(variance);

    double sdThreshold = 3.0;
    double upperbound = mean + sdThreshold * stdDeviation;
    double lowerbound = mean - sdThreshold * stdDeviation;

    // extract keypoints contained by bbox and filter outliers
    for (auto it = kptMatches.begin(); it != kptMatches.end(); ++it)
    {
        if (boundingBox.roi.contains(kptsCurr[(*it).trainIdx].pt))
        {
            double dist = cv::norm(kptsPrev[(*it).queryIdx].pt - kptsCurr[(*it).trainIdx].pt);

            if (dist > lowerbound && dist < upperbound)
            {
                boundingBox.kptMatches.push_back(*it);
            }
        }
    }

    // debug: count filtered keypoint matches
    // cout << "Total keypoint matches: " << kptMatches.size() << endl;
    // cout << "ROI keypoint matches: " << distances.size() << endl;
    // cout << "Filtered keypoint matches: " << boundingBox.kptMatches.size() << endl;

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    std::vector<double> distRatios; // stores the distance ratios for all keypoints between curr and prev frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt loop

        // get current keypoint and its matched partner in the prev frame
        cv::KeyPoint kptOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kptOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt loop

            double minDist = 100.0; // min required distance

            // get next keypoint and its matched partner in the prev frame
            cv::KeyPoint kptInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kptInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kptOuterCurr.pt - kptInnerCurr.pt);
            double distPrev = cv::norm(kptOuterPrev.pt - kptInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; //compute median dist ratio to remove outliers

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);

    std::cout << "Median dist ratio: " << medDistRatio <<endl;
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // apply 1.5IQR rule to filter lidar points
    std::vector<double> xPrev;

    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        xPrev.push_back(it->x);
    }

    std::sort(xPrev.begin(), xPrev.end());
    long midIndexPrev = floor(xPrev.size() / 2.0);
    double medianPrev = xPrev.size() % 2 == 0 ? (xPrev[midIndexPrev - 1] + xPrev[midIndexPrev]) / 2.0 : xPrev[midIndexPrev];

    std::vector<double> q1Prev = std::vector<double>(xPrev.begin(), xPrev.begin() + midIndexPrev);
    long q1MidIndexPrev = floor(q1Prev.size() / 2.0);
    double q1MedianPrev = q1Prev.size() % 2 == 0 ? (q1Prev[q1MidIndexPrev - 1] + q1Prev[q1MidIndexPrev]) / 2.0 : q1Prev[q1MidIndexPrev];

    std::vector<double> q3Prev = std::vector<double>(xPrev.begin() + midIndexPrev, xPrev.end());
    long q3MidIndexPrev = floor(q3Prev.size() / 2.0);
    double q3MedianPrev = q3Prev.size() % 2 == 0 ? (q3Prev[q3MidIndexPrev - 1] + q3Prev[q3MidIndexPrev]) / 2.0 : q3Prev[q3MidIndexPrev];

    double IQRPrev = q3MedianPrev - q1MedianPrev;
    double hiPrev = q3MedianPrev + 1.5 * IQRPrev;
    double loPrev = q1MedianPrev - 1.5 * IQRPrev;

    std::vector<double> xCurr;

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        xCurr.push_back(it->x);
    }

    std::sort(xCurr.begin(), xCurr.end());
    long midIndexCurr = floor(xCurr.size() / 2.0);
    double medianCurr = xCurr.size() % 2 == 0 ? (xCurr[midIndexCurr - 1] + xCurr[midIndexCurr]) / 2.0 : xCurr[midIndexCurr];

    std::vector<double> q1Curr = std::vector<double>(xCurr.begin(), xCurr.begin() + midIndexCurr);
    long q1midIndexCurr = floor(q1Curr.size() / 2.0);
    double q1MedianCurr = q1Curr.size() % 2 == 0 ? (q1Curr[q1midIndexCurr - 1] + q1Curr[q1midIndexCurr]) / 2.0 : q1Curr[q1midIndexCurr];

    std::vector<double> q3Curr = std::vector<double>(xCurr.begin() + midIndexCurr, xCurr.end());
    long q3midIndexCurr = floor(q3Curr.size() / 2.0);
    double q3MedianCurr = q3Curr.size() % 2 == 0 ? (q3Curr[q3midIndexCurr - 1] + q3Curr[q3midIndexCurr]) / 2.0 : q3Curr[q3midIndexCurr];

    double IQRCurr = q3MedianCurr - q1MedianCurr;
    double hiCurr = q3MedianCurr + 1.5 * IQRCurr;
    double loCurr = q1MedianCurr - 1.5 * IQRCurr;

    // find closet distance to lidar points
    double minXPrev = 1e9, minXCurr = 1e9;

    // debug: count filtered lidar points
    // int filteredPrev = 0;
    // int filteredCurr = 0;

    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        if (it->x < hiPrev && it->x > loPrev)
        {
            minXPrev = minXPrev > it->x ? it->x : minXPrev;
            // filteredPrev += 1;
        }
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        if (it->x < hiCurr && it->x > loCurr)
        {
            minXCurr = minXCurr > it->x ? it->x : minXCurr;
            // filteredCurr += 1;
        }
    }

    double dT = 1/frameRate;
    TTC = minXCurr * dT / (minXPrev - minXCurr);

    // cout << "(previous frame) lidar points before filter: " << lidarPointsPrev.size() << endl;
    // cout << "(previous frame) lidar points after filter: " << filteredPrev << endl;
    // cout << "(current frame) lidar points before filter: " << lidarPointsCurr.size() << endl;
    // cout << "(current frame) lidar points after filter: " << filteredCurr << endl;

}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // initialise matrix for counting keypoint correspondances in two frames
    const int size_c = currFrame.boundingBoxes.size();
    const int size_p = prevFrame.boundingBoxes.size();
    cv::Mat count = cv::Mat::zeros(size_c, size_p, CV_32S);

    // count keypoint correspondances
    for (auto it = matches.begin(); it != matches.end(); ++it)
    {
        // get keypoints from pair in current and previous frame
        cv::KeyPoint currKp = currFrame.keypoints.at(it->trainIdx);
        cv::KeyPoint prevKp = prevFrame.keypoints.at(it->queryIdx);

        // check if bounding box contain keypoints
        for (size_t currBB = 0; currBB < size_c; currBB++)
        {
            if (currFrame.boundingBoxes[currBB].roi.contains(currKp.pt))
            {
                for (size_t prevBB = 0; prevBB < size_p; prevBB++)
                  {
                      if (prevFrame.boundingBoxes[prevBB].roi.contains(prevKp.pt))
                      {
                          count.at<int>(currBB, prevBB) += 1;
                      }
                  }
            }
        }
    }

    // match bounding boxes which have the max keypoint matches correspondances
    for (size_t currBB = 0; currBB < size_c; currBB++)
    {
        vector<int> rowCount;
        for (size_t prevBB = 0; prevBB < size_p; prevBB++)
        {
            rowCount.push_back(count.at<int>(currBB,prevBB));
        }
        vector<int>::iterator maxElement = max_element(rowCount.begin(), rowCount.end());
        int matchedBB = distance(rowCount.begin(), maxElement);
        bbBestMatches[currBB] = matchedBB;
    }
}
