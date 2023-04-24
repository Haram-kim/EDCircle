#ifndef EDPF_H
#define EDPF_H

#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "utils.h"

using namespace std;
using namespace cv;

#define MAG_MAX 4096
#define EPSILON 1

// perwitt max value: 255 * 6 * 2 < 4096
// sobel max value: 255 * 8 * 2

class EDPF{
public:
    EDPF();
    ~EDPF();

    void process(Mat img);
    void preprocess();
    
    // edge drawing
    void ComputeAnchor();
    void LinkAnchor();
    void retrieve(int u, int v, int id, int direction);

    // validation
    bool NFA(float min_contrast, int len);
    void segment_validation(int id, int p_st, int p_end);    

    Px_helper px_helper;

    int height = -1;
    int width = -1;
    int channel = -1;

    Mat image;
    Mat blurred_image;
    Mat gx, gy, dir;    
    Mat nfa_gx, nfa_gy;

    Mat mag;
    Mat cv_filter_x, cv_filter_y;
    Mat cv_nfa_filter_x, cv_nfa_filter_y;
    Mat mag_inlier;
    
    float mag_threshold = 8.47;
    float anchor_threshold = 1;
    float len_threshold = 0;
    int scan_interval = 1;

    Mat edge_type;
    Mat edge_segment;    
    Mat edge_segment_valid;
    Mat edge_image;
    Mat edge_loop;

    int valid_id = 0;

    vector<Point2d> anchor_vec;

    EdgeSegment *edge_seg_arr = new EdgeSegment[MAX_SEGMENT_NUM]; // max uint16
    EdgeSegment *edge_seg_valid_arr = new EdgeSegment[MAX_SEGMENT_NUM]; // max uint16

    Parameters param;

    // parameter free
    float H[MAG_MAX];
    float H_elem[MAG_MAX];
    int N_p = 0;

private:
    // bool anchor_comp(Point2d a, Point2d b);
};













#endif // EDPF_H