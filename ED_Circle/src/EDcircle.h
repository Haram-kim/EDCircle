#ifndef EDCIRCLE_H
#define EDCIRCLE_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include "EDPF.h"

using namespace std;
using namespace cv;

class EDcircle{
public:
    EDcircle(EDPF *edpf_);
    ~EDcircle();
    
    void process();
    void preprocess();    
    void visualize();

    // fitting
    bool circle_fitting(EdgeSegment & edge_seg, int p_st, int p_end, Circle &circle);
    bool circle_fitting(vector<Point2d> & arcs, Circle & circle);
    bool ellipse_fitting(vector<Point2d> & arcs, Ellipse & elps);
    void line_fit(EdgeSegment & edge_seg, int p_st, int n_px);

    // segmentation
    void line_segmentation(EdgeSegment & edge_seg);    
    void arc_segmentation(EdgeSegment & edge_seg);
    void arc_joining();

    // validation
    bool NFA(int n, int k);
    
    void validate();
    void validate_circle(Circle &circle);
    void validate_ellipse(Ellipse &ellipse);
    
    // draw functions
    void draw_arc(Circle &circle);
    void draw_circle(Circle &circle);
    void draw_ellipse(Ellipse &elps);
    void draw_edge(EdgeSegment & edge_seg);
    
    EDPF *edpf;

    int height = -1;
    int width = -1;
    int channel = -1;
    
    Mat edge_segment; // segmented edge with id
    Mat edge_image; // segmented edge with id
    Mat edge_loop; // segmented edge with id

    Mat line_segment;
    Mat circle_segment;
    Mat arc_segment;
    Mat arc_segment_candidate;

    Mat nfa_gx;
    Mat nfa_gy;    

    vector<Circle> circle_candidates, circles;
    vector<Circle> arcs;
    vector<Ellipse> ellipse_candidates, ellipses;

    int id = 0;
    EdgeSegment *edge_seg_arr = new EdgeSegment[65536]; // max uint16

    // minimum line length
    int min_line_len = 10;

    // fitting threshold
    float circle_thres = 1.5;
    float ellipse_thres = 1.5;
    float line_thres = 0.5;

    // arc threshold
    float angle_min = 2;
    float angle_max = 70;

    // arc joining threshold

    float dist_thres = 20;
    float radius_thres = 10;
    float join_angle_thres = 120;

    // nfa variables
    float nfa_p = 1.0f/8;
    float nfa_pp = 1.0f/7;
    float nfa_angle_thres = 22.5;
    float nfa_N = 0;
    
};

#endif // EDCIRCLE_H