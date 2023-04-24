#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>

#define MIN(a,b) ((a) > (b) ? (b) : (a))
#define MAX(a,b) ((a) < (b) ? (b) : (a))

#define PI 3.14159265359

#define MAX_SEGMENT_NUM 65536

struct Circle{
    float x;
    float y;
    float r;
    float angle_st = 0;
    float angle_end = 360;
    float angle_btw = 360;
    std::vector<cv::Point2d> pts;

    ~Circle(){
        pts.clear();
    }
};

struct Ellipse{
    float a =  0;
    float b =  0;
    float x = 0;
    float y = 0;
    float t =  0;
    float det = 0;
    std::vector<cv::Point2d> pts;
    ~Ellipse(){
        pts.clear();
    }
};

struct Line{
    float a;
    float b;
    float c;
    cv::Point2d st;
    cv::Point2d ed;
    int p_st;
    int p_end;
    float angle = 0;
    float dir;
};

enum GRAD_TYPE{
    PREWITT_TYPE = 1,
    SOBEL_TYPE = 2
};
enum EDGE_TYPE{
    ANCHOR = 64,
    EDGE = 255,
};
enum DIRECTION{
    LEFT = 0,
    RIGHT = 1,
    UP = 2,
    DOWN = 3
};

struct Parameters{
    int GRAD_TYPE= 1;
};

struct EdgeSegment{
    // int len = 0;
    std::vector<cv::Point2d> pts;
    std::vector<Line> lines;
    bool loop = false;
    ~EdgeSegment(){
        pts.clear();
        lines.clear();
    }
};

struct Px_helper{
    const int neighbor_px[4][3] = {{3, 4, 5}, // left retreive map to idx of dx, dy
                                {7, 0, 1}, // right retreive map to idx of dx, dy
                                {1, 2, 3}, // up retreive map to idx of dx, dy
                                {5, 6, 7}}; // down retreive map to idx of dx, dy
    const int dx[8] = {+1, +1, 0, -1, -1, -1, 0, +1}; // r: right, l: left, u: up, d: down, -: fix
    const int dy[8] = {0, -1, -1, -1, 0, +1, +1, +1}; // 8-neighbor pixel order: r-, ru, -u, lu, l-, ld, -d, rd
};


// 8-neighbor pixel order
// +---+---+---+
// | 3 | 2 | 1 |
// +---+---+---+
// | 4 | - | 0 |
// +---+---+---+
// | 5 | 6 | 7 |
// +---+---+---+


#endif // UTILS_H