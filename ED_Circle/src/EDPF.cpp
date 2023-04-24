#include "EDPF.h"
#include "utils.h"
#include <cmath>

EDPF::EDPF(){
    int grad_type = PREWITT_TYPE;
    // for anchor
    float filter_x[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
    float filter_y[3][3] = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};
    if (grad_type == PREWITT_TYPE){
        // do nothing
    }
    else if (grad_type == SOBEL_TYPE){
        filter_x[1][0] = filter_y[0][1] = -2;
        filter_x[1][2] = filter_y[2][1] = 2;
    }
    
    cv_filter_x = Mat(3, 3, CV_32FC1);
    cv_filter_y = Mat(3, 3, CV_32FC1);
    std::memcpy(cv_filter_x.data, filter_x, 9*sizeof(float));
    std::memcpy(cv_filter_y.data, filter_y, 9*sizeof(float));

    // for nfa
    float nfa_filter_x[2][2] = {{-0.5, 0.5}, {-0.5, 0.5}};
    float nfa_filter_y[2][2] = {{-0.5, -0.5}, {0.5, 0.5}};

    
    cv_nfa_filter_x = Mat(2, 2, CV_32FC1);
    cv_nfa_filter_y = Mat(2, 2, CV_32FC1);
    std::memcpy(cv_nfa_filter_x.data, nfa_filter_x, 4*sizeof(float));
    std::memcpy(cv_nfa_filter_y.data, nfa_filter_y, 4*sizeof(float));
    
}

EDPF::~EDPF(){

}

void EDPF::process(Mat image){
    this->image = image;

    float f_val = 1;
    while(image.rows * image.cols * f_val * f_val > 640*480){
        f_val /= 2;
    }
    resize(image, this->image, Size(), f_val, f_val, INTER_LINEAR);

    height = this->image.rows;
    width = this->image.cols;
    channel = this->image.channels();

    preprocess();
    ComputeAnchor();
    LinkAnchor();
}

void EDPF::preprocess(){
    anchor_vec.clear();
    // init variables for new image
    edge_type = Mat(height, width, CV_8UC1, Scalar(0));
    edge_segment = Mat(height, width, CV_16UC1, Scalar(0));    
    edge_segment_valid = Mat(height, width, CV_16UC1, Scalar(0));
    edge_image = Mat(height, width, CV_8UC1, Scalar(0));    
    edge_loop = Mat(height, width, CV_16UC1, Scalar(0));
    anchor_vec.clear();

    // compute gradient
    cv::GaussianBlur(image, blurred_image, Size(5, 5), 1, 1 );
    filter2D(blurred_image, gx, CV_32F, cv_filter_x, Point(-1, -1), (0, 0), 4);
    filter2D(blurred_image, gy, CV_32F, cv_filter_y, Point(-1, -1), (0, 0), 4);

    filter2D(image, nfa_gx, CV_32F, cv_nfa_filter_x, Point(-1, -1), (0, 0), 4);
    filter2D(image, nfa_gy, CV_32F, cv_nfa_filter_y, Point(-1, -1), (0, 0), 4);


	dir = (abs(gx) >= abs(gy));
    // true: vertical edge
    // false: horizontal edge

    mag = abs(gx) + abs(gy); 
    // perwitt max value: 255 * 6 * 2
    // sobel max value: 255 * 8 * 2

    memset(H, 0, sizeof(H));
    memset(H_elem, 0, sizeof(H));
    // do not clear with memset
    for (int i = 0; i < MAX_SEGMENT_NUM; i++){
        edge_seg_arr[i] = EdgeSegment();
        edge_seg_valid_arr[i] = EdgeSegment();
    }
}

// Also computes cumulative distiburion of mag H
void EDPF::ComputeAnchor(){
    mag_inlier = (mag >= mag_threshold);

    for (int v = 1; v < height - 1; v++){
        for (int u = 1; u < width - 1; u++){
            if (mag_inlier.at<bool>(v, u) == 0 || (v % scan_interval != 0 && u % scan_interval != 0)){
                continue;
            }
            
            // true: vertical edge
            if (dir.at<bool>(v, u) != 0){
                if (mag.at<float_t>(v, u) - mag.at<float_t>(v - 1, u) >= anchor_threshold 
                && mag.at<float_t>(v, u) - mag.at<float_t>(v + 1, u) >= anchor_threshold){
                    edge_type.at<uint8_t>(v, u) = ANCHOR;
                    anchor_vec.push_back(Point2d(u, v));
                }
            }
            // false: horizontal edge
            else{
                if (mag.at<float_t>(v, u) - mag.at<float_t>(v, u - 1) >= anchor_threshold 
                && mag.at<float_t>(v, u) - mag.at<float_t>(v, u + 1) >= anchor_threshold){
                    edge_type.at<uint8_t>(v, u) = ANCHOR;
                    anchor_vec.push_back(Point2d(u, v));
                }
            }
            // cumulative distiburion of mag
            if (mag.at<float_t>(v, u) > 0){
                H_elem[MIN((int)mag.at<float_t>(v, u), MAG_MAX - 1)]++;
            }
        }
    }
    // note that H_elem[MAG_MAX - 1]= M in the paper
    for(int i = 0; i < MAG_MAX - 1; i++){
        H_elem[i + 1] += H_elem[i];
    }
    for(int i = 0; i < MAG_MAX; i++){
        H[i] = 1 - (float)H_elem[i]/(float)H_elem[MAG_MAX - 1];        
    }
    imshow("ANCHOR", edge_type);    
}

void EDPF::LinkAnchor(){
    // linking anchors
    // We should use DFS for EDPF
    int id = 0;
    // sort anchor vec
    auto comp = [&](const Point2d& a, const Point2d& b) {
        return mag.at<float_t>(a.y, a.x) > mag.at<float_t>(b.y, b.x);
    };
    sort(anchor_vec.begin(), anchor_vec.end(), comp);

    for (int i = 0; i < anchor_vec.size(); i++){
        int u = anchor_vec[i].x;
        int v = anchor_vec[i].y;
        // for anchors which became edge
        if(edge_type.at<uint8_t>(v, u) != ANCHOR){
            continue;
        }
        
        // true: vertical edge
        if (dir.at<bool>(v, u) != 0){
            // go up first and then go down
            retrieve(u, v, id, UP);
            edge_type.at<uint8_t>(v, u) = ANCHOR;
            edge_seg_arr[id].pts.erase(edge_seg_arr[id].pts.begin());
            std::reverse(edge_seg_arr[id].pts.begin(), edge_seg_arr[id].pts.end());
            // erase elem for preventing duplicate in retrive down
            retrieve(u, v, id, DOWN);
        }
        else{
            // go left first and then go right
            retrieve(u, v, id, LEFT);
            edge_type.at<uint8_t>(v, u) = ANCHOR;
            edge_seg_arr[id].pts.erase(edge_seg_arr[id].pts.begin());
            std::reverse(edge_seg_arr[id].pts.begin(), edge_seg_arr[id].pts.end());            
            // erase elem for preventing duplicate in retrive right
            retrieve(u, v, id, RIGHT);
        }
        id+=1;
    }

    // compute N_p    
    N_p = 0;
    for (int id_ = 1; id_ < id; id_++){
        int len = edge_seg_arr[id_].pts.size();
        N_p += len * (len - 1)/2;
    }    
    valid_id = 1;
    // segment validation for EDPF
    for (int id_ = 1; id_ < id; id_++){
        segment_validation(id_, 0, (int)edge_seg_arr[id_].pts.size() - 1);
    }

    Mat edge_ = edge_segment_valid.clone();
    for (int v = 0; v < height; v++){
        for (int u = 0; u < width; u++){
            edge_.at<uint16_t>(v, u) = (edge_.at<uint16_t>(v, u) * 7) % 255;
        }
    }
    edge_.convertTo(edge_, CV_8U);
    applyColorMap(edge_, edge_, COLORMAP_JET);
    imshow("edge_segment", edge_);

    edge_ = edge_loop.clone();
    edge_.convertTo(edge_, CV_8U);
    applyColorMap(edge_, edge_, COLORMAP_JET);
    imshow("loop edge_segments", edge_);
}
void EDPF::segment_validation(int id, int p_st, int p_end){
    int len = p_end - p_st + 1;
    if (len < len_threshold || p_st > p_end){
        return;
    }
    float min_contrast = 65535;
    // for connected piece P of S
    // Should be computed on tail or head of pts
    int p_min;
    int u_min;
    int v_min;
    for (int p = p_st; p <= p_end; p++){
        int u = edge_seg_arr[id].pts[p].x;
        int v = edge_seg_arr[id].pts[p].y;
        if(min_contrast > mag.at<float_t>(v, u)){
            min_contrast = mag.at<float_t>(v, u);
            p_min = p;
            u_min = u;
            v_min = v;
        }
    }
    // The edge segment p_st to p_end is valid when
    if(NFA(min_contrast, len)){
        edge_seg_valid_arr[valid_id].pts.clear();
        for (int p = p_st; p <= p_end; p++){
            int u = edge_seg_arr[id].pts[p].x;
            int v = edge_seg_arr[id].pts[p].y;
            edge_seg_valid_arr[valid_id].pts.push_back(edge_seg_arr[id].pts[p]);            
            edge_segment_valid.at<uint16_t>(v, u) = valid_id;
            edge_image.at<uint8_t>(v, u) = EDGE;
        }
        // loop edge check
        for (int i = 0; i < 8; i++){
            int u = edge_seg_arr[id].pts[p_end].x + px_helper.dx[i];
            int v = edge_seg_arr[id].pts[p_end].y + px_helper.dy[i];
            if(u ==  edge_seg_arr[id].pts[p_st].x && v ==  edge_seg_arr[id].pts[p_st].y){
                edge_seg_valid_arr[valid_id].loop = true;
                for (int p = p_st; p <= p_end; p++){
                    int u = edge_seg_arr[id].pts[p].x;
                    int v = edge_seg_arr[id].pts[p].y;
                    edge_loop.at<uint16_t>(v, u) = valid_id;
                }
            }
        }

        valid_id++;
        return;
    }
    int p_st_end = p_min - 1;
    int p_end_st = p_min + 1;        

    segment_validation(id, p_st, p_st_end);
    segment_validation(id, p_end_st, p_end);
    
}

bool EDPF::NFA(float min_contrast, int len){
    return N_p * pow(H[MIN((int)min_contrast, MAG_MAX - 1)], len) <= EPSILON;
}


void EDPF::retrieve(int u, int v, int id, int direction){
    // check 3-neighborhood pixel
    if(mag.at<float_t>(v, u) > 0 
        && edge_type.at<uint8_t>(v, u) != EDGE 
        && u > 0 && u < width
        && v > 0 && v < height){
        edge_type.at<uint8_t>(v, u) = EDGE;
        edge_segment.at<uint16_t>(v, u) = id;
        edge_seg_arr[id].pts.push_back(Point2d(u, v));
        float max_mag = -1.0f;
        int u_next = u;
        int v_next = v;
        int dir_idx = -1;
        for (int i = 0; i < 3; i++){
            int dir_idx_3 = px_helper.neighbor_px[direction][i];
            int u_next_3 = u + px_helper.dx[dir_idx_3];
            int v_next_3 = v + px_helper.dy[dir_idx_3];
            
            if(max_mag < mag.at<float_t>(v_next_3, u_next_3)){
                max_mag = mag.at<float_t>(v_next_3, u_next_3);
                u_next = u_next_3;
                v_next = v_next_3; 
                dir_idx = dir_idx_3;
            }
        }

        if (dir.at<bool>(v_next, u_next) != 0){
            if (dir_idx == 1 || dir_idx == 2 || dir_idx == 3){
                retrieve(u_next, v_next, id, UP);
            }
            else if(dir_idx == 5 || dir_idx == 6 || dir_idx == 7){
                retrieve(u_next, v_next, id, DOWN);
            }
        }
        else{
            if(dir_idx == 3 || dir_idx == 4 || dir_idx == 5) {
                retrieve(u_next, v_next, id, LEFT);
            }
            else if(dir_idx == 7 || dir_idx == 0 || dir_idx == 1){
                retrieve(u_next, v_next, id, RIGHT);
            }
        }
    }
    else if(edge_type.at<uint8_t>(v, u) != EDGE
            && u > 0 && u < width
            && v > 0 && v < height){
        edge_type.at<uint8_t>(v, u) = EDGE;
        edge_segment.at<uint16_t>(v, u) = id;
        edge_seg_arr[id].pts.push_back(Point2d(u, v));
    }
}
