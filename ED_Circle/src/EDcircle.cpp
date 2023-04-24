#include "EDcircle.h"
#include <iostream>
#include <cmath>

float compute_pts_dist_line(Line line, EdgeSegment &edge_seg, int p_st, int line_len);
float compute_pts_dist_arc(Circle arc, EdgeSegment &edge_seg, int p_st, int p_end, int is_front);
void LS_line_fit(EdgeSegment &edge_seg, int p_st, int line_len, Line &line, float &err);
Ellipse convert_elps_coeff(Mat elps_coeff);

EDcircle::EDcircle(EDPF *edpf_):edpf(edpf_){
}

EDcircle::~EDcircle(){
}

void EDcircle::process(){
    preprocess();
    line_segment = Mat(height, width, CV_8UC1, Scalar(0));
    
    for (int i = 1; i < id; i++){  
        EdgeSegment & edge_seg = edge_seg_arr[i];

        Circle circle;
        Ellipse elps;
        if(edge_seg.loop && circle_fitting(edge_seg.pts, circle)){
            circles.push_back(circle);
            circle_candidates.push_back(circle);
        }
        else if(edge_seg.loop && ellipse_fitting(edge_seg.pts, elps)){
            ellipses.push_back(elps);
            ellipse_candidates.push_back(elps);
        }
        else{
            // draw_edge(edge_seg);            
            // Do line segment
            line_segmentation(edge_seg);
            arc_segmentation(edge_seg);
        }        
    }
    arc_joining();
    validate();
    visualize();
}

void EDcircle::preprocess(){
    // initialize
    circle_candidates.clear();
    circles.clear();
    arcs.clear();
    ellipse_candidates.clear();
    ellipses.clear();

    // image config
    height = edpf->height;
    width = edpf->width;
    channel = edpf->channel;

    id = edpf->valid_id - 1;

    // edgea image
    edge_segment = edpf->edge_segment_valid;
    edge_image = edpf->edge_image;
    edge_loop = edpf->edge_loop;
    
    // edge_segment
    edge_seg_arr = edpf->edge_seg_valid_arr; // max uint16

    // intermediate result
    circle_segment = edpf->image.clone();
    arc_segment = Mat(height, width, CV_8UC1, Scalar(0));
    arc_segment_candidate = Mat(height, width, CV_8UC1, Scalar(0));

    // NFA
    nfa_gx = edpf->nfa_gx;
    nfa_gy = edpf->nfa_gy;
    nfa_N = MIN(height, width); 

}

bool EDcircle::circle_fitting(EdgeSegment & edge_seg, int p_st, int p_end, Circle &circle){
    int pts_len = p_end - p_st;

    Mat A = Mat(pts_len, 3, CV_32F, Scalar(1)); // A = x y 1
    Mat B = Mat(pts_len, 1, CV_32F, Scalar(0)); // B = x^2 + y^2
    Mat X = Mat(3, 1, CV_32F, Scalar(0));       // X = a b c
    for(int p = p_st; p < p_end; p++){
        int x = edge_seg.pts[p].x;
        int y = edge_seg.pts[p].y;
        A.at<float>(p - p_st, 0) = x;
        A.at<float>(p - p_st, 1) = y;
        B.at<float>(p - p_st, 0) = x*x + y*y;            
    }

    // B - AX = x^2 + y^2 - ax - by - c = r'^2 - r^2
    // Least square solver X = A.pinv * B
    solve(A, B, X, DECOMP_SVD );
    float a = X.at<float>(0);
    float b = X.at<float>(1);
    float c = X.at<float>(2);  

    // Circle circle;
    circle.x = a*0.5;
    circle.y = b*0.5;
    circle.r = sqrt(4*c + a*a + b*b)*0.5;

    Mat err = B - A*X;
    float err_sum = 0;
    for(int p = 0; p < pts_len; p++){
        err_sum += abs(sqrt(err.at<float>(p) + circle.r*circle.r) - circle.r); // absolute radius error
    }
    err_sum /= pts_len; //mean radius error
    if(err_sum < circle_thres && pts_len > min_line_len){
        // angle sorting
        Point2d pts_st = edge_seg.pts[p_st] - Point2d(circle.x, circle.y);
        Point2d pts_mid = edge_seg.pts[(int)(p_st+p_end - 1)/2] - Point2d(circle.x, circle.y);
        Point2d pts_end = edge_seg.pts[p_end - 1] - Point2d(circle.x, circle.y);
        
        float angle_st = atan2(pts_st.y, pts_st.x)*180.0f/PI;
        float angle_mid = atan2(pts_mid.y, pts_mid.x)*180.0f/PI;
        float angle_end = atan2(pts_end.y, pts_end.x)*180.0f/PI;
        float angle;
        // s: start | m: middle | e: end
        // make: s < m < e or e < m < s
        if(angle_st > angle_end){
            angle_end += 360;
        }
        if(angle_st > angle_mid){
            angle_mid += 360;
        }
        if(angle_end < angle_mid){
            angle_st += 360;        
        }
        // now s < m < e or e < m < s
        // sort angles for compute angle between (draw circles is not affected.)
        if(angle_st < angle_mid){
    
            angle = angle_end - angle_st;    
        }
        else{
            angle = angle_st - angle_end;     
        }    
        circle.angle_st = angle_st;
        circle.angle_end = angle_end; 
        circle.angle_btw = angle;

        circle.pts = vector<Point2d>(edge_seg.pts.begin() + p_st, edge_seg.pts.begin() + p_end);
        return true;
    } 
    else{
        return false;
    }
}

bool EDcircle::circle_fitting(vector<Point2d> & pts, Circle & circle){
    int pts_len = pts.size();
    if(pts_len < 2*min_line_len){
        return false;
    }
    Mat A = Mat(pts_len, 3, CV_32F, Scalar(1)); // A = x y 1
    Mat B = Mat(pts_len, 1, CV_32F, Scalar(0)); // B = x^2 + y^2
    Mat X = Mat(3, 1, CV_32F, Scalar(0));       // X = a b c
    for(int p = 0; p < pts_len; p++){
        int x = pts[p].x;
        int y = pts[p].y;
        A.at<float>(p, 0) = x;
        A.at<float>(p, 1) = y;
        B.at<float>(p, 0) = x*x + y*y;            
    }

    // B - AX = x^2 + y^2 - ax - by - c = r'^2 - r^2
    // Least square solver X = A.pinv * B
    solve(A, B, X, DECOMP_SVD );
    float a = X.at<float>(0);
    float b = X.at<float>(1);
    float c = X.at<float>(2);  

    // Circle circle;
    circle.x = a*0.5;
    circle.y = b*0.5;
    circle.r = sqrt(4*c + a*a + b*b)*0.5;
    circle.pts = pts;

    Mat err = B - A*X;
    float err_sum = 0;
    for(int p = 0; p < pts_len; p++){
        err_sum += abs(sqrt(err.at<float>(p) + circle.r*circle.r) - circle.r); // absolute radius error
    }
    err_sum /= pts_len; //mean radius error
    if(err_sum < circle_thres){
        return true;     
    } 
    else{
        return false;
    }
}

bool EDcircle::ellipse_fitting(vector<Point2d> & pts, Ellipse & elps){
    int pts_len = pts.size();
    if(pts_len < 2*min_line_len){
        return false;
    }
    Mat A = Mat(pts_len, 6, CV_32F, Scalar(1)); // A = x^2 xy y^2 x y 1
    for(int p = 0; p < pts_len; p++){
        float x = pts[p].x;
        float y = pts[p].y;
        A.at<float>(p, 0) = x*x;
        A.at<float>(p, 1) = x*y;
        A.at<float>(p, 2) = y*y;
        A.at<float>(p, 3) = x;
        A.at<float>(p, 4) = y;
        A.at<float>(p, 5) = 1.0;
    }

    // AX = 0 == Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    // Least square solver 
    Mat svd_u, svd_s, svd_vt;
    SVD::compute(A, svd_s, svd_u, svd_vt);
    Mat elps_coeff = svd_vt(cv::Range(5, 6), cv::Range(0, 6)).t();
    
    elps = convert_elps_coeff(elps_coeff);
    elps.pts = pts;

    if(elps.det < 0){
        float elps_r_sq = elps.a*elps.b;
        Mat err = A*elps_coeff.mul(1.0f/elps_r_sq);
        float err_sum = 0;
        for(int p = 0; p < pts_len; p++){
            err_sum += abs(err.at<float>(p)/sqrt(elps_r_sq));
        }        
        err_sum /= pts_len;
        if (err_sum < ellipse_thres && MAX(elps.a/elps.b, elps.b/elps.a) < 3){
            return true;
        }
    }
    return false;
}

void EDcircle::validate(){
    for (int i = 0; i < circle_candidates.size(); i++){
        Circle circle = circle_candidates.at(i);
        validate_circle(circle);

    }
    for (int i = 0; i < ellipse_candidates.size(); i++){
        Ellipse elps = ellipse_candidates.at(i);
        validate_ellipse(elps);
    }
}

void EDcircle::validate_circle(Circle &circle){
    int n = circle.pts.size();
    int k = 0;
    for (int p = 0; p < n; p++){
        Point2d cur_pts = circle.pts.at(p);
        Point2d level_pts = cur_pts - Point2d(circle.x, circle.y);
        float level_angle = atan2(level_pts.x, -level_pts.y);
        float nfa_angle = atan2(nfa_gy.at<float>(cur_pts.y, cur_pts.x), nfa_gx.at<float>(cur_pts.y, cur_pts.x));
        float align_angle = abs(level_angle - nfa_angle) * 180.0f / PI;
        while(align_angle > 90){
            align_angle -= 90;
        }
        if(align_angle < nfa_angle_thres){
            k++;
        }
    }
    if(NFA(n, k)){
        circles.push_back(circle);
    }
}

void EDcircle::validate_ellipse(Ellipse &elps){
    int n = elps.pts.size();
    int k = 0;
    for (int p = 0; p < n; p++){
        Point2d cur_pts = elps.pts.at(p);
        Point2d level_pts = cur_pts - Point2d(elps.x, elps.y);
        float level_angle = atan2(level_pts.x, -level_pts.y);
        float nfa_angle = atan2(nfa_gy.at<float>(cur_pts.y, cur_pts.x), nfa_gx.at<float>(cur_pts.y, cur_pts.x));
        float align_angle = (level_angle - nfa_angle) * 180.0f / PI;
        while(align_angle > 90){
            align_angle -= 180;
        }
        if(abs(align_angle) < nfa_angle_thres){
            k++;
        }
    }
    if(NFA(n, k)){
        ellipses.push_back(elps);
    }
}

bool EDcircle::NFA(int n, int k){
    float b = 0;
    for (int i = k; i <= n ; i++){
        float a = 0; // log scale
        for (int j = 1; j <= i; j++){
            a += log(n - i + j) - log(j) + log(nfa_pp);
        }
        b += exp(a);
    }
    float NFA = exp(log(b) + n*log(1 - nfa_p) + 5*log(nfa_N)); // N^5 * cumulative binomial prob
    if(NFA <= 1){
        return true;
    }
    return false;
}

void EDcircle::visualize(){
    for (int i = 0; i < circle_candidates.size(); i++){
        draw_circle(circle_candidates.at(i));
    }
    for (int i = 0; i < ellipse_candidates.size(); i++){
        draw_ellipse(ellipse_candidates.at(i));
    }
    imshow("circle_candidate", circle_segment);
    circle_segment = edpf->image.clone();
    for (int i = 0; i < circles.size(); i++){
        draw_circle(circles.at(i));
    }
    for (int i = 0; i < ellipses.size(); i++){
        draw_ellipse(ellipses.at(i));
    }
    imshow("circle_segment", circle_segment);
    imshow("line_segment", line_segment);
    imshow("arc_segment", arc_segment);
    imshow("arc candidate", arc_segment_candidate);
    waitKey(10);
}

void EDcircle::draw_arc(Circle &circle){
    cv::ellipse(arc_segment, Point(circle.x, circle.y), Size(circle.r, circle.r), 0, circle.angle_st, circle.angle_end, Scalar(255), 1);
}

void EDcircle::draw_circle(Circle &circle){
    cv::ellipse(circle_segment, Point(circle.x, circle.y), Size(circle.r, circle.r), 0, circle.angle_st, circle.angle_end, Scalar(255), 1);
}

void EDcircle::draw_ellipse(Ellipse &elps){
    cv::ellipse(circle_segment, Point(elps.x, elps.y), Size(elps.a, elps.b), elps.t, 0, 360, Scalar(255), 1); 
}

void EDcircle::draw_edge(EdgeSegment & edge_seg){
    Mat temp = Mat(height, width, CV_8UC1, Scalar(0));
    int pts_len = edge_seg.pts.size();
    for(int p = 0; p < pts_len; p++){
        int u = edge_seg.pts[p].x;
        int v = edge_seg.pts[p].y;
        temp.at<uint8_t>(v, u) = 255;
        imshow("edge", temp);
        waitKey(10);
    }
}

void EDcircle::line_segmentation(EdgeSegment & edge_seg){
    line_fit(edge_seg, 0, edge_seg.pts.size());
    if(edge_seg.lines.size() < 1){
        return;
    }

    // for arc detection
    Line prev_line = edge_seg.lines[0];
    for (int i = 1; i < edge_seg.lines.size(); i++){
        Line cur_line = edge_seg.lines[i];        
        float angle = prev_line.dir - cur_line.dir;
        if(angle < -180){
            angle += 360;
        }
        else if(angle > 180){
            angle -= 360;
        }
        edge_seg.lines[i].angle = angle;
        prev_line = cur_line;
    }
}

void EDcircle::arc_segmentation(EdgeSegment & edge_seg){
    if(edge_seg.lines.size() < 1){
        return;
    }

    int prev_dir = -1;
    int arc_cand_iter = -1;
    vector<vector<Line> > arc_candidate;
    Line & prev_line = edge_seg.lines[0];
    int last_line_iter = edge_seg.lines.size()-1;
    for (int i = 0; i < last_line_iter; i++){
        Line &cur_line = edge_seg.lines[i];
        if(abs(cur_line.angle) > angle_min && abs(cur_line.angle) < angle_max){
            int cur_dir = (cur_line.angle > 0);
            if(prev_dir != cur_dir || abs(prev_line.angle - cur_line.angle) > 30){
                vector<Line> arc_line;
                arc_line.push_back(cur_line);
                arc_candidate.push_back(arc_line);
                arc_cand_iter++;
            }
            else if(i != last_line_iter - 1){
                arc_candidate.at(arc_cand_iter).push_back(cur_line);
            }
            else{
                arc_candidate.at(arc_cand_iter).push_back(cur_line);
                arc_candidate.at(arc_cand_iter).push_back(edge_seg.lines[last_line_iter]);
            }
            prev_dir = cur_dir;
        }
        else{
            prev_dir = -1;
        }
        prev_line = cur_line;
    }
    // check valid arc candidate
    for (int i = 0; i < arc_candidate.size(); i++){
        vector<Line> & cur_arc = arc_candidate.at(i);
        if(cur_arc.size() < 2){
            arc_candidate.erase(arc_candidate.begin() + i);
            i--;
            continue;
        }
        int p_st = cur_arc.front().p_st;
        int p_end = cur_arc.back().p_end;

        Mat temp = Mat(height, width, CV_8UC1, Scalar(0));
        for (int k = 0; k < cur_arc.size(); k++){
            Line & cur_line = cur_arc.at(k);
            // draw arg_segment line
            cv::line(arc_segment_candidate, cur_line.st, cur_line.ed, Scalar(255), 1);
        }

        Circle arc;
        if(circle_fitting(edge_seg, p_st, p_end, arc)){
            // Joining arcs this would makes more circle

            while (p_end < edge_seg.pts.size()){
                float dist = compute_pts_dist_arc(arc, edge_seg, p_st, p_end, false);
                if(dist > circle_thres){
                    break;  
                }
                p_end++;
            }            
            while (p_st >= 0){
                float dist = compute_pts_dist_arc(arc, edge_seg, p_st, p_end, true);
                if(dist > circle_thres){
                    break;
                }
                p_st--;
            }
            circle_fitting(edge_seg, p_st, p_end, arc);
            arcs.push_back(arc);
            draw_arc(arc);
        }
    }
}

void EDcircle::arc_joining(){
    auto comp = [&](const Circle& a, const Circle& b) {
        return a.r*a.angle_btw > b.r*b.angle_btw;
    };
    sort(arcs.begin(), arcs.end(), comp);

    bool is_visit[65535];
    memset(is_visit, 0, sizeof(is_visit));
    for (int i = 0; i < arcs.size(); i++){
        if(is_visit[i]){
            continue;
        }
        is_visit[i] = true;
        Circle &arc = arcs.at(i);
        vector<Point2d> sub_arcs_pts;
        sub_arcs_pts.insert(sub_arcs_pts.end(), arc.pts.begin(), arc.pts.end());
        float angle_sum = arc.angle_btw;
        vector<int> sub_arc_idx;
        for (int j = i + 1; j < arcs.size(); j++){
            Circle &sub_arc = arcs.at(j);
            // center distance
            float dist = sqrt((sub_arc.x - arc.x)*(sub_arc.x - arc.x) 
                            + (sub_arc.y - arc.y)*(sub_arc.y - arc.y));
            // radius error
            float radius_err = abs(arc.r - sub_arc.r);
            // joining arcs
            if(dist < dist_thres && radius_err < radius_thres){
                sub_arcs_pts.insert(sub_arcs_pts.end(), sub_arc.pts.begin(), sub_arc.pts.end());
                angle_sum += sub_arc.angle_btw;
                sub_arc_idx.push_back(j);
                is_visit[j] = true;
            }
        }
        
        // check angle_sum > 180
        if(angle_sum > join_angle_thres){
            Circle circle;
            Ellipse elps;
            if(circle_fitting(sub_arcs_pts, circle)){
                circle_candidates.push_back(circle);
            }            
            else if(ellipse_fitting(sub_arcs_pts, elps)){
                ellipse_candidates.push_back(elps);
            }
        }
        else{
            for (int idx = 0; idx < sub_arc_idx.size(); idx++){
                is_visit[sub_arc_idx.at(idx)] = false;
            }
        }
    }

    for (int i = 0; i < arcs.size(); i++){
        if(is_visit[i]){
            continue;
        }
        is_visit[i] = true;
        Circle &arc = arcs.at(i);
        vector<Point2d> sub_arcs_pts;
        sub_arcs_pts.insert(sub_arcs_pts.end(), arc.pts.begin(), arc.pts.end());
        float angle_sum = arc.angle_btw;
        vector<int> sub_arc_idx;
        for (int j = i + 1; j < arcs.size(); j++){
            Circle &sub_arc = arcs.at(j);
            // center distance
            float dist = sqrt((sub_arc.x - arc.x)*(sub_arc.x - arc.x) 
                            + (sub_arc.y - arc.y)*(sub_arc.y - arc.y));
            // radius error
            float radius_err = abs(arc.r - sub_arc.r);
            // joining arcs            
            sub_arcs_pts.insert(sub_arcs_pts.end(), sub_arc.pts.begin(), sub_arc.pts.end());
            angle_sum += sub_arc.angle_btw;
            sub_arc_idx.push_back(j);
            is_visit[j] = true;
        }
        
        // check angle_sum > 180
        if(angle_sum > join_angle_thres){
            Circle circle;
            Ellipse elps;
            if(circle_fitting(sub_arcs_pts, circle)){
                circle_candidates.push_back(circle);
            }            
            else if(ellipse_fitting(sub_arcs_pts, elps)){
                ellipse_candidates.push_back(elps);
            }
        }
        else{
            for (int idx = 0; idx < sub_arc_idx.size(); idx++){
                is_visit[sub_arc_idx.at(idx)] = false;
            }
        }

    }
}

void EDcircle::line_fit(EdgeSegment & edge_seg, int p_st, int n_px){
    float err = 1e12;
    Line line;

    while(n_px > min_line_len){
        LS_line_fit(edge_seg, p_st, min_line_len, line, err);
        if(err <= line_thres){
            break;
        }
        p_st++;
        n_px--;
    }
    if (err >= line_thres){
        return;
    }

    int line_len = min_line_len;
    while (line_len < n_px){
        float dist = compute_pts_dist_line(line, edge_seg, p_st, line_len);
        if(dist > 2*line_thres){
            break;
        }
        line_len++;
    }

    LS_line_fit(edge_seg, p_st, line_len, line, err);

    //draw line
    cv::line(line_segment, line.st, line.ed, Scalar(255), 1);

    edge_seg.lines.push_back(line);

    line_fit(edge_seg, p_st + line_len, n_px - line_len);
}

float compute_pts_dist_line(Line line, EdgeSegment &edge_seg, int p_st, int line_len){
    int p = p_st + line_len - 1;
    Mat A = (Mat_<float>(1, 3) << edge_seg.pts[p].x, edge_seg.pts[p].y, 1.0f);
    Mat line_coeff = (Mat_<float>(3, 1) << line.a, line.b, line.c);
    Mat dist_err = A*line_coeff;
    float err_sum = abs(dist_err.at<float>(0));
    return err_sum;
}

float compute_pts_dist_arc(Circle arc, EdgeSegment &edge_seg, int p_st, int p_end, int is_front){
    Point2d dist;
    if(is_front){
        dist = edge_seg.pts[p_st] - Point2d(arc.x, arc.x);
    }else{
        dist = edge_seg.pts[p_end-1] - Point2d(arc.x, arc.x);
    }    
    int pts_r = sqrt(dist.dot(dist));
    return abs(pts_r - arc.r);
}

void LS_line_fit(EdgeSegment &edge_seg, int p_st, int line_len, Line &line, float &err){
    int p_end = p_st + line_len;

    Mat A = Mat(line_len, 3, CV_32F, Scalar(1));
    for(int p = p_st; p < p_end; p++){
        int x = edge_seg.pts[p].x;
        int y = edge_seg.pts[p].y;
        A.at<float>(p - p_st, 0) = x;
        A.at<float>(p - p_st, 1) = y;
    }

    // Least square solver
    Mat svd_u, svd_s, svd_vt;
    SVD::compute(A, svd_s, svd_u, svd_vt);
    Mat line_coeff = svd_vt(cv::Range(2, 3), cv::Range(0, 3)).t();
    float line_norm = sqrt(line_coeff.at<float>(0)*line_coeff.at<float>(0) + line_coeff.at<float>(1)*line_coeff.at<float>(1)); 
    line_coeff = line_coeff.mul(1.0f/line_norm);
    
    line.a = line_coeff.at<float>(0);
    line.b = line_coeff.at<float>(1);
    line.c = line_coeff.at<float>(2);

    Mat dist_err = A*line_coeff;
    
    line.st = edge_seg.pts[p_st] - Point2d(dist_err.at<float>(0)*line.a, dist_err.at<float>(0)*line.b);
    line.ed = edge_seg.pts[p_end - 1] - Point2d(dist_err.at<float>(line_len-1)*line.a, dist_err.at<float>(line_len-1)*line.b);
    line.p_st = p_st;
    line.p_end = p_end;

    Point2d pts_vec = line.ed - line.st;
    line.dir = atan2(pts_vec.y, pts_vec.x)*180.0f/PI;

    float err_sum = 0;
    for(int p = 0; p < line_len; p++){
        err_sum += abs(dist_err.at<float>(p));
    }
    err_sum /= line_len;
    err = err_sum;
}

Ellipse convert_elps_coeff(Mat elps_coeff){
    float A = elps_coeff.at<float>(0, 0);
    float B = elps_coeff.at<float>(1, 0);
    float C = elps_coeff.at<float>(2, 0);
    float D = elps_coeff.at<float>(3, 0);
    float E = elps_coeff.at<float>(4, 0);
    float F = elps_coeff.at<float>(5, 0);
    
    float det = B*B-4*A*C;
    float X = 2*(A*E*E+C*D*D-B*D*E+det*F);
    float Y = sqrt((A-C)*(A-C)+B*B);

    float a = -sqrt(X*(A+C+Y))/det;
    float b = -sqrt(X*(A+C-Y))/det;
    float x = (2*C*D-B*E)/det;
    float y = (2*A*E-B*D)/det;
    float t = atan((C-A-Y)/B) * 180.0f / PI;
    
    float A_ = a*a*sin(t)*sin(t) + b*b*cos(t)*cos(t);
    float scale = A_/A;
    elps_coeff = elps_coeff.mul(scale);
    det*= scale*scale;

    Ellipse elps;
    elps.a = a;
    elps.b = b;
    elps.x = x;
    elps.y = y;
    elps.t = t;
    elps.det = det;
    
    return elps;
}