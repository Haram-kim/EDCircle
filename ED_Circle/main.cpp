#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
using namespace cv;
using namespace std;

#include "EDPF.h"
#include "EDcircle.h"
#include "utils.h"

int main(int argc, char** argv )
{
    // TODO: make video image reader
    // should resize the image for large images
    if ( argc != 2 )
    {
        printf("usage: ./main <Image_Path>\n");
        return -1;
    }
    
    EDPF edpf;
    EDcircle ed_circle(&edpf);

    std::string ext = strrchr(argv[1],'.');
    // image
    if(ext.compare(".jpg") == 0 || ext.compare(".png") == 0){
        Mat image_src, image;
        image_src = imread( argv[1], 1 );
        if ( !image_src.data )
        {        
            printf("Not image data: \n");
            printf("%s \n", argv[1]);
            return -1;
        }
        if(image_src.channels() == 3){
            cv::cvtColor(image_src, image, cv::COLOR_RGB2GRAY);
        }
        edpf.process(image);
        ed_circle.process();
    }
    // video
    else if(ext.compare(".mp4") == 0 || ext.compare(".avi") == 0){
        try{
            VideoCapture video_capture(argv[1],  cv::CAP_FFMPEG);
        
            if(!video_capture.isOpened()){
                printf("No video data \n");
                printf("%s \n", argv[0]);
                printf("%s \n", argv[1]);
                return -1;
            }
            
            int frame_id = 0;
            while(1){
                Mat image_src, image;
                cout << frame_id++ << endl;
                video_capture >> image_src;
                if(image_src.empty()){
                    break;
                }
                if(image_src.channels() == 3){
                    cv::cvtColor(image_src, image, cv::COLOR_RGB2GRAY);
                }
                edpf.process(image);
                ed_circle.process();                
            }
        }
        catch(int A){
            printf("Not video data: \n");
            printf("%s \n", argv[1]);
            return -1;            
        }
    }
    waitKey(0);
    return 0;
}