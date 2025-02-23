#include <stdio.h>
#include <list>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#define MODEL_FILENAME RESOURCE_DIR"resnet34_peoplenet_int8.tflite"
//#define INPUT_FILENAME RESOURCE_DIR"sample_1080p_h265_frame_input.png"
#define INPUT_FILENAME RESOURCE_DIR"input.jpg"

#define TFLITE_MINIMAL_CHECK(x)                              \
    if (!(x)) {                                                \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

/* Pre/PostProcess for TensorFlow Lite Model for C/C++ */
class PeopleNetPrePostProcess 
{
    public:

    typedef struct s_boundingbox_type {
        int left;
        int top;
        int right;
        int bottom;
    } boundingbox;

    public:
    const int INFER_CLASSES = 3;

    private:

    const int MODEL_WIDTH = 960;
    const int MODEL_HEIGHT = 544;
    const int MODEL_CHANNEL = 3;

    const int INFER_BBOX_SIZE = 4;
    const int GRID_WIDTH = 60;
    const int GRID_HEIGHT = 34;

    const float GRID_STRIDE = 16.0;
    const float GRID_BBOX_NORM = 35.0;

    int width_orig_image = -1;
    int height_orig_image = -1;

    float grid_centers_w[60];
    float grid_centers_h[34];

    public:
    
    PeopleNetPrePostProcess
    (int width_orig, int height_orig) 
    {
        width_orig_image = width_orig;
        height_orig_image = height_orig;

        printf("original width=%d, height=%d\n",
            width_orig_image, height_orig_image);

        for (int i = 0; i < GRID_WIDTH; i++) {
            grid_centers_w[i] = (i * GRID_STRIDE + 0.5) / GRID_BBOX_NORM;
            //printf("[%d]%lf ", i, grid_centers_w[i]);
        }
    
        for (int i = 0; i < GRID_HEIGHT; i++) {
            grid_centers_h[i] = (i * GRID_STRIDE + 0.5) / GRID_BBOX_NORM;
            //printf("[%d]%lf ", i, grid_centers_h[i]);
        }
    }

    int 
    getSizeOfInputTensor
    (void)
    {
        return sizeof(signed char) * 1 * MODEL_HEIGHT * MODEL_WIDTH * MODEL_CHANNEL;
    }

    int 
    changeModelSizeToReal
    (float model_size, bool axis_W)
    {
        float real_size = 0;
        if (axis_W == true) {
            real_size = (float(model_size) / float(MODEL_WIDTH)) * width_orig_image;
        } else {
            real_size = (float(model_size) / float(MODEL_HEIGHT)) * height_orig_image;
        }
        return int(real_size);
    }
    
    cv::Mat 
    preProcess
    (cv::Mat image)
    {
        cv::Mat img_resize = image.clone();
        cv::resize(img_resize, img_resize, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
        cv::cvtColor(img_resize, img_resize, cv::COLOR_BGR2RGB);
        /* 入力テンソル int8 : -128 ~ 127 にする */
        double mMin, mMax;
        cv::minMaxLoc(img_resize, &mMin, &mMax);
        //printf("Orig: max = %lf, min = %lf\n", mMax, mMin);
        img_resize.convertTo(img_resize, CV_32SC3);
        cv::Mat offset(MODEL_HEIGHT, MODEL_WIDTH, CV_32SC3, cv::Scalar::all(-128));
        img_resize = img_resize + offset;
        cv::minMaxLoc(img_resize, &mMin, &mMax);
        //printf("SUBed : max = %lf, min = %lf\n", mMax, mMin);
        img_resize.convertTo(img_resize, CV_8SC3);
        cv::minMaxLoc(img_resize, &mMin, &mMax);
        //printf("Input : max = %lf, min = %lf\n", mMax, mMin);

        return img_resize;
    }

    int
    postProcess
    (float *output_tensor_classes, float *output_tensor_bbox,
     std::list<PeopleNetPrePostProcess::boundingbox> *pboundingboxes)
    {
        //std::list<PeopleNetPrePostProcess::boundingbox> boundingboxes[INFER_CLASSES];

        for (int c = 0; c < INFER_CLASSES; c++) {
            for (int h = 0; h < GRID_HEIGHT; h++) {
                for (int w = 0; w < GRID_WIDTH; w++) {
                    int offset_output_class = (h * GRID_WIDTH * INFER_CLASSES) + (w * INFER_CLASSES) + c;
                    float grid_acc = output_tensor_classes[offset_output_class];
                    if (grid_acc >= 0.5) {
                        //printf("H=%d, W=%d, C=%d (%lf)\n", h, w, c, grid_acc);
    
                        /* Decode BBOX */
                        int offset_output_bbox = 
                            (h * GRID_WIDTH * INFER_CLASSES * INFER_BBOX_SIZE) 
                            + (w * INFER_CLASSES * INFER_BBOX_SIZE) + c;
                        float o1 = output_tensor_bbox[offset_output_bbox + 0];
                        float o2 = output_tensor_bbox[offset_output_bbox + 1];
                        float o3 = output_tensor_bbox[offset_output_bbox + 2];
                        float o4 = output_tensor_bbox[offset_output_bbox + 3];
                        //printf("BBOX Information = %lf, %lf, %lf, %lf\n", o1, o2, o3, o4);
    
                        //printf("BBOX grid_centers_w = %lf, grid_centers_h =%lf\n", grid_centers_w[w], grid_centers_h[h]);
                        o1 = (grid_centers_w[w] - o1) * GRID_BBOX_NORM;
                        o2 = (grid_centers_h[h] - o2) * GRID_BBOX_NORM;
                        o3 = (o3 + grid_centers_w[w]) * GRID_BBOX_NORM;
                        o4 = (o4 + grid_centers_h[h]) * GRID_BBOX_NORM;
    
                        int left = changeModelSizeToReal(o1, true);
                        int top = changeModelSizeToReal(o2, false);
                        int right = changeModelSizeToReal(o3, true);
                        int bottom = changeModelSizeToReal(o4, false);
                        //printf("BBOX Information = %d, %d, %d, %d\n", left, top, right, bottom);
                        PeopleNetPrePostProcess::boundingbox bbox = 
                            {.left=left, .top=top, .right=right, .bottom=bottom};
                        pboundingboxes[c].push_back(bbox);
                    }
                }
            }
        }
        return 0;
    }
    
};

int main()
{
    /* Capture */
    cv::VideoCapture capture;
    //capture.open(0);
    capture.open(RESOURCE_DIR"sample_1080p_h265.mp4");
    if (!capture.isOpened()) {
        printf("could not found VideoCapture(0)\n");
        return -1;
    }

    /* fetch FirstFrame to get Camera Params */
    cv::Mat image;
    capture.read(image);
    cv::imshow("Video", image);
    const int key = cv::waitKey(1);

    int orig_width = image.cols;
    int orig_height = image.rows;
    PeopleNetPrePostProcess peopleNetPrePost(orig_width, orig_height);

    /* tfliteモデルのパス */
    printf("model file name : %s\n", MODEL_FILENAME);
    /* tfliteのモデルをFlatBufferに読み込む */
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(MODEL_FILENAME);
    /* 開けたかチェック */
    TFLITE_MINIMAL_CHECK(model != nullptr);
    
	/* インタープリタを生成する */
	tflite::ops::builtin::BuiltinOpResolver resolver;
	tflite::InterpreterBuilder builder(*model, resolver);
	std::unique_ptr<tflite::Interpreter> interpreter;
	builder(&interpreter);
    /* 生成できたかチェック */
	TFLITE_MINIMAL_CHECK(interpreter != nullptr);

	/* 入出力のバッファを確保する */
	TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
	//printf("=== Pre-invoke Interpreter State ===\n");
	//tflite::PrintInterpreterState(interpreter.get());

    while (true) 
    {
        capture.read(image);

        /* convert from Image to Tensor */
        cv::Mat input_tensor = peopleNetPrePost.preProcess(image);

        /* 入力テンソルに読み込んだ画像を格納する */
        signed char* input_sc_rgb = interpreter->typed_input_tensor<signed char>(0);
        memcpy(input_sc_rgb, input_tensor.data, peopleNetPrePost.getSizeOfInputTensor());
        
        /* 推論を実行 */
        TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
        //printf("\n\n=== Post-invoke Interpreter State ===\n");
        //tflite::PrintInterpreterState(interpreter.get());

        /* 出力テンソルから結果を取得して表示 */
        float* output_grid_info_bbox = interpreter->typed_output_tensor<float>(0);
        float* output_grid_info_class = interpreter->typed_output_tensor<float>(1);

        /* 出力テンソルからBoundingBoxを生成する */
        std::list<PeopleNetPrePostProcess::boundingbox> bboxes[peopleNetPrePost.INFER_CLASSES];
        peopleNetPrePost.postProcess(output_grid_info_class, output_grid_info_bbox, bboxes);

        /* BoundingBoxを描画する */
        std::list<PeopleNetPrePostProcess::boundingbox>::iterator itr_bbox;
        for (int c = 0; c < peopleNetPrePost.INFER_CLASSES; c++) {
            for (itr_bbox = bboxes[c].begin(); itr_bbox != bboxes[c].end(); itr_bbox++) {
                cv::rectangle(image, 
                    cv::Point(itr_bbox->left, itr_bbox->top),
                    cv::Point(itr_bbox->right, itr_bbox->bottom),
                    cv::Scalar(0, 0, 255), 1);
            }    
        }

        cv::imshow("Video", image);

        const int key = cv::waitKey(1);
        if (key == 'q') {
            break;
        }
    }

    /* 終了 */
	return 0;
}
