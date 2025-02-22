#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#define MODEL_FILENAME RESOURCE_DIR"resnet34_peoplenet_int8.tflite"

#define TFLITE_MINIMAL_CHECK(x)                              \
    if (!(x)) {                                                \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

int main()
{
    /* 入力となる画像データを読み込む "4.jpg" */
    printf("input image path : %s\n", RESOURCE_DIR"input.jpg");
    cv::Mat image = cv::imread(RESOURCE_DIR"input.jpg");
    /* ディスプレイに出力する */
    cv::imwrite("./.tmp.input.jpg", image);
    
    /* 入力画像を入力テンソルへと変換する */
    cv::Mat img_resize = image.clone();
    cv::Mat img_output = image.clone();
    
    /* 出力用の画像 */
    cv::resize(img_output, img_output, cv::Size(960, 544));

    /* 入力テンソル用の画像 */
    cv::cvtColor(img_resize, img_resize, cv::COLOR_BGR2RGB);
    cv::resize(img_resize, img_resize, cv::Size(960, 544));
    /* 入力テンソル int8 : -128 ~ 127 にする */
    double mMin, mMax;
    cv::minMaxLoc(img_resize, &mMin, &mMax);
    printf("max = %lf, min = %lf", mMax, mMin);
    img_resize.convertTo(image, CV_32SC3);
    img_resize = img_resize - 128;
    img_resize.convertTo(img_resize, CV_8SC3);
    cv::minMaxLoc(img_resize, &mMin, &mMax);
    printf("max = %lf, min = %lf", mMax, mMin);

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
	printf("=== Pre-invoke Interpreter State ===\n");
	tflite::PrintInterpreterState(interpreter.get());

	/* 入力テンソルに読み込んだ画像を格納する */
	signed char* input_sc_rgb = interpreter->typed_input_tensor<signed char>(0);
    memcpy(input_sc_rgb, img_resize.reshape(0, 1).data, sizeof(signed char) * 1 * 544 * 960 * 3);
    
	/* 推論を実行 */
	TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
	printf("\n\n=== Post-invoke Interpreter State ===\n");
	tflite::PrintInterpreterState(interpreter.get());

	/* 出力テンソルから結果を取得して表示 */
	float* output_grid_info_bbox = interpreter->typed_output_tensor<float>(0);
    float* output_grid_info_class = interpreter->typed_output_tensor<float>(1);

#if 0
    printf("\n---- BBOX ----\n");
    for (int i = 0; i < 1 * 34 * 60 * 12; i++) {
        printf("%lf ", output_grid_info_bbox[i]);
    }
#endif
    
    float grid_centers_w[60];
    float grid_centers_h[34];

    for (int i = 0; i < 60; i++) {
        grid_centers_w[i] = (i * 16.0 + 0.5) / 35.0;
        printf("[%d]%lf ", i, grid_centers_w[i]);
    }

    for (int i = 0; i < 60; i++) {
        grid_centers_h[i] = (i * 16.0 + 0.5) / 35.0;
        printf("[%d]%lf ", i, grid_centers_h[i]);
    }
    
    printf("\n---- CLASS ----\n");
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < 34; h++) {
            for (int w = 0; w < 60; w++) {
                int offset_output_class = (h * 60 * 3) + (w * 3) + c;
                float grid_acc = output_grid_info_class[offset_output_class];
                if (grid_acc >= 0.5) {
                    printf("H=%d, W=%d, C=%d (%lf)\n", h, w, c, grid_acc);

                    /* Decode BBOX */
                    int offset_output_bbox = (h * 60 * 4 * 3) + (w * 4 * 3) + (c * 4);
                    float o1 = output_grid_info_bbox[offset_output_bbox + 0];
                    float o2 = output_grid_info_bbox[offset_output_bbox + 1];
                    float o3 = output_grid_info_bbox[offset_output_bbox + 2];
                    float o4 = output_grid_info_bbox[offset_output_bbox + 3];
                    printf("BBOX Information = %lf, %lf, %lf, %lf\n", o1, o2, o3, o4);

                    o1 = (grid_centers_w[w] - o1) * 35.0;
                    o2 = (grid_centers_h[h] - o2) * 35.0;
                    o3 = (o3 + grid_centers_w[w]) * 35.0;
                    o4 = (o4 + grid_centers_h[h]) * 35.0;
                    
                    int left = int(o1);
                    int top = int(o2);
                    int right = int(o3);
                    int bottom = int(o4);
                    printf("BBOX Information = %d, %d, %d, %d\n", left, top, right, bottom);
                    cv::rectangle(img_output, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 2);
                }
            }
        }
    }

    /* ディスプレイに出力する */
    cv::imwrite("./.tmp.output.jpg", img_output);

    /* 終了 */
	return 0;
}
