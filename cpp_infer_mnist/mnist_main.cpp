#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#define MODEL_FILENAME RESOURCE_DIR"conv_mnist.tflite"

#define TFLITE_MINIMAL_CHECK(x)                              \
    if (!(x)) {                                                \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

int main()
{
    /* 入力となる画像データを読み込む "4.jpg" */
    printf("input image path : %s\n", RESOURCE_DIR"4.jpg");
    cv::Mat image = cv::imread(RESOURCE_DIR"4.jpg");
    /* ディスプレイに出力する */
    cv::imshow("InputImage", image);
    /* 画面表示が閉じられるまで待機 */
    cv::waitKey(0);
    
    /* 入力画像をgrayscaleへと変換する */
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    /* 28px x 28pxにリサイズする */
    cv::resize(image, image, cv::Size(28, 28));
    /* 背景が黒、文字色が白にする */
    image = ~image;
    /* ディスプレイに出力する */
    cv::imshow("InputImage for CNN", image);
    /* Normalize: 0.0 ~ 1.0 する */
    image.convertTo(image, CV_32FC1, 1.0 / 255);
    /* 画面表示が閉じられるまで待機 */
    cv::waitKey(0);

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
	float* input = interpreter->typed_input_tensor<float>(0);
	memcpy(input, image.reshape(0, 1).data, sizeof(float) * 1 * 28 * 28 * 1);

	/* 推論を実行 */
	TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
	printf("\n\n=== Post-invoke Interpreter State ===\n");
	tflite::PrintInterpreterState(interpreter.get());

	/* 出力テンソルから結果を取得して表示 */
	float* probs = interpreter->typed_output_tensor<float>(0);
	for (int i = 0; i < 10; i++) {
		printf("prob of %d: %.3f\n", i, probs[i]);
	}

    /* 終了 */
	return 0;
}