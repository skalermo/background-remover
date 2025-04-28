#include <libyuv.h>
#include <torch/script.h>

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "include/virtual_camera.hpp"

int IN_WIDTH = 640;
int IN_HEIGHT = 480;

cv::Mat extractCentralSquare(const cv::Mat& img) {
    if (img.empty()) {
        throw std::invalid_argument("Input image is empty");
    }

    int width = img.cols;
    int height = img.rows;

    int square_size = std::min(width, height);
    int x = (width - square_size) / 2;
    int y = (height - square_size) / 2;

    cv::Rect roi(x, y, square_size, square_size);

    return img(roi).clone();
}

cv::Mat applyMaskToImage(const cv::Mat& image, const cv::Mat& mask) {
    if (image.empty() || mask.empty()) {
        throw std::invalid_argument("Input image or mask is empty");
    }
    if (mask.size() != image.size()) {
        throw std::invalid_argument("Mask and image sizes do not match");
    }

    cv::Mat mask_uint8;
    if (mask.type() != CV_8U) {
        mask.convertTo(mask_uint8, CV_8U);
    } else {
        mask_uint8 = mask;
    }

    cv::Mat masked_image;
    image.copyTo(masked_image, mask_uint8);

    return masked_image;
}

cv::Mat processFrame(const cv::Mat& frame, torch::jit::script::Module model) {
    // frame preprocessing before feeding it into model
    cv::Mat rgb_frame;
    cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
    rgb_frame.convertTo(rgb_frame, CV_32F, 1.0 / 255.0);
    cv::Mat squared_frame = extractCentralSquare(rgb_frame);
    cv::Mat preprocessed;
    cv::resize(squared_frame, preprocessed, cv::Size(128, 128));

    // convert frame to tensor
    at::Tensor tensor_image = torch::from_blob(
        preprocessed.data,
        {1, preprocessed.cols, preprocessed.rows, 3},  // batch x width x height x channels
        torch::kFloat);

    // run forward pass to get segmentation mask
    tensor_image = tensor_image.permute({0, 3, 1, 2});
    at::Tensor output = model.forward({tensor_image}).toTensor();
    tensor_image = output.permute({0, 2, 3, 1});  // batch x width x height x channels
    tensor_image = tensor_image.squeeze();
    cv::Mat mask(preprocessed.rows, preprocessed.cols, CV_32FC1, tensor_image.data_ptr<float>());

    // apply segmentation mask onto input frame
    cv::cvtColor(mask, mask, cv::COLOR_RGB2BGR);
    cv::resize(mask, mask, cv::Size(IN_HEIGHT, IN_HEIGHT));
    cv::Mat output_image = applyMaskToImage(squared_frame, mask);
    output_image.convertTo(output_image, CV_8U, 255.0);
    cv::cvtColor(output_image, output_image, cv::COLOR_RGB2BGR);
    return output_image;
}

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: cpp_app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module model;
    try {
        model = torch::jit::load(argv[1]);
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model" << std::endl;
        return -1;
    }

    Camera cam(IN_HEIGHT, IN_HEIGHT, 30, libyuv::FOURCC_24BG, std::nullopt);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }
    cv::Mat frame;
    cap >> frame;

    while (!frame.empty()) {
        cv::imshow("Camera Feed", frame);

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        auto res = processFrame(frame, model);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "fps = " << 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
        cv::imshow("Result", res);

        if (cv::waitKey(1) == 27) {
            std::cout << "ESC pressed. Exiting..." << std::endl;
            break;
        }

        cam.send(res.data);
        cap >> frame;
    }

    cam.close();
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
