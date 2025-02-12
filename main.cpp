#include "src/OcrLite.h"
#include <iostream>
#include <chrono>
#include "windows.h"

#pragma execution_character_set("utf-8")

int padding = 50;
int maxSideLen = 1024;
float boxScoreThresh = 0.5f;
float boxThresh = 0.3f;
float unClipRatio = 1.6f;
bool doAngle = true;
bool mostAngle = true;

int main() {
    SetConsoleOutputCP(CP_UTF8);
    std::cout << "开始" << std::endl;
    auto ocr = OcrLite();
    ocr.initModels(
            "models2/det.onnx",
            "models2/cls.onnx",
            "models2/rec.onnx",
            "models2/keys.txt"
    );
    ocr.setNumThread(1);
    std::cout << "模型初始化完成" << std::endl;

    std::string imgPath;//C:/Users/huhu/Desktop/test.png
    std::cout << "输入图片路径: ";
    std::cin >> imgPath;

    auto mat = cv::imread(imgPath);
//    auto mat = cv::imread("C:/Users/huhu/Desktop/test.png");
    std::cout << "图片已读取" << std::endl;
    while (true){
        auto startTime = std::chrono::steady_clock::now();
        auto res = ocr.detect(
                mat,
                padding,
                maxSideLen,
                boxScoreThresh,
                boxThresh,
                unClipRatio,
                doAngle,
                mostAngle
        );
        auto endTime = std::chrono::steady_clock::now();

        std::cout << "OCR完成" << std::endl;

        for (const auto &i: res.textBlocks) {
            std::cout << i.text.c_str() << ": ";
            for (auto j: i.boxPoint) {
                std::cout << j;
            }
            std::cout << std::endl;
        }

        std::chrono::duration<double, std::milli> duration = endTime - startTime;
        std::cout << "用时" << std::to_string(duration.count()) << "毫秒" << std::endl;
    }

    system("pause");
    return 0;
}