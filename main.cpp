#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

__global__ void convertToGrayscale(unsigned char* d_input, unsigned char* d_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        unsigned char r = d_input[idx];
        unsigned char g = d_input[idx + 1];
        unsigned char b = d_input[idx + 2];
        unsigned char gray = static_cast<unsigned char>(0.298f * r + 0.587f * g + 0.114f * b);
        d_output[y * width + x] = gray;
    }
}

void processImage(const std::string& inputImagePath, const std::string& outputImagePath) {
    cv::Mat image = cv::imread(inputImagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Erro ao ler a imagem: " << inputImagePath << std::endl;
        return;
    }

    int width = image.cols;
    int height = image.rows;
    size_t imageSize = width * height * 3 * sizeof(unsigned char);
    size_t grayImageSize = width * height * sizeof(unsigned char);

    unsigned char* d_input;
    unsigned char* d_output;
    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, grayImageSize);

    cudaMemcpy(d_input, image.data, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    convertToGrayscale<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    unsigned char* h_output = new unsigned char[width * height];
    cudaMemcpy(h_output, d_output, grayImageSize, cudaMemcpyDeviceToHost);

    cv::Mat grayImage(height, width, CV_8UC1, h_output);
    cv::imwrite(outputImagePath, grayImage);

    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    std::string inputImagePath = "C:\\Users\\pagan\\OneDrive\\Área de Trabalho\\alterar-imagem-programa.jpg";
    std::string outputImagePath = "output_image.jpg";

    processImage(inputImagePath, outputImagePath);

    return 0;
}