// nvcc -o "Mandelbrot Set/mandelbrotset" "Mandelbrot Set/mandel brot set.cu"
// "./Mandelbrot Set/mandelbrotset"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>

#define WIDTH 1000
#define HEIGHT 1000
#define MAX_ITERATIONS 255

__global__ void mandelbrotKernel(uint8_t *img, double xmin, double ymin, double xmax, double ymax, double dx, double dy) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    double x = xmin + i * dx;
    double y = ymin + j * dy;
    double zr = x;
    double zi = y;
    int k;
    for (k = 0; k < MAX_ITERATIONS; k++) {
        if (zr * zr + zi * zi > 4.0) break;
        double tmp = zr * zr - zi * zi + x;
        zi = 2.0 * zr * zi + y;
        zr = tmp;
    }
    img[i + j * WIDTH] = k;
}

int main(void) {
    uint8_t *img = (uint8_t*)malloc(WIDTH * HEIGHT * sizeof(uint8_t));
    uint8_t *dev_img;
    double xmin = -2.0;
    double ymin = -2.0;
    double xmax = 2.0;
    double ymax = 2.0;
    double dx = (xmax - xmin) / WIDTH;
    double dy = (ymax - ymin) / HEIGHT;
    cudaMalloc(&dev_img, WIDTH * HEIGHT * sizeof(uint8_t));
    dim3 blockDim(16, 16);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y);
    mandelbrotKernel<<<gridDim, blockDim>>>(dev_img, xmin, ymin, xmax, ymax, dx, dy);
    cudaMemcpy(img, dev_img, WIDTH * HEIGHT * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    FILE *fp = fopen("mandelbrot.pgm", "wb");
    fprintf(fp, "P5\n%d %d\n%d\n", WIDTH, HEIGHT, 255);
    fwrite(img, sizeof(uint8_t), WIDTH * HEIGHT, fp);
    fclose(fp);
    free(img);
    cudaFree(dev_img);
    return 0;
}
