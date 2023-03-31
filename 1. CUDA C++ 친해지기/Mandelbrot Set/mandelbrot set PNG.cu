// nvcc -o "Mandelbrot Set/mandelbrotsetPNG" "Mandelbrot Set/mandelbrot set PNG.cu" -lpng --expt-relaxed-constexpr
// "./Mandelbrot Set/mandelbrotsetPNG"

// log 함수를 디바이스 코드에서 호출할 수 있도록 허용 : --expt-relaxed-constexpr

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <png.h>
#include <cuda_runtime.h>

#define DIM 10000
#define DIM_X 1000
#define DIM_Y 500
#define MAX_ITER 1000
const double X_MIN = -2.0;
const double X_MAX =  1.0;
const double Y_MIN = -1.5;
const double Y_MAX = 1.5;

__global__ void mandelbrot_kernel(uint8_t *img, int width, int height, double xmin, double xmax, double ymin, double ymax) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= width || idy >= height) return;
    double x = xmin + (xmax - xmin) * idx / (double)width;
    double y = ymin + (ymax - ymin) * idy / (double)height;
    double zx = x, zy = y;
    int iter;
    for (iter = 0; iter < MAX_ITER && zx * zx + zy * zy < 4; iter++) {
        double zx_new = zx * zx - zy * zy + x;
        double zy_new = 2 * zx * zy + y;
        zx = zx_new;
        zy = zy_new;
    }
    double log_iter = log(iter + 1);
    img[4 * (idy * width + idx) + 0] = (uint8_t)(255 * log_iter / log(MAX_ITER + 1));
    img[4 * (idy * width + idx) + 1] = (uint8_t)(255 * log_iter / log(MAX_ITER + 1));
    img[4 * (idy * width + idx) + 2] = (uint8_t)(255 * log_iter / log(MAX_ITER + 1));
    img[4 * (idy * width + idx) + 3] = 255;
}

void write_png(const char *filename, uint8_t *image, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);
    png_bytep row_pointer[height];
    for (int i = 0; i < height; i++) {
        row_pointer[i] = (png_bytep)(image + 4 * i * width);
    }
    png_write_image(png_ptr, row_pointer);
    png_write_end(png_ptr, NULL);
    fclose(fp);
    png_destroy_write_struct(&png_ptr, &info_ptr);
}


int main() {
    // set up image buffer
    uint8_t* img = (uint8_t*)malloc(4 * DIM * DIM);
    memset(img, 0, 4 * DIM * DIM);
    // set up kernel launch configuration
    dim3 block(32, 32);
    dim3 grid((DIM + block.x - 1) / block.x, (DIM + block.y - 1) / block.y);

    // set up device memory
    uint8_t *d_img;
    cudaMalloc((void**)&d_img, 4 * DIM * DIM);

    // set up timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // run kernel on device and time it
    cudaEventRecord(start);
    mandelbrot_kernel<<<grid, block>>>(d_img, DIM, DIM, X_MIN, X_MAX, Y_MIN, Y_MAX);
    cudaEventRecord(stop);

    // copy result back to host and write to file
    cudaMemcpy(img, d_img, 4 * DIM * DIM, cudaMemcpyDeviceToHost);
    write_png("mandelbrot.png", img, DIM, DIM);

    // print out timing information
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time elapsed: %f ms\n", milliseconds);

    // clean up
    cudaFree(d_img);
    free(img);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;

}
