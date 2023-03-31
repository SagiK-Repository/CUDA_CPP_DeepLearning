// nvcc -o "Julia Set/JuliaSet" "Julia Set/juliaset.cu" -lpng --expt-relaxed-constexpr
// "./Julia Set/JuliaSet"

#include <iostream>
#include <cstdio>
#include <cmath>
#include <png.h>
#include <cuda_runtime.h>

#define WIDTH 20000
#define HEIGHT 10000
#define N 1000

__global__ void compute_julia_set(unsigned char *image)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = (y * WIDTH + x) * 3;

    double cx = -0.7;
    double cy = 0.27015;
    double zx = ((double)x / WIDTH) * 3.5 - 2.5;
    double zy = ((double)y / HEIGHT) * 2 - 1;

    for (int i = 0; i < N; i++) {
        double tx = zx * zx - zy * zy + cx;
        double ty = 2 * zx * zy + cy;
        zx = tx;
        zy = ty;
        if (zx * zx + zy * zy > 4) {
            image[index] = 0;
            image[index + 1] = i % 256;
            image[index + 2] = 0;
            return;
        }
    }

    image[index] = 0;
    image[index + 1] = 0;
    image[index + 2] = 0;
}

void write_png_file(const char *filename, int width, int height, unsigned char *image)
{
    FILE *fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(png_ptr, info_ptr);

    for (int y = 0; y < height; y++) {
        png_bytep row_pointer = image + y * width * 3;
        png_write_row(png_ptr, row_pointer);
    }

    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main()
{
    unsigned char *image = new unsigned char[WIDTH * HEIGHT * 3];
    
    unsigned char *d_image;
    cudaMalloc(&d_image, WIDTH * HEIGHT * 3);

    dim3 block_size(16, 16);
    dim3 grid_size((WIDTH + block_size.x - 1) / block_size.x, (HEIGHT + block_size.y - 1) / block_size.y);

    compute_julia_set<<<grid_size, block_size>>>(d_image);
    cudaMemcpy(image, d_image, WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost);

    write_png_file("julia_set.png", WIDTH, HEIGHT, image);

    cudaFree(d_image);
    delete[] image;

    return 0;

}
