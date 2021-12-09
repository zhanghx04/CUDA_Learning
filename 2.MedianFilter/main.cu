// Includes
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <time.h>
#include <math.h>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_string.h>    // helper for string parsing
#include <helper_image.h>     // helper for image and data comparison
#include <helper_cuda.h>      // helper for cuda error checking functions

using namespace std;




// ================================================================================================================
// ================================================================================================================

#define BLOCK_WIDTH 8
#define BLOCK_HEIGHT 8


const char* hwName = "HPC-Homework5";

//const char* image_filename = "lena.pgm";

int wind_size       = 3;
unsigned int width, height;

unsigned char* h_img = NULL;  // Host image
unsigned char* d_img = NULL;  // Device image
unsigned char* d_out = NULL;  // Device temp for output







__global__ void AddIntsCUDA(int* a, int* b)
{
    a[0] += b[0];
}


// =======
// Device
// =======

__global__ void median_filter_gpu(unsigned char* input_img, unsigned char* output_img, int img_w, int img_h, int wind_size)
{
    
    const int xx = threadIdx.x;
    const int yy = threadIdx.y;
    
    const int x = blockDim.x * blockIdx.x + xx;
    const int y = blockDim.y * blockIdx.y + yy;
    
    //int datalength = wind_size * wind_size;
    
    int r = wind_size / 2;
    
    unsigned char box[225];
    
  
    
    // Add value to the array
    int itr;
    itr = 0;
    for (int col = x - r; col <= x + r; col++)
    {
        if (col < 0 || col > img_w-1) continue;
        for (int row = y - r; row <= y + r; row++)
        {
            if (row < 0 || row > img_h-1) continue;
            
            box[itr] = input_img[ row * img_w + col ];
            itr++;
        }
    }
    

    
    
    // Bubble sort
    
    for (int i = 0; i < itr; ++i)
    {
        for (int j = i + 1; j < itr; ++j)
        {
            if( box[j] < box[i] )
            {
                unsigned char temp = box[i];
                box[i] = box[j];
                box[j] = temp;
            }
        }
    }
    

    // Save the median value to the output
    //int idx = (itr % 2 == 0) ? (itr/2) : (itr/2 + 1);
    
    output_img[ y * img_w + x ] = box[itr/2];
    //output_img[ ty_g * img_w + tx_g ] = input_img[ ty_g * img_w + tx_g ];
    
    
    //__syncthreads();
    
}



// ==============
// Image Loading
// ==============
void loadImageData(int argc, char **argv)
{
    // load image (needed so we can get the width and height before we create the window
    char *image_path = NULL;

    if (argc >= 1)
    {
        image_path = sdkFindFilePath(argv[1], argv[0]);
    }
    
    if (argc >=3)
    {
        wind_size = std::stoi(argv[2]);
    }

    if (image_path == 0)
    {
        printf("Error finding image file '%s'\n", argv[1]);
        exit(EXIT_FAILURE);
    }

    clock_t time = clock();
    sdkLoadPGM(image_path, (unsigned char **) &h_img, &width, &height);
    time = clock() - time;

    if (!h_img)
    {
        printf("Error opening file '%s'\n", image_path);
        exit(EXIT_FAILURE);
    }

    printf("    Loaded '%s', %d x %d pixels\n", image_path, width, height);
    printf("    Loading Image took %.4f ms\n", (double)time/(CLOCKS_PER_SEC/1000));
    printf("    Wind size: %d\n", wind_size);
}


int grid_divide(int length, int block_size)
{
    return ((length % block_size) == 0) ? (length / block_size) : (length / block_size + 1);
}



void median_filter_cpu(unsigned char* input_img, unsigned char* output_img, int img_w, int img_h, int wind_size)
{
    int r = wind_size / 2;

    // Go thru each pixel to get the median value
    for (int y = 0; y < img_h; y++)
    {
        for (int x = 0; x < img_w; x++)
        {
            // get a arry for storing the neighbors
            unsigned char box[255];
            int itr = 0;
            
            for (int row = y-r; row <=y+r; row++)
            {
                if (row < 0 || row > img_h-1) continue;
                for ( int col = x-r; col <= x+r; col++)
                {
                    if (col < 0 || col > img_w-1) continue;
                    
                    box[itr] = input_img[ row * img_w + col ];
                    itr++;
                }
            }
            
            // Bubble sort
    
            for (int i = 0; i < itr; ++i)
            {
                for (int j = i + 1; j < itr; ++j)
                {
                    if( box[j] < box[i] )
                    {
                        unsigned char temp = box[i];
                        box[i] = box[j];
                        box[j] = temp;
                    }
                }
            }
            
            output_img[ y * img_w + x ] = box[itr/2];
        }
    }
}



void showHelp()
{
    printf("\n%s : Command line options\n", hwName);
    printf("\t-device=n          (where n=0,1,2.... for the GPU device)\n\n");
    printf("> The default matrix size can be overridden with these parameters\n");
    printf("\t-dimX=row_dim_size (matrix row    dimensions)\n");
    printf("\t-dimY=col_dim_size (matrix column dimensions)\n");
}




// =====
// main
// =====


int main(int argc, char **argv)
{
    string imgN_str = argv[3];
    const char* output_name = imgN_str.c_str();


    // ---------------
    // Print CUDA Info
    // ---------------

    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        showHelp();
        return 0;
    }
    
    printf("\n\n[INFO] Printing CUDA Info\n\n");

    int devID = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp deviceProp;

    // --- get number of SMs on this GPU
    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    printf("> Device %d: \"%s\"\n", devID, deviceProp.name);

    
    
    
    
    printf("\n\n");
    printf("========================\n");
    printf(" %s Starting\n", hwName);
    printf("========================\n");
    
    
    printf("\n[INFO] Reading Image\n");
    loadImageData(argc, argv);
    printf("    Output Image Name: %s \n", output_name);
    
    // --- Allocate host and device memory spaces
    unsigned char* h_out = (unsigned char*)malloc(width*height);        // for receiving the results from device
    unsigned char* out_cpu = (unsigned char*)malloc(width*height);
    
    checkCudaErrors(cudaMalloc((void**) &d_img,  (height * width * sizeof(unsigned int))));
    checkCudaErrors(cudaMalloc((void**) &d_out, (height * width * sizeof(unsigned int))));
    
    // --- Copy data from Host to Device
    checkCudaErrors(cudaMemcpy(d_img, h_img, height * width, cudaMemcpyHostToDevice));
    
    // --- Grid and block sizes
    const dim3 grid  (grid_divide(width, BLOCK_WIDTH), grid_divide(height, BLOCK_HEIGHT), 1);
    const dim3 block (BLOCK_WIDTH, BLOCK_HEIGHT, 1);
    
    // -------------
    //  CPU version
    // -------------
    printf("\n[INFO] Median Filtering Image - Golden Standard host code version\n");
    
    clock_t tictok = clock();
    median_filter_cpu(h_img, out_cpu, width, height, wind_size);
    tictok = clock() - tictok;
    
    printf("    Golden Standard host code versio elapsed time: %.3f ms\n", (double)tictok/(CLOCKS_PER_SEC/1000));

    
    // --------
    //  kernel
    // --------
    printf("\n[INFO] Median Filtering Image - Kernel\n");
    
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    
    cudaFuncSetCacheConfig(median_filter_gpu, cudaFuncCachePreferShared);
    
    // Start kernel function
    median_filter_gpu<<<grid, block>>>(d_img, d_out, width, height, wind_size);
    
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    printf("    Kernel elapsed time: %.3f ms\n", time);
    
    
    // --- Copy results from Device to host
    checkCudaErrors(cudaMemcpy(h_out, d_out, height * width, cudaMemcpyDeviceToHost));
    
    
    
    // --- Compare the result between cpu version and gpu version
    printf("\n[INFO] Evaluating result between cpu version and gpu version \n");
    
    printf("    First 20 value from CPU result: %d", out_cpu[0]);
    for (int i = 1; i < 20; i++) printf(", %d", out_cpu[i]);
    printf("\n");
    
    printf("    First 20 value from GPU result: %d", h_out[0]);
    for (int i = 1; i < 20; i++) printf(", %d", h_out[i]);
    printf("\n");
    
    float MSE = 0.0;
    for (int i = 0; i < width*height; i++) MSE += sqrt(out_cpu[i] - h_out[i]);
    
    MSE /= (width*height);
    printf("    Mean Squared Error: %f \n", MSE);
    
    
    
    printf("\n[INFO] Saving Image\n");
    sdkSavePGM(output_name, (unsigned char *)h_out, width, height);
    

    
    
    // --- Clean up
    h_out = NULL;
    delete h_out;
    
    out_cpu = NULL;
    delete out_cpu; 

    if (h_img)
    {
        free(h_img);
        h_img=NULL;
    }

    if (d_img)
    {
        cudaFree(d_img);
        d_img=NULL;
    }

    if (d_out)
    {
        cudaFree(d_out);
        d_out=NULL;
    }
    
    
    return 0;
}

//-Wno-deprecated-gpu-targets