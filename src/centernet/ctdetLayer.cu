#include "ctdetLayer.h"
#include "common.h"
dim3 cudaGridSize(uint32_t n)
{
    uint32_t k = (n - 1) /BLOCK + 1;
    uint32_t x = k ;
    uint32_t y = 1 ;
    if (x > 65535 )
    {
        x = ceil(sqrt(x));
        y = (n - 1 )/(x*BLOCK) + 1;
    }
    dim3 d = {x,y,1} ;
    return d;
}
__device__ float Logist(float data){ return 1./(1. + exp(-data)); }

__global__ void CTdetforward_kernel(const float *hm, const float *reg,const float *wh ,
        float *output,const int w,const int h,const int classes,const int kernel_size,const float visthresh  ) {
    int idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (idx >= w * h * classes) return;
    int padding = (kernel_size - 1) / 2;
    int offset = -padding;
    int stride = w * h;
    int grid_x = idx % w;
    int grid_y = (idx / w) % h;
    int cls = idx/w/h ;
    int  l, m;
    int reg_index = idx - cls*stride;
    float c_x, c_y;
    float objProb = Logist(hm[idx]);
    if (objProb > visthresh) {
        float max = -1;
        int max_index = 0;
        for (l = 0; l < kernel_size; ++l)
            for (m = 0; m < kernel_size; ++m) {
                int cur_x = offset + l + grid_x;
                int cur_y = offset + m + grid_y;
                int cur_index = cur_y * w + cur_x + stride * cls;
                int valid = (cur_x >= 0 && cur_x < w && cur_y >= 0 && cur_y < h);
                float val = (valid != 0) ? Logist(hm[cur_index]) : -1;
                max_index = (val > max) ? cur_index : max_index;
                max = (val > max) ? val : max;
            }

        if(idx == max_index){
            int resCount = (int) atomicAdd(output, 1);
            //printf("%d",resCount);
            char *data = (char *) output + sizeof(float) + resCount * sizeof(BBoxInfo);
            BBoxInfo*det = (BBoxInfo*) (data);
            c_x = grid_x + reg[reg_index];
            c_y = grid_y + reg[reg_index + stride];
            det->box.x1 = (c_x - wh[reg_index] / 2) * 4;
            det->box.y1 = (c_y - wh[reg_index + stride] / 2) * 4;
            det->box.x2 = (c_x + wh[reg_index] / 2) * 4;
            det->box.y2 = (c_y + wh[reg_index + stride] / 2) * 4;
            det->label = cls;
            //det->classId = cls;
            det->prob = objProb;
        }
    }
}

void CTdetforward_gpu(const float *hm, const float *reg,const float *wh ,float *output,
                      const int w,const int h,const int classes,const int kernerl_size, const float visthresh ){
    uint32_t num = w * h * classes;
    CTdetforward_kernel<<<cudaGridSize(num),BLOCK>>>(hm,reg,wh,output,w,h,classes,kernerl_size,visthresh);
}
