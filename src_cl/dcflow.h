#ifndef SGMFLOW_MEX
#define SGMFLOW_MEX

#include <stdint.h>

#define MAX_SOURCE_SIZE (0x100000)

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 32

#define FEAT_DIM 64

void sgmflow(const float *feat1, const float *feat2,
             const float *im1, const float *im2,
             int M, int N, int L,
             int max_offset, float out_of_range,
             int P1, int P2,
             int16_t *unary1, int16_t *unary2);

#endif /* end of include guard:  */
