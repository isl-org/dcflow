#include <mex.h>
#include <math.h>
#include <stdint.h>
#include "dcflow.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  if (nrhs < 8)
    mexErrMsgTxt("Wrong number of input arguments");

  const mwSize *dims = mxGetDimensions(prhs[0]);

  const int M = dims[0];
  const int N = dims[1];

  float *feat1 = (float *)mxGetData(prhs[0]);
  float *feat2 = (float *)mxGetData(prhs[1]);

  float *im1 = (float *)mxGetData(prhs[2]);
  float *im2 = (float *)mxGetData(prhs[3]);

  int max_offset = mxGetScalar(prhs[4]);
  float out_of_range = mxGetScalar(prhs[5]);

  int P1 = (int)mxGetScalar(prhs[6]);
  int P2 = (int)mxGetScalar(prhs[7]);

  int n_disps = 2*max_offset+1;
  n_disps = n_disps*n_disps;

  const mwSize dims_out[] = {2, M, N};

  plhs[0] = mxCreateNumericArray(3, dims_out, mxINT16_CLASS, mxREAL);
  int16_t *unary1 = (int16_t*)mxGetData(plhs[0]);

  plhs[1] = mxCreateNumericArray(3, dims_out, mxINT16_CLASS, mxREAL);
  int16_t *unary2 = (int16_t*)mxGetData(plhs[1]);

  sgmflow(feat1, feat2, im1, im2, M, N, n_disps, max_offset, out_of_range,
        P1, P2, unary1, unary2);
}
