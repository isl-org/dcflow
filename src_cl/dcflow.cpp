#include "dcflow.h"
#include "sgm.h"
#include "helper.h"

#include <vector>
#include <chrono>
#include <CL/cl.hpp>

#define PROFILE

#ifdef PROFILE
using namespace std::chrono;
#endif

void sgmflow(const float *feat1, const float *feat2,
   const float *im1, const float *im2,
    int M, int N, int L,
    int max_offset, float out_of_range,
    int P1, int P2,
    int16_t *unary1, int16_t *unary2) {

  std::vector<cl::Platform> all_plat;

  std::cout << M << "/" << N <<  "/" << max_offset << std::endl;

  cl::Platform::get(&all_plat);

  if (all_plat.size() == 0) {
    std::cout << "ERROR: No OpenCL platforms found, check OpenCL installation"
        << std::endl;
    return;
  }
  cl::Platform default_platform = all_plat[0];

  std::vector<cl::Device> all_dev;
  default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_dev);

  if (all_dev.size() == 0) {
    std::cout  << "ERROR: no OpenCL device found" << std::endl;
    return;
  }


  cl::Device default_device = all_dev[0];

  cl::Context context(default_device);

  cl::Program::Sources sources;

  std::string kernels = readKernel("../src_cl/sgm.cl");

  sources.push_back({kernels.c_str(), kernels.length()});

  cl::Program program(context, sources);
  if (program.build({default_device}, "-cl-mad-enable -cl-fast-relaxed-math -cl-single-precision-constant") != CL_SUCCESS) {
    std::cout<<"ERROR: Error building: "<<
        program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<< std::endl;
  }

  cl::CommandQueue queue(context, default_device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

  // Pitched memory doesn't help
  // Input data
  cl::Buffer unary1_d(context, CL_MEM_READ_WRITE, L*M*N*sizeof(uint8_t));
  queue.enqueueFillBuffer(unary1_d, uint8_t(255*out_of_range), 0, L*M*N*sizeof(uint8_t));

  cl::Buffer unary2_d(context, CL_MEM_READ_WRITE, L*M*N*sizeof(uint8_t));
  queue.enqueueFillBuffer(unary2_d, uint8_t(255*out_of_range), 0, L*M*N*sizeof(uint8_t));

  cl::Buffer feat1_d(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FEAT_DIM*M*N*sizeof(float), (void*)feat1);
  cl::Buffer feat2_d(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FEAT_DIM*M*N*sizeof(float), (void*)feat2);

  queue.finish();

  // Match
  #ifdef PROFILE
  high_resolution_clock::time_point match_start = high_resolution_clock::now();
  #endif

  std::vector<cl::Event> match_evt(1);
  cl::Kernel match_kernel(program, "match");
  match_kernel.setArg(0, feat1_d); match_kernel.setArg(1, feat2_d);
  match_kernel.setArg(2, M); match_kernel.setArg(3, N);
  match_kernel.setArg(4, max_offset);
  match_kernel.setArg(5, unary1_d); match_kernel.setArg(6, unary2_d);
  queue.enqueueNDRangeKernel(match_kernel, cl::NDRange(3, 3),
    cl::NDRange(M-6, N-6),
    cl::NullRange, NULL, &match_evt[0]);
  // queue.enqueueNDRangeKernel(match_kernel, cl::NullRange,
  //   cl::NDRange(M,DIVUP(N,4)),
  //   cl::NDRange(M,4), NULL, &match_evt[0]);

  // Fix border
  cl::Kernel fix_kernel(program, "fix_border");
  fix_kernel.setArg(0, unary1_d); fix_kernel.setArg(1, unary2_d);
  fix_kernel.setArg(2, M); fix_kernel.setArg(3, N); fix_kernel.setArg(4, L);
  queue.enqueueNDRangeKernel(fix_kernel, cl::NullRange,
    cl::NDRange(M, N),
    cl::NullRange, &match_evt, NULL);

  // Set up buffers for SGM
  cl::Buffer im1_d(context, CL_MEM_READ_ONLY, M*N*sizeof(float));
  queue.enqueueWriteBuffer(im1_d, CL_FALSE, 0, M*N*sizeof(float), im1);

  cl::Buffer im2_d(context, CL_MEM_READ_ONLY, M*N*sizeof(float));
  queue.enqueueWriteBuffer(im2_d, CL_FALSE, 0, M*N*sizeof(float), im2);

  #ifdef PROFILE
  match_evt[0].wait();
  high_resolution_clock::time_point match_end = high_resolution_clock::now();

  auto match_dur = duration_cast<milliseconds>(match_end - match_start).count();

  std::cout << "Matching: " << match_dur << " ms" << std::endl;
  #endif

  #ifdef PROFILE
  high_resolution_clock::time_point sgm_start = high_resolution_clock::now();
  #endif

  // Set up SGM
  SGM sgm(M, N, L, max_offset, P1, P2, queue, context, program);

  // Forward
  sgm.process(unary1_d, im1_d);
  queue.enqueueReadBuffer(*sgm.recoverFlow(), CL_FALSE, 0,
      M*N*sizeof(cl_short2), unary1);

  // Backward
  sgm.process(unary2_d, im2_d);
  queue.enqueueReadBuffer(*sgm.recoverFlow(), CL_FALSE, 0,
      M*N*sizeof(cl_short2), unary2);

  queue.finish();


  #ifdef PROFILE
  high_resolution_clock::time_point sgm_end = high_resolution_clock::now();

  auto sgm_dur = duration_cast<milliseconds>(sgm_end - sgm_start).count();

  std::cout << "SGM (fw+bw): " << sgm_dur << " ms" << std::endl;
  #endif
}


void wtaflow(const float *feat1, const float *feat2,
    int M, int N, int L,
    int max_offset, float out_of_range,
    int P1, int P2,
    uint8_t *unary1, uint8_t *unary2) {

  std::vector<cl::Platform> all_plat;


  std::cout << M << "/" << N <<  "/" << max_offset << std::endl;

  cl::Platform::get(&all_plat);

  if (all_plat.size() == 0) {
    std::cout << "ERROR: No OpenCL platforms found, check OpenCL installation"
        << std::endl;
    return;
  }
  cl::Platform default_platform = all_plat[0];

  std::vector<cl::Device> all_dev;
  default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_dev);

  if (all_dev.size() == 0) {
    std::cout  << "ERROR: no OpenCL device found" << std::endl;
    return;
  }

  cl::Device default_device = all_dev[0];

  cl::Context context(default_device);

  cl::Program::Sources sources;

  std::string kernels = readKernel("../src_cl/sgm.cl");

  sources.push_back({kernels.c_str(), kernels.length()});

  cl::Program program(context, sources);
  if (program.build({default_device}) != CL_SUCCESS) {
    std::cout<<"ERROR: Error building: "<<
        program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<< std::endl;
  }

  cl::CommandQueue queue(context, default_device, CL_QUEUE_PROFILING_ENABLE);


  cl::Buffer unary1_d(context, CL_MEM_READ_WRITE, L*M*N*sizeof(uint8_t));
  queue.enqueueFillBuffer(unary1_d, uint8_t(255*out_of_range), 0, L*M*N);

  cl::Buffer unary2_d(context, CL_MEM_READ_WRITE, L*M*N*sizeof(uint8_t));
  queue.enqueueFillBuffer(unary2_d, uint8_t(255*out_of_range), 0, L*M*N);

  cl::Buffer feat1_d(context, CL_MEM_READ_ONLY, FEAT_DIM*M*N*sizeof(float));
  queue.enqueueWriteBuffer(feat1_d, CL_TRUE, 0, FEAT_DIM*M*N*sizeof(float), feat1);

  cl::Buffer feat2_d(context, CL_MEM_READ_ONLY, FEAT_DIM*M*N*sizeof(float));
  queue.enqueueWriteBuffer(feat2_d, CL_TRUE, 0, FEAT_DIM*M*N*sizeof(float), feat2);

  auto match_kernel = cl::make_kernel<cl::Buffer&, cl::Buffer&,
      const int, const int, const int,
      cl::Buffer&, cl::Buffer&>(program, "match");

  cl::EnqueueArgs eargs(queue,
      cl::NullRange,
      cl::NDRange(M, N),
      cl::NullRange);

  match_kernel(eargs, feat1_d, feat2_d, M, N, max_offset,
    unary1_d, unary2_d);

  queue.enqueueReadBuffer(unary1_d, CL_TRUE, 0, L*M*N*sizeof(uint8_t), unary1);
  queue.enqueueReadBuffer(unary2_d, CL_TRUE, 0, L*M*N*sizeof(uint8_t), unary2);
}
