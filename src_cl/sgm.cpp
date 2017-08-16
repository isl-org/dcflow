#include "sgm.h"
#include "helper.h"
#include <iostream>

SGM::SGM(int M, int N, int L, int max_offset, int P1, int P2,
  cl::CommandQueue &queue, cl::Context &context, cl::Program &program) :
  M_(M), N_(N), L_(L), max_offset_(max_offset), P1_(P1), P2_(P2),
  sL_(2*max_offset+1), ps_(L + (4 - (L % 4))),
  queue_(queue), context_(context), program_(program) {

  ngroups_horz_ = DIVUP(ps_,HORZ_DBS)/HORZ_DBS;
  ngroups_vert_ = DIVUP(ps_,VERT_DBS)/VERT_DBS;

  // Set up output
  out_ = new cl::Buffer(context, CL_MEM_READ_WRITE, L*M*N*sizeof(uint16_t));
  flow_ = new cl::Buffer(context, CL_MEM_READ_WRITE, M*N*sizeof(cl_short2));

  // Set up all temporary buffers
  for (int i = 0; i < 4; ++i) {
    min_bufs_[i] = new cl::Buffer(context, CL_MEM_READ_WRITE, M*N*sizeof(uint16_t));

    tmp_bufs_h_[i] = new cl::Image3D(context, CL_MEM_READ_WRITE,
          cl::ImageFormat(CL_R, CL_UNSIGNED_INT16), M, sL_, sL_);
    tmp_bufs_v_[i] = new cl::Image3D(context, CL_MEM_READ_WRITE,
          cl::ImageFormat(CL_R, CL_UNSIGNED_INT16), N, sL_, sL_);
  }

  for (int i = 0; i < 2; ++i) {
    scratch_h_[i] = new cl::Buffer(context, CL_MEM_READ_WRITE, M*ngroups_horz_*sizeof(uint16_t));
    scratch_v_[i] = new cl::Buffer(context, CL_MEM_READ_WRITE, N*ngroups_vert_*sizeof(uint16_t));
  }

  setupKernels();
}

void SGM::process(cl::Buffer &unary, cl::Buffer &im) {
  CL_CHECK_ERR_R(queue_.enqueueFillBuffer(*out_, uint16_t(0), 0, L_*M_*N_*sizeof(uint16_t)));

  sgm_horz_kernel_->setArg(0, unary);
  sgm_horz_kernel_->setArg(1, im);

  sgm_vert_kernel_->setArg(0, unary);
  sgm_vert_kernel_->setArg(1, im);
  CL_CHECK_ERR_R(queue_.finish());

  // Process all 4 canonical directions
  for (int x = 0, y = 0; (x < N_) | (y < M_); ++x, ++y) {
    int xr = N_-1-x;
    int yr = M_-1-y;

    if (x < N_)
      processHorz(x,  0);

    if (xr >= 0)
      processHorz(xr, 1);

    if (y < M_)
      processVert(y, 0);

    if (yr >= 0)
      processVert(yr, 1);
  }
}

void SGM::processHorz(int x, int reverse) {
  auto *tmp_from = tmp_bufs_h_[x % 2 ? reverse : 2 + reverse];
  auto *tmp_to = tmp_bufs_h_[x % 2 ? 2 + reverse : reverse];
  int inc = reverse ? -M_ : M_;

  std::vector<cl::Event> *glob_sync = reverse ? &sync_lr_bw_ : &sync_lr_fw_;
  std::vector<cl::Event> *preq_sync = reverse ? &preq_lr_bw_ : &preq_lr_fw_;

  preq_sync->push_back(cl::Event());

  // SGM pass
  sgm_horz_kernel_->setArg(2, *min_bufs_[reverse]);
  sgm_horz_kernel_->setArg(11, *tmp_from);
  sgm_horz_kernel_->setArg(12, *tmp_to);
  sgm_horz_kernel_->setArg(13, x);
  sgm_horz_kernel_->setArg(14, inc);
  sgm_horz_kernel_->setArg(15, *scratch_h_[reverse]);
  sgm_horz_kernel_->setArg(16, cl::Local(M_*HORZ_DBS*sizeof(ushort)));

  CL_CHECK_ERR_R(queue_.enqueueNDRangeKernel(*sgm_horz_kernel_,
    cl::NullRange,
    cl::NDRange(DIVUP(M_,HORZ_BS_X), DIVUP(L_,HORZ_DBS)),
    cl::NDRange(HORZ_BS_X,HORZ_DBS), glob_sync, &(preq_sync->back())));

  // Final reduction pass
  red_horz_kernel_->setArg(0, *scratch_h_[reverse]);
  red_horz_kernel_->setArg(3, ngroups_horz_);
  red_horz_kernel_->setArg(4, x);
  red_horz_kernel_->setArg(5, *min_bufs_[reverse]);

  glob_sync->push_back(cl::Event());

  CL_CHECK_ERR_R(queue_.enqueueNDRangeKernel(*red_horz_kernel_, cl::NullRange,
    cl::NDRange(DIVUP(M_, HORZ_DBS)),
    cl::NDRange(HORZ_DBS), preq_sync, &(glob_sync->back())));
}

void SGM::processVert(int y, int reverse) {
  auto *tmp_from = tmp_bufs_v_[y % 2 ? reverse : 2 + reverse];
  auto *tmp_to = tmp_bufs_v_[y % 2 ? 2 + reverse : reverse];
  int inc = reverse ? -1 : 1;

  std::vector<cl::Event> *glob_sync = reverse ? &sync_tb_bw_ : &sync_tb_fw_;
  std::vector<cl::Event> *preq_sync = reverse ? &preq_tb_bw_ : &preq_tb_fw_;

  preq_sync->push_back(cl::Event());

  // SGM pass
  sgm_vert_kernel_->setArg(2, *min_bufs_[2+reverse]);
  sgm_vert_kernel_->setArg(11, *tmp_from);
  sgm_vert_kernel_->setArg(12, *tmp_to);
  sgm_vert_kernel_->setArg(13, y);
  sgm_vert_kernel_->setArg(14, inc);
  sgm_vert_kernel_->setArg(15, *scratch_v_[reverse]);
  sgm_vert_kernel_->setArg(16, cl::Local(N_*VERT_DBS*sizeof(ushort)));

  CL_CHECK_ERR_R(queue_.enqueueNDRangeKernel(*sgm_vert_kernel_,
    cl::NullRange,
    cl::NDRange(DIVUP(N_,VERT_BS_X), DIVUP(L_,VERT_DBS)),
    cl::NDRange(VERT_BS_X,VERT_DBS), glob_sync, &(preq_sync->back())));

  // Final reduction pass
  red_vert_kernel_->setArg(0, *scratch_v_[reverse]);
  red_vert_kernel_->setArg(3, ngroups_vert_);
  red_vert_kernel_->setArg(4, y);
  red_vert_kernel_->setArg(5, *min_bufs_[2+reverse]);

  glob_sync->push_back(cl::Event());

  CL_CHECK_ERR_R(queue_.enqueueNDRangeKernel(*red_vert_kernel_, cl::NullRange,
    cl::NDRange(DIVUP(N_, VERT_DBS)),
    cl::NDRange(VERT_DBS), preq_sync, &(glob_sync->back())));
}

cl::Buffer *SGM::recoverFlow() {
  CL_CHECK_ERR(queue_.finish());

  // This is slow -> Implement proper reduction!
  CL_CHECK_ERR(queue_.enqueueNDRangeKernel(*recover_kernel_,
    cl::NullRange,
    cl::NDRange(M_, N_),
    cl::NullRange, NULL, NULL));

  CL_CHECK_ERR(queue_.finish());

  return flow_;
}

void SGM::setupKernels() {
  // Setup constant arguments of kernels
  sgm_horz_kernel_ = new cl::Kernel(program_, "sgm_slice_horz");
  sgm_horz_kernel_->setArg(3, M_);
  sgm_horz_kernel_->setArg(4, N_);
  sgm_horz_kernel_->setArg(5, L_);
  sgm_horz_kernel_->setArg(6, sL_);
  sgm_horz_kernel_->setArg(7, ps_);
  sgm_horz_kernel_->setArg(8, P1_);
  sgm_horz_kernel_->setArg(9, P2_);
  sgm_horz_kernel_->setArg(10, *out_);

  sgm_vert_kernel_ = new cl::Kernel(program_, "sgm_slice_vert");
  sgm_vert_kernel_->setArg(3, M_);
  sgm_vert_kernel_->setArg(4, N_);
  sgm_vert_kernel_->setArg(5, L_);
  sgm_vert_kernel_->setArg(6, sL_);
  sgm_vert_kernel_->setArg(7, ps_);
  sgm_vert_kernel_->setArg(8, P1_);
  sgm_vert_kernel_->setArg(9, P2_);
  sgm_vert_kernel_->setArg(10, *out_);

  red_horz_kernel_ = new cl::Kernel(program_, "min_reduce_horz_fin");
  red_horz_kernel_->setArg(1, M_);
  red_horz_kernel_->setArg(2, N_);
  red_horz_kernel_->setArg(3, L_);

  red_vert_kernel_ = new cl::Kernel(program_, "min_reduce_vert_fin");
  red_vert_kernel_->setArg(1, M_);
  red_vert_kernel_->setArg(2, N_);
  red_vert_kernel_->setArg(3, L_);

  recover_kernel_ = new cl::Kernel(program_, "recover_flow");
  recover_kernel_->setArg(0, *out_);
  recover_kernel_->setArg(1, M_);
  recover_kernel_->setArg(2, N_);
  recover_kernel_->setArg(3, L_);
  recover_kernel_->setArg(4, *flow_);
  recover_kernel_->setArg(5, sL_);
  recover_kernel_->setArg(6, max_offset_);
}

SGM::~SGM() {
  CL_CHECK_ERR(queue_.finish());

  for (int i = 0; i < 4; ++i) {
    delete min_bufs_[i];
    delete tmp_bufs_h_[i];
    delete tmp_bufs_v_[i];
  }

  for (int i = 0; i < 2; ++i) {
    delete scratch_h_[i];
    delete scratch_v_[i];
  }

  delete out_;
  delete flow_;

  delete sgm_horz_kernel_;
  delete sgm_vert_kernel_;
  delete red_horz_kernel_;
  delete red_vert_kernel_;
  delete recover_kernel_;
};
