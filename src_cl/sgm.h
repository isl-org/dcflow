#ifndef SGM_H
#include <CL/cl.hpp>

#define DIVUP(S, A) ((S + (A) - 1)/(A))*(A)

#define HORZ_DBS 32
#define VERT_DBS 32

#define HORZ_BS_X 8
#define VERT_BS_X 8


class SGM
{
public:
  SGM(int M, int N, int L, int max_offset, int P1, int P2,
    cl::CommandQueue &queue, cl::Context &context, cl::Program &program);

  void process(cl::Buffer &unary, cl::Buffer &im);

  cl::Buffer *recoverFlow();

  cl::Buffer *getCostVolume() {
    queue_.finish();
    return out_;
  }

  ~SGM();

private:
  void setupKernels();

  void processHorz(int x, int reverse);
  void processVert(int y, int reverse);

  int M_, N_, L_, max_offset_, P1_, P2_, sL_, ps_, ngroups_horz_, ngroups_vert_;

  cl::CommandQueue &queue_;
  cl::Context &context_;
  cl::Program &program_;

  cl::Buffer *min_bufs_[4];

  cl::Image3D *tmp_bufs_h_[4];
  cl::Image3D *tmp_bufs_v_[4];

  cl::Buffer *scratch_h_[2];
  cl::Buffer *scratch_v_[2];

  cl::Buffer *out_;
  cl::Buffer *flow_;

  cl::Kernel *sgm_horz_kernel_, *sgm_vert_kernel_;
  cl::Kernel *red_horz_kernel_, *red_vert_kernel_;
  cl::Kernel *recover_kernel_;

  std::vector<cl::Event> sync_lr_fw_;
  std::vector<cl::Event> sync_lr_bw_;
  std::vector<cl::Event> sync_tb_fw_;
  std::vector<cl::Event> sync_tb_bw_;

  std::vector<cl::Event> preq_lr_fw_;
  std::vector<cl::Event> preq_lr_bw_;
  std::vector<cl::Event> preq_tb_fw_;
  std::vector<cl::Event> preq_tb_bw_;
};

#endif // SGM_H
