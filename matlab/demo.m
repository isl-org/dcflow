close all; clear all;

addpath('external/caffe/matlab')

param.model_file = '../net/deploy.prototxt';
param.maxDisp = 242; % maximum displacement for both x and y direction
param.ratio   = 3;   % downsample scale
param.P1      = 7;   % SGM param
param.P2      = 485; % SGM param
param.outOfRange    = 0.251; % Default cost for out-of-range displacements
param.occ_threshold = 0.8;   % threshold for fwd+bwd consisntency check

if 1 %% KITTI
  param.weight_file = '../net/kitti.caffemodel';
  im1 = imread('../data/000000_10.png');
  im2 = imread('../data/000000_11.png');
else %% Sintel
  param.P2 = 600;
  param.weight_file = '../net/sintel.caffemodel';
  im1 = imread('../data/frame_0001.png');
  im2 = imread('../data/frame_0002.png');
end

caffe.set_mode_gpu();
net = caffe.Net(param.model_file, param.weight_file, 'test');

matches = dcflow(im1, im2, param, net);
% dlmwrite('matches.txt', matches, ' ');
