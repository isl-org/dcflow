function matches = dcflow(im1_ori, im2_ori, param, net)
  maxDisp = param.maxDisp;   ratio = param.ratio;
  im1 = im2double(im1_ori); im2 = im2double(im2_ori);
  im1 = max(min(imresize(im1(1:end-mod(size(im1_ori,1),ratio),1:end-mod(size(im1_ori,2),ratio),:),1/ratio),1),0);
  im2 = max(min(imresize(im2(1:end-mod(size(im2_ori,1),ratio),1:end-mod(size(im2_ori,2),ratio),:),1/ratio),1),0);

  [M, N, ~] = size(im1);
  range = -floor(maxDisp/ratio):floor(maxDisp/ratio);
  r_max = range(end);

  im1_ = zeros(M+8, N+8, 3, 'single');    im2_ = im1_;
  for i = 1:3
    p1 = single(im1(:,:,i));  p2 = single(im2(:,:,i));
    im1_(:,:,i) = padarray((p1 - mean(p1(:)))/std(p1(:)), [4, 4], 'symmetric', 'both');
    im2_(:,:,i) = padarray((p2 - mean(p2(:)))/std(p2(:)), [4, 4], 'symmetric', 'both');
  end

  im1_ = permute(im1_, [2,1,3]); im2_ = permute(im2_, [2,1,3]);
  [M2, N2, ~] = size(im1_);
  net.blobs('data').reshape([M2 N2 3 2]);   net.reshape();
  feat = net.forward({cat(4,  im1_,  im2_)});
  feat_1 = feat{1}(:,:,:,1);         feat_2 = feat{1}(:,:,:,2);
  feat_1 = permute(feat_1, [2,1,3]); feat_2 = permute(feat_2, [2,1,3]);
  feat_1_n = sqrt(sum(feat_1.^2, 3)+1e-12);   feat_2_n = sqrt(sum(feat_2.^2, 3)+1e-12);
  feat_1 = bsxfun(@rdivide, feat_1, feat_1_n);  feat_2 = bsxfun(@rdivide, feat_2, feat_2_n);

  [forward, backward] = dcflow_mex(feat_1, feat_2, ...
            single(rgb2gray(im1)), single(rgb2gray(im2)), ...
            r_max, param.outOfRange, param.P1, param.P2);

  forward = single(permute(forward, [2, 3, 1]));   backward = single(permute(backward, [2, 3, 1]));
  matches = filter_matches(forward, backward, ratio, param.occ_threshold);
end

function [matches, occ] = filter_matches(forward, backward, ratio, threshold)
  [m,n,~] = size(forward);
  u = forward(:,:,1); v = forward(:,:,2);
  [x1, y1] = meshgrid(1:n, 1:m);
  x2 = x1 + u;  y2 = y1 + v;

  % out-of-boundary pixels
  B = (x2>n) | (x2<1) | (y2>m) | (y2<1);
  x2(B) = x1(B);   y2(B) = y1(B);
  inv_u = interp2(backward(:,:,1), x2, y2, 'linear', 0);  inv_v = interp2(backward(:,:,2), x2, y2, 'linear', 0);
  occ = ((u+inv_u).^2 + (v+inv_v).^2) <= threshold; occ(B) = 0;

  % count bump in the second image
  count = zeros(m,n);
  for r=1:m,
  	for c=1:n
      c1 = round(x2(r,c)); r1 = round(y2(r,c)); count(r1, c1) = count(r1, c1)+1;
    end
  end
  occ(count>=2) = 0;

  j = x1; i = y1;
  matches = [ratio*j(:) - ceil(ratio/2), ratio*i(:) - ceil(ratio/2), ratio*(j(:)+u(:)) - ceil(ratio/2), ratio*(i(:)+v(:)) - ceil(ratio/2)];
  valid = find(occ);
  matches = matches(valid, :);
end
