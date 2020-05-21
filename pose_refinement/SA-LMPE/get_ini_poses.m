%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%           SIMULATED ANNEALING LEAST SQUARES POSE ESTIMATION
%
%
% This package contains the source code which implements the
% Nonlinear Pose Refinement (SA-LMPE) in
%
%       Satellite Pose Estimation with Deep Landmark Regression and 
%                       Nonlinear Pose Refinement  
%                       
%
% The source code, binaries and demo are supplied for academic use only.
% Do not distribute.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Concatinate 6 folds into 1 file
% clear
% 
% predAll = zeros(12000,11,3,'single');
% 
% for f = 1:6
%     load(sprintf('pred768f%01d.mat',f));
%     predAll((f-1)*1000+1 : f*1000,:,:) = preds(1:1000,:,:);
%     predAll((f-1)*1000+6001 : f*1000+6000,:,:) = preds(1001:2000,:,:);
% end

%% Get initial poses for all images

clear
load('pose_gt.mat');
load('pts11.mat');

load('pred768all.mat');

N = size(preds,1);
npts = size(pts,1);
pl = 0.00000586; % pixel length
f = 0.0176/pl;
u = 960;
v = 600;
K = [f, 0, u;
      0, f, v;
      0, 0, 1];
CamParams = cameraIntrinsics([f,f],[u,v],[1920,1200]);

    
Score = zeros(N,3);
meanD = zeros(N,1);
meanConf = zeros(N,1);
ini_poses = zeros(N,7);
fail_id = 0;
num_failed = 0;

for ii = 1:12000
    ii
    i = ii;

    qwi = pose_gt(i, 1:4);
    Rwi = quat2rotm(qwi);
    twi = pose_gt(i, 5:7);

    % get image point coords by model prediction
    img_pts = double(reshape(preds(ii,:,1:2),11,2));
    D = pdist(img_pts);
    meanD(ii) = mean(D);
    conf = reshape(preds(ii,:,3),11,1);
    meanConf(ii) = mean(conf);
    [~,sid] = sort(conf,'descend');
    n_below = sum(conf<=0.1);
    ids = find(conf>0.23,11);

    try
        [Rwi_hat,tiw_hat] = estimateWorldCameraPose(img_pts(ids,1:2),pts(ids,:),CamParams,'MaxReprojectionError', 3);
        twi_hat = -(Rwi_hat*tiw_hat')'.*pl;
        qwi_hat = rotm2quat(Rwi_hat);
        S_R = 2*acos(min(1,abs(qwi*qwi_hat')));
        S_t = norm(twi-twi_hat)/norm(twi);
        S_total = S_R + S_t;
        Score(ii,:) = [S_R, S_t, S_total];
    catch
        Score(ii,:) = [NaN, NaN, NaN];
        num_failed = num_failed + 1;
        fail_id(num_failed) = ii;
    end
    

    ini_poses(ii,:) = [qwi_hat, twi_hat];
end

mean(Score(~isnan(Score(:,1)),:),1)  % print mean scores
sum(isnan(Score(:,1)))  % count number of pnp failures


