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

function [poses_out, scores, res]=salmpe(K,pl, X, obs, ini_poses, ini_poses_quat, pose_gt_quat)
sd = 5;
th = 50; % define outliers
sd_min = 1;
th_min = 4;
lambda_sd = .7;
lambda_th = .7;

alpha = .8;

m = size(ini_poses,2);

%% optimise

it = 1;
inlier_masks = struct();

clear max_res;

scores = struct();

[err_rot, err_tr] = find_error(ini_poses_quat, pose_gt_quat);
scores.rot(it) = mean(err_rot);
scores.tr(it) = mean(err_tr);
%gamma(it) = max(reproj_error(K, obs, ini_poses, X ));
gamma(it) = rms(reproj_error(K, obs, ini_poses, X ));
% do not change input

poses_out = ini_poses;
obs_out = obs; % 

while 1
    it = it+1;
     
    %remove unseen lanmarks if run on single images... todo: update for
    %batch
    if m==1
        unseen_lanmarks_map = isnan(obs_out{1}(1,:));
        X = X(~unseen_lanmarks_map,:);
        obs_out{1}(:,unseen_lanmarks_map) = [];
    end
    
    [~, poses_out, ~] = openmvgBundleAdjustment(obs_out, X, poses_out, K, sd, 'motion_only');
    
    % save score
    pose_moba = make_pose(poses_out, pl);
    [ err_rot, err_tr] = find_error(pose_moba, pose_gt_quat);
    scores.rot(it) = mean(err_rot);
    scores.tr(it) = mean(err_tr);
    
    %% removing outliers
    [obs_out, res, inlier_mask] = outrem(K, obs_out, poses_out, X, th, alpha);
    inlier_masks(it).mask = inlier_mask;
    num_of_outliers = sum(~inlier_mask);
    
    %% stoping criterion
    %gamma(it) = max(res);
    gamma(it) = rms(res);
    dr = gamma(it-1)-gamma(it); 
    
    %% cooling    
    %if (dr>0)
    sd = max(sd_min,sd*lambda_sd);
    th = max(th_min,th*lambda_th); %4
    %end
    
    scores.dr(it) = dr;
    scores.n(it) = size(X,1);
    
%     if abs(dr)<.05 %&& num_of_outliers==0
%         break;
%     end

    if it>10
        break
    end
end


end