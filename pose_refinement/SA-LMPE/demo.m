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

clear all
addpath ba
addpath data

 
% Load data

% K is the Camera Intrinsic matrix. 
load('K');

% pts11.mat is the (11*3) 3D key points expressed in the world frame. 
aux = load('pts11.mat');
pts3d = aux.pts;


% ?pred768all.mat? (12000*11*3) is the predicted 2D key point locations. 
%    (:,:,1) is the horizontal coordinates, 
%    (:,:,2) is the vertical coordinates, and 
%    (:,:,3) is the confidence scores of each key point prediction. 

aux = load('pred768all.mat');
pts2d = aux.preds;

% 'pose_gt.mat' is the ground truth pose (N*7 matrix).
aux = load('pose_gt.mat');
pose_gt  = aux.pose_gt;


% 'ini_poses.mat' is the initial poses (N*7 matrix) that I obtained using the matlab PnP function. 
aux = load('ini_poses.mat'); 
ini_poses_quat = aux.ini_poses;


pl = 0.00000586; % pixel length


% build observations
m = size(pts2d,1); % number of views
n = size(pts2d,2); % number of landmarks

obs = cell(1,m);
num_obs = 0;

for i=1:m
    obs{i} = nan(3,n);
    for j=1:n
        x = pts2d(i,j,1); % check is a finite number
        if isfinite(x)
            u = [pts2d(i,j,1); pts2d(i,j,2); pts2d(i,j,3)];
            obs{i}(:,j) = u; %u(1:2);
            num_obs = num_obs + 1;
        end
    end
end

X = pts3d;

% build camera poses
ini_poses = struct();
for i=1:m
    R = quat2rotm(ini_poses_quat(i,1:4));
    t = ini_poses_quat(i,5:7)'/pl;
    
    ini_poses(i).R = R;
    ini_poses(i).t = t;
end

% compute initial residuals
P = cell(1,m); %projections
for i=1:m
    P{i} = K*[ini_poses(i).R ini_poses(i).t];
end

res = zeros(1,num_obs);
k = 1;
for i=1:m
    for j=1:n
        if isnan(obs{i}(1,j))
            continue
        end
        U = [X(j,:)';1];
        u_proj = P{i}*U;
        x = u_proj(1:2)/u_proj(3); % in pixel coordinates
        res(k) = norm(obs{i}(1:2,j) - x); % I should squared the residual
        k = k+1;
    end
end

sres = sort(res);
%plot(sres);
%plot(sres(1:ceil(num_obs*.98))); % plot 50% best residuals


%%
figure ('name', 'Initial distribution of residuals')
set(gcf, 'Color', 'white')
hist(sres(1:ceil(num_obs*.99)),50)
xlabel('reprojection error')
ylabel('frequency')

%savefig('res_hist_init')
%%

m = size(pts2d,1); % number of views

%[poses_out, scores] = salmpe(K,pl, X, obs, ini_poses, ini_poses_quat, pose_gt)
clear poses_out;
clear scores;
res_out = struct();


tic
for i=1:m
    [poses_out(i), scores(i), res_out(i).res] = salmpe(K,pl, X, obs(i), ini_poses(i), ini_poses_quat(i,:), pose_gt(i,:));
end
total_time = toc
sr = arrayfun(@(x) x.rot(end), scores);
st = arrayfun(@(x) x.tr(end), scores) ;

mean(sr+st)

%
tmax = 10;
iters=0:tmax;
% overall score
figure
set(gcf, 'Color', 'white')
%plot(scores.rot+scores.tr, 'linewidth', 2)

s = mean(reshape([scores.tr], [tmax+1,m])' +reshape([scores.rot], [tmax+1,m])');
plot(iters, s,  'linewidth', 2)
grid on
xlabel('iteration')
ylabel('overall score')
%xlim([1, tmax+1])
xticks(0:tmax)
%savefig('overall_score')


%% score evolution
% translation
figure
st = mean(reshape([scores.tr], [tmax+1,m])' );
plot(iters, st,  'linewidth', 2)
xlabel('iteration')
ylabel('translation score')

grid on
xticks(0:tmax)

set(gcf, 'Color', 'white')
ax = gca;
ax.YAxis.Exponent = 0; % scientific notation

%savefig('tr_score')


%% rotation score
figure
set(gcf, 'Color', 'white')

sr = mean(reshape([scores.rot], [tmax+1,m])' );
plot(iters, sr,  'linewidth', 2)


grid on
xlabel('iteration')
ylabel('rotation score')
%xlim([1, numel(scores.rot)])

xticks(0:tmax)
ax = gca;
ax.YAxis.Exponent = 0; % scientific notation

%%  Create the poses table

poses_table = make_pose(poses_out, pl)

%savefig('rot_score')
