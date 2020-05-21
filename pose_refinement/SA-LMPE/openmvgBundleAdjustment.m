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


function [pts3dOut, camPosesOut, runtime] = openmvgBundleAdjustment(observations, pts3d, camPoses, K, sd, type)

% Create cell array of extrinsics {[r t]}
m = size(camPoses,2);
P = cell(1,m);
for i=1:m
    P{i} = [camPoses(i).R camPoses(i).t];
end

% points in homogeneous coordinates
n = size(pts3d,1);
X = [pts3d ones(n,1)]';


BAOpts = struct();
BAOpts.num_iterations = 150;
if strcmp(type, 'structure_only')
    BAOpts.extrinsic_type = 'none'; %'translation_only all none
    BAOpts.structure_type = 'all';
elseif strcmp(type, 'motion_only')
    BAOpts.extrinsic_type = 'all'; 
    BAOpts.structure_type = 'none';
elseif strcmp(type, 'full')
    BAOpts.extrinsic_type = 'all'; 
    BAOpts.structure_type = 'all';
else
    error('only structure_only supported')
end

[pts3dOut, Pbun, runtime] = openmvg_ba(observations, X, P, sd, BAOpts);


%%transforme output
pts3dOut = pts3dOut(1:3,:)';

camPosesOut = camPoses;
for i=1:m
    r = Pbun{i}(:,1:3);
    t = Pbun{i}(:,4);
    
    camPosesOut(i).R = r;
    camPosesOut(i).t = t;
    
end

end