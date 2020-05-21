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

function [pose_moba] = make_pose(poses_out, pl)

m = numel(poses_out);
pose_moba = zeros(m,7);
for i=1:m
    pose_moba(i,1:4) = rotm2quat(poses_out(i).R);
    pose_moba(i,5:7) =  poses_out(i).t * pl;
end

end