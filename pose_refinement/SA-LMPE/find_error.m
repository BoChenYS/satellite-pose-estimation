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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [rot_err, tr_err] = find_error(pose_moba, pose_gt)

rot_err = 2*real(acos(...
    abs(sum(pose_moba(:,1:4) .* pose_gt(:,1:4), 2))...
    ));

tr_err = vecnorm(pose_moba(:,5:end)-pose_gt(:,5:end),2,2) ./ ...
                    vecnorm(pose_gt(:,5:end),2,2) ;
                
%  fprintf('mean rot error %f\n',rot_err )
%  fprintf('mean tr  error %f\n',tr_err )
end