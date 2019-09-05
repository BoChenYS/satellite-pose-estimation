% Get initial poses for test images from estimated 2D landmarks

clear
load('pose_gt.mat');
load('pts11.mat'); % landmark 3D coords
load('preds.mat'); % landmard 2D coords in test images

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

ini_poses = zeros(N,7);
fail_id = 0;
num_failed = 0;

for ii = 1:N
    ii

    % get image point coords by model prediction
    img_pts = double(reshape(preds(ii,:,1:2),11,2));
    D = pdist(img_pts);
    [~,sid] = sort(conf,'descend');

    try
        [Rwi_hat,tiw_hat] = estimateWorldCameraPose(img_pts(:,1:2),pts,CamParams,'MaxReprojectionError', 3);
        twi_hat = -(Rwi_hat*tiw_hat')'.*pl;
        qwi_hat = rotm2quat(Rwi_hat);
    catch
        num_failed = num_failed + 1;
        fail_id(num_failed) = ii;
    end
    

    ini_poses(ii,:) = [qwi_hat, twi_hat];
end


