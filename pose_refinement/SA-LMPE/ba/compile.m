% Before compile make sure to set camera intrinsics in
% bunle_adjustment_data.cpp !!


CXXFLAGS = 'CXXFLAGS=$CXXFLAGS -Wall -std=c++11';

ICeres  = '-I/usr/local/include/';
LCeres  = '-L/usr/local/lib/'; 
lCeres  = ['-l','ceres'];

Iminglog  = ['-I','third_party/ceres-solver/internal/ceres/miniglog'];

mex ('openmvg_ba.cpp', 'bundle_adjustment_data.cpp', '-largeArrayDims', CXXFLAGS, ...
    '-I.', ...
    'bundle_adjustment_ceres.cpp',...    
    ICeres, LCeres, Iminglog, lCeres, '-I/usr/local/include/eigen3/',...
'-L.', '-lopenMVG_geometry', '-lopenMVG_multiview', '-lopenMVG_sfm')

