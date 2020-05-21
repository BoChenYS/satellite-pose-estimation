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



Libraries
--------

- ceres-solver
    To install ceres-solver in MacOS:
    brew install ceres-solver 
    We tested under version 1.14.0_4.

- Eigen3
    To install Eigen3 in MacOS:
    brew install eigen 
    We tested under version 3.3.7.

- openMVG

    Code contain the compiled library under MacOS. For build details read
    https://github.com/openMVG/openMVG/blob/master/BUILD.md

    We compiled and copy libraries and part of the code into the ba/ 
    directory (for a different OS, similar steps shoud work):

    mkdir tmp
    cd tmp
    git clone https://github.com/openMVG/openMVG.git
    cd openMVG
    git submodule init
    git submodule update
    cd ..
    mkdir openMVG_Build
    cd openMVG_Build
    cmake -DCMAKE_BUILD_TYPE=RELEASE -G "Xcode" . ../openMVG/src/
    xcodebuild -configuration Release

    Copy the libraries to the ba/ directory
    cp Darwin-x86_64-RELEASE/Release/*.a ../../

    Copy the code to the ba/ directory
    cd ..
    cp -r openMVG/src/third_party ../
    cp -r openMVG/src/openMVG ../

    cd ..

    you can delete the tmp folder
    rm -rf tmp


Camera intrinsics
-----------------
Set camera intrinsics in Line 78 in ba/bundle_adjustment_data.cpp

Current camera intrinsics in the code are as in the SPEED dataset.


Compile
-------

Compile openMVG first

In MATLAB, go to the ba folder
cd ba
compile


Run
---

To re-run SA-LMPE experiments in the paper, execute demo.m inside MATLAB.

salmpe.m implements SA-LMPE


