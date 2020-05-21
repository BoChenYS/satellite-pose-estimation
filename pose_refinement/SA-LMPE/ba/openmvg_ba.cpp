#include "openMVG/cameras/Camera_Common.hpp"
//#include "openMVG/cameras/Cameras_Common_command_line_helper.hpp"
#include "sequential_SfM.hpp"
//#include "openMVG/sfm/pipelines/sfm_features_provider.hpp"
//#include "openMVG/sfm/pipelines/sfm_matches_provider.hpp"
//#include "openMVG/sfm/sfm_data.hpp"
//#include "openMVG/sfm/sfm_data_io.hpp"
//#include "openMVG/sfm/sfm_report.hpp"
//#include "openMVG/sfm/sfm_view.hpp"
//#include "openMVG/system/timer.hpp"
#include "openMVG/types.hpp"

#include "third_party/cmdLine/cmdLine.h"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <ctime>

#include "bundle_adjustment_ceres.hpp"
#include "ceres/ceres.h"

#include "mex.h"

using namespace openMVG;
using namespace openMVG::cameras;
using namespace openMVG::sfm;


void writeToMatlab(const BundleAdjustmentData &ba_data, const mxArray *pts, 
        const mxArray *views, const mxArray *runtime, const double ellapsed_secs)
{
    double *runtimePtr;
    runtimePtr = mxGetPr(runtime);
    *runtimePtr = ellapsed_secs;
    
    mxArray *viewArr;
    double *viewPtr;
    
    for (const auto &pose_it : ba_data.poses)
    {
        const IndexT indexPose = pose_it.first;
        
        viewArr = mxGetCell(views, indexPose);
        viewPtr = mxGetPr(viewArr);
        
        const Pose3 &pose = pose_it.second;
        const Mat3 r = pose.rotation();
        const Vec3 t = pose.translation();
        
        // create extrinsics
        //const Mat3 r = camOrientation.transpose();
        //const Vec3 t = -r*camCentre;
        
        std::copy(r.data(),r.data()+9,viewPtr);
        std::copy(t.data(),t.data()+3,viewPtr+9);
        
        //std::cout<<" t = " << t<<std::endl;
    }


    double *x;
    x = mxGetPr(pts);
    
    for (auto & structure_landmark_it : ba_data.structure)
    {
        const IndexT indexPt = structure_landmark_it.first;
        const Vec3 v = structure_landmark_it.second.X;
        std::copy(v.data(), v.data()+3, x+4*indexPt);
        x[4*indexPt+3] = 1.0;
    }
}




// sd: standard deviation for the Huber loss
bool bundleAdjustment(const Optimize_Options &ba_refine_options, 
        const mxArray *obs,
        const mxArray *pts, const mxArray *cams, 
        double sd,
        mxArray *outPts, mxArray *outCams, mxArray *outRuntime)
{
    
    BundleAdjustmentData ba_data(obs, pts, cams);
    
    
    Bundle_Adjustment_Ceres_MEX::BA_Ceres_options options;
    if ( ba_data.numberOfViews() > 100 &&  //if ( ba_data.getPoses().size() > 100 &&
            (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE) ||
            ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CX_SPARSE) ||
            ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE))
            )
        // Enable sparse BA only if a sparse lib is available and if there more than 100 poses
    {
        options.preconditioner_type_ = ceres::JACOBI;
        options.linear_solver_type_ = ceres::SPARSE_SCHUR;
    }
    else
    {
        options.linear_solver_type_ = ceres::DENSE_SCHUR;
   }
    
    
    Bundle_Adjustment_Ceres_MEX bundle_adjustment_obj(options);
    
    //std::clock_t begin = std::clock();
    double ellapsed_seconds;
        
    bool resp = bundle_adjustment_obj.AdjustData(ba_data, ba_refine_options, sd, ellapsed_seconds);
    
    //std::clock_t end = std::clock();
     //double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
     std::cout<<"time = "<< ellapsed_seconds<< std::endl;
    
     //TODO: unlock
    if (resp)
        writeToMatlab(ba_data, outPts, outCams, outRuntime, ellapsed_seconds);
    
    return resp;
}


// Input
#define IN_OBS  prhs[0] // u: 1xm cell array of observations in camera coordinates
#define IN_U    prhs[1] // U: 4xn matrix of point coordinates in homogenous coordinates
#define IN_P    prhs[2] // P: cell array of projection matrices [R|t]
#define IN_SD   prhs[3] // hubber standard deviation
#define IN_OPTS prhs[4] // options struct


// Output
#define OUT_U       plhs[0]
#define OUT_P       plhs[1]
#define OUT_RUNTIME plhs[2]

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Check mex-arguments
    if (nlhs !=3)
    {
        mexErrMsgTxt("Invalid number of output variables. Expected 3 LHS vars.");
    }
    
    if (nrhs != 4 && nrhs != 5)
    {
        mexErrMsgTxt("Invalid number of parameters. Expected 4 or 5 parameters.");
    }
    
    if (mxGetM(IN_U) != 4)
    {
        mexErrMsgTxt("Invalid input. Expected a 4xn matrix");
    }
    
    const int n = mxGetN(IN_U);
    const int m = mxGetN(IN_OBS);
    
    std::cout<<"n = "<<n <<std::endl;
    std::cout<<"m = "<<m <<std::endl;
    
//     if (m<2)
//     {
//         mexErrMsgTxt("Too few cameras. Expected  m>1\n");
//     }
    
    // Allocate output
    OUT_U = mxCreateDoubleMatrix(4, n, mxREAL);
    OUT_P = mxCreateCellMatrix(1, m);
    OUT_RUNTIME = mxCreateDoubleMatrix(1, 1, mxREAL);
    
    for (mwIndex i=0; i<m; i++)
    {
        mxSetCell(OUT_P, i, mxCreateDoubleMatrix(3, 4, mxREAL));
    }
    
    int num_iterations = 500; //, "Number of iterations.");
    
    Extrinsic_Parameter_Type extrinsics = Extrinsic_Parameter_Type::ADJUST_ALL;
    Structure_Parameter_Type structure = Structure_Parameter_Type::ADJUST_ALL;
    
    double sd =3; //huber loss
    
    // parse options
    if (nrhs==5)
    {
        const int nfields = mxGetNumberOfFields(IN_OPTS);
        for(int ifield=0; ifield<nfields; ifield++)
        {
            mxArray *tmpFiled;
            const char *fieldName = mxGetFieldNameByNumber(IN_OPTS, ifield);
            std::cout<<fieldName<<std::endl;
            
            if (strcmp("num_iterations",fieldName)==0)
            {
                tmpFiled = mxGetFieldByNumber(IN_OPTS, 0, ifield);
                num_iterations = *mxGetPr(tmpFiled);
                //std::cout<<"num_iterations = "<<num_iterations<<std::endl;
            }
            else if (strcmp("extrinsic_type",fieldName)==0)
            {
                tmpFiled = mxGetFieldByNumber(IN_OPTS, 0, ifield);
                char *extopt = mxArrayToString(tmpFiled);
                if (strcmp(extopt, "rotation_only")==0)
                {
                    std::cout<<"setting rotation only "<<std::endl;
                    extrinsics = Extrinsic_Parameter_Type::ADJUST_ROTATION;
                }
                else if (strcmp(extopt, "translation_only")==0)
                {
                    std::cout<<"setting translation only "<<std::endl;
                    extrinsics = Extrinsic_Parameter_Type::ADJUST_TRANSLATION;
                }
                else if (strcmp(extopt, "all")==0)
                {
                    std::cout<<"setting translation and rotation "<<std::endl;
                    extrinsics = Extrinsic_Parameter_Type::ADJUST_ALL;
                }
                else if (strcmp(extopt, "none")==0)
                {
                    std::cout<<"no adjusting extrinsics "<<std::endl;
                    extrinsics = Extrinsic_Parameter_Type::NONE;
                }
                else
                {
                    mexErrMsgTxt("extrinsic value must be: rotation_only, translation_only, all, none");
                }
                
                //num_iterations = *mxGetPr(tmpFiled);
                //std::cout<<"num_iterations = "<<num_iterations<<std::endl;
            }
            else if (strcmp("structure_type",fieldName)==0)
            {
                tmpFiled = mxGetFieldByNumber(IN_OPTS, 0, ifield);
                char *extopt = mxArrayToString(tmpFiled);
                if (strcmp(extopt, "all")==0)
                {
                    //std::cout<<"adjusting structure "<<std::endl;
                    structure = Structure_Parameter_Type::ADJUST_ALL;
                }
                else if (strcmp(extopt, "none")==0)
                {
                    //std::cout<<"no adjusting structure "<<std::endl;
                    structure = Structure_Parameter_Type::NONE;
                }
                else
                {
                    mexErrMsgTxt("structure value must be: all, none");
                }
                
                //std::cout<<"num_iterations = "<<num_iterations<<std::endl;
            }
        }
        
        //read standard deviation
        sd = *mxGetPr(IN_SD);
       // std::cout<< "set hubber sd = " <<sd<<std::endl;
    }
    
    
//    const cameras::Intrinsic_Parameter_Type intrinsics = cameras::Intrinsic_Parameter_Type::ADJUST_ALL,
//     const Control_Point_Parameter & control_point = Control_Point_Parameter(0.0, false), // Default setting does not use GCP in the BA
//     const bool use_motion_priors = false
    
    const Optimize_Options refine_options( 
            cameras::Intrinsic_Parameter_Type::NONE, 
            extrinsics, // Adjust camera motion
            structure, // Adjust scene structure
            Control_Point_Parameter(),
            false);
    
    //std::cout<< "calling ba..." <<std::endl;
    
    bundleAdjustment(refine_options, IN_OBS, IN_U, IN_P, sd, OUT_U, OUT_P, OUT_RUNTIME);
    
}
