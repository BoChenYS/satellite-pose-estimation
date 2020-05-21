#include "bundle_adjustment_data.hpp"
#include "openMVG/cameras/Camera_Pinhole.hpp"

namespace openMVG {
namespace sfm {
        
        
/// Generic SfM data container
/// Store structure and camera properties:
BundleAdjustmentData::BundleAdjustmentData(
        const mxArray *obs, const mxArray *pts, const mxArray *cams)
{
    // read poses
    const mwIndex numberOfCameras = mxGetN(cams);
    
    double *p;
    mxArray *camP;
    
    for (mwIndex i=0; i<numberOfCameras; i++)
    {
        camP = mxGetCell(cams, i);
        if (mxGetM(camP)!=3 || mxGetN(camP)!=4)
        {
            mexErrMsgTxt("Expected P to be 3x4 matrix");
        }
        p = mxGetPr(camP);
        
        Eigen::Map<Mat3> r(p); 
        Eigen::Map<Vec3> t(p+9);
        
        //Mat3 orientation = r.transpose();
        Vec3 c = -r.transpose()*t;
        
        geometry::Pose3 pose(r,c);
        poses[i] = pose;
    }
    
    //-------------------------
    // read structure
    const mwIndex numberOfPoints = mxGetN(pts);
    mxArray *camObs;  //Camera observations
    
    p = mxGetPr(pts);
    for (mwIndex j=0; j<numberOfPoints; j++)
    {
        Landmark landmark;
        landmark.X = Eigen::Map<Vec3> (p + j*4);
        // --- read observations for visited point
        bool landmark_is_visile = false;
        double *u;
        for (mwIndex i=0; i<numberOfCameras; i++)
        {
            camObs = mxGetCell(obs, i);
            u = mxGetPr(camObs);
            
            // obtain stride using mx
            // add existing observaitons for point j
            if (!mxIsNaN(u[3*j]))
            {
                Vec2 imagePoint = Eigen::Map<Vec2> (u + 3*j);
                landmark.obs[i] = Observation(imagePoint, 0); //set index of the image point as 0...
                landmark_is_visile = true;
            }
        }
        
        //TODO: assuming lanmark is seen in at least one camera!!!
        structure[j] = landmark;
    }
    
    
    //intrinsics[0] = std::make_shared<cameras::Pinhole_Intrinsic>(cameras::Pinhole_Intrinsic(0.0, 0.0, 1059.718895, 964.44269, 606.17167));
//     intrinsics[0] = std::make_shared<cameras::Pinhole_Intrinsic>(cameras::Pinhole_Intrinsic(0.0, 0.0, 
//             3003.41296928327665, 960, 600));
        // iphone
    //intrinsics[0] = std::make_shared<cameras::Pinhole_Intrinsic>(cameras::Pinhole_Intrinsic(0.0, 0.0, fx, cx, cy));


    intrinsics[0] = std::make_shared<cameras::Pinhole_Intrinsic>(cameras::Pinhole_Intrinsic(0.0, 0.0, 
            3003.41296928327665, 960, 600));
}
    

BundleAdjustmentData::~BundleAdjustmentData()
{}

} //mex

} //openMVG
    
