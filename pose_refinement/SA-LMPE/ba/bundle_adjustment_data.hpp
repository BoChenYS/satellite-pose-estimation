#ifndef BUNDLE_ADJUSTMENT_DATA_HPP
#define BUNDLE_ADJUSTMENT_DATA_HPP

#include "openMVG/cameras/Camera_Intrinsics.hpp"
#include "openMVG/cameras/Camera_Pinhole.hpp"

#include "openMVG/types.hpp"
#include "openMVG/geometry/pose3.hpp"
#include "openMVG/sfm/sfm_landmark.hpp"

#include "mex.h"

namespace openMVG {
namespace sfm {
        
// Store bundle adjustment data
struct BundleAdjustmentData
{
    //poses
    using Poses = Hash_Map<IndexT, geometry::Pose3>;
    
    // Define a collection of IntrinsicParameter (indexed by View::id_intrinsic)
    using Intrinsics = Hash_Map<IndexT, std::shared_ptr<cameras::Pinhole_Intrinsic>>;

    Poses poses;
    
    // Structure (3D points with their 2D observations)
    Landmarks structure;
    
    // Considered camera intrinsics (indexed by view.id_intrinsic)
    Intrinsics intrinsics;
        
    BundleAdjustmentData(const mxArray *obs, const mxArray *pts, const mxArray *cams);
    
    int numberOfViews() const { return poses.size(); }
    int numberOfPoints() const { return structure.size(); }
    
    ~BundleAdjustmentData();
};

} //sfm
} //openMVG
    
#endif // BUNDLE_ADJUSTMENT_DATA_HPP
