///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2018 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

/// @file math/Maps.h

#ifndef OPENVDB_MATH_MAPS_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_MAPS_HAS_BEEN_INCLUDED

#include "Math.h"
#include "Mat4.h"
#include "Vec3.h"
#include "BBox.h"
#include "Coord.h"
#include <openvdb/io/io.h> // for io::getFormatVersion()
#include <openvdb/util/Name.h>
#include <openvdb/Types.h>
#include <cmath> // for std::abs()
#include <iostream>
#include <map>
#include <string>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {


////////////////////////////////////////

/// Forward declarations of the different map types

class MapBase;
class ScaleMap;
class TranslationMap;
class ScaleTranslateMap;
class UniformScaleMap;
class UniformScaleTranslateMap;
class AffineMap;
class UnitaryMap;
class NonlinearFrustumMap;

template<typename T1, typename T2> class CompoundMap;

using UnitaryAndTranslationMap = CompoundMap<UnitaryMap, TranslationMap>;
using SpectralDecomposedMap    = CompoundMap<CompoundMap<UnitaryMap, ScaleMap>, UnitaryMap>;
using SymmetricMap             = SpectralDecomposedMap;
using FullyDecomposedMap       = CompoundMap<SymmetricMap, UnitaryAndTranslationMap>;
using PolarDecomposedMap       = CompoundMap<SymmetricMap, UnitaryMap>;


////////////////////////////////////////

/// Map traits

template<typename T> struct is_linear                 { static const bool value = false; };
template<> struct is_linear<AffineMap>                { static const bool value = true; };
template<> struct is_linear<ScaleMap>                 { static const bool value = true; };
template<> struct is_linear<UniformScaleMap>          { static const bool value = true; };
template<> struct is_linear<UnitaryMap>               { static const bool value = true; };
template<> struct is_linear<TranslationMap>           { static const bool value = true; };
template<> struct is_linear<ScaleTranslateMap>        { static const bool value = true; };
template<> struct is_linear<UniformScaleTranslateMap> { static const bool value = true; };

template<typename T1, typename T2> struct is_linear<CompoundMap<T1, T2> > {
    static const bool value = is_linear<T1>::value && is_linear<T2>::value;
};


template<typename T> struct is_uniform_scale          { static const bool value = false; };
template<> struct is_uniform_scale<UniformScaleMap>   { static const bool value = true; };

template<typename T> struct is_uniform_scale_translate       { static const bool value = false; };
template<> struct is_uniform_scale_translate<TranslationMap> { static const bool value = true; };
template<> struct is_uniform_scale_translate<UniformScaleTranslateMap> {
    static const bool value = true;
};


template<typename T> struct is_scale                  { static const bool value = false; };
template<> struct is_scale<ScaleMap>                  { static const bool value = true; };

template<typename T> struct is_scale_translate        { static const bool value = false; };
template<> struct is_scale_translate<ScaleTranslateMap> { static const bool value = true; };


template<typename T> struct is_uniform_diagonal_jacobian {
    static const bool value = is_uniform_scale<T>::value || is_uniform_scale_translate<T>::value;
};

template<typename T> struct is_diagonal_jacobian {
    static const bool value = is_scale<T>::value || is_scale_translate<T>::value;
};


////////////////////////////////////////

/// Utility methods

/// @brief Create a SymmetricMap from a symmetric matrix.
/// Decomposes the map into Rotation Diagonal Rotation^T
OPENVDB_API SharedPtr<SymmetricMap> createSymmetricMap(const Mat3d& m);


/// @brief General decomposition of a Matrix into a Unitary (e.g. rotation)
/// following a Symmetric (e.g. stretch & shear)
OPENVDB_API SharedPtr<FullyDecomposedMap> createFullyDecomposedMap(const Mat4d& m);


/// @brief Decomposes a general linear into translation following polar decomposition.
///
/// T U S where:
///
///  T: Translation
///  U: Unitary (rotation or reflection)
///  S: Symmetric
///
/// @note: the Symmetric is automatically decomposed into Q D Q^T, where
/// Q is rotation and D is diagonal.
OPENVDB_API SharedPtr<PolarDecomposedMap> createPolarDecomposedMap(const Mat3d& m);


/// @brief reduces an AffineMap to a ScaleMap or a ScaleTranslateMap when it can
OPENVDB_API SharedPtr<MapBase> simplify(SharedPtr<AffineMap> affine);

/// @brief Returns the left pseudoInverse of the input matrix when the 3x3 part is symmetric
/// otherwise it zeros the 3x3 and reverses the translation.
OPENVDB_API Mat4d approxInverse(const Mat4d& mat);


////////////////////////////////////////


/// @brief Abstract base class for maps
class OPENVDB_API MapBase
{
public:
    using Ptr = SharedPtr<MapBase>;
    using ConstPtr = SharedPtr<const MapBase>;
    using MapFactory = Ptr (*)();

    MapBase(const MapBase&) = default;
    virtual ~MapBase() = default;

    virtual SharedPtr<AffineMap> getAffineMap() const = 0;

    /// Return the name of this map's concrete type (e.g., @c "AffineMap").
    virtual Name type() const = 0;

    /// Return @c true if this map is of concrete type @c MapT (e.g., AffineMap).
    template<typename MapT> bool isType() const { return this->type() == MapT::mapType(); }

    /// Return @c true if this map is equal to the given map.
    virtual bool isEqual(const MapBase& other) const = 0;

    /// Return @c true if this map is linear.
    virtual bool isLinear() const = 0;
    /// Return @c true if the spacing between the image of latice is uniform in all directions
    virtual bool hasUniformScale() const = 0;

    virtual Vec3d applyMap(const Vec3d& in) const = 0;
    virtual Vec3d applyInverseMap(const Vec3d& in) const = 0;

    //@{
    /// @brief Apply the Inverse Jacobian Transpose of this map to a vector.
    /// For a linear map this is equivalent to applying the transpose of
    /// inverse map excluding translation.
    virtual Vec3d applyIJT(const Vec3d& in) const = 0;
    virtual Vec3d applyIJT(const Vec3d& in, const Vec3d& domainPos) const = 0;
    //@}

    virtual Mat3d applyIJC(const Mat3d& m) const = 0;
    virtual Mat3d applyIJC(const Mat3d& m, const Vec3d& v, const Vec3d& domainPos) const = 0;


    virtual double determinant() const = 0;
    virtual double determinant(const Vec3d&) const = 0;


    //@{
    /// @brief Method to return the local size of a voxel.
    /// When a location is specified as an argument, it is understood to be
    /// be in the domain of the map (i.e. index space)
    virtual Vec3d voxelSize() const = 0;
    virtual Vec3d voxelSize(const Vec3d&) const = 0;
    //@}

    virtual void read(std::istream&) = 0;
    virtual void write(std::ostream&) const = 0;

    virtual std::string str() const = 0;

    virtual MapBase::Ptr copy() const = 0;

    //@{
    /// @brief Methods to update the map
    virtual MapBase::Ptr preRotate(double radians, Axis axis = X_AXIS) const = 0;
    virtual MapBase::Ptr preTranslate(const Vec3d&) const = 0;
    virtual MapBase::Ptr preScale(const Vec3d&) const = 0;
    virtual MapBase::Ptr preShear(double shear, Axis axis0, Axis axis1) const = 0;

    virtual MapBase::Ptr postRotate(double radians, Axis axis = X_AXIS) const = 0;
    virtual MapBase::Ptr postTranslate(const Vec3d&) const = 0;
    virtual MapBase::Ptr postScale(const Vec3d&) const = 0;
    virtual MapBase::Ptr postShear(double shear, Axis axis0, Axis axis1) const = 0;
    //@}

    //@{
    /// @brief Apply the Jacobian of this map to a vector.
    /// For a linear map this is equivalent to applying the map excluding translation.
    /// @warning Houdini 12.5 uses an earlier version of OpenVDB, and maps created
    /// with that version lack a virtual table entry for this method.  Do not call
    /// this method from Houdini 12.5.
    virtual Vec3d applyJacobian(const Vec3d& in) const = 0;
    virtual Vec3d applyJacobian(const Vec3d& in, const Vec3d& domainPos) const = 0;
    //@}

    //@{
    /// @brief Apply the InverseJacobian of this map to a vector.
    /// For a linear map this is equivalent to applying the map inverse excluding translation.
    /// @warning Houdini 12.5 uses an earlier version of OpenVDB, and maps created
    /// with that version lack a virtual table entry for this method.  Do not call
    /// this method from Houdini 12.5.
    virtual Vec3d applyInverseJacobian(const Vec3d& in) const = 0;
    virtual Vec3d applyInverseJacobian(const Vec3d& in, const Vec3d& domainPos) const = 0;
    //@}


    //@{
    /// @brief Apply the Jacobian transpose of this map to a vector.
    /// For a linear map this is equivalent to applying the transpose of the map
    /// excluding translation.
    /// @warning Houdini 12.5 uses an earlier version of OpenVDB, and maps created
    /// with that version lack a virtual table entry for this method.  Do not call
    /// this method from Houdini 12.5.
    virtual Vec3d applyJT(const Vec3d& in) const = 0;
    virtual Vec3d applyJT(const Vec3d& in, const Vec3d& domainPos) const = 0;
    //@}

    /// @brief Return a new map representing the inverse of this map.
    /// @throw NotImplementedError if the map is a NonlinearFrustumMap.
    /// @warning Houdini 12.5 uses an earlier version of OpenVDB, and maps created
    /// with that version lack a virtual table entry for this method.  Do not call
    /// this method from Houdini 12.5.
    virtual MapBase::Ptr inverseMap() const = 0;

protected:
    MapBase() {}

    template<typename MapT>
    static bool isEqualBase(const MapT& self, const MapBase& other)
    {
        return other.isType<MapT>() && (self == *static_cast<const MapT*>(&other));
    }
};


////////////////////////////////////////


/// @brief Threadsafe singleton object for accessing the map type-name dictionary.
/// Associates a map type-name with a factory function.
class OPENVDB_API MapRegistry
{
public:
    using MapDictionary = std::map<Name, MapBase::MapFactory>;

    static MapRegistry* instance();

    /// Create a new map of the given (registered) type name.
    static MapBase::Ptr createMap(const Name&);

    /// Return @c true if the given map type name is registered.
    static bool isRegistered(const Name&);

    /// Register a map type along with a factory function.
    static void registerMap(const Name&, MapBase::MapFactory);

    /// Remove a map type from the registry.
    static void unregisterMap(const Name&);

    /// Clear the map type registry.
    static void clear();

private:
    MapRegistry() {}

    static MapRegistry* staticInstance();

    static MapRegistry* mInstance;

    MapDictionary mMap;
};


////////////////////////////////////////


/// @brief A general linear transform using homogeneous coordinates to perform
/// rotation, scaling, shear and translation
class OPENVDB_API AffineMap: public MapBase
{
public:
    using Ptr = SharedPtr<AffineMap>;
    using ConstPtr = SharedPtr<const AffineMap>;

    AffineMap():
        mMatrix(Mat4d::identity()),
        mMatrixInv(Mat4d::identity()),
        mJacobianInv(Mat3d::identity()),
        mDeterminant(1),
        mVoxelSize(Vec3d(1,1,1)),
        mIsDiagonal(true),
        mIsIdentity(true)
        // the default constructor for translation is zero
    {
    }

    AffineMap(const Mat3d& m)
    {
        Mat4d mat4(Mat4d::identity());
        mat4.setMat3(m);
        mMatrix = mat4;
        updateAcceleration();
    }

    AffineMap(const Mat4d& m): mMatrix(m)
    {
        if (!isAffine(m)) {
            OPENVDB_THROW(ArithmeticError,
                "Tried to initialize an affine transform from a non-affine 4x4 matrix");
        }
        updateAcceleration();
    }

    AffineMap(const AffineMap& other):
        MapBase(other),
        mMatrix(other.mMatrix),
        mMatrixInv(other.mMatrixInv),
        mJacobianInv(other.mJacobianInv),
        mDeterminant(other.mDeterminant),
        mVoxelSize(other.mVoxelSize),
        mIsDiagonal(other.mIsDiagonal),
        mIsIdentity(other.mIsIdentity)
    {
    }

    /// @brief constructor that merges the matrixes for two affine maps
    AffineMap(const AffineMap& first, const AffineMap& second):
        mMatrix(first.mMatrix * second.mMatrix)
    {
        updateAcceleration();
    }

    ~AffineMap() override = default;

    /// Return a MapBase::Ptr to a new AffineMap
    static MapBase::Ptr create() { return MapBase::Ptr(new AffineMap()); }
    /// Return a MapBase::Ptr to a deep copy of this map
    MapBase::Ptr copy() const override { return MapBase::Ptr(new AffineMap(*this)); }

    MapBase::Ptr inverseMap() const override { return MapBase::Ptr(new AffineMap(mMatrixInv)); }

    static bool isRegistered() { return MapRegistry::isRegistered(AffineMap::mapType()); }

    static void registerMap()
    {
        MapRegistry::registerMap(
            AffineMap::mapType(),
            AffineMap::create);
    }

    Name type() const override { return mapType(); }
    static Name mapType() { return Name("AffineMap"); }

    /// Return @c true (an AffineMap is always linear).
    bool isLinear() const override { return true; }

    /// Return @c false ( test if this is unitary with translation )
    bool hasUniformScale() const override
    {
        Mat3d mat = mMatrix.getMat3();
        const double det = mat.det();
        if (isApproxEqual(det, double(0))) {
            return false;
        } else {
            mat *= (1.0 / pow(std::abs(det), 1.0/3.0));
            return isUnitary(mat);
        }
    }

    bool isEqual(const MapBase& other) const override { return isEqualBase(*this, other); }

    bool operator==(const AffineMap& other) const
    {
        // the Mat.eq() is approximate
        if (!mMatrix.eq(other.mMatrix)) { return false; }
        if (!mMatrixInv.eq(other.mMatrixInv))  { return false; }
        return true;
    }

    bool operator!=(const AffineMap& other) const { return !(*this == other); }

    AffineMap& operator=(const AffineMap& other)
    {
        mMatrix = other.mMatrix;
        mMatrixInv = other.mMatrixInv;

        mJacobianInv = other.mJacobianInv;
        mDeterminant = other.mDeterminant;
        mVoxelSize = other.mVoxelSize;
        mIsDiagonal  = other.mIsDiagonal;
        mIsIdentity  = other.mIsIdentity;
        return *this;
    }
    /// Return the image of @c in under the map
    Vec3d applyMap(const Vec3d& in) const override { return in * mMatrix; }
    /// Return the pre-image of @c in under the map
    Vec3d applyInverseMap(const Vec3d& in) const override {return in * mMatrixInv; }

    /// Return the Jacobian of the map applied to @a in.
    Vec3d applyJacobian(const Vec3d& in, const Vec3d&) const override { return applyJacobian(in); }
    /// Return the Jacobian of the map applied to @a in.
    Vec3d applyJacobian(const Vec3d& in) const override { return mMatrix.transform3x3(in); }

    /// @brief Return the Inverse Jacobian of the map applied to @a in
    /// (i.e. inverse map with out translation)
    Vec3d applyInverseJacobian(const Vec3d& in, const Vec3d&) const override {
        return applyInverseJacobian(in);
    }
    /// @brief Return the Inverse Jacobian of the map applied to @a in
    /// (i.e. inverse map with out translation)
    Vec3d applyInverseJacobian(const Vec3d& in) const override {
        return mMatrixInv.transform3x3(in);
    }

    /// Return the Jacobian Transpose of the map applied to @a in.
    /// This tranforms range-space gradients to domain-space gradients
    Vec3d applyJT(const Vec3d& in, const Vec3d&) const override { return applyJT(in); }
    /// Return the Jacobian Transpose of the map applied to @a in.
    Vec3d applyJT(const Vec3d& in) const override {
        const double* m = mMatrix.asPointer();
        return Vec3d( m[ 0] * in[0] + m[ 1] * in[1] + m[ 2] * in[2],
                      m[ 4] * in[0] + m[ 5] * in[1] + m[ 6] * in[2],
                      m[ 8] * in[0] + m[ 9] * in[1] + m[10] * in[2] );
    }

    /// Return the transpose of the inverse Jacobian of the map applied to @a in.
    Vec3d applyIJT(const Vec3d& in, const Vec3d&) const override { return applyIJT(in); }
    /// Return the transpose of the inverse Jacobian of the map applied to @c in
    Vec3d applyIJT(const Vec3d& in) const override { return in * mJacobianInv; }
    /// Return the Jacobian Curvature: zero for a linear map
    Mat3d applyIJC(const Mat3d& m) const override {
        return mJacobianInv.transpose()* m * mJacobianInv;
    }
    Mat3d applyIJC(const Mat3d& in, const Vec3d& , const Vec3d& ) const override {
        return applyIJC(in);
    }
    /// Return the determinant of the Jacobian, ignores argument
    double determinant(const Vec3d& ) const override { return determinant(); }
    /// Return the determinant of the Jacobian
    double determinant() const override { return mDeterminant; }

    //@{
    /// @brief Return the lengths of the images of the segments
    /// (0,0,0)-(1,0,0), (0,0,0)-(0,1,0) and (0,0,0)-(0,0,1).
    Vec3d voxelSize() const override { return mVoxelSize; }
    Vec3d voxelSize(const Vec3d&) const override { return voxelSize(); }
    //@}

    /// Return @c true if the underlying matrix is approximately an identity
    bool isIdentity() const { return mIsIdentity; }
    /// Return @c true  if the underylying matrix is diagonal
    bool isDiagonal() const { return mIsDiagonal; }
    /// Return @c true if the map is equivalent to a ScaleMap
    bool isScale() const { return isDiagonal(); }
    /// Return @c true if the map is equivalent to a ScaleTranslateMap
    bool isScaleTranslate() const { return math::isDiagonal(mMatrix.getMat3()); }


    // Methods that modify the existing affine map

    //@{
    /// @brief Modify the existing affine map by pre-applying the given operation.
    void accumPreRotation(Axis axis, double radians)
    {
        mMatrix.preRotate(axis, radians);
        updateAcceleration();
    }
    void accumPreScale(const Vec3d& v)
    {
        mMatrix.preScale(v);
        updateAcceleration();
    }
    void accumPreTranslation(const Vec3d& v)
    {
        mMatrix.preTranslate(v);
        updateAcceleration();
    }
    void accumPreShear(Axis axis0, Axis axis1, double shear)
    {
        mMatrix.preShear(axis0, axis1, shear);
        updateAcceleration();
    }
    //@}


    //@{
    /// @brief Modify the existing affine map by post-applying the given operation.
    void accumPostRotation(Axis axis, double radians)
    {
        mMatrix.postRotate(axis, radians);
        updateAcceleration();
    }
    void accumPostScale(const Vec3d& v)
    {
        mMatrix.postScale(v);
        updateAcceleration();
    }
    void accumPostTranslation(const Vec3d& v)
    {
        mMatrix.postTranslate(v);
        updateAcceleration();
    }
    void accumPostShear(Axis axis0, Axis axis1, double shear)
    {
        mMatrix.postShear(axis0, axis1, shear);
        updateAcceleration();
    }
    //@}


    /// read serialization
    void read(std::istream& is) override { mMatrix.read(is); updateAcceleration(); }
    /// write serialization
    void write(std::ostream& os) const override { mMatrix.write(os); }
    /// string serialization, useful for debugging
    std::string str() const override
    {
        std::ostringstream buffer;
        buffer << " - mat4:\n" << mMatrix.str() << std::endl;
        buffer << " - voxel dimensions: " << mVoxelSize << std::endl;
        return buffer.str();
    }

    /// on-demand decomposition of the affine map
    SharedPtr<FullyDecomposedMap> createDecomposedMap()
    {
        return createFullyDecomposedMap(mMatrix);
    }

    /// Return AffineMap::Ptr to  a deep copy of the current AffineMap
    AffineMap::Ptr getAffineMap() const override { return AffineMap::Ptr(new AffineMap(*this)); }

    /// Return AffineMap::Ptr to the inverse of this map
    AffineMap::Ptr inverse() const { return AffineMap::Ptr(new AffineMap(mMatrixInv)); }


    //@{
    /// @brief  Return a MapBase::Ptr to a new map that is the result
    /// of prepending the appropraite operation.
    MapBase::Ptr preRotate(double radians, Axis axis = X_AXIS) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPreRotation(axis, radians);
        return simplify(affineMap);
    }
    MapBase::Ptr preTranslate(const Vec3d& t) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPreTranslation(t);
        return StaticPtrCast<MapBase, AffineMap>(affineMap);
    }
    MapBase::Ptr preScale(const Vec3d& s) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPreScale(s);
        return StaticPtrCast<MapBase, AffineMap>(affineMap);
    }
    MapBase::Ptr preShear(double shear, Axis axis0, Axis axis1) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPreShear(axis0, axis1, shear);
        return simplify(affineMap);
    }
    //@}


    //@{
    /// @brief  Return a MapBase::Ptr to a new map that is the result
    /// of postfixing the appropraite operation.
    MapBase::Ptr postRotate(double radians, Axis axis = X_AXIS) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPostRotation(axis, radians);
        return simplify(affineMap);
    }
    MapBase::Ptr postTranslate(const Vec3d& t) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPostTranslation(t);
        return StaticPtrCast<MapBase, AffineMap>(affineMap);
    }
    MapBase::Ptr postScale(const Vec3d& s) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPostScale(s);
        return StaticPtrCast<MapBase, AffineMap>(affineMap);
    }
    MapBase::Ptr postShear(double shear, Axis axis0, Axis axis1) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPostShear(axis0, axis1, shear);
        return simplify(affineMap);
    }
    //@}

    /// Return the matrix representation of this AffineMap
    Mat4d getMat4() const { return mMatrix;}
    const Mat4d& getConstMat4() const {return mMatrix;}
    const Mat3d& getConstJacobianInv() const {return mJacobianInv;}

private:
    void updateAcceleration() {
        Mat3d mat3 = mMatrix.getMat3();
        mDeterminant = mat3.det();

        if (std::abs(mDeterminant) < (3.0 * math::Tolerance<double>::value())) {
            OPENVDB_THROW(ArithmeticError,
                "Tried to initialize an affine transform from a nearly singular matrix");
        }
        mMatrixInv = mMatrix.inverse();
        mJacobianInv = mat3.inverse().transpose();
        mIsDiagonal = math::isDiagonal(mMatrix);
        mIsIdentity = math::isIdentity(mMatrix);
        Vec3d pos = applyMap(Vec3d(0,0,0));
        mVoxelSize(0) = (applyMap(Vec3d(1,0,0)) - pos).length();
        mVoxelSize(1) = (applyMap(Vec3d(0,1,0)) - pos).length();
        mVoxelSize(2) = (applyMap(Vec3d(0,0,1)) - pos).length();
    }

    // the underlying matrix
    Mat4d  mMatrix;

    // stored for acceleration
    Mat4d  mMatrixInv;
    Mat3d  mJacobianInv;
    double mDeterminant;
    Vec3d  mVoxelSize;
    bool   mIsDiagonal, mIsIdentity;
}; // class AffineMap


////////////////////////////////////////


/// @brief A specialized Affine transform that scales along the principal axis
/// the scaling need not be uniform in the three-directions
class OPENVDB_API ScaleMap: public MapBase
{
public:
    using Ptr = SharedPtr<ScaleMap>;
    using ConstPtr = SharedPtr<const ScaleMap>;

    ScaleMap(): MapBase(), mScaleValues(Vec3d(1,1,1)), mVoxelSize(Vec3d(1,1,1)),
                mScaleValuesInverse(Vec3d(1,1,1)),
                mInvScaleSqr(1,1,1), mInvTwiceScale(0.5,0.5,0.5){}

    ScaleMap(const Vec3d& scale):
        MapBase(),
        mScaleValues(scale),
        mVoxelSize(Vec3d(std::abs(scale(0)),std::abs(scale(1)), std::abs(scale(2))))
    {
        double determinant = scale[0]* scale[1] * scale[2];
        if (std::abs(determinant) < 3.0 * math::Tolerance<double>::value()) {
            OPENVDB_THROW(ArithmeticError, "Non-zero scale values required");
        }
        mScaleValuesInverse = 1.0 / mScaleValues;
        mInvScaleSqr = mScaleValuesInverse * mScaleValuesInverse;
        mInvTwiceScale = mScaleValuesInverse / 2;
    }

    ScaleMap(const ScaleMap& other):
        MapBase(),
        mScaleValues(other.mScaleValues),
        mVoxelSize(other.mVoxelSize),
        mScaleValuesInverse(other.mScaleValuesInverse),
        mInvScaleSqr(other.mInvScaleSqr),
        mInvTwiceScale(other.mInvTwiceScale)
    {
    }

    ~ScaleMap() override = default;

    /// Return a MapBase::Ptr to a new ScaleMap
    static MapBase::Ptr create() { return MapBase::Ptr(new ScaleMap()); }
    /// Return a MapBase::Ptr to a deep copy of this map
    MapBase::Ptr copy() const override { return MapBase::Ptr(new ScaleMap(*this)); }

    MapBase::Ptr inverseMap() const override {
        return MapBase::Ptr(new ScaleMap(mScaleValuesInverse));
    }

    static bool isRegistered() { return MapRegistry::isRegistered(ScaleMap::mapType()); }

    static void registerMap()
    {
        MapRegistry::registerMap(
            ScaleMap::mapType(),
            ScaleMap::create);
    }

    Name type() const override { return mapType(); }
    static Name mapType() { return Name("ScaleMap"); }

    /// Return @c true (a ScaleMap is always linear).
    bool isLinear() const override { return true; }

    /// Return @c true if the values have the same magitude (eg. -1, 1, -1 would be a rotation).
    bool hasUniformScale() const override
    {
        bool value = isApproxEqual(
            std::abs(mScaleValues.x()), std::abs(mScaleValues.y()), double(5e-7));
        value = value && isApproxEqual(
            std::abs(mScaleValues.x()), std::abs(mScaleValues.z()), double(5e-7));
        return value;
    }

    /// Return the image of @c in under the map
    Vec3d applyMap(const Vec3d& in) const override
    {
        return Vec3d(
            in.x() * mScaleValues.x(),
            in.y() * mScaleValues.y(),
            in.z() * mScaleValues.z());
    }
    /// Return the pre-image of @c in under the map
    Vec3d applyInverseMap(const Vec3d& in) const override
    {
        return Vec3d(
            in.x() * mScaleValuesInverse.x(),
            in.y() * mScaleValuesInverse.y(),
            in.z() * mScaleValuesInverse.z());
    }
    /// Return the Jacobian of the map applied to @a in.
    Vec3d applyJacobian(const Vec3d& in, const Vec3d&) const override { return applyJacobian(in); }
    /// Return the Jacobian of the map applied to @a in.
    Vec3d applyJacobian(const Vec3d& in) const override { return applyMap(in); }

    /// @brief Return the Inverse Jacobian of the map applied to @a in
    /// (i.e. inverse map with out translation)
    Vec3d applyInverseJacobian(const Vec3d& in, const Vec3d&) const override {
        return applyInverseJacobian(in);
    }
    /// @brief Return the Inverse Jacobian of the map applied to @a in
    /// (i.e. inverse map with out translation)
    Vec3d applyInverseJacobian(const Vec3d& in) const override { return applyInverseMap(in); }

    /// @brief Return the Jacobian Transpose of the map applied to @a in.
    /// @details This tranforms range-space gradients to domain-space gradients
    Vec3d applyJT(const Vec3d& in, const Vec3d&) const override { return applyJT(in); }
    /// Return the Jacobian Transpose of the map applied to @a in.
    Vec3d applyJT(const Vec3d& in) const override { return applyMap(in); }

    /// @brief Return the transpose of the inverse Jacobian of the map applied to @a in.
    /// @details Ignores second argument
    Vec3d applyIJT(const Vec3d& in, const Vec3d&) const override { return applyIJT(in);}
    /// Return the transpose of the inverse Jacobian of the map applied to @c in
    Vec3d applyIJT(const Vec3d& in) const override { return applyInverseMap(in); }
    /// Return the Jacobian Curvature: zero for a linear map
    Mat3d applyIJC(const Mat3d& in) const override
    {
        Mat3d tmp;
        for (int i = 0; i < 3; i++) {
            tmp.setRow(i, in.row(i) * mScaleValuesInverse(i));
        }
        for (int i = 0; i < 3; i++) {
            tmp.setCol(i, tmp.col(i) * mScaleValuesInverse(i));
        }
        return tmp;
    }
    Mat3d applyIJC(const Mat3d& in, const Vec3d&, const Vec3d&) const override {
        return applyIJC(in);
    }
    /// Return the product of the scale values, ignores argument
    double determinant(const Vec3d&) const override { return determinant(); }
    /// Return the product of the scale values
    double determinant() const override {
        return mScaleValues.x() * mScaleValues.y() * mScaleValues.z();
    }

    /// Return the scale values that define the map
    const Vec3d& getScale() const {return mScaleValues;}

    /// Return the square of the scale.  Used to optimize some finite difference calculations
    const Vec3d& getInvScaleSqr() const { return mInvScaleSqr; }
    /// Return 1/(2 scale). Used to optimize some finite difference calculations
    const Vec3d& getInvTwiceScale() const { return mInvTwiceScale; }
    /// Return 1/(scale)
    const Vec3d& getInvScale() const { return mScaleValuesInverse; }

    //@{
    /// @brief Return the lengths of the images of the segments
    /// (0,0,0) &minus; 1,0,0), (0,0,0) &minus; (0,1,0) and (0,0,0) &minus; (0,0,1).
    /// @details This is equivalent to the absolute values of the scale values
    Vec3d voxelSize() const override { return mVoxelSize; }
    Vec3d voxelSize(const Vec3d&) const override { return voxelSize(); }
    //@}

    /// read serialization
    void read(std::istream& is) override
    {
        mScaleValues.read(is);
        mVoxelSize.read(is);
        mScaleValuesInverse.read(is);
        mInvScaleSqr.read(is);
        mInvTwiceScale.read(is);
    }
    /// write serialization
    void write(std::ostream& os) const override
    {
        mScaleValues.write(os);
        mVoxelSize.write(os);
        mScaleValuesInverse.write(os);
        mInvScaleSqr.write(os);
        mInvTwiceScale.write(os);
    }
    /// string serialization, useful for debuging
    std::string str() const override
    {
        std::ostringstream buffer;
        buffer << " - scale: " << mScaleValues << std::endl;
        buffer << " - voxel dimensions: " << mVoxelSize << std::endl;
        return buffer.str();
    }

    bool isEqual(const MapBase& other) const override { return isEqualBase(*this, other); }

    bool operator==(const ScaleMap& other) const
    {
        // ::eq() uses a tolerance
        if (!mScaleValues.eq(other.mScaleValues)) { return false; }
        return true;
    }

    bool operator!=(const ScaleMap& other) const { return !(*this == other); }

    /// Return a AffineMap equivalent to this map
    AffineMap::Ptr getAffineMap() const override
    {
        return AffineMap::Ptr(new AffineMap(math::scale<Mat4d>(mScaleValues)));
    }



    //@{
    /// @brief  Return a MapBase::Ptr to a new map that is the result
    /// of prepending the appropraite operation to the existing map
    MapBase::Ptr preRotate(double radians, Axis axis) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPreRotation(axis, radians);
        return simplify(affineMap);
    }

    MapBase::Ptr preTranslate(const Vec3d&) const override;
    MapBase::Ptr preScale(const Vec3d&) const override;
    MapBase::Ptr preShear(double shear, Axis axis0, Axis axis1) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPreShear(axis0, axis1, shear);
        return simplify(affineMap);
    }
    //@}


    //@{
    /// @brief  Return a MapBase::Ptr to a new map that is the result
    /// of prepending the appropraite operation to the existing map.
    MapBase::Ptr postRotate(double radians, Axis axis) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPostRotation(axis, radians);
        return simplify(affineMap);
    }
    MapBase::Ptr postTranslate(const Vec3d&) const override;
    MapBase::Ptr postScale(const Vec3d&) const override;
    MapBase::Ptr postShear(double shear, Axis axis0, Axis axis1) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPostShear(axis0, axis1, shear);
        return simplify(affineMap);
    }
    //@}

private:
    Vec3d mScaleValues, mVoxelSize, mScaleValuesInverse, mInvScaleSqr, mInvTwiceScale;
}; // class ScaleMap


/// @brief A specialized Affine transform that scales along the principal axis
/// the scaling is uniform in the three-directions
class OPENVDB_API UniformScaleMap: public ScaleMap
{
public:
    using Ptr = SharedPtr<UniformScaleMap>;
    using ConstPtr = SharedPtr<const UniformScaleMap>;

    UniformScaleMap(): ScaleMap(Vec3d(1,1,1)) {}
    UniformScaleMap(double scale): ScaleMap(Vec3d(scale, scale, scale)) {}
    UniformScaleMap(const UniformScaleMap& other): ScaleMap(other) {}
    ~UniformScaleMap() override = default;

    /// Return a MapBase::Ptr to a new UniformScaleMap
    static MapBase::Ptr create() { return MapBase::Ptr(new UniformScaleMap()); }
    /// Return a MapBase::Ptr to a deep copy of this map
    MapBase::Ptr copy() const override { return MapBase::Ptr(new UniformScaleMap(*this)); }

    MapBase::Ptr inverseMap() const override
    {
        const Vec3d& invScale = getInvScale();
        return MapBase::Ptr(new UniformScaleMap( invScale[0]));
    }

    static bool isRegistered() { return MapRegistry::isRegistered(UniformScaleMap::mapType()); }
    static void registerMap()
    {
        MapRegistry::registerMap(
            UniformScaleMap::mapType(),
            UniformScaleMap::create);
    }

    Name type() const override { return mapType(); }
    static Name mapType() { return Name("UniformScaleMap"); }

    bool isEqual(const MapBase& other) const override { return isEqualBase(*this, other); }

    bool operator==(const UniformScaleMap& other) const { return ScaleMap::operator==(other); }
    bool operator!=(const UniformScaleMap& other) const { return !(*this == other); }

    /// @brief Return a MapBase::Ptr to a UniformScaleTraslateMap that is the result of
    /// pre-translation on this map
    MapBase::Ptr preTranslate(const Vec3d&) const override;

    /// @brief Return a MapBase::Ptr to a UniformScaleTraslateMap that is the result of
    /// post-translation on this map
    MapBase::Ptr postTranslate(const Vec3d&) const override;

}; // class UniformScaleMap


////////////////////////////////////////


inline MapBase::Ptr
ScaleMap::preScale(const Vec3d& v) const
{
    const Vec3d new_scale(v * mScaleValues);
    if (isApproxEqual(new_scale[0],new_scale[1]) && isApproxEqual(new_scale[0],new_scale[2])) {
        return MapBase::Ptr(new UniformScaleMap(new_scale[0]));
    } else {
        return MapBase::Ptr(new ScaleMap(new_scale));
    }
}


inline MapBase::Ptr
ScaleMap::postScale(const Vec3d& v) const
{ // pre-post Scale are the same for a scale map
    return preScale(v);
}


/// @brief A specialized linear transform that performs a translation
class OPENVDB_API TranslationMap: public MapBase
{
public:
    using Ptr = SharedPtr<TranslationMap>;
    using ConstPtr = SharedPtr<const TranslationMap>;

    // default constructor is a translation by zero.
    TranslationMap(): MapBase(), mTranslation(Vec3d(0,0,0)) {}
    TranslationMap(const Vec3d& t): MapBase(), mTranslation(t) {}
    TranslationMap(const TranslationMap& other): MapBase(), mTranslation(other.mTranslation) {}

    ~TranslationMap() override = default;

    /// Return a MapBase::Ptr to a new TranslationMap
    static MapBase::Ptr create() { return MapBase::Ptr(new TranslationMap()); }
    /// Return a MapBase::Ptr to a deep copy of this map
    MapBase::Ptr copy() const override { return MapBase::Ptr(new TranslationMap(*this)); }

    MapBase::Ptr inverseMap() const override {
        return MapBase::Ptr(new TranslationMap(-mTranslation));
    }

    static bool isRegistered() { return MapRegistry::isRegistered(TranslationMap::mapType()); }

    static void registerMap()
    {
        MapRegistry::registerMap(
            TranslationMap::mapType(),
            TranslationMap::create);
    }

    Name type() const override { return mapType(); }
    static Name mapType() { return Name("TranslationMap"); }

    /// Return @c true (a TranslationMap is always linear).
    bool isLinear() const override { return true; }

    /// Return @c false (by convention true)
    bool hasUniformScale() const override { return true; }

    /// Return the image of @c in under the map
    Vec3d applyMap(const Vec3d& in) const override { return in + mTranslation; }
    /// Return the pre-image of @c in under the map
    Vec3d applyInverseMap(const Vec3d& in) const override { return in - mTranslation; }
    /// Return the Jacobian of the map applied to @a in.
    Vec3d applyJacobian(const Vec3d& in, const Vec3d&) const override { return applyJacobian(in); }
    /// Return the Jacobian of the map applied to @a in.
    Vec3d applyJacobian(const Vec3d& in) const override { return in; }

    /// @brief Return the Inverse Jacobian of the map applied to @a in
    /// (i.e. inverse map with out translation)
    Vec3d applyInverseJacobian(const Vec3d& in, const Vec3d&) const override {
        return applyInverseJacobian(in);
    }
    /// @brief Return the Inverse Jacobian of the map applied to @a in
    /// (i.e. inverse map with out translation)
    Vec3d applyInverseJacobian(const Vec3d& in) const override { return in; }


    /// @brief Return the Jacobian Transpose of the map applied to @a in.
    /// @details This tranforms range-space gradients to domain-space gradients
    Vec3d applyJT(const Vec3d& in, const Vec3d&) const override { return applyJT(in); }
    /// Return the Jacobian Transpose of the map applied to @a in.
    Vec3d applyJT(const Vec3d& in) const override { return in; }

    /// @brief Return the transpose of the inverse Jacobian (Identity for TranslationMap)
    /// of the map applied to @c in, ignores second argument
    Vec3d applyIJT(const Vec3d& in, const Vec3d& ) const override { return applyIJT(in);}
    /// @brief Return the transpose of the inverse Jacobian (Identity for TranslationMap)
    /// of the map applied to @c in
    Vec3d applyIJT(const Vec3d& in) const override {return in;}
    /// Return the Jacobian Curvature: zero for a linear map
    Mat3d applyIJC(const Mat3d& mat) const override {return mat;}
    Mat3d applyIJC(const Mat3d& mat, const Vec3d&, const Vec3d&) const override {
        return applyIJC(mat);
    }

    /// Return @c 1
    double determinant(const Vec3d& ) const override { return determinant(); }
    /// Return @c 1
    double determinant() const override { return 1.0; }

    /// Return (1,1,1).
    Vec3d voxelSize() const override { return Vec3d(1,1,1);}
    /// Return (1,1,1).
    Vec3d voxelSize(const Vec3d&) const override { return voxelSize();}

    /// Return the translation vector
    const Vec3d& getTranslation() const { return mTranslation; }

    /// read serialization
    void read(std::istream& is) override { mTranslation.read(is); }
    /// write serialization
    void write(std::ostream& os) const override { mTranslation.write(os); }
    /// string serialization, useful for debuging
    std::string str() const override
    {
        std::ostringstream buffer;
        buffer << " - translation: " << mTranslation << std::endl;
        return buffer.str();
    }

    bool isEqual(const MapBase& other) const override { return isEqualBase(*this, other); }

    bool operator==(const TranslationMap& other) const
    {
        // ::eq() uses a tolerance
        return mTranslation.eq(other.mTranslation);
    }

    bool operator!=(const TranslationMap& other) const { return !(*this == other); }

    /// Return AffineMap::Ptr to an AffineMap equivalent to *this
    AffineMap::Ptr getAffineMap() const override
    {
        Mat4d matrix(Mat4d::identity());
        matrix.setTranslation(mTranslation);

        AffineMap::Ptr affineMap(new AffineMap(matrix));
        return affineMap;
    }

    //@{
    /// @brief Return a MapBase::Ptr to a new map that is the result
    /// of prepending the appropriate operation.
    MapBase::Ptr preRotate(double radians, Axis axis) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPreRotation(axis, radians);
        return simplify(affineMap);

    }
    MapBase::Ptr preTranslate(const Vec3d& t) const override
    {
        return MapBase::Ptr(new TranslationMap(t + mTranslation));
    }

    MapBase::Ptr preScale(const Vec3d& v) const override;

    MapBase::Ptr preShear(double shear, Axis axis0, Axis axis1) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPreShear(axis0, axis1, shear);
        return simplify(affineMap);
    }
    //@}

    //@{
    /// @brief Return a MapBase::Ptr to a new map that is the result
    /// of postfixing the appropriate operation.
    MapBase::Ptr postRotate(double radians, Axis axis) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPostRotation(axis, radians);
        return simplify(affineMap);

    }
    MapBase::Ptr postTranslate(const Vec3d& t) const override
    { // post and pre are the same for this
        return MapBase::Ptr(new TranslationMap(t + mTranslation));
    }

    MapBase::Ptr postScale(const Vec3d& v) const override;

    MapBase::Ptr postShear(double shear, Axis axis0, Axis axis1) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPostShear(axis0, axis1, shear);
        return simplify(affineMap);
    }
    //@}

private:
    Vec3d mTranslation;
}; // class TranslationMap


////////////////////////////////////////


/// @brief A specialized Affine transform that scales along the principal axis
/// the scaling need not be uniform in the three-directions, and then
/// translates the result.
class OPENVDB_API ScaleTranslateMap: public MapBase
{
public:
    using Ptr = SharedPtr<ScaleTranslateMap>;
    using ConstPtr = SharedPtr<const ScaleTranslateMap>;

    ScaleTranslateMap():
        MapBase(),
        mTranslation(Vec3d(0,0,0)),
        mScaleValues(Vec3d(1,1,1)),
        mVoxelSize(Vec3d(1,1,1)),
        mScaleValuesInverse(Vec3d(1,1,1)),
        mInvScaleSqr(1,1,1),
        mInvTwiceScale(0.5,0.5,0.5)
    {
    }

    ScaleTranslateMap(const Vec3d& scale, const Vec3d& translate):
        MapBase(),
        mTranslation(translate),
        mScaleValues(scale),
        mVoxelSize(std::abs(scale(0)), std::abs(scale(1)), std::abs(scale(2)))
    {
        const double determinant = scale[0]* scale[1] * scale[2];
        if (std::abs(determinant) < 3.0 * math::Tolerance<double>::value()) {
            OPENVDB_THROW(ArithmeticError, "Non-zero scale values required");
        }
        mScaleValuesInverse = 1.0 / mScaleValues;
        mInvScaleSqr = mScaleValuesInverse * mScaleValuesInverse;
        mInvTwiceScale = mScaleValuesInverse / 2;
    }

    ScaleTranslateMap(const ScaleMap& scale, const TranslationMap& translate):
        MapBase(),
        mTranslation(translate.getTranslation()),
        mScaleValues(scale.getScale()),
        mVoxelSize(std::abs(mScaleValues(0)),
                         std::abs(mScaleValues(1)),
                         std::abs(mScaleValues(2))),
        mScaleValuesInverse(1.0 / scale.getScale())
    {
        mInvScaleSqr = mScaleValuesInverse * mScaleValuesInverse;
        mInvTwiceScale = mScaleValuesInverse / 2;
    }

    ScaleTranslateMap(const ScaleTranslateMap& other):
        MapBase(),
        mTranslation(other.mTranslation),
        mScaleValues(other.mScaleValues),
        mVoxelSize(other.mVoxelSize),
        mScaleValuesInverse(other.mScaleValuesInverse),
        mInvScaleSqr(other.mInvScaleSqr),
        mInvTwiceScale(other.mInvTwiceScale)
    {}

    ~ScaleTranslateMap() override = default;

    /// Return a MapBase::Ptr to a new ScaleTranslateMap
    static MapBase::Ptr create() { return MapBase::Ptr(new ScaleTranslateMap()); }
    /// Return a MapBase::Ptr to a deep copy of this map
    MapBase::Ptr copy() const override { return MapBase::Ptr(new ScaleTranslateMap(*this)); }

    MapBase::Ptr inverseMap() const override
    {
        return MapBase::Ptr(new ScaleTranslateMap(
            mScaleValuesInverse, -mScaleValuesInverse * mTranslation));
    }

    static bool isRegistered() { return MapRegistry::isRegistered(ScaleTranslateMap::mapType()); }

    static void registerMap()
    {
        MapRegistry::registerMap(
            ScaleTranslateMap::mapType(),
            ScaleTranslateMap::create);
    }

    Name type() const override { return mapType(); }
    static Name mapType() { return Name("ScaleTranslateMap"); }

    /// Return @c true (a ScaleTranslateMap is always linear).
    bool isLinear() const override { return true; }

    /// @brief Return @c true if the scale values have the same magnitude
    /// (eg. -1, 1, -1 would be a rotation).
    bool hasUniformScale() const override
    {
        bool value = isApproxEqual(
            std::abs(mScaleValues.x()), std::abs(mScaleValues.y()), double(5e-7));
        value = value && isApproxEqual(
            std::abs(mScaleValues.x()), std::abs(mScaleValues.z()), double(5e-7));
        return value;
    }

    /// Return the image of @c under the map
    Vec3d applyMap(const Vec3d& in) const override
    {
        return Vec3d(
            in.x() * mScaleValues.x() + mTranslation.x(),
            in.y() * mScaleValues.y() + mTranslation.y(),
            in.z() * mScaleValues.z() + mTranslation.z());
    }
    /// Return the pre-image of @c under the map
    Vec3d applyInverseMap(const Vec3d& in) const override
    {
        return Vec3d(
            (in.x() - mTranslation.x() ) * mScaleValuesInverse.x(),
            (in.y() - mTranslation.y() ) * mScaleValuesInverse.y(),
            (in.z() - mTranslation.z() ) * mScaleValuesInverse.z());
    }

    /// Return the Jacobian of the map applied to @a in.
    Vec3d applyJacobian(const Vec3d& in, const Vec3d&) const override { return applyJacobian(in); }
    /// Return the Jacobian of the map applied to @a in.
    Vec3d applyJacobian(const Vec3d& in) const override { return in * mScaleValues; }

    /// @brief Return the Inverse Jacobian of the map applied to @a in
    /// (i.e. inverse map with out translation)
    Vec3d applyInverseJacobian(const Vec3d& in, const Vec3d&) const override { return applyInverseJacobian(in); }
    /// @brief Return the Inverse Jacobian of the map applied to @a in
    /// (i.e. inverse map with out translation)
    Vec3d applyInverseJacobian(const Vec3d& in) const override { return in * mScaleValuesInverse; }

    /// @brief Return the Jacobian Transpose of the map applied to @a in.
    /// @details This tranforms range-space gradients to domain-space gradients
    Vec3d applyJT(const Vec3d& in, const Vec3d&) const override { return applyJT(in); }
    /// Return the Jacobian Transpose of the map applied to @a in.
    Vec3d applyJT(const Vec3d& in) const override { return applyJacobian(in); }

    /// @brief Return the transpose of the inverse Jacobian of the map applied to @a in
    /// @details Ignores second argument
    Vec3d applyIJT(const Vec3d& in, const Vec3d&) const override { return applyIJT(in);}
    /// Return the transpose of the inverse Jacobian of the map applied to @c in
    Vec3d applyIJT(const Vec3d& in) const override
    {
        return Vec3d(
            in.x() * mScaleValuesInverse.x(),
            in.y() * mScaleValuesInverse.y(),
            in.z() * mScaleValuesInverse.z());
    }
    /// Return the Jacobian Curvature: zero for a linear map
    Mat3d applyIJC(const Mat3d& in) const override
    {
        Mat3d tmp;
        for (int i=0; i<3; i++){
            tmp.setRow(i, in.row(i)*mScaleValuesInverse(i));
        }
        for (int i=0; i<3; i++){
            tmp.setCol(i, tmp.col(i)*mScaleValuesInverse(i));
        }
        return tmp;
    }
    Mat3d applyIJC(const Mat3d& in, const Vec3d&, const Vec3d& ) const override {
        return applyIJC(in);
    }

    /// Return the product of the scale values, ignores argument
    double determinant(const Vec3d&) const override { return determinant(); }
    /// Return the product of the scale values
    double determinant() const override {
        return mScaleValues.x() * mScaleValues.y() * mScaleValues.z();
    }
    /// Return the absolute values of the scale values
    Vec3d voxelSize() const override { return mVoxelSize;}
    /// Return the absolute values of the scale values, ignores argument
    Vec3d voxelSize(const Vec3d&) const override { return voxelSize();}

    /// Returns the scale values
    const Vec3d& getScale() const { return mScaleValues; }
    /// Returns the translation
    const Vec3d& getTranslation() const { return mTranslation; }

    /// Return the square of the scale.  Used to optimize some finite difference calculations
    const Vec3d& getInvScaleSqr() const {return mInvScaleSqr;}
    /// Return 1/(2 scale). Used to optimize some finite difference calculations
    const Vec3d& getInvTwiceScale() const {return mInvTwiceScale;}
    /// Return 1/(scale)
    const Vec3d& getInvScale() const {return mScaleValuesInverse; }

    /// read serialization
    void read(std::istream& is) override
    {
        mTranslation.read(is);
        mScaleValues.read(is);
        mVoxelSize.read(is);
        mScaleValuesInverse.read(is);
        mInvScaleSqr.read(is);
        mInvTwiceScale.read(is);
    }
    /// write serialization
    void write(std::ostream& os) const override
    {
        mTranslation.write(os);
        mScaleValues.write(os);
        mVoxelSize.write(os);
        mScaleValuesInverse.write(os);
        mInvScaleSqr.write(os);
        mInvTwiceScale.write(os);
    }
    /// string serialization, useful for debuging
    std::string str() const override
    {
        std::ostringstream buffer;
        buffer << " - translation: " << mTranslation << std::endl;
        buffer << " - scale: " << mScaleValues << std::endl;
        buffer << " - voxel dimensions: " << mVoxelSize << std::endl;
        return buffer.str();
    }

    bool isEqual(const MapBase& other) const override { return isEqualBase(*this, other); }

    bool operator==(const ScaleTranslateMap& other) const
    {
        // ::eq() uses a tolerance
        if (!mScaleValues.eq(other.mScaleValues)) { return false; }
        if (!mTranslation.eq(other.mTranslation)) { return false; }
        return true;
    }

    bool operator!=(const ScaleTranslateMap& other) const { return !(*this == other); }

    /// Return AffineMap::Ptr to an AffineMap equivalent to *this
    AffineMap::Ptr getAffineMap() const override
    {
        AffineMap::Ptr affineMap(new AffineMap(math::scale<Mat4d>(mScaleValues)));
        affineMap->accumPostTranslation(mTranslation);
        return affineMap;
    }

    //@{
    /// @brief  Return a MapBase::Ptr to a new map that is the result
    /// of prepending the appropraite operation.
    MapBase::Ptr preRotate(double radians, Axis axis) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPreRotation(axis, radians);
        return simplify(affineMap);
    }
    MapBase::Ptr preTranslate(const Vec3d& t) const override
    {
        const Vec3d& s = mScaleValues;
        const Vec3d scaled_trans( t.x() * s.x(),
                                  t.y() * s.y(),
                                  t.z() * s.z() );
        return MapBase::Ptr( new ScaleTranslateMap(mScaleValues, mTranslation + scaled_trans));
    }

    MapBase::Ptr preScale(const Vec3d& v) const override;

    MapBase::Ptr preShear(double shear, Axis axis0, Axis axis1) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPreShear(axis0, axis1, shear);
        return simplify(affineMap);
    }
    //@}

    //@{
    /// @brief  Return a MapBase::Ptr to a new map that is the result
    /// of postfixing the appropraite operation.
    MapBase::Ptr postRotate(double radians, Axis axis) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPostRotation(axis, radians);
        return simplify(affineMap);
    }
    MapBase::Ptr postTranslate(const Vec3d& t) const override
    {
        return MapBase::Ptr( new ScaleTranslateMap(mScaleValues, mTranslation + t));
    }

    MapBase::Ptr postScale(const Vec3d& v) const override;

    MapBase::Ptr postShear(double shear, Axis axis0, Axis axis1) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPostShear(axis0, axis1, shear);
        return simplify(affineMap);
    }
    //@}

private:
    Vec3d mTranslation, mScaleValues, mVoxelSize, mScaleValuesInverse,
        mInvScaleSqr, mInvTwiceScale;
}; // class ScaleTanslateMap


inline MapBase::Ptr
ScaleMap::postTranslate(const Vec3d& t) const
{
    return MapBase::Ptr(new ScaleTranslateMap(mScaleValues, t));
}


inline MapBase::Ptr
ScaleMap::preTranslate(const Vec3d& t) const
{

    const Vec3d& s = mScaleValues;
    const Vec3d scaled_trans( t.x() * s.x(),
                              t.y() * s.y(),
                              t.z() * s.z() );
    return MapBase::Ptr(new ScaleTranslateMap(mScaleValues, scaled_trans));
}


/// @brief A specialized Affine transform that uniformaly scales along the principal axis
/// and then translates the result.
class OPENVDB_API UniformScaleTranslateMap: public ScaleTranslateMap
{
public:
    using Ptr = SharedPtr<UniformScaleTranslateMap>;
    using ConstPtr = SharedPtr<const UniformScaleTranslateMap>;

    UniformScaleTranslateMap():ScaleTranslateMap(Vec3d(1,1,1), Vec3d(0,0,0)) {}
    UniformScaleTranslateMap(double scale, const Vec3d& translate):
        ScaleTranslateMap(Vec3d(scale,scale,scale), translate) {}
    UniformScaleTranslateMap(const UniformScaleMap& scale, const TranslationMap& translate):
        ScaleTranslateMap(scale.getScale(), translate.getTranslation()) {}

    UniformScaleTranslateMap(const UniformScaleTranslateMap& other):ScaleTranslateMap(other) {}
    ~UniformScaleTranslateMap() override = default;

    /// Return a MapBase::Ptr to a new UniformScaleTranslateMap
    static MapBase::Ptr create() { return MapBase::Ptr(new UniformScaleTranslateMap()); }
    /// Return a MapBase::Ptr to a deep copy of this map
    MapBase::Ptr copy() const override { return MapBase::Ptr(new UniformScaleTranslateMap(*this)); }

    MapBase::Ptr inverseMap() const override
    {
        const Vec3d& scaleInv = getInvScale();
        const Vec3d& trans = getTranslation();
        return MapBase::Ptr(new UniformScaleTranslateMap(scaleInv[0], -scaleInv[0] * trans));
    }

    static bool isRegistered()
    {
        return MapRegistry::isRegistered(UniformScaleTranslateMap::mapType());
    }

    static void registerMap()
    {
        MapRegistry::registerMap(
            UniformScaleTranslateMap::mapType(), UniformScaleTranslateMap::create);
    }

    Name type() const override { return mapType(); }
    static Name mapType() { return Name("UniformScaleTranslateMap"); }

    bool isEqual(const MapBase& other) const override { return isEqualBase(*this, other); }

    bool operator==(const UniformScaleTranslateMap& other) const
    {
        return ScaleTranslateMap::operator==(other);
    }
    bool operator!=(const UniformScaleTranslateMap& other) const { return !(*this == other); }

    /// @brief Return a MapBase::Ptr to a UniformScaleTranslateMap that is
    /// the result of prepending translation on this map.
    MapBase::Ptr preTranslate(const Vec3d& t) const override
    {
        const double scale = this->getScale().x();
        const Vec3d  new_trans = this->getTranslation() + scale * t;
        return MapBase::Ptr( new UniformScaleTranslateMap(scale, new_trans));
    }

    /// @brief Return a MapBase::Ptr to a UniformScaleTranslateMap that is
    /// the result of postfixing translation on this map.
    MapBase::Ptr postTranslate(const Vec3d& t) const override
    {
        const double scale = this->getScale().x();
        return MapBase::Ptr( new UniformScaleTranslateMap(scale, this->getTranslation() + t));
    }
}; // class UniformScaleTanslateMap


inline MapBase::Ptr
UniformScaleMap::postTranslate(const Vec3d& t) const
{
    const double scale = this->getScale().x();
    return MapBase::Ptr(new UniformScaleTranslateMap(scale, t));
}


inline MapBase::Ptr
UniformScaleMap::preTranslate(const Vec3d& t) const
{
    const double scale = this->getScale().x();
    return MapBase::Ptr(new UniformScaleTranslateMap(scale, scale*t));
}


inline MapBase::Ptr
TranslationMap::preScale(const Vec3d& v) const
{
    if (isApproxEqual(v[0],v[1]) && isApproxEqual(v[0],v[2])) {
        return MapBase::Ptr(new UniformScaleTranslateMap(v[0], mTranslation));
    } else {
        return MapBase::Ptr(new ScaleTranslateMap(v, mTranslation));
    }
}


inline MapBase::Ptr
TranslationMap::postScale(const Vec3d& v) const
{
    if (isApproxEqual(v[0],v[1]) && isApproxEqual(v[0],v[2])) {
        return MapBase::Ptr(new UniformScaleTranslateMap(v[0], v[0]*mTranslation));
    } else {
        const Vec3d trans(mTranslation.x()*v.x(),
                          mTranslation.y()*v.y(),
                          mTranslation.z()*v.z());
        return MapBase::Ptr(new ScaleTranslateMap(v, trans));
    }
}


inline MapBase::Ptr
ScaleTranslateMap::preScale(const Vec3d& v) const
{
    const Vec3d new_scale( v * mScaleValues );
    if (isApproxEqual(new_scale[0],new_scale[1]) && isApproxEqual(new_scale[0],new_scale[2])) {
        return MapBase::Ptr( new UniformScaleTranslateMap(new_scale[0], mTranslation));
    } else {
        return MapBase::Ptr( new ScaleTranslateMap(new_scale, mTranslation));
    }
}


inline MapBase::Ptr
ScaleTranslateMap::postScale(const Vec3d& v) const
{
    const Vec3d new_scale( v * mScaleValues );
    const Vec3d new_trans( mTranslation.x()*v.x(),
                           mTranslation.y()*v.y(),
                           mTranslation.z()*v.z() );

    if (isApproxEqual(new_scale[0],new_scale[1]) && isApproxEqual(new_scale[0],new_scale[2])) {
        return MapBase::Ptr( new UniformScaleTranslateMap(new_scale[0], new_trans));
    } else {
        return MapBase::Ptr( new ScaleTranslateMap(new_scale, new_trans));
    }
}


////////////////////////////////////////


/// @brief A specialized linear transform that performs a unitary maping
/// i.e. rotation  and or reflection.
class OPENVDB_API UnitaryMap: public MapBase
{
public:
    using Ptr = SharedPtr<UnitaryMap>;
    using ConstPtr = SharedPtr<const UnitaryMap>;

    /// default constructor makes an Idenity.
    UnitaryMap(): mAffineMap(Mat4d::identity())
    {
    }

    UnitaryMap(const Vec3d& axis, double radians)
    {
        Mat3d matrix;
        matrix.setToRotation(axis, radians);
        mAffineMap = AffineMap(matrix);
    }

    UnitaryMap(Axis axis, double radians)
    {
        Mat4d matrix;
        matrix.setToRotation(axis, radians);
        mAffineMap = AffineMap(matrix);
    }

    UnitaryMap(const Mat3d& m)
    {
        // test that the mat3 is a rotation || reflection
        if (!isUnitary(m)) {
            OPENVDB_THROW(ArithmeticError, "Matrix initializing unitary map was not unitary");
        }

        Mat4d matrix(Mat4d::identity());
        matrix.setMat3(m);
        mAffineMap = AffineMap(matrix);
    }

    UnitaryMap(const Mat4d& m)
    {
        if (!isInvertible(m)) {
            OPENVDB_THROW(ArithmeticError,
                "4x4 Matrix initializing unitary map was not unitary: not invertible");
        }

        if (!isAffine(m)) {
            OPENVDB_THROW(ArithmeticError,
                "4x4 Matrix initializing unitary map was not unitary: not affine");
        }

        if (hasTranslation(m)) {
            OPENVDB_THROW(ArithmeticError,
                "4x4 Matrix initializing unitary map was not unitary: had translation");
        }

        if (!isUnitary(m.getMat3())) {
            OPENVDB_THROW(ArithmeticError,
                "4x4 Matrix initializing unitary map was not unitary");
        }

        mAffineMap = AffineMap(m);
    }

    UnitaryMap(const UnitaryMap& other):
        MapBase(other),
        mAffineMap(other.mAffineMap)
    {
    }

    UnitaryMap(const UnitaryMap& first, const UnitaryMap& second):
        mAffineMap(*(first.getAffineMap()), *(second.getAffineMap()))
    {
    }

    ~UnitaryMap() override = default;

    /// Return a MapBase::Ptr to a new UnitaryMap
    static MapBase::Ptr create() { return MapBase::Ptr(new UnitaryMap()); }
    /// Returns a MapBase::Ptr to a deep copy of *this
    MapBase::Ptr copy() const override { return MapBase::Ptr(new UnitaryMap(*this)); }

    MapBase::Ptr inverseMap() const override
    {
        return MapBase::Ptr(new UnitaryMap(mAffineMap.getMat4().inverse()));
    }

    static bool isRegistered() { return MapRegistry::isRegistered(UnitaryMap::mapType()); }

    static void registerMap()
    {
        MapRegistry::registerMap(
            UnitaryMap::mapType(),
            UnitaryMap::create);
    }

    /// Return @c UnitaryMap
    Name type() const override { return mapType(); }
    /// Return @c UnitaryMap
    static Name mapType() { return Name("UnitaryMap"); }

    /// Return @c true (a UnitaryMap is always linear).
    bool isLinear() const override { return true; }

    /// Return @c false (by convention true)
    bool hasUniformScale() const override { return true; }

    bool isEqual(const MapBase& other) const override { return isEqualBase(*this, other); }

    bool operator==(const UnitaryMap& other) const
    {
        // compare underlying linear map.
        if (mAffineMap!=other.mAffineMap)  return false;
        return true;
    }

    bool operator!=(const UnitaryMap& other) const { return !(*this == other); }
    /// Return the image of @c in under the map
    Vec3d applyMap(const Vec3d& in) const override { return mAffineMap.applyMap(in); }
    /// Return the pre-image of @c in under the map
    Vec3d applyInverseMap(const Vec3d& in) const override { return mAffineMap.applyInverseMap(in); }

    Vec3d applyJacobian(const Vec3d& in, const Vec3d&) const override { return applyJacobian(in); }
    /// Return the Jacobian of the map applied to @a in.
    Vec3d applyJacobian(const Vec3d& in) const override { return mAffineMap.applyJacobian(in); }

    /// @brief Return the Inverse Jacobian of the map applied to @a in
    /// (i.e. inverse map with out translation)
    Vec3d applyInverseJacobian(const Vec3d& in, const Vec3d&) const override {
        return applyInverseJacobian(in);
    }
    /// @brief Return the Inverse Jacobian of the map applied to @a in
    /// (i.e. inverse map with out translation)
    Vec3d applyInverseJacobian(const Vec3d& in) const override {
        return mAffineMap.applyInverseJacobian(in);
    }

    /// @brief Return the Jacobian Transpose of the map applied to @a in.
    /// @details This tranforms range-space gradients to domain-space gradients
    Vec3d applyJT(const Vec3d& in, const Vec3d&) const override { return applyJT(in); }
    /// Return the Jacobian Transpose of the map applied to @a in.
    Vec3d applyJT(const Vec3d& in) const override {
        return applyInverseMap(in); // the transpose of the unitary map is its inverse
    }


    /// @brief Return the transpose of the inverse Jacobian of the map applied to @a in
    /// @details Ignores second argument
    Vec3d applyIJT(const Vec3d& in, const Vec3d& ) const override { return applyIJT(in);}
    /// Return the transpose of the inverse Jacobian of the map applied to @c in
    Vec3d applyIJT(const Vec3d& in) const override { return mAffineMap.applyIJT(in); }
    /// Return the Jacobian Curvature: zero for a linear map
    Mat3d applyIJC(const Mat3d& in) const override { return mAffineMap.applyIJC(in); }
    Mat3d applyIJC(const Mat3d& in, const Vec3d&, const Vec3d& ) const override {
        return applyIJC(in);
    }

    /// Return the determinant of the Jacobian, ignores argument
    double determinant(const Vec3d&) const override { return determinant(); }
    /// Return the determinant of the Jacobian
    double determinant() const override { return mAffineMap.determinant(); }


    /// @{
    /// @brief Returns the lengths of the images of the segments
    /// (0,0,0) &minus; (1,0,0), (0,0,0) &minus; (0,1,0) and (0,0,0) &minus; (0,0,1).
    Vec3d voxelSize() const override { return mAffineMap.voxelSize();}
    Vec3d voxelSize(const Vec3d&) const override { return voxelSize();}
    /// @}

    /// read serialization
    void read(std::istream& is) override
    {
        mAffineMap.read(is);
    }

    /// write serialization
    void write(std::ostream& os) const override
    {
        mAffineMap.write(os);
    }
    /// string serialization, useful for debuging
    std::string str() const override
    {
        std::ostringstream buffer;
        buffer << mAffineMap.str();
        return buffer.str();
    }
    /// Return AffineMap::Ptr to an AffineMap equivalent to *this
    AffineMap::Ptr getAffineMap() const override {
        return AffineMap::Ptr(new AffineMap(mAffineMap));
    }

    /// @brief Return a MapBase::Ptr to a new map that is the result
    /// of prepending the given rotation.
    MapBase::Ptr preRotate(double radians, Axis axis) const override
    {
        UnitaryMap first(axis, radians);
        UnitaryMap::Ptr unitaryMap(new UnitaryMap(first, *this));
        return StaticPtrCast<MapBase, UnitaryMap>(unitaryMap);
    }
    /// @brief Return a MapBase::Ptr to a new map that is the result
    /// of prepending the given translation.
    MapBase::Ptr preTranslate(const Vec3d& t) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPreTranslation(t);
        return simplify(affineMap);
    }
    /// @brief Return a MapBase::Ptr to a new map that is the result
    /// of prepending the given scale.
    MapBase::Ptr preScale(const Vec3d& v) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPreScale(v);
        return simplify(affineMap);
    }
    /// @brief Return a MapBase::Ptr to a new map that is the result
    /// of prepending the given shear.
    MapBase::Ptr preShear(double shear, Axis axis0, Axis axis1) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPreShear(axis0, axis1, shear);
        return simplify(affineMap);
    }

    /// @brief Return a MapBase::Ptr to a new map that is the result
    /// of appending the given rotation.
    MapBase::Ptr postRotate(double radians, Axis axis) const override
    {
        UnitaryMap second(axis, radians);
        UnitaryMap::Ptr unitaryMap(new UnitaryMap(*this, second));
        return StaticPtrCast<MapBase, UnitaryMap>(unitaryMap);
    }
    /// @brief Return a MapBase::Ptr to a new map that is the result
    /// of appending the given translation.
    MapBase::Ptr postTranslate(const Vec3d& t) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPostTranslation(t);
        return simplify(affineMap);
    }
    /// @brief Return a MapBase::Ptr to a new map that is the result
    /// of appending the given scale.
    MapBase::Ptr postScale(const Vec3d& v) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPostScale(v);
        return simplify(affineMap);
    }
    /// @brief Return a MapBase::Ptr to a new map that is the result
    /// of appending the given shear.
    MapBase::Ptr postShear(double shear, Axis axis0, Axis axis1) const override
    {
        AffineMap::Ptr affineMap = getAffineMap();
        affineMap->accumPostShear(axis0, axis1, shear);
        return simplify(affineMap);
    }

private:
    AffineMap  mAffineMap;
}; // class UnitaryMap


////////////////////////////////////////


/// @brief  This map is composed of three steps.
/// First it will take a box of size (Lx X  Ly X Lz) defined by a member data bounding box
/// and map it into a frustum with near plane (1 X Ly/Lx) and prescribed depth
/// Then this frustum is transformed by an internal second map: most often a uniform scale,
/// but other effects can be achieved by accumulating translation, shear and rotation: these
/// are all applied to the second map
class OPENVDB_API NonlinearFrustumMap: public MapBase
{
public:
    using Ptr = SharedPtr<NonlinearFrustumMap>;
    using ConstPtr = SharedPtr<const NonlinearFrustumMap>;

    NonlinearFrustumMap():
        MapBase(),
        mBBox(Vec3d(0), Vec3d(1)),
        mTaper(1),
        mDepth(1)
    {
        init();
    }

    /// @brief Constructor that takes an index-space bounding box
    /// to be mapped into a frustum with a given @a depth and @a taper
    /// (defined as ratio of nearplane/farplane).
    NonlinearFrustumMap(const BBoxd& bb, double taper, double depth):
        MapBase(),mBBox(bb), mTaper(taper), mDepth(depth)
    {
        init();
    }

    /// @brief Constructor that takes an index-space bounding box
    /// to be mapped into a frustum with a given @a depth and @a taper
    /// (defined as ratio of nearplane/farplane).
    /// @details This frustum is further modifed by the @a secondMap,
    /// intended to be a simple translation and rotation and uniform scale
   NonlinearFrustumMap(const BBoxd& bb, double taper, double depth,
        const MapBase::Ptr& secondMap):
        mBBox(bb), mTaper(taper), mDepth(depth)
    {
        if (!secondMap->isLinear() ) {
              OPENVDB_THROW(ArithmeticError,
                "The second map in the Frustum transfrom must be linear");
        }
        mSecondMap = *( secondMap->getAffineMap() );
        init();
    }

    NonlinearFrustumMap(const NonlinearFrustumMap& other):
        MapBase(),
        mBBox(other.mBBox),
        mTaper(other.mTaper),
        mDepth(other.mDepth),
        mSecondMap(other.mSecondMap),
        mHasSimpleAffine(other.mHasSimpleAffine)
    {
        init();
    }

    /// @brief Constructor from a camera frustum
    ///
    /// @param position the tip of the frustum (i.e., the camera's position).
    /// @param direction a vector pointing from @a position toward the near plane.
    /// @param up a non-unit vector describing the direction and extent of
    ///     the frustum's intersection on the near plane.  Together,
    ///     @a up must be orthogonal to @a direction.
    /// @param aspect the aspect ratio of the frustum intersection with near plane
    ///     defined as width / height
    /// @param z_near,depth the distance from @a position along @a direction to the
    ///     near and far planes of the frustum.
    /// @param x_count the number of voxels, aligned with @a left,
    ///     across the face of the frustum
    /// @param z_count the number of voxels, aligned with @a direction,
    ///     between the near and far planes
    NonlinearFrustumMap(const Vec3d& position,
                        const Vec3d& direction,
                        const Vec3d& up,
                        double aspect /* width / height */,
                        double z_near, double depth,
                        Coord::ValueType x_count, Coord::ValueType z_count) {

        /// @todo check that depth > 0
        /// @todo check up.length > 0
        /// @todo check that direction dot up = 0
        if (!(depth > 0)) {
            OPENVDB_THROW(ArithmeticError,
                "The frustum depth must be non-zero and positive");
        }
        if (!(up.length() > 0)) {
            OPENVDB_THROW(ArithmeticError,
                "The frustum height must be non-zero and positive");
        }
        if (!(aspect > 0)) {
            OPENVDB_THROW(ArithmeticError,
                "The frustum aspect ratio  must be non-zero and positive");
        }
        if (!(isApproxEqual(up.dot(direction), 0.))) {
            OPENVDB_THROW(ArithmeticError,
                "The frustum up orientation must be perpendicular to into-frustum direction");
        }

        double near_plane_height = 2 * up.length();
        double near_plane_width = aspect * near_plane_height;

        Coord::ValueType y_count = static_cast<int>(Round(x_count / aspect));

        mBBox = BBoxd(Vec3d(0,0,0), Vec3d(x_count, y_count, z_count));
        mDepth = depth / near_plane_width;  // depth non-dimensionalized on width
        double gamma = near_plane_width / z_near;
        mTaper = 1./(mDepth*gamma + 1.);

        Vec3d direction_unit = direction;
        direction_unit.normalize();

        Mat4d r1(Mat4d::identity());
        r1.setToRotation(/*from*/Vec3d(0,0,1), /*to */direction_unit);
        Mat4d r2(Mat4d::identity());
        Vec3d temp = r1.inverse().transform(up);
        r2.setToRotation(/*from*/Vec3d(0,1,0), /*to*/temp );
        Mat4d scale = math::scale<Mat4d>(
            Vec3d(near_plane_width, near_plane_width, near_plane_width));

        // move the near plane to origin, rotate to align with axis, and scale down
        // T_inv * R1_inv * R2_inv * scale_inv
        Mat4d mat = scale * r2 * r1;
        mat.setTranslation(position + z_near*direction_unit);

        mSecondMap = AffineMap(mat);

        init();
    }

    ~NonlinearFrustumMap() override = default;

    /// Return a MapBase::Ptr to a new NonlinearFrustumMap
    static MapBase::Ptr create() { return MapBase::Ptr(new NonlinearFrustumMap()); }
    /// Return a MapBase::Ptr to a deep copy of this map
    MapBase::Ptr copy() const override { return MapBase::Ptr(new NonlinearFrustumMap(*this)); }

    /// @brief Not implemented, since there is currently no map type that can
    /// represent the inverse of a frustum
    /// @throw NotImplementedError
    MapBase::Ptr inverseMap() const override
    {
        OPENVDB_THROW(NotImplementedError,
            "inverseMap() is not implemented for NonlinearFrustumMap");
    }
    static bool isRegistered() { return MapRegistry::isRegistered(NonlinearFrustumMap::mapType()); }

    static void registerMap()
    {
        MapRegistry::registerMap(
            NonlinearFrustumMap::mapType(),
            NonlinearFrustumMap::create);
    }
    /// Return @c NonlinearFrustumMap
    Name type() const override { return mapType(); }
    /// Return @c NonlinearFrustumMap
    static Name mapType() { return Name("NonlinearFrustumMap"); }

    /// Return @c false (a NonlinearFrustumMap is never linear).
    bool isLinear() const override { return false; }

    /// Return @c false (by convention false)
    bool hasUniformScale() const override { return false; }

    /// Return @c true if the map is equivalent to an identity
    bool isIdentity() const
    {
        // The frustum can only be consistent with a linear map if the taper value is 1
        if (!isApproxEqual(mTaper, double(1)) ) return false;

        // There are various ways an identity can decomposed between the two parts of the
        // map.  Best to just check that the principle vectors are stationary.
        const Vec3d e1(1,0,0);
        if (!applyMap(e1).eq(e1)) return false;

        const Vec3d e2(0,1,0);
        if (!applyMap(e2).eq(e2)) return false;

        const Vec3d e3(0,0,1);
        if (!applyMap(e3).eq(e3)) return false;

        return true;
    }

    bool isEqual(const MapBase& other) const override { return isEqualBase(*this, other); }

    bool operator==(const NonlinearFrustumMap& other) const
    {
        if (mBBox!=other.mBBox) return false;
        if (!isApproxEqual(mTaper, other.mTaper)) return false;
        if (!isApproxEqual(mDepth, other.mDepth)) return false;

        // Two linear transforms are equivalent iff they have the same translation
        // and have the same affects on orthongal spanning basis check translation
        Vec3d e(0,0,0);
        if (!mSecondMap.applyMap(e).eq(other.mSecondMap.applyMap(e))) return false;
        /// check spanning vectors
        e(0) = 1;
        if (!mSecondMap.applyMap(e).eq(other.mSecondMap.applyMap(e))) return false;
        e(0) = 0;
        e(1) = 1;
        if (!mSecondMap.applyMap(e).eq(other.mSecondMap.applyMap(e))) return false;
        e(1) = 0;
        e(2) = 1;
        if (!mSecondMap.applyMap(e).eq(other.mSecondMap.applyMap(e))) return false;
        return true;
    }

    bool operator!=(const NonlinearFrustumMap& other) const { return !(*this == other); }

    /// Return the image of @c in under the map
    Vec3d applyMap(const Vec3d& in) const override
    {
        return mSecondMap.applyMap(applyFrustumMap(in));
    }

    /// Return the pre-image of @c in under the map
    Vec3d applyInverseMap(const Vec3d& in) const override
    {
        return applyFrustumInverseMap(mSecondMap.applyInverseMap(in));
    }
    /// Return the Jacobian of the linear second map applied to @c in
    Vec3d applyJacobian(const Vec3d& in) const override { return mSecondMap.applyJacobian(in); }
    /// Return the Jacobian defined at @c isloc applied to @c in
    Vec3d applyJacobian(const Vec3d& in, const Vec3d& isloc) const override
    {
        // Move the center of the x-face of the bbox
        // to the origin in index space.
        Vec3d centered(isloc);
        centered = centered - mBBox.min();
        centered.x() -= mXo;
        centered.y() -= mYo;

        // scale the z-direction on depth / K count
        const double zprime = centered.z()*mDepthOnLz;

        const double scale = (mGamma * zprime + 1.) / mLx;
        const double scale2 = mGamma * mDepthOnLz / mLx;

        const Vec3d tmp(scale * in.x() + scale2 * centered.x()* in.z(),
                        scale * in.y() + scale2 * centered.y()* in.z(),
                        mDepthOnLz * in.z());

        return mSecondMap.applyJacobian(tmp);
    }


    /// @brief Return the Inverse Jacobian of the map applied to @a in
    /// (i.e. inverse map with out translation)
    Vec3d applyInverseJacobian(const Vec3d& in) const override {
        return mSecondMap.applyInverseJacobian(in);
    }
    /// Return the Inverse Jacobian defined at @c isloc of the map applied to @a in.
    Vec3d applyInverseJacobian(const Vec3d& in, const Vec3d& isloc) const override {

        // Move the center of the x-face of the bbox
        // to the origin in index space.
        Vec3d centered(isloc);
        centered = centered - mBBox.min();
        centered.x() -= mXo;
        centered.y() -= mYo;

        // scale the z-direction on depth / K count
        const double zprime = centered.z()*mDepthOnLz;

        const double scale = (mGamma * zprime + 1.) / mLx;
        const double scale2 = mGamma * mDepthOnLz / mLx;


        Vec3d out = mSecondMap.applyInverseJacobian(in);

        out.x() = (out.x() - scale2 * centered.x() * out.z() / mDepthOnLz) / scale;
        out.y() = (out.y() - scale2 * centered.y() * out.z() / mDepthOnLz) / scale;
        out.z() = out.z() / mDepthOnLz;

        return out;
    }

    /// @brief Return the Jacobian Transpose of the map applied to vector @c in at @c indexloc.
    /// @details This tranforms range-space gradients to domain-space gradients.
    Vec3d applyJT(const Vec3d& in, const Vec3d& isloc) const override {
        const Vec3d tmp = mSecondMap.applyJT(in);
        // Move the center of the x-face of the bbox
        // to the origin in index space.
        Vec3d centered(isloc);
        centered = centered - mBBox.min();
        centered.x() -= mXo;
        centered.y() -= mYo;

        // scale the z-direction on depth / K count
        const double zprime = centered.z()*mDepthOnLz;

        const double scale = (mGamma * zprime + 1.) / mLx;
        const double scale2 = mGamma * mDepthOnLz / mLx;

        return Vec3d(scale * tmp.x(),
                     scale * tmp.y(),
                     scale2 * centered.x()* tmp.x() +
                     scale2 * centered.y()* tmp.y() +
                     mDepthOnLz * tmp.z());
    }
    /// Return the Jacobian Transpose of the second map applied to @c in.
    Vec3d applyJT(const Vec3d& in) const override {
        return mSecondMap.applyJT(in);
    }

    /// Return the transpose of the inverse Jacobian of the linear second map applied to @c in
    Vec3d applyIJT(const Vec3d& in) const override { return mSecondMap.applyIJT(in); }

    // the Jacobian of the nonlinear part of the transform is a sparse matrix
    // Jacobian^(-T) =
    //
    //    (Lx)(  1/s               0              0 )
    //        (  0                1/s             0 )
    //        (  -(x-xo)g/(sLx)   -(y-yo)g/(sLx)  Lz/(Depth Lx)   )
    /// Return the transpose of the inverse Jacobain (at @c locW applied to @c in.
    /// @c ijk is the location in the pre-image space (e.g. index space)
    Vec3d applyIJT(const Vec3d& d1_is, const Vec3d& ijk) const override
    {
        const Vec3d loc = applyFrustumMap(ijk);
        const double s = mGamma * loc.z() + 1.;

        // verify that we aren't at the singularity
        if (isApproxEqual(s, 0.)) {
            OPENVDB_THROW(ArithmeticError, "Tried to evaluate the frustum transform"
                " at the singular focal point (e.g. camera)");
        }

        const double sinv = 1.0/s;        // 1/(z*gamma + 1)
        const double pt0 = mLx * sinv;    // Lx / (z*gamma +1)
        const double pt1 = mGamma * pt0;  // gamma * Lx / ( z*gamma +1)
        const double pt2 = pt1 * sinv;    // gamma * Lx / ( z*gamma +1)**2

        const Mat3d& jacinv = mSecondMap.getConstJacobianInv();

        // compute \frac{\partial E_i}{\partial x_j}
        Mat3d gradE(Mat3d::zero());
        for (int j = 0; j < 3; ++j ) {
            gradE(0,j) =  pt0 * jacinv(0,j) -  pt2 * loc.x()*jacinv(2,j);
            gradE(1,j) =  pt0 * jacinv(1,j) -  pt2 * loc.y()*jacinv(2,j);
            gradE(2,j) = (1./mDepthOnLz) * jacinv(2,j);
        }

        Vec3d result;
        for (int i = 0; i < 3; ++i) {
            result(i) = d1_is(0) * gradE(0,i) + d1_is(1) * gradE(1,i) + d1_is(2) * gradE(2,i);
        }

        return result;

    }

    /// Return the Jacobian Curvature for the linear second map
    Mat3d applyIJC(const Mat3d& in) const override { return mSecondMap.applyIJC(in); }
    /// Return the Jacobian Curvature: all the second derivatives in range space
    /// @param d2_is second derivative matrix computed in index space
    /// @param d1_is gradient computed in index space
    /// @param ijk  the index space location where the result is computed
    Mat3d applyIJC(const Mat3d& d2_is, const Vec3d& d1_is, const Vec3d& ijk) const override
    {
        const Vec3d loc = applyFrustumMap(ijk);

        const double s =  mGamma * loc.z()  + 1.;

        // verify that we aren't at the singularity
        if (isApproxEqual(s, 0.)) {
            OPENVDB_THROW(ArithmeticError, "Tried to evaluate the frustum transform"
                " at the singular focal point (e.g. camera)");
        }

        // precompute
        const double sinv = 1.0/s;     // 1/(z*gamma + 1)
        const double pt0 = mLx * sinv;   // Lx / (z*gamma +1)
        const double pt1 = mGamma * pt0;   // gamma * Lx / ( z*gamma +1)
        const double pt2 = pt1 * sinv;   // gamma * Lx / ( z*gamma +1)**2
        const double pt3 = pt2 * sinv;   // gamma * Lx / ( z*gamma +1)**3

        const Mat3d& jacinv = mSecondMap.getConstJacobianInv();

        // compute \frac{\partial^2 E_i}{\partial x_j \partial x_k}

        Mat3d matE0(Mat3d::zero());
        Mat3d matE1(Mat3d::zero()); // matE2 = 0
        for(int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {

                const double pt4 =  2. * jacinv(2,j) * jacinv(2,k) * pt3;

                matE0(j,k) = -(jacinv(0,j) * jacinv(2,k) + jacinv(2,j) * jacinv(0,k)) * pt2 +
                    pt4 * loc.x();

                matE1(j,k) = -(jacinv(1,j) * jacinv(2,k) + jacinv(2,j) * jacinv(1,k)) * pt2 +
                    pt4 * loc.y();
            }
        }

        // compute \frac{\partial E_i}{\partial x_j}
        Mat3d gradE(Mat3d::zero());
        for (int j = 0; j < 3; ++j ) {
            gradE(0,j) =  pt0 * jacinv(0,j) -  pt2 * loc.x()*jacinv(2,j);
            gradE(1,j) =  pt0 * jacinv(1,j) -  pt2 * loc.y()*jacinv(2,j);
            gradE(2,j) = (1./mDepthOnLz) * jacinv(2,j);
        }

        Mat3d result(Mat3d::zero());
        // compute \fac{\partial E_j}{\partial x_m} \fac{\partial E_i}{\partial x_n}
        // \frac{\partial^2 input}{\partial E_i \partial E_j}
        for (int m = 0; m < 3; ++m ) {
            for ( int n = 0; n < 3; ++n) {
                for (int i = 0; i < 3; ++i ) {
                    for (int j = 0; j < 3; ++j) {
                        result(m, n) += gradE(j, m) * gradE(i, n) * d2_is(i, j);
                    }
                }
            }
        }

         for (int m = 0; m < 3; ++m ) {
            for ( int n = 0; n < 3; ++n) {
                result(m, n) +=
                    matE0(m, n) * d1_is(0) + matE1(m, n) * d1_is(1);// + matE2(m, n) * d1_is(2);
            }
        }

         return result;
    }

    /// Return the determinant of the Jacobian of linear second map
    double determinant() const override {return mSecondMap.determinant();} // no implementation

    /// Return the determinate of the Jacobian evaluated at @c loc
    /// @c loc is a location in the pre-image space (e.g., index space)
    double determinant(const Vec3d& loc) const override
    {
        double s = mGamma * loc.z() + 1.0;
        double frustum_determinant = s * s * mDepthOnLzLxLx;
        return mSecondMap.determinant() * frustum_determinant;
    }

    /// Return the size of a voxel at the center of the near plane
    Vec3d voxelSize() const override
    {
        const Vec3d loc( 0.5*(mBBox.min().x() + mBBox.max().x()),
                         0.5*(mBBox.min().y() + mBBox.max().y()),
                         mBBox.min().z());

        return voxelSize(loc);

    }

    /// @brief Returns the lengths of the images of the three segments
    /// from @a loc to @a loc + (1,0,0), from @a loc to @a loc + (0,1,0)
    /// and from @a loc to @a loc + (0,0,1)
    /// @param loc  a location in the pre-image space (e.g., index space)
    Vec3d voxelSize(const Vec3d& loc) const override
    {
        Vec3d out, pos = applyMap(loc);
        out(0) = (applyMap(loc + Vec3d(1,0,0)) - pos).length();
        out(1) = (applyMap(loc + Vec3d(0,1,0)) - pos).length();
        out(2) = (applyMap(loc + Vec3d(0,0,1)) - pos).length();
        return out;
    }

    AffineMap::Ptr getAffineMap() const override { return mSecondMap.getAffineMap(); }

    /// set the taper value, the ratio of nearplane width / far plane width
    void setTaper(double t) { mTaper = t; init();}
    /// Return the taper value.
    double getTaper() const { return mTaper; }
    /// set the frustum depth: distance between near and far plane = frustm depth * frustm x-width
    void setDepth(double d) { mDepth = d; init();}
    /// Return the unscaled frustm depth
    double getDepth() const { return mDepth; }
    // gamma a non-dimensional  number:  nearplane x-width / camera to near plane distance
    double getGamma() const { return mGamma; }

    /// Return the bounding box that defines the frustum in pre-image space
    const BBoxd& getBBox() const { return mBBox; }

    /// Return MapBase::Ptr& to the second map
    const AffineMap& secondMap() const { return mSecondMap; }
    /// Return @c true if the  the bounding box in index space that defines the region that
    /// is maped into the frustum is non-zero, otherwise @c false
    bool isValid() const { return !mBBox.empty();}

    /// Return @c true if the second map is a uniform scale, Rotation and translation
    bool hasSimpleAffine() const { return mHasSimpleAffine; }

    /// read serialization
    void read(std::istream& is) override
    {
        // for backward compatibility with earlier version
        if (io::getFormatVersion(is) < OPENVDB_FILE_VERSION_FLOAT_FRUSTUM_BBOX ) {
            CoordBBox bb;
            bb.read(is);
            mBBox = BBoxd(bb.min().asVec3d(), bb.max().asVec3d());
        } else {
            mBBox.read(is);
        }

        is.read(reinterpret_cast<char*>(&mTaper), sizeof(double));
        is.read(reinterpret_cast<char*>(&mDepth), sizeof(double));

        // Read the second maps type.
        Name type = readString(is);

        // Check if the map has been registered.
        if(!MapRegistry::isRegistered(type)) {
            OPENVDB_THROW(KeyError, "Map " << type << " is not registered");
        }

        // Create the second map of the type and then read it in.
        MapBase::Ptr proxy =  math::MapRegistry::createMap(type);
        proxy->read(is);
        mSecondMap = *(proxy->getAffineMap());
        init();
    }

    /// write serialization
    void write(std::ostream& os) const override
    {
        mBBox.write(os);
        os.write(reinterpret_cast<const char*>(&mTaper), sizeof(double));
        os.write(reinterpret_cast<const char*>(&mDepth), sizeof(double));

        writeString(os, mSecondMap.type());
        mSecondMap.write(os);
    }

    /// string serialization, useful for debuging
    std::string str() const override
    {
        std::ostringstream buffer;
        buffer << " - taper: " << mTaper << std::endl;
        buffer << " - depth: " << mDepth << std::endl;
        buffer << " SecondMap: "<< mSecondMap.type() << std::endl;
        buffer << mSecondMap.str() << std::endl;
        return buffer.str();
    }

    /// @brief Return a MapBase::Ptr to a new map that is the result
    /// of prepending the given rotation to the linear part of this map
    MapBase::Ptr preRotate(double radians, Axis axis = X_AXIS) const override
    {
        return MapBase::Ptr(
            new NonlinearFrustumMap(mBBox, mTaper, mDepth, mSecondMap.preRotate(radians, axis)));
    }
    /// @brief Return a MapBase::Ptr to a new map that is the result
    /// of prepending the given translation to the linear part of this map
    MapBase::Ptr preTranslate(const Vec3d& t) const override
    {
        return MapBase::Ptr(
            new NonlinearFrustumMap(mBBox, mTaper, mDepth, mSecondMap.preTranslate(t)));
    }
    /// @brief Return a MapBase::Ptr to a new map that is the result
    /// of prepending the given scale to the linear part of this map
    MapBase::Ptr preScale(const Vec3d& s) const override
    {
        return MapBase::Ptr(
            new NonlinearFrustumMap(mBBox, mTaper, mDepth, mSecondMap.preScale(s)));
    }
    /// @brief Return a MapBase::Ptr to a new map that is the result
    /// of prepending the given shear to the linear part of this map
    MapBase::Ptr preShear(double shear, Axis axis0, Axis axis1) const override
    {
        return MapBase::Ptr(new NonlinearFrustumMap(
            mBBox, mTaper, mDepth, mSecondMap.preShear(shear, axis0, axis1)));
    }

    /// @brief Return a MapBase::Ptr to a new map that is the result
    /// of appending the given rotation to the linear part of this map.
    MapBase::Ptr postRotate(double radians, Axis axis = X_AXIS) const override
    {
        return MapBase::Ptr(
            new NonlinearFrustumMap(mBBox, mTaper, mDepth, mSecondMap.postRotate(radians, axis)));
    }
    /// @brief Return a MapBase::Ptr to a new map that is the result
    /// of appending the given translation to the linear part of this map.
    MapBase::Ptr postTranslate(const Vec3d& t) const override
    {
        return MapBase::Ptr(
            new NonlinearFrustumMap(mBBox, mTaper, mDepth, mSecondMap.postTranslate(t)));
    }
    /// @brief Return a MapBase::Ptr to a new map that is the result
    /// of appending the given scale to the linear part of this map.
    MapBase::Ptr postScale(const Vec3d& s) const override
    {
        return MapBase::Ptr(
            new NonlinearFrustumMap(mBBox, mTaper, mDepth, mSecondMap.postScale(s)));
    }
    /// @brief Return a MapBase::Ptr to a new map that is the result
    /// of appending the given shear to the linear part of this map.
    MapBase::Ptr postShear(double shear, Axis axis0, Axis axis1) const override
    {
        return MapBase::Ptr(new NonlinearFrustumMap(
            mBBox, mTaper, mDepth, mSecondMap.postShear(shear, axis0, axis1)));
    }

private:
    void init()
    {
        // set up as a frustum
        mLx = mBBox.extents().x();
        mLy = mBBox.extents().y();
        mLz = mBBox.extents().z();

        if (isApproxEqual(mLx,0.) || isApproxEqual(mLy,0.) || isApproxEqual(mLz,0.) ) {
            OPENVDB_THROW(ArithmeticError, "The index space bounding box"
                " must have at least two index points in each direction.");
        }

        mXo = 0.5* mLx;
        mYo = 0.5* mLy;

        // mDepth is non-dimensionalized on near
        mGamma = (1./mTaper - 1) / mDepth;

        mDepthOnLz = mDepth/mLz;
        mDepthOnLzLxLx = mDepthOnLz/(mLx * mLx);

        /// test for shear and non-uniform scale
        mHasSimpleAffine = true;
        Vec3d tmp = mSecondMap.voxelSize();

        /// false if there is non-uniform scale
        if (!isApproxEqual(tmp(0), tmp(1))) { mHasSimpleAffine = false; return; }
        if (!isApproxEqual(tmp(0), tmp(2))) { mHasSimpleAffine = false; return; }

        Vec3d trans = mSecondMap.applyMap(Vec3d(0,0,0));
        /// look for shear
        Vec3d tmp1 = mSecondMap.applyMap(Vec3d(1,0,0)) - trans;
        Vec3d tmp2 = mSecondMap.applyMap(Vec3d(0,1,0)) - trans;
        Vec3d tmp3 = mSecondMap.applyMap(Vec3d(0,0,1)) - trans;

        /// false if there is shear
        if (!isApproxEqual(tmp1.dot(tmp2), 0., 1.e-7)) { mHasSimpleAffine  = false; return; }
        if (!isApproxEqual(tmp2.dot(tmp3), 0., 1.e-7)) { mHasSimpleAffine  = false; return; }
        if (!isApproxEqual(tmp3.dot(tmp1), 0., 1.e-7)) { mHasSimpleAffine  = false; return; }
    }

    Vec3d applyFrustumMap(const Vec3d& in) const
    {

        // Move the center of the x-face of the bbox
        // to the origin in index space.
        Vec3d out(in);
        out = out - mBBox.min();
        out.x() -= mXo;
        out.y() -= mYo;

        // scale the z-direction on depth / K count
        out.z() *= mDepthOnLz;

        double scale = (mGamma * out.z() + 1.)/ mLx;

        // scale the x-y on the length I count and apply tapper
        out.x() *= scale ;
        out.y() *= scale ;

        return out;
    }

    Vec3d applyFrustumInverseMap(const Vec3d& in) const
    {
        // invert taper and resize:  scale = 1/( (z+1)/2 (mt-1) + 1)
        Vec3d out(in);
        double invScale = mLx / (mGamma * out.z() + 1.);
        out.x() *= invScale;
        out.y() *= invScale;

        out.x() += mXo;
        out.y() += mYo;

        out.z() /= mDepthOnLz;

        // move back
        out = out +  mBBox.min();
        return out;
    }

    // bounding box in index space used in Frustum transforms.
    BBoxd   mBBox;

    // taper value used in constructing Frustums.
    double      mTaper;
    double      mDepth;

    // defines the second map
    AffineMap mSecondMap;

    // these are derived from the above.
    double mLx, mLy, mLz;
    double mXo, mYo, mGamma, mDepthOnLz, mDepthOnLzLxLx;

    // true: if the mSecondMap is linear and has no shear, and has no non-uniform scale
    bool mHasSimpleAffine;
}; // class NonlinearFrustumMap


////////////////////////////////////////


///  @brief Creates the composition of two maps, each of which could be a composition.
///  In the case that each component of the composition classified as linear an
///  acceleration AffineMap is stored.
template<typename FirstMapType, typename SecondMapType>
class CompoundMap
{
public:
    using MyType = CompoundMap<FirstMapType, SecondMapType>;

    using Ptr = SharedPtr<MyType>;
    using ConstPtr = SharedPtr<const MyType>;


    CompoundMap() { updateAffineMatrix(); }

    CompoundMap(const FirstMapType& f, const SecondMapType& s): mFirstMap(f), mSecondMap(s)
    {
        updateAffineMatrix();
    }

    CompoundMap(const MyType& other):
        mFirstMap(other.mFirstMap),
        mSecondMap(other.mSecondMap),
        mAffineMap(other.mAffineMap)
    {}

    Name type() const { return mapType(); }
    static Name mapType()
    {
        return (FirstMapType::mapType() + Name(":") + SecondMapType::mapType());
    }

    bool operator==(const MyType& other) const
    {
        if (mFirstMap != other.mFirstMap)   return false;
        if (mSecondMap != other.mSecondMap) return false;
        if (mAffineMap != other.mAffineMap) return false;
        return true;
    }

    bool operator!=(const MyType& other) const { return !(*this == other); }

    MyType& operator=(const MyType& other)
    {
        mFirstMap = other.mFirstMap;
        mSecondMap = other.mSecondMap;
        mAffineMap = other.mAffineMap;
        return *this;
    }

    bool isIdentity() const
    {
        if (is_linear<MyType>::value) {
            return mAffineMap.isIdentity();
        } else {
            return mFirstMap.isIdentity()&&mSecondMap.isIdentity();
        }
    }

    bool isDiagonal() const {
        if (is_linear<MyType>::value) {
            return mAffineMap.isDiagonal();
        } else {
            return mFirstMap.isDiagonal()&&mSecondMap.isDiagonal();
        }
    }

    AffineMap::Ptr getAffineMap() const
    {
        if (is_linear<MyType>::value) {
            AffineMap::Ptr affine(new AffineMap(mAffineMap));
            return affine;
        } else {
            OPENVDB_THROW(ArithmeticError,
                "Constant affine matrix representation not possible for this nonlinear map");
        }
    }

    // direct decompotion
    const FirstMapType& firstMap() const { return mFirstMap; }
    const SecondMapType& secondMap() const {return mSecondMap; }

    void setFirstMap(const FirstMapType& first) { mFirstMap = first; updateAffineMatrix(); }
    void setSecondMap(const SecondMapType& second) { mSecondMap = second; updateAffineMatrix(); }

    void read(std::istream& is)
    {
        mAffineMap.read(is);
        mFirstMap.read(is);
        mSecondMap.read(is);
    }
    void write(std::ostream& os) const
    {
        mAffineMap.write(os);
        mFirstMap.write(os);
        mSecondMap.write(os);
    }

private:
    void updateAffineMatrix()
    {
        if (is_linear<MyType>::value) {
            // both maps need to be linear, these methods are only defined for linear maps
            AffineMap::Ptr first = mFirstMap.getAffineMap();
            AffineMap::Ptr second= mSecondMap.getAffineMap();
            mAffineMap = AffineMap(*first, *second);
        }
    }

    FirstMapType   mFirstMap;
    SecondMapType  mSecondMap;
    // used for acceleration
    AffineMap      mAffineMap;
}; // class CompoundMap

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_MAPS_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
