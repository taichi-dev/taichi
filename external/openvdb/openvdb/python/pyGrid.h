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
//
/// @file pyGrid.h
/// @author Peter Cucka
/// @brief Boost.Python wrapper for openvdb::Grid

#ifndef OPENVDB_PYGRID_HAS_BEEN_INCLUDED
#define OPENVDB_PYGRID_HAS_BEEN_INCLUDED

#include <boost/python.hpp>
#ifndef DWA_BOOST_VERSION
#include <boost/version.hpp>
#define DWA_BOOST_VERSION (10 * BOOST_VERSION)
#endif
#ifdef PY_OPENVDB_USE_NUMPY
  #if DWA_BOOST_VERSION >= 1065000
    // boost::python::numeric was replaced with boost::python::numpy in Boost 1.65.
    // (boost::python::numpy requires NumPy 1.7 or later.)
    #include <boost/python/numpy.hpp>
    //#include <arrayobject.h> // for PyArray_Descr (see pyGrid::arrayTypeId())
    #define PY_OPENVDB_USE_BOOST_PYTHON_NUMPY
  #else
    #define PY_ARRAY_UNIQUE_SYMBOL PY_OPENVDB_ARRAY_API
    #define NO_IMPORT_ARRAY // NumPy gets initialized during module initialization
    #include <numpyconfig.h>
    #ifdef NPY_1_7_API_VERSION
      #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    #endif
    #include <arrayobject.h> // for PyArrayObject
  #endif
  #include "openvdb/tools/MeshToVolume.h"
  #include "openvdb/tools/VolumeToMesh.h" // for tools::volumeToMesh()
#endif
#include "openvdb/openvdb.h"
#include "openvdb/io/Stream.h"
#include "openvdb/math/Math.h" // for math::isExactlyEqual()
#include "openvdb/tools/LevelSetSphere.h"
#include "openvdb/tools/Dense.h"
#include "openvdb/tools/ChangeBackground.h"
#include "openvdb/tools/Prune.h"
#include "openvdb/tools/SignedFloodFill.h"
#include "pyutil.h"
#include "pyAccessor.h" // for pyAccessor::AccessorWrap
#include "pyopenvdb.h"
#include <algorithm> // for std::max()
#include <cstring> // for memcpy()
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace py = boost::python;

#ifdef __clang__
// This is a private header, so it's OK to include a "using namespace" directive.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wheader-hygiene"
#endif

using namespace openvdb::OPENVDB_VERSION_NAME;

#ifdef __clang__
#pragma clang diagnostic pop
#endif


namespace pyopenvdb {

inline py::object
getPyObjectFromGrid(const GridBase::Ptr& grid)
{
    if (!grid) return py::object();

#define CONVERT_BASE_TO_GRID(GridType, grid) \
    if (grid->isType<GridType>()) { \
        return py::object(gridPtrCast<GridType>(grid)); \
    }

    CONVERT_BASE_TO_GRID(FloatGrid, grid);
    CONVERT_BASE_TO_GRID(Vec3SGrid, grid);
    CONVERT_BASE_TO_GRID(BoolGrid, grid);
#ifdef PY_OPENVDB_WRAP_ALL_GRID_TYPES
    CONVERT_BASE_TO_GRID(DoubleGrid, grid);
    CONVERT_BASE_TO_GRID(Int32Grid, grid);
    CONVERT_BASE_TO_GRID(Int64Grid, grid);
    CONVERT_BASE_TO_GRID(Vec3IGrid, grid);
    CONVERT_BASE_TO_GRID(Vec3DGrid, grid);
#endif
#undef CONVERT_BASE_TO_GRID

    OPENVDB_THROW(TypeError, grid->type() + " is not a supported OpenVDB grid type");
}


inline openvdb::GridBase::Ptr
getGridFromPyObject(const boost::python::object& gridObj)
{
    if (!gridObj) return GridBase::Ptr();

#define CONVERT_GRID_TO_BASE(GridPtrType) \
    { \
        py::extract<GridPtrType> x(gridObj); \
        if (x.check()) return x(); \
    }

    // Extract a grid pointer of one of the supported types
    // from the input object, then cast it to a base pointer.
    CONVERT_GRID_TO_BASE(FloatGrid::Ptr);
    CONVERT_GRID_TO_BASE(Vec3SGrid::Ptr);
    CONVERT_GRID_TO_BASE(BoolGrid::Ptr);
#ifdef PY_OPENVDB_WRAP_ALL_GRID_TYPES
    CONVERT_GRID_TO_BASE(DoubleGrid::Ptr);
    CONVERT_GRID_TO_BASE(Int32Grid::Ptr);
    CONVERT_GRID_TO_BASE(Int64Grid::Ptr);
    CONVERT_GRID_TO_BASE(Vec3IGrid::Ptr);
    CONVERT_GRID_TO_BASE(Vec3DGrid::Ptr);
#endif
#undef CONVERT_GRID_TO_BASE

    OPENVDB_THROW(TypeError,
        pyutil::className(gridObj) + " is not a supported OpenVDB grid type");
}


inline openvdb::GridBase::Ptr
getGridFromPyObject(PyObject* gridObj)
{
    return getGridFromPyObject(pyutil::pyBorrow(gridObj));
}

} // namespace pyopenvdb


////////////////////////////////////////


namespace pyGrid {

inline py::object
getGridFromGridBase(GridBase::Ptr grid)
{
    py::object obj;
    try {
        obj = pyopenvdb::getPyObjectFromGrid(grid);
    } catch (openvdb::TypeError& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        py::throw_error_already_set();
        return py::object();
    }
    return obj;
}


/// GridBase is not exposed in Python because it isn't really needed
/// (and because exposing it would be complicated, requiring wrapping
/// pure virtual functions like GridBase::baseTree()), but there are
/// a few cases where, internally, we need to extract a GridBase::Ptr
/// from a py::object.  Hence this converter.
inline GridBase::Ptr
getGridBaseFromGrid(py::object gridObj)
{
    GridBase::Ptr grid;
    try {
        grid = pyopenvdb::getGridFromPyObject(gridObj);
    } catch (openvdb::TypeError& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        py::throw_error_already_set();
        return GridBase::Ptr();
    }
    return grid;
}


////////////////////////////////////////


/// Variant of pyutil::extractArg() that uses the class name of a given grid type
template<typename GridType, typename T>
inline T
extractValueArg(
    py::object obj,
    const char* functionName,
    int argIdx = 0, // args are numbered starting from 1
    const char* expectedType = nullptr)
{
    return pyutil::extractArg<T>(obj,
        functionName, pyutil::GridTraits<GridType>::name(), argIdx, expectedType);
}


/// @brief Variant of pyutil::extractArg() that uses the class name
/// and @c ValueType of a given grid type
template<typename GridType>
inline typename GridType::ValueType
extractValueArg(
    py::object obj,
    const char* functionName,
    int argIdx = 0, // args are numbered starting from 1
    const char* expectedType = nullptr)
{
    return extractValueArg<GridType, typename GridType::ValueType>(
        obj, functionName, argIdx, expectedType);
}


////////////////////////////////////////


template<typename GridType>
inline typename GridType::Ptr
copyGrid(GridType& grid)
{
    return grid.copy();
}


template<typename GridType>
inline bool
sharesWith(const GridType& grid, py::object other)
{
    py::extract<typename GridType::Ptr> x(other);
    if (x.check()) {
        typename GridType::ConstPtr otherGrid = x();
        return (&otherGrid->tree() == &grid.tree());
    }
    return false;
}


////////////////////////////////////////


template<typename GridType>
inline std::string
getValueType()
{
    return pyutil::GridTraits<GridType>::valueTypeName();
}


template<typename GridType>
inline typename GridType::ValueType
getZeroValue()
{
    return openvdb::zeroVal<typename GridType::ValueType>();
}


template<typename GridType>
inline typename GridType::ValueType
getOneValue()
{
    using ValueT = typename GridType::ValueType;
    return ValueT(openvdb::zeroVal<ValueT>() + 1);
}


template<typename GridType>
inline bool
notEmpty(const GridType& grid)
{
    return !grid.empty();
}


template<typename GridType>
inline typename GridType::ValueType
getGridBackground(const GridType& grid)
{
    return grid.background();
}


template<typename GridType>
inline void
setGridBackground(GridType& grid, py::object obj)
{
    tools::changeBackground(grid.tree(), extractValueArg<GridType>(obj, "setBackground"));
}


inline void
setGridName(GridBase::Ptr grid, py::object strObj)
{
    if (grid) {
        if (!strObj) { // if name is None
            grid->removeMeta(GridBase::META_GRID_NAME);
        } else {
            const std::string name = pyutil::extractArg<std::string>(
                strObj, "setName", /*className=*/nullptr, /*argIdx=*/1, "str");
            grid->setName(name);
        }
    }
}


inline void
setGridCreator(GridBase::Ptr grid, py::object strObj)
{
    if (grid) {
        if (!strObj) { // if name is None
            grid->removeMeta(GridBase::META_GRID_CREATOR);
        } else {
            const std::string name = pyutil::extractArg<std::string>(
                strObj, "setCreator", /*className=*/nullptr, /*argIdx=*/1, "str");
            grid->setCreator(name);
        }
    }
}


inline std::string
getGridClass(GridBase::ConstPtr grid)
{
    return GridBase::gridClassToString(grid->getGridClass());
}


inline void
setGridClass(GridBase::Ptr grid, py::object strObj)
{
    if (!strObj) {
        grid->clearGridClass();
    } else {
        const std::string name = pyutil::extractArg<std::string>(
            strObj, "setGridClass", /*className=*/nullptr, /*argIdx=*/1, "str");
        grid->setGridClass(GridBase::stringToGridClass(name));
    }
}


inline std::string
getVecType(GridBase::ConstPtr grid)
{
    return GridBase::vecTypeToString(grid->getVectorType());
}


inline void
setVecType(GridBase::Ptr grid, py::object strObj)
{
    if (!strObj) {
        grid->clearVectorType();
    } else {
        const std::string name = pyutil::extractArg<std::string>(
            strObj, "setVectorType", /*className=*/nullptr, /*argIdx=*/1, "str");
        grid->setVectorType(GridBase::stringToVecType(name));
    }
}


inline std::string
gridInfo(GridBase::ConstPtr grid, int verbosity)
{
    std::ostringstream ostr;
    grid->print(ostr, std::max<int>(1, verbosity));
    return ostr.str();
}


////////////////////////////////////////


inline void
setGridTransform(GridBase::Ptr grid, py::object xformObj)
{
    if (grid) {
        if (math::Transform::Ptr xform = pyutil::extractArg<math::Transform::Ptr>(
            xformObj, "setTransform", /*className=*/nullptr, /*argIdx=*/1, "Transform"))
        {
            grid->setTransform(xform);
        } else {
            PyErr_SetString(PyExc_ValueError, "null transform");
            py::throw_error_already_set();
        }
    }
}


////////////////////////////////////////


// Helper class to construct a pyAccessor::AccessorWrap for a given grid,
// permitting partial specialization for const vs. non-const grids
template<typename GridType>
struct AccessorHelper
{
    using Wrapper = typename pyAccessor::AccessorWrap<GridType>;
    static Wrapper wrap(typename GridType::Ptr grid)
    {
        if (!grid) {
            PyErr_SetString(PyExc_ValueError, "null grid");
            py::throw_error_already_set();
        }
        return Wrapper(grid);
    }
};

// Specialization for const grids
template<typename GridType>
struct AccessorHelper<const GridType>
{
    using Wrapper = typename pyAccessor::AccessorWrap<const GridType>;
    static Wrapper wrap(typename GridType::ConstPtr grid)
    {
        if (!grid) {
            PyErr_SetString(PyExc_ValueError, "null grid");
            py::throw_error_already_set();
        }
        return Wrapper(grid);
    }
};


/// Return a non-const accessor (wrapped in a pyAccessor::AccessorWrap) for the given grid.
template<typename GridType>
inline typename AccessorHelper<GridType>::Wrapper
getAccessor(typename GridType::Ptr grid)
{
    return AccessorHelper<GridType>::wrap(grid);
}

/// @brief Return a const accessor (wrapped in a pyAccessor::AccessorWrap) for the given grid.
/// @internal Note that the grid pointer is non-const, even though the grid is
/// treated as const.  This is because we don't expose a const grid type in Python.
template<typename GridType>
inline typename AccessorHelper<const GridType>::Wrapper
getConstAccessor(typename GridType::Ptr grid)
{
    return AccessorHelper<const GridType>::wrap(grid);
}


////////////////////////////////////////


template<typename GridType>
inline py::tuple
evalLeafBoundingBox(const GridType& grid)
{
    CoordBBox bbox;
    grid.tree().evalLeafBoundingBox(bbox);
    return py::make_tuple(bbox.min(), bbox.max());
}


template<typename GridType>
inline Coord
evalLeafDim(const GridType& grid)
{
    Coord dim;
    grid.tree().evalLeafDim(dim);
    return dim;
}


template<typename GridType>
inline py::tuple
evalActiveVoxelBoundingBox(const GridType& grid)
{
    CoordBBox bbox = grid.evalActiveVoxelBoundingBox();
    return py::make_tuple(bbox.min(), bbox.max());
}


template<typename GridType>
inline py::tuple
getNodeLog2Dims(const GridType& grid)
{
    std::vector<Index> dims;
    grid.tree().getNodeLog2Dims(dims);
    py::list lst;
    for (size_t i = 0, N = dims.size(); i < N; ++i) {
        lst.append(dims[i]);
    }
    return py::tuple(lst);
}


template<typename GridType>
inline Index
treeDepth(const GridType& grid)
{
    return grid.tree().treeDepth();
}


template<typename GridType>
inline Index32
leafCount(const GridType& grid)
{
    return grid.tree().leafCount();
}


template<typename GridType>
inline Index32
nonLeafCount(const GridType& grid)
{
    return grid.tree().nonLeafCount();
}


template<typename GridType>
inline Index64
activeLeafVoxelCount(const GridType& grid)
{
    return grid.tree().activeLeafVoxelCount();
}


template<typename GridType>
inline py::tuple
evalMinMax(const GridType& grid)
{
    typename GridType::ValueType vmin, vmax;
    grid.tree().evalMinMax(vmin, vmax);
    return py::make_tuple(vmin, vmax);
}


template<typename GridType>
inline py::tuple
getIndexRange(const GridType& grid)
{
    CoordBBox bbox;
    grid.tree().getIndexRange(bbox);
    return py::make_tuple(bbox.min(), bbox.max());
}


//template<typename GridType>
//inline void
//expandIndexRange(GridType& grid, py::object coordObj)
//{
//    Coord xyz = extractValueArg<GridType, Coord>(
//        coordObj, "expand", 0, "tuple(int, int, int)");
//    grid.tree().expand(xyz);
//}


////////////////////////////////////////


inline py::dict
getAllMetadata(GridBase::ConstPtr grid)
{
    if (grid) return py::dict(static_cast<const MetaMap&>(*grid));
    return py::dict();
}


inline void
replaceAllMetadata(GridBase::Ptr grid, const MetaMap& metadata)
{
    if (grid) {
        grid->clearMetadata();
        for (MetaMap::ConstMetaIterator it = metadata.beginMeta();
            it != metadata.endMeta(); ++it)
        {
            if (it->second) grid->insertMeta(it->first, *it->second);
        }
    }
}


inline void
updateMetadata(GridBase::Ptr grid, const MetaMap& metadata)
{
    if (grid) {
        for (MetaMap::ConstMetaIterator it = metadata.beginMeta();
            it != metadata.endMeta(); ++it)
        {
            if (it->second) {
                grid->removeMeta(it->first);
                grid->insertMeta(it->first, *it->second);
            }
        }
    }
}


inline py::dict
getStatsMetadata(GridBase::ConstPtr grid)
{
    MetaMap::ConstPtr metadata;
    if (grid) metadata = grid->getStatsMetadata();
    if (metadata) return py::dict(*metadata);
    return py::dict();
}


inline py::object
getMetadataKeys(GridBase::ConstPtr grid)
{
    if (grid) {
#if PY_MAJOR_VERSION >= 3
        // Return an iterator over the "keys" view of a dict.
        return py::import("builtins").attr("iter")(
            py::dict(static_cast<const MetaMap&>(*grid)).keys());
#else
        return py::dict(static_cast<const MetaMap&>(*grid)).iterkeys();
#endif
    }
    return py::object();
}


inline py::object
getMetadata(GridBase::ConstPtr grid, py::object nameObj)
{
    if (!grid) return py::object();

    const std::string name = pyutil::extractArg<std::string>(
        nameObj, "__getitem__", nullptr, /*argIdx=*/1, "str");

    Metadata::ConstPtr metadata = (*grid)[name];
    if (!metadata) {
        PyErr_SetString(PyExc_KeyError, name.c_str());
        py::throw_error_already_set();
    }

    // Use the MetaMap-to-dict converter (see pyOpenVDBModule.cc) to convert
    // the Metadata value to a Python object of the appropriate type.
    /// @todo Would be more efficient to convert the Metadata object
    /// directly to a Python object.
    MetaMap metamap;
    metamap.insertMeta(name, *metadata);
    return py::dict(metamap)[name];
}


inline void
setMetadata(GridBase::Ptr grid, py::object nameObj, py::object valueObj)
{
    if (!grid) return;

    const std::string name = pyutil::extractArg<std::string>(
        nameObj, "__setitem__", nullptr, /*argIdx=*/1, "str");

    // Insert the Python object into a Python dict, then use the dict-to-MetaMap
    // converter (see pyOpenVDBModule.cc) to convert the dict to a MetaMap
    // containing a Metadata object of the appropriate type.
    /// @todo Would be more efficient to convert the Python object
    /// directly to a Metadata object.
    py::dict dictObj;
    dictObj[name] = valueObj;
    MetaMap metamap = py::extract<MetaMap>(dictObj);

    if (Metadata::Ptr metadata = metamap[name]) {
        grid->removeMeta(name);
        grid->insertMeta(name, *metadata);
    }
}


inline void
removeMetadata(GridBase::Ptr grid, const std::string& name)
{
    if (grid) {
        Metadata::Ptr metadata = (*grid)[name];
        if (!metadata) {
            PyErr_SetString(PyExc_KeyError, name.c_str());
            py::throw_error_already_set();
        }
        grid->removeMeta(name);
    }
}


inline bool
hasMetadata(GridBase::ConstPtr grid, const std::string& name)
{
    if (grid) return ((*grid)[name].get() != nullptr);
    return false;
}


////////////////////////////////////////


template<typename GridType>
inline void
prune(GridType& grid, py::object tolerance)
{
    tools::prune(grid.tree(), extractValueArg<GridType>(tolerance, "prune"));
}


template<typename GridType>
inline void
pruneInactive(GridType& grid, py::object valObj)
{
    if (valObj.is_none()) {
        tools::pruneInactive(grid.tree());
    } else {
        tools::pruneInactiveWithValue(
            grid.tree(), extractValueArg<GridType>(valObj, "pruneInactive"));
    }
}


template<typename GridType>
inline void
fill(GridType& grid, py::object minObj, py::object maxObj,
    py::object valObj, bool active)
{
    const Coord
        bmin = extractValueArg<GridType, Coord>(minObj, "fill", 1, "tuple(int, int, int)"),
        bmax = extractValueArg<GridType, Coord>(maxObj, "fill", 2, "tuple(int, int, int)");
    grid.fill(CoordBBox(bmin, bmax), extractValueArg<GridType>(valObj, "fill", 3), active);
}


template<typename GridType>
inline void
signedFloodFill(GridType& grid)
{
    tools::signedFloodFill(grid.tree());
}


////////////////////////////////////////


#ifndef PY_OPENVDB_USE_NUMPY

template<typename GridType>
inline void
copyFromArray(GridType&, const py::object&, py::object, py::object)
{
    PyErr_SetString(PyExc_NotImplementedError, "this module was built without NumPy support");
    boost::python::throw_error_already_set();
}

template<typename GridType>
inline void
copyToArray(GridType&, const py::object&, py::object)
{
    PyErr_SetString(PyExc_NotImplementedError, "this module was built without NumPy support");
    boost::python::throw_error_already_set();
}

#else // if defined(PY_OPENVDB_USE_NUMPY)

using ArrayDimVec = std::vector<size_t>;

#ifdef PY_OPENVDB_USE_BOOST_PYTHON_NUMPY

// ID numbers for supported value types
enum class DtId { NONE, FLOAT, DOUBLE, BOOL, INT16, INT32, INT64, UINT32, UINT64/*, HALF*/ };

using NumPyArrayType = py::numpy::ndarray;

#else // if !defined PY_OPENVDB_USE_BOOST_PYTHON_NUMPY

// NumPy type numbers for supported value types
enum class DtId {
    NONE =   NPY_NOTYPE,
    FLOAT =  NPY_FLOAT,
    DOUBLE = NPY_DOUBLE,
    BOOL =   NPY_BOOL,
    INT16 =  NPY_INT16,
    INT32 =  NPY_INT32,
    INT64 =  NPY_INT64,
    UINT32 = NPY_UINT32,
    UINT64 = NPY_UINT64,
    //HALF =   NPY_HALF
};

using NumPyArrayType = py::numeric::array;

#endif // PY_OPENVDB_USE_BOOST_PYTHON_NUMPY


template<DtId TypeId> struct NumPyToCpp {};
template<> struct NumPyToCpp<DtId::FLOAT>  { using type = float; };
template<> struct NumPyToCpp<DtId::DOUBLE> { using type = double; };
template<> struct NumPyToCpp<DtId::BOOL>   { using type = bool; };
template<> struct NumPyToCpp<DtId::INT16>  { using type = Int16; };
template<> struct NumPyToCpp<DtId::INT32>  { using type = Int32; };
template<> struct NumPyToCpp<DtId::INT64>  { using type = Int64; };
template<> struct NumPyToCpp<DtId::UINT32> { using type = Index32; };
template<> struct NumPyToCpp<DtId::UINT64> { using type = Index64; };
//template<> struct NumPyToCpp<DtId::HALF>   { using type = half; };


#if 0
template<typename T> struct CppToNumPy { static const DtId typeId = DtId::NONE; };
template<> struct CppToNumPy<float>    { static const DtId typeId = DtId::FLOAT; };
template<> struct CppToNumPy<double>   { static const DtId typeId = DtId::DOUBLE; };
template<> struct CppToNumPy<bool>     { static const DtId typeId = DtId::BOOL; };
template<> struct CppToNumPy<Int16>    { static const DtId typeId = DtId::INT16; };
template<> struct CppToNumPy<Int32>    { static const DtId typeId = DtId::INT32; };
template<> struct CppToNumPy<Int64>    { static const DtId typeId = DtId::INT64; };
template<> struct CppToNumPy<Index32>  { static const DtId typeId = DtId::UINT32; };
template<> struct CppToNumPy<Index64>  { static const DtId typeId = DtId::UINT64; };
//template<> struct CppToNumPy<half>     { static const DtId typeId = DtId::HALF; };
#endif


#ifdef PY_OPENVDB_USE_BOOST_PYTHON_NUMPY

// Return the ID number of the given NumPy array's data type.
/// @todo Revisit this if and when py::numpy::dtype ever provides a type number accessor.
inline DtId
arrayTypeId(const py::numpy::ndarray& arrayObj)
{
    namespace np = py::numpy;
    const auto dtype = arrayObj.get_dtype();
#if 0
    // More efficient than np::equivalent(), but requires NumPy headers.
    if (const auto* descr = reinterpret_cast<const PyArray_Descr*>(dtype.ptr())) {
        const auto typeId = static_cast<DtId>(descr->type_num);
        switch (typeId) {
            case DtId::NONE: break;
            case DtId::FLOAT: case DtId::DOUBLE: case DtId::BOOL:
            case DtId::INT16: case DtId::INT32: case DtId::INT64:
            case DtId::UINT32: case DtId::UINT64:
                return typeId;
        }
        throw openvdb::TypeError{};
    }
#else
    if (np::equivalent(dtype, np::dtype::get_builtin<float>())) return DtId::FLOAT;
    if (np::equivalent(dtype, np::dtype::get_builtin<double>())) return DtId::DOUBLE;
    if (np::equivalent(dtype, np::dtype::get_builtin<bool>())) return DtId::BOOL;
    if (np::equivalent(dtype, np::dtype::get_builtin<Int16>())) return DtId::INT16;
    if (np::equivalent(dtype, np::dtype::get_builtin<Int32>())) return DtId::INT32;
    if (np::equivalent(dtype, np::dtype::get_builtin<Int64>())) return DtId::INT64;
    if (np::equivalent(dtype, np::dtype::get_builtin<Index32>())) return DtId::UINT32;
    if (np::equivalent(dtype, np::dtype::get_builtin<Index64>())) return DtId::UINT64;
    //if (np::equivalent(dtype, np::dtype::get_builtin<half>())) return DtId::HALF;
#endif
    throw openvdb::TypeError{};
}


// Return a string description of the given NumPy array's data type.
inline std::string
arrayTypeName(const py::numpy::ndarray& arrayObj)
{
    return pyutil::str(arrayObj.get_dtype());
}


// Return the dimensions of the given NumPy array.
inline ArrayDimVec
arrayDimensions(const py::numpy::ndarray& arrayObj)
{
    ArrayDimVec dims;
    for (int i = 0, N = arrayObj.get_nd(); i < N; ++i) {
        dims.push_back(static_cast<size_t>(arrayObj.shape(i)));
    }
    return dims;
}

#else // !defined PY_OPENVDB_USE_BOOST_PYTHON_NUMPY

// Return the ID number of the given NumPy array's data type.
inline DtId
arrayTypeId(const py::numeric::array& arrayObj)
{
    const PyArray_Descr* dtype = nullptr;
    if (PyArrayObject* arrayObjPtr = reinterpret_cast<PyArrayObject*>(arrayObj.ptr())) {
        dtype = PyArray_DESCR(arrayObjPtr);
    }
    if (dtype) return static_cast<DtId>(dtype->type_num);
    throw openvdb::TypeError{};
}


// Return a string description of the given NumPy array's data type.
inline std::string
arrayTypeName(const py::numeric::array& arrayObj)
{
    std::string name;
    if (PyObject_HasAttrString(arrayObj.ptr(), "dtype")) {
        name = pyutil::str(arrayObj.attr("dtype"));
    } else {
        name = "'_'";
        PyArrayObject* arrayObjPtr = reinterpret_cast<PyArrayObject*>(arrayObj.ptr());
        name[1] = PyArray_DESCR(arrayObjPtr)->kind;
    }
    return name;
}


// Return the dimensions of the given NumPy array.
inline ArrayDimVec
arrayDimensions(const py::numeric::array& arrayObj)
{
    const py::object shape = arrayObj.attr("shape");
    ArrayDimVec dims;
    for (long i = 0, N = py::len(shape); i < N; ++i) {
        dims.push_back(py::extract<size_t>(shape[i]));
    }
    return dims;
}


inline py::object
copyNumPyArray(PyArrayObject* arrayObj, NPY_ORDER order = NPY_CORDER)
{
#ifdef __GNUC__
    // Silence GCC "casting between pointer-to-function and pointer-to-object" warnings.
    __extension__
#endif
    auto obj = pyutil::pyBorrow(PyArray_NewCopy(arrayObj, order));
    return obj;
}

#endif // PY_OPENVDB_USE_BOOST_PYTHON_NUMPY


// Abstract base class for helper classes that copy data between
// NumPy arrays of various types and grids of various types
template<typename GridType>
class CopyOpBase
{
public:
    using ValueT = typename GridType::ValueType;

    CopyOpBase(bool toGrid, GridType& grid, py::object arrObj,
        py::object coordObj, py::object tolObj)
        : mToGrid(toGrid)
        , mGrid(&grid)
    {
        const char* const opName[2] = { "copyToArray", "copyFromArray" };

        // Extract the coordinates (i, j, k) of the voxel at which to start populating data.
        // Voxel (i, j, k) will correspond to array element (0, 0, 0).
        const Coord origin = extractValueArg<GridType, Coord>(
            coordObj, opName[toGrid], 1, "tuple(int, int, int)");

        // Extract a reference to (not a copy of) the NumPy array,
        // or throw an exception if arrObj is not a NumPy array object.
        const auto arrayObj = pyutil::extractArg<NumPyArrayType>(
            arrObj, opName[toGrid], pyutil::GridTraits<GridType>::name(),
            /*argIdx=*/1, "numpy.ndarray");

#ifdef PY_OPENVDB_USE_BOOST_PYTHON_NUMPY
        mArray = arrayObj.get_data();
#else
        mArray = PyArray_DATA(reinterpret_cast<PyArrayObject*>(arrayObj.ptr()));
#endif

        mArrayTypeName = arrayTypeName(arrayObj);
        mArrayTypeId = arrayTypeId(arrayObj);
        mArrayDims = arrayDimensions(arrayObj);

        mTolerance = extractValueArg<GridType>(tolObj, opName[toGrid], 2);

        // Compute the bounding box of the region of the grid that is to be copied from or to.
        Coord bboxMax = origin;
        for (size_t n = 0, N = std::min<size_t>(mArrayDims.size(), 3); n < N; ++n) {
            bboxMax[n] += int(mArrayDims[n]) - 1;
        }
        mBBox.reset(origin, bboxMax);
    }
    virtual ~CopyOpBase() {}

    void operator()() const
    {
        try {
            if (mToGrid) {
                copyFromArray(); // copy data from the array to the grid
            } else {
                copyToArray(); // copy data from the grid to the array
            }
        } catch (openvdb::TypeError&) {
            PyErr_Format(PyExc_TypeError,
                "unsupported NumPy data type %s", mArrayTypeName.c_str());
            boost::python::throw_error_already_set();
        }
    }

protected:
    virtual void validate() const = 0;
    virtual void copyFromArray() const = 0;
    virtual void copyToArray() const = 0;

    template<typename ArrayValueType>
    void fromArray() const
    {
        validate();
        tools::Dense<ArrayValueType> valArray(mBBox, static_cast<ArrayValueType*>(mArray));
        tools::copyFromDense(valArray, *mGrid, mTolerance);
    }

    template<typename ArrayValueType>
    void toArray() const
    {
        validate();
        tools::Dense<ArrayValueType> valArray(mBBox, static_cast<ArrayValueType*>(mArray));
        tools::copyToDense(*mGrid, valArray);
    }


    bool mToGrid; // if true, copy from the array to the grid, else vice-versa
    void* mArray;
    GridType* mGrid;
    DtId mArrayTypeId;
    ArrayDimVec mArrayDims;
    std::string mArrayTypeName;
    CoordBBox mBBox;
    ValueT mTolerance;
}; // class CopyOpBase


// Helper subclass that can be specialized for various grid and NumPy array types
template<typename GridType, int VecSize> class CopyOp: public CopyOpBase<GridType> {};

// Specialization for scalar grids
template<typename GridType>
class CopyOp<GridType, /*VecSize=*/1>: public CopyOpBase<GridType>
{
public:
    CopyOp(bool toGrid, GridType& grid, py::object arrObj, py::object coordObj,
        py::object tolObj = py::object(zeroVal<typename GridType::ValueType>())):
        CopyOpBase<GridType>(toGrid, grid, arrObj, coordObj, tolObj)
    {
    }

protected:
    void validate() const override
    {
        if (this->mArrayDims.size() != 3) {
            std::ostringstream os;
            os << "expected 3-dimensional array, found "
                << this->mArrayDims.size() << "-dimensional array";
            PyErr_SetString(PyExc_ValueError, os.str().c_str());
            boost::python::throw_error_already_set();
        }
    }

    void copyFromArray() const override
    {
        switch (this->mArrayTypeId) {
        case DtId::FLOAT: this->template fromArray<typename NumPyToCpp<DtId::FLOAT>::type>(); break;
        case DtId::DOUBLE:this->template fromArray<typename NumPyToCpp<DtId::DOUBLE>::type>();break;
        case DtId::BOOL:  this->template fromArray<typename NumPyToCpp<DtId::BOOL>::type>(); break;
        case DtId::INT16: this->template fromArray<typename NumPyToCpp<DtId::INT16>::type>(); break;
        case DtId::INT32: this->template fromArray<typename NumPyToCpp<DtId::INT32>::type>(); break;
        case DtId::INT64: this->template fromArray<typename NumPyToCpp<DtId::INT64>::type>(); break;
        case DtId::UINT32:this->template fromArray<typename NumPyToCpp<DtId::UINT32>::type>();break;
        case DtId::UINT64:this->template fromArray<typename NumPyToCpp<DtId::UINT64>::type>();break;
        default: throw openvdb::TypeError(); break;
        }
    }

    void copyToArray() const override
    {
        switch (this->mArrayTypeId) {
        case DtId::FLOAT:  this->template toArray<typename NumPyToCpp<DtId::FLOAT>::type>(); break;
        case DtId::DOUBLE: this->template toArray<typename NumPyToCpp<DtId::DOUBLE>::type>(); break;
        case DtId::BOOL:   this->template toArray<typename NumPyToCpp<DtId::BOOL>::type>(); break;
        case DtId::INT16:  this->template toArray<typename NumPyToCpp<DtId::INT16>::type>(); break;
        case DtId::INT32:  this->template toArray<typename NumPyToCpp<DtId::INT32>::type>(); break;
        case DtId::INT64:  this->template toArray<typename NumPyToCpp<DtId::INT64>::type>(); break;
        case DtId::UINT32: this->template toArray<typename NumPyToCpp<DtId::UINT32>::type>(); break;
        case DtId::UINT64: this->template toArray<typename NumPyToCpp<DtId::UINT64>::type>(); break;
        default: throw openvdb::TypeError(); break;
        }
    }
}; // class CopyOp

// Specialization for Vec3 grids
template<typename GridType>
class CopyOp<GridType, /*VecSize=*/3>: public CopyOpBase<GridType>
{
public:
    CopyOp(bool toGrid, GridType& grid, py::object arrObj, py::object coordObj,
        py::object tolObj = py::object(zeroVal<typename GridType::ValueType>())):
        CopyOpBase<GridType>(toGrid, grid, arrObj, coordObj, tolObj)
    {
    }

protected:
    void validate() const override
    {
        if (this->mArrayDims.size() != 4) {
            std::ostringstream os;
            os << "expected 4-dimensional array, found "
                << this->mArrayDims.size() << "-dimensional array";
            PyErr_SetString(PyExc_ValueError, os.str().c_str());
            boost::python::throw_error_already_set();
        }
        if (this->mArrayDims[3] != 3) {
            std::ostringstream os;
            os << "expected " << this->mArrayDims[0] << "x" << this->mArrayDims[1]
                << "x" << this->mArrayDims[2] << "x3 array, found " << this->mArrayDims[0]
                << "x" << this->mArrayDims[1] << "x" << this->mArrayDims[2]
                << "x" << this->mArrayDims[3] << " array";
            PyErr_SetString(PyExc_ValueError, os.str().c_str());
            boost::python::throw_error_already_set();
        }
    }

    void copyFromArray() const override
    {
        switch (this->mArrayTypeId) {
        case DtId::FLOAT:
            this->template fromArray<math::Vec3<typename NumPyToCpp<DtId::FLOAT>::type>>(); break;
        case DtId::DOUBLE:
            this->template fromArray<math::Vec3<typename NumPyToCpp<DtId::DOUBLE>::type>>(); break;
        case DtId::BOOL:
            this->template fromArray<math::Vec3<typename NumPyToCpp<DtId::BOOL>::type>>(); break;
        case DtId::INT16:
            this->template fromArray<math::Vec3<typename NumPyToCpp<DtId::INT16>::type>>(); break;
        case DtId::INT32:
            this->template fromArray<math::Vec3<typename NumPyToCpp<DtId::INT32>::type>>(); break;
        case DtId::INT64:
            this->template fromArray<math::Vec3<typename NumPyToCpp<DtId::INT64>::type>>(); break;
        case DtId::UINT32:
            this->template fromArray<math::Vec3<typename NumPyToCpp<DtId::UINT32>::type>>(); break;
        case DtId::UINT64:
            this->template fromArray<math::Vec3<typename NumPyToCpp<DtId::UINT64>::type>>(); break;
        default: throw openvdb::TypeError(); break;
        }
    }

    void copyToArray() const override
    {
        switch (this->mArrayTypeId) {
        case DtId::FLOAT:
            this->template toArray<math::Vec3<typename NumPyToCpp<DtId::FLOAT>::type>>(); break;
        case DtId::DOUBLE:
            this->template toArray<math::Vec3<typename NumPyToCpp<DtId::DOUBLE>::type>>(); break;
        case DtId::BOOL:
            this->template toArray<math::Vec3<typename NumPyToCpp<DtId::BOOL>::type>>(); break;
        case DtId::INT16:
            this->template toArray<math::Vec3<typename NumPyToCpp<DtId::INT16>::type>>(); break;
        case DtId::INT32:
            this->template toArray<math::Vec3<typename NumPyToCpp<DtId::INT32>::type>>(); break;
        case DtId::INT64:
            this->template toArray<math::Vec3<typename NumPyToCpp<DtId::INT64>::type>>(); break;
        case DtId::UINT32:
            this->template toArray<math::Vec3<typename NumPyToCpp<DtId::UINT32>::type>>(); break;
        case DtId::UINT64:
            this->template toArray<math::Vec3<typename NumPyToCpp<DtId::UINT64>::type>>(); break;
        default: throw openvdb::TypeError(); break;
        }
    }
}; // class CopyOp


template<typename GridType>
inline void
copyFromArray(GridType& grid, py::object arrayObj, py::object coordObj, py::object toleranceObj)
{
    using ValueT = typename GridType::ValueType;
    CopyOp<GridType, VecTraits<ValueT>::Size>
        op(/*toGrid=*/true, grid, arrayObj, coordObj, toleranceObj);
    op();
}


template<typename GridType>
inline void
copyToArray(GridType& grid, py::object arrayObj, py::object coordObj)
{
    using ValueT = typename GridType::ValueType;
    CopyOp<GridType, VecTraits<ValueT>::Size>
        op(/*toGrid=*/false, grid, arrayObj, coordObj);
    op();
}

#endif // defined(PY_OPENVDB_USE_NUMPY)


////////////////////////////////////////


#ifndef PY_OPENVDB_USE_NUMPY

template<typename GridType>
inline typename GridType::Ptr
meshToLevelSet(py::object, py::object, py::object, py::object, py::object)
{
    PyErr_SetString(PyExc_NotImplementedError, "this module was built without NumPy support");
    boost::python::throw_error_already_set();
    return typename GridType::Ptr();
}

template<typename GridType>
inline py::object
volumeToQuadMesh(const GridType&, py::object)
{
    PyErr_SetString(PyExc_NotImplementedError, "this module was built without NumPy support");
    boost::python::throw_error_already_set();
    return py::object();
}

template<typename GridType>
inline py::object
volumeToMesh(const GridType&, py::object, py::object)
{
    PyErr_SetString(PyExc_NotImplementedError, "this module was built without NumPy support");
    boost::python::throw_error_already_set();
    return py::object();
}

#else // if defined(PY_OPENVDB_USE_NUMPY)

// Helper class for meshToLevelSet()
template<typename SrcT, typename DstT>
struct CopyVecOp {
    void operator()(const void* srcPtr, DstT* dst, size_t count) {
        const SrcT* src = static_cast<const SrcT*>(srcPtr);
        for (size_t i = count; i > 0; --i, ++src, ++dst) {
            *dst = static_cast<DstT>(*src);
        }
    }
};
// Partial specialization for source and destination arrays of the same type
template<typename T>
struct CopyVecOp<T, T> {
    void operator()(const void* srcPtr, T* dst, size_t count) {
        const T* src = static_cast<const T*>(srcPtr);
        ::memcpy(dst, src, count * sizeof(T));
    }
};


// Helper function for use with meshToLevelSet() to copy vectors of various types
// and sizes from NumPy arrays to STL vectors
template<typename VecT>
inline void
copyVecArray(NumPyArrayType& arrayObj, std::vector<VecT>& vec)
{
    using ValueT = typename VecT::ValueType;

    // Get the input array dimensions.
    const auto dims = arrayDimensions(arrayObj);
    const size_t M = dims.empty() ? 0 : dims[0];
    const size_t N = VecT().numElements();
    if (M == 0 || N == 0) return;

    // Preallocate the output vector.
    vec.resize(M);

    // Copy values from the input array to the output vector (with type conversion, if necessary).
#ifdef PY_OPENVDB_USE_BOOST_PYTHON_NUMPY
    const void* src = arrayObj.get_data();
#else
    PyArrayObject* arrayObjPtr = reinterpret_cast<PyArrayObject*>(arrayObj.ptr());
    const void* src = PyArray_DATA(arrayObjPtr);
#endif
    ValueT* dst = &vec[0][0];
    switch (arrayTypeId(arrayObj)) {
    case DtId::FLOAT:  CopyVecOp<NumPyToCpp<DtId::FLOAT>::type, ValueT>()(src, dst, M*N); break;
    case DtId::DOUBLE: CopyVecOp<NumPyToCpp<DtId::DOUBLE>::type, ValueT>()(src, dst, M*N); break;
    case DtId::INT16:  CopyVecOp<NumPyToCpp<DtId::INT16>::type, ValueT>()(src, dst, M*N); break;
    case DtId::INT32:  CopyVecOp<NumPyToCpp<DtId::INT32>::type, ValueT>()(src, dst, M*N); break;
    case DtId::INT64:  CopyVecOp<NumPyToCpp<DtId::INT64>::type, ValueT>()(src, dst, M*N); break;
    case DtId::UINT32: CopyVecOp<NumPyToCpp<DtId::UINT32>::type, ValueT>()(src, dst, M*N); break;
    case DtId::UINT64: CopyVecOp<NumPyToCpp<DtId::UINT64>::type, ValueT>()(src, dst, M*N); break;
    default: break;
    }
}


/// @brief Given NumPy arrays of points, triangle indices and quad indices,
/// call tools::meshToLevelSet() to generate a level set grid.
template<typename GridType>
inline typename GridType::Ptr
meshToLevelSet(py::object pointsObj, py::object trianglesObj, py::object quadsObj,
    py::object xformObj, py::object halfWidthObj)
{
    struct Local {
        // Return the name of the Python grid method (for use in error messages).
        static const char* methodName() { return "createLevelSetFromPolygons"; }

        // Raise a Python exception if the given NumPy array does not have dimensions M x N
        // or does not have an integer or floating-point data type.
        static void validate2DNumPyArray(NumPyArrayType arrayObj,
            const size_t N, const char* desiredType)
        {
            const auto dims = arrayDimensions(arrayObj);

            bool wrongArrayType = false;
            // Check array dimensions.
            if (dims.size() != 2 || dims[1] != N) {
                wrongArrayType = true;
            } else {
                // Check array data type.
                switch (arrayTypeId(arrayObj)) {
                    case DtId::FLOAT: case DtId::DOUBLE: //case DtId::HALF:
                    case DtId::INT16: case DtId::INT32: case DtId::INT64:
                    case DtId::UINT32: case DtId::UINT64: break;
                    default: wrongArrayType = true; break;
                }
            }
            if (wrongArrayType) {
                // Generate an error message and raise a Python TypeError.
                std::ostringstream os;
                os << "expected N x 3 numpy.ndarray of " << desiredType << ", found ";
                switch (dims.size()) {
                    case 0: os << "zero-dimensional"; break;
                    case 1: os << "one-dimensional"; break;
                    default:
                        os << dims[0];
                        for (size_t i = 1; i < dims.size(); ++i) { os << " x " << dims[i]; }
                        break;
                }
                os << " " << arrayTypeName(arrayObj) << " array as argument 1 to "
                    << pyutil::GridTraits<GridType>::name() << "." << methodName() << "()";
                PyErr_SetString(PyExc_TypeError, os.str().c_str());
                py::throw_error_already_set();
            }
        }
    };

    // Extract the narrow band half width from the arguments to this method.
    const float halfWidth = extractValueArg<GridType, float>(
        halfWidthObj, Local::methodName(), /*argIdx=*/5, "float");

    // Extract the transform from the arguments to this method.
    math::Transform::Ptr xform = math::Transform::createLinearTransform();
    if (!xformObj.is_none()) {
        xform = extractValueArg<GridType, math::Transform::Ptr>(
            xformObj, Local::methodName(), /*argIdx=*/4, "Transform");
    }

    // Extract the list of mesh vertices from the arguments to this method.
    std::vector<Vec3s> points;
    if (!pointsObj.is_none()) {
        // Extract a reference to (not a copy of) a NumPy array argument,
        // or throw an exception if the argument is not a NumPy array object.
        auto arrayObj = extractValueArg<GridType, NumPyArrayType>(
            pointsObj, Local::methodName(), /*argIdx=*/1, "numpy.ndarray");

        // Throw an exception if the array has the wrong type or dimensions.
        Local::validate2DNumPyArray(arrayObj, /*N=*/3, /*desiredType=*/"float");

        // Copy values from the array to the vector.
        copyVecArray(arrayObj, points);
    }

    // Extract the list of triangle indices from the arguments to this method.
    std::vector<Vec3I> triangles;
    if (!trianglesObj.is_none()) {
        auto arrayObj = extractValueArg<GridType, NumPyArrayType>(
            trianglesObj, Local::methodName(), /*argIdx=*/2, "numpy.ndarray");
        Local::validate2DNumPyArray(arrayObj, /*N=*/3, /*desiredType=*/"int");
        copyVecArray(arrayObj, triangles);
    }

    // Extract the list of quad indices from the arguments to this method.
    std::vector<Vec4I> quads;
    if (!quadsObj.is_none()) {
        auto arrayObj = extractValueArg<GridType, NumPyArrayType>(
            quadsObj, Local::methodName(), /*argIdx=*/3, "numpy.ndarray");
        Local::validate2DNumPyArray(arrayObj, /*N=*/4, /*desiredType=*/"int");
        copyVecArray(arrayObj, quads);
    }

    // Generate and return a level set grid.
    return tools::meshToLevelSet<GridType>(*xform, points, triangles, quads, halfWidth);
}


template<typename GridType>
inline py::object
volumeToQuadMesh(const GridType& grid, py::object isovalueObj)
{
    const double isovalue = pyutil::extractArg<double>(
        isovalueObj, "convertToQuads", /*className=*/nullptr, /*argIdx=*/2, "float");

    // Mesh the input grid and populate lists of mesh vertices and face vertex indices.
    std::vector<Vec3s> points;
    std::vector<Vec4I> quads;
    tools::volumeToMesh(grid, points, quads, isovalue);

#ifdef PY_OPENVDB_USE_BOOST_PYTHON_NUMPY
    const py::object own;
    auto dtype = py::numpy::dtype::get_builtin<Vec3s::value_type>();
    auto shape = py::make_tuple(points.size(), 3);
    auto stride = py::make_tuple(3 * sizeof(Vec3s::value_type), sizeof(Vec3s::value_type));
    // Create a deep copy of the array (because the point vector will be destroyed
    // when this function returns).
    auto pointArrayObj = py::numpy::from_data(points.data(), dtype, shape, stride, own).copy();

    dtype = py::numpy::dtype::get_builtin<Vec4I::value_type>();
    shape = py::make_tuple(quads.size(), 4);
    stride = py::make_tuple(4 * sizeof(Vec4I::value_type), sizeof(Vec4I::value_type));
    auto quadArrayObj = py::numpy::from_data(
        quads.data(), dtype, shape, stride, own).copy(); // deep copy
#else // !defined PY_OPENVDB_USE_BOOST_PYTHON_NUMPY
    // Copy vertices into an N x 3 NumPy array.
    py::object pointArrayObj = py::numeric::array(py::list(), "float32");
    if (!points.empty()) {
        npy_intp dims[2] = { npy_intp(points.size()), 3 };
        // Construct a NumPy array that wraps the point vector.
        if (PyArrayObject* arrayObj = reinterpret_cast<PyArrayObject*>(
            PyArray_SimpleNewFromData(/*nd=*/2, dims, NPY_FLOAT, &points[0])))
        {
            // Create a deep copy of the array (because the point vector will be
            // destroyed when this function returns).
            pointArrayObj = copyNumPyArray(arrayObj, NPY_CORDER);
        }
    }

    // Copy face indices into an N x 4 NumPy array.
    py::object quadArrayObj = py::numeric::array(py::list(), "uint32");
    if (!quads.empty()) {
        npy_intp dims[2] = { npy_intp(quads.size()), 4 };
        if (PyArrayObject* arrayObj = reinterpret_cast<PyArrayObject*>(
            PyArray_SimpleNewFromData(/*dims=*/2, dims, NPY_UINT32, &quads[0])))
        {
            quadArrayObj = copyNumPyArray(arrayObj, NPY_CORDER);
        }
    }
#endif // PY_OPENVDB_USE_BOOST_PYTHON_NUMPY

    return py::make_tuple(pointArrayObj, quadArrayObj);
}


template<typename GridType>
inline py::object
volumeToMesh(const GridType& grid, py::object isovalueObj, py::object adaptivityObj)
{
    const double isovalue = pyutil::extractArg<double>(
        isovalueObj, "convertToPolygons", /*className=*/nullptr, /*argIdx=*/2, "float");
    const double adaptivity = pyutil::extractArg<double>(
        adaptivityObj, "convertToPolygons", /*className=*/nullptr, /*argIdx=*/3, "float");

    // Mesh the input grid and populate lists of mesh vertices and face vertex indices.
    std::vector<Vec3s> points;
    std::vector<Vec3I> triangles;
    std::vector<Vec4I> quads;
    tools::volumeToMesh(grid, points, triangles, quads, isovalue, adaptivity);

#ifdef PY_OPENVDB_USE_BOOST_PYTHON_NUMPY
    const py::object own;
    auto dtype = py::numpy::dtype::get_builtin<Vec3s::value_type>();
    auto shape = py::make_tuple(points.size(), 3);
    auto stride = py::make_tuple(3 * sizeof(Vec3s::value_type), sizeof(Vec3s::value_type));
    // Create a deep copy of the array (because the point vector will be destroyed
    // when this function returns).
    auto pointArrayObj = py::numpy::from_data(points.data(), dtype, shape, stride, own).copy();

    dtype = py::numpy::dtype::get_builtin<Vec3I::value_type>();
    shape = py::make_tuple(triangles.size(), 3);
    stride = py::make_tuple(3 * sizeof(Vec3I::value_type), sizeof(Vec3I::value_type));
    auto triangleArrayObj = py::numpy::from_data(
        triangles.data(), dtype, shape, stride, own).copy(); // deep copy

    dtype = py::numpy::dtype::get_builtin<Vec4I::value_type>();
    shape = py::make_tuple(quads.size(), 4);
    stride = py::make_tuple(4 * sizeof(Vec4I::value_type), sizeof(Vec4I::value_type));
    auto quadArrayObj = py::numpy::from_data(
        quads.data(), dtype, shape, stride, own).copy(); // deep copy
#else // !defined PY_OPENVDB_USE_BOOST_PYTHON_NUMPY
    // Copy vertices into an N x 3 NumPy array.
    py::object pointArrayObj = py::numeric::array(py::list(), "float32");
    if (!points.empty()) {
        npy_intp dims[2] = { npy_intp(points.size()), 3 };
        // Construct a NumPy array that wraps the point vector.
        if (PyArrayObject* arrayObj = reinterpret_cast<PyArrayObject*>(
            PyArray_SimpleNewFromData(/*dims=*/2, dims, NPY_FLOAT, &points[0])))
        {
            // Create a deep copy of the array (because the point vector will be
            // destroyed when this function returns).
            pointArrayObj = copyNumPyArray(arrayObj, NPY_CORDER);
        }
    }

    // Copy triangular face indices into an N x 3 NumPy array.
    py::object triangleArrayObj = py::numeric::array(py::list(), "uint32");
    if (!triangles.empty()) {
        npy_intp dims[2] = { npy_intp(triangles.size()), 3 };
        if (PyArrayObject* arrayObj = reinterpret_cast<PyArrayObject*>(
            PyArray_SimpleNewFromData(/*dims=*/2, dims, NPY_UINT32, &triangles[0])))
        {
            triangleArrayObj = copyNumPyArray(arrayObj, NPY_CORDER);
        }
    }

    // Copy quadrilateral face indices into an N x 4 NumPy array.
    py::object quadArrayObj = py::numeric::array(py::list(), "uint32");
    if (!quads.empty()) {
        npy_intp dims[2] = { npy_intp(quads.size()), 4 };
        if (PyArrayObject* arrayObj = reinterpret_cast<PyArrayObject*>(
            PyArray_SimpleNewFromData(/*dims=*/2, dims, NPY_UINT32, &quads[0])))
        {
            quadArrayObj = copyNumPyArray(arrayObj, NPY_CORDER);
        }
    }
#endif // PY_OPENVDB_USE_BOOST_PYTHON_NUMPY

    return py::make_tuple(pointArrayObj, triangleArrayObj, quadArrayObj);
}

#endif // defined(PY_OPENVDB_USE_NUMPY)


////////////////////////////////////////


template<typename GridType, typename IterType>
inline void
applyMap(const char* methodName, GridType& grid, py::object funcObj)
{
    using ValueT = typename GridType::ValueType;

    for (IterType it = grid.tree().template begin<IterType>(); it; ++it) {
        // Evaluate the functor.
        py::object result = funcObj(*it);

        // Verify that the result is of type GridType::ValueType.
        py::extract<ValueT> val(result);
        if (!val.check()) {
            PyErr_Format(PyExc_TypeError,
                "expected callable argument to %s.%s() to return %s, found %s",
                pyutil::GridTraits<GridType>::name(),
                methodName,
                openvdb::typeNameAsString<ValueT>(),
                pyutil::className(result).c_str());
            py::throw_error_already_set();
        }

        it.setValue(val());
    }
}


template<typename GridType>
inline void
mapOn(GridType& grid, py::object funcObj)
{
    applyMap<GridType, typename GridType::ValueOnIter>("mapOn", grid, funcObj);
}


template<typename GridType>
inline void
mapOff(GridType& grid, py::object funcObj)
{
    applyMap<GridType, typename GridType::ValueOffIter>("mapOff", grid, funcObj);
}


template<typename GridType>
inline void
mapAll(GridType& grid, py::object funcObj)
{
    applyMap<GridType, typename GridType::ValueAllIter>("mapAll", grid, funcObj);
}


////////////////////////////////////////


template<typename GridType>
struct TreeCombineOp
{
    using TreeT = typename GridType::TreeType;
    using ValueT = typename GridType::ValueType;

    TreeCombineOp(py::object _op): op(_op) {}
    void operator()(const ValueT& a, const ValueT& b, ValueT& result)
    {
        py::object resultObj = op(a, b);

        py::extract<ValueT> val(resultObj);
        if (!val.check()) {
            PyErr_Format(PyExc_TypeError,
                "expected callable argument to %s.combine() to return %s, found %s",
                pyutil::GridTraits<GridType>::name(),
                openvdb::typeNameAsString<ValueT>(),
                pyutil::className(resultObj).c_str());
            py::throw_error_already_set();
        }

        result = val();
    }
    py::object op;
};


template<typename GridType>
inline void
combine(GridType& grid, py::object otherGridObj, py::object funcObj)
{
    using GridPtr = typename GridType::Ptr;
    GridPtr otherGrid = extractValueArg<GridType, GridPtr>(otherGridObj,
        "combine", 1, pyutil::GridTraits<GridType>::name());
    TreeCombineOp<GridType> op(funcObj);
    grid.tree().combine(otherGrid->tree(), op, /*prune=*/true);
}


////////////////////////////////////////


template<typename GridType>
inline typename GridType::Ptr
createLevelSetSphere(float radius, const openvdb::Vec3f& center, float voxelSize, float halfWidth)
{
    return tools::createLevelSetSphere<GridType>(radius, center, voxelSize, halfWidth);
}


////////////////////////////////////////


template<typename GridT, typename IterT> class IterWrap; // forward declaration

//
// Type traits for various iterators
//
template<typename GridT, typename IterT> struct IterTraits
{
    // IterT    the type of the iterator
    // name()   function returning the base name of the iterator type (e.g., "ValueOffIter")
    // descr()  function returning a string describing the iterator
    // begin()  function returning a begin iterator for a given grid
};

template<typename GridT> struct IterTraits<GridT, typename GridT::ValueOnCIter>
{
    using IterT = typename GridT::ValueOnCIter;
    static std::string name() { return "ValueOnCIter"; }
    static std::string descr()
    {
        return std::string("Read-only iterator over the active values (tile and voxel)\nof a ")
            + pyutil::GridTraits<typename std::remove_const<GridT>::type>::name();
    }
    static IterWrap<const GridT, IterT> begin(typename GridT::Ptr g)
    {
        return IterWrap<const GridT, IterT>(g, g->cbeginValueOn());
    }
}; // IterTraits<ValueOnCIter>

template<typename GridT> struct IterTraits<GridT, typename GridT::ValueOffCIter>
{
    using IterT = typename GridT::ValueOffCIter;
    static std::string name() { return "ValueOffCIter"; }
    static std::string descr()
    {
        return std::string("Read-only iterator over the inactive values (tile and voxel)\nof a ")
            + pyutil::GridTraits<typename std::remove_const<GridT>::type>::name();
    }
    static IterWrap<const GridT, IterT> begin(typename GridT::Ptr g)
    {
        return IterWrap<const GridT, IterT>(g, g->cbeginValueOff());
    }
}; // IterTraits<ValueOffCIter>

template<typename GridT> struct IterTraits<GridT, typename GridT::ValueAllCIter>
{
    using IterT = typename GridT::ValueAllCIter;
    static std::string name() { return "ValueAllCIter"; }
    static std::string descr()
    {
        return std::string("Read-only iterator over all tile and voxel values of a ")
            + pyutil::GridTraits<typename std::remove_const<GridT>::type>::name();
    }
    static IterWrap<const GridT, IterT> begin(typename GridT::Ptr g)
    {
        return IterWrap<const GridT, IterT>(g, g->cbeginValueAll());
    }
}; // IterTraits<ValueAllCIter>

template<typename GridT> struct IterTraits<GridT, typename GridT::ValueOnIter>
{
    using IterT = typename GridT::ValueOnIter;
    static std::string name() { return "ValueOnIter"; }
    static std::string descr()
    {
        return std::string("Read/write iterator over the active values (tile and voxel)\nof a ")
            + pyutil::GridTraits<typename std::remove_const<GridT>::type>::name();
    }
    static IterWrap<GridT, IterT> begin(typename GridT::Ptr g)
    {
        return IterWrap<GridT, IterT>(g, g->beginValueOn());
    }
}; // IterTraits<ValueOnIter>

template<typename GridT> struct IterTraits<GridT, typename GridT::ValueOffIter>
{
    using IterT = typename GridT::ValueOffIter;
    static std::string name() { return "ValueOffIter"; }
    static std::string descr()
    {
        return std::string("Read/write iterator over the inactive values (tile and voxel)\nof a ")
            + pyutil::GridTraits<typename std::remove_const<GridT>::type>::name();
    }
    static IterWrap<GridT, IterT> begin(typename GridT::Ptr g)
    {
        return IterWrap<GridT, IterT>(g, g->beginValueOff());
    }
}; // IterTraits<ValueOffIter>

template<typename GridT> struct IterTraits<GridT, typename GridT::ValueAllIter>
{
    using IterT = typename GridT::ValueAllIter;
    static std::string name() { return "ValueAllIter"; }
    static std::string descr()
    {
        return std::string("Read/write iterator over all tile and voxel values of a ")
            + pyutil::GridTraits<typename std::remove_const<GridT>::type>::name();
    }
    static IterWrap<GridT, IterT> begin(typename GridT::Ptr g)
    {
        return IterWrap<GridT, IterT>(g, g->beginValueAll());
    }
}; // IterTraits<ValueAllIter>


////////////////////////////////////////


// Helper class to modify a grid through a non-const iterator
template<typename GridT, typename IterT>
struct IterItemSetter
{
    using ValueT = typename GridT::ValueType;
    static void setValue(const IterT& iter, const ValueT& val) { iter.setValue(val); }
    static void setActive(const IterT& iter, bool on) { iter.setActiveState(on); }
};

// Partial specialization for const iterators
template<typename GridT, typename IterT>
struct IterItemSetter<const GridT, IterT>
{
    using ValueT = typename GridT::ValueType;
    static void setValue(const IterT&, const ValueT&)
    {
        PyErr_SetString(PyExc_AttributeError, "can't set attribute 'value'");
        py::throw_error_already_set();
    }
    static void setActive(const IterT&, bool /*on*/)
    {
        PyErr_SetString(PyExc_AttributeError, "can't set attribute 'active'");
        py::throw_error_already_set();
    }
};


/// @brief Value returned by the next() method of a grid's value iterator
/// @details This class allows both dictionary-style (e.g., items["depth"]) and
/// attribute access (e.g., items.depth) to the items returned by an iterator.
/// @todo Create a reusable base class for "named dicts" like this?
template<typename _GridT, typename _IterT>
class IterValueProxy
{
public:
    using GridT = _GridT;
    using IterT = _IterT;
    using ValueT = typename GridT::ValueType;
    using SetterT = IterItemSetter<GridT, IterT>;

    IterValueProxy(typename GridT::ConstPtr grid, const IterT& iter): mGrid(grid), mIter(iter) {}

    IterValueProxy copy() const { return *this; }

    typename GridT::ConstPtr parent() const { return mGrid; }

    ValueT getValue() const { return *mIter; }
    bool getActive() const { return mIter.isValueOn(); }
    Index getDepth() const { return mIter.getDepth(); }
    Coord getBBoxMin() const { return mIter.getBoundingBox().min(); }
    Coord getBBoxMax() const { return mIter.getBoundingBox().max(); }
    Index64 getVoxelCount() const { return mIter.getVoxelCount(); }

    void setValue(const ValueT& val) { SetterT::setValue(mIter, val); }
    void setActive(bool on) { SetterT::setActive(mIter, on); }

    /// Return this dictionary's keys as a list of C strings.
    static const char* const * keys()
    {
        static const char* const sKeys[] = {
            "value", "active", "depth", "min", "max", "count", nullptr
        };
        return sKeys;
    }

    /// Return @c true if the given string is a valid key.
    static bool hasKey(const std::string& key)
    {
        for (int i = 0; keys()[i] != nullptr; ++i) {
            if (key == keys()[i]) return true;
        }
        return false;
    }

    /// Return this dictionary's keys as a Python list of Python strings.
    static py::list getKeys()
    {
        py::list keyList;
        for (int i = 0; keys()[i] != nullptr; ++i) keyList.append(keys()[i]);
        return keyList;
    }

    /// @brief Return the value for the given key.
    /// @throw KeyError if the key is invalid
    py::object getItem(py::object keyObj) const
    {
        py::extract<std::string> x(keyObj);
        if (x.check()) {
            const std::string key = x();
            if (key == "value") return py::object(this->getValue());
            else if (key == "active") return py::object(this->getActive());
            else if (key == "depth") return py::object(this->getDepth());
            else if (key == "min") return py::object(this->getBBoxMin());
            else if (key == "max") return py::object(this->getBBoxMax());
            else if (key == "count") return py::object(this->getVoxelCount());
        }
        PyErr_SetObject(PyExc_KeyError, ("%s" % keyObj.attr("__repr__")()).ptr());
        py::throw_error_already_set();
        return py::object();
    }

    /// @brief Set the value for the given key.
    /// @throw KeyError if the key is invalid
    /// @throw AttributeError if the key refers to a read-only item
    void setItem(py::object keyObj, py::object valObj)
    {
        py::extract<std::string> x(keyObj);
        if (x.check()) {
            const std::string key = x();
            if (key == "value") {
                this->setValue(py::extract<ValueT>(valObj)); return;
            } else if (key == "active") {
                this->setActive(py::extract<bool>(valObj)); return;
            } else if (this->hasKey(key)) {
                PyErr_SetObject(PyExc_AttributeError,
                    ("can't set attribute '%s'" % keyObj.attr("__repr__")()).ptr());
                py::throw_error_already_set();
            }
        }
        PyErr_SetObject(PyExc_KeyError,
            ("'%s'" % keyObj.attr("__repr__")()).ptr());
        py::throw_error_already_set();
    }

    bool operator==(const IterValueProxy& other) const
    {
        return (other.getActive() == this->getActive()
            && other.getDepth() == this->getDepth()
            && math::isExactlyEqual(other.getValue(), this->getValue())
            && other.getBBoxMin() == this->getBBoxMin()
            && other.getBBoxMax() == this->getBBoxMax()
            && other.getVoxelCount() == this->getVoxelCount());
    }
    bool operator!=(const IterValueProxy& other) const { return !(*this == other); }

    /// Print this dictionary to a stream.
    std::ostream& put(std::ostream& os) const
    {
        // valuesAsStrings = ["%s: %s" % key, repr(this[key]) for key in this.keys()]
        py::list valuesAsStrings;
        for (int i = 0; this->keys()[i] != nullptr; ++i) {
            py::str
                key(this->keys()[i]),
                val(this->getItem(key).attr("__repr__")());
            valuesAsStrings.append("'%s': %s" % py::make_tuple(key, val));
        }
        // print ", ".join(valuesAsStrings)
        py::object joined = py::str(", ").attr("join")(valuesAsStrings);
        std::string s = py::extract<std::string>(joined);
        os << "{" << s << "}";
        return os;
    }
    /// Return a string describing this dictionary.
    std::string info() const { std::ostringstream os; os << *this; return os.str(); }

private:
    // To keep the iterator's grid from being deleted (leaving the iterator dangling),
    // store a shared pointer to the grid.
    const typename GridT::ConstPtr mGrid;
    const IterT mIter; // the iterator may not be incremented
}; // class IterValueProxy


template<typename GridT, typename IterT>
inline std::ostream&
operator<<(std::ostream& os, const IterValueProxy<GridT, IterT>& iv) { return iv.put(os); }


////////////////////////////////////////


/// Wrapper for a grid's value iterator classes
template<typename _GridT, typename _IterT>
class IterWrap
{
public:
    using GridT = _GridT;
    using IterT = _IterT;
    using ValueT = typename GridT::ValueType;
    using IterValueProxyT = IterValueProxy<GridT, IterT>;
    using Traits = IterTraits<GridT, IterT>;

    IterWrap(typename GridT::ConstPtr grid, const IterT& iter): mGrid(grid), mIter(iter) {}

    typename GridT::ConstPtr parent() const { return mGrid; }

    /// Return an IterValueProxy for the current iterator position.
    IterValueProxyT next()
    {
        if (!mIter) {
            PyErr_SetString(PyExc_StopIteration, "no more values");
            py::throw_error_already_set();
        }
        IterValueProxyT result(mGrid, mIter);
        ++mIter;
        return result;
    }

    static py::object returnSelf(const py::object& obj) { return obj; }

    /// @brief Define a Python wrapper class for this C++ class and another for
    /// the IterValueProxy class returned by iterators of this type.
    static void wrap()
    {
        const std::string
            gridClassName = pyutil::GridTraits<typename std::remove_const<GridT>::type>::name(),
            iterClassName = /*gridClassName +*/ Traits::name(),
            valueClassName = /*gridClassName +*/ "Value";

        py::class_<IterWrap>(
            iterClassName.c_str(),
            /*docstring=*/Traits::descr().c_str(),
            /*ctor=*/py::no_init) // can only be instantiated from C++, not from Python

            .add_property("parent", &IterWrap::parent,
                ("the " + gridClassName + " over which to iterate").c_str())

            .def("next", &IterWrap::next, ("next() -> " + valueClassName).c_str())
            .def("__next__", &IterWrap::next, ("__next__() -> " + valueClassName).c_str())
            .def("__iter__", &returnSelf);

        py::class_<IterValueProxyT>(
            valueClassName.c_str(),
            /*docstring=*/("Proxy for a tile or voxel value in a " + gridClassName).c_str(),
            /*ctor=*/py::no_init) // can only be instantiated from C++, not from Python

            .def("copy", &IterValueProxyT::copy,
                ("copy() -> " + valueClassName + "\n\n"
                "Return a shallow copy of this value, i.e., one that shares\n"
                "its data with the original.").c_str())

            .add_property("parent", &IterValueProxyT::parent,
                ("the " + gridClassName + " to which this value belongs").c_str())

            .def("__str__", &IterValueProxyT::info)
            .def("__repr__", &IterValueProxyT::info)

            .def("__eq__", &IterValueProxyT::operator==)
            .def("__ne__", &IterValueProxyT::operator!=)

            .add_property("value", &IterValueProxyT::getValue, &IterValueProxyT::setValue,
                "value of this tile or voxel")
            .add_property("active", &IterValueProxyT::getActive, &IterValueProxyT::setActive,
                "active state of this tile or voxel")
            .add_property("depth", &IterValueProxyT::getDepth,
                "tree depth at which this value is stored")
            .add_property("min", &IterValueProxyT::getBBoxMin,
                "lower bound of the axis-aligned bounding box of this tile or voxel")
            .add_property("max", &IterValueProxyT::getBBoxMax,
                "upper bound of the axis-aligned bounding box of this tile or voxel")
            .add_property("count", &IterValueProxyT::getVoxelCount,
                "number of voxels spanned by this value")

            .def("keys", &IterValueProxyT::getKeys,
                "keys() -> list\n\n"
                "Return a list of keys for this tile or voxel.")
            .staticmethod("keys")
            .def("__contains__", &IterValueProxyT::hasKey,
                "__contains__(key) -> bool\n\n"
                "Return True if the given key exists.")
            .staticmethod("__contains__")
            .def("__getitem__", &IterValueProxyT::getItem,
                "__getitem__(key) -> value\n\n"
                "Return the value of the item with the given key.")
            .def("__setitem__", &IterValueProxyT::getItem,
                "__setitem__(key, value)\n\n"
                "Set the value of the item with the given key.");
    }

private:
    // To keep this iterator's grid from being deleted, leaving the iterator dangling,
    // store a shared pointer to the grid.
    const typename GridT::ConstPtr mGrid;
    IterT mIter;
}; // class IterWrap


////////////////////////////////////////


template<typename GridT>
struct PickleSuite: public py::pickle_suite
{
    using GridPtrT = typename GridT::Ptr;

    /// Return @c true, indicating that this pickler preserves a Grid's __dict__.
    static bool getstate_manages_dict() { return true; }

    /// Return a tuple representing the state of the given Grid.
    static py::tuple getstate(py::object gridObj)
    {
        py::tuple state;

        // Extract a Grid from the Python object.
        GridPtrT grid;
        py::extract<GridPtrT> x(gridObj);
        if (x.check()) grid = x();

        if (grid) {
            // Serialize the Grid to a string.
            std::ostringstream ostr(std::ios_base::binary);
            {
                openvdb::io::Stream strm(ostr);
                strm.setGridStatsMetadataEnabled(false);
                strm.write(openvdb::GridPtrVec(1, grid));
            }
            // Construct a state tuple comprising the Python object's __dict__
            // and the serialized Grid.
#if PY_MAJOR_VERSION >= 3
            // Convert the byte string to a "bytes" sequence.
            const std::string s = ostr.str();
            py::object bytesObj = pyutil::pyBorrow(PyBytes_FromStringAndSize(s.data(), s.size()));
#else
            py::str bytesObj(ostr.str());
#endif
            state = py::make_tuple(gridObj.attr("__dict__"), bytesObj);
        }
        return state;
    }

    /// Restore the given Grid to a saved state.
    static void setstate(py::object gridObj, py::object stateObj)
    {
        GridPtrT grid;
        {
            py::extract<GridPtrT> x(gridObj);
            if (x.check()) grid = x();
        }
        if (!grid) return;

        py::tuple state;
        {
            py::extract<py::tuple> x(stateObj);
            if (x.check()) state = x();
        }
        bool badState = (py::len(state) != 2);

        if (!badState) {
            // Restore the object's __dict__.
            py::extract<py::dict> x(state[0]);
            if (x.check()) {
                py::dict d = py::extract<py::dict>(gridObj.attr("__dict__"))();
                d.update(x());
            } else {
                badState = true;
            }
        }

        std::string serialized;
        if (!badState) {
            // Extract the sequence containing the serialized Grid.
            py::object bytesObj = state[1];
#if PY_MAJOR_VERSION >= 3
            badState = true;
            if (PyBytes_Check(bytesObj.ptr())) {
                // Convert the "bytes" sequence to a byte string.
                char* buf = nullptr;
                Py_ssize_t length = 0;
                if (-1 != PyBytes_AsStringAndSize(bytesObj.ptr(), &buf, &length)) {
                    if (buf != nullptr && length > 0) {
                        serialized.assign(buf, buf + length);
                        badState = false;
                    }
                }
            }
#else
            py::extract<std::string> x(bytesObj);
            if (x.check()) serialized = x();
            else badState = true;
#endif
        }

        if (badState) {
            PyErr_SetObject(PyExc_ValueError,
#if PY_MAJOR_VERSION >= 3
                ("expected (dict, bytes) tuple in call to __setstate__; found %s"
#else
                ("expected (dict, str) tuple in call to __setstate__; found %s"
#endif
                     % stateObj.attr("__repr__")()).ptr());
            py::throw_error_already_set();
        }

        // Restore the internal state of the C++ object.
        GridPtrVecPtr grids;
        {
            std::istringstream istr(serialized, std::ios_base::binary);
            io::Stream strm(istr);
            grids = strm.getGrids(); // (note: file-level metadata is ignored)
        }
        if (grids && !grids->empty()) {
            if (GridPtrT savedGrid = gridPtrCast<GridT>((*grids)[0])) {
                grid->MetaMap::operator=(*savedGrid); ///< @todo add a Grid::setMetadata() method?
                grid->setTransform(savedGrid->transformPtr());
                grid->setTree(savedGrid->treePtr());
            }
        }
    }
}; // struct PickleSuite


////////////////////////////////////////


/// Create a Python wrapper for a particular template instantiation of Grid.
template<typename GridType>
inline void
exportGrid()
{
    using ValueT = typename GridType::ValueType;
    using GridPtr = typename GridType::Ptr;
    using Traits = pyutil::GridTraits<GridType>;

    using ValueOnCIterT = typename GridType::ValueOnCIter;
    using ValueOffCIterT = typename GridType::ValueOffCIter;
    using ValueAllCIterT = typename GridType::ValueAllCIter;
    using ValueOnIterT = typename GridType::ValueOnIter;
    using ValueOffIterT = typename GridType::ValueOffIter;
    using ValueAllIterT = typename GridType::ValueAllIter;

    math::Transform::Ptr (GridType::*getTransform)() = &GridType::transformPtr;

    const std::string pyGridTypeName = Traits::name();
    const std::string defaultCtorDescr = "Initialize with a background value of "
        + pyutil::str(pyGrid::getZeroValue<GridType>()) + ".";

    // Define the Grid wrapper class and make it the current scope.
    {
        py::class_<GridType, /*HeldType=*/GridPtr> clss(
            /*classname=*/pyGridTypeName.c_str(),
            /*docstring=*/(Traits::descr()).c_str(),
            /*ctor=*/py::init<>(defaultCtorDescr.c_str())
        );

        py::scope gridClassScope = clss;

        clss.def(py::init<const ValueT&>(py::args("background"),
                "Initialize with the given background value."))

            .def("copy", &pyGrid::copyGrid<GridType>,
                ("copy() -> " + pyGridTypeName + "\n\n"
                "Return a shallow copy of this grid, i.e., a grid\n"
                "that shares its voxel data with this grid.").c_str())
            .def("deepCopy", &GridType::deepCopy,
                ("deepCopy() -> " + pyGridTypeName + "\n\n"
                "Return a deep copy of this grid.\n").c_str())

            .def_pickle(pyGrid::PickleSuite<GridType>())

            .def("sharesWith", &pyGrid::sharesWith<GridType>,
                ("sharesWith(" + pyGridTypeName + ") -> bool\n\n"
                "Return True if this grid shares its voxel data with the given grid.").c_str())

            /// @todo Any way to set a docstring for a class property?
            .add_static_property("valueTypeName", &pyGrid::getValueType<GridType>)
                /// @todo docstring = "name of this grid's value type"
            .add_static_property("zeroValue", &pyGrid::getZeroValue<GridType>)
                /// @todo docstring = "zero, as expressed in this grid's value type"
            .add_static_property("oneValue", &pyGrid::getOneValue<GridType>)
                /// @todo docstring = "one, as expressed in this grid's value type"
            /// @todo Is Grid.typeName ever needed?
            //.add_static_property("typeName", &GridType::gridType)
                /// @todo docstring = to "name of this grid's type"

            .add_property("background",
                &pyGrid::getGridBackground<GridType>, &pyGrid::setGridBackground<GridType>,
                "value of this grid's background voxels")
            .add_property("name", &GridType::getName, &pyGrid::setGridName,
                "this grid's name")
            .add_property("creator", &GridType::getCreator, &pyGrid::setGridCreator,
                "description of this grid's creator")

            .add_property("transform", getTransform, &pyGrid::setGridTransform,
                "transform associated with this grid")

            .add_property("gridClass", &pyGrid::getGridClass, &pyGrid::setGridClass,
                "the class of volumetric data (level set, fog volume, etc.)\nstored in this grid")

            .add_property("vectorType", &pyGrid::getVecType, &pyGrid::setVecType,
                "how transforms are applied to values stored in this grid")

            .def("getAccessor", &pyGrid::getAccessor<GridType>,
                ("getAccessor() -> " + pyGridTypeName + "Accessor\n\n"
                "Return an accessor that provides random read and write access\n"
                "to this grid's voxels.").c_str())
            .def("getConstAccessor", &pyGrid::getConstAccessor<GridType>,
                ("getConstAccessor() -> " + pyGridTypeName + "Accessor\n\n"
                "Return an accessor that provides random read-only access\n"
                "to this grid's voxels.").c_str())

            //
            // Metadata
            //
            .add_property("metadata", &pyGrid::getAllMetadata, &pyGrid::replaceAllMetadata,
                "dict of this grid's metadata\n\n"
                "Setting this attribute replaces all of this grid's metadata,\n"
                "but mutating it in place has no effect on the grid, since\n"
                "the value of this attribute is a only a copy of the metadata.\n"
                "Use either indexing or updateMetadata() to mutate metadata in place.")
            .def("updateMetadata", &pyGrid::updateMetadata,
                "updateMetadata(dict)\n\n"
                "Add metadata to this grid, replacing any existing items\n"
                "having the same names as the new items.")

            .def("addStatsMetadata", &GridType::addStatsMetadata,
                "addStatsMetadata()\n\n"
                "Add metadata to this grid comprising the current values\n"
                "of statistics like the active voxel count and bounding box.\n"
                "(This metadata is not automatically kept up-to-date with\n"
                "changes to this grid.)")
            .def("getStatsMetadata", &pyGrid::getStatsMetadata,
                "getStatsMetadata() -> dict\n\n"
                "Return a (possibly empty) dict containing just the metadata\n"
                "that was added to this grid with addStatsMetadata().")

            .def("__getitem__", &pyGrid::getMetadata,
                "__getitem__(name) -> value\n\n"
                "Return the metadata value associated with the given name.")
            .def("__setitem__", &pyGrid::setMetadata,
                "__setitem__(name, value)\n\n"
                "Add metadata to this grid, replacing any existing item having\n"
                "the same name as the new item.")
            .def("__delitem__", &pyGrid::removeMetadata,
                "__delitem__(name)\n\n"
                "Remove the metadata with the given name.")
            .def("__contains__", &pyGrid::hasMetadata,
                "__contains__(name) -> bool\n\n"
                "Return True if this grid contains metadata with the given name.")
            .def("__iter__", &pyGrid::getMetadataKeys,
                "__iter__() -> iterator\n\n"
                "Return an iterator over this grid's metadata keys.")
            .def("iterkeys", &pyGrid::getMetadataKeys,
                "iterkeys() -> iterator\n\n"
                "Return an iterator over this grid's metadata keys.")

            .add_property("saveFloatAsHalf",
                &GridType::saveFloatAsHalf, &GridType::setSaveFloatAsHalf,
                "if True, write floating-point voxel values as 16-bit half floats")

            //
            // Statistics
            //
            .def("memUsage", &GridType::memUsage,
                "memUsage() -> int\n\n"
                "Return the memory usage of this grid in bytes.")

            .def("evalLeafBoundingBox", &pyGrid::evalLeafBoundingBox<GridType>,
                "evalLeafBoundingBox() -> xyzMin, xyzMax\n\n"
                "Return the coordinates of opposite corners of the axis-aligned\n"
                "bounding box of all leaf nodes.")
            .def("evalLeafDim", &pyGrid::evalLeafDim<GridType>,
                "evalLeafDim() -> x, y, z\n\n"
                "Return the dimensions of the axis-aligned bounding box\n"
                "of all leaf nodes.")

            .def("evalActiveVoxelBoundingBox", &pyGrid::evalActiveVoxelBoundingBox<GridType>,
                "evalActiveVoxelBoundingBox() -> xyzMin, xyzMax\n\n"
                "Return the coordinates of opposite corners of the axis-aligned\n"
                "bounding box of all active voxels.")
            .def("evalActiveVoxelDim", &GridType::evalActiveVoxelDim,
                "evalActiveVoxelDim() -> x, y, z\n\n"
                "Return the dimensions of the axis-aligned bounding box of all\n"
                "active voxels.")

            .add_property("treeDepth", &pyGrid::treeDepth<GridType>,
                "depth of this grid's tree from root node to leaf node")
            .def("nodeLog2Dims", &pyGrid::getNodeLog2Dims<GridType>,
                "list of Log2Dims of the nodes of this grid's tree\n"
                "in order from root to leaf")

            .def("leafCount", &pyGrid::leafCount<GridType>,
                "leafCount() -> int\n\n"
                "Return the number of leaf nodes in this grid's tree.")
            .def("nonLeafCount", &pyGrid::nonLeafCount<GridType>,
                "nonLeafCount() -> int\n\n"
                "Return the number of non-leaf nodes in this grid's tree.")

            .def("activeVoxelCount", &GridType::activeVoxelCount,
                "activeVoxelCount() -> int\n\n"
                "Return the number of active voxels in this grid.")
            .def("activeLeafVoxelCount", &pyGrid::activeLeafVoxelCount<GridType>,
                "activeLeafVoxelCount() -> int\n\n"
                "Return the number of active voxels that are stored\n"
                "in the leaf nodes of this grid's tree.")

            .def("evalMinMax", &pyGrid::evalMinMax<GridType>,
                "evalMinMax() -> min, max\n\n"
                "Return the minimum and maximum active values in this grid.")

            .def("getIndexRange", &pyGrid::getIndexRange<GridType>,
                "getIndexRange() -> min, max\n\n"
                "Return the minimum and maximum coordinates that are represented\n"
                "in this grid.  These might include background voxels.")
            //.def("expand", &pyGrid::expandIndexRange<GridType>,
            //    py::arg("xyz"),
            //    "expand(xyz)\n\n"
            //    "Expand this grid's index range to include the given coordinates.")

            .def("info", &pyGrid::gridInfo,
                py::arg("verbosity")=1,
                "info(verbosity=1) -> str\n\n"
                "Return a string containing information about this grid\n"
                "with a specified level of verbosity.\n")

            //
            // Tools
            //
            .def("fill", &pyGrid::fill<GridType>,
                (py::arg("min"), py::arg("max"), py::arg("value"), py::arg("active")=true),
                "fill(min, max, value, active=True)\n\n"
                "Set all voxels within a given axis-aligned box to\n"
                "a constant value (either active or inactive).")
            .def("signedFloodFill", &pyGrid::signedFloodFill<GridType>,
                "signedFloodFill()\n\n"
                "Propagate the sign from a narrow-band level set into inactive\n"
                "voxels and tiles.")

            .def("copyFromArray", &pyGrid::copyFromArray<GridType>,
                (py::arg("array"), py::arg("ijk")=Coord(0),
                     py::arg("tolerance")=pyGrid::getZeroValue<GridType>()),
                ("copyFromArray(array, ijk=(0, 0, 0), tolerance=0)\n\n"
                "Populate this grid, starting at voxel (i, j, k), with values\nfrom a "
                + std::string(openvdb::VecTraits<ValueT>::IsVec ? "four" : "three")
                + "-dimensional array.  Mark voxels as inactive\n"
                "if and only if their values are equal to this grid's\n"
                "background value within the given tolerance.").c_str())
            .def("copyToArray", &pyGrid::copyToArray<GridType>,
                (py::arg("array"), py::arg("ijk")=Coord(0)),
                ("copyToArray(array, ijk=(0, 0, 0))\n\nPopulate a "
                + std::string(openvdb::VecTraits<ValueT>::IsVec ? "four" : "three")
                + "-dimensional array with values\n"
                "from this grid, starting at voxel (i, j, k).").c_str())

            .def("convertToQuads",
                &pyGrid::volumeToQuadMesh<GridType>,
                (py::arg("isovalue")=0),
                "convertToQuads(isovalue=0) -> points, quads\n\n"
                "Uniformly mesh a scalar grid that has a continuous isosurface\n"
                "at the given isovalue.  Return a NumPy array of world-space\n"
                "points and a NumPy array of 4-tuples of point indices, which\n"
                "specify the vertices of the quadrilaterals that form the mesh.")
            .def("convertToPolygons",
                &pyGrid::volumeToMesh<GridType>,
                (py::arg("isovalue")=0, py::arg("adaptivity")=0),
                "convertToPolygons(isovalue=0, adaptivity=0) -> points, triangles, quads\n\n"
                "Adaptively mesh a scalar grid that has a continuous isosurface\n"
                "at the given isovalue.  Return a NumPy array of world-space\n"
                "points and NumPy arrays of 3- and 4-tuples of point indices,\n"
                "which specify the vertices of the triangles and quadrilaterals\n"
                "that form the mesh.  Adaptivity can vary from 0 to 1, where 0\n"
                "produces a high-polygon-count mesh that closely approximates\n"
                "the isosurface, and 1 produces a lower-polygon-count mesh\n"
                "with some loss of surface detail.")
            .def("createLevelSetFromPolygons",
                &pyGrid::meshToLevelSet<GridType>,
                (py::arg("points"),
                     py::arg("triangles")=py::object(),
                     py::arg("quads")=py::object(),
                     py::arg("transform")=py::object(),
                     py::arg("halfWidth")=openvdb::LEVEL_SET_HALF_WIDTH),
                ("createLevelSetFromPolygons(points, triangles=None, quads=None,\n"
                 "    transform=None, halfWidth="
                 + std::to_string(openvdb::LEVEL_SET_HALF_WIDTH) + ") -> "
                 + pyGridTypeName + "\n\n"
                "Convert a triangle and/or quad mesh to a narrow-band level set volume.\n"
                "The mesh must form a closed surface, but the surface need not be\n"
                "manifold and may have self intersections and degenerate faces.\n"
                "The mesh is described by a NumPy array of world-space points\n"
                "and NumPy arrays of 3- and 4-tuples of point indices that specify\n"
                "the vertices of the triangles and quadrilaterals that form the mesh.\n"
                "Either the triangle or the quad array may be empty or None.\n"
                "The resulting volume will have the given transform (or the identity\n"
                "transform if no transform is given) and a narrow band width of\n"
                "2 x halfWidth voxels.").c_str())
            .staticmethod("createLevelSetFromPolygons")

            .def("prune", &pyGrid::prune<GridType>,
                (py::arg("tolerance")=0),
                "prune(tolerance=0)\n\n"
                "Remove nodes whose values all have the same active state\n"
                "and are equal to within a given tolerance.")
            .def("pruneInactive", &pyGrid::pruneInactive<GridType>,
                (py::arg("value")=py::object()),
                "pruneInactive(value=None)\n\n"
                "Remove nodes whose values are all inactive and replace them\n"
                "with either background tiles or tiles of the given value\n"
                "(if the value is not None).")

            .def("empty", &GridType::empty,
                "empty() -> bool\n\n"
                "Return True if this grid contains only background voxels.")
            .def("__nonzero__", &pyGrid::notEmpty<GridType>)

            .def("clear", &GridType::clear,
                "clear()\n\n"
                "Remove all tiles from this grid and all nodes other than the root node.")

            .def("merge", &GridType::merge,
                ("merge(" + pyGridTypeName + ")\n\n"
                "Move child nodes from the other grid into this grid wherever\n"
                "those nodes correspond to constant-value tiles in this grid,\n"
                "and replace leaf-level inactive voxels in this grid with\n"
                "corresponding voxels in the other grid that are active.\n\n"
                "Note: this operation always empties the other grid.").c_str())

            .def("mapOn", &pyGrid::mapOn<GridType>,
                py::arg("function"),
                "mapOn(function)\n\n"
                "Iterate over all the active (\"on\") values (tile and voxel)\n"
                "of this grid and replace each value with function(value).\n\n"
                "Example: grid.mapOn(lambda x: x * 2 if x < 0.5 else x)")

            .def("mapOff", &pyGrid::mapOff<GridType>,
                py::arg("function"),
                "mapOff(function)\n\n"
                "Iterate over all the inactive (\"off\") values (tile and voxel)\n"
                "of this grid and replace each value with function(value).\n\n"
                "Example: grid.mapOff(lambda x: x * 2 if x < 0.5 else x)")

            .def("mapAll", &pyGrid::mapAll<GridType>,
                py::arg("function"),
                "mapAll(function)\n\n"
                "Iterate over all values (tile and voxel) of this grid\n"
                "and replace each value with function(value).\n\n"
                "Example: grid.mapAll(lambda x: x * 2 if x < 0.5 else x)")

            .def("combine", &pyGrid::combine<GridType>,
                (py::arg("grid"), py::arg("function")),
                "combine(grid, function)\n\n"
                "Compute function(self, other) over all corresponding pairs\n"
                "of values (tile or voxel) of this grid and the other grid\n"
                "and store the result in this grid.\n\n"
                "Note: this operation always empties the other grid.\n\n"
                "Example: grid.combine(otherGrid, lambda a, b: min(a, b))")

            //
            // Iterators
            //
            .def("citerOnValues", &pyGrid::IterTraits<GridType, ValueOnCIterT>::begin,
                "citerOnValues() -> iterator\n\n"
                "Return a read-only iterator over this grid's active\ntile and voxel values.")
            .def("citerOffValues", &pyGrid::IterTraits<GridType, ValueOffCIterT>::begin,
                "iterOffValues() -> iterator\n\n"
                "Return a read-only iterator over this grid's inactive\ntile and voxel values.")
            .def("citerAllValues", &pyGrid::IterTraits<GridType, ValueAllCIterT>::begin,
                "iterAllValues() -> iterator\n\n"
                "Return a read-only iterator over all of this grid's\ntile and voxel values.")

            .def("iterOnValues", &pyGrid::IterTraits<GridType, ValueOnIterT>::begin,
                "iterOnValues() -> iterator\n\n"
                "Return a read/write iterator over this grid's active\ntile and voxel values.")
            .def("iterOffValues", &pyGrid::IterTraits<GridType, ValueOffIterT>::begin,
                "iterOffValues() -> iterator\n\n"
                "Return a read/write iterator over this grid's inactive\ntile and voxel values.")
            .def("iterAllValues", &pyGrid::IterTraits<GridType, ValueAllIterT>::begin,
                "iterAllValues() -> iterator\n\n"
                "Return a read/write iterator over all of this grid's\ntile and voxel values.")

            ; // py::class_<Grid>

#if DWA_BOOST_VERSION >= 1060000 && DWA_BOOST_VERSION < 1065000
        // Boost versions 1.60 through 1.6x, for some x < 5, require the GridPtr-to-Python
        // object converter to be explicitly registered.
        py::register_ptr_to_python<GridPtr>();
#endif

        py::implicitly_convertible<GridPtr, GridBase::Ptr>();
        py::implicitly_convertible<GridPtr, GridBase::ConstPtr>();
        /// @todo Is there a way to implicitly convert GridType references to GridBase
        /// references without wrapping the GridBase class?  The following doesn't compile,
        /// because GridBase has pure virtual functions:
        /// @code
        /// py::implicitly_convertible<GridType&, GridBase&>();
        /// @endcode

        // Wrap const and non-const value accessors and expose them
        // as nested classes of the Grid class.
        pyAccessor::AccessorWrap<const GridType>::wrap();
        pyAccessor::AccessorWrap<GridType>::wrap();

        // Wrap tree value iterators and expose them as nested classes of the Grid class.
        IterWrap<const GridType, ValueOnCIterT>::wrap();
        IterWrap<const GridType, ValueOffCIterT>::wrap();
        IterWrap<const GridType, ValueAllCIterT>::wrap();
        IterWrap<GridType, ValueOnIterT>::wrap();
        IterWrap<GridType, ValueOffIterT>::wrap();
        IterWrap<GridType, ValueAllIterT>::wrap();

    } // gridClassScope

    // Add the Python type object for this grid type to the module-level list.
    py::extract<py::list>(py::scope().attr("GridTypes"))().append(
        py::scope().attr(pyGridTypeName.c_str()));
}

} // namespace pyGrid

#endif // OPENVDB_PYGRID_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
