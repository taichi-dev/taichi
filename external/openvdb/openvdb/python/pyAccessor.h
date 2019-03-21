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

#ifndef OPENVDB_PYACCESSOR_HAS_BEEN_INCLUDED
#define OPENVDB_PYACCESSOR_HAS_BEEN_INCLUDED

#include <boost/python.hpp>
#include "openvdb/openvdb.h"
#include "pyutil.h"

namespace pyAccessor {

namespace py = boost::python;
using namespace openvdb::OPENVDB_VERSION_NAME;


//@{
/// Type traits for grid accessors
template<typename _GridT>
struct AccessorTraits
{
    typedef _GridT                              GridT;
    typedef GridT                               NonConstGridT;
    typedef typename NonConstGridT::Ptr         GridPtrT;
    typedef typename NonConstGridT::Accessor    AccessorT;
    typedef typename AccessorT::ValueType       ValueT;

    static const bool IsConst = false;

    static const char* typeName() { return "Accessor"; }

    static void setActiveState(AccessorT& acc, const Coord& ijk, bool on) {
        acc.setActiveState(ijk, on);
    }
    static void setValueOnly(AccessorT& acc, const Coord& ijk, const ValueT& val) {
        acc.setValueOnly(ijk, val);
    }
    static void setValueOn(AccessorT& acc, const Coord& ijk) { acc.setValueOn(ijk); }
    static void setValueOn(AccessorT& acc, const Coord& ijk, const ValueT& val) {
        acc.setValueOn(ijk, val);
    }
    static void setValueOff(AccessorT& acc, const Coord& ijk) { acc.setValueOff(ijk); }
    static void setValueOff(AccessorT& acc, const Coord& ijk, const ValueT& val) {
        acc.setValueOff(ijk, val);
    }
};

// Partial specialization for const accessors
template<typename _GridT>
struct AccessorTraits<const _GridT>
{
    typedef const _GridT                            GridT;
    typedef _GridT                                  NonConstGridT;
    typedef typename NonConstGridT::ConstPtr        GridPtrT;
    typedef typename NonConstGridT::ConstAccessor   AccessorT;
    typedef typename AccessorT::ValueType           ValueT;

    static const bool IsConst = true;

    static const char* typeName() { return "ConstAccessor"; }

    static void setActiveState(AccessorT&, const Coord&, bool) { notWritable(); }
    static void setValueOnly(AccessorT&, const Coord&, const ValueT&) { notWritable(); }
    static void setValueOn(AccessorT&, const Coord&) { notWritable(); }
    static void setValueOn(AccessorT&, const Coord&, const ValueT&) { notWritable(); }
    static void setValueOff(AccessorT&, const Coord&) { notWritable(); }
    static void setValueOff(AccessorT&, const Coord&, const ValueT&) { notWritable(); }

    static void notWritable()
    {
        PyErr_SetString(PyExc_TypeError, "accessor is read-only");
        py::throw_error_already_set();
    }
};
//@}


////////////////////////////////////////


/// Variant of pyutil::extractArg() that extracts a Coord from a py::object
/// argument to a given ValueAccessor method
template<typename GridT>
inline Coord
extractCoordArg(py::object obj, const char* functionName, int argIdx = 0)
{
    return pyutil::extractArg<Coord>(obj, functionName,
        AccessorTraits<GridT>::typeName(), argIdx, "tuple(int, int, int)");
}


/// Variant of pyutil::extractArg() that extracts a value of type
/// ValueAccessor::ValueType from an argument to a ValueAccessor method
template<typename GridT>
inline typename GridT::ValueType
extractValueArg(
    py::object obj,
    const char* functionName,
    int argIdx = 0, // args are numbered starting from 1
    const char* expectedType = NULL)
{
    return pyutil::extractArg<typename GridT::ValueType>(
        obj, functionName, AccessorTraits<GridT>::typeName(), argIdx, expectedType);
}


////////////////////////////////////////


/// @brief ValueAccessor wrapper class that also stores a grid pointer,
/// so that the grid doesn't get deleted as long as the accessor is live
///
/// @internal This class could have just been made to inherit from ValueAccessor,
/// but the method wrappers allow for more Pythonic error messages.  For example,
/// if we constructed the Python getValue() method directly from the corresponding
/// ValueAccessor method, as follows,
///
///    .def("getValue", &Accessor::getValue, ...)
///
/// then the conversion from a Python type to a Coord& would be done
/// automatically.  But if the Python method were called with an object of
/// a type that is not convertible to a Coord, then the TypeError message
/// would say something like "TypeError: No registered converter was able to
/// produce a C++ rvalue of type openvdb::math::Coord...".
/// Handling the type conversion manually is more work, but it allows us to
/// instead generate messages like "TypeError: expected tuple(int, int, int),
/// found str as argument to FloatGridAccessor.getValue()".
template<typename _GridType>
class AccessorWrap
{
public:
    typedef AccessorTraits<_GridType>       Traits;
    typedef typename Traits::AccessorT      Accessor;
    typedef typename Traits::ValueT         ValueType;
    typedef typename Traits::NonConstGridT  GridType;
    typedef typename Traits::GridPtrT       GridPtrType;

    AccessorWrap(GridPtrType grid): mGrid(grid), mAccessor(grid->getAccessor()) {}

    AccessorWrap copy() const { return *this; }

    void clear() { mAccessor.clear(); }

    GridPtrType parent() const { return mGrid; }

    ValueType getValue(py::object coordObj)
    {
        const Coord ijk = extractCoordArg<GridType>(coordObj, "getValue");
        return mAccessor.getValue(ijk);
    }

    int getValueDepth(py::object coordObj)
    {
        const Coord ijk = extractCoordArg<GridType>(coordObj, "getValueDepth");
        return mAccessor.getValueDepth(ijk);
    }

    int isVoxel(py::object coordObj)
    {
        const Coord ijk = extractCoordArg<GridType>(coordObj, "isVoxel");
        return mAccessor.isVoxel(ijk);
    }

    py::tuple probeValue(py::object coordObj)
    {
        const Coord ijk = extractCoordArg<GridType>(coordObj, "probeValue");
        ValueType value;
        bool on = mAccessor.probeValue(ijk, value);
        return py::make_tuple(value, on);
    }

    bool isValueOn(py::object coordObj)
    {
        const Coord ijk = extractCoordArg<GridType>(coordObj, "isValueOn");
        return mAccessor.isValueOn(ijk);
    }

    void setActiveState(py::object coordObj, bool on)
    {
        const Coord ijk = extractCoordArg<GridType>(coordObj, "setActiveState", /*argIdx=*/1);
        Traits::setActiveState(mAccessor, ijk, on);
    }

    void setValueOnly(py::object coordObj, py::object valObj)
    {
        Coord ijk = extractCoordArg<GridType>(coordObj, "setValueOnly", 1);
        ValueType val = extractValueArg<GridType>(valObj, "setValueOnly", 2);
        Traits::setValueOnly(mAccessor, ijk, val);
    }

    void setValueOn(py::object coordObj, py::object valObj)
    {
        Coord ijk = extractCoordArg<GridType>(coordObj, "setValueOn", 1);
        if (valObj.is_none()) {
            Traits::setValueOn(mAccessor, ijk);
        } else {
            ValueType val = extractValueArg<GridType>(valObj, "setValueOn", 2);
            Traits::setValueOn(mAccessor, ijk, val);
        }
    }

    void setValueOff(py::object coordObj, py::object valObj)
    {
        Coord ijk = extractCoordArg<GridType>(coordObj, "setValueOff", 1);
        if (valObj.is_none()) {
            Traits::setValueOff(mAccessor, ijk);
        } else {
            ValueType val = extractValueArg<GridType>(valObj, "setValueOff", 2);
            Traits::setValueOff(mAccessor, ijk, val);
        }
    }

    int isCached(py::object coordObj)
    {
        const Coord ijk = extractCoordArg<GridType>(coordObj, "isCached");
        return mAccessor.isCached(ijk);
    }

    /// @brief Define a Python wrapper class for this C++ class.
    static void wrap()
    {
        const std::string
            pyGridTypeName = pyutil::GridTraits<GridType>::name(),
            pyValueTypeName = openvdb::typeNameAsString<typename GridType::ValueType>(),
            pyAccessorTypeName = Traits::typeName();

        py::class_<AccessorWrap> clss(
            pyAccessorTypeName.c_str(),
            (std::string(Traits::IsConst ? "Read-only" : "Read/write")
                + " access by (i, j, k) index coordinates to the voxels\nof a "
                + pyGridTypeName).c_str(),
            py::no_init);

        clss.def("copy", &AccessorWrap::copy,
                ("copy() -> " + pyAccessorTypeName + "\n\n"
                 "Return a copy of this accessor.").c_str())

            .def("clear", &AccessorWrap::clear,
                "clear()\n\n"
                "Clear this accessor of all cached data.")

            .add_property("parent", &AccessorWrap::parent,
                ("this accessor's parent " + pyGridTypeName).c_str())

            //
            // Voxel access
            //
            .def("getValue", &AccessorWrap::getValue,
                py::arg("ijk"),
                ("getValue(ijk) -> " + pyValueTypeName + "\n\n"
                 "Return the value of the voxel at coordinates (i, j, k).").c_str())

            .def("getValueDepth", &AccessorWrap::getValueDepth,
                py::arg("ijk"),
                "getValueDepth(ijk) -> int\n\n"
                "Return the tree depth (0 = root) at which the value of voxel\n"
                "(i, j, k) resides.  If (i, j, k) isn't explicitly represented in\n"
                "the tree (i.e., it is implicitly a background voxel), return -1.")

            .def("isVoxel", &AccessorWrap::isVoxel,
                py::arg("ijk"),
                "isVoxel(ijk) -> bool\n\n"
                "Return True if voxel (i, j, k) resides at the leaf level of the tree.")

            .def("probeValue", &AccessorWrap::probeValue,
                py::arg("ijk"),
                "probeValue(ijk) -> value, bool\n\n"
                "Return the value of the voxel at coordinates (i, j, k)\n"
                "together with the voxel's active state.")

            .def("isValueOn", &AccessorWrap::isValueOn,
                py::arg("ijk"),
                "isValueOn(ijk) -> bool\n\n"
                "Return the active state of the voxel at coordinates (i, j, k).")
            .def("setActiveState", &AccessorWrap::setActiveState,
                (py::arg("ijk"), py::arg("on")),
                "setActiveState(ijk, on)\n\n"
                "Mark voxel (i, j, k) as either active or inactive (True or False),\n"
                "but don't change its value.")

            .def("setValueOnly", &AccessorWrap::setValueOnly,
                (py::arg("ijk"), py::arg("value")),
                "setValueOnly(ijk, value)\n\n"
                "Set the value of voxel (i, j, k), but don't change its active state.")

            .def("setValueOn", &AccessorWrap::setValueOn,
                (py::arg("ijk"), py::arg("value") = py::object()),
                "setValueOn(ijk, value=None)\n\n"
                "Mark voxel (i, j, k) as active and, if the given value\n"
                "is not None, set the voxel's value.\n")
            .def("setValueOff", &AccessorWrap::setValueOff,
                (py::arg("ijk"), py::arg("value") = py::object()),
                "setValueOff(ijk, value=None)\n\n"
                "Mark voxel (i, j, k) as inactive and, if the given value\n"
                "is not None, set the voxel's value.")

            .def("isCached", &AccessorWrap::isCached,
                py::arg("ijk"),
                "isCached(ijk) -> bool\n\n"
                "Return True if this accessor has cached the path to voxel (i, j, k).")

            ; // py::class_<ValueAccessor>
    }

private:
    const GridPtrType mGrid;
    Accessor mAccessor;
}; // class AccessorWrap

} // namespace pyAccessor

#endif // OPENVDB_PYACCESSOR_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
