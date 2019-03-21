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

#ifndef OPENVDB_PYUTIL_HAS_BEEN_INCLUDED
#define OPENVDB_PYUTIL_HAS_BEEN_INCLUDED

#include "openvdb/openvdb.h"
#include <boost/python.hpp>
#include <tbb/mutex.h>
#include <map> // for std::pair
#include <string>
#include <sstream>


namespace pyutil {

/// Return a new @c boost::python::object that borrows (i.e., doesn't
/// take over ownership of) the given @c PyObject's reference.
inline boost::python::object
pyBorrow(PyObject* obj)
{
    return boost::python::object(boost::python::handle<>(boost::python::borrowed(obj)));
}


/// @brief Given a @c PyObject that implements the sequence protocol
/// (e.g., a @c PyListObject), return the value of type @c ValueT
/// at index @a idx in the sequence.
/// @details Raise a Python @c TypeError exception if the value
/// at index @a idx is not convertible to type @c ValueT.
template<typename ValueT>
inline ValueT
getSequenceItem(PyObject* obj, int idx)
{
    return boost::python::extract<ValueT>(pyBorrow(obj)[idx]);
}


////////////////////////////////////////


template<class GridType>
struct GridTraitsBase
{
    /// @brief Return the name of the Python class that wraps this grid type
    /// (e.g., "FloatGrid" for openvdb::FloatGrid).
    ///
    /// @note This name is not the same as GridType::type().
    /// The latter returns a name like "Tree_float_5_4_3".
    static const char* name();

    /// Return the name of this grid type's value type ("bool", "float", "vec3s", etc.).
    static const char* valueTypeName()
    {
        return openvdb::typeNameAsString<typename GridType::ValueType>();
    }

    /// @brief Return a description of this grid type.
    ///
    /// @note This name is generated at runtime for each call to descr().
    static const std::string descr()
    {
        return std::string("OpenVDB grid with voxels of type ") + valueTypeName();
    }
}; // struct GridTraitsBase


template<class GridType>
struct GridTraits: public GridTraitsBase<GridType>
{
};

/// Map a grid type to a traits class that derives from GridTraitsBase
/// and that defines a name() method.
#define GRID_TRAITS(_typ, _name) \
    template<> struct GridTraits<_typ>: public GridTraitsBase<_typ> { \
        static const char* name() { return _name; } \
    }

GRID_TRAITS(openvdb::FloatGrid, "FloatGrid");
GRID_TRAITS(openvdb::Vec3SGrid, "Vec3SGrid");
GRID_TRAITS(openvdb::BoolGrid, "BoolGrid");
#ifdef PY_OPENVDB_WRAP_ALL_GRID_TYPES
GRID_TRAITS(openvdb::DoubleGrid, "DoubleGrid");
GRID_TRAITS(openvdb::Int32Grid, "Int32Grid");
GRID_TRAITS(openvdb::Int64Grid, "Int64Grid");
GRID_TRAITS(openvdb::Vec3IGrid, "Vec3IGrid");
GRID_TRAITS(openvdb::Vec3DGrid, "Vec3DGrid");
#endif

#undef GRID_TRAITS


////////////////////////////////////////


// Note that the elements are pointers to C strings (char**), because
// boost::python::class_::def_readonly() requires a pointer to a static member.
typedef std::pair<const char* const*, const char* const*> CStringPair;


/// @brief Enum-like mapping from string keys to string values, with characteristics
/// of both (Python) classes and class instances (as well as NamedTuples)
/// @details
/// - (@e key, @e value) pairs can be accessed as class attributes (\"<tt>MyClass.MY_KEY</tt>\")
/// - (@e key, @e value) pairs can be accessed via dict lookup on instances
///   (\"<tt>MyClass()['MY_KEY']</tt>\")
/// - (@e key, @e value) pairs can't be modified or reassigned
/// - instances are iterable (\"<tt>for key in MyClass(): ...</tt>\")
///
/// A @c Descr class must implement the following interface:
/// @code
/// struct MyDescr
/// {
///     // Return the Python name for the enum class.
///     static const char* name();
///     // Return the docstring for the enum class.
///     static const char* doc();
///     // Return the ith (key, value) pair, in the form of
///     // a pair of *pointers* to C strings
///     static CStringPair item(int i);
/// };
/// @endcode
template<typename Descr>
struct StringEnum
{
    /// Return the (key, value) map as a Python dict.
    static boost::python::dict items()
    {
        static tbb::mutex sMutex;
        static boost::python::dict itemDict;
        if (!itemDict) {
            // The first time this function is called, populate
            // the static dict with (key, value) pairs.
            tbb::mutex::scoped_lock lock(sMutex);
            if (!itemDict) {
                for (int i = 0; ; ++i) {
                    const CStringPair item = Descr::item(i);
                    OPENVDB_START_THREADSAFE_STATIC_WRITE
                    if (item.first) {
                        itemDict[boost::python::str(*item.first)] =
                            boost::python::str(*item.second);
                    }
                    OPENVDB_FINISH_THREADSAFE_STATIC_WRITE
                    else break;
                }
            }
        }
        return itemDict;
    }

    /// Return the keys as a Python list of strings.
    static boost::python::object keys() { return items().attr("keys")(); }
    /// Return the number of keys as a Python int.
    boost::python::object numItems() const
    {
        return boost::python::object(boost::python::len(items()));
    }
    /// Return the value (as a Python string) for the given key.
    boost::python::object getItem(boost::python::object keyObj) const { return items()[keyObj]; }
    /// Return a Python iterator over the keys.
    boost::python::object iter() const { return items().attr("__iter__")(); }

    /// Register this enum.
    static void wrap()
    {
        boost::python::class_<StringEnum> cls(
            /*classname=*/Descr::name(),
            /*docstring=*/Descr::doc());
        cls.def("keys", &StringEnum::keys, "keys() -> list")
            .staticmethod("keys")
            .def("__len__", &StringEnum::numItems, "__len__() -> int")
            .def("__iter__", &StringEnum::iter, "__iter__() -> iterator")
            .def("__getitem__", &StringEnum::getItem, "__getitem__(str) -> str")
            /*end*/;
        // Add a read-only, class-level attribute for each (key, value) pair.
        for (int i = 0; ; ++i) {
            const CStringPair item = Descr::item(i);
            if (item.first) cls.def_readonly(*item.first, item.second);
            else break;
        }
    }
};


////////////////////////////////////////


/// @brief From the given Python object, extract a value of type @c T.
///
/// If the object cannot be converted to type @c T, raise a @c TypeError with a more
/// Pythonic error message (incorporating the provided class and function names, etc.)
/// than the one that would be generated by boost::python::extract(), e.g.,
/// "TypeError: expected float, found str as argument 2 to FloatGrid.prune()" instead of
/// "TypeError: No registered converter was able to produce a C++ rvalue of type
/// boost::shared_ptr<openvdb::Grid<openvdb::tree::Tree<openvdb::tree::RootNode<...".
template<typename T>
inline T
extractArg(
    boost::python::object obj,
    const char* functionName,
    const char* className = NULL,
    int argIdx = 0, // args are numbered starting from 1
    const char* expectedType = NULL)
{
    boost::python::extract<T> val(obj);
    if (!val.check()) {
        // Generate an error string of the form
        // "expected <expectedType>, found <actualType> as argument <argIdx>
        // to <className>.<functionName>()", where <argIdx> and <className>
        // are optional.
        std::ostringstream os;
        os << "expected ";
        if (expectedType) os << expectedType; else os << openvdb::typeNameAsString<T>();
        const std::string actualType =
            boost::python::extract<std::string>(obj.attr("__class__").attr("__name__"));
        os << ", found " << actualType << " as argument";
        if (argIdx > 0) os << " " << argIdx;
        os << " to ";
        if (className) os << className << ".";
        os << functionName << "()";

        PyErr_SetString(PyExc_TypeError, os.str().c_str());
        boost::python::throw_error_already_set();
    }
    return val();
}


////////////////////////////////////////


/// Return str(val) for the given value.
template<typename T>
inline std::string
str(const T& val)
{
    return boost::python::extract<std::string>(boost::python::str(val));
}


/// Return the name of the given Python object's class.
inline std::string
className(boost::python::object obj)
{
    std::string s = boost::python::extract<std::string>(
        obj.attr("__class__").attr("__name__"));
    return s;
}

} // namespace pyutil

#endif // OPENVDB_PYUTIL_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
