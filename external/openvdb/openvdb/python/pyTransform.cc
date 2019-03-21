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

#include <boost/python.hpp>
#include "openvdb/openvdb.h"
#include "pyutil.h"

namespace py = boost::python;
using namespace openvdb::OPENVDB_VERSION_NAME;

namespace pyTransform {

inline void scale1(math::Transform& t, double s) { t.preScale(s); }
inline void scale3(math::Transform& t, const Vec3d& xyz) { t.preScale(xyz); }

inline Vec3d voxelDim0(math::Transform& t) { return t.voxelSize(); }
inline Vec3d voxelDim1(math::Transform& t, const Vec3d& p) { return t.voxelSize(p); }

inline double voxelVolume0(math::Transform& t) { return t.voxelVolume(); }
inline double voxelVolume1(math::Transform& t, const Vec3d& p) { return t.voxelVolume(p); }

inline Vec3d indexToWorld(math::Transform& t, const Vec3d& p) { return t.indexToWorld(p); }
inline Vec3d worldToIndex(math::Transform& t, const Vec3d& p) { return t.worldToIndex(p); }

inline Coord worldToIndexCellCentered(math::Transform& t, const Vec3d& p) {
    return t.worldToIndexCellCentered(p);
}
inline Coord worldToIndexNodeCentered(math::Transform& t, const Vec3d& p) {
    return t.worldToIndexNodeCentered(p);
}


inline std::string
info(math::Transform& t)
{
    std::ostringstream ostr;
    t.print(ostr);
    return ostr.str();
}


inline math::Transform::Ptr
createLinearFromDim(double dim)
{
    return math::Transform::createLinearTransform(dim);
}


inline math::Transform::Ptr
createLinearFromMat(py::object obj)
{
    Mat4R m;

    // Verify that obj is a four-element sequence.
    bool is4x4Seq = (PySequence_Check(obj.ptr()) && PySequence_Length(obj.ptr()) == 4);
    if (is4x4Seq) {
        for (int row = 0; is4x4Seq && row < 4; ++row) {
            // Verify that each element of obj is itself a four-element sequence.
            py::object rowObj = obj[row];
            if (PySequence_Check(rowObj.ptr()) && PySequence_Length(rowObj.ptr()) == 4) {
                // Extract four numeric values from this row of the sequence.
                for (int col = 0; is4x4Seq && col < 4; ++col) {
                    if (py::extract<double>(rowObj[col]).check()) {
                        m[row][col] = py::extract<double>(rowObj[col]);
                    } else {
                        is4x4Seq = false;
                    }
                }
            } else {
                is4x4Seq = false;
            }
        }
    }
    if (!is4x4Seq) {
        PyErr_Format(PyExc_ValueError, "expected a 4 x 4 sequence of numeric values");
        py::throw_error_already_set();
    }

    return math::Transform::createLinearTransform(m);
}


inline math::Transform::Ptr
createFrustum(const Coord& xyzMin, const Coord& xyzMax,
    double taper, double depth, double voxelDim = 1.0)
{
    return math::Transform::createFrustumTransform(
        BBoxd(xyzMin.asVec3d(), xyzMax.asVec3d()), taper, depth, voxelDim);
}


////////////////////////////////////////


struct PickleSuite: public py::pickle_suite
{
    enum { STATE_DICT = 0, STATE_MAJOR, STATE_MINOR, STATE_FORMAT, STATE_XFORM };

    /// Return @c true, indicating that this pickler preserves a Transform's __dict__.
    static bool getstate_manages_dict() { return true; }

    /// Return a tuple representing the state of the given Transform.
    static py::tuple getstate(py::object xformObj)
    {
        py::tuple state;
        py::extract<math::Transform> x(xformObj);
        if (x.check()) {
            // Extract a Transform from the Python object.
            math::Transform xform = x();
            std::ostringstream ostr(std::ios_base::binary);
            // Serialize the Transform to a string.
            xform.write(ostr);

            // Construct a state tuple comprising the Python object's __dict__,
            // the version numbers of the serialization format,
            // and the serialized Transform.
#if PY_MAJOR_VERSION >= 3
            // Convert the byte string to a "bytes" sequence.
            const std::string s = ostr.str();
            py::object bytesObj = pyutil::pyBorrow(PyBytes_FromStringAndSize(s.data(), s.size()));
#else
            py::str bytesObj(ostr.str());
#endif
            state = py::make_tuple(
                xformObj.attr("__dict__"),
                uint32_t(OPENVDB_LIBRARY_MAJOR_VERSION),
                uint32_t(OPENVDB_LIBRARY_MINOR_VERSION),
                uint32_t(OPENVDB_FILE_VERSION),
                bytesObj);
        }
        return state;
    }

    /// Restore the given Transform to a saved state.
    static void setstate(py::object xformObj, py::object stateObj)
    {
        math::Transform* xform = nullptr;
        {
            py::extract<math::Transform*> x(xformObj);
            if (x.check()) xform = x();
            else return;
        }

        py::tuple state;
        {
            py::extract<py::tuple> x(stateObj);
            if (x.check()) state = x();
        }
        bool badState = (py::len(state) != 5);

        if (!badState) {
            // Restore the object's __dict__.
            py::extract<py::dict> x(state[int(STATE_DICT)]);
            if (x.check()) {
                py::dict d = py::extract<py::dict>(xformObj.attr("__dict__"))();
                d.update(x());
            } else {
                badState = true;
            }
        }

        openvdb::VersionId libVersion;
        uint32_t formatVersion = 0;
        if (!badState) {
            // Extract the serialization format version numbers.
            const int idx[3] = { STATE_MAJOR, STATE_MINOR, STATE_FORMAT };
            uint32_t version[3] = { 0, 0, 0 };
            for (int i = 0; i < 3 && !badState; ++i) {
                py::extract<uint32_t> x(state[idx[i]]);
                if (x.check()) version[i] = x();
                else badState = true;
            }
            libVersion.first = version[0];
            libVersion.second = version[1];
            formatVersion = version[2];
        }

        std::string serialized;
        if (!badState) {
            // Extract the sequence containing the serialized Transform.
            py::object bytesObj = state[int(STATE_XFORM)];
#if PY_MAJOR_VERSION >= 3
            badState = true;
            if (PyBytes_Check(bytesObj.ptr())) {
                // Convert the "bytes" sequence to a byte string.
                char* buf = NULL;
                Py_ssize_t length = 0;
                if (-1 != PyBytes_AsStringAndSize(bytesObj.ptr(), &buf, &length)) {
                    if (buf != NULL && length > 0) {
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
                ("expected (dict, int, int, int, bytes) tuple in call to __setstate__; found %s"
#else
                ("expected (dict, int, int, int, str) tuple in call to __setstate__; found %s"
#endif
                     % stateObj.attr("__repr__")()).ptr());
            py::throw_error_already_set();
        }

        // Restore the internal state of the C++ object.
        std::istringstream istr(serialized, std::ios_base::binary);
        io::setVersion(istr, libVersion, formatVersion);
        xform->read(istr);
    }
}; // struct PickleSuite

} // namespace pyTransform


void exportTransform();

void
exportTransform()
{
    py::enum_<math::Axis>("Axis")
        .value("X", math::X_AXIS)
        .value("Y", math::Y_AXIS)
        .value("Z", math::Z_AXIS);

    py::class_<math::Transform>("Transform", py::init<>())

        .def("deepCopy", &math::Transform::copy,
            "deepCopy() -> Transform\n\n"
            "Return a copy of this transform.")

        /// @todo Should this also be __str__()?
        .def("info", &pyTransform::info,
            "info() -> str\n\n"
            "Return a string containing a description of this transform.\n")

        .def_pickle(pyTransform::PickleSuite())

        .add_property("typeName", &math::Transform::mapType,
            "name of this transform's type")
        .add_property("isLinear", &math::Transform::isLinear,
            "True if this transform is linear")

        .def("rotate", &math::Transform::preRotate,
            (py::arg("radians"), py::arg("axis") = math::X_AXIS),
            "rotate(radians, axis)\n\n"
            "Accumulate a rotation about either Axis.X, Axis.Y or Axis.Z.")
        .def("translate", &math::Transform::postTranslate, py::arg("xyz"),
            "translate((x, y, z))\n\n"
            "Accumulate a translation.")
        .def("scale", &pyTransform::scale1, py::arg("s"),
            "scale(s)\n\n"
            "Accumulate a uniform scale.")
        .def("scale", &pyTransform::scale3, py::arg("sxyz"),
            "scale((sx, sy, sz))\n\n"
            "Accumulate a nonuniform scale.")
        .def("shear", &math::Transform::preShear,
            (py::arg("s"), py::arg("axis0"), py::arg("axis1")),
            "shear(s, axis0, axis1)\n\n"
            "Accumulate a shear (axis0 and axis1 are either\n"
            "Axis.X, Axis.Y or Axis.Z).")

        .def("voxelSize", &pyTransform::voxelDim0,
            "voxelSize() -> (dx, dy, dz)\n\n"
            "Return the size of voxels of the linear component of this transform.")
        .def("voxelSize", &pyTransform::voxelDim1, py::arg("xyz"),
            "voxelSize((x, y, z)) -> (dx, dy, dz)\n\n"
            "Return the size of the voxel at position (x, y, z).")

        .def("voxelVolume", &pyTransform::voxelVolume0,
            "voxelVolume() -> float\n\n"
            "Return the voxel volume of the linear component of this transform.")
        .def("voxelVolume", &pyTransform::voxelVolume1, py::arg("xyz"),
            "voxelVolume((x, y, z)) -> float\n\n"
            "Return the voxel volume at position (x, y, z).")

        .def("indexToWorld", &pyTransform::indexToWorld, py::arg("xyz"),
            "indexToWorld((x, y, z)) -> (x', y', z')\n\n"
            "Apply this transformation to the given coordinates.")
        .def("worldToIndex", &pyTransform::worldToIndex, py::arg("xyz"),
            "worldToIndex((x, y, z)) -> (x', y', z')\n\n"
            "Apply the inverse of this transformation to the given coordinates.")
        .def("worldToIndexCellCentered", &pyTransform::worldToIndexCellCentered,
            py::arg("xyz"),
            "worldToIndexCellCentered((x, y, z)) -> (i, j, k)\n\n"
            "Apply the inverse of this transformation to the given coordinates\n"
            "and round the result to the nearest integer coordinates.")
        .def("worldToIndexNodeCentered", &pyTransform::worldToIndexNodeCentered,
            py::arg("xyz"),
            "worldToIndexNodeCentered((x, y, z)) -> (i, j, k)\n\n"
            "Apply the inverse of this transformation to the given coordinates\n"
            "and round the result down to the nearest integer coordinates.")

        // Allow Transforms to be compared for equality and inequality.
        .def(py::self == py::other<math::Transform>())
        .def(py::self != py::other<math::Transform>())
        ;

    py::def("createLinearTransform", &pyTransform::createLinearFromMat, py::arg("matrix"),
        "createLinearTransform(matrix) -> Transform\n\n"
        "Create a new linear transform from a 4 x 4 matrix given as a sequence\n"
        "of the form [[a, b, c, d], [e, f, g, h], [i, j, k, l], [m, n, o, p]],\n"
        "where [m, n, o, p] is the translation component.");

    py::def("createLinearTransform", &pyTransform::createLinearFromDim,
        (py::arg("voxelSize") = 1.0),
        "createLinearTransform(voxelSize) -> Transform\n\n"
        "Create a new linear transform with the given uniform voxel size.");

    py::def("createFrustumTransform", &pyTransform::createFrustum,
        (py::arg("xyzMin"), py::arg("xyzMax"),
         py::arg("taper"), py::arg("depth"), py::arg("voxelSize") = 1.0),
        "createFrustumTransform(xyzMin, xyzMax, taper, depth, voxelSize) -> Transform\n\n"
        "Create a new frustum transform with unit bounding box (xyzMin, xyzMax)\n"
        "and the given taper, depth and uniform voxel size.");

    // allows Transform::Ptr Grid::getTransform() to work
    py::register_ptr_to_python<math::Transform::Ptr>();
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
