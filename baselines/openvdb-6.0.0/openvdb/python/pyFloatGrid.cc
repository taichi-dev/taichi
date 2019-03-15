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
/// @file pyFloatGrid.cc
/// @author Peter Cucka
/// @brief Boost.Python wrappers for scalar, floating-point openvdb::Grid types

#include "pyGrid.h"


void exportFloatGrid();


/// Create a Python wrapper for each supported Grid type.
void
exportFloatGrid()
{
    // Add a module-level list that gives the types of all supported Grid classes.
    py::scope().attr("GridTypes") = py::list();

#if defined(PY_OPENVDB_USE_NUMPY) && !defined(PY_OPENVDB_USE_BOOST_PYTHON_NUMPY)
    // Specify that py::numeric::array should refer to the Python type numpy.ndarray
    // (rather than the older Numeric.array).
    py::numeric::array::set_module_and_type("numpy", "ndarray");
#endif

    pyGrid::exportGrid<FloatGrid>();
#ifdef PY_OPENVDB_WRAP_ALL_GRID_TYPES
    pyGrid::exportGrid<DoubleGrid>();
#endif

    py::def("createLevelSetSphere",
        &pyGrid::createLevelSetSphere<FloatGrid>,
        (py::arg("radius"), py::arg("center")=openvdb::Coord(), py::arg("voxelSize")=1.0,
             py::arg("halfWidth")=openvdb::LEVEL_SET_HALF_WIDTH),
        "createLevelSetSphere(radius, center, voxelSize, halfWidth) -> FloatGrid\n\n"
        "Return a grid containing a narrow-band level set representation\n"
        "of a sphere.");
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
