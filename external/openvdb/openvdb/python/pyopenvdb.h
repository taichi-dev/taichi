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
/// @file pyopenvdb.h
///
/// @brief Glue functions for access to pyOpenVDB objects from C++ code
/// @details Use these functions in your own Python function implementations
/// to extract an OpenVDB grid from or wrap a grid in a @c PyObject.
/// For example (using Boost.Python),
/// @code
/// #include <openvdb.h>
/// #include <pyopenvdb.h>
/// #include <boost/python.hpp>
///
/// // Implementation of a Python function that processes pyOpenVDB grids
/// boost::python::object
/// processGrid(boost::python::object inObj)
/// {
///     boost::python::object outObj;
///     try {
///         // Extract an OpenVDB grid from the input argument.
///         if (openvdb::GridBase::Ptr grid =
///             pyopenvdb::getGridFromPyObject(inObj))
///         {
///             grid = grid->deepCopyGrid();
///
///             // Process the grid...
///
///             // Wrap the processed grid in a PyObject.
///             outObj = pyopenvdb::getPyObjectFromGrid(grid);
///         }
///     } catch (openvdb::TypeError& e) {
///         PyErr_Format(PyExc_TypeError, e.what());
///         boost::python::throw_error_already_set();
///     }
///     return outObj;
/// }
///
/// BOOST_PYTHON_MODULE(mymodule)
/// {
///     openvdb::initialize();
///
///     // Definition of a Python function that processes pyOpenVDB grids
///     boost::python::def(/*name=*/"processGrid", &processGrid, /*argname=*/"grid");
/// }
/// @endcode
/// Then, from Python,
/// @code
/// import openvdb
/// import mymodule
///
/// grid = openvdb.read('myGrid.vdb', 'MyGrid')
/// grid = mymodule.processGrid(grid)
/// openvdb.write('myProcessedGrid.vdb', [grid])
/// @endcode

#ifndef PYOPENVDB_HAS_BEEN_INCLUDED
#define PYOPENVDB_HAS_BEEN_INCLUDED

#include <boost/python.hpp>
#include "openvdb/Grid.h"


namespace pyopenvdb {

//@{
/// @brief Return a pointer to the OpenVDB grid held by the given Python object.
/// @throw openvdb::TypeError if the Python object is not one of the pyOpenVDB grid types.
///     (See the Python module's GridTypes global variable for the list of supported grid types.)
openvdb::GridBase::Ptr getGridFromPyObject(PyObject*);
openvdb::GridBase::Ptr getGridFromPyObject(const boost::python::object&);
//@}

/// @brief Return a new Python object that holds the given OpenVDB grid.
/// @return @c None if the given grid pointer is null.
/// @throw openvdb::TypeError if the grid is not of a supported type.
///     (See the Python module's GridTypes global variable for the list of supported grid types.)
boost::python::object getPyObjectFromGrid(const openvdb::GridBase::Ptr&);

} // namespace pyopenvdb

#endif // PYOPENVDB_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
