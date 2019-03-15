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

namespace py = boost::python;
using namespace openvdb::OPENVDB_VERSION_NAME;

namespace {

class MetadataWrap: public Metadata, public py::wrapper<Metadata>
{
public:
    Name typeName() const { return static_cast<const Name&>(this->get_override("typeName")()); }
    Metadata::Ptr copy() const {
        return static_cast<const Metadata::Ptr&>(this->get_override("copy")());
    }
    void copy(const Metadata& other) { this->get_override("copy")(other); }
    std::string str() const {return static_cast<const std::string&>(this->get_override("str")());}
    bool asBool() const { return static_cast<const bool&>(this->get_override("asBool")()); }
    Index32 size() const { return static_cast<const Index32&>(this->get_override("size")()); }

protected:
    void readValue(std::istream& is, Index32 numBytes) {
        this->get_override("readValue")(is, numBytes);
    }
    void writeValue(std::ostream& os) const {
        this->get_override("writeValue")(os);
    }
};

// aliases disambiguate the different versions of copy
Metadata::Ptr (MetadataWrap::*copy0)() const = &MetadataWrap::copy;
void (MetadataWrap::*copy1)(const Metadata&) = &MetadataWrap::copy;

} // end anonymous namespace


void exportMetadata();

void
exportMetadata()
{
    py::class_<MetadataWrap, boost::noncopyable> clss(
        /*classname=*/"Metadata",
        /*docstring=*/
            "Class that holds the value of a single item of metadata of a type\n"
            "for which no Python equivalent exists (typically a custom type)",
        /*ctor=*/py::no_init // can only be instantiated from C++, not from Python
    );
    clss.def("copy", py::pure_virtual(copy0),
            "copy() -> Metadata\n\nReturn a copy of this value.")
        .def("copy", py::pure_virtual(copy1),
            "copy() -> Metadata\n\nReturn a copy of this value.")
        .def("type", py::pure_virtual(&Metadata::typeName),
            "type() -> str\n\nReturn the name of this value's type.")
        .def("size", py::pure_virtual(&Metadata::size),
            "size() -> int\n\nReturn the size of this value in bytes.")
        .def("__nonzero__", py::pure_virtual(&Metadata::asBool))
        .def("__str__", py::pure_virtual(&Metadata::str))
        ;
    py::register_ptr_to_python<Metadata::Ptr>();
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
