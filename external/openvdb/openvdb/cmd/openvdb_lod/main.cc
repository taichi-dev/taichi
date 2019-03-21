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

#include <openvdb/openvdb.h>
#include <openvdb/tools/MultiResGrid.h>
#include <openvdb/util/CpuTimer.h>
#include <openvdb/util/logging.h>
#include <boost/algorithm/string/classification.hpp> // for boost::is_any_of()
#include <boost/algorithm/string/split.hpp>
#include <cstdlib> // for std::atof()
#include <iomanip> // for std::setprecision()
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept> // for std::runtime_error
#include <string>


namespace {

const char* gProgName = "";

inline void
usage [[noreturn]] (int exitStatus = EXIT_FAILURE)
{
    std::cerr <<
"Usage: " << gProgName << " in.vdb out.vdb -range FROM[-TO[:STEP]] [options]\n" <<
"Which: generates a volume mipmap from an OpenVDB grid\n" <<
"Where:\n" <<
"    FROM  is the highest-resolution mip level to be generated\n" <<
"    TO    is the lowest-resolution mip level to be generated (default: FROM)\n" <<
"    STEP  is the mip level step size (default: 1)\n" <<
"Options:\n" <<
"    -name S[,S,S,...]  name(s) of the grid(s) to be processed\n" <<
"                       (default: process all grids of supported types)\n" <<
"    -keep              pass through grids that were not processed\n" <<
"                       (default: discard grids that were not processed)\n" <<
"    -nokeep            cancel an earlier -keep option\n" <<
"    -p, -preserve      if only one mip level is generated, give it the same\n" <<
"                       name as the original grid (default: name each level\n" <<
"                       \"NAME_level_N\", where NAME is the original grid name\n" <<
"                       and N is the level number, e.g., \"density_level_0\")\n" <<
"    -nopreserve        cancel an earlier -p or -preserve option\n" <<
"    -version           print version information\n" <<
"\n" <<
"Mip level 0 is the input grid.  Each successive integer level is half\n" <<
"the resolution of the previous level.  Fractional levels are supported.\n" <<
"\n" <<
"Examples:\n" <<
"    Generate levels 0, 1, and 2 (full resolution, half resolution,\n" <<
"    and quarter resolution, respectively) for all grids of supported types\n" <<
"    and ignore all other grids:\n" <<
"\n" <<
"        " << gProgName << " in.vdb out.vdb -range 0-2\n" <<
"\n" <<
"    Generate levels 0, 0.5, and 1 for all grids of supported types\n" <<
"    and pass through all other grids:\n" <<
"\n" <<
"        " << gProgName << " in.vdb out.vdb -range 0-1:0.5 -keep\n" <<
"\n" <<
"    Generate level 3 for the first of multiple grids named \"density\":\n" <<
"\n" <<
"        " << gProgName << " in.vdb out.vdb -range 3 -name 'density[0]'\n" <<
"\n" <<
"    Generate level 1.5 for the second of multiple unnamed grids and for\n" <<
"    the grid named \"velocity\" and give the resulting grids the same names\n"<<
"    as the original grids:\n" <<
"\n" <<
"        " << gProgName << " in.vdb out.vdb -range 1.5 -name '[1],velocity' -p\n" <<
"\n";
    exit(exitStatus);
}


struct Options
{
    Options(): from(0.0), to(0.0), step(1.0), keep(false), preserve(false) {}

    double from, to, step;
    bool keep, preserve;
};


/// @brief Parse a string of the form "from-to:step" and populate the given @a opts
/// with the resulting values.
/// @throw std::runtime_error if parsing fails for any reason
inline void
parseRangeSpec(const std::string& rangeSpec, Options& opts)
{
    // Split on the "-" character, of which there should be at most one.
    std::vector<std::string> rangeItems;
    boost::split(rangeItems, rangeSpec, boost::is_any_of("-"));
    if (rangeItems.empty() || rangeItems.size() > 2) throw std::runtime_error("");

    // Extract the "from" value, and default "to" to "from" and "step" to 1.
    opts.from = opts.to = std::atof(rangeItems[0].c_str());
    opts.step = 1.0;

    if (rangeItems.size() > 1) {
        // Split on the ":" character, of which there should be at most one.
        const std::string item = rangeItems[1];
        boost::split(rangeItems, item, boost::is_any_of(":"));
        if (rangeItems.empty() || rangeItems.size() > 2) throw std::runtime_error("");

        // Extract the "to" value.
        opts.to = std::atof(rangeItems[0].c_str());
        if (rangeItems.size() > 1) {
            // Extract the "step" value.
            opts.step = std::atof(rangeItems[1].c_str());
        }
    }

    if (opts.from < 0.0 || opts.to < opts.from || opts.step <= 0.0) throw std::runtime_error("");
}


/// @brief Mipmap a single grid of a fully-resolved type.
/// @return a vector of pointers to the member grids of the mipmap
template<typename GridType>
inline openvdb::GridPtrVec
mip(const GridType& inGrid, const Options& opts)
{
    OPENVDB_LOG_INFO("processing grid \"" << inGrid.getName() << "\"");

    // MultiResGrid requires at least two mipmap levels, starting from level 0.
    const int levels = std::max(2, openvdb::math::Ceil(opts.to) + 1);

    openvdb::util::CpuTimer timer;
    timer.start();

    // Initialize the mipmap.
    typedef typename GridType::TreeType TreeT;
    openvdb::tools::MultiResGrid<TreeT> mrg(levels, inGrid);

    openvdb::GridPtrVec outGrids;
    for (double level = opts.from; level <= opts.to; level += opts.step) {
        // Request a level from the mipmap.
        if (openvdb::GridBase::Ptr levelGrid =
            mrg.template createGrid</*sampling order=*/1>(static_cast<float>(level)))
        {
            outGrids.push_back(levelGrid);
        }
    }

    const double msec = timer.delta(); // elapsed time

    if (outGrids.size() == 1 && opts.preserve) {
        // If -preserve is in effect and there is only one output grid,
        // give it the same name as the input grid.
        outGrids[0]->setName(inGrid.getName());
    }

    OPENVDB_LOG_INFO("processed grid \"" << inGrid.getName() << "\" in "
        << std::setprecision(3) << (msec / 1000.0) << " sec");

    return outGrids;
}


/// @brief Mipmap a single grid and append the resulting grids to @a outGrids.
inline void
process(const openvdb::GridBase::Ptr& baseGrid, openvdb::GridPtrVec& outGrids, const Options& opts)
{
    using namespace openvdb;

    if (!baseGrid) return;

    GridPtrVec mipmap;
    if (FloatGrid::Ptr g0 = GridBase::grid<FloatGrid>(baseGrid)) { mipmap = mip(*g0, opts); }
    else if (DoubleGrid::Ptr g1 = GridBase::grid<DoubleGrid>(baseGrid)) { mipmap = mip(*g1, opts); }
    else if (Vec3SGrid::Ptr  g2 = GridBase::grid<Vec3SGrid>(baseGrid))  { mipmap = mip(*g2, opts); }
    else if (Vec3DGrid::Ptr  g3 = GridBase::grid<Vec3DGrid>(baseGrid))  { mipmap = mip(*g3, opts); }
    else if (Vec3IGrid::Ptr  g4 = GridBase::grid<Vec3IGrid>(baseGrid))  { mipmap = mip(*g4, opts); }
    else if (Int32Grid::Ptr  g5 = GridBase::grid<Int32Grid>(baseGrid))  { mipmap = mip(*g5, opts); }
    else if (Int64Grid::Ptr  g6 = GridBase::grid<Int64Grid>(baseGrid))  { mipmap = mip(*g6, opts); }
    else if (BoolGrid::Ptr   g7 = GridBase::grid<BoolGrid>(baseGrid))   { mipmap = mip(*g7, opts); }
    else if (MaskGrid::Ptr   g8 = GridBase::grid<MaskGrid>(baseGrid))   { mipmap = mip(*g8, opts); }
    else {
        std::string operation = "skipped";
        if (opts.keep) {
            operation = "passed through";
            outGrids.push_back(baseGrid);
        };
        OPENVDB_LOG_WARN(operation << " grid \"" << baseGrid->getName()
            << "\" of unsupported type " << baseGrid->type());
    }
    outGrids.insert(outGrids.end(), mipmap.begin(), mipmap.end());
}

} // unnamed namespace


int
main(int argc, char *argv[])
{
    OPENVDB_START_THREADSAFE_STATIC_WRITE
    gProgName = argv[0];
    if (const char* ptr = ::strrchr(gProgName, '/')) gProgName = ptr + 1;
    OPENVDB_FINISH_THREADSAFE_STATIC_WRITE

    int exitStatus = EXIT_SUCCESS;

    if (argc == 1) usage();

    openvdb::logging::initialize(argc, argv);
    openvdb::initialize();

    // Parse command-line arguments.
    Options opts;
    bool version = false;
    std::string inFilename, outFilename, gridNameStr, rangeSpec;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg[0] == '-') {
            if (arg == "-name") {
                if (i + 1 < argc && argv[i + 1]) {
                    gridNameStr = argv[i + 1];
                    ++i;
                } else {
                    OPENVDB_LOG_FATAL("missing grid name(s) after -name");
                    usage();
                }
            } else if (arg == "-keep") {
                opts.keep = true;
            } else if (arg == "-nokeep") {
                opts.keep = false;
            } else if (arg == "-p" || arg == "-preserve") {
                opts.preserve = true;
            } else if (arg == "-nopreserve") {
                opts.preserve = false;
            } else if (arg == "-range") {
                if (i + 1 < argc && argv[i + 1]) {
                    rangeSpec = argv[i + 1];
                    ++i;
                } else {
                    OPENVDB_LOG_FATAL("missing level range specification after -range");
                    usage();
                }
            } else if (arg == "-h" || arg == "-help" || arg == "--help") {
                usage(EXIT_SUCCESS);
            } else if (arg == "-version" || arg == "--version") {
                version = true;
            } else {
                OPENVDB_LOG_FATAL("\"" << arg << "\" is not a valid option");
                usage();
            }
        } else if (!arg.empty()) {
            if (inFilename.empty()) {
                inFilename = arg;
            } else if (outFilename.empty()) {
                outFilename = arg;
            } else {
                OPENVDB_LOG_FATAL("unrecognized argument \"" << arg << "\"");
                usage();
            }
        }
    }

    if (version) {
        std::cout << "OpenVDB library version: "
            << openvdb::getLibraryVersionString() << "\n";
        std::cout << "OpenVDB file format version: "
            << openvdb::OPENVDB_FILE_VERSION << std::endl;
        if (outFilename.empty()) return EXIT_SUCCESS;
    }

    if (inFilename.empty()) {
        OPENVDB_LOG_FATAL("missing input OpenVDB filename");
        usage();
    }
    if (outFilename.empty()) {
        OPENVDB_LOG_FATAL("missing output OpenVDB filename");
        usage();
    }
    if (rangeSpec.empty()) {
        OPENVDB_LOG_FATAL("missing level range specification");
        usage();
    }

    try {
        parseRangeSpec(rangeSpec, opts);
    } catch (...) {
        OPENVDB_LOG_FATAL("invalid level range specification \"" << rangeSpec << "\"");
        usage();
    }

    // If -name was specified, generate a white list of names of grids to be processed.
    // Otherwise (if the white list is empty), process all grids of supported types.
    std::set<std::string> whitelist;
    if (!gridNameStr.empty()) {
        boost::split(whitelist, gridNameStr, boost::is_any_of(","));
    }

    // Process the input file.
    try {
        openvdb::io::File file(inFilename);
        file.open();

        const openvdb::MetaMap::ConstPtr fileMetadata = file.getMetadata();

        openvdb::GridPtrVec outGrids;

        // For each input grid...
        for (openvdb::io::File::NameIterator nameIter = file.beginName();
            nameIter != file.endName(); ++nameIter)
        {
            const std::string& name = nameIter.gridName();
            // If there is a white list, check if the grid is on the list.
            const bool skip = (!whitelist.empty() && (whitelist.find(name) == whitelist.end()));

            if (skip && !opts.keep) {
                OPENVDB_LOG_INFO("skipped grid \"" << name << "\"");
            } else {
                // If the grid's name is on the white list or if -keep is in effect, read the grid.
                openvdb::GridBase::Ptr baseGrid = file.readGrid(name);
                if (!baseGrid) {
                    OPENVDB_LOG_WARN("failed to read grid \"" << name << "\"");
                } else {
                    if (skip) {
                        OPENVDB_LOG_INFO("passed through grid \"" << name << "\"");
                        outGrids.push_back(baseGrid);
                    } else {
                        process(baseGrid, outGrids, opts);
                    }
                }
            }
        }
        file.close();

        openvdb::util::CpuTimer timer;
        timer.start();

        openvdb::io::File outFile(outFilename);
        if (fileMetadata) {
            outFile.write(outGrids, *fileMetadata);
        } else {
            outFile.write(outGrids);
        }

        const double msec = timer.delta(); // elapsed time

        if (outGrids.empty()) {
            OPENVDB_LOG_WARN("wrote empty file " << outFilename << " in "
                << std::setprecision(3) << (msec / 1000.0) << " sec");
        } else {
            OPENVDB_LOG_INFO("wrote file " << outFilename << " in "
                << std::setprecision(3) << (msec / 1000.0) << " sec");
        }
    }
    catch (const std::exception& e) {
        OPENVDB_LOG_FATAL(e.what());
        exitStatus = EXIT_FAILURE;
    }
    catch (...) {
        OPENVDB_LOG_FATAL("Exception caught (unexpected type)");
        std::unexpected();
    }

    return exitStatus;
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
