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
#include <openvdb/util/CpuTimer.h>
#include <openvdb/util/logging.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/TestFailure.h>
#include <cppunit/TestListener.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TextTestProgressListener.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <algorithm> // for std::shuffle()
#include <cmath> // for std::round()
#include <cstdlib> // for EXIT_SUCCESS
#include <cstring> // for strrchr()
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>


namespace {

using StringVec = std::vector<std::string>;


void
usage(const char* progName, std::ostream& ostrm)
{
    ostrm <<
"Usage: " << progName << " [options]\n" <<
"Which: runs OpenVDB library unit tests\n" <<
"Options:\n" <<
"    -f file   read whitespace-separated names of tests to be run\n" <<
"              from the given file (\"#\" comments are supported)\n" <<
"    -l        list all available tests\n" <<
"    -shuffle  run tests in random order\n" <<
"    -t test   specific suite or test to run, e.g., \"-t TestGrid\"\n" <<
"              or \"-t TestGrid::testGetGrid\" (default: run all tests)\n" <<
"    -v        verbose output\n";
#ifdef OPENVDB_USE_LOG4CPLUS
    ostrm <<
"\n" <<
"    -error    log fatal and non-fatal errors (default: log only fatal errors)\n" <<
"    -warn     log warnings and errors\n" <<
"    -info     log info messages, warnings and errors\n" <<
"    -debug    log debugging messages, info messages, warnings and errors\n";
#endif
}


void
getTestNames(StringVec& nameVec, const CppUnit::Test* test)
{
    if (test) {
        const int numChildren = test->getChildTestCount();
        if (numChildren == 0) {
            nameVec.push_back(test->getName());
        } else {
            for (int i = 0; i < test->getChildTestCount(); ++i) {
                getTestNames(nameVec, test->getChildTestAt(i));
            }
        }
    }
}


/// Listener that prints the name, elapsed time, and error status of each test
class TimedTestProgressListener: public CppUnit::TestListener
{
public:
    void startTest(CppUnit::Test* test) override
    {
        mFailed = false;
        std::cout << test->getName() << std::flush;
        mTimer.start();
    }

    void addFailure(const CppUnit::TestFailure& failure) override
    {
        std::cout << " : " << (failure.isError() ? "error" : "assertion");
        mFailed  = true;
    }

    void endTest(CppUnit::Test*) override
    {
        if (!mFailed) {
            // Print elapsed time only for successful tests.
            const auto msec = int(std::round(mTimer.delta()));
            std::cout << " : OK (";
            if (msec > 0) {
                std::cout << msec;
            } else {
                std::cout << "<1";
            }
            std::cout << "ms)";
        }
        std::cout << std::endl;
    }

private:
    openvdb::util::CpuTimer mTimer;
    bool mFailed = false;
};


int
run(int argc, char* argv[])
{
    const char* progName = argv[0];
    if (const char* ptr = ::strrchr(progName, '/')) progName = ptr + 1;

    bool shuffle = false, verbose = false;
    StringVec tests;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-l") {
            StringVec allTests;
            getTestNames(allTests,
                CppUnit::TestFactoryRegistry::getRegistry().makeTest());
            for (const auto& name: allTests) { std::cout << name << "\n"; }
            return EXIT_SUCCESS;
        } else if (arg == "-shuffle") {
            shuffle = true;
        } else if (arg == "-v") {
            verbose = true;
        } else if (arg == "-t") {
            if (i + 1 < argc) {
                ++i;
                tests.push_back(argv[i]);
            } else {
                OPENVDB_LOG_FATAL("missing test name after \"-t\"");
                usage(progName, std::cerr);
                return EXIT_FAILURE;
            }
        } else if (arg == "-f") {
            if (i + 1 < argc) {
                ++i;
                std::ifstream file{argv[i]};
                if (file.fail()) {
                    OPENVDB_LOG_FATAL("unable to read file " << argv[i]);
                    return EXIT_FAILURE;
                }
                while (file) {
                    // Read a whitespace-separated string from the file.
                    std::string test;
                    file >> test;
                    if (!test.empty()) {
                        if (test[0] != '#') {
                            tests.push_back(test);
                        } else {
                            // If the string starts with a comment symbol ("#"),
                            // skip it and jump to the end of the line.
                            while (file) { if (file.get() == '\n') break; }
                        }
                    }
                }
            } else {
                OPENVDB_LOG_FATAL("missing filename after \"-f\"");
                usage(progName, std::cerr);
                return EXIT_FAILURE;
            }
        } else if (arg == "-h" || arg == "-help" || arg == "--help") {
            usage(progName, std::cout);
            return EXIT_SUCCESS;
        } else {
            OPENVDB_LOG_FATAL("unrecognized option \"" << arg << "\"");
            usage(progName, std::cerr);
            return EXIT_FAILURE;
        }
    }

    try {
        CppUnit::TestFactoryRegistry& registry =
            CppUnit::TestFactoryRegistry::getRegistry();

        auto* root = registry.makeTest();
        if (!root) {
            throw std::runtime_error(
                "CppUnit test registry was not initialized properly");
        }

        if (!shuffle) {
            if (tests.empty()) tests.push_back("");
        } else {
            // Get the names of all selected tests and their children.
            StringVec allTests;
            if (tests.empty()) {
                getTestNames(allTests, root);
            } else {
                for (const auto& name: tests) {
                    getTestNames(allTests, root->findTest(name));
                }
            }
            // Randomly shuffle the list of names.
            std::random_device randDev;
            std::mt19937 generator(randDev());
            std::shuffle(allTests.begin(), allTests.end(), generator);
            tests.swap(allTests);
        }

        CppUnit::TestRunner runner;
        runner.addTest(root);

        CppUnit::TestResult controller;

        CppUnit::TestResultCollector result;
        controller.addListener(&result);

        CppUnit::TextTestProgressListener progress;
        TimedTestProgressListener vProgress;
        if (verbose) {
            controller.addListener(&vProgress);
        } else {
            controller.addListener(&progress);
        }

        for (size_t i = 0; i < tests.size(); ++i) {
            runner.run(controller, tests[i]);
        }

        CppUnit::CompilerOutputter outputter(&result, std::cerr);
        outputter.write();

        return result.wasSuccessful() ? EXIT_SUCCESS : EXIT_FAILURE;

    } catch (std::exception& e) {
        OPENVDB_LOG_FATAL(e.what());
        return EXIT_FAILURE;
    }
}

} // anonymous namespace


int
main(int argc, char *argv[])
{
    openvdb::logging::initialize(argc, argv);

    return run(argc, argv);
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
