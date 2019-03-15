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

#ifndef OPENVDB_UTIL_LOGGING_HAS_BEEN_INCLUDED
#define OPENVDB_UTIL_LOGGING_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

#ifdef OPENVDB_USE_LOG4CPLUS

#include <log4cplus/appender.h>
#include <log4cplus/configurator.h>
#include <log4cplus/consoleappender.h>
#include <log4cplus/layout.h>
#include <log4cplus/logger.h>
#include <log4cplus/spi/loggingevent.h>
#include <algorithm> // for std::remove()
#include <cstring> // for ::strrchr()
#include <memory>
#include <sstream>
#include <string>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace logging {

/// @brief Message severity level
enum class Level {
    Debug = log4cplus::DEBUG_LOG_LEVEL,
    Info =  log4cplus::INFO_LOG_LEVEL,
    Warn =  log4cplus::WARN_LOG_LEVEL,
    Error = log4cplus::ERROR_LOG_LEVEL,
    Fatal = log4cplus::FATAL_LOG_LEVEL
};


namespace internal {

/// @brief log4cplus layout that outputs text in different colors
/// for different log levels, using ANSI escape codes
class ColoredPatternLayout: public log4cplus::PatternLayout
{
public:
    explicit ColoredPatternLayout(const std::string& progName_, bool useColor = true)
        : log4cplus::PatternLayout(
            progName_.empty() ? std::string{"%5p: %m%n"} : (progName_ + " %5p: %m%n"))
        , mUseColor(useColor)
        , mProgName(progName_)
    {
    }

    ~ColoredPatternLayout() override {}

    const std::string& progName() const { return mProgName; }

    void formatAndAppend(log4cplus::tostream& strm,
        const log4cplus::spi::InternalLoggingEvent& event) override
    {
        if (!mUseColor) {
            log4cplus::PatternLayout::formatAndAppend(strm, event);
            return;
        }
        log4cplus::tostringstream s;
        switch (event.getLogLevel()) {
            case log4cplus::DEBUG_LOG_LEVEL: s << "\033[32m"; break; // green
            case log4cplus::ERROR_LOG_LEVEL:
            case log4cplus::FATAL_LOG_LEVEL: s << "\033[31m"; break; // red
            case log4cplus::INFO_LOG_LEVEL:  s << "\033[36m"; break; // cyan
            case log4cplus::WARN_LOG_LEVEL:  s << "\033[35m"; break; // magenta
        }
        log4cplus::PatternLayout::formatAndAppend(s, event);
        strm << s.str() << "\033[0m" << std::flush;
    }

// Disable deprecation warnings for std::auto_ptr.
#if defined(__ICC)
  #pragma warning push
  #pragma warning disable:1478
#elif defined(__clang__)
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#if defined(LOG4CPLUS_VERSION) && defined(LOG4CPLUS_MAKE_VERSION)
  #if LOG4CPLUS_VERSION >= LOG4CPLUS_MAKE_VERSION(2, 0, 0)
    // In log4cplus 2.0.0, std::auto_ptr was replaced with std::unique_ptr.
    using Ptr = std::unique_ptr<log4cplus::Layout>;
  #else
    using Ptr = std::auto_ptr<log4cplus::Layout>;
  #endif
#else
    using Ptr = std::auto_ptr<log4cplus::Layout>;
#endif

    static Ptr create(const std::string& progName_, bool useColor = true)
    {
        return Ptr{new ColoredPatternLayout{progName_, useColor}};
    }

#if defined(__ICC)
  #pragma warning pop
#elif defined(__clang__)
  #pragma clang diagnostic pop
#elif defined(__GNUC__)
  #pragma GCC diagnostic pop
#endif

private:
    bool mUseColor = true;
    std::string mProgName;
}; // class ColoredPatternLayout


inline log4cplus::Logger
getLogger()
{
    return log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("openvdb"));
}


inline log4cplus::SharedAppenderPtr
getAppender()
{
    return getLogger().getAppender(LOG4CPLUS_TEXT("OPENVDB"));
}

} // namespace internal


/// @brief Return the current logging level.
inline Level
getLevel()
{
    switch (internal::getLogger().getLogLevel()) {
        case log4cplus::DEBUG_LOG_LEVEL: return Level::Debug;
        case log4cplus::INFO_LOG_LEVEL:  return Level::Info;
        case log4cplus::WARN_LOG_LEVEL:  return Level::Warn;
        case log4cplus::ERROR_LOG_LEVEL: return Level::Error;
        case log4cplus::FATAL_LOG_LEVEL: break;
    }
    return Level::Fatal;
}


/// @brief Set the logging level.  (Lower-level messages will be suppressed.)
inline void
setLevel(Level lvl)
{
    internal::getLogger().setLogLevel(static_cast<log4cplus::LogLevel>(lvl));
}


/// @brief If "-debug", "-info", "-warn", "-error" or "-fatal" is found
/// in the given array of command-line arguments, set the logging level
/// appropriately and remove the relevant argument(s) from the array.
inline void
setLevel(int& argc, char* argv[])
{
    for (int i = 1; i < argc; ++i) { // note: skip argv[0]
        const std::string arg{argv[i]};
        bool remove = true;
        if (arg == "-debug")      { setLevel(Level::Debug); }
        else if (arg == "-error") { setLevel(Level::Error); }
        else if (arg == "-fatal") { setLevel(Level::Fatal); }
        else if (arg == "-info")  { setLevel(Level::Info); }
        else if (arg == "-warn")  { setLevel(Level::Warn); }
        else { remove = false; }
        if (remove) argv[i] = nullptr;
    }
    auto end = std::remove(argv + 1, argv + argc, nullptr);
    argc = static_cast<int>(end - argv);
}


/// @brief Specify a program name to be displayed in log messages.
inline void
setProgramName(const std::string& progName, bool useColor = true)
{
    // Change the layout of the OpenVDB appender to use colored text
    // and to incorporate the supplied program name.
    if (auto appender = internal::getAppender()) {
        appender->setLayout(internal::ColoredPatternLayout::create(progName, useColor));
    }
}


/// @brief Initialize the logging system if it is not already initialized.
inline void
initialize(bool useColor = true)
{
    log4cplus::initialize();

    if (internal::getAppender()) return; // already initialized

    // Create the OpenVDB logger if it doesn't already exist.
    auto logger = internal::getLogger();

    // Disable "additivity", so that OpenVDB-related messages are directed
    // to the OpenVDB logger only and are not forwarded up the logger tree.
    logger.setAdditivity(false);

    // Attach a console appender to the OpenVDB logger.
    if (auto appender = log4cplus::SharedAppenderPtr{new log4cplus::ConsoleAppender}) {
        appender->setName(LOG4CPLUS_TEXT("OPENVDB"));
        logger.addAppender(appender);
    }

    setLevel(Level::Warn);
    setProgramName("", useColor);
}


/// @brief Initialize the logging system from command-line arguments.
/// @details If "-debug", "-info", "-warn", "-error" or "-fatal" is found
/// in the given array of command-line arguments, set the logging level
/// appropriately and remove the relevant argument(s) from the array.
inline void
initialize(int& argc, char* argv[], bool useColor = true)
{
    initialize();

    setLevel(argc, argv);

    auto progName = (argc > 0 ? argv[0] : "");
    if (const char* ptr = ::strrchr(progName, '/')) progName = ptr + 1;
    setProgramName(progName, useColor);
}

} // namespace logging
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#define OPENVDB_LOG(level, message) \
    do { \
        auto _log = openvdb::logging::internal::getLogger(); \
        if (_log.isEnabledFor(log4cplus::level##_LOG_LEVEL)) { \
            std::ostringstream _buf; \
            _buf << message; \
            _log.forcedLog(log4cplus::level##_LOG_LEVEL, _buf.str(), __FILE__, __LINE__); \
        } \
    } while (0);

/// Log an info message of the form '<TT>someVar << "some text" << ...</TT>'.
#define OPENVDB_LOG_INFO(message)           OPENVDB_LOG(INFO, message)
/// Log a warning message of the form '<TT>someVar << "some text" << ...</TT>'.
#define OPENVDB_LOG_WARN(message)           OPENVDB_LOG(WARN, message)
/// Log an error message of the form '<TT>someVar << "some text" << ...</TT>'.
#define OPENVDB_LOG_ERROR(message)          OPENVDB_LOG(ERROR, message)
/// Log a fatal error message of the form '<TT>someVar << "some text" << ...</TT>'.
#define OPENVDB_LOG_FATAL(message)          OPENVDB_LOG(FATAL, message)
#ifdef DEBUG
/// In debug builds only, log a debugging message of the form '<TT>someVar << "text" << ...</TT>'.
#define OPENVDB_LOG_DEBUG(message)          OPENVDB_LOG(DEBUG, message)
#else
/// In debug builds only, log a debugging message of the form '<TT>someVar << "text" << ...</TT>'.
#define OPENVDB_LOG_DEBUG(message)
#endif
/// @brief Log a debugging message in both debug and optimized builds.
/// @warning Don't use this in performance-critical code.
#define OPENVDB_LOG_DEBUG_RUNTIME(message)  OPENVDB_LOG(DEBUG, message)

#else // ifdef OPENVDB_USE_LOG4CPLUS

#include <iostream>

#define OPENVDB_LOG_INFO(mesg)
#define OPENVDB_LOG_WARN(mesg)      do { std::cerr << "WARNING: " << mesg << std::endl; } while (0);
#define OPENVDB_LOG_ERROR(mesg)     do { std::cerr << "ERROR: " << mesg << std::endl; } while (0);
#define OPENVDB_LOG_FATAL(mesg)     do { std::cerr << "FATAL: " << mesg << std::endl; } while (0);
#define OPENVDB_LOG_DEBUG(mesg)
#define OPENVDB_LOG_DEBUG_RUNTIME(mesg)

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace logging {

enum class Level { Debug, Info, Warn, Error, Fatal };

inline Level getLevel() { return Level::Warn; }
inline void setLevel(Level) {}
inline void setLevel(int&, char*[]) {}
inline void setProgramName(const std::string&, bool = true) {}
inline void initialize() {}
inline void initialize(int&, char*[], bool = true) {}

} // namespace logging
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_USE_LOG4CPLUS


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace logging {

/// @brief A LevelScope object sets the logging level to a given level
/// and restores it to the current level when the object goes out of scope.
struct LevelScope
{
    Level level;
    explicit LevelScope(Level newLevel): level(getLevel()) { setLevel(newLevel); }
    ~LevelScope() { setLevel(level); }
};

} // namespace logging
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_UTIL_LOGGING_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
