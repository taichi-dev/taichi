#include "timer.h"

TC_NAMESPACE_BEGIN

using namespace std;

std::map<std::string, std::pair<double, int>> Time::Timer::memo;

std::map<std::string, double> Time::FPSCounter::last_refresh;
std::map<std::string, int> Time::FPSCounter::counter;

#ifndef _WIN64
double Time::get_time() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}
double Time::get_tick() {
    return 0.0;
}
#else
#include <intrin.h>
#pragma intrinsic(__rdtsc)
double Time::get_time() {
    //https://msdn.microsoft.com/en-us/library/windows/desktop/dn553408(v=vs.85).aspx
    LARGE_INTEGER EndingTime, ElapsedMicroseconds;
    LARGE_INTEGER Frequency;

    QueryPerformanceFrequency(&Frequency);

    // Activity to be timed

    QueryPerformanceCounter(&EndingTime);
    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart;

    ElapsedMicroseconds.QuadPart *= 1000000;
    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
    return (double)ElapsedMicroseconds.QuadPart / 1000000.0;

    /*
    FILETIME tm;
    GetSystemTimeAsFileTime(&tm);
    unsigned long long t = ((ULONGLONG)tm.dwHighDateTime << 32) | (ULONGLONG)tm.dwLowDateTime;
    return (double)t / 10000000.0;
*/
}
double Time::get_tick() {
    return (double)__rdtsc();
}
#endif

void Time::usleep(double us)
{
#ifdef _WIN64
            Sleep(DWORD(us * 1e-3));
#else
            ::usleep(us);
#endif
}

double Time::Timer::get_time()
{
    return Time::get_time();
}

void Time::Timer::print_record(const char *left, double elapsed, double average)
{
    printf("%s ==> %6.2f ms ~ %6.2f ms\n", left, elapsed * 1e3,
        average * 1e3);
}

void Time::Timer::output()
{
    if (have_output) {
        return;
    }
    else {
        have_output = true;
    }
    double elapsed = get_time() - this->start_time;
    std::string left = this->name;
    if (left.size() < 60) {
        left += std::string(60 - left.size(), '-');
    }
    if (memo.find(name) == memo.end()) {
        memo.insert(make_pair(name, make_pair(0.0, 0)));
    }
    pair<double, int> memo_record = memo[name];
    memo_record.first += elapsed;
    memo_record.second += 1;
    memo[name] = memo_record;
    double avg = memo_record.first / memo_record.second;
    this->print_record(left.c_str(), elapsed, avg);
}

double Time::TickTimer::get_time()
{
    return Time::get_tick();
}

void Time::TickTimer::print_record(const char *left, double elapsed, double average)
{
    string unit;
    double measurement;
    if (elapsed < 1e3) {
        measurement = 1.0;
        unit = "cycles";
    }
    else if (elapsed < 1e6) {
        measurement = 1e3;
        unit = "K cycles";
    }
    else if (elapsed < 1e9) {
        measurement = 1e6;
        unit = "M cycles";
    }
    else {
        measurement = 1e9;
        unit = "G cycles";
    }
    printf("%s ==> %4.2f %s ~ %4.2f %s\n", left, elapsed / measurement, unit.c_str(),
        average / measurement, unit.c_str());
}


Time::Timer::Timer(std::string name) {
    this->name = name;
    this->start_time = get_time();
    this->have_output = false;
}

Time::TickTimer::TickTimer(std::string name) {
    this->name = name;
    this->start_time = get_time();
    this->have_output = false;
}


TC_NAMESPACE_END

