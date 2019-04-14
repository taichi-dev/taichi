/*
*  xxhsum - Command line interface for xxhash algorithms
*  Copyright (C) Yann Collet 2012-2016
*
*  GPL v2 License
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License along
*  with this program; if not, write to the Free Software Foundation, Inc.,
*  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*
*  You can contact the author at :
*  - xxHash homepage : http://www.xxhash.com
*  - xxHash source repository : https://github.com/Cyan4973/xxHash
*/

/* xxhsum :
 * Provides hash value of a file content, or a list of files, or stdin
 * Display convention is Big Endian, for both 32 and 64 bits algorithms
 */

#ifndef XXHASH_C_2097394837
#define XXHASH_C_2097394837

/* ************************************
 *  Compiler Options
 **************************************/
/* MS Visual */
#if defined(_MSC_VER) || defined(_WIN32)
#  define _CRT_SECURE_NO_WARNINGS   /* removes visual warnings */
#endif

/* Under Linux at least, pull in the *64 commands */
#ifndef _LARGEFILE64_SOURCE
#  define _LARGEFILE64_SOURCE
#endif

/* ************************************
 *  Includes
 **************************************/
#include <stdlib.h>     /* malloc, calloc, free, exit */
#include <stdio.h>      /* fprintf, fopen, ftello64, fread, stdin, stdout, _fileno (when present) */
#include <string.h>     /* strcmp */
#include <sys/types.h>  /* stat, stat64, _stat64 */
#include <sys/stat.h>   /* stat, stat64, _stat64 */
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <assert.h>     /* assert */
#include <errno.h>      /* errno */

#define XXH_STATIC_LINKING_ONLY   /* *_state_t */
#include "xxhash.h"


/* ************************************
 *  OS-Specific Includes
 **************************************/
#if !defined(_WIN32) && (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__)) /* UNIX-like OS */ \
   || defined(__midipix__) || defined(__VMS))
#  if (defined(__APPLE__) && defined(__MACH__)) || defined(__SVR4) || defined(_AIX) || defined(__hpux) /* POSIX.1-2001 (SUSv3) conformant */ \
     || defined(__DragonFly__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)  /* BSD distros */
#    define PLATFORM_POSIX_VERSION 200112L
#  else
#    if defined(__linux__) || defined(__linux)
#      ifndef _POSIX_C_SOURCE
#        define _POSIX_C_SOURCE 200112L  /* use feature test macro */
#      endif
#    endif
#    include <unistd.h>  /* declares _POSIX_VERSION */
#    if defined(_POSIX_VERSION)  /* POSIX compliant */
#      define PLATFORM_POSIX_VERSION _POSIX_VERSION
#    else
#      define PLATFORM_POSIX_VERSION 0
#    endif
#  endif
#endif
#if !defined(PLATFORM_POSIX_VERSION)
#  define PLATFORM_POSIX_VERSION -1
#endif

#if (defined(__linux__) && (PLATFORM_POSIX_VERSION >= 1)) \
 || (PLATFORM_POSIX_VERSION >= 200112L) \
 || defined(__DJGPP__) \
 || defined(__MSYS__)
#  include <unistd.h>   /* isatty */
#  define IS_CONSOLE(stdStream) isatty(fileno(stdStream))
#elif defined(MSDOS) || defined(OS2) || defined(__CYGWIN__)
#  include <io.h>       /* _isatty */
#  define IS_CONSOLE(stdStream) _isatty(_fileno(stdStream))
#elif defined(WIN32) || defined(_WIN32)
#  include <io.h>      /* _isatty */
#  include <windows.h> /* DeviceIoControl, HANDLE, FSCTL_SET_SPARSE */
#  include <stdio.h>   /* FILE */
static __inline int IS_CONSOLE(FILE* stdStream) {
    DWORD dummy;
    return _isatty(_fileno(stdStream)) && GetConsoleMode((HANDLE)_get_osfhandle(_fileno(stdStream)), &dummy);
}
#else
#  define IS_CONSOLE(stdStream) 0
#endif

#if defined(MSDOS) || defined(OS2) || defined(WIN32) || defined(_WIN32)
#  include <fcntl.h>   /* _O_BINARY */
#  include <io.h>      /* _setmode, _fileno, _get_osfhandle */
#  if !defined(__DJGPP__)
#    include <windows.h> /* DeviceIoControl, HANDLE, FSCTL_SET_SPARSE */
#    include <winioctl.h> /* FSCTL_SET_SPARSE */
#    define SET_BINARY_MODE(file) { int const unused=_setmode(_fileno(file), _O_BINARY); (void)unused; }
#    define SET_SPARSE_FILE_MODE(file) { DWORD dw; DeviceIoControl((HANDLE) _get_osfhandle(_fileno(file)), FSCTL_SET_SPARSE, 0, 0, 0, 0, &dw, 0); }
#  else
#    define SET_BINARY_MODE(file) setmode(fileno(file), O_BINARY)
#    define SET_SPARSE_FILE_MODE(file)
#  endif
#else
#  define SET_BINARY_MODE(file)
#  define SET_SPARSE_FILE_MODE(file)
#endif

#if !defined(S_ISREG)
#  define S_ISREG(x) (((x) & S_IFMT) == S_IFREG)
#endif


/* ************************************
*  Basic Types
**************************************/
#ifndef MEM_MODULE
# define MEM_MODULE
# if defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L   /* C99 */
#   include <stdint.h>
    typedef uint8_t  BYTE;
    typedef uint16_t U16;
    typedef uint32_t U32;
    typedef  int32_t S32;
    typedef uint64_t U64;
#  else
    typedef unsigned char      BYTE;
    typedef unsigned short     U16;
    typedef unsigned int       U32;
    typedef   signed int       S32;
    typedef unsigned long long U64;
#  endif
#endif

static unsigned BMK_isLittleEndian(void)
{
    const union { U32 u; BYTE c[4]; } one = { 1 };   /* don't use static : performance detrimental  */
    return one.c[0];
}


/* *************************************
 *  Constants
 ***************************************/
#define LIB_VERSION XXH_VERSION_MAJOR.XXH_VERSION_MINOR.XXH_VERSION_RELEASE
#define QUOTE(str) #str
#define EXPAND_AND_QUOTE(str) QUOTE(str)
#define PROGRAM_VERSION EXPAND_AND_QUOTE(LIB_VERSION)

/* Show compiler versions in WELCOME_MESSAGE. VERSION_FMT will return the printf specifiers,
 * and VERSION will contain the comma separated list of arguments to the VERSION_FMT string. */
#if defined(__clang_version__)
/* Clang does its own thing. */
#  ifdef __apple_build_version__
#    define VERSION_FMT ", Apple Clang %s"
#  else
#    define VERSION_FMT ", Clang %s"
#  endif
#  define VERSION  __clang_version__
#elif defined(__VERSION__)
/* GCC and ICC */
#  define VERSION_FMT ", %s"
#  ifdef __INTEL_COMPILER /* icc adds its prefix */
#    define VERSION_STRING __VERSION__
#  else /* assume GCC */
#    define VERSION "GCC " __VERSION__
#  endif
#elif defined(_MSC_FULL_VER) && defined(_MSC_BUILD)
/* "For example, if the version number of the Visual C++ compiler is 15.00.20706.01, the _MSC_FULL_VER macro
 * evaluates to 150020706." https://docs.microsoft.com/en-us/cpp/preprocessor/predefined-macros?view=vs-2017 */
#  define VERSION  _MSC_FULL_VER / 10000000 % 100, _MSC_FULL_VER / 100000 % 100, _MSC_FULL_VER % 100000, _MSC_BUILD
#  define VERSION_FMT ", MSVC %02i.%02i.%05i.%02i"
#elif defined(__TINYC__)
/* tcc stores its version in the __TINYC__ macro. */
#  define VERSION_FMT ", tcc %i.%i.%i"
#  define VERSION __TINYC__ / 10000 % 100, __TINYC__ / 100 % 100, __TINYC__ % 100
#else
#  define VERSION_FMT "%s"
#  define VERSION ""
#endif

/* makes the next part easier */
#if defined(__x86_64__) || defined(_M_AMD64) || defined(_M_X64)
#   define ARCH_X86 "x86_64"
#elif defined(__i386__) || defined(_M_X86) || defined(_M_X86_FP)
#   define ARCH_X86 "i386"
#endif

/* Try to detect the architecture. */
#if defined(ARCH_X86)
#  if defined(__AVX2__)
#    define ARCH ARCH_X86 " + AVX2"
#  elif defined(__AVX__)
#    define ARCH ARCH_X86 " + AVX"
#  elif defined(__SSE2__)
#     define ARCH ARCH_X86 " + SSE2"
#  else
#      define ARCH ARCH_X86
#  endif
#elif defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
#  define ARCH "aarch64"
#elif defined(__arm__) || defined(__thumb__) || defined(__thumb2__) || defined(_M_ARM)
#  if defined(__ARM_NEON) || defined(__ARM_NEON__)
#    define ARCH "arm + NEON"
#  else
#    define ARCH "arm"
#  endif
#elif defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
#  define ARCH "ppc64"
#elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__)
#  define ARCH "ppc"
#elif defined(__AVR)
#  define ARCH "AVR"
#elif defined(__mips64)
#  define ARCH "mips64"
#elif defined(__mips)
#  define ARCH "mips"
#else
#  define ARCH "unknown"
#endif

static const int g_nbBits = (int)(sizeof(void*)*8);
static const char g_lename[] = "little endian";
static const char g_bename[] = "big endian";
#define ENDIAN_NAME (BMK_isLittleEndian() ? g_lename : g_bename)
static const char author[] = "Yann Collet";
#define WELCOME_MESSAGE(exename) "%s %s (%i-bits %s %s)" VERSION_FMT ", by %s \n", \
                    exename, PROGRAM_VERSION, g_nbBits, ARCH, ENDIAN_NAME, VERSION, author

#define KB *( 1<<10)
#define MB *( 1<<20)
#define GB *(1U<<30)

static size_t XXH_DEFAULT_SAMPLE_SIZE = 100 KB;
#define NBLOOPS    3                              /* Default number of benchmark iterations */
#define TIMELOOP_S 1
#define TIMELOOP  (TIMELOOP_S * CLOCKS_PER_SEC)   /* Minimum timing per iteration */
#define XXHSUM32_DEFAULT_SEED 0                   /* Default seed for algo_xxh32 */
#define XXHSUM64_DEFAULT_SEED 0                   /* Default seed for algo_xxh64 */

#define MAX_MEM    (2 GB - 64 MB)

static const char stdinName[] = "-";
typedef enum { algo_xxh32, algo_xxh64 } algoType;
static const algoType g_defaultAlgo = algo_xxh64;    /* required within main() & usage() */

/* <16 hex char> <SPC> <SPC> <filename> <'\0'>
 * '4096' is typical Linux PATH_MAX configuration. */
#define DEFAULT_LINE_LENGTH (sizeof(XXH64_hash_t) * 2 + 2 + 4096 + 1)

/* Maximum acceptable line length. */
#define MAX_LINE_LENGTH (32 KB)


/* ************************************
 *  Display macros
 **************************************/
#define DISPLAY(...)         fprintf(stderr, __VA_ARGS__)
#define DISPLAYRESULT(...)   fprintf(stdout, __VA_ARGS__)
#define DISPLAYLEVEL(l, ...) do { if (g_displayLevel>=l) DISPLAY(__VA_ARGS__); } while (0)
static int g_displayLevel = 2;


/* ************************************
 *  Local variables
 **************************************/
static U32 g_nbIterations = NBLOOPS;


/* ************************************
 *  Benchmark Functions
 **************************************/
static clock_t BMK_clockSpan( clock_t start )
{
    return clock() - start;   /* works even if overflow; Typical max span ~ 30 mn */
}


static size_t BMK_findMaxMem(U64 requiredMem)
{
    size_t const step = 64 MB;
    void* testmem = NULL;

    requiredMem = (((requiredMem >> 26) + 1) << 26);
    requiredMem += 2*step;
    if (requiredMem > MAX_MEM) requiredMem = MAX_MEM;

    while (!testmem) {
        if (requiredMem > step) requiredMem -= step;
        else requiredMem >>= 1;
        testmem = malloc ((size_t)requiredMem);
    }
    free (testmem);

    /* keep some space available */
    if (requiredMem > step) requiredMem -= step;
    else requiredMem >>= 1;

    return (size_t)requiredMem;
}


static U64 BMK_GetFileSize(const char* infilename)
{
    int r;
#if defined(_MSC_VER)
    struct _stat64 statbuf;
    r = _stat64(infilename, &statbuf);
#else
    struct stat statbuf;
    r = stat(infilename, &statbuf);
#endif
    if (r || !S_ISREG(statbuf.st_mode)) return 0;   /* No good... */
    return (U64)statbuf.st_size;
}

typedef U32 (*hashFunction)(const void* buffer, size_t bufferSize, U32 seed);

static U32 localXXH32(const void* buffer, size_t bufferSize, U32 seed) { return XXH32(buffer, bufferSize, seed); }

static U32 localXXH64(const void* buffer, size_t bufferSize, U32 seed) { return (U32)XXH64(buffer, bufferSize, seed); }

static U32 localXXH3_64b(const void* buffer, size_t bufferSize, U32 seed) { (void)seed; return (U32)XXH3_64bits(buffer, bufferSize); }

static void BMK_benchHash(hashFunction h, const char* hName, const void* buffer, size_t bufferSize)
{
    U32 nbh_perIteration = (U32)((300 MB) / (bufferSize+1)) + 1;  /* first loop conservatively aims for 300 MB/s */
    U32 iterationNb;
    double fastestH = 100000000.;

    DISPLAYLEVEL(2, "\r%70s\r", "");       /* Clean display line */
    if (g_nbIterations<1) g_nbIterations=1;
    for (iterationNb = 1; iterationNb <= g_nbIterations; iterationNb++) {
        U32 r=0;
        clock_t cStart;

        DISPLAYLEVEL(2, "%1u-%-17.17s : %10u ->\r", iterationNb, hName, (U32)bufferSize);
        cStart = clock();
        while (clock() == cStart);   /* starts clock() at its exact beginning */
        cStart = clock();

        {   U32 u;
            for (u=0; u<nbh_perIteration; u++)
                r += h(buffer, bufferSize, u);
        }
        if (r==0) DISPLAYLEVEL(3,".\r");  /* do something with r to defeat compiler "optimizing" away hash */

        {   clock_t const nbTicks = BMK_clockSpan(cStart);
            double const timeS = ((double)nbTicks / CLOCKS_PER_SEC) / nbh_perIteration;
            if (nbTicks == 0) { /* faster than resolution timer */
                nbh_perIteration *= 100;
                iterationNb--;   /* try again */
                continue;
            }
            if (timeS < fastestH) fastestH = timeS;
            DISPLAYLEVEL(2, "%1u-%-17.17s : %10u -> %8.0f it/s (%7.1f MB/s) \r",
                    iterationNb, hName, (U32)bufferSize,
                    (double)1 / fastestH,
                    ((double)bufferSize / (1<<20)) / fastestH );
        }
        {   double nbh_perSecond = (1 / fastestH) + 1;
            if (nbh_perSecond > (double)(4000U<<20)) nbh_perSecond = (double)(4000U<<20);
            nbh_perIteration = (U32)nbh_perSecond;
        }
    }
    DISPLAYLEVEL(1, "%-19.19s : %10u -> %8.0f it/s (%7.1f MB/s) \n", hName, (U32)bufferSize,
        (double)1 / fastestH,
        ((double)bufferSize / (1<<20)) / fastestH);
    if (g_displayLevel<1)
        DISPLAYLEVEL(0, "%u, ", (U32)((double)1 / fastestH));
}


/* BMK_benchMem():
 * specificTest : 0 == run all tests, 1+ run only specific test
 * buffer : is supposed 8-bytes aligned (if malloc'ed, it should be)
 * the real allocated size of buffer is supposed to be >= (bufferSize+3).
 * @return : 0 on success, 1 if error (invalid mode selected) */
static int BMK_benchMem(const void* buffer, size_t bufferSize, U32 specificTest)
{
    assert((((size_t)buffer) & 8) == 0);  /* ensure alignment */

    /* XXH32 bench */
    if ((specificTest==0) | (specificTest==1))
        BMK_benchHash(localXXH32, "XXH32", buffer, bufferSize);

    /* Bench XXH32 on Unaligned input */
    if ((specificTest==0) | (specificTest==2))
        BMK_benchHash(localXXH32, "XXH32 unaligned", ((const char*)buffer)+1, bufferSize);

    /* Bench XXH64 */
    if ((specificTest==0) | (specificTest==3))
        BMK_benchHash(localXXH64, "XXH64", buffer, bufferSize);

    /* Bench XXH64 on Unaligned input */
    if ((specificTest==0) | (specificTest==4))
        BMK_benchHash(localXXH64, "XXH64 unaligned", ((const char*)buffer)+3, bufferSize);

    /* Bench XXH3 */
    if ((specificTest==0) | (specificTest==5))
        BMK_benchHash(localXXH3_64b, "XXH3_64bits", buffer, bufferSize);

    /* Bench XXH3 on Unaligned input */
    if ((specificTest==0) | (specificTest==6))
        BMK_benchHash(localXXH3_64b, "XXH3_64b unaligned", ((const char*)buffer)+3, bufferSize);

    if (specificTest > 6) {
        DISPLAY("Benchmark mode invalid.\n");
        return 1;
    }
    return 0;
}


static size_t BMK_selectBenchedSize(const char* fileName)
{   U64 const inFileSize = BMK_GetFileSize(fileName);
    size_t benchedSize = (size_t) BMK_findMaxMem(inFileSize);
    if ((U64)benchedSize > inFileSize) benchedSize = (size_t)inFileSize;
    if (benchedSize < inFileSize) {
        DISPLAY("Not enough memory for '%s' full size; testing %i MB only...\n", fileName, (int)(benchedSize>>20));
    }
    return benchedSize;
}


static int BMK_benchFiles(const char** fileNamesTable, int nbFiles, U32 specificTest)
{
    int result = 0;
    int fileIdx;

    for (fileIdx=0; fileIdx<nbFiles; fileIdx++) {
        const char* const inFileName = fileNamesTable[fileIdx];
        assert(inFileName != NULL);
        {
            FILE* const inFile = fopen( inFileName, "rb" );
            size_t const benchedSize = BMK_selectBenchedSize(inFileName);
            char* const buffer = (char*)calloc(benchedSize+16+3, 1);
            void* const alignedBuffer = (buffer+15) - (((size_t)(buffer+15)) & 0xF);  /* align on next 16 bytes */

            /* Checks */
            if (inFile==NULL){
                DISPLAY("Error: Could not open '%s': %s.\n", inFileName, strerror(errno));
                free(buffer);
                return 11;
            }
            if(!buffer) {
                DISPLAY("\nError: Out of memory.\n");
                fclose(inFile);
                return 12;
            }

            /* Fill input buffer */
            DISPLAYLEVEL(1, "\rLoading %s...        \n", inFileName);
            {   size_t const readSize = fread(alignedBuffer, 1, benchedSize, inFile);
                fclose(inFile);
                if(readSize != benchedSize) {
                    DISPLAY("\nError: Could not read '%s': %s.\n", inFileName, strerror(errno));
                    free(buffer);
                    return 13;
            }   }

            /* bench */
            result |= BMK_benchMem(alignedBuffer, benchedSize, specificTest);

            free(buffer);
        }
    }

    return result;
}


static int BMK_benchInternal(size_t keySize, U32 specificTest)
{
    void* const buffer = calloc(keySize+16+3, 1);
    if (!buffer) {
        DISPLAY("\nError: Out of memory.\n");
        return 12;
    }

    {   const void* const alignedBuffer = ((char*)buffer+15) - (((size_t)((char*)buffer+15)) & 0xF);  /* align on next 16 bytes */

        /* bench */
        DISPLAYLEVEL(1, "Sample of ");
        if (keySize > 10 KB) {
            DISPLAYLEVEL(1, "%u KB", (U32)(keySize >> 10));
        } else {
            DISPLAYLEVEL(1, "%u bytes", (U32)keySize);
        }
        DISPLAYLEVEL(1, "...        \n");

        {   int const result = BMK_benchMem(alignedBuffer, keySize, specificTest);
            free(buffer);
            return result;
        }
    }
}


/* ************************************************
 * Self-test :
 * ensure results consistency accross platforms
 *********************************************** */

static void BMK_checkResult32(XXH32_hash_t r1, XXH32_hash_t r2)
{
    static int nbTests = 1;
    if (r1!=r2) {
        DISPLAY("\rError: 32-bit hash test %i: Internal sanity check failed!\n", nbTests);
        DISPLAY("\rGot 0x%08X, expected 0x%08X.\n", r1, r2);
        DISPLAY("\rNote: If you modified the hash functions, make sure to either update the values\n"
                  "or temporarily comment out the tests in BMK_sanityCheck.\n");
        exit(1);
    }
    nbTests++;
}

static void BMK_checkResult64(XXH64_hash_t r1, XXH64_hash_t r2)
{
    static int nbTests = 1;
    if (r1!=r2) {
        DISPLAY("\rError: 64-bit hash test %i: Internal sanity check failed!\n", nbTests);
        DISPLAY("\rGot 0x%08X%08XULL, expected 0x%08X%08XULL.\n", (U32)(r1>>32), (U32)r1, (U32)(r2>>32), (U32)r2);
        DISPLAY("\rNote: If you modified the hash functions, make sure to either update the values\n"
                  "or temporarily comment out the tests in BMK_sanityCheck.\n");
        exit(1);
    }
    nbTests++;
}

static void BMK_checkResult128(XXH128_hash_t r1, XXH128_hash_t r2)
{
    static int nbTests = 1;
    if ((r1.low64 != r2.low64) || (r1.high64 != r2.high64)) {
        DISPLAY("\rError: 128-bit hash test %i: Internal sanity check failed.\n", nbTests);
        DISPLAY("\rGot { 0x%08X%08XULL, 0x%08X%08XULL }, expected { 0x%08X%08XULL, %08X%08XULL } \n",
                (U32)(r1.low64>>32), (U32)r1.low64, (U32)(r1.high64>>32), (U32)r1.high64,
                (U32)(r2.low64>>32), (U32)r2.low64, (U32)(r2.high64>>32), (U32)r2.high64 );
        DISPLAY("\rNote: If you modified the hash functions, make sure to either update the values\n"
                  "or temporarily comment out the tests in BMK_sanityCheck.\n");
        exit(1);
    }
    nbTests++;
}


static void BMK_testSequence64(const void* sentence, size_t len, U64 seed, U64 Nresult)
{
    XXH64_state_t state;
    U64 Dresult;
    size_t pos;

    Dresult = XXH64(sentence, len, seed);
    BMK_checkResult64(Dresult, Nresult);

    (void)XXH64_reset(&state, seed);
    (void)XXH64_update(&state, sentence, len);
    Dresult = XXH64_digest(&state);
    BMK_checkResult64(Dresult, Nresult);

    (void)XXH64_reset(&state, seed);
    for (pos=0; pos<len; pos++)
        (void)XXH64_update(&state, ((const char*)sentence)+pos, 1);
    Dresult = XXH64_digest(&state);
    BMK_checkResult64(Dresult, Nresult);
}

static void BMK_testXXH3(const void* data, size_t len, U64 seed, U64 Nresult)
{
    {   U64 const Dresult = XXH3_64bits_withSeed(data, len, seed);
        BMK_checkResult64(Dresult, Nresult);
    }

    /* check that the no-seed variant produces same result as seed==0 */
    if (seed == 0) {
        U64 const Dresult = XXH3_64bits(data, len);
        BMK_checkResult64(Dresult, Nresult);
    }
}

static void BMK_testXXH128(const void* data, size_t len, U64 seed, XXH128_hash_t Nresult)
{
    {   XXH128_hash_t const Dresult = XXH3_128bits_withSeed(data, len, seed);
        BMK_checkResult128(Dresult, Nresult);

        /* check that XXH128() is identical to XXH3_128bits_withSeed() */
        {   XXH128_hash_t const Dresult2 = XXH128(data, len, seed);
            BMK_checkResult128(Dresult2, Nresult);
        }

        /* check that first field is equal to _64bits variant */
        {   U64 const result64 = XXH3_64bits_withSeed(data, len, seed);
            BMK_checkResult64(result64, Nresult.low64);
    }   }

    /* check that the no-seed variant produces same result as seed==0 */
    if (seed == 0) {
        XXH128_hash_t const Dresult = XXH3_128bits(data, len);
        BMK_checkResult128(Dresult, Nresult);
    }
}

static void BMK_testSequence(const void* sequence, size_t len, U32 seed, U32 Nresult)
{
    XXH32_state_t state;
    U32 Dresult;
    size_t pos;

    Dresult = XXH32(sequence, len, seed);
    BMK_checkResult32(Dresult, Nresult);

    (void)XXH32_reset(&state, seed);
    (void)XXH32_update(&state, sequence, len);
    Dresult = XXH32_digest(&state);
    BMK_checkResult32(Dresult, Nresult);

    (void)XXH32_reset(&state, seed);
    for (pos=0; pos<len; pos++)
        (void)XXH32_update(&state, ((const char*)sequence)+pos, 1);
    Dresult = XXH32_digest(&state);
    BMK_checkResult32(Dresult, Nresult);
}


#define SANITY_BUFFER_SIZE 2243
static void BMK_sanityCheck(void)
{
    static const U32 prime = 2654435761U;
    BYTE sanityBuffer[SANITY_BUFFER_SIZE];
    U32 byteGen = prime;

    int i;
    for (i=0; i<SANITY_BUFFER_SIZE; i++) {
        sanityBuffer[i] = (BYTE)(byteGen>>24);
        byteGen *= byteGen;
    }

    BMK_testSequence(NULL,          0, 0,     0x02CC5D05);
    BMK_testSequence(NULL,          0, prime, 0x36B78AE7);
    BMK_testSequence(sanityBuffer,  1, 0,     0xB85CBEE5);
    BMK_testSequence(sanityBuffer,  1, prime, 0xD5845D64);
    BMK_testSequence(sanityBuffer, 14, 0,     0xE5AA0AB4);
    BMK_testSequence(sanityBuffer, 14, prime, 0x4481951D);
    BMK_testSequence(sanityBuffer,222, 0,     0xC8070816);
    BMK_testSequence(sanityBuffer,222, prime, 0xF3CFC852);

    BMK_testSequence64(NULL        ,  0, 0,     0xEF46DB3751D8E999ULL);
    BMK_testSequence64(NULL        ,  0, prime, 0xAC75FDA2929B17EFULL);
    BMK_testSequence64(sanityBuffer,  1, 0,     0x4FCE394CC88952D8ULL);
    BMK_testSequence64(sanityBuffer,  1, prime, 0x739840CB819FA723ULL);
    BMK_testSequence64(sanityBuffer, 14, 0,     0xCFFA8DB881BC3A3DULL);
    BMK_testSequence64(sanityBuffer, 14, prime, 0x5B9611585EFCC9CBULL);
    BMK_testSequence64(sanityBuffer,222, 0,     0x9DD507880DEBB03DULL);
    BMK_testSequence64(sanityBuffer,222, prime, 0xDC515172B8EE0600ULL);

    BMK_testXXH3(NULL,           0, 0,     0);                      /* zero-length hash is the seed == 0 by default */
    BMK_testXXH3(NULL,           0, prime, prime);
    BMK_testXXH3(sanityBuffer,   1, 0,     0xE2C6D3B40D6F9203ULL);  /*  1 -  3 */
    BMK_testXXH3(sanityBuffer,   1, prime, 0xCEE5DF124E6135DCULL);  /*  1 -  3 */
    BMK_testXXH3(sanityBuffer,   6, 0,     0x585D6F8D1AAD96A2ULL);  /*  4 -  8 */
    BMK_testXXH3(sanityBuffer,   6, prime, 0x133EC8CA1739250FULL);  /*  4 -  8 */
    BMK_testXXH3(sanityBuffer,  12, 0,     0x0E85E122FE5356ACULL);  /*  9 - 16 */
    BMK_testXXH3(sanityBuffer,  12, prime, 0xE0DB5E70DA67EB16ULL);  /*  9 - 16 */
    BMK_testXXH3(sanityBuffer,  24, 0,     0x6C213B15B89230C9ULL);  /* 17 - 32 */
    BMK_testXXH3(sanityBuffer,  24, prime, 0x71892DB847A8F53CULL);  /* 17 - 32 */
    BMK_testXXH3(sanityBuffer,  48, 0,     0xECED834E8E99DA1EULL);  /* 33 - 64 */
    BMK_testXXH3(sanityBuffer,  48, prime, 0xA901250B336F9133ULL);  /* 33 - 64 */
    BMK_testXXH3(sanityBuffer,  80, 0,     0xC67B3A9C6D69E022ULL);  /* 65 - 96 */
    BMK_testXXH3(sanityBuffer,  80, prime, 0x5054F266D6A65EE4ULL);  /* 65 - 96 */
    BMK_testXXH3(sanityBuffer, 112, 0,     0x84B99B2137A264A5ULL);  /* 97 -128 */
    BMK_testXXH3(sanityBuffer, 112, prime, 0xD6BF88A668E69F2AULL);  /* 97 -128 */
    BMK_testXXH3(sanityBuffer, 192, 0,     0x6D96AC3F415CFCFEULL);  /* one block, finishing at stripe boundary */
    BMK_testXXH3(sanityBuffer, 192, prime, 0xE4BD30AA1673B966ULL);  /* one block, finishing at stripe boundary */
    BMK_testXXH3(sanityBuffer, 222, 0,     0xB62929C362EF3BF5ULL);  /* one block, last stripe is overlapping */
    BMK_testXXH3(sanityBuffer, 222, prime, 0x2782C3C49E3FD25EULL);  /* one block, last stripe is overlapping */
    BMK_testXXH3(sanityBuffer,2048, 0,     0x802EB54C97564FD7ULL);  /* 2 blocks, finishing at block boundary */
    BMK_testXXH3(sanityBuffer,2048, prime, 0xC9F188CFAFDA22CDULL);  /* 2 blocks, finishing at block boundary */
    BMK_testXXH3(sanityBuffer,2240, 0,     0x16B0035F6ABC1F46ULL);  /* 3 blocks, finishing at stripe boundary */
    BMK_testXXH3(sanityBuffer,2240, prime, 0x389E68C2348B9161ULL);  /* 3 blocks, finishing at stripe boundary */
    BMK_testXXH3(sanityBuffer,2243, 0,     0xE7C1890BDBD2B245ULL);  /* 3 blocks, last stripe is overlapping */
    BMK_testXXH3(sanityBuffer,2243, prime, 0x3A68386AED0C50A7ULL);  /* 3 blocks, last stripe is overlapping */

    {   XXH128_hash_t const expected = { 0, 0 };
        BMK_testXXH128(NULL,           0, 0,     expected);         /* zero-length hash is { seed, -seed } by default */
    }
    {   XXH128_hash_t const expected = { prime, -(U64)prime };
        BMK_testXXH128(NULL,           0, prime, expected);
    }
    {   XXH128_hash_t const expected = { 0xE2C6D3B40D6F9203ULL, 0x82895983D246CA74ULL };
        BMK_testXXH128(sanityBuffer,   1, 0,     expected);         /* 1-3 */
    }
    {   XXH128_hash_t const expected = { 0xCEE5DF124E6135DCULL, 0xFA2DA0269396F32DULL };
        BMK_testXXH128(sanityBuffer,   1, prime, expected);         /* 1-3 */
    }
    {   XXH128_hash_t const expected = { 0x585D6F8D1AAD96A2ULL, 0x2791F3B193F0AB86ULL };
        BMK_testXXH128(sanityBuffer,   6, 0,     expected);         /* 4-8 */
    }
    {   XXH128_hash_t const expected = { 0x133EC8CA1739250FULL, 0xDF3F422D70BDE07FULL };
        BMK_testXXH128(sanityBuffer,   6, prime, expected);         /* 4-8 */
    }
    {   XXH128_hash_t const expected = { 0x0E85E122FE5356ACULL, 0xD933CC7EDF4D95DAULL };
        BMK_testXXH128(sanityBuffer,  12, 0,     expected);         /* 9-16 */
    }
    {   XXH128_hash_t const expected = { 0xE0DB5E70DA67EB16ULL, 0x114C8C76E74C669FULL };
        BMK_testXXH128(sanityBuffer,  12, prime, expected);         /* 9-16 */
    }
    {   XXH128_hash_t const expected = { 0x6C213B15B89230C9ULL, 0x3F3AACF5F277AC02ULL };
        BMK_testXXH128(sanityBuffer,  24, 0,     expected);         /* 17-32 */
    }
    {   XXH128_hash_t const expected = { 0x71892DB847A8F53CULL, 0xD11561AC7D0F5ECDULL };
        BMK_testXXH128(sanityBuffer,  24, prime, expected);         /* 17-32 */
    }
    {   XXH128_hash_t const expected = { 0xECED834E8E99DA1EULL, 0x0F85E76A60898313ULL };
        BMK_testXXH128(sanityBuffer,  48, 0,     expected);         /* 33-64 */
    }
    {   XXH128_hash_t const expected = { 0xA901250B336F9133ULL, 0xA35D3FB395E1DDE0ULL };
        BMK_testXXH128(sanityBuffer,  48, prime, expected);         /* 33-64 */
    }
    {   XXH128_hash_t const expected = { 0x338B2F6E103D5B4EULL, 0x5DD1777C8FA671ABULL };
        BMK_testXXH128(sanityBuffer,  81, 0,     expected);         /* 65-96 */
    }
    {   XXH128_hash_t const expected = { 0x0718382B6D4264C3ULL, 0x1D542DAFEFA1790EULL };
        BMK_testXXH128(sanityBuffer,  81, prime, expected);         /* 65-96 */
    }
    {   XXH128_hash_t const expected = { 0x7DE871A4FE41C90EULL, 0x786CB41C46C6B7B6ULL };
        BMK_testXXH128(sanityBuffer, 103, 0,     expected);         /* 97-128 */
    }
    {   XXH128_hash_t const expected = { 0xAD8B0B428C940A2CULL, 0xF8BA6D8B8CB05EB7ULL };
        BMK_testXXH128(sanityBuffer, 103, prime, expected);         /* 97-128 */
    }
    {   XXH128_hash_t const expected = { 0x6D96AC3F415CFCFEULL, 0x947EDFA54DD68990ULL };
        BMK_testXXH128(sanityBuffer, 192, 0,     expected);         /* one block, ends at stripe boundary */
    }
    {   XXH128_hash_t const expected = { 0xE4BD30AA1673B966ULL, 0x8132EF45FF3D51F2ULL };
        BMK_testXXH128(sanityBuffer, 192, prime, expected);         /* one block, ends at stripe boundary */
    }
    {   XXH128_hash_t const expected = { 0xB62929C362EF3BF5ULL, 0x1946A7A9E6DD3032ULL };
        BMK_testXXH128(sanityBuffer, 222, 0,     expected);         /* one block, last stripe is overlapping */
    }
    {   XXH128_hash_t const expected = { 0x2782C3C49E3FD25EULL, 0x98CE16C40C2D59F6ULL };
        BMK_testXXH128(sanityBuffer, 222, prime, expected);         /* one block, last stripe is overlapping */
    }
    {   XXH128_hash_t const expected = { 0x802EB54C97564FD7ULL, 0x384AADF242348D00ULL };
        BMK_testXXH128(sanityBuffer,2048, 0,     expected);         /* two blocks, finishing at block boundary */
    }
    {   XXH128_hash_t const expected = { 0xC9F188CFAFDA22CDULL, 0x7936B69445BE9EEDULL };
        BMK_testXXH128(sanityBuffer,2048, prime, expected);         /* two blocks, finishing at block boundary */
    }
    {   XXH128_hash_t const expected = { 0x16B0035F6ABC1F46ULL, 0x1F6602850A1AA7EEULL };
        BMK_testXXH128(sanityBuffer,2240, 0,     expected);         /* two blocks, ends at stripe boundary */
    }
    {   XXH128_hash_t const expected = { 0x389E68C2348B9161ULL, 0xA7D1E8C96586A052ULL };
        BMK_testXXH128(sanityBuffer,2240, prime, expected);         /* two blocks, ends at stripe boundary */
    }
    {   XXH128_hash_t const expected = { 0x8B1DE79158C397D3ULL, 0x9B6B2EEFAC2DE0ADULL };
        BMK_testXXH128(sanityBuffer,2237, 0,     expected);         /* two blocks, ends at stripe boundary */
    }
    {   XXH128_hash_t const expected = { 0x9DDF09ABA2B93DD6ULL, 0xB9CEDBE2582CA371ULL };
        BMK_testXXH128(sanityBuffer,2237, prime, expected);         /* two blocks, ends at stripe boundary */
    }


    DISPLAYLEVEL(3, "\r%70s\r", "");       /* Clean display line */
    DISPLAYLEVEL(3, "Sanity check -- all tests ok\n");
}


/* ********************************************************
*  File Hashing
**********************************************************/

static void BMK_display_LittleEndian(const void* ptr, size_t length)
{
    const BYTE* p = (const BYTE*)ptr;
    size_t idx;
    for (idx=length-1; idx<length; idx--)    /* intentional underflow to negative to detect end */
        DISPLAYRESULT("%02x", p[idx]);
}

static void BMK_display_BigEndian(const void* ptr, size_t length)
{
    const BYTE* p = (const BYTE*)ptr;
    size_t idx;
    for (idx=0; idx<length; idx++)
        DISPLAYRESULT("%02x", p[idx]);
}

static void BMK_hashStream(void* xxhHashValue, const algoType hashType, FILE* inFile, void* buffer, size_t blockSize)
{
    XXH64_state_t state64;
    XXH32_state_t state32;
    size_t readSize;

    /* Init */
    (void)XXH32_reset(&state32, XXHSUM32_DEFAULT_SEED);
    (void)XXH64_reset(&state64, XXHSUM64_DEFAULT_SEED);

    /* Load file & update hash */
    readSize = 1;
    while (readSize) {
        readSize = fread(buffer, 1, blockSize, inFile);
        switch(hashType)
        {
        case algo_xxh32:
            (void)XXH32_update(&state32, buffer, readSize);
            break;
        case algo_xxh64:
            (void)XXH64_update(&state64, buffer, readSize);
            break;
        default:
            break;
        }
    }

    switch(hashType)
    {
    case algo_xxh32:
        {   U32 const h32 = XXH32_digest(&state32);
            memcpy(xxhHashValue, &h32, sizeof(h32));
            break;
        }
    case algo_xxh64:
        {   U64 const h64 = XXH64_digest(&state64);
            memcpy(xxhHashValue, &h64, sizeof(h64));
            break;
        }
    default:
            break;
    }
}


typedef enum { big_endian, little_endian} endianess;

static int BMK_hash(const char* fileName,
                    const algoType hashType,
                    const endianess displayEndianess)
{
    FILE*  inFile;
    size_t const blockSize = 64 KB;
    void*  buffer;
    U32    h32 = 0;
    U64    h64 = 0;

    /* Check file existence */
    if (fileName == stdinName) {
        inFile = stdin;
        fileName = "stdin";
        SET_BINARY_MODE(stdin);
    }
    else
        inFile = fopen( fileName, "rb" );
    if (inFile==NULL) {
        DISPLAY("Error: Could not open '%s': %s.\n", fileName, strerror(errno));
        return 1;
    }

    /* Memory allocation & restrictions */
    buffer = malloc(blockSize);
    if(!buffer) {
        DISPLAY("\nError: Out of memory.\n");
        fclose(inFile);
        return 1;
    }

    /* loading notification */
    {   const size_t fileNameSize = strlen(fileName);
        const char* const fileNameEnd = fileName + fileNameSize;
        const int maxInfoFilenameSize = (int)(fileNameSize > 30 ? 30 : fileNameSize);
        int infoFilenameSize = 1;
        while ((infoFilenameSize < maxInfoFilenameSize)
            && (fileNameEnd[-1-infoFilenameSize] != '/')
            && (fileNameEnd[-1-infoFilenameSize] != '\\') )
              infoFilenameSize++;
        DISPLAY("\rLoading %s...  \r", fileNameEnd - infoFilenameSize);

        /* Load file & update hash */
        switch(hashType)
        {
        case algo_xxh32:
            BMK_hashStream(&h32, algo_xxh32, inFile, buffer, blockSize);
            break;
        case algo_xxh64:
            BMK_hashStream(&h64, algo_xxh64, inFile, buffer, blockSize);
            break;
        default:
            break;
        }

        fclose(inFile);
        free(buffer);
        DISPLAY("%s             \r", fileNameEnd - infoFilenameSize);  /* erase line */
    }

    /* display Hash */
    switch(hashType)
    {
    case algo_xxh32:
        {   XXH32_canonical_t hcbe32;
            (void)XXH32_canonicalFromHash(&hcbe32, h32);
            displayEndianess==big_endian ?
                BMK_display_BigEndian(&hcbe32, sizeof(hcbe32)) : BMK_display_LittleEndian(&hcbe32, sizeof(hcbe32));
            DISPLAYRESULT("  %s\n", fileName);
            break;
        }
    case algo_xxh64:
        {   XXH64_canonical_t hcbe64;
            (void)XXH64_canonicalFromHash(&hcbe64, h64);
            displayEndianess==big_endian ?
                BMK_display_BigEndian(&hcbe64, sizeof(hcbe64)) : BMK_display_LittleEndian(&hcbe64, sizeof(hcbe64));
            DISPLAYRESULT("  %s\n", fileName);
            break;
        }
    default:
            break;
    }

    return 0;
}


static int BMK_hashFiles(const char** fnList, int fnTotal,
                         algoType hashType, endianess displayEndianess)
{
    int fnNb;
    int result = 0;

    if (fnTotal==0)
        return BMK_hash(stdinName, hashType, displayEndianess);

    for (fnNb=0; fnNb<fnTotal; fnNb++)
        result += BMK_hash(fnList[fnNb], hashType, displayEndianess);
    DISPLAY("\r%70s\r", "");
    return result;
}


typedef enum {
    GetLine_ok,
    GetLine_eof,
    GetLine_exceedMaxLineLength,
    GetLine_outOfMemory,
} GetLineResult;

typedef enum {
    CanonicalFromString_ok,
    CanonicalFromString_invalidFormat,
} CanonicalFromStringResult;

typedef enum {
    ParseLine_ok,
    ParseLine_invalidFormat,
} ParseLineResult;

typedef enum {
    LineStatus_hashOk,
    LineStatus_hashFailed,
    LineStatus_failedToOpen,
} LineStatus;

typedef union {
    XXH32_canonical_t xxh32;
    XXH64_canonical_t xxh64;
} Canonical;

typedef struct {
    Canonical   canonical;
    const char* filename;
    int         xxhBits;    /* canonical type : 32:xxh32, 64:xxh64 */
} ParsedLine;

typedef struct {
    unsigned long   nProperlyFormattedLines;
    unsigned long   nImproperlyFormattedLines;
    unsigned long   nMismatchedChecksums;
    unsigned long   nOpenOrReadFailures;
    unsigned long   nMixedFormatLines;
    int             xxhBits;
    int             quit;
} ParseFileReport;

typedef struct {
    const char*     inFileName;
    FILE*           inFile;
    int             lineMax;
    char*           lineBuf;
    size_t          blockSize;
    char*           blockBuf;
    U32             strictMode;
    U32             statusOnly;
    U32             warn;
    U32             quiet;
    ParseFileReport report;
} ParseFileArg;


/*  Read line from stream.
    Returns GetLine_ok, if it reads line successfully.
    Returns GetLine_eof, if stream reaches EOF.
    Returns GetLine_exceedMaxLineLength, if line length is longer than MAX_LINE_LENGTH.
    Returns GetLine_outOfMemory, if line buffer memory allocation failed.
 */
static GetLineResult getLine(char** lineBuf, int* lineMax, FILE* inFile)
{
    GetLineResult result = GetLine_ok;
    size_t len = 0;

    if ((*lineBuf == NULL) || (*lineMax<1)) {
        free(*lineBuf);  /* in case it's != NULL */
        *lineMax = 0;
        *lineBuf = (char*)malloc(DEFAULT_LINE_LENGTH);
        if(*lineBuf == NULL) return GetLine_outOfMemory;
        *lineMax = DEFAULT_LINE_LENGTH;
    }

    for (;;) {
        const int c = fgetc(inFile);
        if (c == EOF) {
            /* If we meet EOF before first character, returns GetLine_eof,
             * otherwise GetLine_ok.
             */
            if (len == 0) result = GetLine_eof;
            break;
        }

        /* Make enough space for len+1 (for final NUL) bytes. */
        if (len+1 >= (size_t)*lineMax) {
            char* newLineBuf = NULL;
            size_t newBufSize = (size_t)*lineMax;

            newBufSize += (newBufSize/2) + 1; /* x 1.5 */
            if (newBufSize > MAX_LINE_LENGTH) newBufSize = MAX_LINE_LENGTH;
            if (len+1 >= newBufSize) return GetLine_exceedMaxLineLength;

            newLineBuf = (char*) realloc(*lineBuf, newBufSize);
            if (newLineBuf == NULL) return GetLine_outOfMemory;

            *lineBuf = newLineBuf;
            *lineMax = (int)newBufSize;
        }

        if (c == '\n') break;
        (*lineBuf)[len++] = (char) c;
    }

    (*lineBuf)[len] = '\0';
    return result;
}


/*  Converts one hexadecimal character to integer.
 *  Returns -1, if given character is not hexadecimal.
 */
static int charToHex(char c)
{
    int result = -1;
    if (c >= '0' && c <= '9') {
        result = (int) (c - '0');
    } else if (c >= 'A' && c <= 'F') {
        result = (int) (c - 'A') + 0x0a;
    } else if (c >= 'a' && c <= 'f') {
        result = (int) (c - 'a') + 0x0a;
    }
    return result;
}


/*  Converts XXH32 canonical hexadecimal string hashStr to big endian unsigned char array dst.
 *  Returns CANONICAL_FROM_STRING_INVALID_FORMAT, if hashStr is not well formatted.
 *  Returns CANONICAL_FROM_STRING_OK, if hashStr is parsed successfully.
 */
static CanonicalFromStringResult canonicalFromString(unsigned char* dst,
                                                     size_t dstSize,
                                                     const char* hashStr)
{
    size_t i;
    for (i = 0; i < dstSize; ++i) {
        int h0, h1;

        h0 = charToHex(hashStr[i*2 + 0]);
        if (h0 < 0) return CanonicalFromString_invalidFormat;

        h1 = charToHex(hashStr[i*2 + 1]);
        if (h1 < 0) return CanonicalFromString_invalidFormat;

        dst[i] = (unsigned char) ((h0 << 4) | h1);
    }
    return CanonicalFromString_ok;
}


/*  Parse single line of xxHash checksum file.
 *  Returns PARSE_LINE_ERROR_INVALID_FORMAT, if line is not well formatted.
 *  Returns PARSE_LINE_OK if line is parsed successfully.
 *  And members of parseLine will be filled by parsed values.
 *
 *  - line must be ended with '\0'.
 *  - Since parsedLine.filename will point within given argument `line`,
 *    users must keep `line`s content during they are using parsedLine.
 *
 *  Given xxHash checksum line should have the following format:
 *
 *      <8 or 16 hexadecimal char> <space> <space> <filename...> <'\0'>
 */
static ParseLineResult parseLine(ParsedLine* parsedLine, const char* line)
{
    const char* const firstSpace = strchr(line, ' ');
    if (firstSpace == NULL) return ParseLine_invalidFormat;

    {   const char* const secondSpace = firstSpace + 1;
        if (*secondSpace != ' ') return ParseLine_invalidFormat;

        parsedLine->filename = NULL;
        parsedLine->xxhBits = 0;

        switch (firstSpace - line)
        {
        case 8:
            {   XXH32_canonical_t* xxh32c = &parsedLine->canonical.xxh32;
                if (canonicalFromString(xxh32c->digest, sizeof(xxh32c->digest), line)
                    != CanonicalFromString_ok) {
                    return ParseLine_invalidFormat;
                }
                parsedLine->xxhBits = 32;
                break;
            }

        case 16:
            {   XXH64_canonical_t* xxh64c = &parsedLine->canonical.xxh64;
                if (canonicalFromString(xxh64c->digest, sizeof(xxh64c->digest), line)
                    != CanonicalFromString_ok) {
                    return ParseLine_invalidFormat;
                }
                parsedLine->xxhBits = 64;
                break;
            }

        default:
                return ParseLine_invalidFormat;
                break;
        }

        parsedLine->filename = secondSpace + 1;
    }
    return ParseLine_ok;
}


/*!  Parse xxHash checksum file.
 */
static void parseFile1(ParseFileArg* parseFileArg)
{
    const char* const inFileName = parseFileArg->inFileName;
    ParseFileReport* const report = &parseFileArg->report;

    unsigned long lineNumber = 0;
    memset(report, 0, sizeof(*report));

    while (!report->quit) {
        FILE* fp = NULL;
        LineStatus lineStatus = LineStatus_hashFailed;
        GetLineResult getLineResult;
        ParsedLine parsedLine;
        memset(&parsedLine, 0, sizeof(parsedLine));

        lineNumber++;
        if (lineNumber == 0) {
            /* This is unlikely happen, but md5sum.c has this
             * error check. */
            DISPLAY("%s: Error: Too many checksum lines\n", inFileName);
            report->quit = 1;
            break;
        }

        getLineResult = getLine(&parseFileArg->lineBuf, &parseFileArg->lineMax,
                                parseFileArg->inFile);
        if (getLineResult != GetLine_ok) {
            if (getLineResult == GetLine_eof) break;

            switch (getLineResult)
            {
            case GetLine_ok:
            case GetLine_eof:
                /* These cases never happen.  See above getLineResult related "if"s.
                   They exist just for make gcc's -Wswitch-enum happy. */
                break;

            default:
                DISPLAY("%s:%lu: Error: Unknown error.\n", inFileName, lineNumber);
                break;

            case GetLine_exceedMaxLineLength:
                DISPLAY("%s:%lu: Error: Line too long.\n", inFileName, lineNumber);
                break;

            case GetLine_outOfMemory:
                DISPLAY("%s:%lu: Error: Out of memory.\n", inFileName, lineNumber);
                break;
            }
            report->quit = 1;
            break;
        }

        if (parseLine(&parsedLine, parseFileArg->lineBuf) != ParseLine_ok) {
            report->nImproperlyFormattedLines++;
            if (parseFileArg->warn) {
                DISPLAY("%s:%lu: Error: Improperly formatted checksum line.\n"
                    , inFileName, lineNumber);
            }
            continue;
        }

        if (report->xxhBits != 0 && report->xxhBits != parsedLine.xxhBits) {
            /* Don't accept xxh32/xxh64 mixed file */
            report->nImproperlyFormattedLines++;
            report->nMixedFormatLines++;
            if (parseFileArg->warn) {
                DISPLAY("%s : %lu: Error: Multiple hash types in one file.\n"
                    , inFileName, lineNumber);
            }
            continue;
        }

        report->nProperlyFormattedLines++;
        if (report->xxhBits == 0) {
            report->xxhBits = parsedLine.xxhBits;
        }

        fp = fopen(parsedLine.filename, "rb");
        if (fp == NULL) {
            lineStatus = LineStatus_failedToOpen;
        } else {
            lineStatus = LineStatus_hashFailed;
            switch (parsedLine.xxhBits)
            {
            case 32:
                {   XXH32_hash_t xxh;
                    BMK_hashStream(&xxh, algo_xxh32, fp, parseFileArg->blockBuf, parseFileArg->blockSize);
                    if (xxh == XXH32_hashFromCanonical(&parsedLine.canonical.xxh32)) {
                        lineStatus = LineStatus_hashOk;
                }   }
                break;

            case 64:
                {   XXH64_hash_t xxh;
                    BMK_hashStream(&xxh, algo_xxh64, fp, parseFileArg->blockBuf, parseFileArg->blockSize);
                    if (xxh == XXH64_hashFromCanonical(&parsedLine.canonical.xxh64)) {
                        lineStatus = LineStatus_hashOk;
                }   }
                break;

            default:
                break;
            }
            fclose(fp);
        }

        switch (lineStatus)
        {
        default:
            DISPLAY("%s: Error: Unknown error.\n", inFileName);
            report->quit = 1;
            break;

        case LineStatus_failedToOpen:
            report->nOpenOrReadFailures++;
            if (!parseFileArg->statusOnly) {
                DISPLAYRESULT("%s:%lu: Could not open or read '%s': %s.\n",
                    inFileName, lineNumber, parsedLine.filename, strerror(errno));
            }
            break;

        case LineStatus_hashOk:
        case LineStatus_hashFailed:
            {   int b = 1;
                if (lineStatus == LineStatus_hashOk) {
                    /* If --quiet is specified, don't display "OK" */
                    if (parseFileArg->quiet) b = 0;
                } else {
                    report->nMismatchedChecksums++;
                }

                if (b && !parseFileArg->statusOnly) {
                    DISPLAYRESULT("%s: %s\n", parsedLine.filename
                        , lineStatus == LineStatus_hashOk ? "OK" : "FAILED");
            }   }
            break;
        }
    }   /* while (!report->quit) */
}


/*  Parse xxHash checksum file.
 *  Returns 1, if all procedures were succeeded.
 *  Returns 0, if any procedures was failed.
 *
 *  If strictMode != 0, return error code if any line is invalid.
 *  If statusOnly != 0, don't generate any output.
 *  If warn != 0, print a warning message to stderr.
 *  If quiet != 0, suppress "OK" line.
 *
 *  "All procedures are succeeded" means:
 *    - Checksum file contains at least one line and less than SIZE_T_MAX lines.
 *    - All files are properly opened and read.
 *    - All hash values match with its content.
 *    - (strict mode) All lines in checksum file are consistent and well formatted.
 *
 */
static int checkFile(const char* inFileName,
                     const endianess displayEndianess,
                     U32 strictMode,
                     U32 statusOnly,
                     U32 warn,
                     U32 quiet)
{
    int result = 0;
    FILE* inFile = NULL;
    ParseFileArg parseFileArgBody;
    ParseFileArg* const parseFileArg = &parseFileArgBody;
    ParseFileReport* const report = &parseFileArg->report;

    if (displayEndianess != big_endian) {
        /* Don't accept little endian */
        DISPLAY( "Check file mode doesn't support little endian\n" );
        return 0;
    }

    /* note : stdinName is special constant pointer.  It is not a string. */
    if (inFileName == stdinName) {
        /* note : Since we expect text input for xxhash -c mode,
         * Don't set binary mode for stdin */
        inFileName = "stdin";
        inFile = stdin;
    } else {
        inFile = fopen( inFileName, "rt" );
    }

    if (inFile == NULL) {
        DISPLAY("Error: Could not open '%s': %s\n", inFileName, strerror(errno));
        return 0;
    }

    parseFileArg->inFileName    = inFileName;
    parseFileArg->inFile        = inFile;
    parseFileArg->lineMax       = DEFAULT_LINE_LENGTH;
    parseFileArg->lineBuf       = (char*) malloc((size_t) parseFileArg->lineMax);
    parseFileArg->blockSize     = 64 * 1024;
    parseFileArg->blockBuf      = (char*) malloc(parseFileArg->blockSize);
    parseFileArg->strictMode    = strictMode;
    parseFileArg->statusOnly    = statusOnly;
    parseFileArg->warn          = warn;
    parseFileArg->quiet         = quiet;

    parseFile1(parseFileArg);

    free(parseFileArg->blockBuf);
    free(parseFileArg->lineBuf);

    if (inFile != stdin) fclose(inFile);

    /* Show error/warning messages.  All messages are copied from md5sum.c
     */
    if (report->nProperlyFormattedLines == 0) {
        DISPLAY("%s: no properly formatted xxHash checksum lines found\n", inFileName);
    } else if (!statusOnly) {
        if (report->nImproperlyFormattedLines) {
            DISPLAYRESULT("%lu %s are improperly formatted\n"
                , report->nImproperlyFormattedLines
                , report->nImproperlyFormattedLines == 1 ? "line" : "lines");
        }
        if (report->nOpenOrReadFailures) {
            DISPLAYRESULT("%lu listed %s could not be read\n"
                , report->nOpenOrReadFailures
                , report->nOpenOrReadFailures == 1 ? "file" : "files");
        }
        if (report->nMismatchedChecksums) {
            DISPLAYRESULT("%lu computed %s did NOT match\n"
                , report->nMismatchedChecksums
                , report->nMismatchedChecksums == 1 ? "checksum" : "checksums");
    }   }

    /* Result (exit) code logic is copied from
     * gnu coreutils/src/md5sum.c digest_check() */
    result =   report->nProperlyFormattedLines != 0
            && report->nMismatchedChecksums == 0
            && report->nOpenOrReadFailures == 0
            && (!strictMode || report->nImproperlyFormattedLines == 0)
            && report->quit == 0;
    return result;
}


static int checkFiles(const char** fnList, int fnTotal,
                      const endianess displayEndianess,
                      U32 strictMode,
                      U32 statusOnly,
                      U32 warn,
                      U32 quiet)
{
    int ok = 1;

    /* Special case for stdinName "-",
     * note: stdinName is not a string.  It's special pointer. */
    if (fnTotal==0) {
        ok &= checkFile(stdinName, displayEndianess, strictMode, statusOnly, warn, quiet);
    } else {
        int fnNb;
        for (fnNb=0; fnNb<fnTotal; fnNb++)
            ok &= checkFile(fnList[fnNb], displayEndianess, strictMode, statusOnly, warn, quiet);
    }
    return ok ? 0 : 1;
}


/* ********************************************************
*  Main
**********************************************************/

static int usage(const char* exename)
{
    DISPLAY( WELCOME_MESSAGE(exename) );
    DISPLAY( "Usage :\n");
    DISPLAY( "      %s [arg] [filenames]\n", exename);
    DISPLAY( "When no filename provided, or - provided : use stdin as input\n");
    DISPLAY( "Arguments :\n");
    DISPLAY( " -H# : hash selection : 0=32bits, 1=64bits (default: %i)\n", (int)g_defaultAlgo);
    DISPLAY( " -c  : read xxHash sums from the [filenames] and check them\n");
    DISPLAY( " -h  : help \n");
    return 0;
}


static int usage_advanced(const char* exename)
{
    usage(exename);
    DISPLAY( "Advanced :\n");
    DISPLAY( " --little-endian : hash printed using little endian convention (default: big endian)\n");
    DISPLAY( " -V, --version   : display version\n");
    DISPLAY( " -h, --help      : display long help and exit\n");
    DISPLAY( " -b  : benchmark mode \n");
    DISPLAY( " -i# : number of iterations (benchmark mode; default %u)\n", g_nbIterations);
    DISPLAY( "\n");
    DISPLAY( "The following four options are useful only when verifying checksums (-c):\n");
    DISPLAY( "--strict : don't print OK for each successfully verified file\n");
    DISPLAY( "--status : don't output anything, status code shows success\n");
    DISPLAY( "--quiet  : exit non-zero for improperly formatted checksum lines\n");
    DISPLAY( "--warn   : warn about improperly formatted checksum lines\n");
    return 0;
}

static int badusage(const char* exename)
{
    DISPLAY("Wrong parameters\n");
    usage(exename);
    return 1;
}

static void errorOut(const char* msg)
{
    DISPLAY("%s \n", msg); exit(1);
}

/*! readU32FromCharChecked() :
 * @return 0 if success, and store the result in *value.
 *  allows and interprets K, KB, KiB, M, MB and MiB suffix.
 *  Will also modify `*stringPtr`, advancing it to position where it stopped reading.
 * @return 1 if an overflow error occurs */
static int readU32FromCharChecked(const char** stringPtr, unsigned* value)
{
    static unsigned const max = (((unsigned)(-1)) / 10) - 1;
    unsigned result = 0;
    while ((**stringPtr >='0') && (**stringPtr <='9')) {
        if (result > max) return 1; /* overflow error */
        result *= 10;
        result += (unsigned)(**stringPtr - '0');
        (*stringPtr)++ ;
    }
    if ((**stringPtr=='K') || (**stringPtr=='M')) {
        unsigned const maxK = ((unsigned)(-1)) >> 10;
        if (result > maxK) return 1; /* overflow error */
        result <<= 10;
        if (**stringPtr=='M') {
            if (result > maxK) return 1; /* overflow error */
            result <<= 10;
        }
        (*stringPtr)++;  /* skip `K` or `M` */
        if (**stringPtr=='i') (*stringPtr)++;
        if (**stringPtr=='B') (*stringPtr)++;
    }
    *value = result;
    return 0;
}

/*! readU32FromChar() :
 * @return : unsigned integer value read from input in `char` format.
 *  allows and interprets K, KB, KiB, M, MB and MiB suffix.
 *  Will also modify `*stringPtr`, advancing it to position where it stopped reading.
 *  Note : function will exit() program if digit sequence overflows */
static unsigned readU32FromChar(const char** stringPtr) {
    unsigned result;
    if (readU32FromCharChecked(stringPtr, &result)) {
        static const char errorMsg[] = "Error: numeric value too large";
        errorOut(errorMsg);
    }
    return result;
}

int main(int argc, const char** argv)
{
    int i, filenamesStart = 0;
    const char* const exename = argv[0];
    U32 benchmarkMode = 0;
    U32 fileCheckMode = 0;
    U32 strictMode    = 0;
    U32 statusOnly    = 0;
    U32 warn          = 0;
    U32 quiet         = 0;
    U32 specificTest  = 0;
    size_t keySize    = XXH_DEFAULT_SAMPLE_SIZE;
    algoType algo     = g_defaultAlgo;
    endianess displayEndianess = big_endian;

    /* special case : xxh32sum default to 32 bits checksum */
    if (strstr(exename, "xxh32sum") != NULL) algo = algo_xxh32;

    for(i=1; i<argc; i++) {
        const char* argument = argv[i];

        if(!argument) continue;   /* Protection, if argument empty */

        if (!strcmp(argument, "--little-endian")) { displayEndianess = little_endian; continue; }
        if (!strcmp(argument, "--check")) { fileCheckMode = 1; continue; }
        if (!strcmp(argument, "--strict")) { strictMode = 1; continue; }
        if (!strcmp(argument, "--status")) { statusOnly = 1; continue; }
        if (!strcmp(argument, "--quiet")) { quiet = 1; continue; }
        if (!strcmp(argument, "--warn")) { warn = 1; continue; }
        if (!strcmp(argument, "--help")) { return usage_advanced(exename); }
        if (!strcmp(argument, "--version")) { DISPLAY(WELCOME_MESSAGE(exename)); return 0; }

        if (*argument!='-') {
            if (filenamesStart==0) filenamesStart=i;   /* only supports a continuous list of filenames */
            continue;
        }

        /* command selection */
        argument++;   /* note : *argument=='-' */

        while (*argument!=0) {
            switch(*argument)
            {
            /* Display version */
            case 'V':
                DISPLAY(WELCOME_MESSAGE(exename)); return 0;

            /* Display help on usage */
            case 'h':
                return usage_advanced(exename);

            /* select hash algorithm */
            case 'H':
                algo = (algoType)(argument[1] - '0');
                argument+=2;
                break;

            /* File check mode */
            case 'c':
                fileCheckMode=1;
                argument++;
                break;

            /* Warning mode (file check mode only, alias of "--warning") */
            case 'w':
                warn=1;
                argument++;
                break;

            /* Trigger benchmark mode */
            case 'b':
                argument++;
                benchmarkMode = 1;
                specificTest = readU32FromChar(&argument);   /* select one specific test (hidden option) */
                break;

            /* Modify Nb Iterations (benchmark only) */
            case 'i':
                argument++;
                g_nbIterations = readU32FromChar(&argument);
                break;

            /* Modify Block size (benchmark only) */
            case 'B':
                argument++;
                keySize = readU32FromChar(&argument);
                break;

            /* Modify verbosity of benchmark output (hidden option) */
            case 'q':
                argument++;
                g_displayLevel--;
                break;

            default:
                return badusage(exename);
            }
        }
    }   /* for(i=1; i<argc; i++) */

    /* Check benchmark mode */
    if (benchmarkMode) {
        DISPLAYLEVEL(2, WELCOME_MESSAGE(exename) );
        BMK_sanityCheck();
        if (filenamesStart==0) return BMK_benchInternal(keySize, specificTest);
        return BMK_benchFiles(argv+filenamesStart, argc-filenamesStart, specificTest);
    }

    /* Check if input is defined as console; trigger an error in this case */
    if ( (filenamesStart==0) && IS_CONSOLE(stdin) ) return badusage(exename);

    if (filenamesStart==0) filenamesStart = argc;
    if (fileCheckMode) {
        return checkFiles(argv+filenamesStart, argc-filenamesStart,
                          displayEndianess, strictMode, statusOnly, warn, quiet);
    } else {
        return BMK_hashFiles(argv+filenamesStart, argc-filenamesStart, algo, displayEndianess);
    }
}

#endif /* XXHASH_C_2097394837 */
