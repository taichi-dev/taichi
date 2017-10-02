/*
  md5.hpp is a reformulation of the md5.h and md5.c code from
  http://www.opensource.apple.com/source/cups/cups-59/cups/md5.c to allow it to
  function as a component of a header only library. This conversion was done by
  Peter Thorson (webmaster@zaphoyd.com) in 2012 for the WebSocket++ project. The
  changes are released under the same license as the original (listed below)
*/
/*
  Copyright (C) 1999, 2002 Aladdin Enterprises.  All rights reserved.

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  L. Peter Deutsch
  ghost@aladdin.com

 */
/* $Id: md5.h,v 1.4 2002/04/13 19:20:28 lpd Exp $ */
/*
  Independent implementation of MD5 (RFC 1321).

  This code implements the MD5 Algorithm defined in RFC 1321, whose
  text is available at
    http://www.ietf.org/rfc/rfc1321.txt
  The code is derived from the text of the RFC, including the test suite
  (section A.5) but excluding the rest of Appendix A.  It does not include
  any code or documentation that is identified in the RFC as being
  copyrighted.

  The original and principal author of md5.h is L. Peter Deutsch
  <ghost@aladdin.com>.  Other authors are noted in the change history
  that follows (in reverse chronological order):

  2002-04-13 lpd Removed support for non-ANSI compilers; removed
    references to Ghostscript; clarified derivation from RFC 1321;
    now handles byte order either statically or dynamically.
  1999-11-04 lpd Edited comments slightly for automatic TOC extraction.
  1999-10-18 lpd Fixed typo in header comment (ansi2knr rather than md5);
    added conditionalization for C++ compilation from Martin
    Purschke <purschke@bnl.gov>.
  1999-05-03 lpd Original version.
 */

#ifndef WEBSOCKETPP_COMMON_MD5_HPP
#define WEBSOCKETPP_COMMON_MD5_HPP

/*
 * This package supports both compile-time and run-time determination of CPU
 * byte order.  If ARCH_IS_BIG_ENDIAN is defined as 0, the code will be
 * compiled to run only on little-endian CPUs; if ARCH_IS_BIG_ENDIAN is
 * defined as non-zero, the code will be compiled to run only on big-endian
 * CPUs; if ARCH_IS_BIG_ENDIAN is not defined, the code will be compiled to
 * run on either big- or little-endian CPUs, but will run slightly less
 * efficiently on either one than if ARCH_IS_BIG_ENDIAN is defined.
 */

#include <stddef.h>
#include <string>
#include <cstring>

namespace websocketpp {
/// Provides MD5 hashing functionality
namespace md5 {

typedef unsigned char md5_byte_t; /* 8-bit byte */
typedef unsigned int md5_word_t; /* 32-bit word */

/* Define the state of the MD5 Algorithm. */
typedef struct md5_state_s {
    md5_word_t count[2];    /* message length in bits, lsw first */
    md5_word_t abcd[4];     /* digest buffer */
    md5_byte_t buf[64];     /* accumulate block */
} md5_state_t;

/* Initialize the algorithm. */
inline void md5_init(md5_state_t *pms);

/* Append a string to the message. */
inline void md5_append(md5_state_t *pms, md5_byte_t const * data, size_t nbytes);

/* Finish the message and return the digest. */
inline void md5_finish(md5_state_t *pms, md5_byte_t digest[16]);

#undef ZSW_MD5_BYTE_ORDER   /* 1 = big-endian, -1 = little-endian, 0 = unknown */
#ifdef ARCH_IS_BIG_ENDIAN
#  define ZSW_MD5_BYTE_ORDER (ARCH_IS_BIG_ENDIAN ? 1 : -1)
#else
#  define ZSW_MD5_BYTE_ORDER 0
#endif

#define ZSW_MD5_T_MASK ((md5_word_t)~0)
#define ZSW_MD5_T1 /* 0xd76aa478 */ (ZSW_MD5_T_MASK ^ 0x28955b87)
#define ZSW_MD5_T2 /* 0xe8c7b756 */ (ZSW_MD5_T_MASK ^ 0x173848a9)
#define ZSW_MD5_T3    0x242070db
#define ZSW_MD5_T4 /* 0xc1bdceee */ (ZSW_MD5_T_MASK ^ 0x3e423111)
#define ZSW_MD5_T5 /* 0xf57c0faf */ (ZSW_MD5_T_MASK ^ 0x0a83f050)
#define ZSW_MD5_T6    0x4787c62a
#define ZSW_MD5_T7 /* 0xa8304613 */ (ZSW_MD5_T_MASK ^ 0x57cfb9ec)
#define ZSW_MD5_T8 /* 0xfd469501 */ (ZSW_MD5_T_MASK ^ 0x02b96afe)
#define ZSW_MD5_T9    0x698098d8
#define ZSW_MD5_T10 /* 0x8b44f7af */ (ZSW_MD5_T_MASK ^ 0x74bb0850)
#define ZSW_MD5_T11 /* 0xffff5bb1 */ (ZSW_MD5_T_MASK ^ 0x0000a44e)
#define ZSW_MD5_T12 /* 0x895cd7be */ (ZSW_MD5_T_MASK ^ 0x76a32841)
#define ZSW_MD5_T13    0x6b901122
#define ZSW_MD5_T14 /* 0xfd987193 */ (ZSW_MD5_T_MASK ^ 0x02678e6c)
#define ZSW_MD5_T15 /* 0xa679438e */ (ZSW_MD5_T_MASK ^ 0x5986bc71)
#define ZSW_MD5_T16    0x49b40821
#define ZSW_MD5_T17 /* 0xf61e2562 */ (ZSW_MD5_T_MASK ^ 0x09e1da9d)
#define ZSW_MD5_T18 /* 0xc040b340 */ (ZSW_MD5_T_MASK ^ 0x3fbf4cbf)
#define ZSW_MD5_T19    0x265e5a51
#define ZSW_MD5_T20 /* 0xe9b6c7aa */ (ZSW_MD5_T_MASK ^ 0x16493855)
#define ZSW_MD5_T21 /* 0xd62f105d */ (ZSW_MD5_T_MASK ^ 0x29d0efa2)
#define ZSW_MD5_T22    0x02441453
#define ZSW_MD5_T23 /* 0xd8a1e681 */ (ZSW_MD5_T_MASK ^ 0x275e197e)
#define ZSW_MD5_T24 /* 0xe7d3fbc8 */ (ZSW_MD5_T_MASK ^ 0x182c0437)
#define ZSW_MD5_T25    0x21e1cde6
#define ZSW_MD5_T26 /* 0xc33707d6 */ (ZSW_MD5_T_MASK ^ 0x3cc8f829)
#define ZSW_MD5_T27 /* 0xf4d50d87 */ (ZSW_MD5_T_MASK ^ 0x0b2af278)
#define ZSW_MD5_T28    0x455a14ed
#define ZSW_MD5_T29 /* 0xa9e3e905 */ (ZSW_MD5_T_MASK ^ 0x561c16fa)
#define ZSW_MD5_T30 /* 0xfcefa3f8 */ (ZSW_MD5_T_MASK ^ 0x03105c07)
#define ZSW_MD5_T31    0x676f02d9
#define ZSW_MD5_T32 /* 0x8d2a4c8a */ (ZSW_MD5_T_MASK ^ 0x72d5b375)
#define ZSW_MD5_T33 /* 0xfffa3942 */ (ZSW_MD5_T_MASK ^ 0x0005c6bd)
#define ZSW_MD5_T34 /* 0x8771f681 */ (ZSW_MD5_T_MASK ^ 0x788e097e)
#define ZSW_MD5_T35    0x6d9d6122
#define ZSW_MD5_T36 /* 0xfde5380c */ (ZSW_MD5_T_MASK ^ 0x021ac7f3)
#define ZSW_MD5_T37 /* 0xa4beea44 */ (ZSW_MD5_T_MASK ^ 0x5b4115bb)
#define ZSW_MD5_T38    0x4bdecfa9
#define ZSW_MD5_T39 /* 0xf6bb4b60 */ (ZSW_MD5_T_MASK ^ 0x0944b49f)
#define ZSW_MD5_T40 /* 0xbebfbc70 */ (ZSW_MD5_T_MASK ^ 0x4140438f)
#define ZSW_MD5_T41    0x289b7ec6
#define ZSW_MD5_T42 /* 0xeaa127fa */ (ZSW_MD5_T_MASK ^ 0x155ed805)
#define ZSW_MD5_T43 /* 0xd4ef3085 */ (ZSW_MD5_T_MASK ^ 0x2b10cf7a)
#define ZSW_MD5_T44    0x04881d05
#define ZSW_MD5_T45 /* 0xd9d4d039 */ (ZSW_MD5_T_MASK ^ 0x262b2fc6)
#define ZSW_MD5_T46 /* 0xe6db99e5 */ (ZSW_MD5_T_MASK ^ 0x1924661a)
#define ZSW_MD5_T47    0x1fa27cf8
#define ZSW_MD5_T48 /* 0xc4ac5665 */ (ZSW_MD5_T_MASK ^ 0x3b53a99a)
#define ZSW_MD5_T49 /* 0xf4292244 */ (ZSW_MD5_T_MASK ^ 0x0bd6ddbb)
#define ZSW_MD5_T50    0x432aff97
#define ZSW_MD5_T51 /* 0xab9423a7 */ (ZSW_MD5_T_MASK ^ 0x546bdc58)
#define ZSW_MD5_T52 /* 0xfc93a039 */ (ZSW_MD5_T_MASK ^ 0x036c5fc6)
#define ZSW_MD5_T53    0x655b59c3
#define ZSW_MD5_T54 /* 0x8f0ccc92 */ (ZSW_MD5_T_MASK ^ 0x70f3336d)
#define ZSW_MD5_T55 /* 0xffeff47d */ (ZSW_MD5_T_MASK ^ 0x00100b82)
#define ZSW_MD5_T56 /* 0x85845dd1 */ (ZSW_MD5_T_MASK ^ 0x7a7ba22e)
#define ZSW_MD5_T57    0x6fa87e4f
#define ZSW_MD5_T58 /* 0xfe2ce6e0 */ (ZSW_MD5_T_MASK ^ 0x01d3191f)
#define ZSW_MD5_T59 /* 0xa3014314 */ (ZSW_MD5_T_MASK ^ 0x5cfebceb)
#define ZSW_MD5_T60    0x4e0811a1
#define ZSW_MD5_T61 /* 0xf7537e82 */ (ZSW_MD5_T_MASK ^ 0x08ac817d)
#define ZSW_MD5_T62 /* 0xbd3af235 */ (ZSW_MD5_T_MASK ^ 0x42c50dca)
#define ZSW_MD5_T63    0x2ad7d2bb
#define ZSW_MD5_T64 /* 0xeb86d391 */ (ZSW_MD5_T_MASK ^ 0x14792c6e)

static void md5_process(md5_state_t *pms, md5_byte_t const * data /*[64]*/) {
    md5_word_t
    a = pms->abcd[0], b = pms->abcd[1],
    c = pms->abcd[2], d = pms->abcd[3];
    md5_word_t t;
#if ZSW_MD5_BYTE_ORDER > 0
    /* Define storage only for big-endian CPUs. */
    md5_word_t X[16];
#else
    /* Define storage for little-endian or both types of CPUs. */
    md5_word_t xbuf[16];
    md5_word_t const * X;
#endif

    {
#if ZSW_MD5_BYTE_ORDER == 0
    /*
     * Determine dynamically whether this is a big-endian or
     * little-endian machine, since we can use a more efficient
     * algorithm on the latter.
     */
    static int const w = 1;

    if (*((md5_byte_t const *)&w)) /* dynamic little-endian */
#endif
#if ZSW_MD5_BYTE_ORDER <= 0     /* little-endian */
    {
        /*
         * On little-endian machines, we can process properly aligned
         * data without copying it.
         */
        if (!((data - (md5_byte_t const *)0) & 3)) {
        /* data are properly aligned */
        X = (md5_word_t const *)data;
        } else {
        /* not aligned */
        std::memcpy(xbuf, data, 64);
        X = xbuf;
        }
    }
#endif
#if ZSW_MD5_BYTE_ORDER == 0
    else            /* dynamic big-endian */
#endif
#if ZSW_MD5_BYTE_ORDER >= 0     /* big-endian */
    {
        /*
         * On big-endian machines, we must arrange the bytes in the
         * right order.
         */
        const md5_byte_t *xp = data;
        int i;

#  if ZSW_MD5_BYTE_ORDER == 0
        X = xbuf;       /* (dynamic only) */
#  else
#    define xbuf X      /* (static only) */
#  endif
        for (i = 0; i < 16; ++i, xp += 4)
        xbuf[i] = xp[0] + (xp[1] << 8) + (xp[2] << 16) + (xp[3] << 24);
    }
#endif
    }

#define ZSW_MD5_ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

    /* Round 1. */
    /* Let [abcd k s i] denote the operation
       a = b + ((a + F(b,c,d) + X[k] + T[i]) <<< s). */
#define ZSW_MD5_F(x, y, z) (((x) & (y)) | (~(x) & (z)))
#define SET(a, b, c, d, k, s, Ti)\
  t = a + ZSW_MD5_F(b,c,d) + X[k] + Ti;\
  a = ZSW_MD5_ROTATE_LEFT(t, s) + b
    /* Do the following 16 operations. */
    SET(a, b, c, d,  0,  7,  ZSW_MD5_T1);
    SET(d, a, b, c,  1, 12,  ZSW_MD5_T2);
    SET(c, d, a, b,  2, 17,  ZSW_MD5_T3);
    SET(b, c, d, a,  3, 22,  ZSW_MD5_T4);
    SET(a, b, c, d,  4,  7,  ZSW_MD5_T5);
    SET(d, a, b, c,  5, 12,  ZSW_MD5_T6);
    SET(c, d, a, b,  6, 17,  ZSW_MD5_T7);
    SET(b, c, d, a,  7, 22,  ZSW_MD5_T8);
    SET(a, b, c, d,  8,  7,  ZSW_MD5_T9);
    SET(d, a, b, c,  9, 12, ZSW_MD5_T10);
    SET(c, d, a, b, 10, 17, ZSW_MD5_T11);
    SET(b, c, d, a, 11, 22, ZSW_MD5_T12);
    SET(a, b, c, d, 12,  7, ZSW_MD5_T13);
    SET(d, a, b, c, 13, 12, ZSW_MD5_T14);
    SET(c, d, a, b, 14, 17, ZSW_MD5_T15);
    SET(b, c, d, a, 15, 22, ZSW_MD5_T16);
#undef SET

     /* Round 2. */
     /* Let [abcd k s i] denote the operation
          a = b + ((a + G(b,c,d) + X[k] + T[i]) <<< s). */
#define ZSW_MD5_G(x, y, z) (((x) & (z)) | ((y) & ~(z)))
#define SET(a, b, c, d, k, s, Ti)\
  t = a + ZSW_MD5_G(b,c,d) + X[k] + Ti;\
  a = ZSW_MD5_ROTATE_LEFT(t, s) + b
     /* Do the following 16 operations. */
    SET(a, b, c, d,  1,  5, ZSW_MD5_T17);
    SET(d, a, b, c,  6,  9, ZSW_MD5_T18);
    SET(c, d, a, b, 11, 14, ZSW_MD5_T19);
    SET(b, c, d, a,  0, 20, ZSW_MD5_T20);
    SET(a, b, c, d,  5,  5, ZSW_MD5_T21);
    SET(d, a, b, c, 10,  9, ZSW_MD5_T22);
    SET(c, d, a, b, 15, 14, ZSW_MD5_T23);
    SET(b, c, d, a,  4, 20, ZSW_MD5_T24);
    SET(a, b, c, d,  9,  5, ZSW_MD5_T25);
    SET(d, a, b, c, 14,  9, ZSW_MD5_T26);
    SET(c, d, a, b,  3, 14, ZSW_MD5_T27);
    SET(b, c, d, a,  8, 20, ZSW_MD5_T28);
    SET(a, b, c, d, 13,  5, ZSW_MD5_T29);
    SET(d, a, b, c,  2,  9, ZSW_MD5_T30);
    SET(c, d, a, b,  7, 14, ZSW_MD5_T31);
    SET(b, c, d, a, 12, 20, ZSW_MD5_T32);
#undef SET

     /* Round 3. */
     /* Let [abcd k s t] denote the operation
          a = b + ((a + H(b,c,d) + X[k] + T[i]) <<< s). */
#define ZSW_MD5_H(x, y, z) ((x) ^ (y) ^ (z))
#define SET(a, b, c, d, k, s, Ti)\
  t = a + ZSW_MD5_H(b,c,d) + X[k] + Ti;\
  a = ZSW_MD5_ROTATE_LEFT(t, s) + b
     /* Do the following 16 operations. */
    SET(a, b, c, d,  5,  4, ZSW_MD5_T33);
    SET(d, a, b, c,  8, 11, ZSW_MD5_T34);
    SET(c, d, a, b, 11, 16, ZSW_MD5_T35);
    SET(b, c, d, a, 14, 23, ZSW_MD5_T36);
    SET(a, b, c, d,  1,  4, ZSW_MD5_T37);
    SET(d, a, b, c,  4, 11, ZSW_MD5_T38);
    SET(c, d, a, b,  7, 16, ZSW_MD5_T39);
    SET(b, c, d, a, 10, 23, ZSW_MD5_T40);
    SET(a, b, c, d, 13,  4, ZSW_MD5_T41);
    SET(d, a, b, c,  0, 11, ZSW_MD5_T42);
    SET(c, d, a, b,  3, 16, ZSW_MD5_T43);
    SET(b, c, d, a,  6, 23, ZSW_MD5_T44);
    SET(a, b, c, d,  9,  4, ZSW_MD5_T45);
    SET(d, a, b, c, 12, 11, ZSW_MD5_T46);
    SET(c, d, a, b, 15, 16, ZSW_MD5_T47);
    SET(b, c, d, a,  2, 23, ZSW_MD5_T48);
#undef SET

     /* Round 4. */
     /* Let [abcd k s t] denote the operation
          a = b + ((a + I(b,c,d) + X[k] + T[i]) <<< s). */
#define ZSW_MD5_I(x, y, z) ((y) ^ ((x) | ~(z)))
#define SET(a, b, c, d, k, s, Ti)\
  t = a + ZSW_MD5_I(b,c,d) + X[k] + Ti;\
  a = ZSW_MD5_ROTATE_LEFT(t, s) + b
     /* Do the following 16 operations. */
    SET(a, b, c, d,  0,  6, ZSW_MD5_T49);
    SET(d, a, b, c,  7, 10, ZSW_MD5_T50);
    SET(c, d, a, b, 14, 15, ZSW_MD5_T51);
    SET(b, c, d, a,  5, 21, ZSW_MD5_T52);
    SET(a, b, c, d, 12,  6, ZSW_MD5_T53);
    SET(d, a, b, c,  3, 10, ZSW_MD5_T54);
    SET(c, d, a, b, 10, 15, ZSW_MD5_T55);
    SET(b, c, d, a,  1, 21, ZSW_MD5_T56);
    SET(a, b, c, d,  8,  6, ZSW_MD5_T57);
    SET(d, a, b, c, 15, 10, ZSW_MD5_T58);
    SET(c, d, a, b,  6, 15, ZSW_MD5_T59);
    SET(b, c, d, a, 13, 21, ZSW_MD5_T60);
    SET(a, b, c, d,  4,  6, ZSW_MD5_T61);
    SET(d, a, b, c, 11, 10, ZSW_MD5_T62);
    SET(c, d, a, b,  2, 15, ZSW_MD5_T63);
    SET(b, c, d, a,  9, 21, ZSW_MD5_T64);
#undef SET

     /* Then perform the following additions. (That is increment each
        of the four registers by the value it had before this block
        was started.) */
    pms->abcd[0] += a;
    pms->abcd[1] += b;
    pms->abcd[2] += c;
    pms->abcd[3] += d;
}

void md5_init(md5_state_t *pms) {
    pms->count[0] = pms->count[1] = 0;
    pms->abcd[0] = 0x67452301;
    pms->abcd[1] = /*0xefcdab89*/ ZSW_MD5_T_MASK ^ 0x10325476;
    pms->abcd[2] = /*0x98badcfe*/ ZSW_MD5_T_MASK ^ 0x67452301;
    pms->abcd[3] = 0x10325476;
}

void md5_append(md5_state_t *pms, md5_byte_t const * data, size_t nbytes) {
    md5_byte_t const * p = data;
    size_t left = nbytes;
    int offset = (pms->count[0] >> 3) & 63;
    md5_word_t nbits = (md5_word_t)(nbytes << 3);

    if (nbytes <= 0)
    return;

    /* Update the message length. */
    pms->count[1] += nbytes >> 29;
    pms->count[0] += nbits;
    if (pms->count[0] < nbits)
    pms->count[1]++;

    /* Process an initial partial block. */
    if (offset) {
    int copy = (offset + nbytes > 64 ? 64 - offset : static_cast<int>(nbytes));

    std::memcpy(pms->buf + offset, p, copy);
    if (offset + copy < 64)
        return;
    p += copy;
    left -= copy;
    md5_process(pms, pms->buf);
    }

    /* Process full blocks. */
    for (; left >= 64; p += 64, left -= 64)
    md5_process(pms, p);

    /* Process a final partial block. */
    if (left)
    std::memcpy(pms->buf, p, left);
}

void md5_finish(md5_state_t *pms, md5_byte_t digest[16]) {
    static md5_byte_t const pad[64] = {
    0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    md5_byte_t data[8];
    int i;

    /* Save the length before padding. */
    for (i = 0; i < 8; ++i)
    data[i] = (md5_byte_t)(pms->count[i >> 2] >> ((i & 3) << 3));
    /* Pad to 56 bytes mod 64. */
    md5_append(pms, pad, ((55 - (pms->count[0] >> 3)) & 63) + 1);
    /* Append the length. */
    md5_append(pms, data, 8);
    for (i = 0; i < 16; ++i)
    digest[i] = (md5_byte_t)(pms->abcd[i >> 2] >> ((i & 3) << 3));
}

// some convenience c++ functions
inline std::string md5_hash_string(std::string const & s) {
    char digest[16];

    md5_state_t state;

    md5_init(&state);
    md5_append(&state, (md5_byte_t const *)s.c_str(), s.size());
    md5_finish(&state, (md5_byte_t *)digest);

    std::string ret;
    ret.resize(16);
    std::copy(digest,digest+16,ret.begin());

    return ret;
}

const char hexval[16] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};

inline std::string md5_hash_hex(std::string const & input) {
    std::string hash = md5_hash_string(input);
    std::string hex;

    for (size_t i = 0; i < hash.size(); i++) {
        hex.push_back(hexval[((hash[i] >> 4) & 0xF)]);
        hex.push_back(hexval[(hash[i]) & 0x0F]);
    }

    return hex;
}

} // md5
} // websocketpp

#endif // WEBSOCKETPP_COMMON_MD5_HPP
