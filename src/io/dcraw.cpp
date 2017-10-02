/*
dcraw.c -- Dave Coffin's raw photo decoder
Copyright 1997-2016 by Dave Coffin, dcoffin a cybercom o net

This is a command-line ANSI C program to convert raw photos from
any digital camera on any computer running any operating system.

No license is required to download and use dcraw.c.  However,
to lawfully redistribute dcraw, you must either (a) offer, at
no extra charge, full source code* for all executable files
containing RESTRICTED functions, (b) distribute this code under
the GPL Version 2 or later, (c) remove all RESTRICTED functions,
re-implement them, or copy them from an earlier, unrestricted
Revision of dcraw.c, or (d) purchase a license from the author.

The functions that process Foveon images have been RESTRICTED
since Revision 1.237.  All other code remains free for all uses.

*If you have not modified dcraw.c in any way, a link to my
homepage qualifies as "full source code".

$Revision: 1.477 $
$Date: 2016/05/10 21:30:43 $
*/

#pragma warning(push, 0)
#include "dcraw.h"

#define DCRAW_VERSION "9.27"

#define NODEPS

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define _USE_MATH_DEFINES
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>

#if defined(DJGPP) || defined(__MINGW32__)
#define fseeko fseek
#define ftello ftell
#else
#define fgetc getc_unlocked
#endif
#ifdef __CYGWIN__
#include <io.h>
#endif

#ifdef __MINGW32__
#include <sys/utime.h>
#include <windows.h>
#include <winsock2.h>
#define INT64 long long
#define UINT64 unsigned long long

#elif defined WIN32
#include <sys/utime.h>
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#define snprintf _snprintf
#define strcasecmp stricmp
#define strncasecmp strnicmp
typedef __int64 INT64;
typedef unsigned __int64 UINT64;

#define fseeko _fseeki64
#define ftello _ftelli64
#define getc_unlocked _fgetc_nolock

#else
#include <unistd.h>
#include <utime.h>
#include <netinet/in.h>
typedef long long INT64;
typedef unsigned long long UINT64;
#endif

#ifdef NODEPS
#define NO_JASPER
#define NO_JPEG
#define NO_LCMS
#endif
#ifndef NO_JASPER
#include <jasper/jasper.h> /* Decode Red camera movies */
#endif
#ifndef NO_JPEG
#include <jpeglib.h> /* Decode compressed Kodak DC120 photos */
#endif               /* and Adobe Lossy DNGs */
#ifndef NO_LCMS
#include <lcms2.h> /* Support color profiles */
#endif
#ifdef LOCALEDIR
#include <libintl.h>
#define _(String) gettext(String)
#else
#define _(String) (String)
#endif

#if !defined(uchar)
#define uchar unsigned char
#endif
#if !defined(ushort)
#define ushort unsigned short
#endif

/*
All global variables are defined here, and all functions that
access them are prefixed with "CLASS".  Note that a thread-safe
C++ class cannot have non-const static local variables.
*/
FILE *ifp, *ofp;
short order;
const char *ifname;
char *meta_data, xtrans[6][6], xtrans_abs[6][6];
char cdesc[5], desc[512], make[64], model[64], model2[64], artist[64];
float flash_used, canon_ev, iso_speed, shutter, aperture, focal_len;
time_t timestamp;
off_t strip_offset, data_offset;
off_t thumb_offset, meta_offset, profile_offset;
unsigned shot_order, kodak_cbpp, exif_cfa, unique_id;
unsigned thumb_length, meta_length, profile_length;
unsigned thumb_misc, *oprof, fuji_layout, shot_select = 0, multi_out = 0;
unsigned tiff_nifds, tiff_samples, tiff_bps, tiff_compress;
unsigned black, maximum, mix_green, raw_color, zero_is_bad;
unsigned zero_after_ff, is_raw, dng_version, is_foveon, data_error;
unsigned tile_width, tile_length, gpsdata[32], load_flags;
unsigned flip, tiff_flip, filters, colors;
ushort raw_height, raw_width, height, width, top_margin, left_margin;
ushort shrink, iheight, iwidth, fuji_width, thumb_width, thumb_height;
ushort *raw_image, (*image)[4], cblack[4102];
ushort white[8][8], curve[0x10000], cr2_slice[3], sraw_mul[4];
double pixel_aspect, aber[4] = {1, 1, 1, 1}, gamm[6] = {0.45, 4.5, 0, 0, 0, 0};
float bright = 1, user_mul[4] = {0, 0, 0, 0}, threshold = 0;
int mask[8][4];
int half_size = 0, four_color_rgb = 0, document_mode = 0, highlight = 0;
int verbose = 0, use_auto_wb = 0, use_camera_wb = 0, use_camera_matrix = 1;
int output_color = 1, output_bps = 8, output_tiff = 0, med_passes = 0;
int no_auto_bright = 0;
unsigned greybox[4] = {0, 0, UINT_MAX, UINT_MAX};
float cam_mul[4], pre_mul[4], cmatrix[3][4], rgb_cam[3][4];
const double xyz_rgb[3][3] = {/* XYZ from RGB */
                              {0.412453, 0.357580, 0.180423},
                              {0.212671, 0.715160, 0.072169},
                              {0.019334, 0.119193, 0.950227}};
const float d65_white[3] = {0.950456, 1, 1.088754};
int histogram[4][0x2000];
void (*write_thumb)(), (*write_fun)();
void (*load_raw)(), (*thumb_load_raw)();
jmp_buf failure;

void initialize() {
#define MM(X) memset(X, 0, sizeof(X));
#define MMa(X) memset(&X, 0, sizeof(X));
  ifp = ofp = nullptr;
  order = 0;
  ifname = nullptr;
  meta_data = nullptr;
  MM(xtrans);
  MM(xtrans_abs);
  MM(cdesc);
  MM(desc);
  MM(make) MM(model) MM(model2) MM(artist);
  flash_used = canon_ev = iso_speed = shutter = aperture = focal_len = 0;
  timestamp = 0;
  strip_offset = data_offset = 0;
  thumb_offset = meta_offset = profile_offset = 0;
  shot_order = kodak_cbpp = exif_cfa = unique_id = 0;
  thumb_length = meta_length = profile_length = 0;
  thumb_misc = 0;
  oprof = nullptr;
  fuji_layout = shot_select = 0, multi_out = 0;
  tiff_nifds = tiff_samples = tiff_bps = tiff_compress = 0;
  black = maximum = mix_green = raw_color = zero_is_bad = 0;
  zero_after_ff = is_raw = dng_version = is_foveon = data_error = 0;
  tile_width = tile_length = 0;
  MM(gpsdata) load_flags = 0;
  flip = tiff_flip = filters = colors = 0;
  raw_height = raw_width = height = width = 0;
  top_margin = left_margin = 0;
  shrink = iheight = iwidth = fuji_width = thumb_width = thumb_height = 0;
  raw_image = nullptr;
  image = nullptr;
  MM(cblack);
  MM(white);
  MM(curve) MM(cr2_slice) MM(sraw_mul);
  pixel_aspect = 0;
  for (int i = 0; i < 4; i++) {
    aber[i] = 1;
  }
  MM(gamm);
  gamm[0] = gamm[1] = 0.45;
  bright = 1, MM(user_mul) threshold = 0;
  MM(mask);
  half_size = 0, four_color_rgb = 0, document_mode = 0, highlight = 0;
  verbose = 0, use_auto_wb = 0, use_camera_wb = 0, use_camera_matrix = 1;
  output_color = 1, output_bps = 8, output_tiff = 0, med_passes = 0;
  no_auto_bright = 0;
  MM(greybox);
  greybox[3] = greybox[4] = UINT_MAX;
  MM(cam_mul) MM(pre_mul) MM(cmatrix) MM(rgb_cam);
  MM(histogram);
  write_thumb = write_fun = 0;
  load_raw = thumb_load_raw = 0;
  MMa(failure);
  printf("initialized!\n");
}

struct decode {
  struct decode *branch[2];
  int leaf;
} first_decode[2048], *second_decode, *free_decode;

struct tiff_ifd {
  int width, height, bps, comp, phint, offset, flip, samples, bytes;
  int tile_width, tile_length;
  float shutter;
} tiff_ifd[10];

struct ph1 {
  int format, key_off, tag_21a;
  int black, split_col, black_col, split_row, black_row;
  float tag_210;
} ph1;

#define CLASS

#define FORC(cnt) for (c = 0; c < cnt; c++)
#define FORC3 FORC(3)
#define FORC4 FORC(4)
#define FORCC FORC(colors)

#define SQR(x) ((x) * (x))
#define ABS(x) (((int)(x) ^ ((int)(x) >> 31)) - ((int)(x) >> 31))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define LIM(x, min, max) MAX(min, MIN(x, max))
#define ULIM(x, y, z) ((y) < (z) ? LIM(x, y, z) : LIM(x, z, y))
#define CLIP(x) LIM((int)(x), 0, 65535)
#define SWAP(a, b) \
  {                \
    a = a + b;     \
    b = a - b;     \
    a = a - b;     \
  }

/*
In order to inline this calculation, I make the risky
assumption that all filter patterns can be described
by a repeating pattern of eight rows and two columns

Do not use the FC or BAYER macros with the Leaf CatchLight,
because its pattern is 16x16, not 2x8.

Return values are either 0/1/2/3 = G/M/C/Y or 0/1/2/3 = R/G1/B/G2

PowerShot 600    PowerShot A50    PowerShot Pro70    Pro90 & G1
0xe1e4e1e4:    0x1b4e4b1e:    0x1e4b4e1b:    0xb4b4b4b4:

0 1 2 3 4 5      0 1 2 3 4 5      0 1 2 3 4 5      0 1 2 3 4 5
0 G M G M G M    0 C Y C Y C Y    0 Y C Y C Y C    0 G M G M G M
1 C Y C Y C Y    1 M G M G M G    1 M G M G M G    1 Y C Y C Y C
2 M G M G M G    2 Y C Y C Y C    2 C Y C Y C Y
3 C Y C Y C Y    3 G M G M G M    3 G M G M G M
4 C Y C Y C Y    4 Y C Y C Y C
PowerShot A5    5 G M G M G M    5 G M G M G M
0x1e4e1e4e:    6 Y C Y C Y C    6 C Y C Y C Y
7 M G M G M G    7 M G M G M G
0 1 2 3 4 5
0 C Y C Y C Y
1 G M G M G M
2 C Y C Y C Y
3 M G M G M G

All RGB cameras use one of these Bayer grids:

0x16161616:    0x61616161:    0x49494949:    0x94949494:

0 1 2 3 4 5      0 1 2 3 4 5      0 1 2 3 4 5      0 1 2 3 4 5
0 B G B G B G    0 G R G R G R    0 G B G B G B    0 R G R G R G
1 G R G R G R    1 B G B G B G    1 R G R G R G    1 G B G B G B
2 B G B G B G    2 G R G R G R    2 G B G B G B    2 R G R G R G
3 G R G R G R    3 B G B G B G    3 R G R G R G    3 G B G B G B
*/

#define RAW(row, col) raw_image[(row)*raw_width + (col)]

#define FC(row, col) (filters >> ((((row) << 1 & 14) + ((col)&1)) << 1) & 3)

#define BAYER(row, col) \
  image[((row) >> shrink) * iwidth + ((col) >> shrink)][FC(row, col)]

#define BAYER2(row, col) \
  image[((row) >> shrink) * iwidth + ((col) >> shrink)][fcol(row, col)]

int CLASS fcol(int row, int col) {
  static const char filter[16][16] = {
      {2, 1, 1, 3, 2, 3, 2, 0, 3, 2, 3, 0, 1, 2, 1, 0},
      {0, 3, 0, 2, 0, 1, 3, 1, 0, 1, 1, 2, 0, 3, 3, 2},
      {2, 3, 3, 2, 3, 1, 1, 3, 3, 1, 2, 1, 2, 0, 0, 3},
      {0, 1, 0, 1, 0, 2, 0, 2, 2, 0, 3, 0, 1, 3, 2, 1},
      {3, 1, 1, 2, 0, 1, 0, 2, 1, 3, 1, 3, 0, 1, 3, 0},
      {2, 0, 0, 3, 3, 2, 3, 1, 2, 0, 2, 0, 3, 2, 2, 1},
      {2, 3, 3, 1, 2, 1, 2, 1, 2, 1, 1, 2, 3, 0, 0, 1},
      {1, 0, 0, 2, 3, 0, 0, 3, 0, 3, 0, 3, 2, 1, 2, 3},
      {2, 3, 3, 1, 1, 2, 1, 0, 3, 2, 3, 0, 2, 3, 1, 3},
      {1, 0, 2, 0, 3, 0, 3, 2, 0, 1, 1, 2, 0, 1, 0, 2},
      {0, 1, 1, 3, 3, 2, 2, 1, 1, 3, 3, 0, 2, 1, 3, 2},
      {2, 3, 2, 0, 0, 1, 3, 0, 2, 0, 1, 2, 3, 0, 1, 0},
      {1, 3, 1, 2, 3, 2, 3, 2, 0, 2, 0, 1, 1, 0, 3, 0},
      {0, 2, 0, 3, 1, 0, 0, 1, 1, 3, 3, 2, 3, 2, 2, 1},
      {2, 1, 3, 2, 3, 1, 2, 1, 0, 3, 0, 2, 0, 2, 0, 2},
      {0, 3, 1, 0, 0, 2, 0, 3, 2, 1, 3, 1, 1, 3, 1, 3}};

  if (filters == 1)
    return filter[(row + top_margin) & 15][(col + left_margin) & 15];
  if (filters == 9)
    return xtrans[(row + 6) % 6][(col + 6) % 6];
  return FC(row, col);
}

#ifndef __GLIBC__
char *my_memmem(char *haystack,
                size_t haystacklen,
                char *needle,
                size_t needlelen) {
  char *c;
  for (c = haystack; c <= haystack + haystacklen - needlelen; c++)
    if (!memcmp(c, needle, needlelen))
      return c;
  return 0;
}
#define memmem my_memmem
char *my_strcasestr(char *haystack, const char *needle) {
  char *c;
  for (c = haystack; *c; c++)
    if (!strncasecmp(c, needle, strlen(needle)))
      return c;
  return 0;
}
#define strcasestr my_strcasestr
#endif

void CLASS merror(void *ptr, const char *where) {
  if (ptr)
    return;
  fprintf(stderr, _("%s: Out of memory in %s\n"), ifname, where);
  longjmp(failure, 1);
}

void CLASS derror() {
  if (!data_error) {
    fprintf(stderr, "%s: ", ifname);
    if (feof(ifp))
      fprintf(stderr, _("Unexpected end of file\n"));
    else
      fprintf(stderr, _("Corrupt data near 0x%llx\n"), (INT64)ftello(ifp));
  }
  data_error++;
}

ushort CLASS sget2(uchar *s) {
  if (order == 0x4949) /* "II" means little-endian */
    return s[0] | s[1] << 8;
  else /* "MM" means big-endian */
    return s[0] << 8 | s[1];
}

ushort CLASS get2() {
  uchar str[2] = {0xff, 0xff};
  fread(str, 1, 2, ifp);
  return sget2(str);
}

unsigned CLASS sget4(uchar *s) {
  if (order == 0x4949)
    return s[0] | s[1] << 8 | s[2] << 16 | s[3] << 24;
  else
    return s[0] << 24 | s[1] << 16 | s[2] << 8 | s[3];
}
#define sget4(s) sget4((uchar *)s)

unsigned CLASS get4() {
  uchar str[4] = {0xff, 0xff, 0xff, 0xff};
  fread(str, 1, 4, ifp);
  return sget4(str);
}

unsigned CLASS getint(int type) {
  return type == 3 ? get2() : get4();
}

float CLASS int_to_float(int i) {
  union {
    int i;
    float f;
  } u;
  u.i = i;
  return u.f;
}

double CLASS getreal(int type) {
  union {
    char c[8];
    double d;
  } u;
  int i, rev;

  switch (type) {
    case 3:
      return (unsigned short)get2();
    case 4:
      return (unsigned int)get4();
    case 5:
      u.d = (unsigned int)get4();
      return u.d / (unsigned int)get4();
    case 8:
      return (signed short)get2();
    case 9:
      return (signed int)get4();
    case 10:
      u.d = (signed int)get4();
      return u.d / (signed int)get4();
    case 11:
      return int_to_float(get4());
    case 12:
      rev = 7 * ((order == 0x4949) == (ntohs(0x1234) == 0x1234));
      for (i = 0; i < 8; i++)
        u.c[i ^ rev] = fgetc(ifp);
      return u.d;
    default:
      return fgetc(ifp);
  }
}

void CLASS read_shorts(ushort *pixel, int count) {
  if (fread(pixel, 2, count, ifp) < count)
    derror();
  if ((order == 0x4949) == (ntohs(0x1234) == 0x1234))
    swab((char *)pixel, (char *)pixel, count * 2);
}

void CLASS cubic_spline(const int *x_, const int *y_, const int len) {
  float **A, *b, *c, *d, *x, *y;
  int i, j;

  A = (float **)calloc(((2 * len + 4) * sizeof **A + sizeof *A), 2 * len);
  if (!A)
    return;
  A[0] = (float *)(A + 2 * len);
  for (i = 1; i < 2 * len; i++)
    A[i] = A[0] + 2 * len * i;
  y = len + (x = i + (d = i + (c = i + (b = A[0] + i * i))));
  for (i = 0; i < len; i++) {
    x[i] = x_[i] / 65535.0;
    y[i] = y_[i] / 65535.0;
  }
  for (i = len - 1; i > 0; i--) {
    b[i] = (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
    d[i - 1] = x[i] - x[i - 1];
  }
  for (i = 1; i < len - 1; i++) {
    A[i][i] = 2 * (d[i - 1] + d[i]);
    if (i > 1) {
      A[i][i - 1] = d[i - 1];
      A[i - 1][i] = d[i - 1];
    }
    A[i][len - 1] = 6 * (b[i + 1] - b[i]);
  }
  for (i = 1; i < len - 2; i++) {
    float v = A[i + 1][i] / A[i][i];
    for (j = 1; j <= len - 1; j++)
      A[i + 1][j] -= v * A[i][j];
  }
  for (i = len - 2; i > 0; i--) {
    float acc = 0;
    for (j = i; j <= len - 2; j++)
      acc += A[i][j] * c[j];
    c[i] = (A[i][len - 1] - acc) / A[i][i];
  }
  for (i = 0; i < 0x10000; i++) {
    float x_out = (float)(i / 65535.0);
    float y_out = 0;
    for (j = 0; j < len - 1; j++) {
      if (x[j] <= x_out && x_out <= x[j + 1]) {
        float v = x_out - x[j];
        y_out = y[j] +
                ((y[j + 1] - y[j]) / d[j] -
                 (2 * d[j] * c[j] + c[j + 1] * d[j]) / 6) *
                    v +
                (c[j] * 0.5) * v * v +
                ((c[j + 1] - c[j]) / (6 * d[j])) * v * v * v;
      }
    }
    curve[i] = y_out < 0.0
                   ? 0
                   : (y_out >= 1.0 ? 65535 : (ushort)(y_out * 65535.0 + 0.5));
  }
  free(A);
}

void CLASS canon_600_fixed_wb(int temp) {
  static const short mul[4][5] = {{667, 358, 397, 565, 452},
                                  {731, 390, 367, 499, 517},
                                  {1119, 396, 348, 448, 537},
                                  {1399, 485, 431, 508, 688}};
  int lo, hi, i;
  float frac = 0;

  for (lo = 4; --lo;)
    if (*mul[lo] <= temp)
      break;
  for (hi = 0; hi < 3; hi++)
    if (*mul[hi] >= temp)
      break;
  if (lo != hi)
    frac = (float)(temp - *mul[lo]) / (*mul[hi] - *mul[lo]);
  for (i = 1; i < 5; i++)
    pre_mul[i - 1] = 1 / (frac * mul[hi][i] + (1 - frac) * mul[lo][i]);
}

/* Return values:  0 = white  1 = near white  2 = not white */
int CLASS canon_600_color(int ratio[2], int mar) {
  int clipped = 0, target, miss;

  if (flash_used) {
    if (ratio[1] < -104) {
      ratio[1] = -104;
      clipped = 1;
    }
    if (ratio[1] > 12) {
      ratio[1] = 12;
      clipped = 1;
    }
  } else {
    if (ratio[1] < -264 || ratio[1] > 461)
      return 2;
    if (ratio[1] < -50) {
      ratio[1] = -50;
      clipped = 1;
    }
    if (ratio[1] > 307) {
      ratio[1] = 307;
      clipped = 1;
    }
  }
  target = flash_used || ratio[1] < 197 ? -38 - (398 * ratio[1] >> 10)
                                        : -123 + (48 * ratio[1] >> 10);
  if (target - mar <= ratio[0] && target + 20 >= ratio[0] && !clipped)
    return 0;
  miss = target - ratio[0];
  if (abs(miss) >= mar * 4)
    return 2;
  if (miss < -20)
    miss = -20;
  if (miss > mar)
    miss = mar;
  ratio[0] = target - miss;
  return 1;
}

void CLASS canon_600_auto_wb() {
  int mar, row, col, i, j, st, count[] = {0, 0};
  int test[8], total[2][8], ratio[2][2], stat[2];

  memset(&total, 0, sizeof total);
  i = canon_ev + 0.5;
  if (i < 10)
    mar = 150;
  else if (i > 12)
    mar = 20;
  else
    mar = 280 - 20 * i;
  if (flash_used)
    mar = 80;
  for (row = 14; row < height - 14; row += 4)
    for (col = 10; col < width; col += 2) {
      for (i = 0; i < 8; i++)
        test[(i & 4) + FC(row + (i >> 1), col + (i & 1))] =
            BAYER(row + (i >> 1), col + (i & 1));
      for (i = 0; i < 8; i++)
        if (test[i] < 150 || test[i] > 1500)
          goto next;
      for (i = 0; i < 4; i++)
        if (abs(test[i] - test[i + 4]) > 50)
          goto next;
      for (i = 0; i < 2; i++) {
        for (j = 0; j < 4; j += 2)
          ratio[i][j >> 1] =
              ((test[i * 4 + j + 1] - test[i * 4 + j]) << 10) / test[i * 4 + j];
        stat[i] = canon_600_color(ratio[i], mar);
      }
      if ((st = stat[0] | stat[1]) > 1)
        goto next;
      for (i = 0; i < 2; i++)
        if (stat[i])
          for (j = 0; j < 2; j++)
            test[i * 4 + j * 2 + 1] =
                test[i * 4 + j * 2] * (0x400 + ratio[i][j]) >> 10;
      for (i = 0; i < 8; i++)
        total[st][i] += test[i];
      count[st]++;
    next:;
    }
  if (count[0] | count[1]) {
    st = count[0] * 200 < count[1];
    for (i = 0; i < 4; i++)
      pre_mul[i] = 1.0 / (total[st][i] + total[st][i + 4]);
  }
}

void CLASS canon_600_coeff() {
  static const short table[6][12] = {
      {-190, 702, -1878, 2390, 1861, -1349, 905, -393, -432, 944, 2617, -2105},
      {-1203, 1715, -1136, 1648, 1388, -876, 267, 245, -1641, 2153, 3921,
       -3409},
      {-615, 1127, -1563, 2075, 1437, -925, 509, 3, -756, 1268, 2519, -2007},
      {-190, 702, -1886, 2398, 2153, -1641, 763, -251, -452, 964, 3040, -2528},
      {-190, 702, -1878, 2390, 1861, -1349, 905, -393, -432, 944, 2617, -2105},
      {-807, 1319, -1785, 2297, 1388, -876, 769, -257, -230, 742, 2067, -1555}};
  int t = 0, i, c;
  float mc, yc;

  mc = pre_mul[1] / pre_mul[2];
  yc = pre_mul[3] / pre_mul[2];
  if (mc > 1 && mc <= 1.28 && yc < 0.8789)
    t = 1;
  if (mc > 1.28 && mc <= 2) {
    if (yc < 0.8789)
      t = 3;
    else if (yc <= 2)
      t = 4;
  }
  if (flash_used)
    t = 5;
  for (raw_color = i = 0; i < 3; i++)
    FORCC rgb_cam[i][c] = table[t][i * 4 + c] / 1024.0;
}

void CLASS canon_600_load_raw() {
  uchar data[1120], *dp;
  ushort *pix;
  int irow, row;

  for (irow = row = 0; irow < height; irow++) {
    if (fread(data, 1, 1120, ifp) < 1120)
      derror();
    pix = raw_image + row * raw_width;
    for (dp = data; dp < data + 1120; dp += 10, pix += 8) {
      pix[0] = (dp[0] << 2) + (dp[1] >> 6);
      pix[1] = (dp[2] << 2) + (dp[1] >> 4 & 3);
      pix[2] = (dp[3] << 2) + (dp[1] >> 2 & 3);
      pix[3] = (dp[4] << 2) + (dp[1] & 3);
      pix[4] = (dp[5] << 2) + (dp[9] & 3);
      pix[5] = (dp[6] << 2) + (dp[9] >> 2 & 3);
      pix[6] = (dp[7] << 2) + (dp[9] >> 4 & 3);
      pix[7] = (dp[8] << 2) + (dp[9] >> 6);
    }
    if ((row += 2) > height)
      row = 1;
  }
}

void CLASS canon_600_correct() {
  int row, col, val;
  static const short mul[4][2] = {
      {1141, 1145}, {1128, 1109}, {1178, 1149}, {1128, 1109}};

  for (row = 0; row < height; row++)
    for (col = 0; col < width; col++) {
      if ((val = BAYER(row, col) - black) < 0)
        val = 0;
      val = val * mul[row & 3][col & 1] >> 9;
      BAYER(row, col) = val;
    }
  canon_600_fixed_wb(1311);
  canon_600_auto_wb();
  canon_600_coeff();
  maximum = (0x3ff - black) * 1109 >> 9;
  black = 0;
}

int CLASS canon_s2is() {
  unsigned row;

  for (row = 0; row < 100; row++) {
    fseek(ifp, row * 3340 + 3284, SEEK_SET);
    if (getc(ifp) > 15)
      return 1;
  }
  return 0;
}

unsigned CLASS getbithuff(int nbits, ushort *huff) {
  static unsigned bitbuf = 0;
  static int vbits = 0, reset = 0;
  unsigned c;

  if (nbits > 25)
    return 0;
  if (nbits < 0)
    return bitbuf = vbits = reset = 0;
  if (nbits == 0 || vbits < 0)
    return 0;
  while (!reset && vbits < nbits && (c = fgetc(ifp)) != EOF &&
         !(reset = zero_after_ff && c == 0xff && fgetc(ifp))) {
    bitbuf = (bitbuf << 8) + (uchar)c;
    vbits += 8;
  }
  c = bitbuf << (32 - vbits) >> (32 - nbits);
  if (huff) {
    vbits -= huff[c] >> 8;
    c = (uchar)huff[c];
  } else
    vbits -= nbits;
  if (vbits < 0)
    derror();
  return c;
}

#define getbits(n) getbithuff(n, 0)
#define gethuff(h) getbithuff(*h, h + 1)

/*
Construct a decode tree according the specification in *source.
The first 16 bytes specify how many codes should be 1-bit, 2-bit
3-bit, etc.  Bytes after that are the leaf values.

For example, if the source is

{ 0,1,4,2,3,1,2,0,0,0,0,0,0,0,0,0,
0x04,0x03,0x05,0x06,0x02,0x07,0x01,0x08,0x09,0x00,0x0a,0x0b,0xff  },

then the code is

00        0x04
010        0x03
011        0x05
100        0x06
101        0x02
1100        0x07
1101        0x01
11100        0x08
11101        0x09
11110        0x00
111110        0x0a
1111110        0x0b
1111111        0xff
*/
ushort *CLASS make_decoder_ref(const uchar **source) {
  int max, len, h, i, j;
  const uchar *count;
  ushort *huff;

  count = (*source += 16) - 17;
  for (max = 16; max && !count[max]; max--)
    ;
  huff = (ushort *)calloc(1 + (1 << max), sizeof *huff);
  merror(huff, "make_decoder()");
  huff[0] = max;
  for (h = len = 1; len <= max; len++)
    for (i = 0; i < count[len]; i++, ++*source)
      for (j = 0; j < 1 << (max - len); j++)
        if (h <= 1 << max)
          huff[h++] = len << 8 | **source;
  return huff;
}

ushort *CLASS make_decoder(const uchar *source) {
  return make_decoder_ref(&source);
}

void CLASS crw_init_tables(unsigned table, ushort *huff[2]) {
  static const uchar first_tree[3][29] = {
      {0,    1,    4,    2,    3,    1,    2,    0,    0,    0,
       0,    0,    0,    0,    0,    0,    0x04, 0x03, 0x05, 0x06,
       0x02, 0x07, 0x01, 0x08, 0x09, 0x00, 0x0a, 0x0b, 0xff},
      {0,    2,    2,    3,    1,    1,    1,    1,    2,    0,
       0,    0,    0,    0,    0,    0,    0x03, 0x02, 0x04, 0x01,
       0x05, 0x00, 0x06, 0x07, 0x09, 0x08, 0x0a, 0x0b, 0xff},
      {0,    0,    6,    3,    1,    1,    2,    0,    0,    0,
       0,    0,    0,    0,    0,    0,    0x06, 0x05, 0x07, 0x04,
       0x08, 0x03, 0x09, 0x02, 0x00, 0x0a, 0x01, 0x0b, 0xff},
  };
  static const uchar second_tree[3][180] = {
      {0,    2,    2,    2,    1,    4,    2,    1,    2,    5,    1,    1,
       0,    0,    0,    139,  0x03, 0x04, 0x02, 0x05, 0x01, 0x06, 0x07, 0x08,
       0x12, 0x13, 0x11, 0x14, 0x09, 0x15, 0x22, 0x00, 0x21, 0x16, 0x0a, 0xf0,
       0x23, 0x17, 0x24, 0x31, 0x32, 0x18, 0x19, 0x33, 0x25, 0x41, 0x34, 0x42,
       0x35, 0x51, 0x36, 0x37, 0x38, 0x29, 0x79, 0x26, 0x1a, 0x39, 0x56, 0x57,
       0x28, 0x27, 0x52, 0x55, 0x58, 0x43, 0x76, 0x59, 0x77, 0x54, 0x61, 0xf9,
       0x71, 0x78, 0x75, 0x96, 0x97, 0x49, 0xb7, 0x53, 0xd7, 0x74, 0xb6, 0x98,
       0x47, 0x48, 0x95, 0x69, 0x99, 0x91, 0xfa, 0xb8, 0x68, 0xb5, 0xb9, 0xd6,
       0xf7, 0xd8, 0x67, 0x46, 0x45, 0x94, 0x89, 0xf8, 0x81, 0xd5, 0xf6, 0xb4,
       0x88, 0xb1, 0x2a, 0x44, 0x72, 0xd9, 0x87, 0x66, 0xd4, 0xf5, 0x3a, 0xa7,
       0x73, 0xa9, 0xa8, 0x86, 0x62, 0xc7, 0x65, 0xc8, 0xc9, 0xa1, 0xf4, 0xd1,
       0xe9, 0x5a, 0x92, 0x85, 0xa6, 0xe7, 0x93, 0xe8, 0xc1, 0xc6, 0x7a, 0x64,
       0xe1, 0x4a, 0x6a, 0xe6, 0xb3, 0xf1, 0xd3, 0xa5, 0x8a, 0xb2, 0x9a, 0xba,
       0x84, 0xa4, 0x63, 0xe5, 0xc5, 0xf3, 0xd2, 0xc4, 0x82, 0xaa, 0xda, 0xe4,
       0xf2, 0xca, 0x83, 0xa3, 0xa2, 0xc3, 0xea, 0xc2, 0xe2, 0xe3, 0xff, 0xff},
      {0,    2,    2,    1,    4,    1,    4,    1,    3,    3,    1,    0,
       0,    0,    0,    140,  0x02, 0x03, 0x01, 0x04, 0x05, 0x12, 0x11, 0x06,
       0x13, 0x07, 0x08, 0x14, 0x22, 0x09, 0x21, 0x00, 0x23, 0x15, 0x31, 0x32,
       0x0a, 0x16, 0xf0, 0x24, 0x33, 0x41, 0x42, 0x19, 0x17, 0x25, 0x18, 0x51,
       0x34, 0x43, 0x52, 0x29, 0x35, 0x61, 0x39, 0x71, 0x62, 0x36, 0x53, 0x26,
       0x38, 0x1a, 0x37, 0x81, 0x27, 0x91, 0x79, 0x55, 0x45, 0x28, 0x72, 0x59,
       0xa1, 0xb1, 0x44, 0x69, 0x54, 0x58, 0xd1, 0xfa, 0x57, 0xe1, 0xf1, 0xb9,
       0x49, 0x47, 0x63, 0x6a, 0xf9, 0x56, 0x46, 0xa8, 0x2a, 0x4a, 0x78, 0x99,
       0x3a, 0x75, 0x74, 0x86, 0x65, 0xc1, 0x76, 0xb6, 0x96, 0xd6, 0x89, 0x85,
       0xc9, 0xf5, 0x95, 0xb4, 0xc7, 0xf7, 0x8a, 0x97, 0xb8, 0x73, 0xb7, 0xd8,
       0xd9, 0x87, 0xa7, 0x7a, 0x48, 0x82, 0x84, 0xea, 0xf4, 0xa6, 0xc5, 0x5a,
       0x94, 0xa4, 0xc6, 0x92, 0xc3, 0x68, 0xb5, 0xc8, 0xe4, 0xe5, 0xe6, 0xe9,
       0xa2, 0xa3, 0xe3, 0xc2, 0x66, 0x67, 0x93, 0xaa, 0xd4, 0xd5, 0xe7, 0xf8,
       0x88, 0x9a, 0xd7, 0x77, 0xc4, 0x64, 0xe2, 0x98, 0xa5, 0xca, 0xda, 0xe8,
       0xf3, 0xf6, 0xa9, 0xb2, 0xb3, 0xf2, 0xd2, 0x83, 0xba, 0xd3, 0xff, 0xff},
      {0,    0,    6,    2,    1,    3,    3,    2,    5,    1,    2,    2,
       8,    10,   0,    117,  0x04, 0x05, 0x03, 0x06, 0x02, 0x07, 0x01, 0x08,
       0x09, 0x12, 0x13, 0x14, 0x11, 0x15, 0x0a, 0x16, 0x17, 0xf0, 0x00, 0x22,
       0x21, 0x18, 0x23, 0x19, 0x24, 0x32, 0x31, 0x25, 0x33, 0x38, 0x37, 0x34,
       0x35, 0x36, 0x39, 0x79, 0x57, 0x58, 0x59, 0x28, 0x56, 0x78, 0x27, 0x41,
       0x29, 0x77, 0x26, 0x42, 0x76, 0x99, 0x1a, 0x55, 0x98, 0x97, 0xf9, 0x48,
       0x54, 0x96, 0x89, 0x47, 0xb7, 0x49, 0xfa, 0x75, 0x68, 0xb6, 0x67, 0x69,
       0xb9, 0xb8, 0xd8, 0x52, 0xd7, 0x88, 0xb5, 0x74, 0x51, 0x46, 0xd9, 0xf8,
       0x3a, 0xd6, 0x87, 0x45, 0x7a, 0x95, 0xd5, 0xf6, 0x86, 0xb4, 0xa9, 0x94,
       0x53, 0x2a, 0xa8, 0x43, 0xf5, 0xf7, 0xd4, 0x66, 0xa7, 0x5a, 0x44, 0x8a,
       0xc9, 0xe8, 0xc8, 0xe7, 0x9a, 0x6a, 0x73, 0x4a, 0x61, 0xc7, 0xf4, 0xc6,
       0x65, 0xe9, 0x72, 0xe6, 0x71, 0x91, 0x93, 0xa6, 0xda, 0x92, 0x85, 0x62,
       0xf3, 0xc5, 0xb2, 0xa4, 0x84, 0xba, 0x64, 0xa5, 0xb3, 0xd2, 0x81, 0xe5,
       0xd3, 0xaa, 0xc4, 0xca, 0xf2, 0xb1, 0xe4, 0xd1, 0x83, 0x63, 0xea, 0xc3,
       0xe2, 0x82, 0xf1, 0xa3, 0xc2, 0xa1, 0xc1, 0xe3, 0xa2, 0xe1, 0xff, 0xff}};
  if (table > 2)
    table = 2;
  huff[0] = make_decoder(first_tree[table]);
  huff[1] = make_decoder(second_tree[table]);
}

/*
Return 0 if the image starts with compressed data,
1 if it starts with uncompressed low-order bits.

In Canon compressed data, 0xff is always followed by 0x00.
*/
int CLASS canon_has_lowbits() {
  uchar test[0x4000];
  int ret = 1, i;

  fseek(ifp, 0, SEEK_SET);
  fread(test, 1, sizeof test, ifp);
  for (i = 540; i < sizeof test - 1; i++)
    if (test[i] == 0xff) {
      if (test[i + 1])
        return 1;
      ret = 0;
    }
  return ret;
}

void CLASS canon_load_raw() {
  ushort *pixel, *prow, *huff[2];
  int nblocks, lowbits, i, c, row, r, save, val;
  int block, diffbuf[64], leaf, len, diff, carry = 0, pnum = 0, base[2];

  crw_init_tables(tiff_compress, huff);
  lowbits = canon_has_lowbits();
  if (!lowbits)
    maximum = 0x3ff;
  fseek(ifp, 540 + lowbits * raw_height * raw_width / 4, SEEK_SET);
  zero_after_ff = 1;
  getbits(-1);
  for (row = 0; row < raw_height; row += 8) {
    pixel = raw_image + row * raw_width;
    nblocks = MIN(8, raw_height - row) * raw_width >> 6;
    for (block = 0; block < nblocks; block++) {
      memset(diffbuf, 0, sizeof diffbuf);
      for (i = 0; i < 64; i++) {
        leaf = gethuff(huff[i > 0]);
        if (leaf == 0 && i)
          break;
        if (leaf == 0xff)
          continue;
        i += leaf >> 4;
        len = leaf & 15;
        if (len == 0)
          continue;
        diff = getbits(len);
        if ((diff & (1 << (len - 1))) == 0)
          diff -= (1 << len) - 1;
        if (i < 64)
          diffbuf[i] = diff;
      }
      diffbuf[0] += carry;
      carry = diffbuf[0];
      for (i = 0; i < 64; i++) {
        if (pnum++ % raw_width == 0)
          base[0] = base[1] = 512;
        if ((pixel[(block << 6) + i] = base[i & 1] += diffbuf[i]) >> 10)
          derror();
      }
    }
    if (lowbits) {
      save = ftell(ifp);
      fseek(ifp, 26 + row * raw_width / 4, SEEK_SET);
      for (prow = pixel, i = 0; i < raw_width * 2; i++) {
        c = fgetc(ifp);
        for (r = 0; r < 8; r += 2, prow++) {
          val = (*prow << 2) + ((c >> r) & 3);
          if (raw_width == 2672 && val < 512)
            val += 2;
          *prow = val;
        }
      }
      fseek(ifp, save, SEEK_SET);
    }
  }
  FORC(2) free(huff[c]);
}

struct jhead {
  int algo, bits, high, wide, clrs, sraw, psv, restart, vpred[6];
  ushort quant[64], idct[64], *huff[20], *free[20], *row;
};

int CLASS ljpeg_start(struct jhead *jh, int info_only) {
  ushort c, tag, len;
  uchar data[0x10000];
  const uchar *dp;

  memset(jh, 0, sizeof *jh);
  jh->restart = INT_MAX;
  if ((fgetc(ifp), fgetc(ifp)) != 0xd8)
    return 0;
  do {
    if (!fread(data, 2, 2, ifp))
      return 0;
    tag = data[0] << 8 | data[1];
    len = (data[2] << 8 | data[3]) - 2;
    if (tag <= 0xff00)
      return 0;
    fread(data, 1, len, ifp);
    switch (tag) {
      case 0xffc3:
        jh->sraw = ((data[7] >> 4) * (data[7] & 15) - 1) & 3;
      case 0xffc1:
      case 0xffc0:
        jh->algo = tag & 0xff;
        jh->bits = data[0];
        jh->high = data[1] << 8 | data[2];
        jh->wide = data[3] << 8 | data[4];
        jh->clrs = data[5] + jh->sraw;
        if (len == 9 && !dng_version)
          getc(ifp);
        break;
      case 0xffc4:
        if (info_only)
          break;
        for (dp = data; dp < data + len && !((c = *dp++) & -20);)
          jh->free[c] = jh->huff[c] = make_decoder_ref(&dp);
        break;
      case 0xffda:
        jh->psv = data[1 + data[0] * 2];
        jh->bits -= data[3 + data[0] * 2] & 15;
        break;
      case 0xffdb:
        FORC(64) jh->quant[c] = data[c * 2 + 1] << 8 | data[c * 2 + 2];
        break;
      case 0xffdd:
        jh->restart = data[0] << 8 | data[1];
    }
  } while (tag != 0xffda);
  if (jh->bits > 16 || jh->clrs > 6 || !jh->bits || !jh->high || !jh->wide ||
      !jh->clrs)
    return 0;
  if (info_only)
    return 1;
  if (!jh->huff[0])
    return 0;
  FORC(19) if (!jh->huff[c + 1]) jh->huff[c + 1] = jh->huff[c];
  if (jh->sraw) {
    FORC(4) jh->huff[2 + c] = jh->huff[1];
    FORC(jh->sraw) jh->huff[1 + c] = jh->huff[0];
  }
  jh->row = (ushort *)calloc(jh->wide * jh->clrs, 4);
  merror(jh->row, "ljpeg_start()");
  return zero_after_ff = 1;
}

void CLASS ljpeg_end(struct jhead *jh) {
  int c;
  FORC4 if (jh->free[c]) free(jh->free[c]);
  free(jh->row);
}

int CLASS ljpeg_diff(ushort *huff) {
  int len, diff;

  len = gethuff(huff);
  if (len == 16 && (!dng_version || dng_version >= 0x1010000))
    return -32768;
  diff = getbits(len);
  if ((diff & (1 << (len - 1))) == 0)
    diff -= (1 << len) - 1;
  return diff;
}

ushort *CLASS ljpeg_row(int jrow, struct jhead *jh) {
  int col, c, diff, pred, spred = 0;
  ushort mark = 0, *row[3];

  if (jrow * jh->wide % jh->restart == 0) {
    FORC(6) jh->vpred[c] = 1 << (jh->bits - 1);
    if (jrow) {
      fseek(ifp, -2, SEEK_CUR);
      do
        mark = (mark << 8) + (c = fgetc(ifp));
      while (c != EOF && mark >> 4 != 0xffd);
    }
    getbits(-1);
  }
  FORC3 row[c] = jh->row + jh->wide * jh->clrs * ((jrow + c) & 1);
  for (col = 0; col < jh->wide; col++)
    FORC(jh->clrs) {
      diff = ljpeg_diff(jh->huff[c]);
      if (jh->sraw && c <= jh->sraw && (col | c))
        pred = spred;
      else if (col)
        pred = row[0][-jh->clrs];
      else
        pred = (jh->vpred[c] += diff) - diff;
      if (jrow && col)
        switch (jh->psv) {
          case 1:
            break;
          case 2:
            pred = row[1][0];
            break;
          case 3:
            pred = row[1][-jh->clrs];
            break;
          case 4:
            pred = pred + row[1][0] - row[1][-jh->clrs];
            break;
          case 5:
            pred = pred + ((row[1][0] - row[1][-jh->clrs]) >> 1);
            break;
          case 6:
            pred = row[1][0] + ((pred - row[1][-jh->clrs]) >> 1);
            break;
          case 7:
            pred = (pred + row[1][0]) >> 1;
            break;
          default:
            pred = 0;
        }
      if ((**row = pred + diff) >> jh->bits)
        derror();
      if (c <= jh->sraw)
        spred = **row;
      row[0]++;
      row[1]++;
    }
  return row[2];
}

void CLASS lossless_jpeg_load_raw() {
  int jwide, jrow, jcol, val, jidx, i, j, row = 0, col = 0;
  struct jhead jh;
  ushort *rp;

  if (!ljpeg_start(&jh, 0))
    return;
  jwide = jh.wide * jh.clrs;

  for (jrow = 0; jrow < jh.high; jrow++) {
    rp = ljpeg_row(jrow, &jh);
    if (load_flags & 1)
      row = jrow & 1 ? height - 1 - jrow / 2 : jrow / 2;
    for (jcol = 0; jcol < jwide; jcol++) {
      val = curve[*rp++];
      if (cr2_slice[0]) {
        jidx = jrow * jwide + jcol;
        i = jidx / (cr2_slice[1] * raw_height);
        if ((j = i >= cr2_slice[0]))
          i = cr2_slice[0];
        jidx -= i * (cr2_slice[1] * raw_height);
        row = jidx / cr2_slice[1 + j];
        col = jidx % cr2_slice[1 + j] + i * cr2_slice[1];
      }
      if (raw_width == 3984 && (col -= 2) < 0)
        col += (row--, raw_width);
      if ((unsigned)row < raw_height)
        RAW(row, col) = val;
      if (++col >= raw_width)
        col = (row++, 0);
    }
  }
  ljpeg_end(&jh);
}

void CLASS canon_sraw_load_raw() {
  struct jhead jh;
  short *rp = 0, (*ip)[4];
  int jwide, slice, scol, ecol, row, col, jrow = 0, jcol = 0, pix[3], c;
  int v[3] = {0, 0, 0}, ver, hue;
  char *cp;

  if (!ljpeg_start(&jh, 0) || jh.clrs < 4)
    return;
  jwide = (jh.wide >>= 1) * jh.clrs;

  for (ecol = slice = 0; slice <= cr2_slice[0]; slice++) {
    scol = ecol;
    ecol += cr2_slice[1] * 2 / jh.clrs;
    if (!cr2_slice[0] || ecol > raw_width - 1)
      ecol = raw_width & -2;
    for (row = 0; row < height; row += (jh.clrs >> 1) - 1) {
      ip = (short(*)[4])image + row * width;
      for (col = scol; col < ecol; col += 2, jcol += jh.clrs) {
        if ((jcol %= jwide) == 0)
          rp = (short *)ljpeg_row(jrow++, &jh);
        if (col >= width)
          continue;
        FORC(jh.clrs - 2)
        ip[col + (c >> 1) * width + (c & 1)][0] = rp[jcol + c];
        ip[col][1] = rp[jcol + jh.clrs - 2] - 16384;
        ip[col][2] = rp[jcol + jh.clrs - 1] - 16384;
      }
    }
  }
  for (cp = model2; *cp && !isdigit(*cp); cp++)
    ;
  sscanf(cp, "%d.%d.%d", v, v + 1, v + 2);
  ver = (v[0] * 1000 + v[1]) * 1000 + v[2];
  hue = (jh.sraw + 1) << 2;
  if (unique_id >= 0x80000281 || (unique_id == 0x80000218 && ver > 1000006))
    hue = jh.sraw << 1;
  ip = (short(*)[4])image;
  rp = ip[0];
  for (row = 0; row < height; row++, ip += width) {
    if (row & (jh.sraw >> 1)) {
      for (col = 0; col < width; col += 2)
        for (c = 1; c < 3; c++) {
          if (row == height - 1)
            ip[col][c] = ip[col - width][c];
          else {
            ip[col][c] = (ip[col - width][c] + ip[col + width][c] + 1) >> 1;
          }
          for (col = 1; col < width; col += 2)
            for (c = 1; c < 3; c++)
              if (col == width - 1)
                ip[col][c] = ip[col - 1][c];
              else
                ip[col][c] = (ip[col - 1][c] + ip[col + 1][c] + 1) >> 1;
        }
    }
  }
  for (; rp < ip[0]; rp += 4) {
    if (unique_id == 0x80000218 || unique_id == 0x80000250 ||
        unique_id == 0x80000261 || unique_id == 0x80000281 ||
        unique_id == 0x80000287) {
      rp[1] = (rp[1] << 2) + hue;
      rp[2] = (rp[2] << 2) + hue;
      pix[0] = rp[0] + ((50 * rp[1] + 22929 * rp[2]) >> 14);
      pix[1] = rp[0] + ((-5640 * rp[1] - 11751 * rp[2]) >> 14);
      pix[2] = rp[0] + ((29040 * rp[1] - 101 * rp[2]) >> 14);
    } else {
      if (unique_id < 0x80000218)
        rp[0] -= 512;
      pix[0] = rp[0] + rp[2];
      pix[2] = rp[0] + rp[1];
      pix[1] = rp[0] + ((-778 * rp[1] - (rp[2] << 11)) >> 12);
    }
    FORC3 rp[c] = CLIP(pix[c] * sraw_mul[c] >> 10);
  }
  ljpeg_end(&jh);
  maximum = 0x3fff;
}

void CLASS adobe_copy_pixel(unsigned row, unsigned col, ushort **rp) {
  int c;

  if (tiff_samples == 2 && shot_select)
    (*rp)++;
  if (raw_image) {
    if (row < raw_height && col < raw_width)
      RAW(row, col) = curve[**rp];
    *rp += tiff_samples;
  } else {
    if (row < height && col < width)
      FORC(tiff_samples)
    image[row * width + col][c] = curve[(*rp)[c]];
    *rp += tiff_samples;
  }
  if (tiff_samples == 2 && shot_select)
    (*rp)--;
}

void CLASS ljpeg_idct(struct jhead *jh) {
  int c, i, j, len, skip, coef;
  float work[3][8][8];
  static float cs[106] = {0};
  static const uchar zigzag[80] = {
      0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
      12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
      35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
      58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
      63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63};

  if (!cs[0])
    FORC(106) cs[c] = cos((c & 31) * M_PI / 16) / 2;
  memset(work, 0, sizeof work);
  work[0][0][0] = jh->vpred[0] += ljpeg_diff(jh->huff[0]) * jh->quant[0];
  for (i = 1; i < 64; i++) {
    len = gethuff(jh->huff[16]);
    i += skip = len >> 4;
    if (!(len &= 15) && skip < 15)
      break;
    coef = getbits(len);
    if ((coef & (1 << (len - 1))) == 0)
      coef -= (1 << len) - 1;
    ((float *)work)[zigzag[i]] = coef * jh->quant[i];
  }
  FORC(8) work[0][0][c] *= M_SQRT1_2;
  FORC(8) work[0][c][0] *= M_SQRT1_2;
  for (i = 0; i < 8; i++)
    for (j = 0; j < 8; j++)
      FORC(8) work[1][i][j] += work[0][i][c] * cs[(j * 2 + 1) * c];
  for (i = 0; i < 8; i++)
    for (j = 0; j < 8; j++)
      FORC(8) work[2][i][j] += work[1][c][j] * cs[(i * 2 + 1) * c];

  FORC(64) jh->idct[c] = CLIP(((float *)work[2])[c] + 0.5);
}

void CLASS lossless_dng_load_raw() {
  unsigned save, trow = 0, tcol = 0, jwide, jrow, jcol, row, col, i, j;
  struct jhead jh;
  ushort *rp;

  while (trow < raw_height) {
    save = ftell(ifp);
    if (tile_length < INT_MAX)
      fseek(ifp, get4(), SEEK_SET);
    if (!ljpeg_start(&jh, 0))
      break;
    jwide = jh.wide;
    if (filters)
      jwide *= jh.clrs;
    jwide /= MIN(is_raw, tiff_samples);
    switch (jh.algo) {
      case 0xc1:
        jh.vpred[0] = 16384;
        getbits(-1);
        for (jrow = 0; jrow + 7 < jh.high; jrow += 8) {
          for (jcol = 0; jcol + 7 < jh.wide; jcol += 8) {
            ljpeg_idct(&jh);
            rp = jh.idct;
            row = trow + jcol / tile_width + jrow * 2;
            col = tcol + jcol % tile_width;
            for (i = 0; i < 16; i += 2)
              for (j = 0; j < 8; j++)
                adobe_copy_pixel(row + i, col + j, &rp);
          }
        }
        break;
      case 0xc3:
        for (row = col = jrow = 0; jrow < jh.high; jrow++) {
          rp = ljpeg_row(jrow, &jh);
          for (jcol = 0; jcol < jwide; jcol++) {
            adobe_copy_pixel(trow + row, tcol + col, &rp);
            if (++col >= tile_width || col >= raw_width)
              row += 1 + (col = 0);
          }
        }
    }
    fseek(ifp, save + 4, SEEK_SET);
    if ((tcol += tile_width) >= raw_width)
      trow += tile_length + (tcol = 0);
    ljpeg_end(&jh);
  }
}

void CLASS packed_dng_load_raw() {
  ushort *pixel, *rp;
  int row, col;

  pixel = (ushort *)calloc(raw_width, tiff_samples * sizeof *pixel);
  merror(pixel, "packed_dng_load_raw()");
  for (row = 0; row < raw_height; row++) {
    if (tiff_bps == 16)
      read_shorts(pixel, raw_width * tiff_samples);
    else {
      getbits(-1);
      for (col = 0; col < raw_width * tiff_samples; col++)
        pixel[col] = getbits(tiff_bps);
    }
    for (rp = pixel, col = 0; col < raw_width; col++)
      adobe_copy_pixel(row, col, &rp);
  }
  free(pixel);
}

void CLASS pentax_load_raw() {
  ushort bit[2][15], huff[4097];
  int dep, row, col, diff, c, i;
  ushort vpred[2][2] = {{0, 0}, {0, 0}}, hpred[2];

  fseek(ifp, meta_offset, SEEK_SET);
  dep = (get2() + 12) & 15;
  fseek(ifp, 12, SEEK_CUR);
  FORC(dep) bit[0][c] = get2();
  FORC(dep) bit[1][c] = fgetc(ifp);
  FORC(dep)
  for (i = bit[0][c]; i <= ((bit[0][c] + (4096 >> bit[1][c]) - 1) & 4095);)
    huff[++i] = bit[1][c] << 8 | c;
  huff[0] = 12;
  fseek(ifp, data_offset, SEEK_SET);
  getbits(-1);
  for (row = 0; row < raw_height; row++)
    for (col = 0; col < raw_width; col++) {
      diff = ljpeg_diff(huff);
      if (col < 2)
        hpred[col] = vpred[row & 1][col] += diff;
      else
        hpred[col & 1] += diff;
      RAW(row, col) = hpred[col & 1];
      if (hpred[col & 1] >> tiff_bps)
        derror();
    }
}

void CLASS nikon_load_raw() {
  static const uchar nikon_tree[][32] = {
      {0, 1, 5, 1, 1, 1, 1, 1, 1, 2, 0,  0,  0, 0, 0, 0, /* 12-bit lossy */
       5, 4, 3, 6, 2, 7, 1, 0, 8, 9, 11, 10, 12},
      {0,    1,    5,    1,    1,    1, 1, 1, 1, 2, 0, 0,  0,  0,
       0,    0, /* 12-bit lossy after split */
       0x39, 0x5a, 0x38, 0x27, 0x16, 5, 4, 3, 2, 1, 0, 11, 12, 12},
      {0, 1, 4, 2, 3, 1, 2, 0, 0, 0, 0,  0,  0, 0, 0, 0, /* 12-bit lossless */
       5, 4, 6, 3, 7, 2, 8, 1, 9, 0, 10, 11, 12},
      {0, 1, 4, 3, 1, 1, 1, 1, 1, 2, 0,  0,  0,  0,  0, 0, /* 14-bit lossy */
       5, 6, 4, 7, 8, 3, 9, 2, 1, 0, 10, 11, 12, 13, 14},
      {0, 1,    5,    1,    1,    1, 1, 1, 1, 1, 2, 0, 0, 0,  0,
       0, /* 14-bit lossy after split */
       8, 0x5c, 0x4b, 0x3a, 0x29, 7, 6, 5, 4, 3, 2, 1, 0, 13, 14},
      {0, 1, 4, 2, 2, 3, 1,  2, 0,  0,  0, 0, 0, 0,  0, 0, /* 14-bit lossless */
       7, 6, 8, 5, 9, 4, 10, 3, 11, 12, 2, 0, 1, 13, 14}};
  ushort *huff, ver0, ver1, vpred[2][2], hpred[2], csize;
  int i, min, max, step = 0, tree = 0, split = 0, row, col, len, shl, diff;

  fseek(ifp, meta_offset, SEEK_SET);
  ver0 = fgetc(ifp);
  ver1 = fgetc(ifp);
  if (ver0 == 0x49 || ver1 == 0x58)
    fseek(ifp, 2110, SEEK_CUR);
  if (ver0 == 0x46)
    tree = 2;
  if (tiff_bps == 14)
    tree += 3;
  read_shorts(vpred[0], 4);
  max = 1 << tiff_bps & 0x7fff;
  if ((csize = get2()) > 1)
    step = max / (csize - 1);
  if (ver0 == 0x44 && ver1 == 0x20 && step > 0) {
    for (i = 0; i < csize; i++)
      curve[i * step] = get2();
    for (i = 0; i < max; i++)
      curve[i] = (curve[i - i % step] * (step - i % step) +
                  curve[i - i % step + step] * (i % step)) /
                 step;
    fseek(ifp, meta_offset + 562, SEEK_SET);
    split = get2();
  } else if (ver0 != 0x46 && csize <= 0x4001)
    read_shorts(curve, max = csize);
  while (curve[max - 2] == curve[max - 1])
    max--;
  huff = make_decoder(nikon_tree[tree]);
  fseek(ifp, data_offset, SEEK_SET);
  getbits(-1);
  for (min = row = 0; row < height; row++) {
    if (split && row == split) {
      free(huff);
      huff = make_decoder(nikon_tree[tree + 1]);
      max += (min = 16) << 1;
    }
    for (col = 0; col < raw_width; col++) {
      i = gethuff(huff);
      len = i & 15;
      shl = i >> 4;
      diff = ((getbits(len - shl) << 1) + 1) << shl >> 1;
      if ((diff & (1 << (len - 1))) == 0)
        diff -= (1 << len) - !shl;
      if (col < 2)
        hpred[col] = vpred[row & 1][col] += diff;
      else
        hpred[col & 1] += diff;
      if ((ushort)(hpred[col & 1] + min) >= max)
        derror();
      RAW(row, col) = curve[LIM((short)hpred[col & 1], 0, 0x3fff)];
    }
  }
  free(huff);
}

void CLASS nikon_yuv_load_raw() {
  int row, col, yuv[4], rgb[3], b, c;
  UINT64 bitbuf = 0;

  for (row = 0; row < raw_height; row++)
    for (col = 0; col < raw_width; col++) {
      if (!(b = col & 1)) {
        bitbuf = 0;
        FORC(6) bitbuf |= (UINT64)fgetc(ifp) << c * 8;
        FORC(4) yuv[c] = (bitbuf >> c * 12 & 0xfff) - (c >> 1 << 11);
      }
      rgb[0] = yuv[b] + 1.370705 * yuv[3];
      rgb[1] = yuv[b] - 0.337633 * yuv[2] - 0.698001 * yuv[3];
      rgb[2] = yuv[b] + 1.732446 * yuv[2];
      FORC3 image[row * width + col][c] =
          curve[LIM(rgb[c], 0, 0xfff)] / cam_mul[c];
    }
}

/*
Returns 1 for a Coolpix 995, 0 for anything else.
*/
int CLASS nikon_e995() {
  int i, histo[256];
  const uchar often[] = {0x00, 0x55, 0xaa, 0xff};

  memset(histo, 0, sizeof histo);
  fseek(ifp, -2000, SEEK_END);
  for (i = 0; i < 2000; i++)
    histo[fgetc(ifp)]++;
  for (i = 0; i < 4; i++)
    if (histo[often[i]] < 200)
      return 0;
  return 1;
}

/*
Returns 1 for a Coolpix 2100, 0 for anything else.
*/
int CLASS nikon_e2100() {
  uchar t[12];
  int i;

  fseek(ifp, 0, SEEK_SET);
  for (i = 0; i < 1024; i++) {
    fread(t, 1, 12, ifp);
    if (((t[2] & t[4] & t[7] & t[9]) >> 4 & t[1] & t[6] & t[8] & t[11] & 3) !=
        3)
      return 0;
  }
  return 1;
}

void CLASS nikon_3700() {
  int bits, i;
  uchar dp[24];
  static const struct {
    int bits;
    char make[12], model[15];
  } table[] = {{0x00, "Pentax", "Optio 33WR"},
               {0x03, "Nikon", "E3200"},
               {0x32, "Nikon", "E3700"},
               {0x33, "Olympus", "C740UZ"}};

  fseek(ifp, 3072, SEEK_SET);
  fread(dp, 1, 24, ifp);
  bits = (dp[8] & 3) << 4 | (dp[20] & 3);
  for (i = 0; i < sizeof table / sizeof *table; i++)
    if (bits == table[i].bits) {
      strcpy(make, table[i].make);
      strcpy(model, table[i].model);
    }
}

/*
Separates a Minolta DiMAGE Z2 from a Nikon E4300.
*/
int CLASS minolta_z2() {
  int i, nz;
  char tail[424];

  fseek(ifp, -sizeof tail, SEEK_END);
  fread(tail, 1, sizeof tail, ifp);
  for (nz = i = 0; i < sizeof tail; i++)
    if (tail[i])
      nz++;
  return nz > 20;
}

void CLASS jpeg_thumb();

void CLASS ppm_thumb() {
  char *thumb;
  thumb_length = thumb_width * thumb_height * 3;
  thumb = (char *)malloc(thumb_length);
  merror(thumb, "ppm_thumb()");
  fprintf(ofp, "P6\n%d %d\n255\n", thumb_width, thumb_height);
  fread(thumb, 1, thumb_length, ifp);
  fwrite(thumb, 1, thumb_length, ofp);
  free(thumb);
}

void CLASS ppm16_thumb() {
  int i;
  char *thumb;
  thumb_length = thumb_width * thumb_height * 3;
  thumb = (char *)calloc(thumb_length, 2);
  merror(thumb, "ppm16_thumb()");
  read_shorts((ushort *)thumb, thumb_length);
  for (i = 0; i < thumb_length; i++)
    thumb[i] = ((ushort *)thumb)[i] >> 8;
  fprintf(ofp, "P6\n%d %d\n255\n", thumb_width, thumb_height);
  fwrite(thumb, 1, thumb_length, ofp);
  free(thumb);
}

void CLASS layer_thumb() {
  int i, c;
  char *thumb, map[][4] = {"012", "102"};

  colors = thumb_misc >> 5 & 7;
  thumb_length = thumb_width * thumb_height;
  thumb = (char *)calloc(colors, thumb_length);
  merror(thumb, "layer_thumb()");
  fprintf(ofp, "P%d\n%d %d\n255\n", 5 + (colors >> 1), thumb_width,
          thumb_height);
  fread(thumb, thumb_length, colors, ifp);
  for (i = 0; i < thumb_length; i++)
    FORCC putc(thumb[i + thumb_length * (map[thumb_misc >> 8][c] - '0')], ofp);
  free(thumb);
}

void CLASS rollei_thumb() {
  unsigned i;
  ushort *thumb;

  thumb_length = thumb_width * thumb_height;
  thumb = (ushort *)calloc(thumb_length, 2);
  merror(thumb, "rollei_thumb()");
  fprintf(ofp, "P6\n%d %d\n255\n", thumb_width, thumb_height);
  read_shorts(thumb, thumb_length);
  for (i = 0; i < thumb_length; i++) {
    putc(thumb[i] << 3, ofp);
    putc(thumb[i] >> 5 << 2, ofp);
    putc(thumb[i] >> 11 << 3, ofp);
  }
  free(thumb);
}

void CLASS rollei_load_raw() {
  uchar pixel[10];
  unsigned iten = 0, isix, i, buffer = 0, todo[16];

  isix = raw_width * raw_height * 5 / 8;
  while (fread(pixel, 1, 10, ifp) == 10) {
    for (i = 0; i < 10; i += 2) {
      todo[i] = iten++;
      todo[i + 1] = pixel[i] << 8 | pixel[i + 1];
      buffer = pixel[i] >> 2 | buffer << 6;
    }
    for (; i < 16; i += 2) {
      todo[i] = isix++;
      todo[i + 1] = buffer >> (14 - i) * 5;
    }
    for (i = 0; i < 16; i += 2)
      raw_image[todo[i]] = (todo[i + 1] & 0x3ff);
  }
  maximum = 0x3ff;
}

int CLASS raw(unsigned row, unsigned col) {
  return (row < raw_height && col < raw_width) ? RAW(row, col) : 0;
}

void CLASS phase_one_flat_field(int is_float, int nc) {
  ushort head[8];
  unsigned wide, high, y, x, c, rend, cend, row, col;
  float *mrow, num, mult[4];

  read_shorts(head, 8);
  if (head[2] * head[3] * head[4] * head[5] == 0)
    return;
  wide = head[2] / head[4] + (head[2] % head[4] != 0);
  high = head[3] / head[5] + (head[3] % head[5] != 0);
  mrow = (float *)calloc(nc * wide, sizeof *mrow);
  merror(mrow, "phase_one_flat_field()");
  for (y = 0; y < high; y++) {
    for (x = 0; x < wide; x++)
      for (c = 0; c < nc; c += 2) {
        num = is_float ? getreal(11) : get2() / 32768.0;
        if (y == 0)
          mrow[c * wide + x] = num;
        else
          mrow[(c + 1) * wide + x] = (num - mrow[c * wide + x]) / head[5];
      }
    if (y == 0)
      continue;
    rend = head[1] + y * head[5];
    for (row = rend - head[5];
         row < raw_height && row < rend && row < head[1] + head[3] - head[5];
         row++) {
      for (x = 1; x < wide; x++) {
        for (c = 0; c < nc; c += 2) {
          mult[c] = mrow[c * wide + x - 1];
          mult[c + 1] = (mrow[c * wide + x] - mult[c]) / head[4];
        }
        cend = head[0] + x * head[4];
        for (col = cend - head[4];
             col < raw_width && col < cend && col < head[0] + head[2] - head[4];
             col++) {
          c = nc > 2 ? FC(row - top_margin, col - left_margin) : 0;
          if (!(c & 1)) {
            c = RAW(row, col) * mult[c];
            RAW(row, col) = LIM(c, 0, 65535);
          }
          for (c = 0; c < nc; c += 2)
            mult[c] += mult[c + 1];
        }
      }
      for (x = 0; x < wide; x++)
        for (c = 0; c < nc; c += 2)
          mrow[c * wide + x] += mrow[(c + 1) * wide + x];
    }
  }
  free(mrow);
}

void CLASS phase_one_correct() {
  unsigned entries, tag, data, save, col, row, type;
  int len, i, j, k, cip, val[4], dev[4], sum, max;
  int head[9], diff, mindiff = INT_MAX, off_412 = 0;
  static const signed char dir[12][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1},
                                         {-2, 0},  {0, -2}, {0, 2},  {2, 0},
                                         {-2, -2}, {-2, 2}, {2, -2}, {2, 2}};
  float poly[8], num, cfrac, frac, mult[2], *yval[2];
  ushort *xval[2];
  int qmult_applied = 0, qlin_applied = 0;

  if (half_size || !meta_length)
    return;
  if (verbose)
    fprintf(stderr, _("Phase One correction...\n"));
  fseek(ifp, meta_offset, SEEK_SET);
  order = get2();
  fseek(ifp, 6, SEEK_CUR);
  fseek(ifp, meta_offset + get4(), SEEK_SET);
  entries = get4();
  get4();
  while (entries--) {
    tag = get4();
    len = get4();
    data = get4();
    save = ftell(ifp);
    fseek(ifp, meta_offset + data, SEEK_SET);
    if (tag == 0x419) { /* Polynomial curve */
      for (get4(), i = 0; i < 8; i++)
        poly[i] = getreal(11);
      poly[3] += (ph1.tag_210 - poly[7]) * poly[6] + 1;
      for (i = 0; i < 0x10000; i++) {
        num = (poly[5] * i + poly[3]) * i + poly[1];
        curve[i] = LIM(num, 0, 65535);
      }
      goto apply;              /* apply to right half */
    } else if (tag == 0x41a) { /* Polynomial curve */
      for (i = 0; i < 4; i++)
        poly[i] = getreal(11);
      for (i = 0; i < 0x10000; i++) {
        for (num = 0, j = 4; j--;)
          num = num * i + poly[j];
        curve[i] = LIM(num + i, 0, 65535);
      }
    apply: /* apply to whole image */
      for (row = 0; row < raw_height; row++)
        for (col = (tag & 1) * ph1.split_col; col < raw_width; col++)
          RAW(row, col) = curve[RAW(row, col)];
    } else if (tag == 0x400) { /* Sensor defects */
      while ((len -= 8) >= 0) {
        col = get2();
        row = get2();
        type = get2();
        get2();
        if (col >= raw_width)
          continue;
        if (type == 131 || type == 137) /* Bad column */
          for (row = 0; row < raw_height; row++)
            if (FC(row - top_margin, col - left_margin) == 1) {
              for (sum = i = 0; i < 4; i++)
                sum += val[i] = raw(row + dir[i][0], col + dir[i][1]);
              for (max = i = 0; i < 4; i++) {
                dev[i] = abs((val[i] << 2) - sum);
                if (dev[max] < dev[i])
                  max = i;
              }
              RAW(row, col) = (sum - val[max]) / 3.0 + 0.5;
            } else {
              for (sum = 0, i = 8; i < 12; i++)
                sum += raw(row + dir[i][0], col + dir[i][1]);
              RAW(row, col) =
                  0.5 + sum * 0.0732233 +
                  (raw(row, col - 2) + raw(row, col + 2)) * 0.3535534;
            }
        else if (type == 129) { /* Bad pixel */
          if (row >= raw_height)
            continue;
          j = (FC(row - top_margin, col - left_margin) != 1) * 4;
          for (sum = 0, i = j; i < j + 8; i++)
            sum += raw(row + dir[i][0], col + dir[i][1]);
          RAW(row, col) = (sum + 4) >> 3;
        }
      }
    } else if (tag == 0x401) { /* All-color flat fields */
      phase_one_flat_field(1, 2);
    } else if (tag == 0x416 || tag == 0x410) {
      phase_one_flat_field(0, 2);
    } else if (tag == 0x40b) { /* Red+blue flat field */
      phase_one_flat_field(0, 4);
    } else if (tag == 0x412) {
      fseek(ifp, 36, SEEK_CUR);
      diff = abs(get2() - ph1.tag_21a);
      if (mindiff > diff) {
        mindiff = diff;
        off_412 = ftell(ifp) - 38;
      }
    } else if (tag == 0x41f && !qlin_applied) { /* Quadrant linearization */
      ushort lc[2][2][16], ref[16];
      int qr, qc;
      for (qr = 0; qr < 2; qr++)
        for (qc = 0; qc < 2; qc++)
          for (i = 0; i < 16; i++)
            lc[qr][qc][i] = get4();
      for (i = 0; i < 16; i++) {
        int v = 0;
        for (qr = 0; qr < 2; qr++)
          for (qc = 0; qc < 2; qc++)
            v += lc[qr][qc][i];
        ref[i] = (v + 2) >> 2;
      }
      for (qr = 0; qr < 2; qr++) {
        for (qc = 0; qc < 2; qc++) {
          int cx[19], cf[19];
          for (i = 0; i < 16; i++) {
            cx[1 + i] = lc[qr][qc][i];
            cf[1 + i] = ref[i];
          }
          cx[0] = cf[0] = 0;
          cx[17] = cf[17] = ((unsigned)ref[15] * 65535) / lc[qr][qc][15];
          cx[18] = cf[18] = 65535;
          cubic_spline(cx, cf, 19);
          for (row = (qr ? ph1.split_row : 0);
               row < (qr ? raw_height : ph1.split_row); row++)
            for (col = (qc ? ph1.split_col : 0);
                 col < (qc ? raw_width : ph1.split_col); col++)
              RAW(row, col) = curve[RAW(row, col)];
        }
      }
      qlin_applied = 1;
    } else if (tag == 0x41e && !qmult_applied) { /* Quadrant multipliers */
      float qmult[2][2] = {{1, 1}, {1, 1}};
      get4();
      get4();
      get4();
      get4();
      qmult[0][0] = 1.0 + getreal(11);
      get4();
      get4();
      get4();
      get4();
      get4();
      qmult[0][1] = 1.0 + getreal(11);
      get4();
      get4();
      get4();
      qmult[1][0] = 1.0 + getreal(11);
      get4();
      get4();
      get4();
      qmult[1][1] = 1.0 + getreal(11);
      for (row = 0; row < raw_height; row++)
        for (col = 0; col < raw_width; col++) {
          i = qmult[row >= ph1.split_row][col >= ph1.split_col] * RAW(row, col);
          RAW(row, col) = LIM(i, 0, 65535);
        }
      qmult_applied = 1;
    } else if (tag == 0x431 && !qmult_applied) { /* Quadrant combined */
      ushort lc[2][2][7], ref[7];
      int qr, qc;
      for (i = 0; i < 7; i++)
        ref[i] = get4();
      for (qr = 0; qr < 2; qr++)
        for (qc = 0; qc < 2; qc++)
          for (i = 0; i < 7; i++)
            lc[qr][qc][i] = get4();
      for (qr = 0; qr < 2; qr++) {
        for (qc = 0; qc < 2; qc++) {
          int cx[9], cf[9];
          for (i = 0; i < 7; i++) {
            cx[1 + i] = ref[i];
            cf[1 + i] = ((unsigned)ref[i] * lc[qr][qc][i]) / 10000;
          }
          cx[0] = cf[0] = 0;
          cx[8] = cf[8] = 65535;
          cubic_spline(cx, cf, 9);
          for (row = (qr ? ph1.split_row : 0);
               row < (qr ? raw_height : ph1.split_row); row++)
            for (col = (qc ? ph1.split_col : 0);
                 col < (qc ? raw_width : ph1.split_col); col++)
              RAW(row, col) = curve[RAW(row, col)];
        }
      }
      qmult_applied = 1;
      qlin_applied = 1;
    }
    fseek(ifp, save, SEEK_SET);
  }
  if (off_412) {
    fseek(ifp, off_412, SEEK_SET);
    for (i = 0; i < 9; i++)
      head[i] = get4() & 0x7fff;
    yval[0] = (float *)calloc(head[1] * head[3] + head[2] * head[4], 6);
    merror(yval[0], "phase_one_correct()");
    yval[1] = (float *)(yval[0] + head[1] * head[3]);
    xval[0] = (ushort *)(yval[1] + head[2] * head[4]);
    xval[1] = (ushort *)(xval[0] + head[1] * head[3]);
    get2();
    for (i = 0; i < 2; i++)
      for (j = 0; j < head[i + 1] * head[i + 3]; j++)
        yval[i][j] = getreal(11);
    for (i = 0; i < 2; i++)
      for (j = 0; j < head[i + 1] * head[i + 3]; j++)
        xval[i][j] = get2();
    for (row = 0; row < raw_height; row++)
      for (col = 0; col < raw_width; col++) {
        cfrac = (float)col * head[3] / raw_width;
        cfrac -= cip = cfrac;
        num = RAW(row, col) * 0.5;
        for (i = cip; i < cip + 2; i++) {
          for (k = j = 0; j < head[1]; j++)
            if (num < xval[0][k = head[1] * i + j])
              break;
          frac = (j == 0 || j == head[1])
                     ? 0
                     : (xval[0][k] - num) / (xval[0][k] - xval[0][k - 1]);
          mult[i - cip] = yval[0][k - 1] * frac + yval[0][k] * (1 - frac);
        }
        i = ((mult[0] * (1 - cfrac) + mult[1] * cfrac) * row + num) * 2;
        RAW(row, col) = LIM(i, 0, 65535);
      }
    free(yval[0]);
  }
}

void CLASS phase_one_load_raw() {
  int a, b, i;
  ushort akey, bkey, mask;

  fseek(ifp, ph1.key_off, SEEK_SET);
  akey = get2();
  bkey = get2();
  mask = ph1.format == 1 ? 0x5555 : 0x1354;
  fseek(ifp, data_offset, SEEK_SET);
  read_shorts(raw_image, raw_width * raw_height);
  if (ph1.format)
    for (i = 0; i < raw_width * raw_height; i += 2) {
      a = raw_image[i + 0] ^ akey;
      b = raw_image[i + 1] ^ bkey;
      raw_image[i + 0] = (a & mask) | (b & ~mask);
      raw_image[i + 1] = (b & mask) | (a & ~mask);
    }
}

unsigned CLASS ph1_bithuff(int nbits, ushort *huff) {
  static UINT64 bitbuf = 0;
  static int vbits = 0;
  unsigned c;

  if (nbits == -1)
    return bitbuf = vbits = 0;
  if (nbits == 0)
    return 0;
  if (vbits < nbits) {
    bitbuf = bitbuf << 32 | get4();
    vbits += 32;
  }
  c = bitbuf << (64 - vbits) >> (64 - nbits);
  if (huff) {
    vbits -= huff[c] >> 8;
    return (uchar)huff[c];
  }
  vbits -= nbits;
  return c;
}
#define ph1_bits(n) ph1_bithuff(n, 0)
#define ph1_huff(h) ph1_bithuff(*h, h + 1)

void CLASS phase_one_load_raw_c() {
  static const int length[] = {8, 7, 6, 9, 11, 10, 5, 12, 14, 13};
  int *offset, len[2], pred[2], row, col, i, j;
  ushort *pixel;
  short(*cblack)[2], (*rblack)[2];

  pixel = (ushort *)calloc(raw_width * 3 + raw_height * 4, 2);
  merror(pixel, "phase_one_load_raw_c()");
  offset = (int *)(pixel + raw_width);
  fseek(ifp, strip_offset, SEEK_SET);
  for (row = 0; row < raw_height; row++)
    offset[row] = get4();
  cblack = (short(*)[2])(offset + raw_height);
  fseek(ifp, ph1.black_col, SEEK_SET);
  if (ph1.black_col)
    read_shorts((ushort *)cblack[0], raw_height * 2);
  rblack = cblack + raw_height;
  fseek(ifp, ph1.black_row, SEEK_SET);
  if (ph1.black_row)
    read_shorts((ushort *)rblack[0], raw_width * 2);
  for (i = 0; i < 256; i++)
    curve[i] = i * i / 3.969 + 0.5;
  for (row = 0; row < raw_height; row++) {
    fseek(ifp, data_offset + offset[row], SEEK_SET);
    ph1_bits(-1);
    pred[0] = pred[1] = 0;
    for (col = 0; col < raw_width; col++) {
      if (col >= (raw_width & -8))
        len[0] = len[1] = 14;
      else if ((col & 7) == 0)
        for (i = 0; i < 2; i++) {
          for (j = 0; j < 5 && !ph1_bits(1); j++)
            ;
          if (j--)
            len[i] = length[j * 2 + ph1_bits(1)];
        }
      if ((i = len[col & 1]) == 14)
        pixel[col] = pred[col & 1] = ph1_bits(16);
      else
        pixel[col] = pred[col & 1] += ph1_bits(i) + 1 - (1 << (i - 1));
      if (pred[col & 1] >> 16)
        derror();
      if (ph1.format == 5 && pixel[col] < 256)
        pixel[col] = curve[pixel[col]];
    }
    for (col = 0; col < raw_width; col++) {
      i = (pixel[col] << 2 * (ph1.format != 8)) - ph1.black +
          cblack[row][col >= ph1.split_col] + rblack[col][row >= ph1.split_row];
      if (i > 0)
        RAW(row, col) = i;
    }
  }
  free(pixel);
  maximum = 0xfffc - ph1.black;
}

void CLASS hasselblad_load_raw() {
  struct jhead jh;
  int shot, row, col, *back[5], len[2], diff[12], pred, sh, f, s, c;
  unsigned upix, urow, ucol;
  ushort *ip;

  if (!ljpeg_start(&jh, 0))
    return;
  order = 0x4949;
  ph1_bits(-1);
  back[4] = (int *)calloc(raw_width, 3 * sizeof **back);
  merror(back[4], "hasselblad_load_raw()");
  FORC3 back[c] = back[4] + c * raw_width;
  cblack[6] >>= sh = tiff_samples > 1;
  shot = LIM(shot_select, 1, tiff_samples) - 1;
  for (row = 0; row < raw_height; row++) {
    FORC4 back[(c + 3) & 3] = back[c];
    for (col = 0; col < raw_width; col += 2) {
      for (s = 0; s < tiff_samples * 2; s += 2) {
        FORC(2) len[c] = ph1_huff(jh.huff[0]);
        FORC(2) {
          diff[s + c] = ph1_bits(len[c]);
          if ((diff[s + c] & (1 << (len[c] - 1))) == 0)
            diff[s + c] -= (1 << len[c]) - 1;
          if (diff[s + c] == 65535)
            diff[s + c] = -32768;
        }
      }
      for (s = col; s < col + 2; s++) {
        pred = 0x8000 + load_flags;
        if (col)
          pred = back[2][s - 2];
        if (col && row > 1)
          switch (jh.psv) {
            case 11:
              pred += back[0][s] / 2 - back[0][s - 2] / 2;
              break;
          }
        f = (row & 1) * 3 ^ ((col + s) & 1);
        FORC(tiff_samples) {
          pred += diff[(s & 1) * tiff_samples + c];
          upix = pred >> sh & 0xffff;
          if (raw_image && c == shot)
            RAW(row, s) = upix;
          if (image) {
            urow = row - top_margin + (c & 1);
            ucol = col - left_margin - ((c >> 1) & 1);
            ip = &image[urow * width + ucol][f];
            if (urow < height && ucol < width)
              *ip = c < 4 ? upix : (*ip + upix) >> 1;
          }
        }
        back[2][s] = pred;
      }
    }
  }
  free(back[4]);
  ljpeg_end(&jh);
  if (image)
    mix_green = 1;
}

void CLASS leaf_hdr_load_raw() {
  ushort *pixel = 0;
  unsigned tile = 0, r, c, row, col;

  if (!filters) {
    pixel = (ushort *)calloc(raw_width, sizeof *pixel);
    merror(pixel, "leaf_hdr_load_raw()");
  }
  FORC(tiff_samples)
  for (r = 0; r < raw_height; r++) {
    if (r % tile_length == 0) {
      fseek(ifp, data_offset + 4 * tile++, SEEK_SET);
      fseek(ifp, get4(), SEEK_SET);
    }
    if (filters && c != shot_select)
      continue;
    if (filters)
      pixel = raw_image + r * raw_width;
    read_shorts(pixel, raw_width);
    if (!filters && (row = r - top_margin) < height)
      for (col = 0; col < width; col++)
        image[row * width + col][c] = pixel[col + left_margin];
  }
  if (!filters) {
    maximum = 0xffff;
    raw_color = 1;
    free(pixel);
  }
}

void CLASS unpacked_load_raw() {
  int row, col, bits = 0;

  while (1 << ++bits < maximum)
    ;
  read_shorts(raw_image, raw_width * raw_height);
  for (row = 0; row < raw_height; row++)
    for (col = 0; col < raw_width; col++)
      if ((RAW(row, col) >>= load_flags) >> bits &&
          (unsigned)(row - top_margin) < height &&
          (unsigned)(col - left_margin) < width)
        derror();
}

void CLASS sinar_4shot_load_raw() {
  ushort *pixel;
  unsigned shot, row, col, r, c;

  if (raw_image) {
    shot = LIM(shot_select, 1, 4) - 1;
    fseek(ifp, data_offset + shot * 4, SEEK_SET);
    fseek(ifp, get4(), SEEK_SET);
    unpacked_load_raw();
    return;
  }
  pixel = (ushort *)calloc(raw_width, sizeof *pixel);
  merror(pixel, "sinar_4shot_load_raw()");
  for (shot = 0; shot < 4; shot++) {
    fseek(ifp, data_offset + shot * 4, SEEK_SET);
    fseek(ifp, get4(), SEEK_SET);
    for (row = 0; row < raw_height; row++) {
      read_shorts(pixel, raw_width);
      if ((r = row - top_margin - (shot >> 1 & 1)) >= height)
        continue;
      for (col = 0; col < raw_width; col++) {
        if ((c = col - left_margin - (shot & 1)) >= width)
          continue;
        image[r * width + c][(row & 1) * 3 ^ (~col & 1)] = pixel[col];
      }
    }
  }
  free(pixel);
  mix_green = 1;
}

void CLASS imacon_full_load_raw() {
  int row, col;

  if (!image)
    return;
  for (row = 0; row < height; row++)
    for (col = 0; col < width; col++)
      read_shorts(image[row * width + col], 3);
}

void CLASS packed_load_raw() {
  int vbits = 0, bwide, rbits, bite, half, irow, row, col, val, i;
  UINT64 bitbuf = 0;

  bwide = raw_width * tiff_bps / 8;
  bwide += bwide & load_flags >> 7;
  rbits = bwide * 8 - raw_width * tiff_bps;
  if (load_flags & 1)
    bwide = bwide * 16 / 15;
  bite = 8 + (load_flags & 24);
  half = (raw_height + 1) >> 1;
  for (irow = 0; irow < raw_height; irow++) {
    row = irow;
    if (load_flags & 2 && (row = irow % half * 2 + irow / half) == 1 &&
        load_flags & 4) {
      if (vbits = 0, tiff_compress)
        fseek(ifp, data_offset - (-half * bwide & -2048), SEEK_SET);
      else {
        fseek(ifp, 0, SEEK_END);
        fseek(ifp, ftell(ifp) >> 3 << 2, SEEK_SET);
      }
    }
    for (col = 0; col < raw_width; col++) {
      for (vbits -= tiff_bps; vbits < 0; vbits += bite) {
        bitbuf <<= bite;
        for (i = 0; i < bite; i += 8)
          bitbuf |= (unsigned)(fgetc(ifp) << i);
      }
      val = bitbuf << (64 - tiff_bps - vbits) >> (64 - tiff_bps);
      RAW(row, col ^ (load_flags >> 6 & 1)) = val;
      if (load_flags & 1 && (col % 10) == 9 && fgetc(ifp) &&
          row < height + top_margin && col < width + left_margin)
        derror();
    }
    vbits -= rbits;
  }
}

void CLASS nokia_load_raw() {
  uchar *data, *dp;
  int rev, dwide, row, col, c;
  double sum[] = {0, 0};

  rev = 3 * (order == 0x4949);
  dwide = (raw_width * 5 + 1) / 4;
  data = (uchar *)malloc(dwide * 2);
  merror(data, "nokia_load_raw()");
  for (row = 0; row < raw_height; row++) {
    if (fread(data + dwide, 1, dwide, ifp) < dwide)
      derror();
    FORC(dwide) data[c] = data[dwide + (c ^ rev)];
    for (dp = data, col = 0; col < raw_width; dp += 5, col += 4)
      FORC4 RAW(row, col + c) = (dp[c] << 2) | (dp[4] >> (c << 1) & 3);
  }
  free(data);
  maximum = 0x3ff;
  if (strcmp(make, "OmniVision"))
    return;
  row = raw_height / 2;
  FORC(width - 1) {
    sum[c & 1] += SQR(RAW(row, c) - RAW(row + 1, c + 1));
    sum[~c & 1] += SQR(RAW(row + 1, c) - RAW(row, c + 1));
  }
  if (sum[1] > sum[0])
    filters = 0x4b4b4b4b;
}

void CLASS canon_rmf_load_raw() {
  int row, col, bits, orow, ocol, c;

  for (row = 0; row < raw_height; row++)
    for (col = 0; col < raw_width - 2; col += 3) {
      bits = get4();
      FORC3 {
        orow = row;
        if ((ocol = col + c - 4) < 0) {
          ocol += raw_width;
          if ((orow -= 2) < 0)
            orow += raw_height;
        }
        RAW(orow, ocol) = curve[bits >> (10 * c + 2) & 0x3ff];
      }
    }
  maximum = curve[0x3ff];
}

unsigned CLASS pana_bits(int nbits) {
  static uchar buf[0x4000];
  static int vbits;
  int byte;

  if (!nbits)
    return vbits = 0;
  if (!vbits) {
    fread(buf + load_flags, 1, 0x4000 - load_flags, ifp);
    fread(buf, 1, load_flags, ifp);
  }
  vbits = (vbits - nbits) & 0x1ffff;
  byte = vbits >> 3 ^ 0x3ff0;
  return (buf[byte] | buf[byte + 1] << 8) >> (vbits & 7) & ~(-1 << nbits);
}

void CLASS panasonic_load_raw() {
  int row, col, i, j, sh = 0, pred[2], nonz[2];

  pana_bits(0);
  for (row = 0; row < height; row++)
    for (col = 0; col < raw_width; col++) {
      if ((i = col % 14) == 0)
        pred[0] = pred[1] = nonz[0] = nonz[1] = 0;
      if (i % 3 == 2)
        sh = 4 >> (3 - pana_bits(2));
      if (nonz[i & 1]) {
        if ((j = pana_bits(8))) {
          if ((pred[i & 1] -= 0x80 << sh) < 0 || sh == 4)
            pred[i & 1] &= ~(-1 << sh);
          pred[i & 1] += j << sh;
        }
      } else if ((nonz[i & 1] = pana_bits(8)) || i > 11)
        pred[i & 1] = nonz[i & 1] << 4 | pana_bits(4);
      if ((RAW(row, col) = pred[col & 1]) > 4098 && col < width)
        derror();
    }
}

void CLASS olympus_load_raw() {
  ushort huff[4096];
  int row, col, nbits, sign, low, high, i, c, w, n, nw;
  int acarry[2][3], *carry, pred, diff;

  huff[n = 0] = 0xc0c;
  for (i = 12; i--;)
    FORC(2048 >> i) huff[++n] = (i + 1) << 8 | i;
  fseek(ifp, 7, SEEK_CUR);
  getbits(-1);
  for (row = 0; row < height; row++) {
    memset(acarry, 0, sizeof acarry);
    for (col = 0; col < raw_width; col++) {
      carry = acarry[col & 1];
      i = 2 * (carry[2] < 3);
      for (nbits = 2 + i; (ushort)carry[0] >> (nbits + i); nbits++)
        ;
      low = (sign = getbits(3)) & 3;
      sign = sign << 29 >> 31;
      if ((high = getbithuff(12, huff)) == 12)
        high = getbits(16 - nbits) >> 1;
      carry[0] = (high << nbits) | getbits(nbits);
      diff = (carry[0] ^ sign) + carry[1];
      carry[1] = (diff * 3 + carry[1]) >> 5;
      carry[2] = carry[0] > 16 ? 0 : carry[2] + 1;
      if (col >= width)
        continue;
      if (row < 2 && col < 2)
        pred = 0;
      else if (row < 2)
        pred = RAW(row, col - 2);
      else if (col < 2)
        pred = RAW(row - 2, col);
      else {
        w = RAW(row, col - 2);
        n = RAW(row - 2, col);
        nw = RAW(row - 2, col - 2);
        if ((w < nw && nw < n) || (n < nw && nw < w)) {
          if (ABS(w - nw) > 32 || ABS(n - nw) > 32)
            pred = w + n - nw;
          else
            pred = (w + n) >> 1;
        } else
          pred = ABS(w - nw) > ABS(n - nw) ? w : n;
      }
      if ((RAW(row, col) = pred + ((diff << 2) | low)) >> 12)
        derror();
    }
  }
}

void CLASS minolta_rd175_load_raw() {
  uchar pixel[768];
  unsigned irow, box, row, col;

  for (irow = 0; irow < 1481; irow++) {
    if (fread(pixel, 1, 768, ifp) < 768)
      derror();
    box = irow / 82;
    row = irow % 82 * 12 + ((box < 12) ? box | 1 : (box - 12) * 2);
    switch (irow) {
      case 1477:
      case 1479:
        continue;
      case 1476:
        row = 984;
        break;
      case 1480:
        row = 985;
        break;
      case 1478:
        row = 985;
        box = 1;
    }
    if ((box < 12) && (box & 1)) {
      for (col = 0; col < 1533; col++, row ^= 1)
        if (col != 1)
          RAW(row, col) = (col + 1) & 2
                              ? pixel[col / 2 - 1] + pixel[col / 2 + 1]
                              : pixel[col / 2] << 1;
      RAW(row, 1) = pixel[1] << 1;
      RAW(row, 1533) = pixel[765] << 1;
    } else
      for (col = row & 1; col < 1534; col += 2)
        RAW(row, col) = pixel[col / 2] << 1;
  }
  maximum = 0xff << 1;
}

void CLASS quicktake_100_load_raw() {
  uchar pixel[484][644];
  static const short gstep[16] = {-89, -60, -44, -32, -22, -15, -8, -2,
                                  2,   8,   15,  22,  32,  44,  60, 89};
  static const short rstep[6][4] = {{-3, -1, 1, 3},   {-5, -1, 1, 5},
                                    {-8, -2, 2, 8},   {-13, -3, 3, 13},
                                    {-19, -4, 4, 19}, {-28, -6, 6, 28}};
  static const short curve[256] = {
      0,   1,    2,    3,   4,   5,   6,   7,   8,   9,   11,  12,  13,  14,
      15,  16,   17,   18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,
      29,  30,   32,   33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
      44,  45,   46,   47,  48,  49,  50,  51,  53,  54,  55,  56,  57,  58,
      59,  60,   61,   62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,
      74,  75,   76,   77,  78,  79,  80,  81,  82,  83,  84,  86,  88,  90,
      92,  94,   97,   99,  101, 103, 105, 107, 110, 112, 114, 116, 118, 120,
      123, 125,  127,  129, 131, 134, 136, 138, 140, 142, 144, 147, 149, 151,
      153, 155,  158,  160, 162, 164, 166, 168, 171, 173, 175, 177, 179, 181,
      184, 186,  188,  190, 192, 195, 197, 199, 201, 203, 205, 208, 210, 212,
      214, 216,  218,  221, 223, 226, 230, 235, 239, 244, 248, 252, 257, 261,
      265, 270,  274,  278, 283, 287, 291, 296, 300, 305, 309, 313, 318, 322,
      326, 331,  335,  339, 344, 348, 352, 357, 361, 365, 370, 374, 379, 383,
      387, 392,  396,  400, 405, 409, 413, 418, 422, 426, 431, 435, 440, 444,
      448, 453,  457,  461, 466, 470, 474, 479, 483, 487, 492, 496, 500, 508,
      519, 531,  542,  553, 564, 575, 587, 598, 609, 620, 631, 643, 654, 665,
      676, 687,  698,  710, 721, 732, 743, 754, 766, 777, 788, 799, 810, 822,
      833, 844,  855,  866, 878, 889, 900, 911, 922, 933, 945, 956, 967, 978,
      989, 1001, 1012, 1023};
  int rb, row, col, sharp, val = 0;

  getbits(-1);
  memset(pixel, 0x80, sizeof pixel);
  for (row = 2; row < height + 2; row++) {
    for (col = 2 + (row & 1); col < width + 2; col += 2) {
      val = ((pixel[row - 1][col - 1] + 2 * pixel[row - 1][col + 1] +
              pixel[row][col - 2]) >>
             2) +
            gstep[getbits(4)];
      pixel[row][col] = val = LIM(val, 0, 255);
      if (col < 4)
        pixel[row][col - 2] = pixel[row + 1][~row & 1] = val;
      if (row == 2)
        pixel[row - 1][col + 1] = pixel[row - 1][col + 3] = val;
    }
    pixel[row][col] = val;
  }
  for (rb = 0; rb < 2; rb++)
    for (row = 2 + rb; row < height + 2; row += 2)
      for (col = 3 - (row & 1); col < width + 2; col += 2) {
        if (row < 4 || col < 4)
          sharp = 2;
        else {
          val = ABS(pixel[row - 2][col] - pixel[row][col - 2]) +
                ABS(pixel[row - 2][col] - pixel[row - 2][col - 2]) +
                ABS(pixel[row][col - 2] - pixel[row - 2][col - 2]);
          sharp = val < 4
                      ? 0
                      : val < 8
                            ? 1
                            : val < 16 ? 2 : val < 32 ? 3 : val < 48 ? 4 : 5;
        }
        val = ((pixel[row - 2][col] + pixel[row][col - 2]) >> 1) +
              rstep[sharp][getbits(2)];
        pixel[row][col] = val = LIM(val, 0, 255);
        if (row < 4)
          pixel[row - 2][col + 2] = val;
        if (col < 4)
          pixel[row + 2][col - 2] = val;
      }
  for (row = 2; row < height + 2; row++)
    for (col = 3 - (row & 1); col < width + 2; col += 2) {
      val = ((pixel[row][col - 1] + (pixel[row][col] << 2) +
              pixel[row][col + 1]) >>
             1) -
            0x100;
      pixel[row][col] = LIM(val, 0, 255);
    }
  for (row = 0; row < height; row++)
    for (col = 0; col < width; col++)
      RAW(row, col) = curve[pixel[row + 2][col + 2]];
  maximum = 0x3ff;
}

#define radc_token(tree) ((signed char)getbithuff(8, huff[tree]))

#define FORYX             \
  for (y = 1; y < 3; y++) \
    for (x = col + 1; x >= col; x--)

#define PREDICTOR                                \
  (c ? (buf[c][y - 1][x] + buf[c][y][x + 1]) / 2 \
     : (buf[c][y - 1][x + 1] + 2 * buf[c][y - 1][x] + buf[c][y][x + 1]) / 4)

void CLASS kodak_radc_load_raw() {
  static const char src[] = {
      1, 1,   2, 3,   3, 4,   4, 2,   5, 7,   6, 5,   7, 6,   7, 8,   1, 0,
      2, 1,   3, 3,   4, 4,   5, 2,   6, 7,   7, 6,   8, 5,   8, 8,   2, 1,
      2, 3,   3, 0,   3, 2,   3, 4,   4, 6,   5, 5,   6, 7,   6, 8,   2, 0,
      2, 1,   2, 3,   3, 2,   4, 4,   5, 6,   6, 7,   7, 5,   7, 8,   2, 1,
      2, 4,   3, 0,   3, 2,   3, 3,   4, 7,   5, 5,   6, 6,   6, 8,   2, 3,
      3, 1,   3, 2,   3, 4,   3, 5,   3, 6,   4, 7,   5, 0,   5, 8,   2, 3,
      2, 6,   3, 0,   3, 1,   4, 4,   4, 5,   4, 7,   5, 2,   5, 8,   2, 4,
      2, 7,   3, 3,   3, 6,   4, 1,   4, 2,   4, 5,   5, 0,   5, 8,   2, 6,
      3, 1,   3, 3,   3, 5,   3, 7,   3, 8,   4, 0,   5, 2,   5, 4,   2, 0,
      2, 1,   3, 2,   3, 3,   4, 4,   4, 5,   5, 6,   5, 7,   4, 8,   1, 0,
      2, 2,   2, -2,  1, -3,  1, 3,   2, -17, 2, -5,  2, 5,   2, 17,  2, -7,
      2, 2,   2, 9,   2, 18,  2, -18, 2, -9,  2, -2,  2, 7,   2, -28, 2, 28,
      3, -49, 3, -9,  3, 9,   4, 49,  5, -79, 5, 79,  2, -1,  2, 13,  2, 26,
      3, 39,  4, -16, 5, 55,  6, -37, 6, 76,  2, -26, 2, -13, 2, 1,   3, -39,
      4, 16,  5, -55, 6, -76, 6, 37};
  ushort huff[19][256];
  int row, col, tree, nreps, rep, step, i, c, s, r, x, y, val;
  short last[3] = {16, 16, 16}, mul[3], buf[3][3][386];
  static const ushort pt[] = {0,    0,    1280, 1344,  2320,  3616,
                              3328, 8000, 4095, 16383, 65535, 16383};

  for (i = 2; i < 12; i += 2)
    for (c = pt[i - 2]; c <= pt[i]; c++)
      curve[c] = (float)(c - pt[i - 2]) / (pt[i] - pt[i - 2]) *
                     (pt[i + 1] - pt[i - 1]) +
                 pt[i - 1] + 0.5;
  for (s = i = 0; i < sizeof src; i += 2)
    FORC(256 >> src[i])
  ((ushort *)huff)[s++] = src[i] << 8 | (uchar)src[i + 1];
  s = kodak_cbpp == 243 ? 2 : 3;
  FORC(256) huff[18][c] = (8 - s) << 8 | c >> s << s | 1 << (s - 1);
  getbits(-1);
  for (i = 0; i < sizeof(buf) / sizeof(short); i++)
    ((short *)buf)[i] = 2048;
  for (row = 0; row < height; row += 4) {
    FORC3 mul[c] = getbits(6);
    FORC3 {
      val = ((0x1000000 / last[c] + 0x7ff) >> 12) * mul[c];
      s = val > 65564 ? 10 : 12;
      x = ~(-1 << (s - 1));
      val <<= 12 - s;
      for (i = 0; i < sizeof(buf[0]) / sizeof(short); i++)
        ((short *)buf[c])[i] = (((short *)buf[c])[i] * val + x) >> s;
      last[c] = mul[c];
      for (r = 0; r <= !c; r++) {
        buf[c][1][width / 2] = buf[c][2][width / 2] = mul[c] << 7;
        for (tree = 1, col = width / 2; col > 0;) {
          if ((tree = radc_token(tree))) {
            col -= 2;
            if (tree == 8)
              FORYX buf[c][y][x] = (uchar)radc_token(18) * mul[c];
            else
              FORYX buf[c][y][x] = radc_token(tree + 10) * 16 + PREDICTOR;
          } else
            do {
              nreps = (col > 2) ? radc_token(9) + 1 : 1;
              for (rep = 0; rep < 8 && rep < nreps && col > 0; rep++) {
                col -= 2;
                FORYX buf[c][y][x] = PREDICTOR;
                if (rep & 1) {
                  step = radc_token(10) << 4;
                  FORYX buf[c][y][x] += step;
                }
              }
            } while (nreps == 9);
        }
        for (y = 0; y < 2; y++)
          for (x = 0; x < width / 2; x++) {
            val = (buf[c][y + 1][x] << 4) / mul[c];
            if (val < 0)
              val = 0;
            if (c)
              RAW(row + y * 2 + c - 1, x * 2 + 2 - c) = val;
            else
              RAW(row + r * 2 + y, x * 2 + y) = val;
          }
        memcpy(buf[c][0] + !c, buf[c][2], sizeof buf[c][0] - 2 * !c);
      }
    }
    for (y = row; y < row + 4; y++)
      for (x = 0; x < width; x++)
        if ((x + y) & 1) {
          r = x ? x - 1 : x + 1;
          s = x + 1 < width ? x + 1 : x - 1;
          val = (RAW(y, x) - 2048) * 2 + (RAW(y, r) + RAW(y, s)) / 2;
          if (val < 0)
            val = 0;
          RAW(y, x) = val;
        }
  }
  for (i = 0; i < height * width; i++)
    raw_image[i] = curve[raw_image[i]];
  maximum = 0x3fff;
}

#undef FORYX
#undef PREDICTOR

#ifdef NO_JPEG
void CLASS kodak_jpeg_load_raw() {}
void CLASS lossy_dng_load_raw() {}
#else

METHODDEF(boolean)
fill_input_buffer(j_decompress_ptr cinfo) {
  static uchar jpeg_buffer[4096];
  size_t nbytes;

  nbytes = fread(jpeg_buffer, 1, 4096, ifp);
  swab(jpeg_buffer, jpeg_buffer, nbytes);
  cinfo->src->next_input_byte = jpeg_buffer;
  cinfo->src->bytes_in_buffer = nbytes;
  return TRUE;
}

void CLASS kodak_jpeg_load_raw() {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPARRAY buf;
  JSAMPLE(*pixel)[3];
  int row, col;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, ifp);
  cinfo.src->fill_input_buffer = fill_input_buffer;
  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);
  if ((cinfo.output_width != width) || (cinfo.output_height * 2 != height) ||
      (cinfo.output_components != 3)) {
    fprintf(stderr, _("%s: incorrect JPEG dimensions\n"), ifname);
    jpeg_destroy_decompress(&cinfo);
    longjmp(failure, 3);
  }
  buf = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, width * 3,
                                   1);

  while (cinfo.output_scanline < cinfo.output_height) {
    row = cinfo.output_scanline * 2;
    jpeg_read_scanlines(&cinfo, buf, 1);
    pixel = (JSAMPLE(*)[3])buf[0];
    for (col = 0; col < width; col += 2) {
      RAW(row + 0, col + 0) = pixel[col + 0][1] << 1;
      RAW(row + 1, col + 1) = pixel[col + 1][1] << 1;
      RAW(row + 0, col + 1) = pixel[col][0] + pixel[col + 1][0];
      RAW(row + 1, col + 0) = pixel[col][2] + pixel[col + 1][2];
    }
  }
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  maximum = 0xff << 1;
}

void CLASS gamma_curve(double pwr, double ts, int mode, int imax);

void CLASS lossy_dng_load_raw() {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPARRAY buf;
  JSAMPLE(*pixel)[3];
  unsigned sorder = order, ntags, opcode, deg, i, j, c;
  unsigned save = data_offset - 4, trow = 0, tcol = 0, row, col;
  ushort cur[3][256];
  double coeff[9], tot;

  if (meta_offset) {
    fseek(ifp, meta_offset, SEEK_SET);
    order = 0x4d4d;
    ntags = get4();
    while (ntags--) {
      opcode = get4();
      get4();
      get4();
      if (opcode != 8) {
        fseek(ifp, get4(), SEEK_CUR);
        continue;
      }
      fseek(ifp, 20, SEEK_CUR);
      if ((c = get4()) > 2)
        break;
      fseek(ifp, 12, SEEK_CUR);
      if ((deg = get4()) > 8)
        break;
      for (i = 0; i <= deg && i < 9; i++)
        coeff[i] = getreal(12);
      for (i = 0; i < 256; i++) {
        for (tot = j = 0; j <= deg; j++)
          tot += coeff[j] * pow(i / 255.0, j);
        cur[c][i] = tot * 0xffff;
      }
    }
    order = sorder;
  } else {
    gamma_curve(1 / 2.4, 12.92, 1, 255);
    FORC3 memcpy(cur[c], curve, sizeof cur[0]);
  }
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  while (trow < raw_height) {
    fseek(ifp, save += 4, SEEK_SET);
    if (tile_length < INT_MAX)
      fseek(ifp, get4(), SEEK_SET);
    jpeg_stdio_src(&cinfo, ifp);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);
    buf = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE,
                                     cinfo.output_width * 3, 1);
    while (cinfo.output_scanline < cinfo.output_height &&
           (row = trow + cinfo.output_scanline) < height) {
      jpeg_read_scanlines(&cinfo, buf, 1);
      pixel = (JSAMPLE(*)[3])buf[0];
      for (col = 0; col < cinfo.output_width && tcol + col < width; col++) {
        FORC3 image[row * width + tcol + col][c] = cur[c][pixel[col][c]];
      }
    }
    jpeg_abort_decompress(&cinfo);
    if ((tcol += tile_width) >= raw_width)
      trow += tile_length + (tcol = 0);
  }
  jpeg_destroy_decompress(&cinfo);
  maximum = 0xffff;
}
#endif

void CLASS kodak_dc120_load_raw() {
  static const int mul[4] = {162, 192, 187, 92};
  static const int add[4] = {0, 636, 424, 212};
  uchar pixel[848];
  int row, shift, col;

  for (row = 0; row < height; row++) {
    if (fread(pixel, 1, 848, ifp) < 848)
      derror();
    shift = row * mul[row & 3] + add[row & 3];
    for (col = 0; col < width; col++)
      RAW(row, col) = (ushort)pixel[(col + shift) % 848];
  }
  maximum = 0xff;
}

void CLASS eight_bit_load_raw() {
  uchar *pixel;
  unsigned row, col;

  pixel = (uchar *)calloc(raw_width, sizeof *pixel);
  merror(pixel, "eight_bit_load_raw()");
  for (row = 0; row < raw_height; row++) {
    if (fread(pixel, 1, raw_width, ifp) < raw_width)
      derror();
    for (col = 0; col < raw_width; col++)
      RAW(row, col) = curve[pixel[col]];
  }
  free(pixel);
  maximum = curve[0xff];
}

void CLASS kodak_c330_load_raw() {
  uchar *pixel;
  int row, col, y, cb, cr, rgb[3], c;

  pixel = (uchar *)calloc(raw_width, 2 * sizeof *pixel);
  merror(pixel, "kodak_c330_load_raw()");
  for (row = 0; row < height; row++) {
    if (fread(pixel, raw_width, 2, ifp) < 2)
      derror();
    if (load_flags && (row & 31) == 31)
      fseek(ifp, raw_width * 32, SEEK_CUR);
    for (col = 0; col < width; col++) {
      y = pixel[col * 2];
      cb = pixel[(col * 2 & -4) | 1] - 128;
      cr = pixel[(col * 2 & -4) | 3] - 128;
      rgb[1] = y - ((cb + cr + 2) >> 2);
      rgb[2] = rgb[1] + cb;
      rgb[0] = rgb[1] + cr;
      FORC3 image[row * width + col][c] = curve[LIM(rgb[c], 0, 255)];
    }
  }
  free(pixel);
  maximum = curve[0xff];
}

void CLASS kodak_c603_load_raw() {
  uchar *pixel;
  int row, col, y, cb, cr, rgb[3], c;

  pixel = (uchar *)calloc(raw_width, 3 * sizeof *pixel);
  merror(pixel, "kodak_c603_load_raw()");
  for (row = 0; row < height; row++) {
    if (~row & 1)
      if (fread(pixel, raw_width, 3, ifp) < 3)
        derror();
    for (col = 0; col < width; col++) {
      y = pixel[width * 2 * (row & 1) + col];
      cb = pixel[width + (col & -2)] - 128;
      cr = pixel[width + (col & -2) + 1] - 128;
      rgb[1] = y - ((cb + cr + 2) >> 2);
      rgb[2] = rgb[1] + cb;
      rgb[0] = rgb[1] + cr;
      FORC3 image[row * width + col][c] = curve[LIM(rgb[c], 0, 255)];
    }
  }
  free(pixel);
  maximum = curve[0xff];
}

void CLASS kodak_262_load_raw() {
  static const uchar kodak_tree[2][26] = {
      {0, 1, 5, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      {0, 3, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};
  ushort *huff[2];
  uchar *pixel;
  int *strip, ns, c, row, col, chess, pi = 0, pi1, pi2, pred, val;

  FORC(2) huff[c] = make_decoder(kodak_tree[c]);
  ns = (raw_height + 63) >> 5;
  pixel = (uchar *)malloc(raw_width * 32 + ns * 4);
  merror(pixel, "kodak_262_load_raw()");
  strip = (int *)(pixel + raw_width * 32);
  order = 0x4d4d;
  FORC(ns) strip[c] = get4();
  for (row = 0; row < raw_height; row++) {
    if ((row & 31) == 0) {
      fseek(ifp, strip[row >> 5], SEEK_SET);
      getbits(-1);
      pi = 0;
    }
    for (col = 0; col < raw_width; col++) {
      chess = (row + col) & 1;
      pi1 = chess ? pi - 2 : pi - raw_width - 1;
      pi2 = chess ? pi - 2 * raw_width : pi - raw_width + 1;
      if (col <= chess)
        pi1 = -1;
      if (pi1 < 0)
        pi1 = pi2;
      if (pi2 < 0)
        pi2 = pi1;
      if (pi1 < 0 && col > 1)
        pi1 = pi2 = pi - 2;
      pred = (pi1 < 0) ? 0 : (pixel[pi1] + pixel[pi2]) >> 1;
      pixel[pi] = val = pred + ljpeg_diff(huff[chess]);
      if (val >> 8)
        derror();
      val = curve[pixel[pi++]];
      RAW(row, col) = val;
    }
  }
  free(pixel);
  FORC(2) free(huff[c]);
}

int CLASS kodak_65000_decode(short *out, int bsize) {
  uchar c, blen[768];
  ushort raw[6];
  INT64 bitbuf = 0;
  int save, bits = 0, i, j, len, diff;

  save = ftell(ifp);
  bsize = (bsize + 3) & -4;
  for (i = 0; i < bsize; i += 2) {
    c = fgetc(ifp);
    if ((blen[i] = c & 15) > 12 || (blen[i + 1] = c >> 4) > 12) {
      fseek(ifp, save, SEEK_SET);
      for (i = 0; i < bsize; i += 8) {
        read_shorts(raw, 6);
        out[i] = raw[0] >> 12 << 8 | raw[2] >> 12 << 4 | raw[4] >> 12;
        out[i + 1] = raw[1] >> 12 << 8 | raw[3] >> 12 << 4 | raw[5] >> 12;
        for (j = 0; j < 6; j++)
          out[i + 2 + j] = raw[j] & 0xfff;
      }
      return 1;
    }
  }
  if ((bsize & 7) == 4) {
    bitbuf = fgetc(ifp) << 8;
    bitbuf += fgetc(ifp);
    bits = 16;
  }
  for (i = 0; i < bsize; i++) {
    len = blen[i];
    if (bits < len) {
      for (j = 0; j < 32; j += 8)
        bitbuf += (INT64)fgetc(ifp) << (bits + (j ^ 8));
      bits += 32;
    }
    diff = bitbuf & (0xffff >> (16 - len));
    bitbuf >>= len;
    bits -= len;
    if ((diff & (1 << (len - 1))) == 0)
      diff -= (1 << len) - 1;
    out[i] = diff;
  }
  return 0;
}

void CLASS kodak_65000_load_raw() {
  short buf[256];
  int row, col, len, pred[2], ret, i;

  for (row = 0; row < height; row++)
    for (col = 0; col < width; col += 256) {
      pred[0] = pred[1] = 0;
      len = MIN(256, width - col);
      ret = kodak_65000_decode(buf, len);
      for (i = 0; i < len; i++)
        if ((RAW(row, col + i) =
                 curve[ret ? buf[i] : (pred[i & 1] += buf[i])]) >>
            12)
          derror();
    }
}

void CLASS kodak_ycbcr_load_raw() {
  short buf[384], *bp;
  int row, col, len, c, i, j, k, y[2][2], cb, cr, rgb[3];
  ushort *ip;

  if (!image)
    return;
  for (row = 0; row < height; row += 2)
    for (col = 0; col < width; col += 128) {
      len = MIN(128, width - col);
      kodak_65000_decode(buf, len * 3);
      y[0][1] = y[1][1] = cb = cr = 0;
      for (bp = buf, i = 0; i < len; i += 2, bp += 2) {
        cb += bp[4];
        cr += bp[5];
        rgb[1] = -((cb + cr + 2) >> 2);
        rgb[2] = rgb[1] + cb;
        rgb[0] = rgb[1] + cr;
        for (j = 0; j < 2; j++)
          for (k = 0; k < 2; k++) {
            if ((y[j][k] = y[j][k ^ 1] + *bp++) >> 10)
              derror();
            ip = image[(row + j) * width + col + i + k];
            FORC3 ip[c] = curve[LIM(y[j][k] + rgb[c], 0, 0xfff)];
          }
      }
    }
}

void CLASS kodak_rgb_load_raw() {
  short buf[768], *bp;
  int row, col, len, c, i, rgb[3];
  ushort *ip = image[0];

  for (row = 0; row < height; row++)
    for (col = 0; col < width; col += 256) {
      len = MIN(256, width - col);
      kodak_65000_decode(buf, len * 3);
      memset(rgb, 0, sizeof rgb);
      for (bp = buf, i = 0; i < len; i++, ip += 4)
        FORC3 if ((ip[c] = rgb[c] += *bp++) >> 12) derror();
    }
}

void CLASS kodak_thumb_load_raw() {
  int row, col;
  colors = thumb_misc >> 5;
  for (row = 0; row < height; row++)
    for (col = 0; col < width; col++)
      read_shorts(image[row * width + col], colors);
  maximum = (1 << (thumb_misc & 31)) - 1;
}

void CLASS sony_decrypt(unsigned *data, int len, int start, int key) {
  static unsigned pad[128], p;

  if (start) {
    for (p = 0; p < 4; p++)
      pad[p] = key = key * 48828125 + 1;
    pad[3] = pad[3] << 1 | (pad[0] ^ pad[2]) >> 31;
    for (p = 4; p < 127; p++)
      pad[p] = (pad[p - 4] ^ pad[p - 2]) << 1 | (pad[p - 3] ^ pad[p - 1]) >> 31;
    for (p = 0; p < 127; p++)
      pad[p] = htonl(pad[p]);
  }
  while (len-- && p++)
    *data++ ^= pad[(p - 1) & 127] = pad[p & 127] ^ pad[(p + 64) & 127];
}

void CLASS sony_load_raw() {
  uchar head[40];
  ushort *pixel;
  unsigned i, key, row, col;

  fseek(ifp, 200896, SEEK_SET);
  fseek(ifp, (unsigned)fgetc(ifp) * 4 - 1, SEEK_CUR);
  order = 0x4d4d;
  key = get4();
  fseek(ifp, 164600, SEEK_SET);
  fread(head, 1, 40, ifp);
  sony_decrypt((unsigned *)head, 10, 1, key);
  for (i = 26; i-- > 22;)
    key = key << 8 | head[i];
  fseek(ifp, data_offset, SEEK_SET);
  for (row = 0; row < raw_height; row++) {
    pixel = raw_image + row * raw_width;
    if (fread(pixel, 2, raw_width, ifp) < raw_width)
      derror();
    sony_decrypt((unsigned *)pixel, raw_width / 2, !row, key);
    for (col = 0; col < raw_width; col++)
      if ((pixel[col] = ntohs(pixel[col])) >> 14)
        derror();
  }
  maximum = 0x3ff0;
}

void CLASS sony_arw_load_raw() {
  ushort huff[32770];
  static const ushort tab[18] = {0xf11, 0xf10, 0xe0f, 0xd0e, 0xc0d, 0xb0c,
                                 0xa0b, 0x90a, 0x809, 0x708, 0x607, 0x506,
                                 0x405, 0x304, 0x303, 0x300, 0x202, 0x201};
  int i, c, n, col, row, sum = 0;

  huff[0] = 15;
  for (n = i = 0; i < 18; i++)
    FORC(32768 >> (tab[i] >> 8)) huff[++n] = tab[i];
  getbits(-1);
  for (col = raw_width; col--;)
    for (row = 0; row < raw_height + 1; row += 2) {
      if (row == raw_height)
        row = 1;
      if ((sum += ljpeg_diff(huff)) >> 12)
        derror();
      if (row < height)
        RAW(row, col) = sum;
    }
}

void CLASS sony_arw2_load_raw() {
  uchar *data, *dp;
  ushort pix[16];
  int row, col, val, max, min, imax, imin, sh, bit, i;

  data = (uchar *)malloc(raw_width + 1);
  merror(data, "sony_arw2_load_raw()");
  for (row = 0; row < height; row++) {
    fread(data, 1, raw_width, ifp);
    for (dp = data, col = 0; col < raw_width - 30; dp += 16) {
      max = 0x7ff & (val = sget4(dp));
      min = 0x7ff & val >> 11;
      imax = 0x0f & val >> 22;
      imin = 0x0f & val >> 26;
      for (sh = 0; sh < 4 && 0x80 << sh <= max - min; sh++)
        ;
      for (bit = 30, i = 0; i < 16; i++)
        if (i == imax)
          pix[i] = max;
        else if (i == imin)
          pix[i] = min;
        else {
          pix[i] = ((sget2(dp + (bit >> 3)) >> (bit & 7) & 0x7f) << sh) + min;
          if (pix[i] > 0x7ff)
            pix[i] = 0x7ff;
          bit += 7;
        }
      for (i = 0; i < 16; i++, col += 2)
        RAW(row, col) = curve[pix[i] << 1] >> 2;
      col -= col & 1 ? 1 : 31;
    }
  }
  free(data);
}

void CLASS samsung_load_raw() {
  int row, col, c, i, dir, op[4], len[4];

  order = 0x4949;
  for (row = 0; row < raw_height; row++) {
    fseek(ifp, strip_offset + row * 4, SEEK_SET);
    fseek(ifp, data_offset + get4(), SEEK_SET);
    ph1_bits(-1);
    FORC4 len[c] = row < 2 ? 7 : 4;
    for (col = 0; col < raw_width; col += 16) {
      dir = ph1_bits(1);
      FORC4 op[c] = ph1_bits(2);
      FORC4 switch (op[c]) {
        case 3:
          len[c] = ph1_bits(4);
          break;
        case 2:
          len[c]--;
          break;
        case 1:
          len[c]++;
      }
      for (c = 0; c < 16; c += 2) {
        i = len[((c & 1) << 1) | (c >> 3)];
        RAW(row, col + c) = ((signed)ph1_bits(i) << (32 - i) >> (32 - i)) +
                            (dir ? RAW(row + (~c | -2), col + c)
                                 : col ? RAW(row, col + (c | -2)) : 128);
        if (c == 14)
          c = -1;
      }
    }
  }
  for (row = 0; row < raw_height - 1; row += 2)
    for (col = 0; col < raw_width - 1; col += 2)
      SWAP(RAW(row, col + 1), RAW(row + 1, col));
}

void CLASS samsung2_load_raw() {
  static const ushort tab[14] = {0x304, 0x307, 0x206, 0x205, 0x403,
                                 0x600, 0x709, 0x80a, 0x90b, 0xa0c,
                                 0xa0d, 0x501, 0x408, 0x402};
  ushort huff[1026], vpred[2][2] = {{0, 0}, {0, 0}}, hpred[2];
  int i, c, n, row, col, diff;

  huff[0] = 10;
  for (n = i = 0; i < 14; i++)
    FORC(1024 >> (tab[i] >> 8)) huff[++n] = tab[i];
  getbits(-1);
  for (row = 0; row < raw_height; row++)
    for (col = 0; col < raw_width; col++) {
      diff = ljpeg_diff(huff);
      if (col < 2)
        hpred[col] = vpred[row & 1][col] += diff;
      else
        hpred[col & 1] += diff;
      RAW(row, col) = hpred[col & 1];
      if (hpred[col & 1] >> tiff_bps)
        derror();
    }
}

void CLASS samsung3_load_raw() {
  int opt, init, mag, pmode, row, tab, col, pred, diff, i, c;
  ushort lent[3][2], len[4], *prow[2];

  order = 0x4949;
  fseek(ifp, 9, SEEK_CUR);
  opt = fgetc(ifp);
  init = (get2(), get2());
  for (row = 0; row < raw_height; row++) {
    fseek(ifp, (data_offset - ftell(ifp)) & 15, SEEK_CUR);
    ph1_bits(-1);
    mag = 0;
    pmode = 7;
    FORC(6)((ushort *)lent)[c] = row < 2 ? 7 : 4;
    prow[row & 1] = &RAW(row - 1, 1 - ((row & 1) << 1));  // green
    prow[~row & 1] = &RAW(row - 2, 0);                    // red and blue
    for (tab = 0; tab + 15 < raw_width; tab += 16) {
      if (~opt & 4 && !(tab & 63)) {
        i = ph1_bits(2);
        mag = i < 3 ? mag - '2' + "204"[i] : ph1_bits(12);
      }
      if (opt & 2)
        pmode = 7 - 4 * ph1_bits(1);
      else if (!ph1_bits(1))
        pmode = ph1_bits(3);
      if (opt & 1 || !ph1_bits(1)) {
        FORC4 len[c] = ph1_bits(2);
        FORC4 {
          i = ((row & 1) << 1 | (c & 1)) % 3;
          len[c] = len[c] < 3 ? lent[i][0] - '1' + "120"[len[c]] : ph1_bits(4);
          lent[i][0] = lent[i][1];
          lent[i][1] = len[c];
        }
      }
      FORC(16) {
        col = tab + (((c & 7) << 1) ^ (c >> 3) ^ (row & 1));
        pred = (pmode == 7 || row < 2)
                   ? (tab ? RAW(row, tab - 2 + (col & 1)) : init)
                   : (prow[col & 1][col - '4' + "0224468"[pmode]] +
                      prow[col & 1][col - '4' + "0244668"[pmode]] + 1) >>
                         1;
        diff = ph1_bits(i = len[c >> 2]);
        if (diff >> (i - 1))
          diff -= 1 << i;
        diff = diff * (mag * 2 + 1) + mag;
        RAW(row, col) = pred + diff;
      }
    }
  }
}

#define HOLE(row) ((holes >> (((row)-raw_height) & 7)) & 1)

/* Kudos to Rich Taylor for figuring out SMaL's compression algorithm. */
void CLASS smal_decode_segment(unsigned seg[2][2], int holes) {
  uchar hist[3][13] = {{7, 7, 0, 0, 63, 55, 47, 39, 31, 23, 15, 7, 0},
                       {7, 7, 0, 0, 63, 55, 47, 39, 31, 23, 15, 7, 0},
                       {3, 3, 0, 0, 63, 47, 31, 15, 0}};
  int low, high = 0xff, carry = 0, nbits = 8;
  int pix, s, count, bin, next, i, sym[3];
  uchar diff, pred[] = {0, 0};
  ushort data = 0, range = 0;

  fseek(ifp, seg[0][1] + 1, SEEK_SET);
  getbits(-1);
  if (seg[1][0] > raw_width * raw_height)
    seg[1][0] = raw_width * raw_height;
  for (pix = seg[0][0]; pix < seg[1][0]; pix++) {
    for (s = 0; s < 3; s++) {
      data = data << nbits | getbits(nbits);
      if (carry < 0)
        carry = (nbits += carry + 1) < 1 ? nbits - 1 : 0;
      while (--nbits >= 0)
        if ((data >> nbits & 0xff) == 0xff)
          break;
      if (nbits > 0)
        data = ((data & ((1 << (nbits - 1)) - 1)) << 1) |
               ((data + (((data & (1 << (nbits - 1)))) << 1)) & (-1 << nbits));
      if (nbits >= 0) {
        data += getbits(1);
        carry = nbits - 8;
      }
      count = ((((data - range + 1) & 0xffff) << 2) - 1) / (high >> 4);
      for (bin = 0; hist[s][bin + 5] > count; bin++)
        ;
      low = hist[s][bin + 5] * (high >> 4) >> 2;
      if (bin)
        high = hist[s][bin + 4] * (high >> 4) >> 2;
      high -= low;
      for (nbits = 0; high << nbits < 128; nbits++)
        ;
      range = (range + low) << nbits;
      high <<= nbits;
      next = hist[s][1];
      if (++hist[s][2] > hist[s][3]) {
        next = (next + 1) & hist[s][0];
        hist[s][3] = (hist[s][next + 4] - hist[s][next + 5]) >> 2;
        hist[s][2] = 1;
      }
      if (hist[s][hist[s][1] + 4] - hist[s][hist[s][1] + 5] > 1) {
        if (bin < hist[s][1])
          for (i = bin; i < hist[s][1]; i++)
            hist[s][i + 5]--;
        else if (next <= bin)
          for (i = hist[s][1]; i < bin; i++)
            hist[s][i + 5]++;
      }
      hist[s][1] = next;
      sym[s] = bin;
    }
    diff = sym[2] << 5 | sym[1] << 2 | (sym[0] & 3);
    if (sym[0] & 4)
      diff = diff ? -diff : 0x80;
    if (ftell(ifp) + 12 >= seg[1][1])
      diff = 0;
    raw_image[pix] = pred[pix & 1] += diff;
    if (!(pix & 1) && HOLE(pix / raw_width))
      pix += 2;
  }
  maximum = 0xff;
}

void CLASS smal_v6_load_raw() {
  unsigned seg[2][2];

  fseek(ifp, 16, SEEK_SET);
  seg[0][0] = 0;
  seg[0][1] = get2();
  seg[1][0] = raw_width * raw_height;
  seg[1][1] = INT_MAX;
  smal_decode_segment(seg, 0);
}

int CLASS median4(int *p) {
  int min, max, sum, i;

  min = max = sum = p[0];
  for (i = 1; i < 4; i++) {
    sum += p[i];
    if (min > p[i])
      min = p[i];
    if (max < p[i])
      max = p[i];
  }
  return (sum - min - max) >> 1;
}

void CLASS fill_holes(int holes) {
  int row, col, val[4];

  for (row = 2; row < height - 2; row++) {
    if (!HOLE(row))
      continue;
    for (col = 1; col < width - 1; col += 4) {
      val[0] = RAW(row - 1, col - 1);
      val[1] = RAW(row - 1, col + 1);
      val[2] = RAW(row + 1, col - 1);
      val[3] = RAW(row + 1, col + 1);
      RAW(row, col) = median4(val);
    }
    for (col = 2; col < width - 2; col += 4)
      if (HOLE(row - 2) || HOLE(row + 2))
        RAW(row, col) = (RAW(row, col - 2) + RAW(row, col + 2)) >> 1;
      else {
        val[0] = RAW(row, col - 2);
        val[1] = RAW(row, col + 2);
        val[2] = RAW(row - 2, col);
        val[3] = RAW(row + 2, col);
        RAW(row, col) = median4(val);
      }
  }
}

void CLASS smal_v9_load_raw() {
  unsigned seg[256][2], offset, nseg, holes, i;

  fseek(ifp, 67, SEEK_SET);
  offset = get4();
  nseg = (uchar)fgetc(ifp);
  fseek(ifp, offset, SEEK_SET);
  for (i = 0; i < nseg * 2; i++)
    ((unsigned *)seg)[i] = get4() + data_offset * (i & 1);
  fseek(ifp, 78, SEEK_SET);
  holes = fgetc(ifp);
  fseek(ifp, 88, SEEK_SET);
  seg[nseg][0] = raw_height * raw_width;
  seg[nseg][1] = get4() + data_offset;
  for (i = 0; i < nseg; i++)
    smal_decode_segment(seg + i, holes);
  if (holes)
    fill_holes(holes);
}

void CLASS redcine_load_raw() {
#ifndef NO_JASPER
  int c, row, col;
  jas_stream_t *in;
  jas_image_t *jimg;
  jas_matrix_t *jmat;
  jas_seqent_t *data;
  ushort *img, *pix;

  jas_init();
  in = jas_stream_fopen(ifname, "rb");
  jas_stream_seek(in, data_offset + 20, SEEK_SET);
  jimg = jas_image_decode(in, -1, 0);
  if (!jimg)
    longjmp(failure, 3);
  jmat = jas_matrix_create(height / 2, width / 2);
  merror(jmat, "redcine_load_raw()");
  img = (ushort *)calloc((height + 2), (width + 2) * 2);
  merror(img, "redcine_load_raw()");
  FORC4 {
    jas_image_readcmpt(jimg, c, 0, 0, width / 2, height / 2, jmat);
    data = jas_matrix_getref(jmat, 0, 0);
    for (row = c >> 1; row < height; row += 2)
      for (col = c & 1; col < width; col += 2)
        img[(row + 1) * (width + 2) + col + 1] =
            data[(row / 2) * (width / 2) + col / 2];
  }
  for (col = 1; col <= width; col++) {
    img[col] = img[2 * (width + 2) + col];
    img[(height + 1) * (width + 2) + col] =
        img[(height - 1) * (width + 2) + col];
  }
  for (row = 0; row < height + 2; row++) {
    img[row * (width + 2)] = img[row * (width + 2) + 2];
    img[(row + 1) * (width + 2) - 1] = img[(row + 1) * (width + 2) - 3];
  }
  for (row = 1; row <= height; row++) {
    pix = img + row * (width + 2) + (col = 1 + (FC(row, 1) & 1));
    for (; col <= width; col += 2, pix += 2) {
      c = (((pix[0] - 0x800) << 3) + pix[-(width + 2)] + pix[width + 2] +
           pix[-1] + pix[1]) >>
          2;
      pix[0] = LIM(c, 0, 4095);
    }
  }
  for (row = 0; row < height; row++)
    for (col = 0; col < width; col++)
      RAW(row, col) = curve[img[(row + 1) * (width + 2) + col + 1]];
  free(img);
  jas_matrix_destroy(jmat);
  jas_image_destroy(jimg);
  jas_stream_close(in);
#endif
}

/* RESTRICTED code starts here */

void CLASS foveon_decoder(unsigned size, unsigned code) {
  static unsigned huff[1024];
  struct decode *cur;
  int i, len;

  if (!code) {
    for (i = 0; i < size; i++)
      huff[i] = get4();
    memset(first_decode, 0, sizeof first_decode);
    free_decode = first_decode;
  }
  cur = free_decode++;
  if (free_decode > first_decode + 2048) {
    fprintf(stderr, _("%s: decoder table overflow\n"), ifname);
    longjmp(failure, 2);
  }
  if (code)
    for (i = 0; i < size; i++)
      if (huff[i] == code) {
        cur->leaf = i;
        return;
      }
  if ((len = code >> 27) > 26)
    return;
  code = (len + 1) << 27 | (code & 0x3ffffff) << 1;

  cur->branch[0] = free_decode;
  foveon_decoder(size, code);
  cur->branch[1] = free_decode;
  foveon_decoder(size, code + 1);
}

void CLASS foveon_thumb() {
  unsigned bwide, row, col, bitbuf = 0, bit = 1, c, i;
  char *buf;
  struct decode *dindex;
  short pred[3];

  bwide = get4();
  fprintf(ofp, "P6\n%d %d\n255\n", thumb_width, thumb_height);
  if (bwide > 0) {
    if (bwide < thumb_width * 3)
      return;
    buf = (char *)malloc(bwide);
    merror(buf, "foveon_thumb()");
    for (row = 0; row < thumb_height; row++) {
      fread(buf, 1, bwide, ifp);
      fwrite(buf, 3, thumb_width, ofp);
    }
    free(buf);
    return;
  }
  foveon_decoder(256, 0);

  for (row = 0; row < thumb_height; row++) {
    memset(pred, 0, sizeof pred);
    if (!bit)
      get4();
    for (bit = col = 0; col < thumb_width; col++)
      FORC3 {
        for (dindex = first_decode; dindex->branch[0];) {
          if ((bit = (bit - 1) & 31) == 31)
            for (i = 0; i < 4; i++)
              bitbuf = (bitbuf << 8) + fgetc(ifp);
          dindex = dindex->branch[bitbuf >> bit & 1];
        }
        pred[c] += dindex->leaf;
        fputc(pred[c], ofp);
      }
  }
}

void CLASS foveon_sd_load_raw() {
  struct decode *dindex;
  short diff[1024];
  unsigned bitbuf = 0;
  int pred[3], row, col, bit = -1, c, i;

  read_shorts((ushort *)diff, 1024);
  if (!load_flags)
    foveon_decoder(1024, 0);

  for (row = 0; row < height; row++) {
    memset(pred, 0, sizeof pred);
    if (!bit && !load_flags && atoi(model + 2) < 14)
      get4();
    for (col = bit = 0; col < width; col++) {
      if (load_flags) {
        bitbuf = get4();
        FORC3 pred[2 - c] += diff[bitbuf >> c * 10 & 0x3ff];
      } else
        FORC3 {
          for (dindex = first_decode; dindex->branch[0];) {
            if ((bit = (bit - 1) & 31) == 31)
              for (i = 0; i < 4; i++)
                bitbuf = (bitbuf << 8) + fgetc(ifp);
            dindex = dindex->branch[bitbuf >> bit & 1];
          }
          pred[c] += diff[dindex->leaf];
          if (pred[c] >> 16 && ~pred[c] >> 16)
            derror();
        }
      FORC3 image[row * width + col][c] = pred[c];
    }
  }
}

void CLASS foveon_huff(ushort *huff) {
  int i, j, clen, code;

  huff[0] = 8;
  for (i = 0; i < 13; i++) {
    clen = getc(ifp);
    code = getc(ifp);
    for (j = 0; j<256>> clen;)
      huff[code + ++j] = clen << 8 | i;
  }
  get2();
}

void CLASS foveon_dp_load_raw() {
  unsigned c, roff[4], row, col, diff;
  ushort huff[512], vpred[2][2], hpred[2];

  fseek(ifp, 8, SEEK_CUR);
  foveon_huff(huff);
  roff[0] = 48;
  FORC3 roff[c + 1] = -(-(roff[c] + get4()) & -16);
  FORC3 {
    fseek(ifp, data_offset + roff[c], SEEK_SET);
    getbits(-1);
    vpred[0][0] = vpred[0][1] = vpred[1][0] = vpred[1][1] = 512;
    for (row = 0; row < height; row++) {
      for (col = 0; col < width; col++) {
        diff = ljpeg_diff(huff);
        if (col < 2)
          hpred[col] = vpred[row & 1][col] += diff;
        else
          hpred[col & 1] += diff;
        image[row * width + col][c] = hpred[col & 1];
      }
    }
  }
}

void CLASS foveon_load_camf() {
  unsigned type, wide, high, i, j, row, col, diff;
  ushort huff[258], vpred[2][2] = {{512, 512}, {512, 512}}, hpred[2];

  fseek(ifp, meta_offset, SEEK_SET);
  type = get4();
  get4();
  get4();
  wide = get4();
  high = get4();
  if (type == 2) {
    fread(meta_data, 1, meta_length, ifp);
    for (i = 0; i < meta_length; i++) {
      high = (high * 1597 + 51749) % 244944;
      wide = high * (INT64)301593171 >> 24;
      meta_data[i] ^= ((((high << 8) - wide) >> 1) + wide) >> 17;
    }
  } else if (type == 4) {
    free(meta_data);
    meta_data = (char *)malloc(meta_length = wide * high * 3 / 2);
    merror(meta_data, "foveon_load_camf()");
    foveon_huff(huff);
    get4();
    getbits(-1);
    for (j = row = 0; row < high; row++) {
      for (col = 0; col < wide; col++) {
        diff = ljpeg_diff(huff);
        if (col < 2)
          hpred[col] = vpred[row & 1][col] += diff;
        else
          hpred[col & 1] += diff;
        if (col & 1) {
          meta_data[j++] = hpred[0] >> 4;
          meta_data[j++] = hpred[0] << 4 | hpred[1] >> 8;
          meta_data[j++] = hpred[1];
        }
      }
    }
  } else
    fprintf(stderr, _("%s has unknown CAMF type %d.\n"), ifname, type);
}

const char *CLASS foveon_camf_param(const char *block, const char *param) {
  unsigned idx, num;
  char *pos, *cp, *dp;

  for (idx = 0; idx < meta_length; idx += sget4(pos + 8)) {
    pos = meta_data + idx;
    if (strncmp(pos, "CMb", 3))
      break;
    if (pos[3] != 'P')
      continue;
    if (strcmp(block, pos + sget4(pos + 12)))
      continue;
    cp = pos + sget4(pos + 16);
    num = sget4(cp);
    dp = pos + sget4(cp + 4);
    while (num--) {
      cp += 8;
      if (!strcmp(param, dp + sget4(cp)))
        return dp + sget4(cp + 4);
    }
  }
  return 0;
}

void *CLASS foveon_camf_matrix(unsigned dim[3], const char *name) {
  unsigned i, idx, type, ndim, size, *mat;
  char *pos, *cp, *dp;
  double dsize;

  for (idx = 0; idx < meta_length; idx += sget4(pos + 8)) {
    pos = meta_data + idx;
    if (strncmp(pos, "CMb", 3))
      break;
    if (pos[3] != 'M')
      continue;
    if (strcmp(name, pos + sget4(pos + 12)))
      continue;
    dim[0] = dim[1] = dim[2] = 1;
    cp = pos + sget4(pos + 16);
    type = sget4(cp);
    if ((ndim = sget4(cp + 4)) > 3)
      break;
    dp = pos + sget4(cp + 8);
    for (i = ndim; i--;) {
      cp += 12;
      dim[i] = sget4(cp);
    }
    if ((dsize = (double)dim[0] * dim[1] * dim[2]) > meta_length / 4)
      break;
    mat = (unsigned *)malloc((size = dsize) * 4);
    merror(mat, "foveon_camf_matrix()");
    for (i = 0; i < size; i++)
      if (type && type != 6)
        mat[i] = sget4(dp + i * 4);
      else
        mat[i] = sget4(dp + i * 2) & 0xffff;
    return mat;
  }
  fprintf(stderr, _("%s: \"%s\" matrix not found!\n"), ifname, name);
  return 0;
}

int CLASS foveon_fixed(void *ptr, int size, const char *name) {
  void *dp;
  unsigned dim[3];

  if (!name)
    return 0;
  dp = foveon_camf_matrix(dim, name);
  if (!dp)
    return 0;
  memcpy(ptr, dp, size * 4);
  free(dp);
  return 1;
}

float CLASS foveon_avg(short *pix, int range[2], float cfilt) {
  int i;
  float val, min = FLT_MAX, max = -FLT_MAX, sum = 0;

  for (i = range[0]; i <= range[1]; i++) {
    sum += val = pix[i * 4] + (pix[i * 4] - pix[(i - 1) * 4]) * cfilt;
    if (min > val)
      min = val;
    if (max < val)
      max = val;
  }
  if (range[1] - range[0] == 1)
    return sum / 2;
  return (sum - min - max) / (range[1] - range[0] - 1);
}

short *CLASS foveon_make_curve(double max, double mul, double filt) {
  short *curve;
  unsigned i, size;
  double x;

  if (!filt)
    filt = 0.8;
  size = 4 * M_PI * max / filt;
  if (size == UINT_MAX)
    size--;
  curve = (short *)calloc(size + 1, sizeof *curve);
  merror(curve, "foveon_make_curve()");
  curve[0] = size;
  for (i = 0; i < size; i++) {
    x = i * filt / max / 4;
    curve[i + 1] = (cos(x) + 1) / 2 * tanh(i * filt / mul) * mul + 0.5;
  }
  return curve;
}

void CLASS foveon_make_curves(short **curvep,
                              float dq[3],
                              float div[3],
                              float filt) {
  double mul[3], max = 0;
  int c;

  FORC3 mul[c] = dq[c] / div[c];
  FORC3 if (max < mul[c]) max = mul[c];
  FORC3 curvep[c] = foveon_make_curve(max, mul[c], filt);
}

int CLASS foveon_apply_curve(short *curve, int i) {
  if (abs(i) >= curve[0])
    return 0;
  return i < 0 ? -curve[1 - i] : curve[1 + i];
}

#define image ((short(*)[4])image)

void CLASS foveon_interpolate() {
  static const short hood[] = {-1, -1, -1, 0,  -1, 1, 0, -1,
                               0,  1,  1,  -1, 1,  0, 1, 1};
  short *pix, prev[3], *curve[8], (*shrink)[3];
  float cfilt = 0, ddft[3][3][2], ppm[3][3][3];
  float cam_xyz[3][3], correct[3][3], last[3][3], trans[3][3];
  float chroma_dq[3], color_dq[3], diag[3][3], div[3];
  float(*black)[3], (*sgain)[3], (*sgrow)[3];
  float fsum[3], val, frow, num;
  int row, col, c, i, j, diff, sgx, irow, sum, min, max, limit;
  int dscr[2][2], dstb[4], (*smrow[7])[3], total[4], ipix[3];
  int work[3][3], smlast, smred, smred_p = 0, dev[3];
  int satlev[3], keep[4], active[4];
  unsigned dim[3], *badpix;
  double dsum = 0, trsum[3];
  char str[128];
  const char *cp;

  if (verbose)
    fprintf(stderr, _("Foveon interpolation...\n"));

  foveon_load_camf();
  foveon_fixed(dscr, 4, "DarkShieldColRange");
  foveon_fixed(ppm[0][0], 27, "PostPolyMatrix");
  foveon_fixed(satlev, 3, "SaturationLevel");
  foveon_fixed(keep, 4, "KeepImageArea");
  foveon_fixed(active, 4, "ActiveImageArea");
  foveon_fixed(chroma_dq, 3, "ChromaDQ");
  foveon_fixed(color_dq, 3,
               foveon_camf_param("IncludeBlocks", "ColorDQ") ? "ColorDQ"
                                                             : "ColorDQCamRGB");
  if (foveon_camf_param("IncludeBlocks", "ColumnFilter"))
    foveon_fixed(&cfilt, 1, "ColumnFilter");

  memset(ddft, 0, sizeof ddft);
  if (!foveon_camf_param("IncludeBlocks", "DarkDrift") ||
      !foveon_fixed(ddft[1][0], 12, "DarkDrift"))
    for (i = 0; i < 2; i++) {
      foveon_fixed(dstb, 4, i ? "DarkShieldBottom" : "DarkShieldTop");
      for (row = dstb[1]; row <= dstb[3]; row++)
        for (col = dstb[0]; col <= dstb[2]; col++)
          FORC3 ddft[i + 1][c][1] += (short)image[row * width + col][c];
      FORC3 ddft[i + 1][c][1] /=
          (dstb[3] - dstb[1] + 1) * (dstb[2] - dstb[0] + 1);
    }

  if (!(cp = foveon_camf_param("WhiteBalanceIlluminants", model2))) {
    fprintf(stderr, _("%s: Invalid white balance \"%s\"\n"), ifname, model2);
    return;
  }
  foveon_fixed(cam_xyz, 9, cp);
  foveon_fixed(correct, 9,
               foveon_camf_param("WhiteBalanceCorrections", model2));
  memset(last, 0, sizeof last);
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      FORC3 last[i][j] += correct[i][c] * cam_xyz[c][j];

#define LAST(x, y) last[(i + x) % 3][(c + y) % 3]
  for (i = 0; i < 3; i++)
    FORC3 diag[c][i] = LAST(1, 1) * LAST(2, 2) - LAST(1, 2) * LAST(2, 1);
#undef LAST
  FORC3 div[c] = diag[c][0] * 0.3127 + diag[c][1] * 0.329 + diag[c][2] * 0.3583;
  sprintf(str, "%sRGBNeutral", model2);
  if (foveon_camf_param("IncludeBlocks", str))
    foveon_fixed(div, 3, str);
  num = 0;
  FORC3 if (num < div[c]) num = div[c];
  FORC3 div[c] /= num;

  memset(trans, 0, sizeof trans);
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      FORC3 trans[i][j] += rgb_cam[i][c] * last[c][j] * div[j];
  FORC3 trsum[c] = trans[c][0] + trans[c][1] + trans[c][2];
  dsum = (6 * trsum[0] + 11 * trsum[1] + 3 * trsum[2]) / 20;
  for (i = 0; i < 3; i++)
    FORC3 last[i][c] = trans[i][c] * dsum / trsum[i];
  memset(trans, 0, sizeof trans);
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      FORC3 trans[i][j] += (i == c ? 32 : -1) * last[c][j] / 30;

  foveon_make_curves(curve, color_dq, div, cfilt);
  FORC3 chroma_dq[c] /= 3;
  foveon_make_curves(curve + 3, chroma_dq, div, cfilt);
  FORC3 dsum += chroma_dq[c] / div[c];
  curve[6] = foveon_make_curve(dsum, dsum, cfilt);
  curve[7] = foveon_make_curve(dsum * 2, dsum * 2, cfilt);

  sgain = (float(*)[3])foveon_camf_matrix(dim, "SpatialGain");
  if (!sgain)
    return;
  sgrow = (float(*)[3])calloc(dim[1], sizeof *sgrow);
  sgx = (width + dim[1] - 2) / (dim[1] - 1);

  black = (float(*)[3])calloc(height, sizeof *black);
  for (row = 0; row < height; row++) {
    for (i = 0; i < 6; i++)
      ((float *)ddft[0])[i] = ((float *)ddft[1])[i] +
                              row / (height - 1.0) * (((float *)ddft[2])[i] -
                                                      ((float *)ddft[1])[i]);
    FORC3 black[row][c] =
        (foveon_avg(image[row * width] + c, dscr[0], cfilt) +
         foveon_avg(image[row * width] + c, dscr[1], cfilt) * 3 -
         ddft[0][c][0]) /
            4 -
        ddft[0][c][1];
  }
  memcpy(black, black + 8, sizeof *black * 8);
  memcpy(black + height - 11, black + height - 22, 11 * sizeof *black);
  memcpy(last, black, sizeof last);

  for (row = 1; row < height - 1; row++) {
    FORC3 if (last[1][c] > last[0][c]) {
      if (last[1][c] > last[2][c])
        black[row][c] = (last[0][c] > last[2][c]) ? last[0][c] : last[2][c];
    }
    else if (last[1][c] < last[2][c]) black[row][c] =
        (last[0][c] < last[2][c]) ? last[0][c] : last[2][c];
    memmove(last, last + 1, 2 * sizeof last[0]);
    memcpy(last[2], black[row + 1], sizeof last[2]);
  }
  FORC3 black[row][c] = (last[0][c] + last[1][c]) / 2;
  FORC3 black[0][c] = (black[1][c] + black[3][c]) / 2;

  val = 1 - exp(-1 / 24.0);
  memcpy(fsum, black, sizeof fsum);
  for (row = 1; row < height; row++)
    FORC3 fsum[c] += black[row][c] =
        (black[row][c] - black[row - 1][c]) * val + black[row - 1][c];
  memcpy(last[0], black[height - 1], sizeof last[0]);
  FORC3 fsum[c] /= height;
  for (row = height; row--;)
    FORC3 last[0][c] = black[row][c] =
        (black[row][c] - fsum[c] - last[0][c]) * val + last[0][c];

  memset(total, 0, sizeof total);
  for (row = 2; row < height; row += 4)
    for (col = 2; col < width; col += 4) {
      FORC3 total[c] += (short)image[row * width + col][c];
      total[3]++;
    }
  for (row = 0; row < height; row++)
    FORC3 black[row][c] += fsum[c] / 2 + total[c] / (total[3] * 100.0);

  for (row = 0; row < height; row++) {
    for (i = 0; i < 6; i++)
      ((float *)ddft[0])[i] = ((float *)ddft[1])[i] +
                              row / (height - 1.0) * (((float *)ddft[2])[i] -
                                                      ((float *)ddft[1])[i]);
    pix = image[row * width];
    memcpy(prev, pix, sizeof prev);
    frow = row / (height - 1.0) * (dim[2] - 1);
    if ((irow = frow) == dim[2] - 1)
      irow--;
    frow -= irow;
    for (i = 0; i < dim[1]; i++)
      FORC3 sgrow[i][c] = sgain[irow * dim[1] + i][c] * (1 - frow) +
                          sgain[(irow + 1) * dim[1] + i][c] * frow;
    for (col = 0; col < width; col++) {
      FORC3 {
        diff = pix[c] - prev[c];
        prev[c] = pix[c];
        ipix[c] =
            pix[c] +
            floor((diff + (diff * diff >> 14)) * cfilt - ddft[0][c][1] -
                  ddft[0][c][0] * ((float)col / width - 0.5) - black[row][c]);
      }
      FORC3 {
        work[0][c] = ipix[c] * ipix[c] >> 14;
        work[2][c] = ipix[c] * work[0][c] >> 14;
        work[1][2 - c] = ipix[(c + 1) % 3] * ipix[(c + 2) % 3] >> 14;
      }
      FORC3 {
        for (val = i = 0; i < 3; i++)
          for (j = 0; j < 3; j++)
            val += ppm[c][i][j] * work[i][j];
        ipix[c] = floor((ipix[c] + floor(val)) *
                        (sgrow[col / sgx][c] * (sgx - col % sgx) +
                         sgrow[col / sgx + 1][c] * (col % sgx)) /
                        sgx / div[c]);
        if (ipix[c] > 32000)
          ipix[c] = 32000;
        pix[c] = ipix[c];
      }
      pix += 4;
    }
  }
  free(black);
  free(sgrow);
  free(sgain);

  if ((badpix = (unsigned *)foveon_camf_matrix(dim, "BadPixels"))) {
    for (i = 0; i < dim[0]; i++) {
      col = (badpix[i] >> 8 & 0xfff) - keep[0];
      row = (badpix[i] >> 20) - keep[1];
      if ((unsigned)(row - 1) > height - 3 || (unsigned)(col - 1) > width - 3)
        continue;
      memset(fsum, 0, sizeof fsum);
      for (sum = j = 0; j < 8; j++)
        if (badpix[i] & (1 << j)) {
          FORC3 fsum[c] += (short)
              image[(row + hood[j * 2]) * width + col + hood[j * 2 + 1]][c];
          sum++;
        }
      if (sum)
        FORC3 image[row * width + col][c] = fsum[c] / sum;
    }
    free(badpix);
  }

  /* Array for 5x5 Gaussian averaging of red values */
  smrow[6] = (int(*)[3])calloc(width * 5, sizeof **smrow);
  merror(smrow[6], "foveon_interpolate()");
  for (i = 0; i < 5; i++)
    smrow[i] = smrow[6] + i * width;

  /* Sharpen the reds against these Gaussian averages */
  for (smlast = -1, row = 2; row < height - 2; row++) {
    while (smlast < row + 2) {
      for (i = 0; i < 6; i++)
        smrow[(i + 5) % 6] = smrow[i];
      pix = image[++smlast * width + 2];
      for (col = 2; col < width - 2; col++) {
        smrow[4][col][0] =
            (pix[0] * 6 + (pix[-4] + pix[4]) * 4 + pix[-8] + pix[8] + 8) >> 4;
        pix += 4;
      }
    }
    pix = image[row * width + 2];
    for (col = 2; col < width - 2; col++) {
      smred =
          (6 * smrow[2][col][0] + 4 * (smrow[1][col][0] + smrow[3][col][0]) +
           smrow[0][col][0] + smrow[4][col][0] + 8) >>
          4;
      if (col == 2)
        smred_p = smred;
      i = pix[0] + ((pix[0] - ((smred * 7 + smred_p) >> 3)) >> 3);
      if (i > 32000)
        i = 32000;
      pix[0] = i;
      smred_p = smred;
      pix += 4;
    }
  }

  /* Adjust the brighter pixels for better linearity */
  min = 0xffff;
  FORC3 {
    i = satlev[c] / div[c];
    if (min > i)
      min = i;
  }
  limit = min * 9 >> 4;
  for (pix = image[0]; pix < image[height * width]; pix += 4) {
    if (pix[0] <= limit || pix[1] <= limit || pix[2] <= limit)
      continue;
    min = max = pix[0];
    for (c = 1; c < 3; c++) {
      if (min > pix[c])
        min = pix[c];
      if (max < pix[c])
        max = pix[c];
    }
    if (min >= limit * 2) {
      pix[0] = pix[1] = pix[2] = max;
    } else {
      i = 0x4000 - ((min - limit) << 14) / limit;
      i = 0x4000 - (i * i >> 14);
      i = i * i >> 14;
      FORC3 pix[c] += (max - pix[c]) * i >> 14;
    }
  }
  /*
  Because photons that miss one detector often hit another,
  the sum R+G+B is much less noisy than the individual colors.
  So smooth the hues without smoothing the total.
  */
  for (smlast = -1, row = 2; row < height - 2; row++) {
    while (smlast < row + 2) {
      for (i = 0; i < 6; i++)
        smrow[(i + 5) % 6] = smrow[i];
      pix = image[++smlast * width + 2];
      for (col = 2; col < width - 2; col++) {
        FORC3 smrow[4][col][c] =
            (pix[c - 4] + 2 * pix[c] + pix[c + 4] + 2) >> 2;
        pix += 4;
      }
    }
    pix = image[row * width + 2];
    for (col = 2; col < width - 2; col++) {
      FORC3 dev[c] = -foveon_apply_curve(
          curve[7],
          pix[c] -
              ((smrow[1][col][c] + 2 * smrow[2][col][c] + smrow[3][col][c]) >>
               2));
      sum = (dev[0] + dev[1] + dev[2]) >> 3;
      FORC3 pix[c] += dev[c] - sum;
      pix += 4;
    }
  }
  for (smlast = -1, row = 2; row < height - 2; row++) {
    while (smlast < row + 2) {
      for (i = 0; i < 6; i++)
        smrow[(i + 5) % 6] = smrow[i];
      pix = image[++smlast * width + 2];
      for (col = 2; col < width - 2; col++) {
        FORC3 smrow[4][col][c] =
            (pix[c - 8] + pix[c - 4] + pix[c] + pix[c + 4] + pix[c + 8] + 2) >>
            2;
        pix += 4;
      }
    }
    pix = image[row * width + 2];
    for (col = 2; col < width - 2; col++) {
      for (total[3] = 375, sum = 60, c = 0; c < 3; c++) {
        for (total[c] = i = 0; i < 5; i++)
          total[c] += smrow[i][col][c];
        total[3] += total[c];
        sum += pix[c];
      }
      if (sum < 0)
        sum = 0;
      j = total[3] > 375 ? (sum << 16) / total[3] : sum * 174;
      FORC3 pix[c] += foveon_apply_curve(
          curve[6], ((j * total[c] + 0x8000) >> 16) - pix[c]);
      pix += 4;
    }
  }

  /* Transform the image to a different colorspace */
  for (pix = image[0]; pix < image[height * width]; pix += 4) {
    FORC3 pix[c] -= foveon_apply_curve(curve[c], pix[c]);
    sum = (pix[0] + pix[1] + pix[1] + pix[2]) >> 2;
    FORC3 pix[c] -= foveon_apply_curve(curve[c], pix[c] - sum);
    FORC3 {
      for (dsum = i = 0; i < 3; i++)
        dsum += trans[c][i] * pix[i];
      if (dsum < 0)
        dsum = 0;
      if (dsum > 24000)
        dsum = 24000;
      ipix[c] = dsum + 0.5;
    }
    FORC3 pix[c] = ipix[c];
  }

  /* Smooth the image bottom-to-top and save at 1/4 scale */
  shrink = (short(*)[3])calloc((height / 4), (width / 4) * sizeof *shrink);
  merror(shrink, "foveon_interpolate()");
  for (row = height / 4; row--;)
    for (col = 0; col < width / 4; col++) {
      ipix[0] = ipix[1] = ipix[2] = 0;
      for (i = 0; i < 4; i++)
        for (j = 0; j < 4; j++)
          FORC3 ipix[c] += image[(row * 4 + i) * width + col * 4 + j][c];
      FORC3
      if (row + 2 > height / 4)
        shrink[row * (width / 4) + col][c] = ipix[c] >> 4;
      else
        shrink[row * (width / 4) + col][c] =
            (shrink[(row + 1) * (width / 4) + col][c] * 1840 + ipix[c] * 141 +
             2048) >>
            12;
    }
  /* From the 1/4-scale image, smooth right-to-left */
  for (row = 0; row < (height & ~3); row++) {
    ipix[0] = ipix[1] = ipix[2] = 0;
    if ((row & 3) == 0)
      for (col = width & ~3; col--;)
        FORC3 smrow[0][col][c] = ipix[c] =
            (shrink[(row / 4) * (width / 4) + col / 4][c] * 1485 +
             ipix[c] * 6707 + 4096) >>
            13;

    /* Then smooth left-to-right */
    ipix[0] = ipix[1] = ipix[2] = 0;
    for (col = 0; col < (width & ~3); col++)
      FORC3 smrow[1][col][c] = ipix[c] =
          (smrow[0][col][c] * 1485 + ipix[c] * 6707 + 4096) >> 13;

    /* Smooth top-to-bottom */
    if (row == 0)
      memcpy(smrow[2], smrow[1], sizeof **smrow * width);
    else
      for (col = 0; col < (width & ~3); col++)
        FORC3 smrow[2][col][c] =
            (smrow[2][col][c] * 6707 + smrow[1][col][c] * 1485 + 4096) >> 13;

    /* Adjust the chroma toward the smooth values */
    for (col = 0; col < (width & ~3); col++) {
      for (i = j = 30, c = 0; c < 3; c++) {
        i += smrow[2][col][c];
        j += image[row * width + col][c];
      }
      j = (j << 16) / i;
      for (sum = c = 0; c < 3; c++) {
        ipix[c] = foveon_apply_curve(curve[c + 3],
                                     ((smrow[2][col][c] * j + 0x8000) >> 16) -
                                         image[row * width + col][c]);
        sum += ipix[c];
      }
      sum >>= 3;
      FORC3 {
        i = image[row * width + col][c] + ipix[c] - sum;
        if (i < 0)
          i = 0;
        image[row * width + col][c] = i;
      }
    }
  }
  free(shrink);
  free(smrow[6]);
  for (i = 0; i < 8; i++)
    free(curve[i]);

  /* Trim off the black border */
  active[1] -= keep[1];
  active[3] -= 2;
  i = active[2] - active[0];
  for (row = 0; row < active[3] - active[1]; row++)
    memcpy(image[row * i], image[(row + active[1]) * width + active[0]],
           i * sizeof *image);
  width = i;
  height = row;
}
#undef image

/* RESTRICTED code ends here */

void CLASS crop_masked_pixels() {
  int row, col;
  unsigned r, c, m, mblack[8], zero, val;

  if (load_raw == &CLASS phase_one_load_raw ||
      load_raw == &CLASS phase_one_load_raw_c)
    phase_one_correct();
  if (fuji_width) {
    for (row = 0; row < raw_height - top_margin * 2; row++) {
      for (col = 0; col < fuji_width << !fuji_layout; col++) {
        if (fuji_layout) {
          r = fuji_width - 1 - col + (row >> 1);
          c = col + ((row + 1) >> 1);
        } else {
          r = fuji_width - 1 + row - (col >> 1);
          c = row + ((col + 1) >> 1);
        }
        if (r < height && c < width)
          BAYER(r, c) = RAW(row + top_margin, col + left_margin);
      }
    }
  } else {
    for (row = 0; row < height; row++)
      for (col = 0; col < width; col++)
        BAYER2(row, col) = RAW(row + top_margin, col + left_margin);
  }
  if (mask[0][3] > 0)
    goto mask_set;
  if (load_raw == &CLASS canon_load_raw ||
      load_raw == &CLASS lossless_jpeg_load_raw) {
    mask[0][1] = mask[1][1] += 2;
    mask[0][3] -= 2;
    goto sides;
  }
  if (load_raw == &CLASS canon_600_load_raw ||
      load_raw == &CLASS sony_load_raw ||
      (load_raw == &CLASS eight_bit_load_raw && strncmp(model, "DC2", 3)) ||
      load_raw == &CLASS kodak_262_load_raw ||
      (load_raw == &CLASS packed_load_raw && (load_flags & 32))) {
  sides:
    mask[0][0] = mask[1][0] = top_margin;
    mask[0][2] = mask[1][2] = top_margin + height;
    mask[0][3] += left_margin;
    mask[1][1] += left_margin + width;
    mask[1][3] += raw_width;
  }
  if (load_raw == &CLASS nokia_load_raw) {
    mask[0][2] = top_margin;
    mask[0][3] = width;
  }
mask_set:
  memset(mblack, 0, sizeof mblack);
  for (zero = m = 0; m < 8; m++)
    for (row = MAX(mask[m][0], 0); row < MIN(mask[m][2], raw_height); row++)
      for (col = MAX(mask[m][1], 0); col < MIN(mask[m][3], raw_width); col++) {
        c = FC(row - top_margin, col - left_margin);
        mblack[c] += val = RAW(row, col);
        mblack[4 + c]++;
        zero += !val;
      }
  if (load_raw == &CLASS canon_600_load_raw && width < raw_width) {
    black = (mblack[0] + mblack[1] + mblack[2] + mblack[3]) /
                (mblack[4] + mblack[5] + mblack[6] + mblack[7]) -
            4;
    canon_600_correct();
  } else if (zero < mblack[4] && mblack[5] && mblack[6] && mblack[7]) {
    FORC4 cblack[c] = mblack[c] / mblack[4 + c];
    cblack[4] = cblack[5] = cblack[6] = 0;
  }
}

void CLASS remove_zeroes() {
  unsigned row, col, tot, n, r, c;

  for (row = 0; row < height; row++)
    for (col = 0; col < width; col++)
      if (BAYER(row, col) == 0) {
        tot = n = 0;
        for (r = row - 2; r <= row + 2; r++)
          for (c = col - 2; c <= col + 2; c++)
            if (r < height && c < width && FC(r, c) == FC(row, col) &&
                BAYER(r, c))
              tot += (n++, BAYER(r, c));
        if (n)
          BAYER(row, col) = tot / n;
      }
}

/*
Seach from the current directory up to the root looking for
a ".badpixels" file, and fix those pixels now.
*/
void CLASS bad_pixels(const char *cfname) {
  // Note: deleted, since there's no getcwd on Windows.
}

void CLASS subtract(const char *fname) {
  FILE *fp;
  int dim[3] = {0, 0, 0}, comment = 0, number = 0, error = 0, nd = 0, c, row,
      col;
  ushort *pixel;

  if (!(fp = fopen(fname, "rb"))) {
    perror(fname);
    return;
  }
  if (fgetc(fp) != 'P' || fgetc(fp) != '5')
    error = 1;
  while (!error && nd < 3 && (c = fgetc(fp)) != EOF) {
    if (c == '#')
      comment = 1;
    if (c == '\n')
      comment = 0;
    if (comment)
      continue;
    if (isdigit(c))
      number = 1;
    if (number) {
      if (isdigit(c))
        dim[nd] = dim[nd] * 10 + c - '0';
      else if (isspace(c)) {
        number = 0;
        nd++;
      } else
        error = 1;
    }
  }
  if (error || nd < 3) {
    fprintf(stderr, _("%s is not a valid PGM file!\n"), fname);
    fclose(fp);
    return;
  } else if (dim[0] != width || dim[1] != height || dim[2] != 65535) {
    fprintf(stderr, _("%s has the wrong dimensions!\n"), fname);
    fclose(fp);
    return;
  }
  pixel = (ushort *)calloc(width, sizeof *pixel);
  merror(pixel, "subtract()");
  for (row = 0; row < height; row++) {
    fread(pixel, 2, width, fp);
    for (col = 0; col < width; col++)
      BAYER(row, col) = MAX(BAYER(row, col) - ntohs(pixel[col]), 0);
  }
  free(pixel);
  fclose(fp);
  memset(cblack, 0, sizeof cblack);
  black = 0;
}

void CLASS gamma_curve(double pwr, double ts, int mode, int imax) {
  int i;
  double g[6], bnd[2] = {0, 0}, r;

  g[0] = pwr;
  g[1] = ts;
  g[2] = g[3] = g[4] = 0;
  bnd[g[1] >= 1] = 1;
  if (g[1] && (g[1] - 1) * (g[0] - 1) <= 0) {
    for (i = 0; i < 48; i++) {
      g[2] = (bnd[0] + bnd[1]) / 2;
      if (g[0])
        bnd[(pow(g[2] / g[1], -g[0]) - 1) / g[0] - 1 / g[2] > -1] = g[2];
      else
        bnd[g[2] / exp(1 - 1 / g[2]) < g[1]] = g[2];
    }
    g[3] = g[2] / g[1];
    if (g[0])
      g[4] = g[2] * (1 / g[0] - 1);
  }
  if (g[0])
    g[5] = 1 / (g[1] * SQR(g[3]) / 2 - g[4] * (1 - g[3]) +
                (1 - pow(g[3], 1 + g[0])) * (1 + g[4]) / (1 + g[0])) -
           1;
  else
    g[5] = 1 / (g[1] * SQR(g[3]) / 2 + 1 - g[2] - g[3] -
                g[2] * g[3] * (log(g[3]) - 1)) -
           1;
  if (!mode--) {
    memcpy(gamm, g, sizeof gamm);
    return;
  }
  for (i = 0; i < 0x10000; i++) {
    curve[i] = 0xffff;
    if ((r = (double)i / imax) < 1)
      curve[i] =
          0x10000 *
          (mode
               ? (r < g[3] ? r * g[1] : (g[0] ? pow(r, g[0]) * (1 + g[4]) - g[4]
                                              : log(r) * g[2] + 1))
               : (r < g[2] ? r / g[1]
                           : (g[0] ? pow((r + g[4]) / (1 + g[4]), 1 / g[0])
                                   : exp((r - 1) / g[2]))));
  }
}

void CLASS pseudoinverse(double (*in)[3], double (*out)[3], int size) {
  double work[3][6], num;
  int i, j, k;

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 6; j++)
      work[i][j] = j == i + 3;
    for (j = 0; j < 3; j++)
      for (k = 0; k < size; k++)
        work[i][j] += in[k][i] * in[k][j];
  }
  for (i = 0; i < 3; i++) {
    num = work[i][i];
    for (j = 0; j < 6; j++)
      work[i][j] /= num;
    for (k = 0; k < 3; k++) {
      if (k == i)
        continue;
      num = work[k][i];
      for (j = 0; j < 6; j++)
        work[k][j] -= work[i][j] * num;
    }
  }
  for (i = 0; i < size; i++)
    for (j = 0; j < 3; j++)
      for (out[i][j] = k = 0; k < 3; k++)
        out[i][j] += work[j][k + 3] * in[i][k];
}

void CLASS cam_xyz_coeff(float rgb_cam[3][4], double cam_xyz[4][3]) {
  double cam_rgb[4][3], inverse[4][3], num;
  int i, j, k;

  for (i = 0; i < colors; i++) /* Multiply out XYZ colorspace */
    for (j = 0; j < 3; j++)
      for (cam_rgb[i][j] = k = 0; k < 3; k++)
        cam_rgb[i][j] += cam_xyz[i][k] * xyz_rgb[k][j];

  for (i = 0; i < colors; i++) {  /* Normalize cam_rgb so that */
    for (num = j = 0; j < 3; j++) /* cam_rgb * (1,1,1) is (1,1,1,1) */
      num += cam_rgb[i][j];
    for (j = 0; j < 3; j++)
      cam_rgb[i][j] /= num;
    pre_mul[i] = 1 / num;
  }
  pseudoinverse(cam_rgb, inverse, colors);
  for (i = 0; i < 3; i++)
    for (j = 0; j < colors; j++)
      rgb_cam[i][j] = inverse[j][i];
}

#ifdef COLORCHECK
void CLASS colorcheck() {
#define NSQ 24
  // Coordinates of the GretagMacbeth ColorChecker squares
  // width, height, 1st_column, 1st_row
  int cut[NSQ][4];  // you must set these
                    // ColorChecker Chart under 6500-kelvin illumination
  static const double gmb_xyY[NSQ][3] = {{0.400, 0.350, 10.1},  // Dark Skin
                                         {0.377, 0.345, 35.8},  // Light Skin
                                         {0.247, 0.251, 19.3},  // Blue Sky
                                         {0.337, 0.422, 13.3},  // Foliage
                                         {0.265, 0.240, 24.3},  // Blue Flower
                                         {0.261, 0.343, 43.1},  // Bluish Green
                                         {0.506, 0.407, 30.1},  // Orange
                                         {0.211, 0.175, 12.0},  // Purplish Blue
                                         {0.453, 0.306, 19.8},  // Moderate Red
                                         {0.285, 0.202, 6.6},   // Purple
                                         {0.380, 0.489, 44.3},  // Yellow Green
                                         {0.473, 0.438, 43.1},  // Orange Yellow
                                         {0.187, 0.129, 6.1},   // Blue
                                         {0.305, 0.478, 23.4},  // Green
                                         {0.539, 0.313, 12.0},  // Red
                                         {0.448, 0.470, 59.1},  // Yellow
                                         {0.364, 0.233, 19.8},  // Magenta
                                         {0.196, 0.252, 19.8},  // Cyan
                                         {0.310, 0.316, 90.0},  // White
                                         {0.310, 0.316, 59.1},  // Neutral 8
                                         {0.310, 0.316, 36.2},  // Neutral 6.5
                                         {0.310, 0.316, 19.8},  // Neutral 5
                                         {0.310, 0.316, 9.0},   // Neutral 3.5
                                         {0.310, 0.316, 3.1}};  // Black
  double gmb_cam[NSQ][4], gmb_xyz[NSQ][3];
  double inverse[NSQ][3], cam_xyz[4][3], balance[4], num;
  int c, i, j, k, sq, row, col, pass, count[4];

  memset(gmb_cam, 0, sizeof gmb_cam);
  for (sq = 0; sq < NSQ; sq++) {
    FORCC count[c] = 0;
    for (row = cut[sq][3]; row < cut[sq][3] + cut[sq][1]; row++)
      for (col = cut[sq][2]; col < cut[sq][2] + cut[sq][0]; col++) {
        c = FC(row, col);
        if (c >= colors)
          c -= 2;
        gmb_cam[sq][c] += BAYER2(row, col);
        BAYER2(row, col) = black + (BAYER2(row, col) - black) / 2;
        count[c]++;
      }
    FORCC gmb_cam[sq][c] = gmb_cam[sq][c] / count[c] - black;
    gmb_xyz[sq][0] = gmb_xyY[sq][2] * gmb_xyY[sq][0] / gmb_xyY[sq][1];
    gmb_xyz[sq][1] = gmb_xyY[sq][2];
    gmb_xyz[sq][2] =
        gmb_xyY[sq][2] * (1 - gmb_xyY[sq][0] - gmb_xyY[sq][1]) / gmb_xyY[sq][1];
  }
  pseudoinverse(gmb_xyz, inverse, NSQ);
  for (pass = 0; pass < 2; pass++) {
    for (raw_color = i = 0; i < colors; i++)
      for (j = 0; j < 3; j++)
        for (cam_xyz[i][j] = k = 0; k < NSQ; k++)
          cam_xyz[i][j] += gmb_cam[k][i] * inverse[k][j];
    cam_xyz_coeff(rgb_cam, cam_xyz);
    FORCC balance[c] = pre_mul[c] * gmb_cam[20][c];
    for (sq = 0; sq < NSQ; sq++)
      FORCC gmb_cam[sq][c] *= balance[c];
  }
  if (verbose) {
    printf("    { \"%s %s\", %d,\n\t{", make, model, black);
    num = 10000 / (cam_xyz[1][0] + cam_xyz[1][1] + cam_xyz[1][2]);
    FORCC for (j = 0; j < 3; j++)
        printf("%c%d", (c | j) ? ',' : ' ', (int)(cam_xyz[c][j] * num + 0.5));
    puts(" } },");
  }
#undef NSQ
}
#endif

void CLASS hat_transform(float *temp, float *base, int st, int size, int sc) {
  int i;
  for (i = 0; i < sc; i++)
    temp[i] = 2 * base[st * i] + base[st * (sc - i)] + base[st * (i + sc)];
  for (; i + sc < size; i++)
    temp[i] = 2 * base[st * i] + base[st * (i - sc)] + base[st * (i + sc)];
  for (; i < size; i++)
    temp[i] = 2 * base[st * i] + base[st * (i - sc)] +
              base[st * (2 * size - 2 - (i + sc))];
}

void CLASS wavelet_denoise() {
  float *fimg = 0, *temp, thold, mul[2], avg, diff;
  int scale = 1, size, lev, hpass, lpass, row, col, nc, c, i, wlast, blk[2];
  ushort *window[4];
  static const float noise[] = {0.8002, 0.2735, 0.1202, 0.0585,
                                0.0291, 0.0152, 0.0080, 0.0044};

  if (verbose)
    fprintf(stderr, _("Wavelet denoising...\n"));

  while (maximum << scale < 0x10000)
    scale++;
  maximum <<= --scale;
  black <<= scale;
  FORC4 cblack[c] <<= scale;
  if ((size = iheight * iwidth) < 0x15550000)
    fimg = (float *)malloc((size * 3 + iheight + iwidth) * sizeof *fimg);
  merror(fimg, "wavelet_denoise()");
  temp = fimg + size * 3;
  if ((nc = colors) == 3 && filters)
    nc++;
  FORC(nc) { /* denoise R,G1,B,G3 individually */
    for (i = 0; i < size; i++)
      fimg[i] = 256 * sqrt(image[i][c] << scale);
    for (hpass = lev = 0; lev < 5; lev++) {
      lpass = size * ((lev & 1) + 1);
      for (row = 0; row < iheight; row++) {
        hat_transform(temp, fimg + hpass + row * iwidth, 1, iwidth, 1 << lev);
        for (col = 0; col < iwidth; col++)
          fimg[lpass + row * iwidth + col] = temp[col] * 0.25;
      }
      for (col = 0; col < iwidth; col++) {
        hat_transform(temp, fimg + lpass + col, iwidth, iheight, 1 << lev);
        for (row = 0; row < iheight; row++)
          fimg[lpass + row * iwidth + col] = temp[row] * 0.25;
      }
      thold = threshold * noise[lev];
      for (i = 0; i < size; i++) {
        fimg[hpass + i] -= fimg[lpass + i];
        if (fimg[hpass + i] < -thold)
          fimg[hpass + i] += thold;
        else if (fimg[hpass + i] > thold)
          fimg[hpass + i] -= thold;
        else
          fimg[hpass + i] = 0;
        if (hpass)
          fimg[i] += fimg[hpass + i];
      }
      hpass = lpass;
    }
    for (i = 0; i < size; i++)
      image[i][c] = CLIP(SQR(fimg[i] + fimg[lpass + i]) / 0x10000);
  }
  if (filters && colors == 3) { /* pull G1 and G3 closer together */
    for (row = 0; row < 2; row++) {
      mul[row] = 0.125 * pre_mul[FC(row + 1, 0) | 1] / pre_mul[FC(row, 0) | 1];
      blk[row] = cblack[FC(row, 0) | 1];
    }
    for (i = 0; i < 4; i++)
      window[i] = (ushort *)fimg + width * i;
    for (wlast = -1, row = 1; row < height - 1; row++) {
      while (wlast < row + 1) {
        for (wlast++, i = 0; i < 4; i++)
          window[(i + 3) & 3] = window[i];
        for (col = FC(wlast, 1) & 1; col < width; col += 2)
          window[2][col] = BAYER(wlast, col);
      }
      thold = threshold / 512;
      for (col = (FC(row, 0) & 1) + 1; col < width - 1; col += 2) {
        avg = (window[0][col - 1] + window[0][col + 1] + window[2][col - 1] +
               window[2][col + 1] - blk[~row & 1] * 4) *
                  mul[row & 1] +
              (window[1][col] + blk[row & 1]) * 0.5;
        avg = avg < 0 ? 0 : sqrt(avg);
        diff = sqrt(BAYER(row, col)) - avg;
        if (diff < -thold)
          diff += thold;
        else if (diff > thold)
          diff -= thold;
        else
          diff = 0;
        BAYER(row, col) = CLIP(SQR(avg + diff) + 0.5);
      }
    }
  }
  free(fimg);
}

void CLASS scale_colors() {
  unsigned bottom, right, size, row, col, ur, uc, i, x, y, c, sum[8];
  int val, dark, sat;
  double dsum[8], dmin, dmax;
  float scale_mul[4], fr, fc;
  ushort *img = 0, *pix;

  if (user_mul[0])
    memcpy(pre_mul, user_mul, sizeof pre_mul);
  if (use_auto_wb || (use_camera_wb && cam_mul[0] == -1)) {
    memset(dsum, 0, sizeof dsum);
    bottom = MIN(greybox[1] + greybox[3], height);
    right = MIN(greybox[0] + greybox[2], width);
    for (row = greybox[1]; row < bottom; row += 8)
      for (col = greybox[0]; col < right; col += 8) {
        memset(sum, 0, sizeof sum);
        for (y = row; y < row + 8 && y < bottom; y++)
          for (x = col; x < col + 8 && x < right; x++)
            FORC4 {
              if (filters) {
                c = fcol(y, x);
                val = BAYER2(y, x);
              } else
                val = image[y * width + x][c];
              if (val > maximum - 25)
                goto skip_block;
              if ((val -= cblack[c]) < 0)
                val = 0;
              sum[c] += val;
              sum[c + 4]++;
              if (filters)
                break;
            }
        FORC(8) dsum[c] += sum[c];
      skip_block:;
      }
    FORC4 if (dsum[c]) pre_mul[c] = dsum[c + 4] / dsum[c];
  }
  if (use_camera_wb && cam_mul[0] != -1) {
    memset(sum, 0, sizeof sum);
    for (row = 0; row < 8; row++)
      for (col = 0; col < 8; col++) {
        c = FC(row, col);
        if ((val = white[row][col] - cblack[c]) > 0)
          sum[c] += val;
        sum[c + 4]++;
      }
    if (sum[0] && sum[1] && sum[2] && sum[3])
      FORC4 pre_mul[c] = (float)sum[c + 4] / sum[c];
    else if (cam_mul[0] && cam_mul[2])
      memcpy(pre_mul, cam_mul, sizeof pre_mul);
    else
      fprintf(stderr, _("%s: Cannot use camera white balance.\n"), ifname);
  }
  if (pre_mul[1] == 0)
    pre_mul[1] = 1;
  if (pre_mul[3] == 0)
    pre_mul[3] = colors < 4 ? pre_mul[1] : 1;
  dark = black;
  sat = maximum;
  if (threshold)
    wavelet_denoise();
  maximum -= black;
  for (dmin = DBL_MAX, dmax = c = 0; c < 4; c++) {
    if (dmin > pre_mul[c])
      dmin = pre_mul[c];
    if (dmax < pre_mul[c])
      dmax = pre_mul[c];
  }
  if (!highlight)
    dmax = dmin;
  FORC4 scale_mul[c] = (pre_mul[c] /= dmax) * 65535.0 / maximum;
  if (verbose) {
    fprintf(stderr,
            _("Scaling with darkness %d, saturation %d, and\nmultipliers"),
            dark, sat);
    FORC4 fprintf(stderr, " %f", pre_mul[c]);
    fputc('\n', stderr);
  }
  if (filters > 1000 && (cblack[4] + 1) / 2 == 1 && (cblack[5] + 1) / 2 == 1) {
    FORC4 cblack[FC(c / 2, c % 2)] +=
        cblack[6 + c / 2 % cblack[4] * cblack[5] + c % 2 % cblack[5]];
    cblack[4] = cblack[5] = 0;
  }
  size = iheight * iwidth;
  for (i = 0; i < size * 4; i++) {
    if (!(val = ((ushort *)image)[i]))
      continue;
    if (cblack[4] && cblack[5])
      val -= cblack[6 + i / 4 / iwidth % cblack[4] * cblack[5] +
                    i / 4 % iwidth % cblack[5]];
    val -= cblack[i & 3];
    val *= scale_mul[i & 3];
    ((ushort *)image)[i] = CLIP(val);
  }
  if ((aber[0] != 1 || aber[2] != 1) && colors == 3) {
    if (verbose)
      fprintf(stderr, _("Correcting chromatic aberration...\n"));
    for (c = 0; c < 4; c += 2) {
      if (aber[c] == 1)
        continue;
      img = (ushort *)malloc(size * sizeof *img);
      merror(img, "scale_colors()");
      for (i = 0; i < size; i++)
        img[i] = image[i][c];
      for (row = 0; row < iheight; row++) {
        ur = fr = (row - iheight * 0.5) * aber[c] + iheight * 0.5;
        if (ur > iheight - 2)
          continue;
        fr -= ur;
        for (col = 0; col < iwidth; col++) {
          uc = fc = (col - iwidth * 0.5) * aber[c] + iwidth * 0.5;
          if (uc > iwidth - 2)
            continue;
          fc -= uc;
          pix = img + ur * iwidth + uc;
          image[row * iwidth + col][c] =
              (pix[0] * (1 - fc) + pix[1] * fc) * (1 - fr) +
              (pix[iwidth] * (1 - fc) + pix[iwidth + 1] * fc) * fr;
        }
      }
      free(img);
    }
  }
}

void CLASS pre_interpolate() {
  ushort(*img)[4];
  int row, col, c;

  if (shrink) {
    if (half_size) {
      height = iheight;
      width = iwidth;
      if (filters == 9) {
        for (row = 0; row < 3; row++)
          for (col = 1; col < 4; col++)
            if (!(image[row * width + col][0] | image[row * width + col][2]))
              goto break2;
      break2:
        for (; row < height; row += 3)
          for (col = (col - 1) % 3 + 1; col < width - 1; col += 3) {
            img = image + row * width + col;
            for (c = 0; c < 3; c += 2)
              img[0][c] = (img[-1][c] + img[1][c]) >> 1;
          }
      }
    } else {
      img = (ushort(*)[4])calloc(height, width * sizeof *img);
      merror(img, "pre_interpolate()");
      for (row = 0; row < height; row++)
        for (col = 0; col < width; col++) {
          c = fcol(row, col);
          img[row * width + col][c] =
              image[(row >> 1) * iwidth + (col >> 1)][c];
        }
      free(image);
      image = img;
      shrink = 0;
    }
  }
  if (filters > 1000 && colors == 3) {
    mix_green = four_color_rgb ^ half_size;
    if (four_color_rgb | half_size)
      colors++;
    else {
      for (row = FC(1, 0) >> 1; row < height; row += 2)
        for (col = FC(row, 1) & 1; col < width; col += 2)
          image[row * width + col][1] = image[row * width + col][3];
      filters &= ~((filters & 0x55555555) << 1);
    }
  }
  if (half_size)
    filters = 0;
}

void CLASS border_interpolate(int border) {
  unsigned row, col, y, x, f, c, sum[8];

  for (row = 0; row < height; row++)
    for (col = 0; col < width; col++) {
      if (col == border && row >= border && row < height - border)
        col = width - border;
      memset(sum, 0, sizeof sum);
      for (y = row - 1; y != row + 2; y++)
        for (x = col - 1; x != col + 2; x++)
          if (y < height && x < width) {
            f = fcol(y, x);
            sum[f] += image[y * width + x][f];
            sum[f + 4]++;
          }
      f = fcol(row, col);
      FORCC if (c != f && sum[c + 4]) image[row * width + col][c] =
          sum[c] / sum[c + 4];
    }
}

void CLASS lin_interpolate() {
  int code[16][16][32], size = 16, *ip, sum[4];
  int f, c, i, x, y, row, col, shift, color;
  ushort *pix;

  if (verbose)
    fprintf(stderr, _("Bilinear interpolation...\n"));
  if (filters == 9)
    size = 6;
  border_interpolate(1);
  for (row = 0; row < size; row++)
    for (col = 0; col < size; col++) {
      ip = code[row][col] + 1;
      f = fcol(row, col);
      memset(sum, 0, sizeof sum);
      for (y = -1; y <= 1; y++)
        for (x = -1; x <= 1; x++) {
          shift = (y == 0) + (x == 0);
          color = fcol(row + y, col + x);
          if (color == f)
            continue;
          *ip++ = (width * y + x) * 4 + color;
          *ip++ = shift;
          *ip++ = color;
          sum[color] += 1 << shift;
        }
      code[row][col][0] = (ip - code[row][col]) / 3;
      FORCC
      if (c != f) {
        *ip++ = c;
        *ip++ = 256 / sum[c];
      }
    }
  for (row = 1; row < height - 1; row++)
    for (col = 1; col < width - 1; col++) {
      pix = image[row * width + col];
      ip = code[row % size][col % size];
      memset(sum, 0, sizeof sum);
      for (i = *ip++; i--; ip += 3)
        sum[ip[2]] += pix[ip[0]] << ip[1];
      for (i = colors; --i; ip += 2)
        pix[ip[0]] = sum[ip[0]] * ip[1] >> 8;
    }
}

/*
This algorithm is officially called:

"Interpolation using a Threshold-based variable number of gradients"

described in
http://scien.stanford.edu/pages/labsite/1999/psych221/projects/99/tingchen/algodep/vargra.html

I've extended the basic idea to work with non-Bayer filter arrays.
Gradients are numbered clockwise from NW=0 to W=7.
*/
void CLASS vng_interpolate() {
  static const signed char terms[] = {-2, -2, +0, -1, 0, 0x01,
                                      -2, -2, +0, +0, 1, 0x01,
                                      -2, -1, -1, +0, 0, 0x01,
                                      -2, -1, +0, -1, 0, 0x02,
                                      -2, -1, +0, +0, 0, 0x03,
                                      -2, -1, +0, +1, 1, 0x01,
                                      -2, +0, +0, -1, 0, 0x06,
                                      -2, +0, +0, +0, 1, 0x02,
                                      -2, +0, +0, +1, 0, 0x03,
                                      -2, +1, -1, +0, 0, 0x04,
                                      -2, +1, +0, -1, 1, 0x04,
                                      -2, +1, +0, +0, 0, 0x06,
                                      -2, +1, +0, +1, 0, 0x02,
                                      -2, +2, +0, +0, 1, 0x04,
                                      -2, +2, +0, +1, 0, 0x04,
                                      -1, -2, -1, +0, 0, (signed char)0x80,
                                      -1, -2, +0, -1, 0, 0x01,
                                      -1, -2, +1, -1, 0, 0x01,
                                      -1, -2, +1, +0, 1, 0x01,
                                      -1, -1, -1, +1, 0, (signed char)0x88,
                                      -1, -1, +1, -2, 0, 0x40,
                                      -1, -1, +1, -1, 0, 0x22,
                                      -1, -1, +1, +0, 0, 0x33,
                                      -1, -1, +1, +1, 1, 0x11,
                                      -1, +0, -1, +2, 0, 0x08,
                                      -1, +0, +0, -1, 0, 0x44,
                                      -1, +0, +0, +1, 0, 0x11,
                                      -1, +0, +1, -2, 1, 0x40,
                                      -1, +0, +1, -1, 0, 0x66,
                                      -1, +0, +1, +0, 1, 0x22,
                                      -1, +0, +1, +1, 0, 0x33,
                                      -1, +0, +1, +2, 1, 0x10,
                                      -1, +1, +1, -1, 1, 0x44,
                                      -1, +1, +1, +0, 0, 0x66,
                                      -1, +1, +1, +1, 0, 0x22,
                                      -1, +1, +1, +2, 0, 0x10,
                                      -1, +2, +0, +1, 0, 0x04,
                                      -1, +2, +1, +0, 1, 0x04,
                                      -1, +2, +1, +1, 0, 0x04,
                                      +0, -2, +0, +0, 1, (signed char)0x80,
                                      +0, -1, +0, +1, 1, (signed char)0x88,
                                      +0, -1, +1, -2, 0, 0x40,
                                      +0, -1, +1, +0, 0, 0x11,
                                      +0, -1, +2, -2, 0, 0x40,
                                      +0, -1, +2, -1, 0, 0x20,
                                      +0, -1, +2, +0, 0, 0x30,
                                      +0, -1, +2, +1, 1, 0x10,
                                      +0, +0, +0, +2, 1, 0x08,
                                      +0, +0, +2, -2, 1, 0x40,
                                      +0, +0, +2, -1, 0, 0x60,
                                      +0, +0, +2, +0, 1, 0x20,
                                      +0, +0, +2, +1, 0, 0x30,
                                      +0, +0, +2, +2, 1, 0x10,
                                      +0, +1, +1, +0, 0, 0x44,
                                      +0, +1, +1, +2, 0, 0x10,
                                      +0, +1, +2, -1, 1, 0x40,
                                      +0, +1, +2, +0, 0, 0x60,
                                      +0, +1, +2, +1, 0, 0x20,
                                      +0, +1, +2, +2, 0, 0x10,
                                      +1, -2, +1, +0, 0, (signed char)0x80,
                                      +1, -1, +1, +1, 0, (signed char)0x88,
                                      +1, +0, +1, +2, 0, 0x08,
                                      +1, +0, +2, -1, 0, 0x40,
                                      +1, +0, +2, +1, 0, 0x10};
  static const signed char *cp,
      chood[] = {-1, -1, -1, 0, -1, +1, 0, +1, +1, +1, +1, 0, +1, -1, 0, -1};
  ushort(*brow[5])[4], *pix;
  int prow = 8, pcol = 2, *ip, *code[16][16], gval[8], gmin, gmax, sum[4];
  int row, col, x, y, x1, x2, y1, y2, t, weight, grads, color, diag;
  int g, diff, thold, num, c;

  lin_interpolate();
  if (verbose)
    fprintf(stderr, _("VNG interpolation...\n"));

  if (filters == 1)
    prow = pcol = 16;
  if (filters == 9)
    prow = pcol = 6;
  ip = (int *)calloc(prow * pcol, 1280);
  merror(ip, "vng_interpolate()");
  for (row = 0; row < prow; row++) /* Precalculate for VNG */
    for (col = 0; col < pcol; col++) {
      code[row][col] = ip;
      for (cp = terms, t = 0; t < 64; t++) {
        y1 = *cp++;
        x1 = *cp++;
        y2 = *cp++;
        x2 = *cp++;
        weight = *cp++;
        grads = *cp++;
        color = fcol(row + y1, col + x1);
        if (fcol(row + y2, col + x2) != color)
          continue;
        diag = (fcol(row, col + 1) == color && fcol(row + 1, col) == color) ? 2
                                                                            : 1;
        if (abs(y1 - y2) == diag && abs(x1 - x2) == diag)
          continue;
        *ip++ = (y1 * width + x1) * 4 + color;
        *ip++ = (y2 * width + x2) * 4 + color;
        *ip++ = weight;
        for (g = 0; g < 8; g++)
          if (grads & 1 << g)
            *ip++ = g;
        *ip++ = -1;
      }
      *ip++ = INT_MAX;
      for (cp = chood, g = 0; g < 8; g++) {
        y = *cp++;
        x = *cp++;
        *ip++ = (y * width + x) * 4;
        color = fcol(row, col);
        if (fcol(row + y, col + x) != color &&
            fcol(row + y * 2, col + x * 2) == color)
          *ip++ = (y * width + x) * 8 + color;
        else
          *ip++ = 0;
      }
    }
  brow[4] = (ushort(*)[4])calloc(width * 3, sizeof **brow);
  merror(brow[4], "vng_interpolate()");
  for (row = 0; row < 3; row++)
    brow[row] = brow[4] + row * width;
  for (row = 2; row < height - 2; row++) { /* Do VNG interpolation */
    for (col = 2; col < width - 2; col++) {
      pix = image[row * width + col];
      ip = code[row % prow][col % pcol];
      memset(gval, 0, sizeof gval);
      while ((g = ip[0]) != INT_MAX) { /* Calculate gradients */
        diff = ABS(pix[g] - pix[ip[1]]) << ip[2];
        gval[ip[3]] += diff;
        ip += 5;
        if ((g = ip[-1]) == -1)
          continue;
        gval[g] += diff;
        while ((g = *ip++) != -1)
          gval[g] += diff;
      }
      ip++;
      gmin = gmax = gval[0]; /* Choose a threshold */
      for (g = 1; g < 8; g++) {
        if (gmin > gval[g])
          gmin = gval[g];
        if (gmax < gval[g])
          gmax = gval[g];
      }
      if (gmax == 0) {
        memcpy(brow[2][col], pix, sizeof *image);
        continue;
      }
      thold = gmin + (gmax >> 1);
      memset(sum, 0, sizeof sum);
      color = fcol(row, col);
      for (num = g = 0; g < 8; g++, ip += 2) { /* Average the neighbors */
        if (gval[g] <= thold) {
          FORCC
          if (c == color && ip[1])
            sum[c] += (pix[c] + pix[ip[1]]) >> 1;
          else
            sum[c] += pix[ip[0] + c];
          num++;
        }
      }
      FORCC { /* Save to buffer */
        t = pix[color];
        if (c != color)
          t += (sum[c] - sum[color]) / num;
        brow[2][col][c] = CLIP(t);
      }
    }
    if (row > 3) /* Write buffer to image */
      memcpy(image[(row - 2) * width + 2], brow[0] + 2,
             (width - 4) * sizeof *image);
    for (g = 0; g < 4; g++)
      brow[(g - 1) & 3] = brow[g];
  }
  memcpy(image[(row - 2) * width + 2], brow[0] + 2,
         (width - 4) * sizeof *image);
  memcpy(image[(row - 1) * width + 2], brow[1] + 2,
         (width - 4) * sizeof *image);
  free(brow[4]);
  free(code[0][0]);
}

/*
Patterned Pixel Grouping Interpolation by Alain Desbiolles
*/
void CLASS ppg_interpolate() {
  int dir[5] = {1, width, -1, -width, 1};
  int row, col, diff[2], guess[2], c, d, i;
  ushort(*pix)[4];

  border_interpolate(3);
  if (verbose)
    fprintf(stderr, _("PPG interpolation...\n"));

  /*  Fill in the green layer with gradients and pattern recognition: */
  for (row = 3; row < height - 3; row++)
    for (col = 3 + (FC(row, 3) & 1), c = FC(row, col); col < width - 3;
         col += 2) {
      pix = image + row * width + col;
      for (i = 0; (d = dir[i]) > 0; i++) {
        guess[i] = (pix[-d][1] + pix[0][c] + pix[d][1]) * 2 - pix[-2 * d][c] -
                   pix[2 * d][c];
        diff[i] =
            (ABS(pix[-2 * d][c] - pix[0][c]) + ABS(pix[2 * d][c] - pix[0][c]) +
             ABS(pix[-d][1] - pix[d][1])) *
                3 +
            (ABS(pix[3 * d][1] - pix[d][1]) +
             ABS(pix[-3 * d][1] - pix[-d][1])) *
                2;
      }
      d = dir[i = diff[0] > diff[1]];
      pix[0][1] = ULIM(guess[i] >> 2, pix[d][1], pix[-d][1]);
    }
  /*  Calculate red and blue for each green pixel:        */
  for (row = 1; row < height - 1; row++)
    for (col = 1 + (FC(row, 2) & 1), c = FC(row, col + 1); col < width - 1;
         col += 2) {
      pix = image + row * width + col;
      for (i = 0; (d = dir[i]) > 0; c = 2 - c, i++)
        pix[0][c] = CLIP(
            (pix[-d][c] + pix[d][c] + 2 * pix[0][1] - pix[-d][1] - pix[d][1]) >>
            1);
    }
  /*  Calculate blue for red pixels and vice versa:        */
  for (row = 1; row < height - 1; row++)
    for (col = 1 + (FC(row, 1) & 1), c = 2 - FC(row, col); col < width - 1;
         col += 2) {
      pix = image + row * width + col;
      for (i = 0; (d = dir[i] + dir[i + 1]) > 0; i++) {
        diff[i] = ABS(pix[-d][c] - pix[d][c]) + ABS(pix[-d][1] - pix[0][1]) +
                  ABS(pix[d][1] - pix[0][1]);
        guess[i] =
            pix[-d][c] + pix[d][c] + 2 * pix[0][1] - pix[-d][1] - pix[d][1];
      }
      if (diff[0] != diff[1])
        pix[0][c] = CLIP(guess[diff[0] > diff[1]] >> 1);
      else
        pix[0][c] = CLIP((guess[0] + guess[1]) >> 2);
    }
}

void CLASS cielab(ushort rgb[3], short lab[3]) {
  int c, i, j, k;
  float r, xyz[3];
  static float cbrt[0x10000], xyz_cam[3][4];

  if (!rgb) {
    for (i = 0; i < 0x10000; i++) {
      r = i / 65535.0;
      cbrt[i] = r > 0.008856 ? pow(r, 1 / 3.0) : 7.787 * r + 16 / 116.0;
    }
    for (i = 0; i < 3; i++)
      for (j = 0; j < colors; j++)
        for (xyz_cam[i][j] = k = 0; k < 3; k++)
          xyz_cam[i][j] += xyz_rgb[i][k] * rgb_cam[k][j] / d65_white[i];
    return;
  }
  xyz[0] = xyz[1] = xyz[2] = 0.5;
  FORCC {
    xyz[0] += xyz_cam[0][c] * rgb[c];
    xyz[1] += xyz_cam[1][c] * rgb[c];
    xyz[2] += xyz_cam[2][c] * rgb[c];
  }
  xyz[0] = cbrt[CLIP((int)xyz[0])];
  xyz[1] = cbrt[CLIP((int)xyz[1])];
  xyz[2] = cbrt[CLIP((int)xyz[2])];
  lab[0] = 64 * (116 * xyz[1] - 16);
  lab[1] = 64 * 500 * (xyz[0] - xyz[1]);
  lab[2] = 64 * 200 * (xyz[1] - xyz[2]);
}

#define TS 512 /* Tile Size */
#define fcol(row, col) xtrans[(row + 6) % 6][(col + 6) % 6]

/*
Frank Markesteijn's algorithm for Fuji X-Trans sensors
*/
void CLASS xtrans_interpolate(int passes) {
  int c, d, f, g, h, i, v, ng, row, col, top, left, mrow, mcol;
  int val, ndir, pass, hm[8], avg[4], color[3][8];
  static const short orth[12] = {1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 1},
                     patt[2][16] = {{0, 1, 0, -1, 2, 0, -1, 0, 1, 1, 1, -1, 0,
                                     0, 0, 0},
                                    {0, 1, 0, -2, 1, 0, -2, 0, 1, 1, -2, -2, 1,
                                     -1, -1, 1}},
                     dir[4] = {1, TS, TS + 1, TS - 1};
  short allhex[3][3][2][8], *hex;
  ushort min, max, sgrow, sgcol;
  ushort(*rgb)[TS][TS][3], (*rix)[3], (*pix)[4];
  short(*lab)[TS][3], (*lix)[3];
  float(*drv)[TS][TS], diff[6], tr;
  char(*homo)[TS][TS], *buffer;

  if (verbose)
    fprintf(stderr, _("%d-pass X-Trans interpolation...\n"), passes);

  cielab(0, 0);
  ndir = 4 << (passes > 1);
  buffer = (char *)malloc(TS * TS * (ndir * 11 + 6));
  merror(buffer, "xtrans_interpolate()");
  rgb = (ushort(*)[TS][TS][3])buffer;
  lab = (short(*)[TS][3])(buffer + TS * TS * (ndir * 6));
  drv = (float(*)[TS][TS])(buffer + TS * TS * (ndir * 6 + 6));
  homo = (char(*)[TS][TS])(buffer + TS * TS * (ndir * 10 + 6));

  /* Map a green hexagon around each non-green pixel and vice versa:    */
  for (row = 0; row < 3; row++)
    for (col = 0; col < 3; col++)
      for (ng = d = 0; d < 10; d += 2) {
        g = fcol(row, col) == 1;
        if (fcol(row + orth[d], col + orth[d + 2]) == 1)
          ng = 0;
        else
          ng++;
        if (ng == 4) {
          sgrow = row;
          sgcol = col;
        }
        if (ng == g + 1)
          FORC(8) {
            v = orth[d] * patt[g][c * 2] + orth[d + 1] * patt[g][c * 2 + 1];
            h = orth[d + 2] * patt[g][c * 2] + orth[d + 3] * patt[g][c * 2 + 1];
            allhex[row][col][0][c ^ (g * 2 & d)] = h + v * width;
            allhex[row][col][1][c ^ (g * 2 & d)] = h + v * TS;
          }
      }

  /* Set green1 and green3 to the minimum and maximum allowed values:    */
  for (row = 2; row < height - 2; row++)
    for (min = ~(max = 0), col = 2; col < width - 2; col++) {
      if (fcol(row, col) == 1 && (min = ~(max = 0)))
        continue;
      pix = image + row * width + col;
      hex = allhex[row % 3][col % 3][0];
      if (!max)
        FORC(6) {
          val = pix[hex[c]][1];
          if (min > val)
            min = val;
          if (max < val)
            max = val;
        }
      pix[0][1] = min;
      pix[0][3] = max;
      switch ((row - sgrow) % 3) {
        case 1:
          if (row < height - 3) {
            row++;
            col--;
          }
          break;
        case 2:
          if ((min = ~(max = 0)) && (col += 2) < width - 3 && row > 2)
            row--;
      }
    }

  for (top = 3; top < height - 19; top += TS - 16)
    for (left = 3; left < width - 19; left += TS - 16) {
      mrow = MIN(top + TS, height - 3);
      mcol = MIN(left + TS, width - 3);
      for (row = top; row < mrow; row++)
        for (col = left; col < mcol; col++)
          memcpy(rgb[0][row - top][col - left], image[row * width + col], 6);
      FORC3 memcpy(rgb[c + 1], rgb[0], sizeof *rgb);

      /* Interpolate green horizontally, vertically, and along both diagonals:
       */
      for (row = top; row < mrow; row++)
        for (col = left; col < mcol; col++) {
          if ((f = fcol(row, col)) == 1)
            continue;
          pix = image + row * width + col;
          hex = allhex[row % 3][col % 3][0];
          color[1][0] = 174 * (pix[hex[1]][1] + pix[hex[0]][1]) -
                        46 * (pix[2 * hex[1]][1] + pix[2 * hex[0]][1]);
          color[1][1] = 223 * pix[hex[3]][1] + pix[hex[2]][1] * 33 +
                        92 * (pix[0][f] - pix[-hex[2]][f]);
          FORC(2)
          color[1][2 + c] = 164 * pix[hex[4 + c]][1] +
                            92 * pix[-2 * hex[4 + c]][1] +
                            33 * (2 * pix[0][f] - pix[3 * hex[4 + c]][f] -
                                  pix[-3 * hex[4 + c]][f]);
          FORC4 rgb[c ^ !((row - sgrow) % 3)][row - top][col - left][1] =
              LIM(color[1][c] >> 8, pix[0][1], pix[0][3]);
        }

      for (pass = 0; pass < passes; pass++) {
        if (pass == 1)
          memcpy(rgb += 4, buffer, 4 * sizeof *rgb);

        /* Recalculate green from interpolated values of closer pixels:    */
        if (pass) {
          for (row = top + 2; row < mrow - 2; row++)
            for (col = left + 2; col < mcol - 2; col++) {
              if ((f = fcol(row, col)) == 1)
                continue;
              pix = image + row * width + col;
              hex = allhex[row % 3][col % 3][1];
              for (d = 3; d < 6; d++) {
                rix =
                    &rgb[(d - 2) ^ !((row - sgrow) % 3)][row - top][col - left];
                val = rix[-2 * hex[d]][1] + 2 * rix[hex[d]][1] -
                      rix[-2 * hex[d]][f] - 2 * rix[hex[d]][f] + 3 * rix[0][f];
                rix[0][1] = LIM(val / 3, pix[0][1], pix[0][3]);
              }
            }
        }

        /* Interpolate red and blue values for solitary green pixels:    */
        for (row = (top - sgrow + 4) / 3 * 3 + sgrow; row < mrow - 2; row += 3)
          for (col = (left - sgcol + 4) / 3 * 3 + sgcol; col < mcol - 2;
               col += 3) {
            rix = &rgb[0][row - top][col - left];
            h = fcol(row, col + 1);
            memset(diff, 0, sizeof diff);
            for (i = 1, d = 0; d < 6; d++, i ^= TS ^ 1, h ^= 2) {
              for (c = 0; c < 2; c++, h ^= 2) {
                g = 2 * rix[0][1] - rix[i << c][1] - rix[-i << c][1];
                color[h][d] = g + rix[i << c][h] + rix[-i << c][h];
                if (d > 1)
                  diff[d] += SQR(rix[i << c][1] - rix[-i << c][1] -
                                 rix[i << c][h] + rix[-i << c][h]) +
                             SQR(g);
              }
              if (d > 1 && (d & 1))
                if (diff[d - 1] < diff[d])
                  FORC(2) color[c * 2][d] = color[c * 2][d - 1];
              if (d < 2 || (d & 1)) {
                FORC(2) rix[0][c * 2] = CLIP(color[c * 2][d] / 2);
                rix += TS * TS;
              }
            }
          }

        /* Interpolate red for blue pixels and vice versa:        */
        for (row = top + 3; row < mrow - 3; row++)
          for (col = left + 3; col < mcol - 3; col++) {
            if ((f = 2 - fcol(row, col)) == 1)
              continue;
            rix = &rgb[0][row - top][col - left];
            c = (row - sgrow) % 3 ? TS : 1;
            h = 3 * (c ^ TS ^ 1);
            for (d = 0; d < 4; d++, rix += TS * TS) {
              i = d > 1 || ((d ^ c) & 1) || ((ABS(rix[0][1] - rix[c][1]) +
                                              ABS(rix[0][1] - rix[-c][1])) <
                                             2 * (ABS(rix[0][1] - rix[h][1]) +
                                                  ABS(rix[0][1] - rix[-h][1])))
                      ? c
                      : h;
              rix[0][f] = CLIP((rix[i][f] + rix[-i][f] + 2 * rix[0][1] -
                                rix[i][1] - rix[-i][1]) /
                               2);
            }
          }

        /* Fill in red and blue for 2x2 blocks of green:        */
        for (row = top + 2; row < mrow - 2; row++)
          if ((row - sgrow) % 3)
            for (col = left + 2; col < mcol - 2; col++)
              if ((col - sgcol) % 3) {
                rix = &rgb[0][row - top][col - left];
                hex = allhex[row % 3][col % 3][1];
                for (d = 0; d < ndir; d += 2, rix += TS * TS)
                  if (hex[d] + hex[d + 1]) {
                    g = 3 * rix[0][1] - 2 * rix[hex[d]][1] - rix[hex[d + 1]][1];
                    for (c = 0; c < 4; c += 2)
                      rix[0][c] = CLIP(
                          (g + 2 * rix[hex[d]][c] + rix[hex[d + 1]][c]) / 3);
                  } else {
                    g = 2 * rix[0][1] - rix[hex[d]][1] - rix[hex[d + 1]][1];
                    for (c = 0; c < 4; c += 2)
                      rix[0][c] =
                          CLIP((g + rix[hex[d]][c] + rix[hex[d + 1]][c]) / 2);
                  }
              }
      }
      rgb = (ushort(*)[TS][TS][3])buffer;
      mrow -= top;
      mcol -= left;

      /* Convert to CIELab and differentiate in all directions:    */
      for (d = 0; d < ndir; d++) {
        for (row = 2; row < mrow - 2; row++)
          for (col = 2; col < mcol - 2; col++)
            cielab(rgb[d][row][col], lab[row][col]);
        for (f = dir[d & 3], row = 3; row < mrow - 3; row++)
          for (col = 3; col < mcol - 3; col++) {
            lix = &lab[row][col];
            g = 2 * lix[0][0] - lix[f][0] - lix[-f][0];
            drv[d][row][col] =
                SQR(g) +
                SQR((2 * lix[0][1] - lix[f][1] - lix[-f][1] + g * 500 / 232)) +
                SQR((2 * lix[0][2] - lix[f][2] - lix[-f][2] - g * 500 / 580));
          }
      }

      /* Build homogeneity maps from the derivatives:            */
      memset(homo, 0, ndir * TS * TS);
      for (row = 4; row < mrow - 4; row++)
        for (col = 4; col < mcol - 4; col++) {
          for (tr = FLT_MAX, d = 0; d < ndir; d++)
            if (tr > drv[d][row][col])
              tr = drv[d][row][col];
          tr *= 8;
          for (d = 0; d < ndir; d++)
            for (v = -1; v <= 1; v++)
              for (h = -1; h <= 1; h++)
                if (drv[d][row + v][col + h] <= tr)
                  homo[d][row][col]++;
        }

      /* Average the most homogenous pixels for the final result:    */
      if (height - top < TS + 4)
        mrow = height - top + 2;
      if (width - left < TS + 4)
        mcol = width - left + 2;
      for (row = MIN(top, 8); row < mrow - 8; row++)
        for (col = MIN(left, 8); col < mcol - 8; col++) {
          for (d = 0; d < ndir; d++)
            for (hm[d] = 0, v = -2; v <= 2; v++)
              for (h = -2; h <= 2; h++)
                hm[d] += homo[d][row + v][col + h];
          for (d = 0; d < ndir - 4; d++)
            if (hm[d] < hm[d + 4])
              hm[d] = 0;
            else if (hm[d] > hm[d + 4])
              hm[d + 4] = 0;
          for (max = hm[0], d = 1; d < ndir; d++)
            if (max < hm[d])
              max = hm[d];
          max -= max >> 3;
          memset(avg, 0, sizeof avg);
          for (d = 0; d < ndir; d++)
            if (hm[d] >= max) {
              FORC3 avg[c] += rgb[d][row][col][c];
              avg[3]++;
            }
          FORC3 image[(row + top) * width + col + left][c] = avg[c] / avg[3];
        }
    }
  free(buffer);
  border_interpolate(8);
}
#undef fcol

/*
Adaptive Homogeneity-Directed interpolation is based on
the work of Keigo Hirakawa, Thomas Parks, and Paul Lee.
*/
void CLASS ahd_interpolate() {
  int i, j, top, left, row, col, tr, tc, c, d, val, hm[2];
  static const int dir[4] = {-1, 1, -TS, TS};
  unsigned ldiff[2][4], abdiff[2][4], leps, abeps;
  ushort(*rgb)[TS][TS][3], (*rix)[3], (*pix)[4];
  short(*lab)[TS][TS][3], (*lix)[3];
  char(*homo)[TS][TS], *buffer;

  if (verbose)
    fprintf(stderr, _("AHD interpolation...\n"));

  cielab(0, 0);
  border_interpolate(5);
  buffer = (char *)malloc(26 * TS * TS);
  merror(buffer, "ahd_interpolate()");
  rgb = (ushort(*)[TS][TS][3])buffer;
  lab = (short(*)[TS][TS][3])(buffer + 12 * TS * TS);
  homo = (char(*)[TS][TS])(buffer + 24 * TS * TS);

  for (top = 2; top < height - 5; top += TS - 6)
    for (left = 2; left < width - 5; left += TS - 6) {
      /*  Interpolate green horizontally and vertically:        */
      for (row = top; row < top + TS && row < height - 2; row++) {
        col = left + (FC(row, left) & 1);
        for (c = FC(row, col); col < left + TS && col < width - 2; col += 2) {
          pix = image + row * width + col;
          val = ((pix[-1][1] + pix[0][c] + pix[1][1]) * 2 - pix[-2][c] -
                 pix[2][c]) >>
                2;
          rgb[0][row - top][col - left][1] = ULIM(val, pix[-1][1], pix[1][1]);
          val = ((pix[-width][1] + pix[0][c] + pix[width][1]) * 2 -
                 pix[-2 * width][c] - pix[2 * width][c]) >>
                2;
          rgb[1][row - top][col - left][1] =
              ULIM(val, pix[-width][1], pix[width][1]);
        }
      }
      /*  Interpolate red and blue, and convert to CIELab:        */
      for (d = 0; d < 2; d++)
        for (row = top + 1; row < top + TS - 1 && row < height - 3; row++)
          for (col = left + 1; col < left + TS - 1 && col < width - 3; col++) {
            pix = image + row * width + col;
            rix = &rgb[d][row - top][col - left];
            lix = &lab[d][row - top][col - left];
            if ((c = 2 - FC(row, col)) == 1) {
              c = FC(row + 1, col);
              val =
                  pix[0][1] +
                  ((pix[-1][2 - c] + pix[1][2 - c] - rix[-1][1] - rix[1][1]) >>
                   1);
              rix[0][2 - c] = CLIP(val);
              val = pix[0][1] + ((pix[-width][c] + pix[width][c] - rix[-TS][1] -
                                  rix[TS][1]) >>
                                 1);
            } else
              val = rix[0][1] + ((pix[-width - 1][c] + pix[-width + 1][c] +
                                  pix[+width - 1][c] + pix[+width + 1][c] -
                                  rix[-TS - 1][1] - rix[-TS + 1][1] -
                                  rix[+TS - 1][1] - rix[+TS + 1][1] + 1) >>
                                 2);
            rix[0][c] = CLIP(val);
            c = FC(row, col);
            rix[0][c] = pix[0][c];
            cielab(rix[0], lix[0]);
          }
      /*  Build homogeneity maps from the CIELab images:        */
      memset(homo, 0, 2 * TS * TS);
      for (row = top + 2; row < top + TS - 2 && row < height - 4; row++) {
        tr = row - top;
        for (col = left + 2; col < left + TS - 2 && col < width - 4; col++) {
          tc = col - left;
          for (d = 0; d < 2; d++) {
            lix = &lab[d][tr][tc];
            for (i = 0; i < 4; i++) {
              ldiff[d][i] = ABS(lix[0][0] - lix[dir[i]][0]);
              abdiff[d][i] = SQR(lix[0][1] - lix[dir[i]][1]) +
                             SQR(lix[0][2] - lix[dir[i]][2]);
            }
          }
          leps =
              MIN(MAX(ldiff[0][0], ldiff[0][1]), MAX(ldiff[1][2], ldiff[1][3]));
          abeps = MIN(MAX(abdiff[0][0], abdiff[0][1]),
                      MAX(abdiff[1][2], abdiff[1][3]));
          for (d = 0; d < 2; d++)
            for (i = 0; i < 4; i++)
              if (ldiff[d][i] <= leps && abdiff[d][i] <= abeps)
                homo[d][tr][tc]++;
        }
      }
      /*  Combine the most homogenous pixels for the final result:    */
      for (row = top + 3; row < top + TS - 3 && row < height - 5; row++) {
        tr = row - top;
        for (col = left + 3; col < left + TS - 3 && col < width - 5; col++) {
          tc = col - left;
          for (d = 0; d < 2; d++)
            for (hm[d] = 0, i = tr - 1; i <= tr + 1; i++)
              for (j = tc - 1; j <= tc + 1; j++)
                hm[d] += homo[d][i][j];
          if (hm[0] != hm[1])
            FORC3 image[row * width + col][c] = rgb[hm[1] > hm[0]][tr][tc][c];
          else
            FORC3 image[row * width + col][c] =
                (rgb[0][tr][tc][c] + rgb[1][tr][tc][c]) >> 1;
        }
      }
    }
  free(buffer);
}
#undef TS

void CLASS median_filter() {
  ushort(*pix)[4];
  int pass, c, i, j, k, med[9];
  static const uchar opt[] = /* Optimal 9-element median search */
      {1, 2, 4, 5, 7, 8, 0, 1, 3, 4, 6, 7, 1, 2, 4, 5, 7, 8, 0,
       3, 5, 8, 4, 7, 3, 6, 1, 4, 2, 5, 4, 7, 4, 2, 6, 4, 4, 2};

  for (pass = 1; pass <= med_passes; pass++) {
    if (verbose)
      fprintf(stderr, _("Median filter pass %d...\n"), pass);
    for (c = 0; c < 3; c += 2) {
      for (pix = image; pix < image + width * height; pix++)
        pix[0][3] = pix[0][c];
      for (pix = image + width; pix < image + width * (height - 1); pix++) {
        if ((pix - image + 1) % width < 2)
          continue;
        for (k = 0, i = -width; i <= width; i += width)
          for (j = i - 1; j <= i + 1; j++)
            med[k++] = pix[j][3] - pix[j][1];
        for (i = 0; i < sizeof opt; i += 2)
          if (med[opt[i]] > med[opt[i + 1]])
            SWAP(med[opt[i]], med[opt[i + 1]]);
        pix[0][c] = CLIP(med[4] + pix[0][1]);
      }
    }
  }
}

void CLASS blend_highlights() {
  int clip = INT_MAX, row, col, c, i, j;
  static const float trans[2][4][4] = {
      {{1, 1, 1}, {1.7320508, -1.7320508, 0}, {-1, -1, 2}},
      {{1, 1, 1, 1}, {1, -1, 1, -1}, {1, 1, -1, -1}, {1, -1, -1, 1}}};
  static const float itrans[2][4][4] = {
      {{1, 0.8660254, -0.5}, {1, -0.8660254, -0.5}, {1, 0, 1}},
      {{1, 1, 1, 1}, {1, -1, 1, -1}, {1, 1, -1, -1}, {1, -1, -1, 1}}};
  float cam[2][4], lab[2][4], sum[2], chratio;

  if ((unsigned)(colors - 3) > 1)
    return;
  if (verbose)
    fprintf(stderr, _("Blending highlights...\n"));
  FORCC if (clip > (i = 65535 * pre_mul[c])) clip = i;
  for (row = 0; row < height; row++)
    for (col = 0; col < width; col++) {
      FORCC if (image[row * width + col][c] > clip) break;
      if (c == colors)
        continue;
      FORCC {
        cam[0][c] = image[row * width + col][c];
        cam[1][c] = MIN(cam[0][c], clip);
      }
      for (i = 0; i < 2; i++) {
        FORCC for (lab[i][c] = j = 0; j < colors; j++) lab[i][c] +=
            trans[colors - 3][c][j] * cam[i][j];
        for (sum[i] = 0, c = 1; c < colors; c++)
          sum[i] += SQR(lab[i][c]);
      }
      chratio = sqrt(sum[1] / sum[0]);
      for (c = 1; c < colors; c++)
        lab[0][c] *= chratio;
      FORCC for (cam[0][c] = j = 0; j < colors; j++) cam[0][c] +=
          itrans[colors - 3][c][j] * lab[0][j];
      FORCC image[row * width + col][c] = cam[0][c] / colors;
    }
}

#define SCALE (4 >> shrink)
void CLASS recover_highlights() {
  float *map, sum, wgt, grow;
  int hsat[4], count, spread, change, val, i;
  unsigned high, wide, mrow, mcol, row, col, kc, c, d, y, x;
  ushort *pixel;
  static const signed char dir[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, 1},
                                        {1, 1},   {1, 0},  {1, -1}, {0, -1}};

  if (verbose)
    fprintf(stderr, _("Rebuilding highlights...\n"));

  grow = pow(2, 4 - highlight);
  FORCC hsat[c] = 32000 * pre_mul[c];
  for (kc = 0, c = 1; c < colors; c++)
    if (pre_mul[kc] < pre_mul[c])
      kc = c;
  high = height / SCALE;
  wide = width / SCALE;
  map = (float *)calloc(high, wide * sizeof *map);
  merror(map, "recover_highlights()");
  FORCC if (c != kc) {
    memset(map, 0, high * wide * sizeof *map);
    for (mrow = 0; mrow < high; mrow++)
      for (mcol = 0; mcol < wide; mcol++) {
        sum = wgt = count = 0;
        for (row = mrow * SCALE; row < (mrow + 1) * SCALE; row++)
          for (col = mcol * SCALE; col < (mcol + 1) * SCALE; col++) {
            pixel = image[row * width + col];
            if (pixel[c] / hsat[c] == 1 && pixel[kc] > 24000) {
              sum += pixel[c];
              wgt += pixel[kc];
              count++;
            }
          }
        if (count == SCALE * SCALE)
          map[mrow * wide + mcol] = sum / wgt;
      }
    for (spread = 32 / grow; spread--;) {
      for (mrow = 0; mrow < high; mrow++)
        for (mcol = 0; mcol < wide; mcol++) {
          if (map[mrow * wide + mcol])
            continue;
          sum = count = 0;
          for (d = 0; d < 8; d++) {
            y = mrow + dir[d][0];
            x = mcol + dir[d][1];
            if (y < high && x < wide && map[y * wide + x] > 0) {
              sum += (1 + (d & 1)) * map[y * wide + x];
              count += 1 + (d & 1);
            }
          }
          if (count > 3)
            map[mrow * wide + mcol] = -(sum + grow) / (count + grow);
        }
      for (change = i = 0; i < high * wide; i++)
        if (map[i] < 0) {
          map[i] = -map[i];
          change = 1;
        }
      if (!change)
        break;
    }
    for (i = 0; i < high * wide; i++)
      if (map[i] == 0)
        map[i] = 1;
    for (mrow = 0; mrow < high; mrow++)
      for (mcol = 0; mcol < wide; mcol++) {
        for (row = mrow * SCALE; row < (mrow + 1) * SCALE; row++)
          for (col = mcol * SCALE; col < (mcol + 1) * SCALE; col++) {
            pixel = image[row * width + col];
            if (pixel[c] / hsat[c] > 1) {
              val = pixel[kc] * map[mrow * wide + mcol];
              if (pixel[c] < val)
                pixel[c] = CLIP(val);
            }
          }
      }
  }
  free(map);
}
#undef SCALE

void CLASS tiff_get(unsigned base,
                    unsigned *tag,
                    unsigned *type,
                    unsigned *len,
                    unsigned *save) {
  *tag = get2();
  *type = get2();
  *len = get4();
  *save = ftell(ifp) + 4;
  if (*len * ("11124811248484"[*type < 14 ? *type : 0] - '0') > 4)
    fseek(ifp, get4() + base, SEEK_SET);
}

void CLASS parse_thumb_note(int base, unsigned toff, unsigned tlen) {
  unsigned entries, tag, type, len, save;

  entries = get2();
  while (entries--) {
    tiff_get(base, &tag, &type, &len, &save);
    if (tag == toff)
      thumb_offset = get4() + base;
    if (tag == tlen)
      thumb_length = get4();
    fseek(ifp, save, SEEK_SET);
  }
}

int CLASS parse_tiff_ifd(int base);

void CLASS parse_makernote(int base, int uptag) {
  static const uchar xlat[2][256] = {
      {0xc1, 0xbf, 0x6d, 0x0d, 0x59, 0xc5, 0x13, 0x9d, 0x83, 0x61, 0x6b, 0x4f,
       0xc7, 0x7f, 0x3d, 0x3d, 0x53, 0x59, 0xe3, 0xc7, 0xe9, 0x2f, 0x95, 0xa7,
       0x95, 0x1f, 0xdf, 0x7f, 0x2b, 0x29, 0xc7, 0x0d, 0xdf, 0x07, 0xef, 0x71,
       0x89, 0x3d, 0x13, 0x3d, 0x3b, 0x13, 0xfb, 0x0d, 0x89, 0xc1, 0x65, 0x1f,
       0xb3, 0x0d, 0x6b, 0x29, 0xe3, 0xfb, 0xef, 0xa3, 0x6b, 0x47, 0x7f, 0x95,
       0x35, 0xa7, 0x47, 0x4f, 0xc7, 0xf1, 0x59, 0x95, 0x35, 0x11, 0x29, 0x61,
       0xf1, 0x3d, 0xb3, 0x2b, 0x0d, 0x43, 0x89, 0xc1, 0x9d, 0x9d, 0x89, 0x65,
       0xf1, 0xe9, 0xdf, 0xbf, 0x3d, 0x7f, 0x53, 0x97, 0xe5, 0xe9, 0x95, 0x17,
       0x1d, 0x3d, 0x8b, 0xfb, 0xc7, 0xe3, 0x67, 0xa7, 0x07, 0xf1, 0x71, 0xa7,
       0x53, 0xb5, 0x29, 0x89, 0xe5, 0x2b, 0xa7, 0x17, 0x29, 0xe9, 0x4f, 0xc5,
       0x65, 0x6d, 0x6b, 0xef, 0x0d, 0x89, 0x49, 0x2f, 0xb3, 0x43, 0x53, 0x65,
       0x1d, 0x49, 0xa3, 0x13, 0x89, 0x59, 0xef, 0x6b, 0xef, 0x65, 0x1d, 0x0b,
       0x59, 0x13, 0xe3, 0x4f, 0x9d, 0xb3, 0x29, 0x43, 0x2b, 0x07, 0x1d, 0x95,
       0x59, 0x59, 0x47, 0xfb, 0xe5, 0xe9, 0x61, 0x47, 0x2f, 0x35, 0x7f, 0x17,
       0x7f, 0xef, 0x7f, 0x95, 0x95, 0x71, 0xd3, 0xa3, 0x0b, 0x71, 0xa3, 0xad,
       0x0b, 0x3b, 0xb5, 0xfb, 0xa3, 0xbf, 0x4f, 0x83, 0x1d, 0xad, 0xe9, 0x2f,
       0x71, 0x65, 0xa3, 0xe5, 0x07, 0x35, 0x3d, 0x0d, 0xb5, 0xe9, 0xe5, 0x47,
       0x3b, 0x9d, 0xef, 0x35, 0xa3, 0xbf, 0xb3, 0xdf, 0x53, 0xd3, 0x97, 0x53,
       0x49, 0x71, 0x07, 0x35, 0x61, 0x71, 0x2f, 0x43, 0x2f, 0x11, 0xdf, 0x17,
       0x97, 0xfb, 0x95, 0x3b, 0x7f, 0x6b, 0xd3, 0x25, 0xbf, 0xad, 0xc7, 0xc5,
       0xc5, 0xb5, 0x8b, 0xef, 0x2f, 0xd3, 0x07, 0x6b, 0x25, 0x49, 0x95, 0x25,
       0x49, 0x6d, 0x71, 0xc7},
      {0xa7, 0xbc, 0xc9, 0xad, 0x91, 0xdf, 0x85, 0xe5, 0xd4, 0x78, 0xd5, 0x17,
       0x46, 0x7c, 0x29, 0x4c, 0x4d, 0x03, 0xe9, 0x25, 0x68, 0x11, 0x86, 0xb3,
       0xbd, 0xf7, 0x6f, 0x61, 0x22, 0xa2, 0x26, 0x34, 0x2a, 0xbe, 0x1e, 0x46,
       0x14, 0x68, 0x9d, 0x44, 0x18, 0xc2, 0x40, 0xf4, 0x7e, 0x5f, 0x1b, 0xad,
       0x0b, 0x94, 0xb6, 0x67, 0xb4, 0x0b, 0xe1, 0xea, 0x95, 0x9c, 0x66, 0xdc,
       0xe7, 0x5d, 0x6c, 0x05, 0xda, 0xd5, 0xdf, 0x7a, 0xef, 0xf6, 0xdb, 0x1f,
       0x82, 0x4c, 0xc0, 0x68, 0x47, 0xa1, 0xbd, 0xee, 0x39, 0x50, 0x56, 0x4a,
       0xdd, 0xdf, 0xa5, 0xf8, 0xc6, 0xda, 0xca, 0x90, 0xca, 0x01, 0x42, 0x9d,
       0x8b, 0x0c, 0x73, 0x43, 0x75, 0x05, 0x94, 0xde, 0x24, 0xb3, 0x80, 0x34,
       0xe5, 0x2c, 0xdc, 0x9b, 0x3f, 0xca, 0x33, 0x45, 0xd0, 0xdb, 0x5f, 0xf5,
       0x52, 0xc3, 0x21, 0xda, 0xe2, 0x22, 0x72, 0x6b, 0x3e, 0xd0, 0x5b, 0xa8,
       0x87, 0x8c, 0x06, 0x5d, 0x0f, 0xdd, 0x09, 0x19, 0x93, 0xd0, 0xb9, 0xfc,
       0x8b, 0x0f, 0x84, 0x60, 0x33, 0x1c, 0x9b, 0x45, 0xf1, 0xf0, 0xa3, 0x94,
       0x3a, 0x12, 0x77, 0x33, 0x4d, 0x44, 0x78, 0x28, 0x3c, 0x9e, 0xfd, 0x65,
       0x57, 0x16, 0x94, 0x6b, 0xfb, 0x59, 0xd0, 0xc8, 0x22, 0x36, 0xdb, 0xd2,
       0x63, 0x98, 0x43, 0xa1, 0x04, 0x87, 0x86, 0xf7, 0xa6, 0x26, 0xbb, 0xd6,
       0x59, 0x4d, 0xbf, 0x6a, 0x2e, 0xaa, 0x2b, 0xef, 0xe6, 0x78, 0xb6, 0x4e,
       0xe0, 0x2f, 0xdc, 0x7c, 0xbe, 0x57, 0x19, 0x32, 0x7e, 0x2a, 0xd0, 0xb8,
       0xba, 0x29, 0x00, 0x3c, 0x52, 0x7d, 0xa8, 0x49, 0x3b, 0x2d, 0xeb, 0x25,
       0x49, 0xfa, 0xa3, 0xaa, 0x39, 0xa7, 0xc5, 0xa7, 0x50, 0x11, 0x36, 0xfb,
       0xc6, 0x67, 0x4a, 0xf5, 0xa5, 0x12, 0x65, 0x7e, 0xb0, 0xdf, 0xaf, 0x4e,
       0xb3, 0x61, 0x7f, 0x2f}};
  unsigned offset = 0, entries, tag, type, len, save, c;
  unsigned ver97 = 0, serial = 0, i, wbi = 0, wb[4] = {0, 0, 0, 0};
  uchar buf97[324], ci, cj, ck;
  short morder, sorder = order;
  char buf[10];
  /*
  The MakerNote might have its own TIFF header (possibly with
  its own byte-order!), or it might just be a table.
  */
  if (!strcmp(make, "Nokia"))
    return;
  fread(buf, 1, 10, ifp);
  if (!strncmp(buf, "KDK", 3) || /* these aren't TIFF tables */
      !strncmp(buf, "VER", 3) || !strncmp(buf, "IIII", 4) ||
      !strncmp(buf, "MMMM", 4))
    return;
  if (!strncmp(buf, "KC", 2) ||  /* Konica KD-400Z, KD-510Z */
      !strncmp(buf, "MLY", 3)) { /* Minolta DiMAGE G series */
    order = 0x4d4d;
    while ((i = ftell(ifp)) < data_offset && i < 16384) {
      wb[0] = wb[2];
      wb[2] = wb[1];
      wb[1] = wb[3];
      wb[3] = get2();
      if (wb[1] == 256 && wb[3] == 256 && wb[0] > 256 && wb[0] < 640 &&
          wb[2] > 256 && wb[2] < 640)
        FORC4 cam_mul[c] = wb[c];
    }
    goto quit;
  }
  if (!strcmp(buf, "Nikon")) {
    base = ftell(ifp);
    order = get2();
    if (get2() != 42)
      goto quit;
    offset = get4();
    fseek(ifp, offset - 8, SEEK_CUR);
  } else if (!strcmp(buf, "OLYMPUS") || !strcmp(buf, "PENTAX ")) {
    base = ftell(ifp) - 10;
    fseek(ifp, -2, SEEK_CUR);
    order = get2();
    if (buf[0] == 'O')
      get2();
  } else if (!strncmp(buf, "SONY", 4) || !strcmp(buf, "Panasonic")) {
    goto nf;
  } else if (!strncmp(buf, "FUJIFILM", 8)) {
    base = ftell(ifp) - 10;
  nf:
    order = 0x4949;
    fseek(ifp, 2, SEEK_CUR);
  } else if (!strcmp(buf, "OLYMP") || !strcmp(buf, "LEICA") ||
             !strcmp(buf, "Ricoh") || !strcmp(buf, "EPSON"))
    fseek(ifp, -2, SEEK_CUR);
  else if (!strcmp(buf, "AOC") || !strcmp(buf, "QVC"))
    fseek(ifp, -4, SEEK_CUR);
  else {
    fseek(ifp, -10, SEEK_CUR);
    if (!strncmp(make, "SAMSUNG", 7))
      base = ftell(ifp);
  }
  entries = get2();
  if (entries > 1000)
    return;
  morder = order;
  while (entries--) {
    order = morder;
    tiff_get(base, &tag, &type, &len, &save);
    tag |= uptag << 16;
    if (tag == 2 && strstr(make, "NIKON") && !iso_speed)
      iso_speed = (get2(), get2());
    if (tag == 4 && len > 26 && len < 35) {
      if ((i = (get4(), get2())) != 0x7fff && !iso_speed)
        iso_speed = 50 * pow(2, i / 32.0 - 4);
      if ((i = (get2(), get2())) != 0x7fff && !aperture)
        aperture = pow(2, i / 64.0);
      if ((i = get2()) != 0xffff && !shutter)
        shutter = pow(2, (short)i / -32.0);
      wbi = (get2(), get2());
      shot_order = (get2(), get2());
    }
    if ((tag == 4 || tag == 0x114) && !strncmp(make, "KONICA", 6)) {
      fseek(ifp, tag == 4 ? 140 : 160, SEEK_CUR);
      switch (get2()) {
        case 72:
          flip = 0;
          break;
        case 76:
          flip = 6;
          break;
        case 82:
          flip = 5;
          break;
      }
    }
    if (tag == 7 && type == 2 && len > 20)
      fgets(model2, 64, ifp);
    if (tag == 8 && type == 4)
      shot_order = get4();
    if (tag == 9 && !strcmp(make, "Canon"))
      fread(artist, 64, 1, ifp);
    if (tag == 0xc && len == 4)
      FORC3 cam_mul[(c << 1 | c >> 1) & 3] = getreal(type);
    if (tag == 0xd && type == 7 && get2() == 0xaaaa) {
      for (c = i = 2; (ushort)c != 0xbbbb && i < len; i++)
        c = c << 8 | fgetc(ifp);
      while ((i += 4) < len - 5)
        if (get4() == 257 && (i = len) && (c = (get4(), fgetc(ifp))) < 3)
          flip = "065"[c] - '0';
    }
    if (tag == 0x10 && type == 4)
      unique_id = get4();
    if (tag == 0x11 && is_raw && !strncmp(make, "NIKON", 5)) {
      fseek(ifp, get4() + base, SEEK_SET);
      parse_tiff_ifd(base);
    }
    if (tag == 0x14 && type == 7) {
      if (len == 2560) {
        fseek(ifp, 1248, SEEK_CUR);
        goto get2_256;
      }
      fread(buf, 1, 10, ifp);
      if (!strncmp(buf, "NRW ", 4)) {
        fseek(ifp, strcmp(buf + 4, "0100") ? 46 : 1546, SEEK_CUR);
        cam_mul[0] = get4() << 2;
        cam_mul[1] = get4() + get4();
        cam_mul[2] = get4() << 2;
      }
    }
    if (tag == 0x15 && type == 2 && is_raw)
      fread(model, 64, 1, ifp);
    if (strstr(make, "PENTAX")) {
      if (tag == 0x1b)
        tag = 0x1018;
      if (tag == 0x1c)
        tag = 0x1017;
    }
    if (tag == 0x1d)
      while ((c = fgetc(ifp)) && c != EOF)
        serial = serial * 10 + (isdigit(c) ? c - '0' : c % 10);
    if (tag == 0x29 && type == 1) {
      c = wbi < 18 ? "012347800000005896"[wbi] - '0' : 0;
      fseek(ifp, 8 + c * 32, SEEK_CUR);
      FORC4 cam_mul[c ^ (c >> 1) ^ 1] = get4();
    }
    if (tag == 0x3d && type == 3 && len == 4)
      FORC4 cblack[c ^ c >> 1] = get2() >> (14 - tiff_bps);
    if (tag == 0x81 && type == 4) {
      data_offset = get4();
      fseek(ifp, data_offset + 41, SEEK_SET);
      raw_height = get2() * 2;
      raw_width = get2();
      filters = 0x61616161;
    }
    if ((tag == 0x81 && type == 7) || (tag == 0x100 && type == 7) ||
        (tag == 0x280 && type == 1)) {
      thumb_offset = ftell(ifp);
      thumb_length = len;
    }
    if (tag == 0x88 && type == 4 && (thumb_offset = get4()))
      thumb_offset += base;
    if (tag == 0x89 && type == 4)
      thumb_length = get4();
    if (tag == 0x8c || tag == 0x96)
      meta_offset = ftell(ifp);
    if (tag == 0x97) {
      for (i = 0; i < 4; i++)
        ver97 = ver97 * 10 + fgetc(ifp) - '0';
      switch (ver97) {
        case 100:
          fseek(ifp, 68, SEEK_CUR);
          FORC4 cam_mul[(c >> 1) | ((c & 1) << 1)] = get2();
          break;
        case 102:
          fseek(ifp, 6, SEEK_CUR);
          FORC4 cam_mul[c ^ (c >> 1)] = get2();
          break;
        case 103:
          fseek(ifp, 16, SEEK_CUR);
          FORC4 cam_mul[c] = get2();
      }
      if (ver97 >= 200) {
        if (ver97 != 205)
          fseek(ifp, 280, SEEK_CUR);
        fread(buf97, 324, 1, ifp);
      }
    }
    if (tag == 0xa1 && type == 7) {
      order = 0x4949;
      fseek(ifp, 140, SEEK_CUR);
      FORC3 cam_mul[c] = get4();
    }
    if (tag == 0xa4 && type == 3) {
      fseek(ifp, wbi * 48, SEEK_CUR);
      FORC3 cam_mul[c] = get2();
    }
    if (tag == 0xa7 && (unsigned)(ver97 - 200) < 17) {
      ci = xlat[0][serial & 0xff];
      cj = xlat[1][fgetc(ifp) ^ fgetc(ifp) ^ fgetc(ifp) ^ fgetc(ifp)];
      ck = 0x60;
      for (i = 0; i < 324; i++)
        buf97[i] ^= (cj += ci * ck++);
      i = "66666>666;6A;:;55"[ver97 - 200] - '0';
      FORC4 cam_mul[c ^ (c >> 1) ^ (i & 1)] = sget2(buf97 + (i & -2) + c * 2);
    }
    if (tag == 0x200 && len == 3)
      shot_order = (get4(), get4());
    if (tag == 0x200 && len == 4)
      FORC4 cblack[c ^ c >> 1] = get2();
    if (tag == 0x201 && len == 4)
      FORC4 cam_mul[c ^ (c >> 1)] = get2();
    if (tag == 0x220 && type == 7)
      meta_offset = ftell(ifp);
    if (tag == 0x401 && type == 4 && len == 4)
      FORC4 cblack[c ^ c >> 1] = get4();
    if (tag == 0xe01) { /* Nikon Capture Note */
      order = 0x4949;
      fseek(ifp, 22, SEEK_CUR);
      for (offset = 22; offset + 22 < len; offset += 22 + i) {
        tag = get4();
        fseek(ifp, 14, SEEK_CUR);
        i = get4() - 4;
        if (tag == 0x76a43207)
          flip = get2();
        else
          fseek(ifp, i, SEEK_CUR);
      }
    }
    if (tag == 0xe80 && len == 256 && type == 7) {
      fseek(ifp, 48, SEEK_CUR);
      cam_mul[0] = get2() * 508 * 1.078 / 0x10000;
      cam_mul[2] = get2() * 382 * 1.173 / 0x10000;
    }
    if (tag == 0xf00 && type == 7) {
      if (len == 614)
        fseek(ifp, 176, SEEK_CUR);
      else if (len == 734 || len == 1502)
        fseek(ifp, 148, SEEK_CUR);
      else
        goto next;
      goto get2_256;
    }
    if ((tag == 0x1011 && len == 9) || tag == 0x20400200)
      for (i = 0; i < 3; i++)
        FORC3 cmatrix[i][c] = ((short)get2()) / 256.0;
    if ((tag == 0x1012 || tag == 0x20400600) && len == 4)
      FORC4 cblack[c ^ c >> 1] = get2();
    if (tag == 0x1017 || tag == 0x20400100)
      cam_mul[0] = get2() / 256.0;
    if (tag == 0x1018 || tag == 0x20400100)
      cam_mul[2] = get2() / 256.0;
    if (tag == 0x2011 && len == 2) {
    get2_256:
      order = 0x4d4d;
      cam_mul[0] = get2() / 256.0;
      cam_mul[2] = get2() / 256.0;
    }
    if ((tag | 0x70) == 0x2070 && (type == 4 || type == 13))
      fseek(ifp, get4() + base, SEEK_SET);
    if (tag == 0x2020 && !strncmp(buf, "OLYMP", 5))
      parse_thumb_note(base, 257, 258);
    if (tag == 0x2040)
      parse_makernote(base, 0x2040);
    if (tag == 0xb028) {
      fseek(ifp, get4() + base, SEEK_SET);
      parse_thumb_note(base, 136, 137);
    }
    if (tag == 0x4001 && len > 500) {
      i = len == 582 ? 50 : len == 653 ? 68 : len == 5120 ? 142 : 126;
      fseek(ifp, i, SEEK_CUR);
      FORC4 cam_mul[c ^ (c >> 1)] = get2();
      for (i += 18; i <= len; i += 10) {
        get2();
        FORC4 sraw_mul[c ^ (c >> 1)] = get2();
        if (sraw_mul[1] == 1170)
          break;
      }
    }
    if (tag == 0x4021 && get4() && get4())
      FORC4 cam_mul[c] = 1024;
    if (tag == 0xa021)
      FORC4 cam_mul[c ^ (c >> 1)] = get4();
    if (tag == 0xa028)
      FORC4 cam_mul[c ^ (c >> 1)] -= get4();
    if (tag == 0xb001)
      unique_id = get2();
  next:
    fseek(ifp, save, SEEK_SET);
  }
quit:
  order = sorder;
}

/*
Since the TIFF DateTime string has no timezone information,
assume that the camera's clock was set to Universal Time.
*/
void CLASS get_timestamp(int reversed) {
  struct tm t;
  char str[20];
  int i;

  str[19] = 0;
  if (reversed)
    for (i = 19; i--;)
      str[i] = fgetc(ifp);
  else
    fread(str, 19, 1, ifp);
  memset(&t, 0, sizeof t);
  if (sscanf(str, "%d:%d:%d %d:%d:%d", &t.tm_year, &t.tm_mon, &t.tm_mday,
             &t.tm_hour, &t.tm_min, &t.tm_sec) != 6)
    return;
  t.tm_year -= 1900;
  t.tm_mon -= 1;
  t.tm_isdst = -1;
  if (mktime(&t) > 0)
    timestamp = mktime(&t);
}

void CLASS parse_exif(int base) {
  unsigned kodak, entries, tag, type, len, save, c;
  double expo;

  kodak = !strncmp(make, "EASTMAN", 7) && tiff_nifds < 3;
  entries = get2();
  while (entries--) {
    tiff_get(base, &tag, &type, &len, &save);
    switch (tag) {
      case 33434:
        tiff_ifd[tiff_nifds - 1].shutter = shutter = getreal(type);
        break;
      case 33437:
        aperture = getreal(type);
        break;
      case 34855:
        iso_speed = get2();
        break;
      case 36867:
      case 36868:
        get_timestamp(0);
        break;
      case 37377:
        if ((expo = -getreal(type)) < 128)
          tiff_ifd[tiff_nifds - 1].shutter = shutter = pow(2, expo);
        break;
      case 37378:
        aperture = pow(2, getreal(type) / 2);
        break;
      case 37386:
        focal_len = getreal(type);
        break;
      case 37500:
        parse_makernote(base, 0);
        break;
      case 40962:
        if (kodak)
          raw_width = get4();
        break;
      case 40963:
        if (kodak)
          raw_height = get4();
        break;
      case 41730:
        if (get4() == 0x20002)
          for (exif_cfa = c = 0; c < 8; c += 2)
            exif_cfa |= fgetc(ifp) * 0x01010101 << c;
    }
    fseek(ifp, save, SEEK_SET);
  }
}

void CLASS parse_gps(int base) {
  unsigned entries, tag, type, len, save, c;

  entries = get2();
  while (entries--) {
    tiff_get(base, &tag, &type, &len, &save);
    switch (tag) {
      case 1:
      case 3:
      case 5:
        gpsdata[29 + tag / 2] = getc(ifp);
        break;
      case 2:
      case 4:
      case 7:
        FORC(6) gpsdata[tag / 3 * 6 + c] = get4();
        break;
      case 6:
        FORC(2) gpsdata[18 + c] = get4();
        break;
      case 18:
      case 29:
        fgets((char *)(gpsdata + 14 + tag / 3), MIN(len, 12), ifp);
    }
    fseek(ifp, save, SEEK_SET);
  }
}

void CLASS romm_coeff(float romm_cam[3][3]) {
  static const float rgb_romm[3][3] = /* ROMM == Kodak ProPhoto */
      {{2.034193, -0.727420, -0.306766},
       {-0.228811, 1.231729, -0.002922},
       {-0.008565, -0.153273, 1.161839}};
  int i, j, k;

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      for (cmatrix[i][j] = k = 0; k < 3; k++)
        cmatrix[i][j] += rgb_romm[i][k] * romm_cam[k][j];
}

void CLASS parse_mos(int offset) {
  char data[40];
  int skip, from, i, c, neut[4], planes = 0, frot = 0;
  static const char *mod[] = {"",
                              "DCB2",
                              "Volare",
                              "Cantare",
                              "CMost",
                              "Valeo 6",
                              "Valeo 11",
                              "Valeo 22",
                              "Valeo 11p",
                              "Valeo 17",
                              "",
                              "Aptus 17",
                              "Aptus 22",
                              "Aptus 75",
                              "Aptus 65",
                              "Aptus 54S",
                              "Aptus 65S",
                              "Aptus 75S",
                              "AFi 5",
                              "AFi 6",
                              "AFi 7",
                              "AFi-II 7",
                              "Aptus-II 7",
                              "",
                              "Aptus-II 6",
                              "",
                              "",
                              "Aptus-II 10",
                              "Aptus-II 5",
                              "",
                              "",
                              "",
                              "",
                              "Aptus-II 10R",
                              "Aptus-II 8",
                              "",
                              "Aptus-II 12",
                              "",
                              "AFi-II 12"};
  float romm_cam[3][3];

  fseek(ifp, offset, SEEK_SET);
  while (1) {
    if (get4() != 0x504b5453)
      break;
    get4();
    fread(data, 1, 40, ifp);
    skip = get4();
    from = ftell(ifp);
    if (!strcmp(data, "JPEG_preview_data")) {
      thumb_offset = from;
      thumb_length = skip;
    }
    if (!strcmp(data, "icc_camera_profile")) {
      profile_offset = from;
      profile_length = skip;
    }
    if (!strcmp(data, "ShootObj_back_type")) {
      fscanf(ifp, "%d", &i);
      if ((unsigned)i < sizeof mod / sizeof(*mod))
        strcpy(model, mod[i]);
    }
    if (!strcmp(data, "icc_camera_to_tone_matrix")) {
      for (i = 0; i < 9; i++)
        ((float *)romm_cam)[i] = int_to_float(get4());
      romm_coeff(romm_cam);
    }
    if (!strcmp(data, "CaptProf_color_matrix")) {
      for (i = 0; i < 9; i++)
        fscanf(ifp, "%f", (float *)romm_cam + i);
      romm_coeff(romm_cam);
    }
    if (!strcmp(data, "CaptProf_number_of_planes"))
      fscanf(ifp, "%d", &planes);
    if (!strcmp(data, "CaptProf_raw_data_rotation"))
      fscanf(ifp, "%d", &flip);
    if (!strcmp(data, "CaptProf_mosaic_pattern"))
      FORC4 {
        fscanf(ifp, "%d", &i);
        if (i == 1)
          frot = c ^ (c >> 1);
      }
    if (!strcmp(data, "ImgProf_rotation_angle")) {
      fscanf(ifp, "%d", &i);
      flip = i - flip;
    }
    if (!strcmp(data, "NeutObj_neutrals") && !cam_mul[0]) {
      FORC4 fscanf(ifp, "%d", neut + c);
      FORC3 cam_mul[c] = (float)neut[0] / neut[c + 1];
    }
    if (!strcmp(data, "Rows_data"))
      load_flags = get4();
    parse_mos(from);
    fseek(ifp, skip + from, SEEK_SET);
  }
  if (planes)
    filters = (planes == 1) * 0x01010101 *
              (uchar) "\x94\x61\x16\x49"[(flip / 90 + frot) & 3];
}

void CLASS linear_table(unsigned len) {
  int i;
  if (len > 0x1000)
    len = 0x1000;
  read_shorts(curve, len);
  for (i = len; i < 0x1000; i++)
    curve[i] = curve[i - 1];
  maximum = curve[0xfff];
}

void CLASS parse_kodak_ifd(int base) {
  unsigned entries, tag, type, len, save;
  int i, c, wbi = -2, wbtemp = 6500;
  float mul[3] = {1, 1, 1}, num;
  static const int wbtag[] = {64037, 64040, 64039, 64041, -1, -1, 64042};

  entries = get2();
  if (entries > 1024)
    return;
  while (entries--) {
    tiff_get(base, &tag, &type, &len, &save);
    if (tag == 1020)
      wbi = getint(type);
    if (tag == 1021 && len == 72) { /* WB set in software */
      fseek(ifp, 40, SEEK_CUR);
      FORC3 cam_mul[c] = 2048.0 / get2();
      wbi = -2;
    }
    if (tag == 2118)
      wbtemp = getint(type);
    if (tag == 2120 + wbi && wbi >= 0)
      FORC3 cam_mul[c] = 2048.0 / getreal(type);
    if (tag == 2130 + wbi)
      FORC3 mul[c] = getreal(type);
    if (tag == 2140 + wbi && wbi >= 0)
      FORC3 {
        for (num = i = 0; i < 4; i++)
          num += getreal(type) * pow(wbtemp / 100.0, i);
        cam_mul[c] = 2048 / (num * mul[c]);
      }
    if (tag == 2317)
      linear_table(len);
    if (tag == 6020)
      iso_speed = getint(type);
    if (tag == 64013)
      wbi = fgetc(ifp);
    if ((unsigned)wbi < 7 && tag == wbtag[wbi])
      FORC3 cam_mul[c] = get4();
    if (tag == 64019)
      width = getint(type);
    if (tag == 64020)
      height = (getint(type) + 1) & -2;
    fseek(ifp, save, SEEK_SET);
  }
}

void CLASS parse_minolta(int base);
int CLASS parse_tiff(int base);

int CLASS parse_tiff_ifd(int base) {
  unsigned entries, tag, type, len, plen = 16, save;
  int ifd, use_cm = 0, cfa, i, j, c, ima_len = 0;
  char software[64], *cbuf, *cp;
  uchar cfa_pat[16], cfa_pc[] = {0, 1, 2, 3}, tab[256];
  double cc[4][4], cm[4][3], cam_xyz[4][3], num;
  double ab[] = {1, 1, 1, 1}, asn[] = {0, 0, 0, 0}, xyz[] = {1, 1, 1};
  unsigned sony_curve[] = {0, 0, 0, 0, 0, 4095};
  unsigned *buf, sony_offset = 0, sony_length = 0, sony_key = 0;
  struct jhead jh;
  FILE *sfp;

  if (tiff_nifds >= sizeof tiff_ifd / sizeof tiff_ifd[0])
    return 1;
  ifd = tiff_nifds++;
  for (j = 0; j < 4; j++)
    for (i = 0; i < 4; i++)
      cc[j][i] = i == j;
  entries = get2();
  if (entries > 512)
    return 1;
  while (entries--) {
    tiff_get(base, &tag, &type, &len, &save);
    switch (tag) {
      case 5:
        width = get2();
        break;
      case 6:
        height = get2();
        break;
      case 7:
        width += get2();
        break;
      case 9:
        if ((i = get2()))
          filters = i;
        break;
      case 17:
      case 18:
        if (type == 3 && len == 1)
          cam_mul[(tag - 17) * 2] = get2() / 256.0;
        break;
      case 23:
        if (type == 3)
          iso_speed = get2();
        break;
      case 28:
      case 29:
      case 30:
        cblack[tag - 28] = get2();
        cblack[3] = cblack[1];
        break;
      case 36:
      case 37:
      case 38:
        cam_mul[tag - 36] = get2();
        break;
      case 39:
        if (len < 50 || cam_mul[0])
          break;
        fseek(ifp, 12, SEEK_CUR);
        FORC3 cam_mul[c] = get2();
        break;
      case 46:
        if (type != 7 || fgetc(ifp) != 0xff || fgetc(ifp) != 0xd8)
          break;
        thumb_offset = ftell(ifp) - 2;
        thumb_length = len;
        break;
      case 61440: /* Fuji HS10 table */
        fseek(ifp, get4() + base, SEEK_SET);
        parse_tiff_ifd(base);
        break;
      case 2:
      case 256:
      case 61441: /* ImageWidth */
        tiff_ifd[ifd].width = getint(type);
        break;
      case 3:
      case 257:
      case 61442: /* ImageHeight */
        tiff_ifd[ifd].height = getint(type);
        break;
      case 258: /* BitsPerSample */
      case 61443:
        tiff_ifd[ifd].samples = len & 7;
        tiff_ifd[ifd].bps = getint(type);
        if (tiff_bps < tiff_ifd[ifd].bps)
          tiff_bps = tiff_ifd[ifd].bps;
        break;
      case 61446:
        raw_height = 0;
        if (tiff_ifd[ifd].bps > 12)
          break;
        load_raw = &CLASS packed_load_raw;
        load_flags = get4() ? 24 : 80;
        break;
      case 259: /* Compression */
        tiff_ifd[ifd].comp = getint(type);
        break;
      case 262: /* PhotometricInterpretation */
        tiff_ifd[ifd].phint = get2();
        break;
      case 270: /* ImageDescription */
        fread(desc, 512, 1, ifp);
        break;
      case 271: /* Make */
        fgets(make, 64, ifp);
        break;
      case 272: /* Model */
        fgets(model, 64, ifp);
        break;
      case 280: /* Panasonic RW2 offset */
        if (type != 4)
          break;
        load_raw = &CLASS panasonic_load_raw;
        load_flags = 0x2008;
      case 273: /* StripOffset */
      case 513: /* JpegIFOffset */
      case 61447:
        tiff_ifd[ifd].offset = get4() + base;
        if (!tiff_ifd[ifd].bps && tiff_ifd[ifd].offset > 0) {
          fseek(ifp, tiff_ifd[ifd].offset, SEEK_SET);
          if (ljpeg_start(&jh, 1)) {
            tiff_ifd[ifd].comp = 6;
            tiff_ifd[ifd].width = jh.wide;
            tiff_ifd[ifd].height = jh.high;
            tiff_ifd[ifd].bps = jh.bits;
            tiff_ifd[ifd].samples = jh.clrs;
            if (!(jh.sraw || (jh.clrs & 1)))
              tiff_ifd[ifd].width *= jh.clrs;
            if ((tiff_ifd[ifd].width > 4 * tiff_ifd[ifd].height) & ~jh.clrs) {
              tiff_ifd[ifd].width /= 2;
              tiff_ifd[ifd].height *= 2;
            }
            i = order;
            parse_tiff(tiff_ifd[ifd].offset + 12);
            order = i;
          }
        }
        break;
      case 274: /* Orientation */
        tiff_ifd[ifd].flip = "50132467"[get2() & 7] - '0';
        break;
      case 277: /* SamplesPerPixel */
        tiff_ifd[ifd].samples = getint(type) & 7;
        break;
      case 279: /* StripByteCounts */
      case 514:
      case 61448:
        tiff_ifd[ifd].bytes = get4();
        break;
      case 61454:
        FORC3 cam_mul[(4 - c) % 3] = getint(type);
        break;
      case 305:
      case 11: /* Software */
        fgets(software, 64, ifp);
        if (!strncmp(software, "Adobe", 5) || !strncmp(software, "dcraw", 5) ||
            !strncmp(software, "UFRaw", 5) || !strncmp(software, "Bibble", 6) ||
            !strncmp(software, "Nikon Scan", 10) ||
            !strcmp(software, "Digital Photo Professional"))
          is_raw = 0;
        break;
      case 306: /* DateTime */
        get_timestamp(0);
        break;
      case 315: /* Artist */
        fread(artist, 64, 1, ifp);
        break;
      case 322: /* TileWidth */
        tiff_ifd[ifd].tile_width = getint(type);
        break;
      case 323: /* TileLength */
        tiff_ifd[ifd].tile_length = getint(type);
        break;
      case 324: /* TileOffsets */
        tiff_ifd[ifd].offset = len > 1 ? ftell(ifp) : get4();
        if (len == 1)
          tiff_ifd[ifd].tile_width = tiff_ifd[ifd].tile_length = 0;
        if (len == 4) {
          load_raw = &CLASS sinar_4shot_load_raw;
          is_raw = 5;
        }
        break;
      case 330: /* SubIFDs */
        if (!strcmp(model, "DSLR-A100") && tiff_ifd[ifd].width == 3872) {
          load_raw = &CLASS sony_arw_load_raw;
          data_offset = get4() + base;
          ifd++;
          break;
        }
        while (len--) {
          i = ftell(ifp);
          fseek(ifp, get4() + base, SEEK_SET);
          if (parse_tiff_ifd(base))
            break;
          fseek(ifp, i + 4, SEEK_SET);
        }
        break;
      case 400:
        strcpy(make, "Sarnoff");
        maximum = 0xfff;
        break;
      case 28688:
        FORC4 sony_curve[c + 1] = get2() >> 2 & 0xfff;
        for (i = 0; i < 5; i++)
          for (j = sony_curve[i] + 1; j <= sony_curve[i + 1]; j++)
            curve[j] = curve[j - 1] + (1 << i);
        break;
      case 29184:
        sony_offset = get4();
        break;
      case 29185:
        sony_length = get4();
        break;
      case 29217:
        sony_key = get4();
        break;
      case 29264:
        parse_minolta(ftell(ifp));
        raw_width = 0;
        break;
      case 29443:
        FORC4 cam_mul[c ^ (c < 2)] = get2();
        break;
      case 29459:
        FORC4 cam_mul[c] = get2();
        i = (cam_mul[1] == 1024 && cam_mul[2] == 1024) << 1;
        SWAP(cam_mul[i], cam_mul[i + 1])
        break;
      case 33405: /* Model2 */
        fgets(model2, 64, ifp);
        break;
      case 33421: /* CFARepeatPatternDim */
        if (get2() == 6 && get2() == 6)
          filters = 9;
        break;
      case 33422: /* CFAPattern */
        if (filters == 9) {
          FORC(36)((char *)xtrans)[c] = fgetc(ifp) & 3;
          break;
        }
      case 64777: /* Kodak P-series */
        if ((plen = len) > 16)
          plen = 16;
        fread(cfa_pat, 1, plen, ifp);
        for (colors = cfa = i = 0; i < plen && colors < 4; i++) {
          colors += !(cfa & (1 << cfa_pat[i]));
          cfa |= 1 << cfa_pat[i];
        }
        if (cfa == 070)
          memcpy(cfa_pc, "\003\004\005", 3); /* CMY */
        if (cfa == 072)
          memcpy(cfa_pc, "\005\003\004\001", 4); /* GMCY */
        goto guess_cfa_pc;
      case 33424:
      case 65024:
        fseek(ifp, get4() + base, SEEK_SET);
        parse_kodak_ifd(base);
        break;
      case 33434: /* ExposureTime */
        tiff_ifd[ifd].shutter = shutter = getreal(type);
        break;
      case 33437: /* FNumber */
        aperture = getreal(type);
        break;
      case 34306: /* Leaf white balance */
        FORC4 cam_mul[c ^ 1] = 4096.0 / get2();
        break;
      case 34307: /* Leaf CatchLight color matrix */
        fread(software, 1, 7, ifp);
        if (strncmp(software, "MATRIX", 6))
          break;
        colors = 4;
        for (raw_color = i = 0; i < 3; i++) {
          FORC4 fscanf(ifp, "%f", &rgb_cam[i][c ^ 1]);
          if (!use_camera_wb)
            continue;
          num = 0;
          FORC4 num += rgb_cam[i][c];
          FORC4 rgb_cam[i][c] /= num;
        }
        break;
      case 34310: /* Leaf metadata */
        parse_mos(ftell(ifp));
      case 34303:
        strcpy(make, "Leaf");
        break;
      case 34665: /* EXIF tag */
        fseek(ifp, get4() + base, SEEK_SET);
        parse_exif(base);
        break;
      case 34853: /* GPSInfo tag */
        fseek(ifp, get4() + base, SEEK_SET);
        parse_gps(base);
        break;
      case 34675: /* InterColorProfile */
      case 50831: /* AsShotICCProfile */
        profile_offset = ftell(ifp);
        profile_length = len;
        break;
      case 37122: /* CompressedBitsPerPixel */
        kodak_cbpp = get4();
        break;
      case 37386: /* FocalLength */
        focal_len = getreal(type);
        break;
      case 37393: /* ImageNumber */
        shot_order = getint(type);
        break;
      case 37400: /* old Kodak KDC tag */
        for (raw_color = i = 0; i < 3; i++) {
          getreal(type);
          FORC3 rgb_cam[i][c] = getreal(type);
        }
        break;
      case 40976:
        strip_offset = get4();
        switch (tiff_ifd[ifd].comp) {
          case 32770:
            load_raw = &CLASS samsung_load_raw;
            break;
          case 32772:
            load_raw = &CLASS samsung2_load_raw;
            break;
          case 32773:
            load_raw = &CLASS samsung3_load_raw;
            break;
        }
        break;
      case 46275: /* Imacon tags */
        strcpy(make, "Imacon");
        data_offset = ftell(ifp);
        ima_len = len;
        break;
      case 46279:
        if (!ima_len)
          break;
        fseek(ifp, 38, SEEK_CUR);
      case 46274:
        fseek(ifp, 40, SEEK_CUR);
        raw_width = get4();
        raw_height = get4();
        left_margin = get4() & 7;
        width = raw_width - left_margin - (get4() & 7);
        top_margin = get4() & 7;
        height = raw_height - top_margin - (get4() & 7);
        if (raw_width == 7262) {
          height = 5444;
          width = 7244;
          left_margin = 7;
        }
        fseek(ifp, 52, SEEK_CUR);
        FORC3 cam_mul[c] = getreal(11);
        fseek(ifp, 114, SEEK_CUR);
        flip = (get2() >> 7) * 90;
        if (width * height * 6 == ima_len) {
          if (flip % 180 == 90)
            SWAP(width, height);
          raw_width = width;
          raw_height = height;
          left_margin = top_margin = filters = flip = 0;
        }
        sprintf(model, "Ixpress %d-Mp", height * width / 1000000);
        load_raw = &CLASS imacon_full_load_raw;
        if (filters) {
          if (left_margin & 1)
            filters = 0x61616161;
          load_raw = &CLASS unpacked_load_raw;
        }
        maximum = 0xffff;
        break;
      case 50454: /* Sinar tag */
      case 50455:
        if (!(cbuf = (char *)malloc(len)))
          break;
        fread(cbuf, 1, len, ifp);
        for (cp = cbuf - 1; cp && cp < cbuf + len; cp = strchr(cp, '\n'))
          if (!strncmp(++cp, "Neutral ", 8))
            sscanf(cp + 8, "%f %f %f", cam_mul, cam_mul + 1, cam_mul + 2);
        free(cbuf);
        break;
      case 50458:
        if (!make[0])
          strcpy(make, "Hasselblad");
        break;
      case 50459: /* Hasselblad tag */
        i = order;
        j = ftell(ifp);
        c = tiff_nifds;
        order = get2();
        fseek(ifp, j + (get2(), get4()), SEEK_SET);
        parse_tiff_ifd(j);
        maximum = 0xffff;
        tiff_nifds = c;
        order = i;
        break;
      case 50706: /* DNGVersion */
        FORC4 dng_version = (dng_version << 8) + fgetc(ifp);
        if (!make[0])
          strcpy(make, "DNG");
        is_raw = 1;
        break;
      case 50708: /* UniqueCameraModel */
        if (model[0])
          break;
        fgets(make, 64, ifp);
        if ((cp = strchr(make, ' '))) {
          strcpy(model, cp + 1);
          *cp = 0;
        }
        break;
      case 50710: /* CFAPlaneColor */
        if (filters == 9)
          break;
        if (len > 4)
          len = 4;
        colors = len;
        fread(cfa_pc, 1, colors, ifp);
      guess_cfa_pc:
        FORCC tab[cfa_pc[c]] = c;
        cdesc[c] = 0;
        for (i = 16; i--;)
          filters = filters << 2 | tab[cfa_pat[i % plen]];
        filters -= !filters;
        break;
      case 50711: /* CFALayout */
        if (get2() == 2)
          fuji_width = 1;
        break;
      case 291:
      case 50712: /* LinearizationTable */
        linear_table(len);
        break;
      case 50713: /* BlackLevelRepeatDim */
        cblack[4] = get2();
        cblack[5] = get2();
        if (cblack[4] * cblack[5] > sizeof cblack / sizeof *cblack - 6)
          cblack[4] = cblack[5] = 1;
        break;
      case 61450:
        cblack[4] = cblack[5] = MIN(sqrt(len), 64);
      case 50714: /* BlackLevel */
        if (!(cblack[4] * cblack[5]))
          cblack[4] = cblack[5] = 1;
        FORC(cblack[4] * cblack[5])
        cblack[6 + c] = getreal(type);
        black = 0;
        break;
      case 50715: /* BlackLevelDeltaH */
      case 50716: /* BlackLevelDeltaV */
        for (num = i = 0; i < (len & 0xffff); i++)
          num += getreal(type);
        black += num / len + 0.5;
        break;
      case 50717: /* WhiteLevel */
        maximum = getint(type);
        break;
      case 50718: /* DefaultScale */
        pixel_aspect = getreal(type);
        pixel_aspect /= getreal(type);
        break;
      case 50721: /* ColorMatrix1 */
      case 50722: /* ColorMatrix2 */
        FORCC for (j = 0; j < 3; j++) cm[c][j] = getreal(type);
        use_cm = 1;
        break;
      case 50723: /* CameraCalibration1 */
      case 50724: /* CameraCalibration2 */
        for (i = 0; i < colors; i++)
          FORCC cc[i][c] = getreal(type);
        break;
      case 50727: /* AnalogBalance */
        FORCC ab[c] = getreal(type);
        break;
      case 50728: /* AsShotNeutral */
        FORCC asn[c] = getreal(type);
        break;
      case 50729: /* AsShotWhiteXY */
        xyz[0] = getreal(type);
        xyz[1] = getreal(type);
        xyz[2] = 1 - xyz[0] - xyz[1];
        FORC3 xyz[c] /= d65_white[c];
        break;
      case 50740: /* DNGPrivateData */
        if (dng_version)
          break;
        parse_minolta(j = get4() + base);
        fseek(ifp, j, SEEK_SET);
        parse_tiff_ifd(base);
        break;
      case 50752:
        read_shorts(cr2_slice, 3);
        break;
      case 50829: /* ActiveArea */
        top_margin = getint(type);
        left_margin = getint(type);
        height = getint(type) - top_margin;
        width = getint(type) - left_margin;
        break;
      case 50830: /* MaskedAreas */
        for (i = 0; i < len && i < 32; i++)
          ((int *)mask)[i] = getint(type);
        black = 0;
        break;
      case 51009: /* OpcodeList2 */
        meta_offset = ftell(ifp);
        break;
      case 64772: /* Kodak P-series */
        if (len < 13)
          break;
        fseek(ifp, 16, SEEK_CUR);
        data_offset = get4();
        fseek(ifp, 28, SEEK_CUR);
        data_offset += get4();
        load_raw = &CLASS packed_load_raw;
        break;
      case 65026:
        if (type == 2)
          fgets(model2, 64, ifp);
    }
    fseek(ifp, save, SEEK_SET);
  }
  if (sony_length && (buf = (unsigned *)malloc(sony_length))) {
    fseek(ifp, sony_offset, SEEK_SET);
    fread(buf, sony_length, 1, ifp);
    sony_decrypt(buf, sony_length / 4, 1, sony_key);
    sfp = ifp;
    if ((ifp = tmpfile())) {
      fwrite(buf, sony_length, 1, ifp);
      fseek(ifp, 0, SEEK_SET);
      parse_tiff_ifd(-sony_offset);
      fclose(ifp);
    }
    ifp = sfp;
    free(buf);
  }
  for (i = 0; i < colors; i++)
    FORCC cc[i][c] *= ab[i];
  if (use_cm) {
    FORCC for (i = 0; i < 3; i++) for (cam_xyz[c][i] = j = 0; j < colors; j++)
        cam_xyz[c][i] += cc[c][j] * cm[j][i] * xyz[i];
    cam_xyz_coeff(cmatrix, cam_xyz);
  }
  if (asn[0]) {
    cam_mul[3] = 0;
    FORCC cam_mul[c] = 1 / asn[c];
  }
  if (!use_cm)
    FORCC pre_mul[c] /= cc[c][c];
  return 0;
}

int CLASS parse_tiff(int base) {
  int doff;

  fseek(ifp, base, SEEK_SET);
  order = get2();
  if (order != 0x4949 && order != 0x4d4d)
    return 0;
  get2();
  while ((doff = get4())) {
    fseek(ifp, doff + base, SEEK_SET);
    if (parse_tiff_ifd(base))
      break;
  }
  return 1;
}

void CLASS apply_tiff() {
  int max_samp = 0, ties = 0, os, ns, raw = -1, thm = -1, i;
  struct jhead jh;

  thumb_misc = 16;
  if (thumb_offset) {
    fseek(ifp, thumb_offset, SEEK_SET);
    if (ljpeg_start(&jh, 1)) {
      thumb_misc = jh.bits;
      thumb_width = jh.wide;
      thumb_height = jh.high;
    }
  }
  for (i = tiff_nifds; i--;) {
    if (tiff_ifd[i].shutter)
      shutter = tiff_ifd[i].shutter;
    tiff_ifd[i].shutter = shutter;
  }
  for (i = 0; i < tiff_nifds; i++) {
    if (max_samp < tiff_ifd[i].samples)
      max_samp = tiff_ifd[i].samples;
    if (max_samp > 3)
      max_samp = 3;
    os = raw_width * raw_height;
    ns = tiff_ifd[i].width * tiff_ifd[i].height;
    if (tiff_bps) {
      os *= tiff_bps;
      ns *= tiff_ifd[i].bps;
    }
    if ((tiff_ifd[i].comp != 6 || tiff_ifd[i].samples != 3) &&
        (tiff_ifd[i].width | tiff_ifd[i].height) < 0x10000 && ns &&
        ((ns > os && (ties = 1)) || (ns == os && shot_select == ties++))) {
      raw_width = tiff_ifd[i].width;
      raw_height = tiff_ifd[i].height;
      tiff_bps = tiff_ifd[i].bps;
      tiff_compress = tiff_ifd[i].comp;
      data_offset = tiff_ifd[i].offset;
      tiff_flip = tiff_ifd[i].flip;
      tiff_samples = tiff_ifd[i].samples;
      tile_width = tiff_ifd[i].tile_width;
      tile_length = tiff_ifd[i].tile_length;
      shutter = tiff_ifd[i].shutter;
      raw = i;
    }
  }
  if (is_raw == 1 && ties)
    is_raw = ties;
  if (!tile_width)
    tile_width = INT_MAX;
  if (!tile_length)
    tile_length = INT_MAX;
  for (i = tiff_nifds; i--;)
    if (tiff_ifd[i].flip)
      tiff_flip = tiff_ifd[i].flip;
  if (raw >= 0 && !load_raw)
    switch (tiff_compress) {
      case 32767:
        if (tiff_ifd[raw].bytes == raw_width * raw_height) {
          tiff_bps = 12;
          load_raw = &CLASS sony_arw2_load_raw;
          break;
        }
        if (tiff_ifd[raw].bytes * 8 != raw_width * raw_height * tiff_bps) {
          raw_height += 8;
          load_raw = &CLASS sony_arw_load_raw;
          break;
        }
        load_flags = 79;
      case 32769:
        load_flags++;
      case 32770:
      case 32773:
        goto slr;
      case 0:
      case 1:
        if (!strncmp(make, "OLYMPUS", 7) &&
            tiff_ifd[raw].bytes * 2 == raw_width * raw_height * 3)
          load_flags = 24;
        if (tiff_ifd[raw].bytes * 5 == raw_width * raw_height * 8) {
          load_flags = 81;
          tiff_bps = 12;
        }
      slr:
        switch (tiff_bps) {
          case 8:
            load_raw = &CLASS eight_bit_load_raw;
            break;
          case 12:
            if (tiff_ifd[raw].phint == 2)
              load_flags = 6;
            load_raw = &CLASS packed_load_raw;
            break;
          case 14:
            load_flags = 0;
          case 16:
            load_raw = &CLASS unpacked_load_raw;
            if (!strncmp(make, "OLYMPUS", 7) &&
                tiff_ifd[raw].bytes * 7 > raw_width * raw_height)
              load_raw = &CLASS olympus_load_raw;
        }
        break;
      case 6:
      case 7:
      case 99:
        load_raw = &CLASS lossless_jpeg_load_raw;
        break;
      case 262:
        load_raw = &CLASS kodak_262_load_raw;
        break;
      case 34713:
        if ((raw_width + 9) / 10 * 16 * raw_height == tiff_ifd[raw].bytes) {
          load_raw = &CLASS packed_load_raw;
          load_flags = 1;
        } else if (raw_width * raw_height * 3 == tiff_ifd[raw].bytes * 2) {
          load_raw = &CLASS packed_load_raw;
          if (model[0] == 'N')
            load_flags = 80;
        } else if (raw_width * raw_height * 3 == tiff_ifd[raw].bytes) {
          load_raw = &CLASS nikon_yuv_load_raw;
          gamma_curve(1 / 2.4, 12.92, 1, 4095);
          memset(cblack, 0, sizeof cblack);
          filters = 0;
        } else if (raw_width * raw_height * 2 == tiff_ifd[raw].bytes) {
          load_raw = &CLASS unpacked_load_raw;
          load_flags = 4;
          order = 0x4d4d;
        } else
          load_raw = &CLASS nikon_load_raw;
        break;
      case 65535:
        load_raw = &CLASS pentax_load_raw;
        break;
      case 65000:
        switch (tiff_ifd[raw].phint) {
          case 2:
            load_raw = &CLASS kodak_rgb_load_raw;
            filters = 0;
            break;
          case 6:
            load_raw = &CLASS kodak_ycbcr_load_raw;
            filters = 0;
            break;
          case 32803:
            load_raw = &CLASS kodak_65000_load_raw;
        }
      case 32867:
      case 34892:
        break;
      default:
        is_raw = 0;
    }
  if (!dng_version)
    if ((tiff_samples == 3 && tiff_ifd[raw].bytes && tiff_bps != 14 &&
         (tiff_compress & -16) != 32768) ||
        (tiff_bps == 8 && strncmp(make, "Phase", 5) &&
         !strcasestr(make, "Kodak") && !strstr(model2, "DEBUG RAW")))
      is_raw = 0;
  for (i = 0; i < tiff_nifds; i++)
    if (i != raw && tiff_ifd[i].samples == max_samp &&
        tiff_ifd[i].width * tiff_ifd[i].height / (SQR(tiff_ifd[i].bps) + 1) >
            thumb_width * thumb_height / (SQR(thumb_misc) + 1) &&
        tiff_ifd[i].comp != 34892) {
      thumb_width = tiff_ifd[i].width;
      thumb_height = tiff_ifd[i].height;
      thumb_offset = tiff_ifd[i].offset;
      thumb_length = tiff_ifd[i].bytes;
      thumb_misc = tiff_ifd[i].bps;
      thm = i;
    }
  if (thm >= 0) {
    thumb_misc |= tiff_ifd[thm].samples << 5;
    switch (tiff_ifd[thm].comp) {
      case 0:
        write_thumb = &CLASS layer_thumb;
        break;
      case 1:
        if (tiff_ifd[thm].bps <= 8)
          write_thumb = &CLASS ppm_thumb;
        else if (!strcmp(make, "Imacon"))
          write_thumb = &CLASS ppm16_thumb;
        else
          thumb_load_raw = &CLASS kodak_thumb_load_raw;
        break;
      case 65000:
        thumb_load_raw = tiff_ifd[thm].phint == 6 ? &CLASS kodak_ycbcr_load_raw
                                                  : &CLASS kodak_rgb_load_raw;
    }
  }
}

void CLASS parse_minolta(int base) {
  int save, tag, len, offset, high = 0, wide = 0, i, c;
  short sorder = order;

  fseek(ifp, base, SEEK_SET);
  if (fgetc(ifp) || fgetc(ifp) - 'M' || fgetc(ifp) - 'R')
    return;
  order = fgetc(ifp) * 0x101;
  offset = base + get4() + 8;
  while ((save = ftell(ifp)) < offset) {
    for (tag = i = 0; i < 4; i++)
      tag = tag << 8 | fgetc(ifp);
    len = get4();
    switch (tag) {
      case 0x505244: /* PRD */
        fseek(ifp, 8, SEEK_CUR);
        high = get2();
        wide = get2();
        break;
      case 0x574247: /* WBG */
        get4();
        i = strcmp(model, "DiMAGE A200") ? 0 : 3;
        FORC4 cam_mul[c ^ (c >> 1) ^ i] = get2();
        break;
      case 0x545457: /* TTW */
        parse_tiff(ftell(ifp));
        data_offset = offset;
    }
    fseek(ifp, save + len + 8, SEEK_SET);
  }
  raw_height = high;
  raw_width = wide;
  order = sorder;
}

/*
Many cameras have a "debug mode" that writes JPEG and raw
at the same time.  The raw file has no header, so try to
to open the matching JPEG file and read its metadata.
*/
void CLASS parse_external_jpeg() {
  const char *file, *ext;
  char *jname, *jfile, *jext;
  FILE *save = ifp;

  ext = strrchr(ifname, '.');
  file = strrchr(ifname, '/');
  if (!file)
    file = strrchr(ifname, '\\');
  if (!file)
    file = ifname - 1;
  file++;
  if (!ext || strlen(ext) != 4 || ext - file != 8)
    return;
  jname = (char *)malloc(strlen(ifname) + 1);
  merror(jname, "parse_external_jpeg()");
  strcpy(jname, ifname);
  jfile = file - ifname + jname;
  jext = ext - ifname + jname;
  if (strcasecmp(ext, ".jpg")) {
    strcpy(jext, isupper(ext[1]) ? ".JPG" : ".jpg");
    if (isdigit(*file)) {
      memcpy(jfile, file + 4, 4);
      memcpy(jfile + 4, file, 4);
    }
  } else
    while (isdigit(*--jext)) {
      if (*jext != '9') {
        (*jext)++;
        break;
      }
      *jext = '0';
    }
  if (strcmp(jname, ifname)) {
    if ((ifp = fopen(jname, "rb"))) {
      if (verbose)
        fprintf(stderr, _("Reading metadata from %s ...\n"), jname);
      parse_tiff(12);
      thumb_offset = 0;
      is_raw = 1;
      fclose(ifp);
    }
  }
  if (!timestamp)
    fprintf(stderr, _("Failed to read metadata from %s\n"), jname);
  free(jname);
  ifp = save;
}

/*
CIFF block 0x1030 contains an 8x8 white sample.
Load this into white[][] for use in scale_colors().
*/
void CLASS ciff_block_1030() {
  static const ushort key[] = {0x410, 0x45f3};
  int i, bpp, row, col, vbits = 0;
  unsigned long bitbuf = 0;

  if ((get2(), get4()) != 0x80008 || !get4())
    return;
  bpp = get2();
  if (bpp != 10 && bpp != 12)
    return;
  for (i = row = 0; row < 8; row++)
    for (col = 0; col < 8; col++) {
      if (vbits < bpp) {
        bitbuf = bitbuf << 16 | (get2() ^ key[i++ & 1]);
        vbits += 16;
      }
      white[row][col] = bitbuf >> (vbits -= bpp) & ~(-1 << bpp);
    }
}

/*
Parse a CIFF file, better known as Canon CRW format.
*/
void CLASS parse_ciff(int offset, int length, int depth) {
  int tboff, nrecs, c, type, len, save, wbi = -1;
  ushort key[] = {0x410, 0x45f3};

  fseek(ifp, offset + length - 4, SEEK_SET);
  tboff = get4() + offset;
  fseek(ifp, tboff, SEEK_SET);
  nrecs = get2();
  if ((nrecs | depth) > 127)
    return;
  while (nrecs--) {
    type = get2();
    len = get4();
    save = ftell(ifp) + 4;
    fseek(ifp, offset + get4(), SEEK_SET);
    if ((((type >> 8) + 8) | 8) == 0x38)
      parse_ciff(ftell(ifp), len, depth + 1); /* Parse a sub-table */
    if (type == 0x0810)
      fread(artist, 64, 1, ifp);
    if (type == 0x080a) {
      fread(make, 64, 1, ifp);
      fseek(ifp, strlen(make) - 63, SEEK_CUR);
      fread(model, 64, 1, ifp);
    }
    if (type == 0x1810) {
      width = get4();
      height = get4();
      pixel_aspect = int_to_float(get4());
      flip = get4();
    }
    if (type == 0x1835) /* Get the decoder table */
      tiff_compress = get4();
    if (type == 0x2007) {
      thumb_offset = ftell(ifp);
      thumb_length = len;
    }
    if (type == 0x1818) {
      shutter = pow(2, -int_to_float((get4(), get4())));
      aperture = pow(2, int_to_float(get4()) / 2);
    }
    if (type == 0x102a) {
      iso_speed = pow(2, (get4(), get2()) / 32.0 - 4) * 50;
      aperture = pow(2, (get2(), (short)get2()) / 64.0);
      shutter = pow(2, -((short)get2()) / 32.0);
      wbi = (get2(), get2());
      if (wbi > 17)
        wbi = 0;
      fseek(ifp, 32, SEEK_CUR);
      if (shutter > 1e6)
        shutter = get2() / 10.0;
    }
    if (type == 0x102c) {
      if (get2() > 512) { /* Pro90, G1 */
        fseek(ifp, 118, SEEK_CUR);
        FORC4 cam_mul[c ^ 2] = get2();
      } else { /* G2, S30, S40 */
        fseek(ifp, 98, SEEK_CUR);
        FORC4 cam_mul[c ^ (c >> 1) ^ 1] = get2();
      }
    }
    if (type == 0x0032) {
      if (len == 768) { /* EOS D30 */
        fseek(ifp, 72, SEEK_CUR);
        FORC4 cam_mul[c ^ (c >> 1)] = 1024.0 / get2();
        if (!wbi)
          cam_mul[0] = -1; /* use my auto white balance */
      } else if (!cam_mul[0]) {
        if (get2() == key[0]) /* Pro1, G6, S60, S70 */
          c = (strstr(model, "Pro1") ? "012346000000000000"
                                     : "01345:000000006008")[wbi] -
              '0' + 2;
        else { /* G3, G5, S45, S50 */
          c = "023457000000006000"[wbi] - '0';
          key[0] = key[1] = 0;
        }
        fseek(ifp, 78 + c * 8, SEEK_CUR);
        FORC4 cam_mul[c ^ (c >> 1) ^ 1] = get2() ^ key[c & 1];
        if (!wbi)
          cam_mul[0] = -1;
      }
    }
    if (type == 0x10a9) { /* D60, 10D, 300D, and clones */
      if (len > 66)
        wbi = "0134567028"[wbi] - '0';
      fseek(ifp, 2 + wbi * 8, SEEK_CUR);
      FORC4 cam_mul[c ^ (c >> 1)] = get2();
    }
    if (type == 0x1030 && (0x18040 >> wbi & 1))
      ciff_block_1030(); /* all that don't have 0x10a9 */
    if (type == 0x1031) {
      raw_width = (get2(), get2());
      raw_height = get2();
    }
    if (type == 0x5029) {
      focal_len = len >> 16;
      if ((len & 0xffff) == 2)
        focal_len /= 32;
    }
    if (type == 0x5813)
      flash_used = int_to_float(len);
    if (type == 0x5814)
      canon_ev = int_to_float(len);
    if (type == 0x5817)
      shot_order = len;
    if (type == 0x5834)
      unique_id = len;
    if (type == 0x580e)
      timestamp = len;
    if (type == 0x180e)
      timestamp = get4();
#ifdef LOCALTIME
    if ((type | 0x4000) == 0x580e)
      timestamp = mktime(gmtime(&timestamp));
#endif
    fseek(ifp, save, SEEK_SET);
  }
}

void CLASS parse_rollei() {
  char line[128], *val;
  struct tm t;

  fseek(ifp, 0, SEEK_SET);
  memset(&t, 0, sizeof t);
  do {
    fgets(line, 128, ifp);
    if ((val = strchr(line, '=')))
      *val++ = 0;
    else
      val = line + strlen(line);
    if (!strcmp(line, "DAT"))
      sscanf(val, "%d.%d.%d", &t.tm_mday, &t.tm_mon, &t.tm_year);
    if (!strcmp(line, "TIM"))
      sscanf(val, "%d:%d:%d", &t.tm_hour, &t.tm_min, &t.tm_sec);
    if (!strcmp(line, "HDR"))
      thumb_offset = atoi(val);
    if (!strcmp(line, "X  "))
      raw_width = atoi(val);
    if (!strcmp(line, "Y  "))
      raw_height = atoi(val);
    if (!strcmp(line, "TX "))
      thumb_width = atoi(val);
    if (!strcmp(line, "TY "))
      thumb_height = atoi(val);
  } while (strncmp(line, "EOHD", 4));
  data_offset = thumb_offset + thumb_width * thumb_height * 2;
  t.tm_year -= 1900;
  t.tm_mon -= 1;
  if (mktime(&t) > 0)
    timestamp = mktime(&t);
  strcpy(make, "Rollei");
  strcpy(model, "d530flex");
  write_thumb = &CLASS rollei_thumb;
}

void CLASS parse_sinar_ia() {
  int entries, off;
  char str[8], *cp;

  order = 0x4949;
  fseek(ifp, 4, SEEK_SET);
  entries = get4();
  fseek(ifp, get4(), SEEK_SET);
  while (entries--) {
    off = get4();
    get4();
    fread(str, 8, 1, ifp);
    if (!strcmp(str, "META"))
      meta_offset = off;
    if (!strcmp(str, "THUMB"))
      thumb_offset = off;
    if (!strcmp(str, "RAW0"))
      data_offset = off;
  }
  fseek(ifp, meta_offset + 20, SEEK_SET);
  fread(make, 64, 1, ifp);
  make[63] = 0;
  if ((cp = strchr(make, ' '))) {
    strcpy(model, cp + 1);
    *cp = 0;
  }
  raw_width = get2();
  raw_height = get2();
  load_raw = &CLASS unpacked_load_raw;
  thumb_width = (get4(), get2());
  thumb_height = get2();
  write_thumb = &CLASS ppm_thumb;
  maximum = 0x3fff;
}

void CLASS parse_phase_one(int base) {
  unsigned entries, tag, type, len, data, save, i, c;
  float romm_cam[3][3];
  char *cp;

  memset(&ph1, 0, sizeof ph1);
  fseek(ifp, base, SEEK_SET);
  order = get4() & 0xffff;
  if (get4() >> 8 != 0x526177)
    return; /* "Raw" */
  fseek(ifp, get4() + base, SEEK_SET);
  entries = get4();
  get4();
  while (entries--) {
    tag = get4();
    type = get4();
    len = get4();
    data = get4();
    save = ftell(ifp);
    fseek(ifp, base + data, SEEK_SET);
    switch (tag) {
      case 0x100:
        flip = "0653"[data & 3] - '0';
        break;
      case 0x106:
        for (i = 0; i < 9; i++)
          ((float *)romm_cam)[i] = getreal(11);
        romm_coeff(romm_cam);
        break;
      case 0x107:
        FORC3 cam_mul[c] = getreal(11);
        break;
      case 0x108:
        raw_width = data;
        break;
      case 0x109:
        raw_height = data;
        break;
      case 0x10a:
        left_margin = data;
        break;
      case 0x10b:
        top_margin = data;
        break;
      case 0x10c:
        width = data;
        break;
      case 0x10d:
        height = data;
        break;
      case 0x10e:
        ph1.format = data;
        break;
      case 0x10f:
        data_offset = data + base;
        break;
      case 0x110:
        meta_offset = data + base;
        meta_length = len;
        break;
      case 0x112:
        ph1.key_off = save - 4;
        break;
      case 0x210:
        ph1.tag_210 = int_to_float(data);
        break;
      case 0x21a:
        ph1.tag_21a = data;
        break;
      case 0x21c:
        strip_offset = data + base;
        break;
      case 0x21d:
        ph1.black = data;
        break;
      case 0x222:
        ph1.split_col = data;
        break;
      case 0x223:
        ph1.black_col = data + base;
        break;
      case 0x224:
        ph1.split_row = data;
        break;
      case 0x225:
        ph1.black_row = data + base;
        break;
      case 0x301:
        model[63] = 0;
        fread(model, 1, 63, ifp);
        if ((cp = strstr(model, " camera")))
          *cp = 0;
    }
    fseek(ifp, save, SEEK_SET);
  }
  load_raw =
      ph1.format < 3 ? &CLASS phase_one_load_raw : &CLASS phase_one_load_raw_c;
  maximum = 0xffff;
  strcpy(make, "Phase One");
  if (model[0])
    return;
  switch (raw_height) {
    case 2060:
      strcpy(model, "LightPhase");
      break;
    case 2682:
      strcpy(model, "H 10");
      break;
    case 4128:
      strcpy(model, "H 20");
      break;
    case 5488:
      strcpy(model, "H 25");
      break;
  }
}

void CLASS parse_fuji(int offset) {
  unsigned entries, tag, len, save, c;

  fseek(ifp, offset, SEEK_SET);
  entries = get4();
  if (entries > 255)
    return;
  while (entries--) {
    tag = get2();
    len = get2();
    save = ftell(ifp);
    if (tag == 0x100) {
      raw_height = get2();
      raw_width = get2();
    } else if (tag == 0x121) {
      height = get2();
      if ((width = get2()) == 4284)
        width += 3;
    } else if (tag == 0x130) {
      fuji_layout = fgetc(ifp) >> 7;
      fuji_width = !(fgetc(ifp) & 8);
    } else if (tag == 0x131) {
      filters = 9;
      FORC(36) xtrans_abs[0][35 - c] = fgetc(ifp) & 3;
    } else if (tag == 0x2ff0) {
      FORC4 cam_mul[c ^ 1] = get2();
    } else if (tag == 0xc000) {
      c = order;
      order = 0x4949;
      while ((tag = get4()) > raw_width)
        ;
      width = tag;
      height = get4();
      order = c;
    }
    fseek(ifp, save + len, SEEK_SET);
  }
  height <<= fuji_layout;
  width >>= fuji_layout;
}

int CLASS parse_jpeg(int offset) {
  int len, save, hlen, mark;

  fseek(ifp, offset, SEEK_SET);
  if (fgetc(ifp) != 0xff || fgetc(ifp) != 0xd8)
    return 0;

  while (fgetc(ifp) == 0xff && (mark = fgetc(ifp)) != 0xda) {
    order = 0x4d4d;
    len = get2() - 2;
    save = ftell(ifp);
    if (mark == 0xc0 || mark == 0xc3 || mark == 0xc9) {
      fgetc(ifp);
      raw_height = get2();
      raw_width = get2();
    }
    order = get2();
    hlen = get4();
    if (get4() == 0x48454150) /* "HEAP" */
      parse_ciff(save + hlen, len - hlen, 0);
    if (parse_tiff(save + 6))
      apply_tiff();
    fseek(ifp, save + len, SEEK_SET);
  }
  return 1;
}

void CLASS parse_riff() {
  unsigned i, size, end;
  char tag[4], date[64], month[64];
  static const char mon[12][4] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
  struct tm t;

  order = 0x4949;
  fread(tag, 4, 1, ifp);
  size = get4();
  end = ftell(ifp) + size;
  if (!memcmp(tag, "RIFF", 4) || !memcmp(tag, "LIST", 4)) {
    get4();
    while (ftell(ifp) + 7 < end && !feof(ifp))
      parse_riff();
  } else if (!memcmp(tag, "nctg", 4)) {
    while (ftell(ifp) + 7 < end) {
      i = get2();
      size = get2();
      if ((i + 1) >> 1 == 10 && size == 20)
        get_timestamp(0);
      else
        fseek(ifp, size, SEEK_CUR);
    }
  } else if (!memcmp(tag, "IDIT", 4) && size < 64) {
    fread(date, 64, 1, ifp);
    date[size] = 0;
    memset(&t, 0, sizeof t);
    if (sscanf(date, "%*s %s %d %d:%d:%d %d", month, &t.tm_mday, &t.tm_hour,
               &t.tm_min, &t.tm_sec, &t.tm_year) == 6) {
      for (i = 0; i < 12 && strcasecmp(mon[i], month); i++)
        ;
      t.tm_mon = i;
      t.tm_year -= 1900;
      if (mktime(&t) > 0)
        timestamp = mktime(&t);
    }
  } else
    fseek(ifp, size, SEEK_CUR);
}

void CLASS parse_qt(int end) {
  unsigned save, size;
  char tag[4];

  order = 0x4d4d;
  while (ftell(ifp) + 7 < end) {
    save = ftell(ifp);
    if ((size = get4()) < 8)
      return;
    fread(tag, 4, 1, ifp);
    if (!memcmp(tag, "moov", 4) || !memcmp(tag, "udta", 4) ||
        !memcmp(tag, "CNTH", 4))
      parse_qt(save + size);
    if (!memcmp(tag, "CNDA", 4))
      parse_jpeg(ftell(ifp));
    fseek(ifp, save + size, SEEK_SET);
  }
}

void CLASS parse_smal(int offset, int fsize) {
  int ver;

  fseek(ifp, offset + 2, SEEK_SET);
  order = 0x4949;
  ver = fgetc(ifp);
  if (ver == 6)
    fseek(ifp, 5, SEEK_CUR);
  if (get4() != fsize)
    return;
  if (ver > 6)
    data_offset = get4();
  raw_height = height = get2();
  raw_width = width = get2();
  strcpy(make, "SMaL");
  sprintf(model, "v%d %dx%d", ver, width, height);
  if (ver == 6)
    load_raw = &CLASS smal_v6_load_raw;
  if (ver == 9)
    load_raw = &CLASS smal_v9_load_raw;
}

void CLASS parse_cine() {
  unsigned off_head, off_setup, off_image, i;

  order = 0x4949;
  fseek(ifp, 4, SEEK_SET);
  is_raw = get2() == 2;
  fseek(ifp, 14, SEEK_CUR);
  is_raw *= get4();
  off_head = get4();
  off_setup = get4();
  off_image = get4();
  timestamp = get4();
  if ((i = get4()))
    timestamp = i;
  fseek(ifp, off_head + 4, SEEK_SET);
  raw_width = get4();
  raw_height = get4();
  switch (get2(), get2()) {
    case 8:
      load_raw = &CLASS eight_bit_load_raw;
      break;
    case 16:
      load_raw = &CLASS unpacked_load_raw;
  }
  fseek(ifp, off_setup + 792, SEEK_SET);
  strcpy(make, "CINE");
  sprintf(model, "%d", get4());
  fseek(ifp, 12, SEEK_CUR);
  switch ((i = get4()) & 0xffffff) {
    case 3:
      filters = 0x94949494;
      break;
    case 4:
      filters = 0x49494949;
      break;
    default:
      is_raw = 0;
  }
  fseek(ifp, 72, SEEK_CUR);
  switch ((get4() + 3600) % 360) {
    case 270:
      flip = 4;
      break;
    case 180:
      flip = 1;
      break;
    case 90:
      flip = 7;
      break;
    case 0:
      flip = 2;
  }
  cam_mul[0] = getreal(11);
  cam_mul[2] = getreal(11);
  maximum = ~(-1 << get4());
  fseek(ifp, 668, SEEK_CUR);
  shutter = get4() / 1000000000.0;
  fseek(ifp, off_image, SEEK_SET);
  if (shot_select < is_raw)
    fseek(ifp, shot_select * 8, SEEK_CUR);
  data_offset = (INT64)get4() + 8;
  data_offset += (INT64)get4() << 32;
}

void CLASS parse_redcine() {
  unsigned i, len, rdvo;

  order = 0x4d4d;
  is_raw = 0;
  fseek(ifp, 52, SEEK_SET);
  width = get4();
  height = get4();
  fseek(ifp, 0, SEEK_END);
  fseek(ifp, -(i = ftello(ifp) & 511), SEEK_CUR);
  if (get4() != i || get4() != 0x52454f42) {
    fprintf(stderr, _("%s: Tail is missing, parsing from head...\n"), ifname);
    fseek(ifp, 0, SEEK_SET);
    while ((len = get4()) != EOF) {
      if (get4() == 0x52454456)
        if (is_raw++ == shot_select)
          data_offset = ftello(ifp) - 8;
      fseek(ifp, len - 8, SEEK_CUR);
    }
  } else {
    rdvo = get4();
    fseek(ifp, 12, SEEK_CUR);
    is_raw = get4();
    fseeko(ifp, rdvo + 8 + shot_select * 4, SEEK_SET);
    data_offset = get4();
  }
}

char *CLASS foveon_gets(int offset, char *str, int len) {
  int i;
  fseek(ifp, offset, SEEK_SET);
  for (i = 0; i < len - 1; i++)
    if ((str[i] = get2()) == 0)
      break;
  str[i] = 0;
  return str;
}

void CLASS parse_foveon() {
  int entries, img = 0, off, len, tag, save, i, wide, high, pent, poff[256][2];
  char name[64], value[64];

  order = 0x4949; /* Little-endian */
  fseek(ifp, 36, SEEK_SET);
  flip = get4();
  fseek(ifp, -4, SEEK_END);
  fseek(ifp, get4(), SEEK_SET);
  if (get4() != 0x64434553)
    return; /* SECd */
  entries = (get4(), get4());
  while (entries--) {
    off = get4();
    len = get4();
    tag = get4();
    save = ftell(ifp);
    fseek(ifp, off, SEEK_SET);
    if (get4() != (0x20434553 | (tag << 24)))
      return;
    switch (tag) {
      case 0x47414d49: /* IMAG */
      case 0x32414d49: /* IMA2 */
        fseek(ifp, 8, SEEK_CUR);
        pent = get4();
        wide = get4();
        high = get4();
        if (wide > raw_width && high > raw_height) {
          switch (pent) {
            case 5:
              load_flags = 1;
            case 6:
              load_raw = &CLASS foveon_sd_load_raw;
              break;
            case 30:
              load_raw = &CLASS foveon_dp_load_raw;
              break;
            default:
              load_raw = 0;
          }
          raw_width = wide;
          raw_height = high;
          data_offset = off + 28;
          is_foveon = 1;
        }
        fseek(ifp, off + 28, SEEK_SET);
        if (fgetc(ifp) == 0xff && fgetc(ifp) == 0xd8 &&
            thumb_length < len - 28) {
          thumb_offset = off + 28;
          thumb_length = len - 28;
          write_thumb = &CLASS jpeg_thumb;
        }
        if (++img == 2 && !thumb_length) {
          thumb_offset = off + 24;
          thumb_width = wide;
          thumb_height = high;
          write_thumb = &CLASS foveon_thumb;
        }
        break;
      case 0x464d4143: /* CAMF */
        meta_offset = off + 8;
        meta_length = len - 28;
        break;
      case 0x504f5250: /* PROP */
        pent = (get4(), get4());
        fseek(ifp, 12, SEEK_CUR);
        off += pent * 8 + 24;
        if ((unsigned)pent > 256)
          pent = 256;
        for (i = 0; i < pent * 2; i++)
          ((int *)poff)[i] = off + get4() * 2;
        for (i = 0; i < pent; i++) {
          foveon_gets(poff[i][0], name, 64);
          foveon_gets(poff[i][1], value, 64);
          if (!strcmp(name, "ISO"))
            iso_speed = atoi(value);
          if (!strcmp(name, "CAMMANUF"))
            strcpy(make, value);
          if (!strcmp(name, "CAMMODEL"))
            strcpy(model, value);
          if (!strcmp(name, "WB_DESC"))
            strcpy(model2, value);
          if (!strcmp(name, "TIME"))
            timestamp = atoi(value);
          if (!strcmp(name, "EXPTIME"))
            shutter = atoi(value) / 1000000.0;
          if (!strcmp(name, "APERTURE"))
            aperture = atof(value);
          if (!strcmp(name, "FLENGTH"))
            focal_len = atof(value);
        }
#ifdef LOCALTIME
        timestamp = mktime(gmtime(&timestamp));
#endif
    }
    fseek(ifp, save, SEEK_SET);
  }
}

/*
All matrices are from Adobe DNG Converter unless otherwise noted.
*/
void CLASS adobe_coeff(const char *make, const char *model) {
  static const struct {
    const char *prefix;
    short black, maximum, trans[12];
  } table[] = {
      {"AgfaPhoto DC-833m",
       0,
       0, /* DJC */
       {11438, -3762, -1115, -2409, 9914, 2497, -1227, 2295, 5300}},
      {"Apple QuickTake",
       0,
       0, /* DJC */
       {21392, -5653, -3353, 2406, 8010, -415, 7166, 1427, 2078}},
      {"Canon EOS D2000",
       0,
       0,
       {24542, -10860, -3401, -1490, 11370, -297, 2858, -605, 3225}},
      {"Canon EOS D6000",
       0,
       0,
       {20482, -7172, -3125, -1033, 10410, -285, 2542, 226, 3136}},
      {"Canon EOS D30",
       0,
       0,
       {9805, -2689, -1312, -5803, 13064, 3068, -2438, 3075, 8775}},
      {"Canon EOS D60",
       0,
       0xfa0,
       {6188, -1341, -890, -7168, 14489, 2937, -2640, 3228, 8483}},
      {"Canon EOS 5DS",
       0,
       0x3c96,
       {6250, -711, -808, -5153, 12794, 2636, -1249, 2198, 5610}},
      {"Canon EOS 5D Mark III",
       0,
       0x3c80,
       {6722, -635, -963, -4287, 12460, 2028, -908, 2162, 5668}},
      {"Canon EOS 5D Mark II",
       0,
       0x3cf0,
       {4716, 603, -830, -7798, 15474, 2480, -1496, 1937, 6651}},
      {"Canon EOS 5D",
       0,
       0xe6c,
       {6347, -479, -972, -8297, 15954, 2480, -1968, 2131, 7649}},
      {"Canon EOS 6D",
       0,
       0x3c82,
       {7034, -804, -1014, -4420, 12564, 2058, -851, 1994, 5758}},
      {"Canon EOS 7D Mark II",
       0,
       0x3510,
       {7268, -1082, -969, -4186, 11839, 2663, -825, 2029, 5839}},
      {"Canon EOS 7D",
       0,
       0x3510,
       {6844, -996, -856, -3876, 11761, 2396, -593, 1772, 6198}},
      {"Canon EOS 10D",
       0,
       0xfa0,
       {8197, -2000, -1118, -6714, 14335, 2592, -2536, 3178, 8266}},
      {"Canon EOS 20Da",
       0,
       0,
       {14155, -5065, -1382, -6550, 14633, 2039, -1623, 1824, 6561}},
      {"Canon EOS 20D",
       0,
       0xfff,
       {6599, -537, -891, -8071, 15783, 2424, -1983, 2234, 7462}},
      {"Canon EOS 30D",
       0,
       0,
       {6257, -303, -1000, -7880, 15621, 2396, -1714, 1904, 7046}},
      {"Canon EOS 40D",
       0,
       0x3f60,
       {6071, -747, -856, -7653, 15365, 2441, -2025, 2553, 7315}},
      {"Canon EOS 50D",
       0,
       0x3d93,
       {4920, 616, -593, -6493, 13964, 2784, -1774, 3178, 7005}},
      {"Canon EOS 60D",
       0,
       0x2ff7,
       {6719, -994, -925, -4408, 12426, 2211, -887, 2129, 6051}},
      {"Canon EOS 70D",
       0,
       0x3bc7,
       {7034, -804, -1014, -4420, 12564, 2058, -851, 1994, 5758}},
      {"Canon EOS 80D",
       0,
       0,
       {7457, -671, -937, -4849, 12495, 2643, -1213, 2354, 5492}},
      {"Canon EOS 100D",
       0,
       0x350f,
       {6602, -841, -939, -4472, 12458, 2247, -975, 2039, 6148}},
      {"Canon EOS 300D",
       0,
       0xfa0,
       {8197, -2000, -1118, -6714, 14335, 2592, -2536, 3178, 8266}},
      {"Canon EOS 350D",
       0,
       0xfff,
       {6018, -617, -965, -8645, 15881, 2975, -1530, 1719, 7642}},
      {"Canon EOS 400D",
       0,
       0xe8e,
       {7054, -1501, -990, -8156, 15544, 2812, -1278, 1414, 7796}},
      {"Canon EOS 450D",
       0,
       0x390d,
       {5784, -262, -821, -7539, 15064, 2672, -1982, 2681, 7427}},
      {"Canon EOS 500D",
       0,
       0x3479,
       {4763, 712, -646, -6821, 14399, 2640, -1921, 3276, 6561}},
      {"Canon EOS 550D",
       0,
       0x3dd7,
       {6941, -1164, -857, -3825, 11597, 2534, -416, 1540, 6039}},
      {"Canon EOS 600D",
       0,
       0x3510,
       {6461, -907, -882, -4300, 12184, 2378, -819, 1944, 5931}},
      {"Canon EOS 650D",
       0,
       0x354d,
       {6602, -841, -939, -4472, 12458, 2247, -975, 2039, 6148}},
      {"Canon EOS 700D",
       0,
       0x3c00,
       {6602, -841, -939, -4472, 12458, 2247, -975, 2039, 6148}},
      {"Canon EOS 750D",
       0,
       0x368e,
       {6362, -823, -847, -4426, 12109, 2616, -743, 1857, 5635}},
      {"Canon EOS 760D",
       0,
       0x350f,
       {6362, -823, -847, -4426, 12109, 2616, -743, 1857, 5635}},
      {"Canon EOS 1000D",
       0,
       0xe43,
       {6771, -1139, -977, -7818, 15123, 2928, -1244, 1437, 7533}},
      {"Canon EOS 1100D",
       0,
       0x3510,
       {6444, -904, -893, -4563, 12308, 2535, -903, 2016, 6728}},
      {"Canon EOS 1200D",
       0,
       0x37c2,
       {6461, -907, -882, -4300, 12184, 2378, -819, 1944, 5931}},
      {"Canon EOS 1300D",
       0,
       0x3510,
       {6939, -1016, -866, -4428, 12473, 2177, -1175, 2178, 6162}},
      {"Canon EOS M3",
       0,
       0,
       {6362, -823, -847, -4426, 12109, 2616, -743, 1857, 5635}},
      {"Canon EOS M10",
       0,
       0,
       {6400, -480, -888, -5294, 13416, 2047, -1296, 2203, 6137}},
      {"Canon EOS M",
       0,
       0,
       {6602, -841, -939, -4472, 12458, 2247, -975, 2039, 6148}},
      {"Canon EOS-1Ds Mark III",
       0,
       0x3bb0,
       {5859, -211, -930, -8255, 16017, 2353, -1732, 1887, 7448}},
      {"Canon EOS-1Ds Mark II",
       0,
       0xe80,
       {6517, -602, -867, -8180, 15926, 2378, -1618, 1771, 7633}},
      {"Canon EOS-1D Mark IV",
       0,
       0x3bb0,
       {6014, -220, -795, -4109, 12014, 2361, -561, 1824, 5787}},
      {"Canon EOS-1D Mark III",
       0,
       0x3bb0,
       {6291, -540, -976, -8350, 16145, 2311, -1714, 1858, 7326}},
      {"Canon EOS-1D Mark II N",
       0,
       0xe80,
       {6240, -466, -822, -8180, 15825, 2500, -1801, 1938, 8042}},
      {"Canon EOS-1D Mark II",
       0,
       0xe80,
       {6264, -582, -724, -8312, 15948, 2504, -1744, 1919, 8664}},
      {"Canon EOS-1DS",
       0,
       0xe20,
       {4374, 3631, -1743, -7520, 15212, 2472, -2892, 3632, 8161}},
      {"Canon EOS-1D C",
       0,
       0x3c4e,
       {6847, -614, -1014, -4669, 12737, 2139, -1197, 2488, 6846}},
      {"Canon EOS-1D X Mark II",
       0,
       0,
       {7596, -978, -967, -4808, 12571, 2503, -1398, 2567, 5752}},
      {"Canon EOS-1D X",
       0,
       0x3c4e,
       {6847, -614, -1014, -4669, 12737, 2139, -1197, 2488, 6846}},
      {"Canon EOS-1D",
       0,
       0xe20,
       {6806, -179, -1020, -8097, 16415, 1687, -3267, 4236, 7690}},
      {"Canon EOS C500",
       853,
       0, /* DJC */
       {17851, -10604, 922, -7425, 16662, 763, -3660, 3636, 22278}},
      {"Canon PowerShot A530", 0, 0, {0}}, /* don't want the A5 matrix */
      {"Canon PowerShot A50",
       0,
       0,
       {-5300, 9846, 1776, 3436, 684, 3939, -5540, 9879, 6200, -1404, 11175,
        217}},
      {"Canon PowerShot A5",
       0,
       0,
       {-4801, 9475, 1952, 2926, 1611, 4094, -5259, 10164, 5947, -1554, 10883,
        547}},
      {"Canon PowerShot G10",
       0,
       0,
       {11093, -3906, -1028, -5047, 12492, 2879, -1003, 1750, 5561}},
      {"Canon PowerShot G11",
       0,
       0,
       {12177, -4817, -1069, -1612, 9864, 2049, -98, 850, 4471}},
      {"Canon PowerShot G12",
       0,
       0,
       {13244, -5501, -1248, -1508, 9858, 1935, -270, 1083, 4366}},
      {"Canon PowerShot G15",
       0,
       0,
       {7474, -2301, -567, -4056, 11456, 2975, -222, 716, 4181}},
      {"Canon PowerShot G16",
       0,
       0,
       {8020, -2687, -682, -3704, 11879, 2052, -965, 1921, 5556}},
      {"Canon PowerShot G1 X",
       0,
       0,
       {7378, -1255, -1043, -4088, 12251, 2048, -876, 1946, 5805}},
      {"Canon PowerShot G1",
       0,
       0,
       {-4778, 9467, 2172, 4743, -1141, 4344, -5146, 9908, 6077, -1566, 11051,
        557}},
      {"Canon PowerShot G2",
       0,
       0,
       {9087, -2693, -1049, -6715, 14382, 2537, -2291, 2819, 7790}},
      {"Canon PowerShot G3 X",
       0,
       0,
       {9701, -3857, -921, -3149, 11537, 1817, -786, 1817, 5147}},
      {"Canon PowerShot G3",
       0,
       0,
       {9212, -2781, -1073, -6573, 14189, 2605, -2300, 2844, 7664}},
      {"Canon PowerShot G5 X",
       0,
       0,
       {9602, -3823, -937, -2984, 11495, 1675, -407, 1415, 5049}},
      {"Canon PowerShot G5",
       0,
       0,
       {9757, -2872, -933, -5972, 13861, 2301, -1622, 2328, 7212}},
      {"Canon PowerShot G6",
       0,
       0,
       {9877, -3775, -871, -7613, 14807, 3072, -1448, 1305, 7485}},
      {"Canon PowerShot G7 X",
       0,
       0,
       {9602, -3823, -937, -2984, 11495, 1675, -407, 1415, 5049}},
      {"Canon PowerShot G9 X",
       0,
       0,
       {9602, -3823, -937, -2984, 11495, 1675, -407, 1415, 5049}},
      {"Canon PowerShot G9",
       0,
       0,
       {7368, -2141, -598, -5621, 13254, 2625, -1418, 1696, 5743}},
      {"Canon PowerShot Pro1",
       0,
       0,
       {10062, -3522, -999, -7643, 15117, 2730, -765, 817, 7323}},
      {"Canon PowerShot Pro70",
       34,
       0,
       {-4155, 9818, 1529, 3939, -25, 4522, -5521, 9870, 6610, -2238, 10873,
        1342}},
      {"Canon PowerShot Pro90",
       0,
       0,
       {-4963, 9896, 2235, 4642, -987, 4294, -5162, 10011, 5859, -1770, 11230,
        577}},
      {"Canon PowerShot S30",
       0,
       0,
       {10566, -3652, -1129, -6552, 14662, 2006, -2197, 2581, 7670}},
      {"Canon PowerShot S40",
       0,
       0,
       {8510, -2487, -940, -6869, 14231, 2900, -2318, 2829, 9013}},
      {"Canon PowerShot S45",
       0,
       0,
       {8163, -2333, -955, -6682, 14174, 2751, -2077, 2597, 8041}},
      {"Canon PowerShot S50",
       0,
       0,
       {8882, -2571, -863, -6348, 14234, 2288, -1516, 2172, 6569}},
      {"Canon PowerShot S60",
       0,
       0,
       {8795, -2482, -797, -7804, 15403, 2573, -1422, 1996, 7082}},
      {"Canon PowerShot S70",
       0,
       0,
       {9976, -3810, -832, -7115, 14463, 2906, -901, 989, 7889}},
      {"Canon PowerShot S90",
       0,
       0,
       {12374, -5016, -1049, -1677, 9902, 2078, -83, 852, 4683}},
      {"Canon PowerShot S95",
       0,
       0,
       {13440, -5896, -1279, -1236, 9598, 1931, -180, 1001, 4651}},
      {"Canon PowerShot S100",
       0,
       0,
       {7968, -2565, -636, -2873, 10697, 2513, 180, 667, 4211}},
      {"Canon PowerShot S110",
       0,
       0,
       {8039, -2643, -654, -3783, 11230, 2930, -206, 690, 4194}},
      {"Canon PowerShot S120",
       0,
       0,
       {6961, -1685, -695, -4625, 12945, 1836, -1114, 2152, 5518}},
      {"Canon PowerShot SX1 IS",
       0,
       0,
       {6578, -259, -502, -5974, 13030, 3309, -308, 1058, 4970}},
      {"Canon PowerShot SX50 HS",
       0,
       0,
       {12432, -4753, -1247, -2110, 10691, 1629, -412, 1623, 4926}},
      {"Canon PowerShot SX60 HS",
       0,
       0,
       {13161, -5451, -1344, -1989, 10654, 1531, -47, 1271, 4955}},
      {"Canon PowerShot A3300",
       0,
       0, /* DJC */
       {10826, -3654, -1023, -3215, 11310, 1906, 0, 999, 4960}},
      {"Canon PowerShot A470",
       0,
       0, /* DJC */
       {12513, -4407, -1242, -2680, 10276, 2405, -878, 2215, 4734}},
      {"Canon PowerShot A610",
       0,
       0, /* DJC */
       {15591, -6402, -1592, -5365, 13198, 2168, -1300, 1824, 5075}},
      {"Canon PowerShot A620",
       0,
       0, /* DJC */
       {15265, -6193, -1558, -4125, 12116, 2010, -888, 1639, 5220}},
      {"Canon PowerShot A630",
       0,
       0, /* DJC */
       {14201, -5308, -1757, -6087, 14472, 1617, -2191, 3105, 5348}},
      {"Canon PowerShot A640",
       0,
       0, /* DJC */
       {13124, -5329, -1390, -3602, 11658, 1944, -1612, 2863, 4885}},
      {"Canon PowerShot A650",
       0,
       0, /* DJC */
       {9427, -3036, -959, -2581, 10671, 1911, -1039, 1982, 4430}},
      {"Canon PowerShot A720",
       0,
       0, /* DJC */
       {14573, -5482, -1546, -1266, 9799, 1468, -1040, 1912, 3810}},
      {"Canon PowerShot S3 IS",
       0,
       0, /* DJC */
       {14062, -5199, -1446, -4712, 12470, 2243, -1286, 2028, 4836}},
      {"Canon PowerShot SX110 IS",
       0,
       0, /* DJC */
       {14134, -5576, -1527, -1991, 10719, 1273, -1158, 1929, 3581}},
      {"Canon PowerShot SX220",
       0,
       0, /* DJC */
       {13898, -5076, -1447, -1405, 10109, 1297, -244, 1860, 3687}},
      {"Canon IXUS 160",
       0,
       0, /* DJC */
       {11657, -3781, -1136, -3544, 11262, 2283, -160, 1219, 4700}},
      {"Casio EX-S20",
       0,
       0, /* DJC */
       {11634, -3924, -1128, -4968, 12954, 2015, -1588, 2648, 7206}},
      {"Casio EX-Z750",
       0,
       0, /* DJC */
       {10819, -3873, -1099, -4903, 13730, 1175, -1755, 3751, 4632}},
      {"Casio EX-Z10",
       128,
       0xfff, /* DJC */
       {9790, -3338, -603, -2321, 10222, 2099, -344, 1273, 4799}},
      {"CINE 650", 0, 0, {3390, 480, -500, -800, 3610, 340, -550, 2336, 1192}},
      {"CINE 660", 0, 0, {3390, 480, -500, -800, 3610, 340, -550, 2336, 1192}},
      {"CINE",
       0,
       0,
       {20183, -4295, -423, -3940, 15330, 3985, -280, 4870, 9800}},
      {"Contax N Digital",
       0,
       0xf1e,
       {7777, 1285, -1053, -9280, 16543, 2916, -3677, 5679, 7060}},
      {"DXO ONE",
       0,
       0,
       {6596, -2079, -562, -4782, 13016, 1933, -970, 1581, 5181}},
      {"Epson R-D1",
       0,
       0,
       {6827, -1878, -732, -8429, 16012, 2564, -704, 592, 7145}},
      {"Fujifilm E550",
       0,
       0,
       {11044, -3888, -1120, -7248, 15168, 2208, -1531, 2277, 8069}},
      {"Fujifilm E900",
       0,
       0,
       {9183, -2526, -1078, -7461, 15071, 2574, -2022, 2440, 8639}},
      {"Fujifilm F5",
       0,
       0,
       {13690, -5358, -1474, -3369, 11600, 1998, -132, 1554, 4395}},
      {"Fujifilm F6",
       0,
       0,
       {13690, -5358, -1474, -3369, 11600, 1998, -132, 1554, 4395}},
      {"Fujifilm F77",
       0,
       0xfe9,
       {13690, -5358, -1474, -3369, 11600, 1998, -132, 1554, 4395}},
      {"Fujifilm F7",
       0,
       0,
       {10004, -3219, -1201, -7036, 15047, 2107, -1863, 2565, 7736}},
      {"Fujifilm F8",
       0,
       0,
       {13690, -5358, -1474, -3369, 11600, 1998, -132, 1554, 4395}},
      {"Fujifilm S100FS",
       514,
       0,
       {11521, -4355, -1065, -6524, 13767, 3058, -1466, 1984, 6045}},
      {"Fujifilm S1",
       0,
       0,
       {12297, -4882, -1202, -2106, 10691, 1623, -88, 1312, 4790}},
      {"Fujifilm S20Pro",
       0,
       0,
       {10004, -3219, -1201, -7036, 15047, 2107, -1863, 2565, 7736}},
      {"Fujifilm S20",
       512,
       0x3fff,
       {11401, -4498, -1312, -5088, 12751, 2613, -838, 1568, 5941}},
      {"Fujifilm S2Pro",
       128,
       0,
       {12492, -4690, -1402, -7033, 15423, 1647, -1507, 2111, 7697}},
      {"Fujifilm S3Pro",
       0,
       0,
       {11807, -4612, -1294, -8927, 16968, 1988, -2120, 2741, 8006}},
      {"Fujifilm S5Pro",
       0,
       0,
       {12300, -5110, -1304, -9117, 17143, 1998, -1947, 2448, 8100}},
      {"Fujifilm S5000",
       0,
       0,
       {8754, -2732, -1019, -7204, 15069, 2276, -1702, 2334, 6982}},
      {"Fujifilm S5100",
       0,
       0,
       {11940, -4431, -1255, -6766, 14428, 2542, -993, 1165, 7421}},
      {"Fujifilm S5500",
       0,
       0,
       {11940, -4431, -1255, -6766, 14428, 2542, -993, 1165, 7421}},
      {"Fujifilm S5200",
       0,
       0,
       {9636, -2804, -988, -7442, 15040, 2589, -1803, 2311, 8621}},
      {"Fujifilm S5600",
       0,
       0,
       {9636, -2804, -988, -7442, 15040, 2589, -1803, 2311, 8621}},
      {"Fujifilm S6",
       0,
       0,
       {12628, -4887, -1401, -6861, 14996, 1962, -2198, 2782, 7091}},
      {"Fujifilm S7000",
       0,
       0,
       {10190, -3506, -1312, -7153, 15051, 2238, -2003, 2399, 7505}},
      {"Fujifilm S9000",
       0,
       0,
       {10491, -3423, -1145, -7385, 15027, 2538, -1809, 2275, 8692}},
      {"Fujifilm S9500",
       0,
       0,
       {10491, -3423, -1145, -7385, 15027, 2538, -1809, 2275, 8692}},
      {"Fujifilm S9100",
       0,
       0,
       {12343, -4515, -1285, -7165, 14899, 2435, -1895, 2496, 8800}},
      {"Fujifilm S9600",
       0,
       0,
       {12343, -4515, -1285, -7165, 14899, 2435, -1895, 2496, 8800}},
      {"Fujifilm SL1000",
       0,
       0,
       {11705, -4262, -1107, -2282, 10791, 1709, -555, 1713, 4945}},
      {"Fujifilm IS-1",
       0,
       0,
       {21461, -10807, -1441, -2332, 10599, 1999, 289, 875, 7703}},
      {"Fujifilm IS Pro",
       0,
       0,
       {12300, -5110, -1304, -9117, 17143, 1998, -1947, 2448, 8100}},
      {"Fujifilm HS10 HS11",
       0,
       0xf68,
       {12440, -3954, -1183, -1123, 9674, 1708, -83, 1614, 4086}},
      {"Fujifilm HS2",
       0,
       0,
       {13690, -5358, -1474, -3369, 11600, 1998, -132, 1554, 4395}},
      {"Fujifilm HS3",
       0,
       0,
       {13690, -5358, -1474, -3369, 11600, 1998, -132, 1554, 4395}},
      {"Fujifilm HS50EXR",
       0,
       0,
       {12085, -4727, -953, -3257, 11489, 2002, -511, 2046, 4592}},
      {"Fujifilm F900EXR",
       0,
       0,
       {12085, -4727, -953, -3257, 11489, 2002, -511, 2046, 4592}},
      {"Fujifilm X100S",
       0,
       0,
       {10592, -4262, -1008, -3514, 11355, 2465, -870, 2025, 6386}},
      {"Fujifilm X100T",
       0,
       0,
       {10592, -4262, -1008, -3514, 11355, 2465, -870, 2025, 6386}},
      {"Fujifilm X100",
       0,
       0,
       {12161, -4457, -1069, -5034, 12874, 2400, -795, 1724, 6904}},
      {"Fujifilm X10",
       0,
       0,
       {13509, -6199, -1254, -4430, 12733, 1865, -331, 1441, 5022}},
      {"Fujifilm X20",
       0,
       0,
       {11768, -4971, -1133, -4904, 12927, 2183, -480, 1723, 4605}},
      {"Fujifilm X30",
       0,
       0,
       {12328, -5256, -1144, -4469, 12927, 1675, -87, 1291, 4351}},
      {"Fujifilm X70",
       0,
       0,
       {10450, -4329, -878, -3217, 11105, 2421, -752, 1758, 6519}},
      {"Fujifilm X-Pro1",
       0,
       0,
       {10413, -3996, -993, -3721, 11640, 2361, -733, 1540, 6011}},
      {"Fujifilm X-Pro2",
       0,
       0,
       {11434, -4948, -1210, -3746, 12042, 1903, -666, 1479, 5235}},
      {"Fujifilm X-A1",
       0,
       0,
       {11086, -4555, -839, -3512, 11310, 2517, -815, 1341, 5940}},
      {"Fujifilm X-A2",
       0,
       0,
       {10763, -4560, -917, -3346, 11311, 2322, -475, 1135, 5843}},
      {"Fujifilm X-E1",
       0,
       0,
       {10413, -3996, -993, -3721, 11640, 2361, -733, 1540, 6011}},
      {"Fujifilm X-E2S",
       0,
       0,
       {11562, -5118, -961, -3022, 11007, 2311, -525, 1569, 6097}},
      {"Fujifilm X-E2",
       0,
       0,
       {8458, -2451, -855, -4597, 12447, 2407, -1475, 2482, 6526}},
      {"Fujifilm X-M1",
       0,
       0,
       {10413, -3996, -993, -3721, 11640, 2361, -733, 1540, 6011}},
      {"Fujifilm X-S1",
       0,
       0,
       {13509, -6199, -1254, -4430, 12733, 1865, -331, 1441, 5022}},
      {"Fujifilm X-T1",
       0,
       0, /* also X-T10 */
       {8458, -2451, -855, -4597, 12447, 2407, -1475, 2482, 6526}},
      {"Fujifilm XF1",
       0,
       0,
       {13509, -6199, -1254, -4430, 12733, 1865, -331, 1441, 5022}},
      {"Fujifilm XQ",
       0,
       0, /* XQ1 and XQ2 */
       {9252, -2704, -1064, -5893, 14265, 1717, -1101, 2341, 4349}},
      {"Imacon Ixpress",
       0,
       0, /* DJC */
       {7025, -1415, -704, -5188, 13765, 1424, -1248, 2742, 6038}},
      {"Kodak NC2000",
       0,
       0,
       {13891, -6055, -803, -465, 9919, 642, 2121, 82, 1291}},
      {"Kodak DCS315C",
       8,
       0,
       {17523, -4827, -2510, 756, 8546, -137, 6113, 1649, 2250}},
      {"Kodak DCS330C",
       8,
       0,
       {20620, -7572, -2801, -103, 10073, -396, 3551, -233, 2220}},
      {"Kodak DCS420",
       0,
       0,
       {10868, -1852, -644, -1537, 11083, 484, 2343, 628, 2216}},
      {"Kodak DCS460",
       0,
       0,
       {10592, -2206, -967, -1944, 11685, 230, 2206, 670, 1273}},
      {"Kodak EOSDCS1",
       0,
       0,
       {10592, -2206, -967, -1944, 11685, 230, 2206, 670, 1273}},
      {"Kodak EOSDCS3B",
       0,
       0,
       {9898, -2700, -940, -2478, 12219, 206, 1985, 634, 1031}},
      {"Kodak DCS520C",
       178,
       0,
       {24542, -10860, -3401, -1490, 11370, -297, 2858, -605, 3225}},
      {"Kodak DCS560C",
       177,
       0,
       {20482, -7172, -3125, -1033, 10410, -285, 2542, 226, 3136}},
      {"Kodak DCS620C",
       177,
       0,
       {23617, -10175, -3149, -2054, 11749, -272, 2586, -489, 3453}},
      {"Kodak DCS620X",
       176,
       0,
       {13095, -6231, 154, 12221, -21, -2137, 895, 4602, 2258}},
      {"Kodak DCS660C",
       173,
       0,
       {18244, -6351, -2739, -791, 11193, -521, 3711, -129, 2802}},
      {"Kodak DCS720X",
       0,
       0,
       {11775, -5884, 950, 9556, 1846, -1286, -1019, 6221, 2728}},
      {"Kodak DCS760C",
       0,
       0,
       {16623, -6309, -1411, -4344, 13923, 323, 2285, 274, 2926}},
      {"Kodak DCS Pro SLR",
       0,
       0,
       {5494, 2393, -232, -6427, 13850, 2846, -1876, 3997, 5445}},
      {"Kodak DCS Pro 14nx",
       0,
       0,
       {5494, 2393, -232, -6427, 13850, 2846, -1876, 3997, 5445}},
      {"Kodak DCS Pro 14",
       0,
       0,
       {7791, 3128, -776, -8588, 16458, 2039, -2455, 4006, 6198}},
      {"Kodak ProBack645",
       0,
       0,
       {16414, -6060, -1470, -3555, 13037, 473, 2545, 122, 4948}},
      {"Kodak ProBack",
       0,
       0,
       {21179, -8316, -2918, -915, 11019, -165, 3477, -180, 4210}},
      {"Kodak P712",
       0,
       0,
       {9658, -3314, -823, -5163, 12695, 2768, -1342, 1843, 6044}},
      {"Kodak P850",
       0,
       0xf7c,
       {10511, -3836, -1102, -6946, 14587, 2558, -1481, 1792, 6246}},
      {"Kodak P880",
       0,
       0xfff,
       {12805, -4662, -1376, -7480, 15267, 2360, -1626, 2194, 7904}},
      {"Kodak EasyShare Z980",
       0,
       0,
       {11313, -3559, -1101, -3893, 11891, 2257, -1214, 2398, 4908}},
      {"Kodak EasyShare Z981",
       0,
       0,
       {12729, -4717, -1188, -1367, 9187, 2582, 274, 860, 4411}},
      {"Kodak EasyShare Z990",
       0,
       0xfed,
       {11749, -4048, -1309, -1867, 10572, 1489, -138, 1449, 4522}},
      {"Kodak EASYSHARE Z1015",
       0,
       0xef1,
       {11265, -4286, -992, -4694, 12343, 2647, -1090, 1523, 5447}},
      {"Leaf CMost",
       0,
       0,
       {3952, 2189, 449, -6701, 14585, 2275, -4536, 7349, 6536}},
      {"Leaf Valeo 6",
       0,
       0,
       {3952, 2189, 449, -6701, 14585, 2275, -4536, 7349, 6536}},
      {"Leaf Aptus 54S",
       0,
       0,
       {8236, 1746, -1314, -8251, 15953, 2428, -3673, 5786, 5771}},
      {"Leaf Aptus 65",
       0,
       0,
       {7914, 1414, -1190, -8777, 16582, 2280, -2811, 4605, 5562}},
      {"Leaf Aptus 75",
       0,
       0,
       {7914, 1414, -1190, -8777, 16582, 2280, -2811, 4605, 5562}},
      {"Leaf",
       0,
       0,
       {8236, 1746, -1314, -8251, 15953, 2428, -3673, 5786, 5771}},
      {"Mamiya ZD",
       0,
       0,
       {7645, 2579, -1363, -8689, 16717, 2015, -3712, 5941, 5961}},
      {"Micron 2010",
       110,
       0, /* DJC */
       {16695, -3761, -2151, 155, 9682, 163, 3433, 951, 4904}},
      {"Minolta DiMAGE 5",
       0,
       0xf7d,
       {8983, -2942, -963, -6556, 14476, 2237, -2426, 2887, 8014}},
      {"Minolta DiMAGE 7Hi",
       0,
       0xf7d,
       {11368, -3894, -1242, -6521, 14358, 2339, -2475, 3056, 7285}},
      {"Minolta DiMAGE 7",
       0,
       0xf7d,
       {9144, -2777, -998, -6676, 14556, 2281, -2470, 3019, 7744}},
      {"Minolta DiMAGE A1",
       0,
       0xf8b,
       {9274, -2547, -1167, -8220, 16323, 1943, -2273, 2720, 8340}},
      {"Minolta DiMAGE A200",
       0,
       0,
       {8560, -2487, -986, -8112, 15535, 2771, -1209, 1324, 7743}},
      {"Minolta DiMAGE A2",
       0,
       0xf8f,
       {9097, -2726, -1053, -8073, 15506, 2762, -966, 981, 7763}},
      {"Minolta DiMAGE Z2",
       0,
       0, /* DJC */
       {11280, -3564, -1370, -4655, 12374, 2282, -1423, 2168, 5396}},
      {"Minolta DYNAX 5",
       0,
       0xffb,
       {10284, -3283, -1086, -7957, 15762, 2316, -829, 882, 6644}},
      {"Minolta DYNAX 7",
       0,
       0xffb,
       {10239, -3104, -1099, -8037, 15727, 2451, -927, 925, 6871}},
      {"Motorola PIXL",
       0,
       0, /* DJC */
       {8898, -989, -1033, -3292, 11619, 1674, -661, 3178, 5216}},
      {"Nikon D100",
       0,
       0,
       {5902, -933, -782, -8983, 16719, 2354, -1402, 1455, 6464}},
      {"Nikon D1H",
       0,
       0,
       {7577, -2166, -926, -7454, 15592, 1934, -2377, 2808, 8606}},
      {"Nikon D1X",
       0,
       0,
       {7702, -2245, -975, -9114, 17242, 1875, -2679, 3055, 8521}},
      {"Nikon D1",
       0,
       0, /* multiplied by 2.218750, 1.0, 1.148438 */
       {16772, -4726, -2141, -7611, 15713, 1972, -2846, 3494, 9521}},
      {"Nikon D200",
       0,
       0xfbc,
       {8367, -2248, -763, -8758, 16447, 2422, -1527, 1550, 8053}},
      {"Nikon D2H",
       0,
       0,
       {5710, -901, -615, -8594, 16617, 2024, -2975, 4120, 6830}},
      {"Nikon D2X",
       0,
       0,
       {10231, -2769, -1255, -8301, 15900, 2552, -797, 680, 7148}},
      {"Nikon D3000",
       0,
       0,
       {8736, -2458, -935, -9075, 16894, 2251, -1354, 1242, 8263}},
      {"Nikon D3100",
       0,
       0,
       {7911, -2167, -813, -5327, 13150, 2408, -1288, 2483, 7968}},
      {"Nikon D3200",
       0,
       0xfb9,
       {7013, -1408, -635, -5268, 12902, 2640, -1470, 2801, 7379}},
      {"Nikon D3300",
       0,
       0,
       {6988, -1384, -714, -5631, 13410, 2447, -1485, 2204, 7318}},
      {"Nikon D300",
       0,
       0,
       {9030, -1992, -715, -8465, 16302, 2255, -2689, 3217, 8069}},
      {"Nikon D3X",
       0,
       0,
       {7171, -1986, -648, -8085, 15555, 2718, -2170, 2512, 7457}},
      {"Nikon D3S",
       0,
       0,
       {8828, -2406, -694, -4874, 12603, 2541, -660, 1509, 7587}},
      {"Nikon D3",
       0,
       0,
       {8139, -2171, -663, -8747, 16541, 2295, -1925, 2008, 8093}},
      {"Nikon D40X",
       0,
       0,
       {8819, -2543, -911, -9025, 16928, 2151, -1329, 1213, 8449}},
      {"Nikon D40",
       0,
       0,
       {6992, -1668, -806, -8138, 15748, 2543, -874, 850, 7897}},
      {"Nikon D4S",
       0,
       0,
       {8598, -2848, -857, -5618, 13606, 2195, -1002, 1773, 7137}},
      {"Nikon D4",
       0,
       0,
       {8598, -2848, -857, -5618, 13606, 2195, -1002, 1773, 7137}},
      {"Nikon Df",
       0,
       0,
       {8598, -2848, -857, -5618, 13606, 2195, -1002, 1773, 7137}},
      {"Nikon D5000",
       0,
       0xf00,
       {7309, -1403, -519, -8474, 16008, 2622, -2433, 2826, 8064}},
      {"Nikon D5100",
       0,
       0x3de6,
       {8198, -2239, -724, -4871, 12389, 2798, -1043, 2050, 7181}},
      {"Nikon D5200",
       0,
       0,
       {8322, -3112, -1047, -6367, 14342, 2179, -988, 1638, 6394}},
      {"Nikon D5300",
       0,
       0,
       {6988, -1384, -714, -5631, 13410, 2447, -1485, 2204, 7318}},
      {"Nikon D5500",
       0,
       0,
       {8821, -2938, -785, -4178, 12142, 2287, -824, 1651, 6860}},
      {"Nikon D500",
       0,
       0,
       {8813, -3210, -1036, -4703, 12868, 2021, -1054, 1940, 6129}},
      {"Nikon D50",
       0,
       0,
       {7732, -2422, -789, -8238, 15884, 2498, -859, 783, 7330}},
      {"Nikon D5",
       0,
       0,
       {9200, -3522, -992, -5755, 13803, 2117, -753, 1486, 6338}},
      {"Nikon D600",
       0,
       0x3e07,
       {8178, -2245, -609, -4857, 12394, 2776, -1207, 2086, 7298}},
      {"Nikon D610",
       0,
       0,
       {8178, -2245, -609, -4857, 12394, 2776, -1207, 2086, 7298}},
      {"Nikon D60",
       0,
       0,
       {8736, -2458, -935, -9075, 16894, 2251, -1354, 1242, 8263}},
      {"Nikon D7000",
       0,
       0,
       {8198, -2239, -724, -4871, 12389, 2798, -1043, 2050, 7181}},
      {"Nikon D7100",
       0,
       0,
       {8322, -3112, -1047, -6367, 14342, 2179, -988, 1638, 6394}},
      {"Nikon D7200",
       0,
       0,
       {8322, -3112, -1047, -6367, 14342, 2179, -988, 1638, 6394}},
      {"Nikon D750",
       0,
       0,
       {9020, -2890, -715, -4535, 12436, 2348, -934, 1919, 7086}},
      {"Nikon D700",
       0,
       0,
       {8139, -2171, -663, -8747, 16541, 2295, -1925, 2008, 8093}},
      {"Nikon D70",
       0,
       0,
       {7732, -2422, -789, -8238, 15884, 2498, -859, 783, 7330}},
      {"Nikon D810",
       0,
       0,
       {9369, -3195, -791, -4488, 12430, 2301, -893, 1796, 6872}},
      {"Nikon D800",
       0,
       0,
       {7866, -2108, -555, -4869, 12483, 2681, -1176, 2069, 7501}},
      {"Nikon D80",
       0,
       0,
       {8629, -2410, -883, -9055, 16940, 2171, -1490, 1363, 8520}},
      {"Nikon D90",
       0,
       0xf00,
       {7309, -1403, -519, -8474, 16008, 2622, -2434, 2826, 8064}},
      {"Nikon E700",
       0,
       0x3dd, /* DJC */
       {-3746, 10611, 1665, 9621, -1734, 2114, -2389, 7082, 3064, 3406, 6116,
        -244}},
      {"Nikon E800",
       0,
       0x3dd, /* DJC */
       {-3746, 10611, 1665, 9621, -1734, 2114, -2389, 7082, 3064, 3406, 6116,
        -244}},
      {"Nikon E950",
       0,
       0x3dd, /* DJC */
       {-3746, 10611, 1665, 9621, -1734, 2114, -2389, 7082, 3064, 3406, 6116,
        -244}},
      {"Nikon E995",
       0,
       0, /* copied from E5000 */
       {-5547, 11762, 2189, 5814, -558, 3342, -4924, 9840, 5949, 688, 9083,
        96}},
      {"Nikon E2100",
       0,
       0, /* copied from Z2, new white balance */
       {13142, -4152, -1596, -4655, 12374, 2282, -1769, 2696, 6711}},
      {"Nikon E2500",
       0,
       0,
       {-5547, 11762, 2189, 5814, -558, 3342, -4924, 9840, 5949, 688, 9083,
        96}},
      {"Nikon E3200",
       0,
       0, /* DJC */
       {9846, -2085, -1019, -3278, 11109, 2170, -774, 2134, 5745}},
      {"Nikon E4300",
       0,
       0, /* copied from Minolta DiMAGE Z2 */
       {11280, -3564, -1370, -4655, 12374, 2282, -1423, 2168, 5396}},
      {"Nikon E4500",
       0,
       0,
       {-5547, 11762, 2189, 5814, -558, 3342, -4924, 9840, 5949, 688, 9083,
        96}},
      {"Nikon E5000",
       0,
       0,
       {-5547, 11762, 2189, 5814, -558, 3342, -4924, 9840, 5949, 688, 9083,
        96}},
      {"Nikon E5400",
       0,
       0,
       {9349, -2987, -1001, -7919, 15766, 2266, -2098, 2680, 6839}},
      {"Nikon E5700",
       0,
       0,
       {-5368, 11478, 2368, 5537, -113, 3148, -4969, 10021, 5782, 778, 9028,
        211}},
      {"Nikon E8400",
       0,
       0,
       {7842, -2320, -992, -8154, 15718, 2599, -1098, 1342, 7560}},
      {"Nikon E8700",
       0,
       0,
       {8489, -2583, -1036, -8051, 15583, 2643, -1307, 1407, 7354}},
      {"Nikon E8800",
       0,
       0,
       {7971, -2314, -913, -8451, 15762, 2894, -1442, 1520, 7610}},
      {"Nikon COOLPIX A",
       0,
       0,
       {8198, -2239, -724, -4871, 12389, 2798, -1043, 2050, 7181}},
      {"Nikon COOLPIX P330",
       200,
       0,
       {10321, -3920, -931, -2750, 11146, 1824, -442, 1545, 5539}},
      {"Nikon COOLPIX P340",
       200,
       0,
       {10321, -3920, -931, -2750, 11146, 1824, -442, 1545, 5539}},
      {"Nikon COOLPIX P6000",
       0,
       0,
       {9698, -3367, -914, -4706, 12584, 2368, -837, 968, 5801}},
      {"Nikon COOLPIX P7000",
       0,
       0,
       {11432, -3679, -1111, -3169, 11239, 2202, -791, 1380, 4455}},
      {"Nikon COOLPIX P7100",
       0,
       0,
       {11053, -4269, -1024, -1976, 10182, 2088, -526, 1263, 4469}},
      {"Nikon COOLPIX P7700",
       200,
       0,
       {10321, -3920, -931, -2750, 11146, 1824, -442, 1545, 5539}},
      {"Nikon COOLPIX P7800",
       200,
       0,
       {10321, -3920, -931, -2750, 11146, 1824, -442, 1545, 5539}},
      {"Nikon 1 V3",
       0,
       0,
       {5958, -1559, -571, -4021, 11453, 2939, -634, 1548, 5087}},
      {"Nikon 1 J4",
       0,
       0,
       {5958, -1559, -571, -4021, 11453, 2939, -634, 1548, 5087}},
      {"Nikon 1 J5",
       0,
       0,
       {7520, -2518, -645, -3844, 12102, 1945, -913, 2249, 6835}},
      {"Nikon 1 S2",
       200,
       0,
       {6612, -1342, -618, -3338, 11055, 2623, -174, 1792, 5075}},
      {"Nikon 1 V2",
       0,
       0,
       {6588, -1305, -693, -3277, 10987, 2634, -355, 2016, 5106}},
      {"Nikon 1 J3",
       0,
       0,
       {6588, -1305, -693, -3277, 10987, 2634, -355, 2016, 5106}},
      {"Nikon 1 AW1",
       0,
       0,
       {6588, -1305, -693, -3277, 10987, 2634, -355, 2016, 5106}},
      {"Nikon 1 ",
       0,
       0, /* J1, J2, S1, V1 */
       {8994, -2667, -865, -4594, 12324, 2552, -699, 1786, 6260}},
      {"Olympus AIR A01",
       0,
       0,
       {8992, -3093, -639, -2563, 10721, 2122, -437, 1270, 5473}},
      {"Olympus C5050",
       0,
       0,
       {10508, -3124, -1273, -6079, 14294, 1901, -1653, 2306, 6237}},
      {"Olympus C5060",
       0,
       0,
       {10445, -3362, -1307, -7662, 15690, 2058, -1135, 1176, 7602}},
      {"Olympus C7070",
       0,
       0,
       {10252, -3531, -1095, -7114, 14850, 2436, -1451, 1723, 6365}},
      {"Olympus C70",
       0,
       0,
       {10793, -3791, -1146, -7498, 15177, 2488, -1390, 1577, 7321}},
      {"Olympus C80",
       0,
       0,
       {8606, -2509, -1014, -8238, 15714, 2703, -942, 979, 7760}},
      {"Olympus E-10",
       0,
       0xffc,
       {12745, -4500, -1416, -6062, 14542, 1580, -1934, 2256, 6603}},
      {"Olympus E-1",
       0,
       0,
       {11846, -4767, -945, -7027, 15878, 1089, -2699, 4122, 8311}},
      {"Olympus E-20",
       0,
       0xffc,
       {13173, -4732, -1499, -5807, 14036, 1895, -2045, 2452, 7142}},
      {"Olympus E-300",
       0,
       0,
       {7828, -1761, -348, -5788, 14071, 1830, -2853, 4518, 6557}},
      {"Olympus E-330",
       0,
       0,
       {8961, -2473, -1084, -7979, 15990, 2067, -2319, 3035, 8249}},
      {"Olympus E-30",
       0,
       0xfbc,
       {8144, -1861, -1111, -7763, 15894, 1929, -1865, 2542, 7607}},
      {"Olympus E-3",
       0,
       0xf99,
       {9487, -2875, -1115, -7533, 15606, 2010, -1618, 2100, 7389}},
      {"Olympus E-400",
       0,
       0,
       {6169, -1483, -21, -7107, 14761, 2536, -2904, 3580, 8568}},
      {"Olympus E-410",
       0,
       0xf6a,
       {8856, -2582, -1026, -7761, 15766, 2082, -2009, 2575, 7469}},
      {"Olympus E-420",
       0,
       0xfd7,
       {8746, -2425, -1095, -7594, 15612, 2073, -1780, 2309, 7416}},
      {"Olympus E-450",
       0,
       0xfd2,
       {8745, -2425, -1095, -7594, 15613, 2073, -1780, 2309, 7416}},
      {"Olympus E-500",
       0,
       0,
       {8136, -1968, -299, -5481, 13742, 1871, -2556, 4205, 6630}},
      {"Olympus E-510",
       0,
       0xf6a,
       {8785, -2529, -1033, -7639, 15624, 2112, -1783, 2300, 7817}},
      {"Olympus E-520",
       0,
       0xfd2,
       {8344, -2322, -1020, -7596, 15635, 2048, -1748, 2269, 7287}},
      {"Olympus E-5",
       0,
       0xeec,
       {11200, -3783, -1325, -4576, 12593, 2206, -695, 1742, 7504}},
      {"Olympus E-600",
       0,
       0xfaf,
       {8453, -2198, -1092, -7609, 15681, 2008, -1725, 2337, 7824}},
      {"Olympus E-620",
       0,
       0xfaf,
       {8453, -2198, -1092, -7609, 15681, 2008, -1725, 2337, 7824}},
      {"Olympus E-P1",
       0,
       0xffd,
       {8343, -2050, -1021, -7715, 15705, 2103, -1831, 2380, 8235}},
      {"Olympus E-P2",
       0,
       0xffd,
       {8343, -2050, -1021, -7715, 15705, 2103, -1831, 2380, 8235}},
      {"Olympus E-P3",
       0,
       0,
       {7575, -2159, -571, -3722, 11341, 2725, -1434, 2819, 6271}},
      {"Olympus E-P5",
       0,
       0,
       {8380, -2630, -639, -2887, 10725, 2496, -627, 1427, 5438}},
      {"Olympus E-PL1s",
       0,
       0,
       {11409, -3872, -1393, -4572, 12757, 2003, -709, 1810, 7415}},
      {"Olympus E-PL1",
       0,
       0,
       {11408, -4289, -1215, -4286, 12385, 2118, -387, 1467, 7787}},
      {"Olympus E-PL2",
       0,
       0xcf3,
       {15030, -5552, -1806, -3987, 12387, 1767, -592, 1670, 7023}},
      {"Olympus E-PL3",
       0,
       0,
       {7575, -2159, -571, -3722, 11341, 2725, -1434, 2819, 6271}},
      {"Olympus E-PL5",
       0,
       0xfcb,
       {8380, -2630, -639, -2887, 10725, 2496, -627, 1427, 5438}},
      {"Olympus E-PL6",
       0,
       0,
       {8380, -2630, -639, -2887, 10725, 2496, -627, 1427, 5438}},
      {"Olympus E-PL7",
       0,
       0,
       {9197, -3190, -659, -2606, 10830, 2039, -458, 1250, 5458}},
      {"Olympus E-PM1",
       0,
       0,
       {7575, -2159, -571, -3722, 11341, 2725, -1434, 2819, 6271}},
      {"Olympus E-PM2",
       0,
       0,
       {8380, -2630, -639, -2887, 10725, 2496, -627, 1427, 5438}},
      {"Olympus E-M10",
       0,
       0, /* also E-M10 Mark II */
       {8380, -2630, -639, -2887, 10725, 2496, -627, 1427, 5438}},
      {"Olympus E-M1",
       0,
       0,
       {7687, -1984, -606, -4327, 11928, 2721, -1381, 2339, 6452}},
      {"Olympus E-M5MarkII",
       0,
       0,
       {9422, -3258, -711, -2655, 10898, 2015, -512, 1354, 5512}},
      {"Olympus E-M5",
       0,
       0xfe1,
       {8380, -2630, -639, -2887, 10725, 2496, -627, 1427, 5438}},
      {"Olympus PEN-F",
       0,
       0,
       {9476, -3182, -765, -2613, 10958, 1893, -449, 1315, 5268}},
      {"Olympus SH-2",
       0,
       0,
       {10156, -3425, -1077, -2611, 11177, 1624, -385, 1592, 5080}},
      {"Olympus SP350",
       0,
       0,
       {12078, -4836, -1069, -6671, 14306, 2578, -786, 939, 7418}},
      {"Olympus SP3",
       0,
       0,
       {11766, -4445, -1067, -6901, 14421, 2707, -1029, 1217, 7572}},
      {"Olympus SP500UZ",
       0,
       0xfff,
       {9493, -3415, -666, -5211, 12334, 3260, -1548, 2262, 6482}},
      {"Olympus SP510UZ",
       0,
       0xffe,
       {10593, -3607, -1010, -5881, 13127, 3084, -1200, 1805, 6721}},
      {"Olympus SP550UZ",
       0,
       0xffe,
       {11597, -4006, -1049, -5432, 12799, 2957, -1029, 1750, 6516}},
      {"Olympus SP560UZ",
       0,
       0xff9,
       {10915, -3677, -982, -5587, 12986, 2911, -1168, 1968, 6223}},
      {"Olympus SP570UZ",
       0,
       0,
       {11522, -4044, -1146, -4736, 12172, 2904, -988, 1829, 6039}},
      {"Olympus STYLUS1",
       0,
       0,
       {8360, -2420, -880, -3928, 12353, 1739, -1381, 2416, 5173}},
      {"Olympus TG-4",
       0,
       0,
       {11426, -4159, -1126, -2066, 10678, 1593, -120, 1327, 4998}},
      {"Olympus XZ-10",
       0,
       0,
       {9777, -3483, -925, -2886, 11297, 1800, -602, 1663, 5134}},
      {"Olympus XZ-1",
       0,
       0,
       {10901, -4095, -1074, -1141, 9208, 2293, -62, 1417, 5158}},
      {"Olympus XZ-2",
       0,
       0,
       {9777, -3483, -925, -2886, 11297, 1800, -602, 1663, 5134}},
      {"OmniVision",
       0,
       0, /* DJC */
       {12782, -4059, -379, -478, 9066, 1413, 1340, 1513, 5176}},
      {"Pentax *ist DL2",
       0,
       0,
       {10504, -2438, -1189, -8603, 16207, 2531, -1022, 863, 12242}},
      {"Pentax *ist DL",
       0,
       0,
       {10829, -2838, -1115, -8339, 15817, 2696, -837, 680, 11939}},
      {"Pentax *ist DS2",
       0,
       0,
       {10504, -2438, -1189, -8603, 16207, 2531, -1022, 863, 12242}},
      {"Pentax *ist DS",
       0,
       0,
       {10371, -2333, -1206, -8688, 16231, 2602, -1230, 1116, 11282}},
      {"Pentax *ist D",
       0,
       0,
       {9651, -2059, -1189, -8881, 16512, 2487, -1460, 1345, 10687}},
      {"Pentax K10D",
       0,
       0,
       {9566, -2863, -803, -7170, 15172, 2112, -818, 803, 9705}},
      {"Pentax K1",
       0,
       0,
       {11095, -3157, -1324, -8377, 15834, 2720, -1108, 947, 11688}},
      {"Pentax K20D",
       0,
       0,
       {9427, -2714, -868, -7493, 16092, 1373, -2199, 3264, 7180}},
      {"Pentax K200D",
       0,
       0,
       {9186, -2678, -907, -8693, 16517, 2260, -1129, 1094, 8524}},
      {"Pentax K2000",
       0,
       0,
       {11057, -3604, -1155, -5152, 13046, 2329, -282, 375, 8104}},
      {"Pentax K-m",
       0,
       0,
       {11057, -3604, -1155, -5152, 13046, 2329, -282, 375, 8104}},
      {"Pentax K-x",
       0,
       0,
       {8843, -2837, -625, -5025, 12644, 2668, -411, 1234, 7410}},
      {"Pentax K-r",
       0,
       0,
       {9895, -3077, -850, -5304, 13035, 2521, -883, 1768, 6936}},
      {"Pentax K-1",
       0,
       0,
       {8566, -2746, -1201, -3612, 12204, 1550, -893, 1680, 6264}},
      {"Pentax K-30",
       0,
       0,
       {8710, -2632, -1167, -3995, 12301, 1881, -981, 1719, 6535}},
      {"Pentax K-3 II",
       0,
       0,
       {8626, -2607, -1155, -3995, 12301, 1881, -1039, 1822, 6925}},
      {"Pentax K-3",
       0,
       0,
       {7415, -2052, -721, -5186, 12788, 2682, -1446, 2157, 6773}},
      {"Pentax K-5 II",
       0,
       0,
       {8170, -2725, -639, -4440, 12017, 2744, -771, 1465, 6599}},
      {"Pentax K-5",
       0,
       0,
       {8713, -2833, -743, -4342, 11900, 2772, -722, 1543, 6247}},
      {"Pentax K-7",
       0,
       0,
       {9142, -2947, -678, -8648, 16967, 1663, -2224, 2898, 8615}},
      {"Pentax K-S1",
       0,
       0,
       {8512, -3211, -787, -4167, 11966, 2487, -638, 1288, 6054}},
      {"Pentax K-S2",
       0,
       0,
       {8662, -3280, -798, -3928, 11771, 2444, -586, 1232, 6054}},
      {"Pentax Q-S1",
       0,
       0,
       {12995, -5593, -1107, -1879, 10139, 2027, -64, 1233, 4919}},
      {"Pentax 645D",
       0,
       0x3e00,
       {10646, -3593, -1158, -3329, 11699, 1831, -667, 2874, 6287}},
      {"Panasonic DMC-CM1",
       15,
       0,
       {8770, -3194, -820, -2871, 11281, 1803, -513, 1552, 4434}},
      {"Panasonic DMC-FZ8",
       0,
       0xf7f,
       {8986, -2755, -802, -6341, 13575, 3077, -1476, 2144, 6379}},
      {"Panasonic DMC-FZ18",
       0,
       0,
       {9932, -3060, -935, -5809, 13331, 2753, -1267, 2155, 5575}},
      {"Panasonic DMC-FZ28",
       15,
       0xf96,
       {10109, -3488, -993, -5412, 12812, 2916, -1305, 2140, 5543}},
      {"Panasonic DMC-FZ330",
       15,
       0,
       {8378, -2798, -769, -3068, 11410, 1877, -538, 1792, 4623}},
      {"Panasonic DMC-FZ300",
       15,
       0,
       {8378, -2798, -769, -3068, 11410, 1877, -538, 1792, 4623}},
      {"Panasonic DMC-FZ30",
       0,
       0xf94,
       {10976, -4029, -1141, -7918, 15491, 2600, -1670, 2071, 8246}},
      {"Panasonic DMC-FZ3",
       15,
       0,
       {9938, -2780, -890, -4604, 12393, 2480, -1117, 2304, 4620}},
      {"Panasonic DMC-FZ4",
       15,
       0,
       {13639, -5535, -1371, -1698, 9633, 2430, 316, 1152, 4108}},
      {"Panasonic DMC-FZ50",
       0,
       0,
       {7906, -2709, -594, -6231, 13351, 3220, -1922, 2631, 6537}},
      {"Panasonic DMC-FZ7",
       15,
       0,
       {11532, -4324, -1066, -2375, 10847, 1749, -564, 1699, 4351}},
      {"Leica V-LUX1",
       0,
       0,
       {7906, -2709, -594, -6231, 13351, 3220, -1922, 2631, 6537}},
      {"Panasonic DMC-L10",
       15,
       0xf96,
       {8025, -1942, -1050, -7920, 15904, 2100, -2456, 3005, 7039}},
      {"Panasonic DMC-L1",
       0,
       0xf7f,
       {8054, -1885, -1025, -8349, 16367, 2040, -2805, 3542, 7629}},
      {"Leica DIGILUX 3",
       0,
       0xf7f,
       {8054, -1885, -1025, -8349, 16367, 2040, -2805, 3542, 7629}},
      {"Panasonic DMC-LC1",
       0,
       0,
       {11340, -4069, -1275, -7555, 15266, 2448, -2960, 3426, 7685}},
      {"Leica DIGILUX 2",
       0,
       0,
       {11340, -4069, -1275, -7555, 15266, 2448, -2960, 3426, 7685}},
      {"Panasonic DMC-LX100",
       15,
       0,
       {8844, -3538, -768, -3709, 11762, 2200, -698, 1792, 5220}},
      {"Leica D-LUX (Typ 109)",
       15,
       0,
       {8844, -3538, -768, -3709, 11762, 2200, -698, 1792, 5220}},
      {"Panasonic DMC-LF1",
       15,
       0,
       {9379, -3267, -816, -3227, 11560, 1881, -926, 1928, 5340}},
      {"Leica C (Typ 112)",
       15,
       0,
       {9379, -3267, -816, -3227, 11560, 1881, -926, 1928, 5340}},
      {"Panasonic DMC-LX1",
       0,
       0xf7f,
       {10704, -4187, -1230, -8314, 15952, 2501, -920, 945, 8927}},
      {"Leica D-LUX2",
       0,
       0xf7f,
       {10704, -4187, -1230, -8314, 15952, 2501, -920, 945, 8927}},
      {"Panasonic DMC-LX2",
       0,
       0,
       {8048, -2810, -623, -6450, 13519, 3272, -1700, 2146, 7049}},
      {"Leica D-LUX3",
       0,
       0,
       {8048, -2810, -623, -6450, 13519, 3272, -1700, 2146, 7049}},
      {"Panasonic DMC-LX3",
       15,
       0,
       {8128, -2668, -655, -6134, 13307, 3161, -1782, 2568, 6083}},
      {"Leica D-LUX 4",
       15,
       0,
       {8128, -2668, -655, -6134, 13307, 3161, -1782, 2568, 6083}},
      {"Panasonic DMC-LX5",
       15,
       0,
       {10909, -4295, -948, -1333, 9306, 2399, 22, 1738, 4582}},
      {"Leica D-LUX 5",
       15,
       0,
       {10909, -4295, -948, -1333, 9306, 2399, 22, 1738, 4582}},
      {"Panasonic DMC-LX7",
       15,
       0,
       {10148, -3743, -991, -2837, 11366, 1659, -701, 1893, 4899}},
      {"Leica D-LUX 6",
       15,
       0,
       {10148, -3743, -991, -2837, 11366, 1659, -701, 1893, 4899}},
      {"Panasonic DMC-FZ1000",
       15,
       0,
       {7830, -2696, -763, -3325, 11667, 1866, -641, 1712, 4824}},
      {"Leica V-LUX (Typ 114)",
       15,
       0,
       {7830, -2696, -763, -3325, 11667, 1866, -641, 1712, 4824}},
      {"Panasonic DMC-FZ100",
       15,
       0xfff,
       {16197, -6146, -1761, -2393, 10765, 1869, 366, 2238, 5248}},
      {"Leica V-LUX 2",
       15,
       0xfff,
       {16197, -6146, -1761, -2393, 10765, 1869, 366, 2238, 5248}},
      {"Panasonic DMC-FZ150",
       15,
       0xfff,
       {11904, -4541, -1189, -2355, 10899, 1662, -296, 1586, 4289}},
      {"Leica V-LUX 3",
       15,
       0xfff,
       {11904, -4541, -1189, -2355, 10899, 1662, -296, 1586, 4289}},
      {"Panasonic DMC-FZ200",
       15,
       0xfff,
       {8112, -2563, -740, -3730, 11784, 2197, -941, 2075, 4933}},
      {"Leica V-LUX 4",
       15,
       0xfff,
       {8112, -2563, -740, -3730, 11784, 2197, -941, 2075, 4933}},
      {"Panasonic DMC-FX150",
       15,
       0xfff,
       {9082, -2907, -925, -6119, 13377, 3058, -1797, 2641, 5609}},
      {"Panasonic DMC-G10",
       0,
       0,
       {10113, -3400, -1114, -4765, 12683, 2317, -377, 1437, 6710}},
      {"Panasonic DMC-G1",
       15,
       0xf94,
       {8199, -2065, -1056, -8124, 16156, 2033, -2458, 3022, 7220}},
      {"Panasonic DMC-G2",
       15,
       0xf3c,
       {10113, -3400, -1114, -4765, 12683, 2317, -377, 1437, 6710}},
      {"Panasonic DMC-G3",
       15,
       0xfff,
       {6763, -1919, -863, -3868, 11515, 2684, -1216, 2387, 5879}},
      {"Panasonic DMC-G5",
       15,
       0xfff,
       {7798, -2562, -740, -3879, 11584, 2613, -1055, 2248, 5434}},
      {"Panasonic DMC-G6",
       15,
       0xfff,
       {8294, -2891, -651, -3869, 11590, 2595, -1183, 2267, 5352}},
      {"Panasonic DMC-G7",
       15,
       0xfff,
       {7610, -2780, -576, -4614, 12195, 2733, -1375, 2393, 6490}},
      {"Panasonic DMC-GF1",
       15,
       0xf92,
       {7888, -1902, -1011, -8106, 16085, 2099, -2353, 2866, 7330}},
      {"Panasonic DMC-GF2",
       15,
       0xfff,
       {7888, -1902, -1011, -8106, 16085, 2099, -2353, 2866, 7330}},
      {"Panasonic DMC-GF3",
       15,
       0xfff,
       {9051, -2468, -1204, -5212, 13276, 2121, -1197, 2510, 6890}},
      {"Panasonic DMC-GF5",
       15,
       0xfff,
       {8228, -2945, -660, -3938, 11792, 2430, -1094, 2278, 5793}},
      {"Panasonic DMC-GF6",
       15,
       0,
       {8130, -2801, -946, -3520, 11289, 2552, -1314, 2511, 5791}},
      {"Panasonic DMC-GF7",
       15,
       0,
       {7610, -2780, -576, -4614, 12195, 2733, -1375, 2393, 6490}},
      {"Panasonic DMC-GF8",
       15,
       0,
       {7610, -2780, -576, -4614, 12195, 2733, -1375, 2393, 6490}},
      {"Panasonic DMC-GH1",
       15,
       0xf92,
       {6299, -1466, -532, -6535, 13852, 2969, -2331, 3112, 5984}},
      {"Panasonic DMC-GH2",
       15,
       0xf95,
       {7780, -2410, -806, -3913, 11724, 2484, -1018, 2390, 5298}},
      {"Panasonic DMC-GH3",
       15,
       0,
       {6559, -1752, -491, -3672, 11407, 2586, -962, 1875, 5130}},
      {"Panasonic DMC-GH4",
       15,
       0,
       {7122, -2108, -512, -3155, 11201, 2231, -541, 1423, 5045}},
      {"Panasonic DMC-GM1",
       15,
       0,
       {6770, -1895, -744, -5232, 13145, 2303, -1664, 2691, 5703}},
      {"Panasonic DMC-GM5",
       15,
       0,
       {8238, -3244, -679, -3921, 11814, 2384, -836, 2022, 5852}},
      {"Panasonic DMC-GX1",
       15,
       0,
       {6763, -1919, -863, -3868, 11515, 2684, -1216, 2387, 5879}},
      {"Panasonic DMC-GX7",
       15,
       0,
       {7610, -2780, -576, -4614, 12195, 2733, -1375, 2393, 6490}},
      {"Panasonic DMC-GX8",
       15,
       0,
       {7564, -2263, -606, -3148, 11239, 2177, -540, 1435, 4853}},
      {"Panasonic DMC-TZ1",
       15,
       0,
       {7790, -2736, -755, -3452, 11870, 1769, -628, 1647, 4898}},
      {"Panasonic DMC-ZS1",
       15,
       0,
       {7790, -2736, -755, -3452, 11870, 1769, -628, 1647, 4898}},
      {"Panasonic DMC-TZ6",
       15,
       0,
       {8607, -2822, -808, -3755, 11930, 2049, -820, 2060, 5224}},
      {"Panasonic DMC-ZS4",
       15,
       0,
       {8607, -2822, -808, -3755, 11930, 2049, -820, 2060, 5224}},
      {"Panasonic DMC-TZ7",
       15,
       0,
       {8802, -3135, -789, -3151, 11468, 1904, -550, 1745, 4810}},
      {"Panasonic DMC-ZS5",
       15,
       0,
       {8802, -3135, -789, -3151, 11468, 1904, -550, 1745, 4810}},
      {"Panasonic DMC-TZ8",
       15,
       0,
       {8550, -2908, -842, -3195, 11529, 1881, -338, 1603, 4631}},
      {"Panasonic DMC-ZS6",
       15,
       0,
       {8550, -2908, -842, -3195, 11529, 1881, -338, 1603, 4631}},
      {"Leica S (Typ 007)",
       0,
       0,
       {6063, -2234, -231, -5210, 13787, 1500, -1043, 2866, 6997}},
      {"Leica X",
       0,
       0, /* X and X-U, both (Typ 113) */
       {7712, -2059, -653, -3882, 11494, 2726, -710, 1332, 5958}},
      {"Leica Q (Typ 116)",
       0,
       0,
       {11865, -4523, -1441, -5423, 14458, 935, -1587, 2687, 4830}},
      {"Leica M (Typ 262)",
       0,
       0,
       {6653, -1486, -611, -4221, 13303, 929, -881, 2416, 7226}},
      {"Leica SL (Typ 601)",
       0,
       0,
       {11865, -4523, -1441, -5423, 14458, 935, -1587, 2687, 4830}},
      {"Phase One H 20",
       0,
       0, /* DJC */
       {1313, 1855, -109, -6715, 15908, 808, -327, 1840, 6020}},
      {"Phase One H 25",
       0,
       0,
       {2905, 732, -237, -8134, 16626, 1476, -3038, 4253, 7517}},
      {"Phase One P 2",
       0,
       0,
       {2905, 732, -237, -8134, 16626, 1476, -3038, 4253, 7517}},
      {"Phase One P 30",
       0,
       0,
       {4516, -245, -37, -7020, 14976, 2173, -3206, 4671, 7087}},
      {"Phase One P 45",
       0,
       0,
       {5053, -24, -117, -5684, 14076, 1702, -2619, 4492, 5849}},
      {"Phase One P40",
       0,
       0,
       {8035, 435, -962, -6001, 13872, 2320, -1159, 3065, 5434}},
      {"Phase One P65",
       0,
       0,
       {8035, 435, -962, -6001, 13872, 2320, -1159, 3065, 5434}},
      {"Photron BC2-HD",
       0,
       0, /* DJC */
       {14603, -4122, -528, -1810, 9794, 2017, -297, 2763, 5936}},
      {"Red One",
       704,
       (short int)0xffff, /* DJC */
       {21014, -7891, -2613, -3056, 12201, 856, -2203, 5125, 8042}},
      {"Ricoh GR II",
       0,
       0,
       {4630, -834, -423, -4977, 12805, 2417, -638, 1467, 6115}},
      {"Ricoh GR",
       0,
       0,
       {3708, -543, -160, -5381, 12254, 3556, -1471, 1929, 8234}},
      {"Samsung EX1",
       0,
       0x3e00,
       {8898, -2498, -994, -3144, 11328, 2066, -760, 1381, 4576}},
      {"Samsung EX2F",
       0,
       0x7ff,
       {10648, -3897, -1055, -2022, 10573, 1668, -492, 1611, 4742}},
      {"Samsung EK-GN120",
       0,
       0,
       {7557, -2522, -739, -4679, 12949, 1894, -840, 1777, 5311}},
      {"Samsung NX mini",
       0,
       0,
       {5222, -1196, -550, -6540, 14649, 2009, -1666, 2819, 5657}},
      {"Samsung NX3300",
       0,
       0,
       {8060, -2933, -761, -4504, 12890, 1762, -630, 1489, 5227}},
      {"Samsung NX3000",
       0,
       0,
       {8060, -2933, -761, -4504, 12890, 1762, -630, 1489, 5227}},
      {"Samsung NX30",
       0,
       0, /* NX30, NX300, NX300M */
       {7557, -2522, -739, -4679, 12949, 1894, -840, 1777, 5311}},
      {"Samsung NX2000",
       0,
       0,
       {7557, -2522, -739, -4679, 12949, 1894, -840, 1777, 5311}},
      {"Samsung NX2",
       0,
       0xfff, /* NX20, NX200, NX210 */
       {6933, -2268, -753, -4921, 13387, 1647, -803, 1641, 6096}},
      {"Samsung NX1000",
       0,
       0,
       {6933, -2268, -753, -4921, 13387, 1647, -803, 1641, 6096}},
      {"Samsung NX1100",
       0,
       0,
       {6933, -2268, -753, -4921, 13387, 1647, -803, 1641, 6096}},
      {"Samsung NX11",
       0,
       0,
       {10332, -3234, -1168, -6111, 14639, 1520, -1352, 2647, 8331}},
      {"Samsung NX10",
       0,
       0, /* also NX100 */
       {10332, -3234, -1168, -6111, 14639, 1520, -1352, 2647, 8331}},
      {"Samsung NX500",
       0,
       0,
       {10686, -4042, -1052, -3595, 13238, 276, -464, 1259, 5931}},
      {"Samsung NX5",
       0,
       0,
       {10332, -3234, -1168, -6111, 14639, 1520, -1352, 2647, 8331}},
      {"Samsung NX1",
       0,
       0,
       {10686, -4042, -1052, -3595, 13238, 276, -464, 1259, 5931}},
      {"Samsung WB2000",
       0,
       0xfff,
       {12093, -3557, -1155, -1000, 9534, 1733, -22, 1787, 4576}},
      {"Samsung GX-1",
       0,
       0,
       {10504, -2438, -1189, -8603, 16207, 2531, -1022, 863, 12242}},
      {"Samsung GX20",
       0,
       0, /* copied from Pentax K20D */
       {9427, -2714, -868, -7493, 16092, 1373, -2199, 3264, 7180}},
      {"Samsung S85",
       0,
       0, /* DJC */
       {11885, -3968, -1473, -4214, 12299, 1916, -835, 1655, 5549}},
      {"Sinar",
       0,
       0, /* DJC */
       {16442, -2956, -2422, -2877, 12128, 750, -1136, 6066, 4559}},
      {"Sony DSC-F828",
       0,
       0,
       {7924, -1910, -777, -8226, 15459, 2998, -1517, 2199, 6818, -7242, 11401,
        3481}},
      {"Sony DSC-R1",
       0,
       0,
       {8512, -2641, -694, -8042, 15670, 2526, -1821, 2117, 7414}},
      {"Sony DSC-V3",
       0,
       0,
       {7511, -2571, -692, -7894, 15088, 3060, -948, 1111, 8128}},
      {"Sony DSC-RX100M",
       0,
       0, /* M2, M3, and M4 */
       {6596, -2079, -562, -4782, 13016, 1933, -970, 1581, 5181}},
      {"Sony DSC-RX100",
       0,
       0,
       {8651, -2754, -1057, -3464, 12207, 1373, -568, 1398, 4434}},
      {"Sony DSC-RX10",
       0,
       0, /* also RX10M2 */
       {6679, -1825, -745, -5047, 13256, 1953, -1580, 2422, 5183}},
      {"Sony DSC-RX1RM2",
       0,
       0,
       {6629, -1900, -483, -4618, 12349, 2550, -622, 1381, 6514}},
      {"Sony DSC-RX1",
       0,
       0,
       {6344, -1612, -462, -4863, 12477, 2681, -865, 1786, 6899}},
      {"Sony DSLR-A100",
       0,
       0xfeb,
       {9437, -2811, -774, -8405, 16215, 2290, -710, 596, 7181}},
      {"Sony DSLR-A290",
       0,
       0,
       {6038, -1484, -579, -9145, 16746, 2512, -875, 746, 7218}},
      {"Sony DSLR-A2",
       0,
       0,
       {9847, -3091, -928, -8485, 16345, 2225, -715, 595, 7103}},
      {"Sony DSLR-A300",
       0,
       0,
       {9847, -3091, -928, -8485, 16345, 2225, -715, 595, 7103}},
      {"Sony DSLR-A330",
       0,
       0,
       {9847, -3091, -929, -8485, 16346, 2225, -714, 595, 7103}},
      {"Sony DSLR-A350",
       0,
       0xffc,
       {6038, -1484, -578, -9146, 16746, 2513, -875, 746, 7217}},
      {"Sony DSLR-A380",
       0,
       0,
       {6038, -1484, -579, -9145, 16746, 2512, -875, 746, 7218}},
      {"Sony DSLR-A390",
       0,
       0,
       {6038, -1484, -579, -9145, 16746, 2512, -875, 746, 7218}},
      {"Sony DSLR-A450",
       0,
       0xfeb,
       {4950, -580, -103, -5228, 12542, 3029, -709, 1435, 7371}},
      {"Sony DSLR-A580",
       0,
       0xfeb,
       {5932, -1492, -411, -4813, 12285, 2856, -741, 1524, 6739}},
      {"Sony DSLR-A500",
       0,
       0xfeb,
       {6046, -1127, -278, -5574, 13076, 2786, -691, 1419, 7625}},
      {"Sony DSLR-A5",
       0,
       0xfeb,
       {4950, -580, -103, -5228, 12542, 3029, -709, 1435, 7371}},
      {"Sony DSLR-A700",
       0,
       0,
       {5775, -805, -359, -8574, 16295, 2391, -1943, 2341, 7249}},
      {"Sony DSLR-A850",
       0,
       0,
       {5413, -1162, -365, -5665, 13098, 2866, -608, 1179, 8440}},
      {"Sony DSLR-A900",
       0,
       0,
       {5209, -1072, -397, -8845, 16120, 2919, -1618, 1803, 8654}},
      {"Sony ILCA-68",
       0,
       0,
       {6435, -1903, -536, -4722, 12449, 2550, -663, 1363, 6517}},
      {"Sony ILCA-77M2",
       0,
       0,
       {5991, -1732, -443, -4100, 11989, 2381, -704, 1467, 5992}},
      {"Sony ILCE-6300",
       0,
       0,
       {5973, -1695, -419, -3826, 11797, 2293, -639, 1398, 5789}},
      {"Sony ILCE-7M2",
       0,
       0,
       {5271, -712, -347, -6153, 13653, 2763, -1601, 2366, 7242}},
      {"Sony ILCE-7S",
       0,
       0, /* also ILCE-7SM2 */
       {5838, -1430, -246, -3497, 11477, 2297, -748, 1885, 5778}},
      {"Sony ILCE-7RM2",
       0,
       0,
       {6629, -1900, -483, -4618, 12349, 2550, -622, 1381, 6514}},
      {"Sony ILCE-7R",
       0,
       0,
       {4913, -541, -202, -6130, 13513, 2906, -1564, 2151, 7183}},
      {"Sony ILCE-7",
       0,
       0,
       {5271, -712, -347, -6153, 13653, 2763, -1601, 2366, 7242}},
      {"Sony ILCE",
       0,
       0, /* 3000, 5000, 5100, 6000, and QX1 */
       {5991, -1456, -455, -4764, 12135, 2980, -707, 1425, 6701}},
      {"Sony NEX-5N",
       0,
       0,
       {5991, -1456, -455, -4764, 12135, 2980, -707, 1425, 6701}},
      {"Sony NEX-5R",
       0,
       0,
       {6129, -1545, -418, -4930, 12490, 2743, -977, 1693, 6615}},
      {"Sony NEX-5T",
       0,
       0,
       {6129, -1545, -418, -4930, 12490, 2743, -977, 1693, 6615}},
      {"Sony NEX-3N",
       0,
       0,
       {6129, -1545, -418, -4930, 12490, 2743, -977, 1693, 6615}},
      {"Sony NEX-3",
       138,
       0, /* DJC */
       {6907, -1256, -645, -4940, 12621, 2320, -1710, 2581, 6230}},
      {"Sony NEX-5",
       116,
       0, /* DJC */
       {6807, -1350, -342, -4216, 11649, 2567, -1089, 2001, 6420}},
      {"Sony NEX-3",
       0,
       0, /* Adobe */
       {6549, -1550, -436, -4880, 12435, 2753, -854, 1868, 6976}},
      {"Sony NEX-5",
       0,
       0, /* Adobe */
       {6549, -1550, -436, -4880, 12435, 2753, -854, 1868, 6976}},
      {"Sony NEX-6",
       0,
       0,
       {6129, -1545, -418, -4930, 12490, 2743, -977, 1693, 6615}},
      {"Sony NEX-7",
       0,
       0,
       {5491, -1192, -363, -4951, 12342, 2948, -911, 1722, 7192}},
      {"Sony NEX",
       0,
       0, /* NEX-C3, NEX-F3 */
       {5991, -1456, -455, -4764, 12135, 2980, -707, 1425, 6701}},
      {"Sony SLT-A33",
       0,
       0,
       {6069, -1221, -366, -5221, 12779, 2734, -1024, 2066, 6834}},
      {"Sony SLT-A35",
       0,
       0,
       {5986, -1618, -415, -4557, 11820, 3120, -681, 1404, 6971}},
      {"Sony SLT-A37",
       0,
       0,
       {5991, -1456, -455, -4764, 12135, 2980, -707, 1425, 6701}},
      {"Sony SLT-A55",
       0,
       0,
       {5932, -1492, -411, -4813, 12285, 2856, -741, 1524, 6739}},
      {"Sony SLT-A57",
       0,
       0,
       {5991, -1456, -455, -4764, 12135, 2980, -707, 1425, 6701}},
      {"Sony SLT-A58",
       0,
       0,
       {5991, -1456, -455, -4764, 12135, 2980, -707, 1425, 6701}},
      {"Sony SLT-A65",
       0,
       0,
       {5491, -1192, -363, -4951, 12342, 2948, -911, 1722, 7192}},
      {"Sony SLT-A77",
       0,
       0,
       {5491, -1192, -363, -4951, 12342, 2948, -911, 1722, 7192}},
      {"Sony SLT-A99",
       0,
       0,
       {6344, -1612, -462, -4863, 12477, 2681, -865, 1786, 6899}},
  };
  double cam_xyz[4][3];
  char name[130];
  int i, j;

  sprintf(name, "%s %s", make, model);
  for (i = 0; i < sizeof table / sizeof *table; i++)
    if (!strncmp(name, table[i].prefix, strlen(table[i].prefix))) {
      if (table[i].black)
        black = (ushort)table[i].black;
      if (table[i].maximum)
        maximum = (ushort)table[i].maximum;
      if (table[i].trans[0]) {
        for (raw_color = j = 0; j < 12; j++)
          ((double *)cam_xyz)[j] = table[i].trans[j] / 10000.0;
        cam_xyz_coeff(rgb_cam, cam_xyz);
      }
      break;
    }
}

void CLASS simple_coeff(int index) {
  static const float table[][12] = {
      /* index 0 -- all Foveon cameras */
      {1.4032, -0.2231, -0.1016, -0.5263, 1.4816, 0.017, -0.0112, 0.0183,
       0.9113},
      /* index 1 -- Kodak DC20 and DC25 */
      {2.25, 0.75, -1.75, -0.25, -0.25, 0.75, 0.75, -0.25, -0.25, -1.75, 0.75,
       2.25},
      /* index 2 -- Logitech Fotoman Pixtura */
      {1.893, -0.418, -0.476, -0.495, 1.773, -0.278, -1.017, -0.655, 2.672},
      /* index 3 -- Nikon E880, E900, and E990 */
      {-1.936280, 1.800443, -1.448486, 2.584324, 1.405365, -0.524955, -0.289090,
       0.408680, -1.204965, 1.082304, 2.941367, -1.818705}};
  int i, c;

  for (raw_color = i = 0; i < 3; i++)
    FORCC rgb_cam[i][c] = table[index][i * colors + c];
}

short CLASS guess_byte_order(int words) {
  uchar test[4][2];
  int t = 2, msb;
  double diff, sum[2] = {0, 0};

  fread(test[0], 2, 2, ifp);
  for (words -= 2; words--;) {
    fread(test[t], 2, 1, ifp);
    for (msb = 0; msb < 2; msb++) {
      diff = (test[t ^ 2][msb] << 8 | test[t ^ 2][!msb]) -
             (test[t][msb] << 8 | test[t][!msb]);
      sum[msb] += diff * diff;
    }
    t = (t + 1) & 3;
  }
  return sum[0] < sum[1] ? 0x4d4d : 0x4949;
}

float CLASS find_green(int bps, int bite, int off0, int off1) {
  UINT64 bitbuf = 0;
  int vbits, col, i, c;
  ushort img[2][2064];
  double sum[] = {0, 0};

  FORC(2) {
    fseek(ifp, c ? off1 : off0, SEEK_SET);
    for (vbits = col = 0; col < width; col++) {
      for (vbits -= bps; vbits < 0; vbits += bite) {
        bitbuf <<= bite;
        for (i = 0; i < bite; i += 8)
          bitbuf |= (unsigned)(fgetc(ifp) << i);
      }
      img[c][col] = bitbuf << (64 - bps - vbits) >> (64 - bps);
    }
  }
  FORC(width - 1) {
    sum[c & 1] += ABS(img[0][c] - img[1][c + 1]);
    sum[~c & 1] += ABS(img[1][c] - img[0][c + 1]);
  }
  return 100 * log(sum[0] / sum[1]);
}

/*
Identify which camera created this file, and set global variables
accordingly.
*/
void CLASS identify() {
  static const short pana[][6] = {
      {3130, 1743, 4, 0, -6, 0},      {3130, 2055, 4, 0, -6, 0},
      {3130, 2319, 4, 0, -6, 0},      {3170, 2103, 18, 0, -42, 20},
      {3170, 2367, 18, 13, -42, -21}, {3177, 2367, 0, 0, -1, 0},
      {3304, 2458, 0, 0, -1, 0},      {3330, 2463, 9, 0, -5, 0},
      {3330, 2479, 9, 0, -17, 4},     {3370, 1899, 15, 0, -44, 20},
      {3370, 2235, 15, 0, -44, 20},   {3370, 2511, 15, 10, -44, -21},
      {3690, 2751, 3, 0, -8, -3},     {3710, 2751, 0, 0, -3, 0},
      {3724, 2450, 0, 0, 0, -2},      {3770, 2487, 17, 0, -44, 19},
      {3770, 2799, 17, 15, -44, -19}, {3880, 2170, 6, 0, -6, 0},
      {4060, 3018, 0, 0, 0, -2},      {4290, 2391, 3, 0, -8, -1},
      {4330, 2439, 17, 15, -44, -19}, {4508, 2962, 0, 0, -3, -4},
      {4508, 3330, 0, 0, -3, -6},
  };
  static const ushort canon[][11] = {
      {1944, 1416, 0, 0, 48, 0},
      {2144, 1560, 4, 8, 52, 2, 0, 0, 0, 25},
      {2224, 1456, 48, 6, 0, 2},
      {2376, 1728, 12, 6, 52, 2},
      {2672, 1968, 12, 6, 44, 2},
      {3152, 2068, 64, 12, 0, 0, 16},
      {3160, 2344, 44, 12, 4, 4},
      {3344, 2484, 4, 6, 52, 6},
      {3516, 2328, 42, 14, 0, 0},
      {3596, 2360, 74, 12, 0, 0},
      {3744, 2784, 52, 12, 8, 12},
      {3944, 2622, 30, 18, 6, 2},
      {3948, 2622, 42, 18, 0, 2},
      {3984, 2622, 76, 20, 0, 2, 14},
      {4104, 3048, 48, 12, 24, 12},
      {4116, 2178, 4, 2, 0, 0},
      {4152, 2772, 192, 12, 0, 0},
      {4160, 3124, 104, 11, 8, 65},
      {4176, 3062, 96, 17, 8, 0, 0, 16, 0, 7, 0x49},
      {4192, 3062, 96, 17, 24, 0, 0, 16, 0, 0, 0x49},
      {4312, 2876, 22, 18, 0, 2},
      {4352, 2874, 62, 18, 0, 0},
      {4476, 2954, 90, 34, 0, 0},
      {4480, 3348, 12, 10, 36, 12, 0, 0, 0, 18, 0x49},
      {4480, 3366, 80, 50, 0, 0},
      {4496, 3366, 80, 50, 12, 0},
      {4768, 3516, 96, 16, 0, 0, 0, 16},
      {4832, 3204, 62, 26, 0, 0},
      {4832, 3228, 62, 51, 0, 0},
      {5108, 3349, 98, 13, 0, 0},
      {5120, 3318, 142, 45, 62, 0},
      {5280, 3528, 72, 52, 0, 0},
      {5344, 3516, 142, 51, 0, 0},
      {5344, 3584, 126, 100, 0, 2},
      {5360, 3516, 158, 51, 0, 0},
      {5568, 3708, 72, 38, 0, 0},
      {5632, 3710, 96, 17, 0, 0, 0, 16, 0, 0, 0x49},
      {5712, 3774, 62, 20, 10, 2},
      {5792, 3804, 158, 51, 0, 0},
      {5920, 3950, 122, 80, 2, 0},
      {6096, 4056, 72, 34, 0, 0},
      {6288, 4056, 264, 34, 0, 0},
      {8896, 5920, 160, 64, 0, 0},
  };
  static const struct {
    ushort id;
    char model[20];
  } unique[] =
      {
          {0x168, "EOS 10D"},          {0x001, "EOS-1D"},
          {0x175, "EOS 20D"},          {0x174, "EOS-1D Mark II"},
          {0x234, "EOS 30D"},          {0x232, "EOS-1D Mark II N"},
          {0x190, "EOS 40D"},          {0x169, "EOS-1D Mark III"},
          {0x261, "EOS 50D"},          {0x281, "EOS-1D Mark IV"},
          {0x287, "EOS 60D"},          {0x167, "EOS-1DS"},
          {0x325, "EOS 70D"},          {0x350, "EOS 80D"},
          {0x328, "EOS-1D X Mark II"}, {0x170, "EOS 300D"},
          {0x188, "EOS-1Ds Mark II"},  {0x176, "EOS 450D"},
          {0x215, "EOS-1Ds Mark III"}, {0x189, "EOS 350D"},
          {0x324, "EOS-1D C"},         {0x236, "EOS 400D"},
          {0x269, "EOS-1D X"},         {0x252, "EOS 500D"},
          {0x213, "EOS 5D"},           {0x270, "EOS 550D"},
          {0x218, "EOS 5D Mark II"},   {0x286, "EOS 600D"},
          {0x285, "EOS 5D Mark III"},  {0x301, "EOS 650D"},
          {0x302, "EOS 6D"},           {0x326, "EOS 700D"},
          {0x250, "EOS 7D"},           {0x393, "EOS 750D"},
          {0x289, "EOS 7D Mark II"},   {0x347, "EOS 760D"},
          {0x254, "EOS 1000D"},        {0x288, "EOS 1100D"},
          {0x327, "EOS 1200D"},        {0x382, "Canon EOS 5DS"},
          {0x404, "EOS 1300D"},        {0x401, "Canon EOS 5DS R"},
          {0x346, "EOS 100D"},
      },
    sonique[] = {
        {0x002, "DSC-R1"},      {0x100, "DSLR-A100"},  {0x101, "DSLR-A900"},
        {0x102, "DSLR-A700"},   {0x103, "DSLR-A200"},  {0x104, "DSLR-A350"},
        {0x105, "DSLR-A300"},   {0x108, "DSLR-A330"},  {0x109, "DSLR-A230"},
        {0x10a, "DSLR-A290"},   {0x10d, "DSLR-A850"},  {0x111, "DSLR-A550"},
        {0x112, "DSLR-A500"},   {0x113, "DSLR-A450"},  {0x116, "NEX-5"},
        {0x117, "NEX-3"},       {0x118, "SLT-A33"},    {0x119, "SLT-A55V"},
        {0x11a, "DSLR-A560"},   {0x11b, "DSLR-A580"},  {0x11c, "NEX-C3"},
        {0x11d, "SLT-A35"},     {0x11e, "SLT-A65V"},   {0x11f, "SLT-A77V"},
        {0x120, "NEX-5N"},      {0x121, "NEX-7"},      {0x123, "SLT-A37"},
        {0x124, "SLT-A57"},     {0x125, "NEX-F3"},     {0x126, "SLT-A99V"},
        {0x127, "NEX-6"},       {0x128, "NEX-5R"},     {0x129, "DSC-RX100"},
        {0x12a, "DSC-RX1"},     {0x12e, "ILCE-3000"},  {0x12f, "SLT-A58"},
        {0x131, "NEX-3N"},      {0x132, "ILCE-7"},     {0x133, "NEX-5T"},
        {0x134, "DSC-RX100M2"}, {0x135, "DSC-RX10"},   {0x136, "DSC-RX1R"},
        {0x137, "ILCE-7R"},     {0x138, "ILCE-6000"},  {0x139, "ILCE-5000"},
        {0x13d, "DSC-RX100M3"}, {0x13e, "ILCE-7S"},    {0x13f, "ILCA-77M2"},
        {0x153, "ILCE-5100"},   {0x154, "ILCE-7M2"},   {0x155, "DSC-RX100M4"},
        {0x156, "DSC-RX10M2"},  {0x158, "DSC-RX1RM2"}, {0x15a, "ILCE-QX1"},
        {0x15b, "ILCE-7RM2"},   {0x15e, "ILCE-7SM2"},  {0x161, "ILCA-68"},
        {0x165, "ILCE-6300"},
    };
  static const struct {
    unsigned fsize;
    ushort rw, rh;
    uchar lm, tm, rm, bm, lf, cf, max, flags;
    char make[10], model[20];
    ushort offset;
  } table[] = {
      {786432, 1024, 768, 0, 0, 0, 0, 0, 0x94, 0, 0, "AVT", "F-080C"},
      {1447680, 1392, 1040, 0, 0, 0, 0, 0, 0x94, 0, 0, "AVT", "F-145C"},
      {1920000, 1600, 1200, 0, 0, 0, 0, 0, 0x94, 0, 0, "AVT", "F-201C"},
      {5067304, 2588, 1958, 0, 0, 0, 0, 0, 0x94, 0, 0, "AVT", "F-510C"},
      {5067316, 2588, 1958, 0, 0, 0, 0, 0, 0x94, 0, 0, "AVT", "F-510C", 12},
      {10134608, 2588, 1958, 0, 0, 0, 0, 9, 0x94, 0, 0, "AVT", "F-510C"},
      {10134620, 2588, 1958, 0, 0, 0, 0, 9, 0x94, 0, 0, "AVT", "F-510C", 12},
      {16157136, 3272, 2469, 0, 0, 0, 0, 9, 0x94, 0, 0, "AVT", "F-810C"},
      {15980544, 3264, 2448, 0, 0, 0, 0, 8, 0x61, 0, 1, "AgfaPhoto", "DC-833m"},
      {9631728, 2532, 1902, 0, 0, 0, 0, 96, 0x61, 0, 0, "Alcatel", "5035D"},
      {2868726, 1384, 1036, 0, 0, 0, 0, 64, 0x49, 0, 8, "Baumer", "TXG14",
       1078},
      {5298000, 2400, 1766, 12, 12, 44, 2, 40, 0x94, 0, 2, "Canon",
       "PowerShot SD300"},
      {6553440, 2664, 1968, 4, 4, 44, 4, 40, 0x94, 0, 2, "Canon",
       "PowerShot A460"},
      {6573120, 2672, 1968, 12, 8, 44, 0, 40, 0x94, 0, 2, "Canon",
       "PowerShot A610"},
      {6653280, 2672, 1992, 10, 6, 42, 2, 40, 0x94, 0, 2, "Canon",
       "PowerShot A530"},
      {7710960, 2888, 2136, 44, 8, 4, 0, 40, 0x94, 0, 2, "Canon",
       "PowerShot S3 IS"},
      {9219600, 3152, 2340, 36, 12, 4, 0, 40, 0x94, 0, 2, "Canon",
       "PowerShot A620"},
      {9243240, 3152, 2346, 12, 7, 44, 13, 40, 0x49, 0, 2, "Canon",
       "PowerShot A470"},
      {10341600, 3336, 2480, 6, 5, 32, 3, 40, 0x94, 0, 2, "Canon",
       "PowerShot A720 IS"},
      {10383120, 3344, 2484, 12, 6, 44, 6, 40, 0x94, 0, 2, "Canon",
       "PowerShot A630"},
      {12945240, 3736, 2772, 12, 6, 52, 6, 40, 0x94, 0, 2, "Canon",
       "PowerShot A640"},
      {15636240, 4104, 3048, 48, 12, 24, 12, 40, 0x94, 0, 2, "Canon",
       "PowerShot A650"},
      {15467760, 3720, 2772, 6, 12, 30, 0, 40, 0x94, 0, 2, "Canon",
       "PowerShot SX110 IS"},
      {15534576, 3728, 2778, 12, 9, 44, 9, 40, 0x94, 0, 2, "Canon",
       "PowerShot SX120 IS"},
      {18653760, 4080, 3048, 24, 12, 24, 12, 40, 0x94, 0, 2, "Canon",
       "PowerShot SX20 IS"},
      {19131120, 4168, 3060, 92, 16, 4, 1, 40, 0x94, 0, 2, "Canon",
       "PowerShot SX220 HS"},
      {21936096, 4464, 3276, 25, 10, 73, 12, 40, 0x16, 0, 2, "Canon",
       "PowerShot SX30 IS"},
      {24724224, 4704, 3504, 8, 16, 56, 8, 40, 0x94, 0, 2, "Canon",
       "PowerShot A3300 IS"},
      {30858240, 5248, 3920, 8, 16, 56, 16, 40, 0x94, 0, 2, "Canon",
       "IXUS 160"},
      {1976352, 1632, 1211, 0, 2, 0, 1, 0, 0x94, 0, 1, "Casio", "QV-2000UX"},
      {3217760, 2080, 1547, 0, 0, 10, 1, 0, 0x94, 0, 1, "Casio", "QV-3*00EX"},
      {6218368, 2585, 1924, 0, 0, 9, 0, 0, 0x94, 0, 1, "Casio", "QV-5700"},
      {7816704, 2867, 2181, 0, 0, 34, 36, 0, 0x16, 0, 1, "Casio", "EX-Z60"},
      {2937856, 1621, 1208, 0, 0, 1, 0, 0, 0x94, 7, 13, "Casio", "EX-S20"},
      {4948608, 2090, 1578, 0, 0, 32, 34, 0, 0x94, 7, 1, "Casio", "EX-S100"},
      {6054400, 2346, 1720, 2, 0, 32, 0, 0, 0x94, 7, 1, "Casio", "QV-R41"},
      {7426656, 2568, 1928, 0, 0, 0, 0, 0, 0x94, 0, 1, "Casio", "EX-P505"},
      {7530816, 2602, 1929, 0, 0, 22, 0, 0, 0x94, 7, 1, "Casio", "QV-R51"},
      {7542528, 2602, 1932, 0, 0, 32, 0, 0, 0x94, 7, 1, "Casio", "EX-Z50"},
      {7562048, 2602, 1937, 0, 0, 25, 0, 0, 0x16, 7, 1, "Casio", "EX-Z500"},
      {7753344, 2602, 1986, 0, 0, 32, 26, 0, 0x94, 7, 1, "Casio", "EX-Z55"},
      {9313536, 2858, 2172, 0, 0, 14, 30, 0, 0x94, 7, 1, "Casio", "EX-P600"},
      {10834368, 3114, 2319, 0, 0, 27, 0, 0, 0x94, 0, 1, "Casio", "EX-Z750"},
      {10843712, 3114, 2321, 0, 0, 25, 0, 0, 0x94, 0, 1, "Casio", "EX-Z75"},
      {10979200, 3114, 2350, 0, 0, 32, 32, 0, 0x94, 7, 1, "Casio", "EX-P700"},
      {12310144, 3285, 2498, 0, 0, 6, 30, 0, 0x94, 0, 1, "Casio", "EX-Z850"},
      {12489984, 3328, 2502, 0, 0, 47, 35, 0, 0x94, 0, 1, "Casio", "EX-Z8"},
      {15499264, 3754, 2752, 0, 0, 82, 0, 0, 0x94, 0, 1, "Casio", "EX-Z1050"},
      {18702336, 4096, 3044, 0, 0, 24, 0, 80, 0x94, 7, 1, "Casio", "EX-ZR100"},
      {7684000, 2260, 1700, 0, 0, 0, 0, 13, 0x94, 0, 1, "Casio", "QV-4000"},
      {787456, 1024, 769, 0, 1, 0, 0, 0, 0x49, 0, 0, "Creative", "PC-CAM 600"},
      {28829184, 4384, 3288, 0, 0, 0, 0, 36, 0x61, 0, 0, "DJI"},
      {15151104, 4608, 3288, 0, 0, 0, 0, 0, 0x94, 0, 0, "Matrix"},
      {3840000, 1600, 1200, 0, 0, 0, 0, 65, 0x49, 0, 0, "Foculus", "531C"},
      {307200, 640, 480, 0, 0, 0, 0, 0, 0x94, 0, 0, "Generic"},
      {62464, 256, 244, 1, 1, 6, 1, 0, 0x8d, 0, 0, "Kodak", "DC20"},
      {124928, 512, 244, 1, 1, 10, 1, 0, 0x8d, 0, 0, "Kodak", "DC20"},
      {1652736, 1536, 1076, 0, 52, 0, 0, 0, 0x61, 0, 0, "Kodak", "DCS200"},
      {4159302, 2338, 1779, 1, 33, 1, 2, 0, 0x94, 0, 0, "Kodak", "C330"},
      {4162462, 2338, 1779, 1, 33, 1, 2, 0, 0x94, 0, 0, "Kodak", "C330", 3160},
      {2247168, 1232, 912, 0, 0, 16, 0, 0, 0x00, 0, 0, "Kodak", "C330"},
      {3370752, 1232, 912, 0, 0, 16, 0, 0, 0x00, 0, 0, "Kodak", "C330"},
      {6163328, 2864, 2152, 0, 0, 0, 0, 0, 0x94, 0, 0, "Kodak", "C603"},
      {6166488, 2864, 2152, 0, 0, 0, 0, 0, 0x94, 0, 0, "Kodak", "C603", 3160},
      {460800, 640, 480, 0, 0, 0, 0, 0, 0x00, 0, 0, "Kodak", "C603"},
      {9116448, 2848, 2134, 0, 0, 0, 0, 0, 0x00, 0, 0, "Kodak", "C603"},
      {12241200, 4040, 3030, 2, 0, 0, 13, 0, 0x49, 0, 0, "Kodak", "12MP"},
      {12272756, 4040, 3030, 2, 0, 0, 13, 0, 0x49, 0, 0, "Kodak", "12MP",
       31556},
      {18000000, 4000, 3000, 0, 0, 0, 0, 0, 0x00, 0, 0, "Kodak", "12MP"},
      {614400, 640, 480, 0, 3, 0, 0, 64, 0x94, 0, 0, "Kodak", "KAI-0340"},
      {15360000, 3200, 2400, 0, 0, 0, 0, 96, 0x16, 0, 0, "Lenovo", "A820"},
      {3884928, 1608, 1207, 0, 0, 0, 0, 96, 0x16, 0, 0, "Micron", "2010", 3212},
      {1138688, 1534, 986, 0, 0, 0, 0, 0, 0x61, 0, 0, "Minolta", "RD175", 513},
      {1581060, 1305, 969, 0, 0, 18, 6, 6, 0x1e, 4, 1, "Nikon", "E900"},
      {2465792, 1638, 1204, 0, 0, 22, 1, 6, 0x4b, 5, 1, "Nikon", "E950"},
      {2940928, 1616, 1213, 0, 0, 0, 7, 30, 0x94, 0, 1, "Nikon", "E2100"},
      {4771840, 2064, 1541, 0, 0, 0, 1, 6, 0xe1, 0, 1, "Nikon", "E990"},
      {4775936, 2064, 1542, 0, 0, 0, 0, 30, 0x94, 0, 1, "Nikon", "E3700"},
      {5865472, 2288, 1709, 0, 0, 0, 1, 6, 0xb4, 0, 1, "Nikon", "E4500"},
      {5869568, 2288, 1710, 0, 0, 0, 0, 6, 0x16, 0, 1, "Nikon", "E4300"},
      {7438336, 2576, 1925, 0, 0, 0, 1, 6, 0xb4, 0, 1, "Nikon", "E5000"},
      {8998912, 2832, 2118, 0, 0, 0, 0, 30, 0x94, 7, 1, "Nikon", "COOLPIX S6"},
      {5939200, 2304, 1718, 0, 0, 0, 0, 30, 0x16, 0, 0, "Olympus", "C770UZ"},
      {3178560, 2064, 1540, 0, 0, 0, 0, 0, 0x94, 0, 1, "Pentax", "Optio S"},
      {4841984, 2090, 1544, 0, 0, 22, 0, 0, 0x94, 7, 1, "Pentax", "Optio S"},
      {6114240, 2346, 1737, 0, 0, 22, 0, 0, 0x94, 7, 1, "Pentax", "Optio S4"},
      {10702848, 3072, 2322, 0, 0, 0, 21, 30, 0x94, 0, 1, "Pentax",
       "Optio 750Z"},
      {4147200, 1920, 1080, 0, 0, 0, 0, 0, 0x49, 0, 0, "Photron", "BC2-HD"},
      {4151666, 1920, 1080, 0, 0, 0, 0, 0, 0x49, 0, 0, "Photron", "BC2-HD", 8},
      {13248000, 2208, 3000, 0, 0, 0, 0, 13, 0x61, 0, 0, "Pixelink", "A782"},
      {6291456, 2048, 1536, 0, 0, 0, 0, 96, 0x61, 0, 0, "RoverShot", "3320AF"},
      {311696, 644, 484, 0, 0, 0, 0, 0, 0x16, 0, 8, "ST Micro", "STV680 VGA"},
      {16098048, 3288, 2448, 0, 0, 24, 0, 9, 0x94, 0, 1, "Samsung", "S85"},
      {16215552, 3312, 2448, 0, 0, 48, 0, 9, 0x94, 0, 1, "Samsung", "S85"},
      {20487168, 3648, 2808, 0, 0, 0, 0, 13, 0x94, 5, 1, "Samsung", "WB550"},
      {24000000, 4000, 3000, 0, 0, 0, 0, 13, 0x94, 5, 1, "Samsung", "WB550"},
      {12582980, 3072, 2048, 0, 0, 0, 0, 33, 0x61, 0, 0, "Sinar", "", 68},
      {33292868, 4080, 4080, 0, 0, 0, 0, 33, 0x61, 0, 0, "Sinar", "", 68},
      {44390468, 4080, 5440, 0, 0, 0, 0, 33, 0x61, 0, 0, "Sinar", "", 68},
      {1409024, 1376, 1024, 0, 0, 1, 0, 0, 0x49, 0, 0, "Sony", "XCD-SX910CR"},
      {2818048, 1376, 1024, 0, 0, 1, 0, 97, 0x49, 0, 0, "Sony", "XCD-SX910CR"},
  };
  static const char *corp[] = {
      "AgfaPhoto", "Canon",    "Casio", "Epson",  "Fujifilm",  "Mamiya",
      "Minolta",   "Motorola", "Kodak", "Konica", "Leica",     "Nikon",
      "Nokia",     "Olympus",  "Ricoh", "Pentax", "Phase One", "Samsung",
      "Sigma",     "Sinar",    "Sony"};
  char head[32], *cp;
  int hlen, flen, fsize, zero_fsize = 1, i, c;
  struct jhead jh;

  tiff_flip = flip = filters = UINT_MAX; /* unknown */
  raw_height = raw_width = fuji_width = fuji_layout = cr2_slice[0] = 0;
  maximum = height = width = top_margin = left_margin = 0;
  cdesc[0] = desc[0] = artist[0] = make[0] = model[0] = model2[0] = 0;
  iso_speed = shutter = aperture = focal_len = unique_id = 0;
  tiff_nifds = 0;
  memset(tiff_ifd, 0, sizeof tiff_ifd);
  memset(gpsdata, 0, sizeof gpsdata);
  memset(cblack, 0, sizeof cblack);
  memset(white, 0, sizeof white);
  memset(mask, 0, sizeof mask);
  thumb_offset = thumb_length = thumb_width = thumb_height = 0;
  load_raw = thumb_load_raw = 0;
  write_thumb = &CLASS jpeg_thumb;
  data_offset = meta_offset = meta_length = tiff_bps = tiff_compress = 0;
  kodak_cbpp = zero_after_ff = dng_version = load_flags = 0;
  timestamp = shot_order = tiff_samples = black = is_foveon = 0;
  mix_green = profile_length = data_error = zero_is_bad = 0;
  pixel_aspect = is_raw = raw_color = 1;
  tile_width = tile_length = 0;
  for (i = 0; i < 4; i++) {
    cam_mul[i] = i == 1;
    pre_mul[i] = i < 3;
    FORC3 cmatrix[c][i] = 0;
    FORC3 rgb_cam[c][i] = c == i;
  }
  colors = 3;
  for (i = 0; i < 0x10000; i++)
    curve[i] = i;

  order = get2();
  hlen = get4();
  fseek(ifp, 0, SEEK_SET);
  fread(head, 1, 32, ifp);
  fseek(ifp, 0, SEEK_END);
  flen = fsize = ftell(ifp);
  if ((cp = (char *)memmem(head, 32, (char *)"MMMM", 4)) ||
      (cp = (char *)memmem(head, 32, (char *)"IIII", 4))) {
    parse_phase_one(cp - head);
    if (cp - head && parse_tiff(0))
      apply_tiff();
  } else if (order == 0x4949 || order == 0x4d4d) {
    if (!memcmp(head + 6, "HEAPCCDR", 8)) {
      data_offset = hlen;
      parse_ciff(hlen, flen - hlen, 0);
      load_raw = &CLASS canon_load_raw;
    } else if (parse_tiff(0))
      apply_tiff();
  } else if (!memcmp(head, "\xff\xd8\xff\xe1", 4) &&
             !memcmp(head + 6, "Exif", 4)) {
    fseek(ifp, 4, SEEK_SET);
    data_offset = 4 + get2();
    fseek(ifp, data_offset, SEEK_SET);
    if (fgetc(ifp) != 0xff)
      parse_tiff(12);
    thumb_offset = 0;
  } else if (!memcmp(head + 25, "ARECOYK", 7)) {
    strcpy(make, "Contax");
    strcpy(model, "N Digital");
    fseek(ifp, 33, SEEK_SET);
    get_timestamp(1);
    fseek(ifp, 60, SEEK_SET);
    FORC4 cam_mul[c ^ (c >> 1)] = get4();
  } else if (!strcmp(head, "PXN")) {
    strcpy(make, "Logitech");
    strcpy(model, "Fotoman Pixtura");
  } else if (!strcmp(head, "qktk")) {
    strcpy(make, "Apple");
    strcpy(model, "QuickTake 100");
    load_raw = &CLASS quicktake_100_load_raw;
  } else if (!strcmp(head, "qktn")) {
    strcpy(make, "Apple");
    strcpy(model, "QuickTake 150");
    load_raw = &CLASS kodak_radc_load_raw;
  } else if (!memcmp(head, "FUJIFILM", 8)) {
    fseek(ifp, 84, SEEK_SET);
    thumb_offset = get4();
    thumb_length = get4();
    fseek(ifp, 92, SEEK_SET);
    parse_fuji(get4());
    if (thumb_offset > 120) {
      fseek(ifp, 120, SEEK_SET);
      is_raw += (i = get4()) && (bool)1;
      if (is_raw == 2 && shot_select)
        parse_fuji(i);
    }
    load_raw = &CLASS unpacked_load_raw;
    fseek(ifp, 100 + 28 * (shot_select > 0), SEEK_SET);
    parse_tiff(data_offset = get4());
    parse_tiff(thumb_offset + 12);
    apply_tiff();
  } else if (!memcmp(head, "RIFF", 4)) {
    fseek(ifp, 0, SEEK_SET);
    parse_riff();
  } else if (!memcmp(head + 4, "ftypqt   ", 9)) {
    fseek(ifp, 0, SEEK_SET);
    parse_qt(fsize);
    is_raw = 0;
  } else if (!memcmp(head, "\0\001\0\001\0@", 6)) {
    fseek(ifp, 6, SEEK_SET);
    fread(make, 1, 8, ifp);
    fread(model, 1, 8, ifp);
    fread(model2, 1, 16, ifp);
    data_offset = get2();
    get2();
    raw_width = get2();
    raw_height = get2();
    load_raw = &CLASS nokia_load_raw;
    filters = 0x61616161;
  } else if (!memcmp(head, "NOKIARAW", 8)) {
    strcpy(make, "NOKIA");
    order = 0x4949;
    fseek(ifp, 300, SEEK_SET);
    data_offset = get4();
    i = get4();
    width = get2();
    height = get2();
    switch (tiff_bps = i * 8 / (width * height)) {
      case 8:
        load_raw = &CLASS eight_bit_load_raw;
        break;
      case 10:
        load_raw = &CLASS nokia_load_raw;
    }
    raw_height = height + (top_margin = i / (width * tiff_bps / 8) - height);
    mask[0][3] = 1;
    filters = 0x61616161;
  } else if (!memcmp(head, "ARRI", 4)) {
    order = 0x4949;
    fseek(ifp, 20, SEEK_SET);
    width = get4();
    height = get4();
    strcpy(make, "ARRI");
    fseek(ifp, 668, SEEK_SET);
    fread(model, 1, 64, ifp);
    data_offset = 4096;
    load_raw = &CLASS packed_load_raw;
    load_flags = 88;
    filters = 0x61616161;
  } else if (!memcmp(head, "XPDS", 4)) {
    order = 0x4949;
    fseek(ifp, 0x800, SEEK_SET);
    fread(make, 1, 41, ifp);
    raw_height = get2();
    raw_width = get2();
    fseek(ifp, 56, SEEK_CUR);
    fread(model, 1, 30, ifp);
    data_offset = 0x10000;
    load_raw = &CLASS canon_rmf_load_raw;
    gamma_curve(0, 12.25, 1, 1023);
  } else if (!memcmp(head + 4, "RED1", 4)) {
    strcpy(make, "Red");
    strcpy(model, "One");
    parse_redcine();
    load_raw = &CLASS redcine_load_raw;
    gamma_curve(1 / 2.4, 12.92, 1, 4095);
    filters = 0x49494949;
  } else if (!memcmp(head, "DSC-Image", 9))
    parse_rollei();
  else if (!memcmp(head, "PWAD", 4))
    parse_sinar_ia();
  else if (!memcmp(head, "\0MRM", 4))
    parse_minolta(0);
  else if (!memcmp(head, "FOVb", 4))
    parse_foveon();
  else if (!memcmp(head, "CI", 2))
    parse_cine();
  if (make[0] == 0)
    for (zero_fsize = i = 0; i < sizeof table / sizeof *table; i++)
      if (fsize == table[i].fsize) {
        strcpy(make, table[i].make);
        strcpy(model, table[i].model);
        flip = table[i].flags >> 2;
        zero_is_bad = table[i].flags & 2;
        if (table[i].flags & 1)
          parse_external_jpeg();
        data_offset = table[i].offset;
        raw_width = table[i].rw;
        raw_height = table[i].rh;
        left_margin = table[i].lm;
        top_margin = table[i].tm;
        width = raw_width - left_margin - table[i].rm;
        height = raw_height - top_margin - table[i].bm;
        filters = 0x1010101 * table[i].cf;
        colors = 4 - !((filters & filters >> 1) & 0x5555);
        load_flags = table[i].lf;
        switch (tiff_bps =
                    (fsize - data_offset) * 8 / (raw_width * raw_height)) {
          case 6:
            load_raw = &CLASS minolta_rd175_load_raw;
            break;
          case 8:
            load_raw = &CLASS eight_bit_load_raw;
            break;
          case 10:
          case 12:
            load_flags |= 128;
            load_raw = &CLASS packed_load_raw;
            break;
          case 16:
            order = 0x4949 | 0x404 * (load_flags & 1);
            tiff_bps -= load_flags >> 4;
            tiff_bps -= load_flags = load_flags >> 1 & 7;
            load_raw = &CLASS unpacked_load_raw;
        }
        maximum = (1 << tiff_bps) - (1 << table[i].max);
      }
  if (zero_fsize)
    fsize = 0;
  if (make[0] == 0)
    parse_smal(0, flen);
  if (make[0] == 0) {
    parse_jpeg(0);
    if (!(strncmp(model, "ov", 2) && strncmp(model, "RP_OV", 5)) &&
        !fseek(ifp, -6404096, SEEK_END) && fread(head, 1, 32, ifp) &&
        !strcmp(head, "BRCMn")) {
      strcpy(make, "OmniVision");
      data_offset = ftell(ifp) + 0x8000 - 32;
      width = raw_width;
      raw_width = 2611;
      load_raw = &CLASS nokia_load_raw;
      filters = 0x16161616;
    } else
      is_raw = 0;
  }

  for (i = 0; i < sizeof corp / sizeof *corp; i++)
    if (strcasestr(make, corp[i])) /* Simplify company names */
      strcpy(make, corp[i]);
  if ((!strcmp(make, "Kodak") || !strcmp(make, "Leica")) &&
      ((cp = strcasestr(model, " DIGITAL CAMERA")) ||
       (cp = strstr(model, "FILE VERSION"))))
    *cp = 0;
  if (!strncasecmp(model, "PENTAX", 6))
    strcpy(make, "Pentax");
  cp = make + strlen(make); /* Remove trailing spaces */
  while (*--cp == ' ')
    *cp = 0;
  cp = model + strlen(model);
  while (*--cp == ' ')
    *cp = 0;
  i = strlen(make); /* Remove make from model */
  if (!strncasecmp(model, make, i) && model[i++] == ' ')
    memmove(model, model + i, 64 - i);
  if (!strncmp(model, "FinePix ", 8))
    strcpy(model, model + 8);
  if (!strncmp(model, "Digital Camera ", 15))
    strcpy(model, model + 15);
  desc[511] = artist[63] = make[63] = model[63] = model2[63] = 0;
  if (!is_raw)
    goto notraw;

  if (!height)
    height = raw_height;
  if (!width)
    width = raw_width;
  if (height == 2624 && width == 3936) /* Pentax K10D and Samsung GX10 */
  {
    height = 2616;
    width = 3896;
  }
  if (height == 3136 && width == 4864) /* Pentax K20D and Samsung GX20 */
  {
    height = 3124;
    width = 4688;
    filters = 0x16161616;
  }
  if (width == 4352 && (!strcmp(model, "K-r") || !strcmp(model, "K-x"))) {
    width = 4309;
    filters = 0x16161616;
  }
  if (width >= 4960 && !strncmp(model, "K-5", 3)) {
    left_margin = 10;
    width = 4950;
    filters = 0x16161616;
  }
  if (width == 4736 && !strcmp(model, "K-7")) {
    height = 3122;
    width = 4684;
    filters = 0x16161616;
    top_margin = 2;
  }
  if (width == 6080 && !strcmp(model, "K-3")) {
    left_margin = 4;
    width = 6040;
  }
  if (width == 7424 && !strcmp(model, "645D")) {
    height = 5502;
    width = 7328;
    filters = 0x61616161;
    top_margin = 29;
    left_margin = 48;
  }
  if (height == 3014 && width == 4096) /* Ricoh GX200 */
    width = 4014;
  if (dng_version) {
    if (filters == UINT_MAX)
      filters = 0;
    if (filters)
      is_raw *= tiff_samples;
    else
      colors = tiff_samples;
    switch (tiff_compress) {
      case 0:
      case 1:
        load_raw = &CLASS packed_dng_load_raw;
        break;
      case 7:
        load_raw = &CLASS lossless_dng_load_raw;
        break;
      case 34892:
        load_raw = &CLASS lossy_dng_load_raw;
        break;
      default:
        load_raw = 0;
    }
    goto dng_skip;
  }
  if (!strcmp(make, "Canon") && !fsize && tiff_bps != 15) {
    if (!load_raw)
      load_raw = &CLASS lossless_jpeg_load_raw;
    for (i = 0; i < sizeof canon / sizeof *canon; i++)
      if (raw_width == canon[i][0] && raw_height == canon[i][1]) {
        width = raw_width - (left_margin = canon[i][2]);
        height = raw_height - (top_margin = canon[i][3]);
        width -= canon[i][4];
        height -= canon[i][5];
        mask[0][1] = canon[i][6];
        mask[0][3] = -canon[i][7];
        mask[1][1] = canon[i][8];
        mask[1][3] = -canon[i][9];
        if (canon[i][10])
          filters = canon[i][10] * 0x01010101;
      }
    if ((unique_id | 0x20000) == 0x2720000) {
      left_margin = 8;
      top_margin = 16;
    }
  }
  for (i = 0; i < sizeof unique / sizeof *unique; i++)
    if (unique_id == 0x80000000 + unique[i].id) {
      adobe_coeff("Canon", unique[i].model);
      if (model[4] == 'K' && strlen(model) == 8)
        strcpy(model, unique[i].model);
    }
  for (i = 0; i < sizeof sonique / sizeof *sonique; i++)
    if (unique_id == sonique[i].id)
      strcpy(model, sonique[i].model);
  if (!strcmp(make, "Nikon")) {
    if (!load_raw)
      load_raw = &CLASS packed_load_raw;
    if (model[0] == 'E')
      load_flags |= !data_offset << 2 | 2;
  }

  /* Set parameters based on camera name (for non-DNG files). */

  if (!strcmp(model, "KAI-0340") && find_green(16, 16, 3840, 5120) < 25) {
    height = 480;
    top_margin = filters = 0;
    strcpy(model, "C603");
  }
  if (!strcmp(make, "Sony") && raw_width > 3888)
    black = 128 << (tiff_bps - 12);
  if (is_foveon) {
    if (height * 2 < width)
      pixel_aspect = 0.5;
    if (height > width)
      pixel_aspect = 2;
    filters = 0;
    simple_coeff(0);
  } else if (!strcmp(make, "Canon") && tiff_bps == 15) {
    switch (width) {
      case 3344:
        width -= 66;
      case 3872:
        width -= 6;
    }
    if (height > width) {
      SWAP(height, width);
      SWAP(raw_height, raw_width);
    }
    if (width == 7200 && height == 3888) {
      raw_width = width = 6480;
      raw_height = height = 4320;
    }
    filters = 0;
    tiff_samples = colors = 3;
    load_raw = &CLASS canon_sraw_load_raw;
  } else if (!strcmp(model, "PowerShot 600")) {
    height = 613;
    width = 854;
    raw_width = 896;
    colors = 4;
    filters = 0xe1e4e1e4;
    load_raw = &CLASS canon_600_load_raw;
  } else if (!strcmp(model, "PowerShot A5") ||
             !strcmp(model, "PowerShot A5 Zoom")) {
    height = 773;
    width = 960;
    raw_width = 992;
    pixel_aspect = 256 / 235.0;
    filters = 0x1e4e1e4e;
    goto canon_a5;
  } else if (!strcmp(model, "PowerShot A50")) {
    height = 968;
    width = 1290;
    raw_width = 1320;
    filters = 0x1b4e4b1e;
    goto canon_a5;
  } else if (!strcmp(model, "PowerShot Pro70")) {
    height = 1024;
    width = 1552;
    filters = 0x1e4b4e1b;
  canon_a5:
    colors = 4;
    tiff_bps = 10;
    load_raw = &CLASS packed_load_raw;
    load_flags = 40;
  } else if (!strcmp(model, "PowerShot Pro90 IS") ||
             !strcmp(model, "PowerShot G1")) {
    colors = 4;
    filters = 0xb4b4b4b4;
  } else if (!strcmp(model, "PowerShot A610")) {
    if (canon_s2is())
      strcpy(model + 10, "S2 IS");
  } else if (!strcmp(model, "PowerShot SX220 HS")) {
    mask[1][3] = -4;
  } else if (!strcmp(model, "EOS D2000C")) {
    filters = 0x61616161;
    black = curve[200];
  } else if (!strcmp(model, "D1")) {
    cam_mul[0] *= 256 / 527.0;
    cam_mul[2] *= 256 / 317.0;
  } else if (!strcmp(model, "D1X")) {
    width -= 4;
    pixel_aspect = 0.5;
  } else if (!strcmp(model, "D40X") || !strcmp(model, "D60") ||
             !strcmp(model, "D80") || !strcmp(model, "D3000")) {
    height -= 3;
    width -= 4;
  } else if (!strcmp(model, "D3") || !strcmp(model, "D3S") ||
             !strcmp(model, "D700")) {
    width -= 4;
    left_margin = 2;
  } else if (!strcmp(model, "D3100")) {
    width -= 28;
    left_margin = 6;
  } else if (!strcmp(model, "D5000") || !strcmp(model, "D90")) {
    width -= 42;
  } else if (!strcmp(model, "D5100") || !strcmp(model, "D7000") ||
             !strcmp(model, "COOLPIX A")) {
    width -= 44;
  } else if (!strcmp(model, "D3200") || !strncmp(model, "D6", 2) ||
             !strncmp(model, "D800", 4)) {
    width -= 46;
  } else if (!strcmp(model, "D4") || !strcmp(model, "Df")) {
    width -= 52;
    left_margin = 2;
  } else if (!strncmp(model, "D40", 3) || !strncmp(model, "D50", 3) ||
             !strncmp(model, "D70", 3)) {
    width--;
  } else if (!strcmp(model, "D100")) {
    if (load_flags)
      raw_width = (width += 3) + 3;
  } else if (!strcmp(model, "D200")) {
    left_margin = 1;
    width -= 4;
    filters = 0x94949494;
  } else if (!strncmp(model, "D2H", 3)) {
    left_margin = 6;
    width -= 14;
  } else if (!strncmp(model, "D2X", 3)) {
    if (width == 3264)
      width -= 32;
    else
      width -= 8;
  } else if (!strncmp(model, "D300", 4)) {
    width -= 32;
  } else if (!strncmp(model, "COOLPIX P", 9) && raw_width != 4032) {
    load_flags = 24;
    filters = 0x94949494;
    if (model[9] == '7' && iso_speed >= 400)
      black = 255;
  } else if (!strncmp(model, "1 ", 2)) {
    height -= 2;
  } else if (fsize == 1581060) {
    simple_coeff(3);
    pre_mul[0] = 1.2085;
    pre_mul[1] = 1.0943;
    pre_mul[3] = 1.1103;
  } else if (fsize == 3178560) {
    cam_mul[0] *= 4;
    cam_mul[2] *= 4;
  } else if (fsize == 4771840) {
    if (!timestamp && nikon_e995())
      strcpy(model, "E995");
    if (strcmp(model, "E995")) {
      filters = 0xb4b4b4b4;
      simple_coeff(3);
      pre_mul[0] = 1.196;
      pre_mul[1] = 1.246;
      pre_mul[2] = 1.018;
    }
  } else if (fsize == 2940928) {
    if (!timestamp && !nikon_e2100())
      strcpy(model, "E2500");
    if (!strcmp(model, "E2500")) {
      height -= 2;
      load_flags = 6;
      colors = 4;
      filters = 0x4b4b4b4b;
    }
  } else if (fsize == 4775936) {
    if (!timestamp)
      nikon_3700();
    if (model[0] == 'E' && atoi(model + 1) < 3700)
      filters = 0x49494949;
    if (!strcmp(model, "Optio 33WR")) {
      flip = 1;
      filters = 0x16161616;
    }
    if (make[0] == 'O') {
      i = find_green(12, 32, 1188864, 3576832);
      c = find_green(12, 32, 2383920, 2387016);
      if (abs(i) < abs(c)) {
        SWAP(i, c);
        load_flags = 24;
      }
      if (i < 0)
        filters = 0x61616161;
    }
  } else if (fsize == 5869568) {
    if (!timestamp && minolta_z2()) {
      strcpy(make, "Minolta");
      strcpy(model, "DiMAGE Z2");
    }
    load_flags = 6 + 24 * (make[0] == 'M');
  } else if (fsize == 6291456) {
    fseek(ifp, 0x300000, SEEK_SET);
    if ((order = guess_byte_order(0x10000)) == 0x4d4d) {
      height -= (top_margin = 16);
      width -= (left_margin = 28);
      maximum = 0xf5c0;
      strcpy(make, "ISG");
      model[0] = 0;
    }
  } else if (!strcmp(make, "Fujifilm")) {
    if (!strcmp(model + 7, "S2Pro")) {
      strcpy(model, "S2Pro");
      height = 2144;
      width = 2880;
      flip = 6;
    } else if (load_raw != &CLASS packed_load_raw)
      maximum = (is_raw == 2 && shot_select) ? 0x2f00 : 0x3e00;
    top_margin = (raw_height - height) >> 2 << 1;
    left_margin = (raw_width - width) >> 2 << 1;
    if (width == 2848 || width == 3664)
      filters = 0x16161616;
    if (width == 4032 || width == 4952 || width == 6032)
      left_margin = 0;
    if (width == 3328 && (width -= 66))
      left_margin = 34;
    if (width == 4936)
      left_margin = 4;
    if (!strcmp(model, "HS50EXR") || !strcmp(model, "F900EXR")) {
      width += 2;
      left_margin = 0;
      filters = 0x16161616;
    }
    if (fuji_layout)
      raw_width *= is_raw;
    if (filters == 9)
      FORC(36)
    ((char *)xtrans)[c] =
        xtrans_abs[(c / 6 + top_margin) % 6][(c + left_margin) % 6];
  } else if (!strcmp(model, "KD-400Z")) {
    height = 1712;
    width = 2312;
    raw_width = 2336;
    goto konica_400z;
  } else if (!strcmp(model, "KD-510Z")) {
    goto konica_510z;
  } else if (!strcasecmp(make, "Minolta")) {
    if (!load_raw && (maximum = 0xfff))
      load_raw = &CLASS unpacked_load_raw;
    if (!strncmp(model, "DiMAGE A", 8)) {
      if (!strcmp(model, "DiMAGE A200"))
        filters = 0x49494949;
      tiff_bps = 12;
      load_raw = &CLASS packed_load_raw;
    } else if (!strncmp(model, "ALPHA", 5) || !strncmp(model, "DYNAX", 5) ||
               !strncmp(model, "MAXXUM", 6)) {
      sprintf(model + 20, "DYNAX %-10s", model + 6 + (model[0] == 'M'));
      adobe_coeff(make, model + 20);
      load_raw = &CLASS packed_load_raw;
    } else if (!strncmp(model, "DiMAGE G", 8)) {
      if (model[8] == '4') {
        height = 1716;
        width = 2304;
      } else if (model[8] == '5') {
      konica_510z:
        height = 1956;
        width = 2607;
        raw_width = 2624;
      } else if (model[8] == '6') {
        height = 2136;
        width = 2848;
      }
      data_offset += 14;
      filters = 0x61616161;
    konica_400z:
      load_raw = &CLASS unpacked_load_raw;
      maximum = 0x3df;
      order = 0x4d4d;
    }
  } else if (!strcmp(model, "*ist D")) {
    load_raw = &CLASS unpacked_load_raw;
    data_error = -1;
  } else if (!strcmp(model, "*ist DS")) {
    height -= 2;
  } else if (!strcmp(make, "Samsung") && raw_width == 4704) {
    height -= top_margin = 8;
    width -= 2 * (left_margin = 8);
    load_flags = 32;
  } else if (!strcmp(make, "Samsung") && raw_height == 3714) {
    height -= top_margin = 18;
    left_margin = raw_width - (width = 5536);
    if (raw_width != 5600)
      left_margin = top_margin = 0;
    filters = 0x61616161;
    colors = 3;
  } else if (!strcmp(make, "Samsung") && raw_width == 5632) {
    order = 0x4949;
    height = 3694;
    top_margin = 2;
    width = 5574 - (left_margin = 32 + tiff_bps);
    if (tiff_bps == 12)
      load_flags = 80;
  } else if (!strcmp(make, "Samsung") && raw_width == 5664) {
    height -= top_margin = 17;
    left_margin = 96;
    width = 5544;
    filters = 0x49494949;
  } else if (!strcmp(make, "Samsung") && raw_width == 6496) {
    filters = 0x61616161;
    black = 1 << (tiff_bps - 7);
  } else if (!strcmp(model, "EX1")) {
    order = 0x4949;
    height -= 20;
    top_margin = 2;
    if ((width -= 6) > 3682) {
      height -= 10;
      width -= 46;
      top_margin = 8;
    }
  } else if (!strcmp(model, "WB2000")) {
    order = 0x4949;
    height -= 3;
    top_margin = 2;
    if ((width -= 10) > 3718) {
      height -= 28;
      width -= 56;
      top_margin = 8;
    }
  } else if (strstr(model, "WB550")) {
    strcpy(model, "WB550");
  } else if (!strcmp(model, "EX2F")) {
    height = 3045;
    width = 4070;
    top_margin = 3;
    order = 0x4949;
    filters = 0x49494949;
    load_raw = &CLASS unpacked_load_raw;
  } else if (!strcmp(model, "STV680 VGA")) {
    black = 16;
  } else if (!strcmp(model, "N95")) {
    height = raw_height - (top_margin = 2);
  } else if (!strcmp(model, "640x480")) {
    gamma_curve(0.45, 4.5, 1, 255);
  } else if (!strcmp(make, "Hasselblad")) {
    if (load_raw == &CLASS lossless_jpeg_load_raw)
      load_raw = &CLASS hasselblad_load_raw;
    if (raw_width == 7262) {
      height = 5444;
      width = 7248;
      top_margin = 4;
      left_margin = 7;
      filters = 0x61616161;
    } else if (raw_width == 7410 || raw_width == 8282) {
      height -= 84;
      width -= 82;
      top_margin = 4;
      left_margin = 41;
      filters = 0x61616161;
    } else if (raw_width == 9044) {
      height = 6716;
      width = 8964;
      top_margin = 8;
      left_margin = 40;
      black += load_flags = 256;
      maximum = 0x8101;
    } else if (raw_width == 4090) {
      strcpy(model, "V96C");
      height -= (top_margin = 6);
      width -= (left_margin = 3) + 7;
      filters = 0x61616161;
    }
    if (tiff_samples > 1) {
      is_raw = tiff_samples + 1;
      if (!shot_select && !half_size)
        filters = 0;
    }
  } else if (!strcmp(make, "Sinar")) {
    if (!load_raw)
      load_raw = &CLASS unpacked_load_raw;
    if (is_raw > 1 && !shot_select && !half_size)
      filters = 0;
    maximum = 0x3fff;
  } else if (!strcmp(make, "Leaf")) {
    maximum = 0x3fff;
    fseek(ifp, data_offset, SEEK_SET);
    if (ljpeg_start(&jh, 1) && jh.bits == 15)
      maximum = 0x1fff;
    if (tiff_samples > 1)
      filters = 0;
    if (tiff_samples > 1 || tile_length < raw_height) {
      load_raw = &CLASS leaf_hdr_load_raw;
      raw_width = tile_width;
    }
    if ((width | height) == 2048) {
      if (tiff_samples == 1) {
        filters = 1;
        strcpy(cdesc, "RBTG");
        strcpy(model, "CatchLight");
        top_margin = 8;
        left_margin = 18;
        height = 2032;
        width = 2016;
      } else {
        strcpy(model, "DCB2");
        top_margin = 10;
        left_margin = 16;
        height = 2028;
        width = 2022;
      }
    } else if (width + height == 3144 + 2060) {
      if (!model[0])
        strcpy(model, "Cantare");
      if (width > height) {
        top_margin = 6;
        left_margin = 32;
        height = 2048;
        width = 3072;
        filters = 0x61616161;
      } else {
        left_margin = 6;
        top_margin = 32;
        width = 2048;
        height = 3072;
        filters = 0x16161616;
      }
      if (!cam_mul[0] || model[0] == 'V')
        filters = 0;
      else
        is_raw = tiff_samples;
    } else if (width == 2116) {
      strcpy(model, "Valeo 6");
      height -= 2 * (top_margin = 30);
      width -= 2 * (left_margin = 55);
      filters = 0x49494949;
    } else if (width == 3171) {
      strcpy(model, "Valeo 6");
      height -= 2 * (top_margin = 24);
      width -= 2 * (left_margin = 24);
      filters = 0x16161616;
    }
  } else if (!strcmp(make, "Leica") || !strcmp(make, "Panasonic")) {
    if ((flen - data_offset) / (raw_width * 8 / 7) == raw_height)
      load_raw = &CLASS panasonic_load_raw;
    if (!load_raw) {
      load_raw = &CLASS unpacked_load_raw;
      load_flags = 4;
    }
    zero_is_bad = 1;
    if ((height += 12) > raw_height)
      height = raw_height;
    for (i = 0; i < sizeof pana / sizeof *pana; i++)
      if (raw_width == pana[i][0] && raw_height == pana[i][1]) {
        left_margin = pana[i][2];
        top_margin = pana[i][3];
        width += pana[i][4];
        height += pana[i][5];
      }
    filters = 0x01010101 *
              (uchar) "\x94\x61\x49\x16"[((filters - 1) ^ (left_margin & 1) ^
                                          (top_margin << 1)) &
                                         3];
  } else if (!strcmp(model, "C770UZ")) {
    height = 1718;
    width = 2304;
    filters = 0x16161616;
    load_raw = &CLASS packed_load_raw;
    load_flags = 30;
  } else if (!strcmp(make, "Olympus")) {
    height += height & 1;
    if (exif_cfa)
      filters = exif_cfa;
    if (width == 4100)
      width -= 4;
    if (width == 4080)
      width -= 24;
    if (width == 9280) {
      width -= 6;
      height -= 6;
    }
    if (load_raw == &CLASS unpacked_load_raw)
      load_flags = 4;
    tiff_bps = 12;
    if (!strcmp(model, "E-300") || !strcmp(model, "E-500")) {
      width -= 20;
      if (load_raw == &CLASS unpacked_load_raw) {
        maximum = 0xfc3;
        memset(cblack, 0, sizeof cblack);
      }
    } else if (!strcmp(model, "E-330")) {
      width -= 30;
      if (load_raw == &CLASS unpacked_load_raw)
        maximum = 0xf79;
    } else if (!strcmp(model, "SP550UZ")) {
      thumb_length = flen - (thumb_offset = 0xa39800);
      thumb_height = 480;
      thumb_width = 640;
    } else if (!strcmp(model, "TG-4")) {
      width -= 16;
    }
  } else if (!strcmp(model, "N Digital")) {
    height = 2047;
    width = 3072;
    filters = 0x61616161;
    data_offset = 0x1a00;
    load_raw = &CLASS packed_load_raw;
  } else if (!strcmp(model, "DSC-F828")) {
    width = 3288;
    left_margin = 5;
    mask[1][3] = -17;
    data_offset = 862144;
    load_raw = &CLASS sony_load_raw;
    filters = 0x9c9c9c9c;
    colors = 4;
    strcpy(cdesc, "RGBE");
  } else if (!strcmp(model, "DSC-V3")) {
    width = 3109;
    left_margin = 59;
    mask[0][1] = 9;
    data_offset = 787392;
    load_raw = &CLASS sony_load_raw;
  } else if (!strcmp(make, "Sony") && raw_width == 3984) {
    width = 3925;
    order = 0x4d4d;
  } else if (!strcmp(make, "Sony") && raw_width == 4288) {
    width -= 32;
  } else if (!strcmp(make, "Sony") && raw_width == 4600) {
    if (!strcmp(model, "DSLR-A350"))
      height -= 4;
    black = 0;
  } else if (!strcmp(make, "Sony") && raw_width == 4928) {
    if (height < 3280)
      width -= 8;
  } else if (!strcmp(make, "Sony") && raw_width == 5504) {
    width -= height > 3664 ? 8 : 32;
    if (!strncmp(model, "DSC", 3))
      black = 200 << (tiff_bps - 12);
  } else if (!strcmp(make, "Sony") && raw_width == 6048) {
    width -= 24;
    if (strstr(model, "RX1") || strstr(model, "A99"))
      width -= 6;
  } else if (!strcmp(make, "Sony") && raw_width == 7392) {
    width -= 30;
  } else if (!strcmp(make, "Sony") && raw_width == 8000) {
    width -= 32;
    if (!strncmp(model, "DSC", 3)) {
      tiff_bps = 14;
      load_raw = &CLASS unpacked_load_raw;
      black = 512;
    }
  } else if (!strcmp(model, "DSLR-A100")) {
    if (width == 3880) {
      height--;
      width = ++raw_width;
    } else {
      height -= 4;
      width -= 4;
      order = 0x4d4d;
      load_flags = 2;
    }
    filters = 0x61616161;
  } else if (!strcmp(model, "PIXL")) {
    height -= top_margin = 4;
    width -= left_margin = 32;
    gamma_curve(0, 7, 1, 255);
  } else if (!strcmp(model, "C603") || !strcmp(model, "C330") ||
             !strcmp(model, "12MP")) {
    order = 0x4949;
    if (filters && data_offset) {
      fseek(ifp, data_offset < 4096 ? 168 : 5252, SEEK_SET);
      read_shorts(curve, 256);
    } else
      gamma_curve(0, 3.875, 1, 255);
    load_raw = filters ? &CLASS eight_bit_load_raw
                       : strcmp(model, "C330") ? &CLASS kodak_c603_load_raw
                                               : &CLASS kodak_c330_load_raw;
    load_flags = tiff_bps > 16;
    tiff_bps = 8;
  } else if (!strncasecmp(model, "EasyShare", 9)) {
    data_offset = data_offset < 0x15000 ? 0x15000 : 0x17000;
    load_raw = &CLASS packed_load_raw;
  } else if (!strcasecmp(make, "Kodak")) {
    if (filters == UINT_MAX)
      filters = 0x61616161;
    if (!strncmp(model, "NC2000", 6) || !strncmp(model, "EOSDCS", 6) ||
        !strncmp(model, "DCS4", 4)) {
      width -= 4;
      left_margin = 2;
      if (model[6] == ' ')
        model[6] = 0;
      if (!strcmp(model, "DCS460A"))
        goto bw;
    } else if (!strcmp(model, "DCS660M")) {
      black = 214;
      goto bw;
    } else if (!strcmp(model, "DCS760M")) {
    bw:
      colors = 1;
      filters = 0;
    }
    if (!strcmp(model + 4, "20X"))
      strcpy(cdesc, "MYCY");
    if (strstr(model, "DC25")) {
      strcpy(model, "DC25");
      data_offset = 15424;
    }
    if (!strncmp(model, "DC2", 3)) {
      raw_height = 2 + (height = 242);
      if (flen < 100000) {
        raw_width = 256;
        width = 249;
        pixel_aspect = (4.0 * height) / (3.0 * width);
      } else {
        raw_width = 512;
        width = 501;
        pixel_aspect = (493.0 * height) / (373.0 * width);
      }
      top_margin = left_margin = 1;
      colors = 4;
      filters = 0x8d8d8d8d;
      simple_coeff(1);
      pre_mul[1] = 1.179;
      pre_mul[2] = 1.209;
      pre_mul[3] = 1.036;
      load_raw = &CLASS eight_bit_load_raw;
    } else if (!strcmp(model, "40")) {
      strcpy(model, "DC40");
      height = 512;
      width = 768;
      data_offset = 1152;
      load_raw = &CLASS kodak_radc_load_raw;
      tiff_bps = 12;
    } else if (strstr(model, "DC50")) {
      strcpy(model, "DC50");
      height = 512;
      width = 768;
      data_offset = 19712;
      load_raw = &CLASS kodak_radc_load_raw;
    } else if (strstr(model, "DC120")) {
      strcpy(model, "DC120");
      height = 976;
      width = 848;
      pixel_aspect = height / 0.75 / width;
      load_raw = tiff_compress == 7 ? &CLASS kodak_jpeg_load_raw
                                    : &CLASS kodak_dc120_load_raw;
    } else if (!strcmp(model, "DCS200")) {
      thumb_height = 128;
      thumb_width = 192;
      thumb_offset = 6144;
      thumb_misc = 360;
      write_thumb = &CLASS layer_thumb;
      black = 17;
    }
  } else if (!strcmp(model, "Fotoman Pixtura")) {
    height = 512;
    width = 768;
    data_offset = 3632;
    load_raw = &CLASS kodak_radc_load_raw;
    filters = 0x61616161;
    simple_coeff(2);
  } else if (!strncmp(model, "QuickTake", 9)) {
    if (head[5])
      strcpy(model + 10, "200");
    fseek(ifp, 544, SEEK_SET);
    height = get2();
    width = get2();
    data_offset = (get4(), get2()) == 30 ? 738 : 736;
    if (height > width) {
      SWAP(height, width);
      fseek(ifp, data_offset - 6, SEEK_SET);
      flip = ~get2() & 3 ? 5 : 6;
    }
    filters = 0x61616161;
  } else if (!strcmp(make, "Rollei") && !load_raw) {
    switch (raw_width) {
      case 1316:
        height = 1030;
        width = 1300;
        top_margin = 1;
        left_margin = 6;
        break;
      case 2568:
        height = 1960;
        width = 2560;
        top_margin = 2;
        left_margin = 8;
    }
    filters = 0x16161616;
    load_raw = &CLASS rollei_load_raw;
  }
  if (!model[0])
    sprintf(model, "%dx%d", width, height);
  if (filters == UINT_MAX)
    filters = 0x94949494;
  if (thumb_offset && !thumb_height) {
    fseek(ifp, thumb_offset, SEEK_SET);
    if (ljpeg_start(&jh, 1)) {
      thumb_width = jh.wide;
      thumb_height = jh.high;
    }
  }
dng_skip:
  if ((use_camera_matrix & (use_camera_wb || dng_version)) &&
      cmatrix[0][0] > 0.125) {
    memcpy(rgb_cam, cmatrix, sizeof cmatrix);
    raw_color = 0;
  }
  if (raw_color)
    adobe_coeff(make, model);
  if (load_raw == &CLASS kodak_radc_load_raw)
    if (raw_color)
      adobe_coeff("Apple", "Quicktake");
  if (fuji_width) {
    fuji_width = width >> !fuji_layout;
    filters = fuji_width & 1 ? 0x94949494 : 0x49494949;
    width = (height >> fuji_layout) + fuji_width;
    height = width - 1;
    pixel_aspect = 1;
  } else {
    if (raw_height < height)
      raw_height = height;
    if (raw_width < width)
      raw_width = width;
  }
  if (!tiff_bps)
    tiff_bps = 12;
  if (!maximum)
    maximum = (1 << tiff_bps) - 1;
  if (!load_raw || height < 22 || width < 22 || tiff_bps > 16 ||
      tiff_samples > 6 || colors > 4)
    is_raw = 0;
#ifdef NO_JASPER
  if (load_raw == &CLASS redcine_load_raw) {
    fprintf(stderr, _("%s: You must link dcraw with %s!!\n"), ifname,
            "libjasper");
    is_raw = 0;
  }
#endif
#ifdef NO_JPEG
  if (load_raw == &CLASS kodak_jpeg_load_raw ||
      load_raw == &CLASS lossy_dng_load_raw) {
    fprintf(stderr, _("%s: You must link dcraw with %s!!\n"), ifname,
            "libjpeg");
    is_raw = 0;
  }
#endif
  if (!cdesc[0])
    strcpy(cdesc, colors == 3 ? "RGBG" : "GMCY");
  if (!raw_height)
    raw_height = height;
  if (!raw_width)
    raw_width = width;
  if (filters > 999 && colors == 3)
    filters |= ((filters >> 2 & 0x22222222) | (filters << 2 & 0x88888888)) &
               filters << 1;
notraw:
  if (flip == UINT_MAX)
    flip = tiff_flip;
  if (flip == UINT_MAX)
    flip = 0;
}

#ifndef NO_LCMS
void CLASS apply_profile(const char *input, const char *output) {
  char *prof;
  cmsHPROFILE hInProfile = 0, hOutProfile = 0;
  cmsHTRANSFORM hTransform;
  FILE *fp;
  unsigned size;

  if (strcmp(input, "embed"))
    hInProfile = cmsOpenProfileFromFile(input, "r");
  else if (profile_length) {
    prof = (char *)malloc(profile_length);
    merror(prof, "apply_profile()");
    fseek(ifp, profile_offset, SEEK_SET);
    fread(prof, 1, profile_length, ifp);
    hInProfile = cmsOpenProfileFromMem(prof, profile_length);
    free(prof);
  } else
    fprintf(stderr, _("%s has no embedded profile.\n"), ifname);
  if (!hInProfile)
    return;
  if (!output)
    hOutProfile = cmsCreate_sRGBProfile();
  else if ((fp = fopen(output, "rb"))) {
    fread(&size, 4, 1, fp);
    fseek(fp, 0, SEEK_SET);
    oprof = (unsigned *)malloc(size = ntohl(size));
    merror(oprof, "apply_profile()");
    fread(oprof, 1, size, fp);
    fclose(fp);
    if (!(hOutProfile = cmsOpenProfileFromMem(oprof, size))) {
      free(oprof);
      oprof = 0;
    }
  } else
    fprintf(stderr, _("Cannot open file %s!\n"), output);
  if (!hOutProfile)
    goto quit;
  if (verbose)
    fprintf(stderr, _("Applying color profile...\n"));
  hTransform = cmsCreateTransform(hInProfile, TYPE_RGBA_16, hOutProfile,
                                  TYPE_RGBA_16, INTENT_PERCEPTUAL, 0);
  cmsDoTransform(hTransform, image, image, width * height);
  raw_color = 1; /* Don't use rgb_cam with a profile */
  cmsDeleteTransform(hTransform);
  cmsCloseProfile(hOutProfile);
quit:
  cmsCloseProfile(hInProfile);
}
#endif

void CLASS convert_to_rgb() {
  int row, col, c, i, j, k;
  ushort *img;
  float out[3], out_cam[3][4];
  double num, inverse[3][3];
  static const double xyzd50_srgb[3][3] = {{0.436083, 0.385083, 0.143055},
                                           {0.222507, 0.716888, 0.060608},
                                           {0.013930, 0.097097, 0.714022}};
  static const double rgb_rgb[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  static const double adobe_rgb[3][3] = {{0.715146, 0.284856, 0.000000},
                                         {0.000000, 1.000000, 0.000000},
                                         {0.000000, 0.041166, 0.958839}};
  static const double wide_rgb[3][3] = {{0.593087, 0.404710, 0.002206},
                                        {0.095413, 0.843149, 0.061439},
                                        {0.011621, 0.069091, 0.919288}};
  static const double prophoto_rgb[3][3] = {{0.529317, 0.330092, 0.140588},
                                            {0.098368, 0.873465, 0.028169},
                                            {0.016879, 0.117663, 0.865457}};
  static const double aces_rgb[3][3] = {{0.432996, 0.375380, 0.189317},
                                        {0.089427, 0.816523, 0.102989},
                                        {0.019165, 0.118150, 0.941914}};
  static const double(*out_rgb[])[3] = {rgb_rgb,      adobe_rgb, wide_rgb,
                                        prophoto_rgb, xyz_rgb,   aces_rgb};
  static const char *name[] = {"sRGB",          "Adobe RGB (1998)",
                               "WideGamut D65", "ProPhoto D65",
                               "XYZ",           "ACES"};
  static const unsigned phead[] = {
      1024, 0, 0x2100000,  0x6d6e7472, 0x52474220, 0x58595a20, 0,
      0,    0, 0x61637370, 0,          0,          0x6e6f6e65, 0,
      0,    0, 0,          0xf6d6,     0x10000,    0xd32d};
  unsigned pbody[] = {10,         0x63707274, 0,  36, /* cprt */
                      0x64657363, 0,          40,     /* desc */
                      0x77747074, 0,          20,     /* wtpt */
                      0x626b7074, 0,          20,     /* bkpt */
                      0x72545243, 0,          14,     /* rTRC */
                      0x67545243, 0,          14,     /* gTRC */
                      0x62545243, 0,          14,     /* bTRC */
                      0x7258595a, 0,          20,     /* rXYZ */
                      0x6758595a, 0,          20,     /* gXYZ */
                      0x6258595a, 0,          20};    /* bXYZ */
  static const unsigned pwhite[] = {0xf351, 0x10000, 0x116cc};
  unsigned pcurve[] = {0x63757276, 0, 1, 0x1000000};

  gamma_curve(gamm[0], gamm[1], 0, 0);
  memcpy(out_cam, rgb_cam, sizeof out_cam);
  raw_color |=
      colors == 1 || document_mode || output_color < 1 || output_color > 6;
  if (!raw_color) {
    oprof = (unsigned *)calloc(phead[0], 1);
    merror(oprof, "convert_to_rgb()");
    memcpy(oprof, phead, sizeof phead);
    if (output_color == 5)
      oprof[4] = oprof[5];
    oprof[0] = 132 + 12 * pbody[0];
    for (i = 0; i < pbody[0]; i++) {
      oprof[oprof[0] / 4] = i ? (i > 1 ? 0x58595a20 : 0x64657363) : 0x74657874;
      pbody[i * 3 + 2] = oprof[0];
      oprof[0] += (pbody[i * 3 + 3] + 3) & -4;
    }
    memcpy(oprof + 32, pbody, sizeof pbody);
    oprof[pbody[5] / 4 + 2] = strlen(name[output_color - 1]) + 1;
    memcpy((char *)oprof + pbody[8] + 8, pwhite, sizeof pwhite);
    pcurve[3] = (short)(256 / gamm[5] + 0.5) << 16;
    for (i = 4; i < 7; i++)
      memcpy((char *)oprof + pbody[i * 3 + 2], pcurve, sizeof pcurve);
    pseudoinverse((double(*)[3])out_rgb[output_color - 1], inverse, 3);
    for (i = 0; i < 3; i++)
      for (j = 0; j < 3; j++) {
        for (num = k = 0; k < 3; k++)
          num += xyzd50_srgb[i][k] * inverse[j][k];
        oprof[pbody[j * 3 + 23] / 4 + i + 2] = num * 0x10000 + 0.5;
      }
    for (i = 0; i < phead[0] / 4; i++)
      oprof[i] = htonl(oprof[i]);
    strcpy((char *)oprof + pbody[2] + 8, "auto-generated by dcraw");
    strcpy((char *)oprof + pbody[5] + 12, name[output_color - 1]);
    for (i = 0; i < 3; i++)
      for (j = 0; j < colors; j++)
        for (out_cam[i][j] = k = 0; k < 3; k++)
          out_cam[i][j] += out_rgb[output_color - 1][i][k] * rgb_cam[k][j];
  }
  if (verbose)
    fprintf(stderr,
            raw_color ? _("Building histograms...\n")
                      : _("Converting to %s colorspace...\n"),
            name[output_color - 1]);

  memset(histogram, 0, sizeof histogram);
  for (img = image[0], row = 0; row < height; row++)
    for (col = 0; col < width; col++, img += 4) {
      if (!raw_color) {
        out[0] = out[1] = out[2] = 0;
        FORCC {
          out[0] += out_cam[0][c] * img[c];
          out[1] += out_cam[1][c] * img[c];
          out[2] += out_cam[2][c] * img[c];
        }
        FORC3 img[c] = CLIP((int)out[c]);
      } else if (document_mode)
        img[0] = img[fcol(row, col)];
      FORCC histogram[c][img[c] >> 3]++;
    }
  if (colors == 4 && output_color)
    colors = 3;
  if (document_mode && filters)
    colors = 1;
}

void CLASS fuji_rotate() {
  int i, row, col;
  double step;
  float r, c, fr, fc;
  unsigned ur, uc;
  ushort wide, high, (*img)[4], (*pix)[4];

  if (!fuji_width)
    return;
  if (verbose)
    fprintf(stderr, _("Rotating image 45 degrees...\n"));
  fuji_width = (fuji_width - 1 + shrink) >> shrink;
  step = sqrt(0.5);
  wide = fuji_width / step;
  high = (height - fuji_width) / step;
  img = (ushort(*)[4])calloc(high, wide * sizeof *img);
  merror(img, "fuji_rotate()");

  for (row = 0; row < high; row++)
    for (col = 0; col < wide; col++) {
      ur = r = fuji_width + (row - col) * step;
      uc = c = (row + col) * step;
      if (ur > height - 2 || uc > width - 2)
        continue;
      fr = r - ur;
      fc = c - uc;
      pix = image + ur * width + uc;
      for (i = 0; i < colors; i++)
        img[row * wide + col][i] =
            (pix[0][i] * (1 - fc) + pix[1][i] * fc) * (1 - fr) +
            (pix[width][i] * (1 - fc) + pix[width + 1][i] * fc) * fr;
    }
  free(image);
  width = wide;
  height = high;
  image = img;
  fuji_width = 0;
}

void CLASS stretch() {
  ushort newdim, (*img)[4], *pix0, *pix1;
  int row, col, c;
  double rc, frac;

  if (pixel_aspect == 1)
    return;
  if (verbose)
    fprintf(stderr, _("Stretching the image...\n"));
  if (pixel_aspect < 1) {
    newdim = height / pixel_aspect + 0.5;
    img = (ushort(*)[4])calloc(width, newdim * sizeof *img);
    merror(img, "stretch()");
    for (rc = row = 0; row < newdim; row++, rc += pixel_aspect) {
      frac = rc - (c = rc);
      pix0 = pix1 = image[c * width];
      if (c + 1 < height)
        pix1 += width * 4;
      for (col = 0; col < width; col++, pix0 += 4, pix1 += 4)
        FORCC img[row * width + col][c] =
            pix0[c] * (1 - frac) + pix1[c] * frac + 0.5;
    }
    height = newdim;
  } else {
    newdim = width * pixel_aspect + 0.5;
    img = (ushort(*)[4])calloc(height, newdim * sizeof *img);
    merror(img, "stretch()");
    for (rc = col = 0; col < newdim; col++, rc += 1 / pixel_aspect) {
      frac = rc - (c = rc);
      pix0 = pix1 = image[c];
      if (c + 1 < width)
        pix1 += 4;
      for (row = 0; row < height; row++, pix0 += width * 4, pix1 += width * 4)
        FORCC img[row * newdim + col][c] =
            pix0[c] * (1 - frac) + pix1[c] * frac + 0.5;
    }
    width = newdim;
  }
  free(image);
  image = img;
}

int CLASS flip_index(int row, int col) {
  if (flip & 4)
    SWAP(row, col);
  if (flip & 2)
    row = iheight - 1 - row;
  if (flip & 1)
    col = iwidth - 1 - col;
  return row * iwidth + col;
}

struct tiff_tag {
  ushort tag, type;
  int count;
  union {
    char c[4];
    short s[2];
    int i;
  } val;
};

struct tiff_hdr {
  ushort order, magic;
  int ifd;
  ushort pad, ntag;
  struct tiff_tag tag[23];
  int nextifd;
  ushort pad2, nexif;
  struct tiff_tag exif[4];
  ushort pad3, ngps;
  struct tiff_tag gpst[10];
  short bps[4];
  int rat[10];
  unsigned gps[26];
  char desc[512], make[64], model[64], soft[32], date[20], artist[64];
};

void CLASS tiff_set(struct tiff_hdr *th,
                    ushort *ntag,
                    ushort tag,
                    ushort type,
                    int count,
                    int val) {
  struct tiff_tag *tt;
  int c;

  tt = (struct tiff_tag *)(ntag + 1) + (*ntag)++;
  tt->val.i = val;
  if (type == 1 && count <= 4)
    FORC(4) tt->val.c[c] = val >> (c << 3);
  else if (type == 2) {
    count = strnlen((char *)th + val, count - 1) + 1;
    if (count <= 4)
      FORC(4) tt->val.c[c] = ((char *)th)[val + c];
  } else if (type == 3 && count <= 2)
    FORC(2) tt->val.s[c] = val >> (c << 4);
  tt->count = count;
  tt->type = type;
  tt->tag = tag;
}

#define TOFF(ptr) ((char *)(&(ptr)) - (char *)th)

void CLASS tiff_head(struct tiff_hdr *th, int full) {
  int c, psize = 0;
  struct tm *t;

  memset(th, 0, sizeof *th);
  th->order = htonl(0x4d4d4949) >> 16;
  th->magic = 42;
  th->ifd = 10;
  th->rat[0] = th->rat[2] = 300;
  th->rat[1] = th->rat[3] = 1;
  FORC(6) th->rat[4 + c] = 1000000;
  th->rat[4] *= shutter;
  th->rat[6] *= aperture;
  th->rat[8] *= focal_len;
  strncpy(th->desc, desc, 512);
  strncpy(th->make, make, 64);
  strncpy(th->model, model, 64);
  strcpy(th->soft, "dcraw v" DCRAW_VERSION);
  t = localtime(&timestamp);
  sprintf(th->date, "%04d:%02d:%02d %02d:%02d:%02d", t->tm_year + 1900,
          t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
  strncpy(th->artist, artist, 64);
  if (full) {
    tiff_set(th, &th->ntag, 254, 4, 1, 0);
    tiff_set(th, &th->ntag, 256, 4, 1, width);
    tiff_set(th, &th->ntag, 257, 4, 1, height);
    tiff_set(th, &th->ntag, 258, 3, colors, output_bps);
    if (colors > 2)
      th->tag[th->ntag - 1].val.i = TOFF(th->bps);
    FORC4 th->bps[c] = output_bps;
    tiff_set(th, &th->ntag, 259, 3, 1, 1);
    tiff_set(th, &th->ntag, 262, 3, 1, 1 + (colors > 1));
  }
  tiff_set(th, &th->ntag, 270, 2, 512, TOFF(th->desc));
  tiff_set(th, &th->ntag, 271, 2, 64, TOFF(th->make));
  tiff_set(th, &th->ntag, 272, 2, 64, TOFF(th->model));
  if (full) {
    if (oprof)
      psize = ntohl(oprof[0]);
    tiff_set(th, &th->ntag, 273, 4, 1, sizeof *th + psize);
    tiff_set(th, &th->ntag, 277, 3, 1, colors);
    tiff_set(th, &th->ntag, 278, 4, 1, height);
    tiff_set(th, &th->ntag, 279, 4, 1,
             height * width * colors * output_bps / 8);
  } else
    tiff_set(th, &th->ntag, 274, 3, 1, "12435867"[flip] - '0');
  tiff_set(th, &th->ntag, 282, 5, 1, TOFF(th->rat[0]));
  tiff_set(th, &th->ntag, 283, 5, 1, TOFF(th->rat[2]));
  tiff_set(th, &th->ntag, 284, 3, 1, 1);
  tiff_set(th, &th->ntag, 296, 3, 1, 2);
  tiff_set(th, &th->ntag, 305, 2, 32, TOFF(th->soft));
  tiff_set(th, &th->ntag, 306, 2, 20, TOFF(th->date));
  tiff_set(th, &th->ntag, 315, 2, 64, TOFF(th->artist));
  tiff_set(th, &th->ntag, 34665, 4, 1, TOFF(th->nexif));
  if (psize)
    tiff_set(th, &th->ntag, 34675, 7, psize, sizeof *th);
  tiff_set(th, &th->nexif, 33434, 5, 1, TOFF(th->rat[4]));
  tiff_set(th, &th->nexif, 33437, 5, 1, TOFF(th->rat[6]));
  tiff_set(th, &th->nexif, 34855, 3, 1, iso_speed);
  tiff_set(th, &th->nexif, 37386, 5, 1, TOFF(th->rat[8]));
  if (gpsdata[1]) {
    tiff_set(th, &th->ntag, 34853, 4, 1, TOFF(th->ngps));
    tiff_set(th, &th->ngps, 0, 1, 4, 0x202);
    tiff_set(th, &th->ngps, 1, 2, 2, gpsdata[29]);
    tiff_set(th, &th->ngps, 2, 5, 3, TOFF(th->gps[0]));
    tiff_set(th, &th->ngps, 3, 2, 2, gpsdata[30]);
    tiff_set(th, &th->ngps, 4, 5, 3, TOFF(th->gps[6]));
    tiff_set(th, &th->ngps, 5, 1, 1, gpsdata[31]);
    tiff_set(th, &th->ngps, 6, 5, 1, TOFF(th->gps[18]));
    tiff_set(th, &th->ngps, 7, 5, 3, TOFF(th->gps[12]));
    tiff_set(th, &th->ngps, 18, 2, 12, TOFF(th->gps[20]));
    tiff_set(th, &th->ngps, 29, 2, 12, TOFF(th->gps[23]));
    memcpy(th->gps, gpsdata, sizeof th->gps);
  }
}

void CLASS jpeg_thumb() {
  char *thumb;
  ushort exif[5];
  struct tiff_hdr th;

  thumb = (char *)malloc(thumb_length);
  merror(thumb, "jpeg_thumb()");
  fread(thumb, 1, thumb_length, ifp);
  fputc(0xff, ofp);
  fputc(0xd8, ofp);
  if (strcmp(thumb + 6, "Exif")) {
    memcpy(exif, "\xff\xe1  Exif\0\0", 10);
    exif[1] = htons(8 + sizeof th);
    fwrite(exif, 1, sizeof exif, ofp);
    tiff_head(&th, 0);
    fwrite(&th, 1, sizeof th, ofp);
  }
  fwrite(thumb + 2, 1, thumb_length - 2, ofp);
  free(thumb);
}

// We no longer need to write to a tiff file but to a taichi::Array2D<Vector4>

void CLASS write_ppm_tiff(DCRawOutput &output) {
  struct tiff_hdr th;
  uchar *ppm;
  ushort *ppm2;
  int c, row, col, soff, rstep, cstep;
  int perc, val, total, white = 0x2000;

  perc = width * height * 0.01; /* 99th percentile white level */
  if (fuji_width)
    perc /= 2;
  if (!((highlight & ~2) || no_auto_bright))
    for (white = c = 0; c < colors; c++) {
      for (val = 0x2000, total = 0; --val > 32;)
        if ((total += histogram[c][val]) > perc)
          break;
      if (white < val)
        white = val;
    }
  gamma_curve(gamm[0], gamm[1], 2, (white << 3) / bright);
  iheight = height;
  iwidth = width;
  if (flip & 4)
    SWAP(height, width);
  ppm = (uchar *)calloc(width, colors * output_bps / 8);
  ppm2 = (ushort *)ppm;
  if (output_tiff) {
    tiff_head(&th, 1);
    fwrite(&th, sizeof th, 1, ofp);
    if (oprof)
      fwrite(oprof, ntohl(oprof[0]), 1, ofp);
  }
  soff = flip_index(0, 0);
  cstep = flip_index(0, 1) - soff;
  rstep = flip_index(1, 0) - flip_index(0, width);
  output.initialize(width, height, colors);
  for (row = 0; row < height; row++, soff += rstep) {
    for (col = 0; col < width; col++, soff += cstep) {
      FORCC ppm2[col * colors + c] = curve[image[soff][c]];
      for (int i = 0; i < colors; i++) {
        output.data[colors * (row * width + col) + i] =
            ppm2[col * colors + i] * (1.0f / (1 << 16));
      }
    }
    // fwrite(ppm, colors*output_bps / 8, width, ofp);
  }
  free(ppm);
}

/*
void CLASS write_ppm_tiff()
{
    struct tiff_hdr th;
    uchar *ppm;
    ushort *ppm2;
    int c, row, col, soff, rstep, cstep;
    int perc, val, total, white = 0x2000;

    perc = width * height * 0.01;        //99th percentile white level
    if (fuji_width) perc /= 2;
    if (!((highlight & ~2) || no_auto_bright))
        for (white = c = 0; c < colors; c++) {
            for (val = 0x2000, total = 0; --val > 32; )
                if ((total += histogram[c][val]) > perc) break;
            if (white < val) white = val;
        }
    gamma_curve(gamm[0], gamm[1], 2, (white << 3) / bright);
    iheight = height;
    iwidth = width;
    if (flip & 4) SWAP(height, width);
    ppm = (uchar *)calloc(width, colors*output_bps / 8);
    ppm2 = (ushort *)ppm;
    merror(ppm, "write_ppm_tiff()");
    if (output_tiff) {
        tiff_head(&th, 1);
        fwrite(&th, sizeof th, 1, ofp);
        if (oprof)
            fwrite(oprof, ntohl(oprof[0]), 1, ofp);
    }
    else if (colors > 3)
        fprintf(ofp,
            "P7\nWIDTH %d\nHEIGHT %d\nDEPTH %d\nMAXVAL %d\nTUPLTYPE
%s\nENDHDR\n",
            width, height, colors, (1 << output_bps) - 1, cdesc);
    else
        fprintf(ofp, "P%d\n%d %d\n%d\n",
            colors / 2 + 5, width, height, (1 << output_bps) - 1);
    soff = flip_index(0, 0);
    cstep = flip_index(0, 1) - soff;
    rstep = flip_index(1, 0) - flip_index(0, width);
    for (row = 0; row < height; row++, soff += rstep) {
        for (col = 0; col < width; col++, soff += cstep)
            if (output_bps == 8)
                FORCC ppm[col*colors + c] = curve[image[soff][c]] >> 8;
            else FORCC ppm2[col*colors + c] = curve[image[soff][c]];
            if (output_bps == 16 && !output_tiff && htons(0x55aa) != 0x55aa)
                swab((char *)ppm2, (char *)ppm2, width*colors * 2);
            fwrite(ppm, colors*output_bps / 8, width, ofp);
    }
    free(ppm);
}
*/

int CLASS dcraw_main(int argc, const char **argv, DCRawOutput &output) {
  // initialize();
  int arg, status = 0, quality, i, c;
  int timestamp_only = 0, thumbnail_only = 0, identify_only = 0;
  int user_qual = -1, user_black = -1, user_sat = -1, user_flip = -1;
  int use_fuji_rotate = 1, write_to_stdout = 0, read_from_stdin = 0;
  const char *sp, *bpfile = 0, *dark_frame = 0, *write_ext;
  char opm, opt, *ofname, *cp;
  struct utimbuf ut;
#ifndef NO_LCMS
  const char *cam_profile = 0, *out_profile = 0;
#endif

#ifndef LOCALTIME
  putenv((char *)"TZ=UTC");
#endif
#ifdef LOCALEDIR
  setlocale(LC_CTYPE, "");
  setlocale(LC_MESSAGES, "");
  bindtextdomain("dcraw", LOCALEDIR);
  textdomain("dcraw");
#endif

  if (argc == 1) {
    printf(_("\nRaw photo decoder \"dcraw\" v%s"), DCRAW_VERSION);
    printf(_("\nby Dave Coffin, dcoffin a cybercom o net\n"));
    printf(_("\nUsage:  %s [OPTION]... [FILE]...\n\n"), argv[0]);
    puts(_("-v        Print verbose messages"));
    puts(_("-c        Write image data to standard output"));
    puts(_("-e        Extract embedded thumbnail image"));
    puts(_("-i        Identify files without decoding them"));
    puts(_("-i -v     Identify files and show metadata"));
    puts(_("-z        Change file dates to camera timestamp"));
    puts(_("-w        Use camera white balance, if possible"));
    puts(_("-a        Average the whole image for white balance"));
    puts(_("-A <x y w h> Average a grey box for white balance"));
    puts(_("-r <r g b g> Set custom white balance"));
    puts(_("+M/-M     Use/don't use an embedded color matrix"));
    puts(_("-C <r b>  Correct chromatic aberration"));
    puts(_("-P <file> Fix the dead pixels listed in this file"));
    puts(_("-K <file> Subtract dark frame (16-bit raw PGM)"));
    puts(_("-k <num>  Set the darkness level"));
    puts(_("-S <num>  Set the saturation level"));
    puts(_("-n <num>  Set threshold for wavelet denoising"));
    puts(_("-H [0-9]  Highlight mode (0=clip, 1=unclip, 2=blend, 3+=rebuild)"));
    puts(_("-t [0-7]  Flip image (0=none, 3=180, 5=90CCW, 6=90CW)"));
    puts(_(
        "-o [0-6]  Output colorspace (raw,sRGB,Adobe,Wide,ProPhoto,XYZ,ACES)"));
#ifndef NO_LCMS
    puts(_("-o <file> Apply output ICC profile from file"));
    puts(_("-p <file> Apply camera ICC profile from file or \"embed\""));
#endif
    puts(_("-d        Document mode (no color, no interpolation)"));
    puts(_("-D        Document mode without scaling (totally raw)"));
    puts(_("-j        Don't stretch or rotate raw pixels"));
    puts(_("-W        Don't automatically brighten the image"));
    puts(_("-b <num>  Adjust brightness (default = 1.0)"));
    puts(_("-g <p ts> Set custom gamma curve (default = 2.222 4.5)"));
    puts(_("-q [0-3]  Set the interpolation quality"));
    puts(_("-h        Half-size color image (twice as fast as \"-q 0\")"));
    puts(_("-f        Interpolate RGGB as four colors"));
    puts(_("-m <num>  Apply a 3x3 median filter to R-G and B-G"));
    puts(_("-s [0..N-1] Select one raw image or \"all\" from each file"));
    puts(_("-6        Write 16-bit instead of 8-bit"));
    puts(_("-4        Linear 16-bit, same as \"-6 -W -g 1 1\""));
    puts(_("-T        Write TIFF instead of PPM"));
    puts("");
    return 1;
  }
  argv[argc] = "";
  for (arg = 1; (((opm = argv[arg][0]) - 2) | 2) == '+';) {
    opt = argv[arg++][1];
    if ((cp = (char *)strchr(sp = "nbrkStqmHACg", opt)))
      for (i = 0; i < "114111111422"[cp - sp] - '0'; i++)
        if (!isdigit(argv[arg + i][0])) {
          fprintf(stderr, _("Non-numeric argument to \"-%c\"\n"), opt);
          return 1;
        }
    switch (opt) {
      case 'n':
        threshold = atof(argv[arg++]);
        break;
      case 'b':
        bright = atof(argv[arg++]);
        break;
      case 'r':
        FORC4 user_mul[c] = atof(argv[arg++]);
        break;
      case 'C':
        aber[0] = 1 / atof(argv[arg++]);
        aber[2] = 1 / atof(argv[arg++]);
        break;
      case 'g':
        gamm[0] = atof(argv[arg++]);
        gamm[1] = atof(argv[arg++]);
        if (gamm[0])
          gamm[0] = 1 / gamm[0];
        break;
      case 'k':
        user_black = atoi(argv[arg++]);
        break;
      case 'S':
        user_sat = atoi(argv[arg++]);
        break;
      case 't':
        user_flip = atoi(argv[arg++]);
        break;
      case 'q':
        user_qual = atoi(argv[arg++]);
        break;
      case 'm':
        med_passes = atoi(argv[arg++]);
        break;
      case 'H':
        highlight = atoi(argv[arg++]);
        break;
      case 's':
        shot_select = abs(atoi(argv[arg]));
        multi_out = !strcmp(argv[arg++], "all");
        break;
      case 'o':
        if (isdigit(argv[arg][0]) && !argv[arg][1])
          output_color = atoi(argv[arg++]);
#ifndef NO_LCMS
        else
          out_profile = argv[arg++];
        break;
      case 'p':
        cam_profile = argv[arg++];
#endif
        break;
      case 'P':
        bpfile = argv[arg++];
        break;
      case 'K':
        dark_frame = argv[arg++];
        break;
      case 'z':
        timestamp_only = 1;
        break;
      case 'e':
        thumbnail_only = 1;
        break;
      case 'i':
        identify_only = 1;
        break;
      case 'c':
        write_to_stdout = 1;
        break;
      case 'v':
        verbose = 1;
        break;
      case 'h':
        half_size = 1;
        break;
      case 'f':
        four_color_rgb = 1;
        break;
      case 'A':
        FORC4 greybox[c] = atoi(argv[arg++]);
      case 'a':
        use_auto_wb = 1;
        break;
      case 'w':
        use_camera_wb = 1;
        break;
      case 'M':
        use_camera_matrix = 3 * (opm == '+');
        break;
      case 'I':
        read_from_stdin = 1;
        break;
      case 'E':
        document_mode++;
      case 'D':
        document_mode++;
      case 'd':
        document_mode++;
      case 'j':
        use_fuji_rotate = 0;
        break;
      case 'W':
        no_auto_bright = 1;
        break;
      case 'T':
        output_tiff = 1;
        break;
      case '4':
        gamm[0] = gamm[1] = no_auto_bright = 1;
      case '6':
        output_bps = 16;
        break;
      default:
        fprintf(stderr, _("Unknown option \"-%c\".\n"), opt);
        return 1;
    }
  }
  if (arg == argc) {
    fprintf(stderr, _("No files to process.\n"));
    return 1;
  }
  if (write_to_stdout) {
    // NOTE: deleted.
    /*
    if (isatty(1)) {
        fprintf(stderr, _("Will not write an image to the terminal!\n"));
        return 1;
    }
#if defined(WIN32) || defined(DJGPP) || defined(__CYGWIN__)
    if (setmode(1, O_BINARY) < 0) {
        perror("setmode()");
        return 1;
    }
#endif
    */
  }
  for (; arg < argc; arg++) {
    status = 1;
    raw_image = 0;
    image = 0;
    oprof = 0;
    meta_data = ofname = 0;
    ofp = stdout;
    if (setjmp(failure)) {
      if (fileno(ifp) > 2)
        fclose(ifp);
      if (fileno(ofp) > 2)
        fclose(ofp);
      status = 1;
      goto cleanup;
    }
    ifname = argv[arg];
    if (!(ifp = fopen(ifname, "rb"))) {
      perror(ifname);
      continue;
    }
    status = (identify(), !is_raw);
    if (user_flip >= 0)
      flip = user_flip;
    switch ((flip + 3600) % 360) {
      case 270:
        flip = 5;
        break;
      case 180:
        flip = 3;
        break;
      case 90:
        flip = 6;
    }
    if (timestamp_only) {
      if ((status = !timestamp))
        fprintf(stderr, _("%s has no timestamp.\n"), ifname);
      else if (identify_only)
        printf("%10ld%10d %s\n", (long)timestamp, shot_order, ifname);
      else {
        if (verbose)
          fprintf(stderr, _("%s time set to %d.\n"), ifname, (int)timestamp);
        ut.actime = ut.modtime = timestamp;
        utime(ifname, &ut);
      }
      goto next;
    }
    write_fun = nullptr;
    if (thumbnail_only) {
      if ((status = !thumb_offset)) {
        fprintf(stderr, _("%s has no thumbnail.\n"), ifname);
        goto next;
      } else if (thumb_load_raw) {
        load_raw = thumb_load_raw;
        data_offset = thumb_offset;
        height = thumb_height;
        width = thumb_width;
        filters = 0;
        colors = 3;
      } else {
        fseek(ifp, thumb_offset, SEEK_SET);
        write_fun = write_thumb;
        goto thumbnail;
      }
    }
    if (load_raw == &CLASS kodak_ycbcr_load_raw) {
      height += height & 1;
      width += width & 1;
    }
    if (identify_only && verbose && make[0]) {
      printf(_("\nFilename: %s\n"), ifname);
      printf(_("Timestamp: %s"), ctime(&timestamp));
      printf(_("Camera: %s %s\n"), make, model);
      if (artist[0])
        printf(_("Owner: %s\n"), artist);
      if (dng_version) {
        printf(_("DNG Version: "));
        for (i = 24; i >= 0; i -= 8)
          printf("%d%c", dng_version >> i & 255, i ? '.' : '\n');
      }
      printf(_("ISO speed: %d\n"), (int)iso_speed);
      printf(_("Shutter: "));
      if (shutter > 0 && shutter < 1)
        shutter = (printf("1/"), 1 / shutter);
      printf(_("%0.1f sec\n"), shutter);
      printf(_("Aperture: f/%0.1f\n"), aperture);
      printf(_("Focal length: %0.1f mm\n"), focal_len);
      printf(_("Embedded ICC profile: %s\n"),
             profile_length ? _("yes") : _("no"));
      printf(_("Number of raw images: %d\n"), is_raw);
      if (pixel_aspect != 1)
        printf(_("Pixel Aspect Ratio: %0.6f\n"), pixel_aspect);
      if (thumb_offset)
        printf(_("Thumb size:  %4d x %d\n"), thumb_width, thumb_height);
      printf(_("Full size:   %4d x %d\n"), raw_width, raw_height);
    } else if (!is_raw)
      fprintf(stderr, _("Cannot decode file %s\n"), ifname);
    if (!is_raw)
      goto next;
    shrink = filters &&
             (half_size ||
              (!identify_only && (threshold || aber[0] != 1 || aber[2] != 1)));
    iheight = (height + shrink) >> shrink;
    iwidth = (width + shrink) >> shrink;
    if (identify_only) {
      if (verbose) {
        if (document_mode == 3) {
          top_margin = left_margin = fuji_width = 0;
          height = raw_height;
          width = raw_width;
        }
        iheight = (height + shrink) >> shrink;
        iwidth = (width + shrink) >> shrink;
        if (use_fuji_rotate) {
          if (fuji_width) {
            fuji_width = (fuji_width - 1 + shrink) >> shrink;
            iwidth = fuji_width / sqrt(0.5);
            iheight = (iheight - fuji_width) / sqrt(0.5);
          } else {
            if (pixel_aspect < 1)
              iheight = iheight / pixel_aspect + 0.5;
            if (pixel_aspect > 1)
              iwidth = iwidth * pixel_aspect + 0.5;
          }
        }
        if (flip & 4)
          SWAP(iheight, iwidth);
        printf(_("Image size:  %4d x %d\n"), width, height);
        printf(_("Output size: %4d x %d\n"), iwidth, iheight);
        printf(_("Raw colors: %d"), colors);
        if (filters) {
          int fhigh = 2, fwide = 2;
          if ((filters ^ (filters >> 8)) & 0xff)
            fhigh = 4;
          if ((filters ^ (filters >> 16)) & 0xffff)
            fhigh = 8;
          if (filters == 1)
            fhigh = fwide = 16;
          if (filters == 9)
            fhigh = fwide = 6;
          printf(_("\nFilter pattern: "));
          for (i = 0; i < fhigh; i++)
            for (c = i && putchar('/') && 0; c < fwide; c++)
              putchar(cdesc[fcol(i, c)]);
        }
        printf(_("\nDaylight multipliers:"));
        FORCC printf(" %f", pre_mul[c]);
        if (cam_mul[0] > 0) {
          printf(_("\nCamera multipliers:"));
          FORC4 printf(" %f", cam_mul[c]);
        }
        putchar('\n');
      } else
        printf(_("%s is a %s %s image.\n"), ifname, make, model);
    next:
      fclose(ifp);
      continue;
    }
    if (meta_length) {
      meta_data = (char *)malloc(meta_length);
      merror(meta_data, "main()");
    }
    if (filters || colors == 1) {
      raw_image = (ushort *)calloc((raw_height + 7), raw_width * 2);
      merror(raw_image, "main()");
    } else {
      image = (ushort(*)[4])calloc(iheight, iwidth * sizeof *image);
      merror(image, "main()");
    }
    if (verbose)
      fprintf(stderr, _("Loading %s %s image from %s ...\n"), make, model,
              ifname);
    if (shot_select >= is_raw)
      fprintf(stderr, _("%s: \"-s %d\" requests a nonexistent image!\n"),
              ifname, shot_select);
    fseeko(ifp, data_offset, SEEK_SET);
    if (raw_image && read_from_stdin)
      fread(raw_image, 2, raw_height * raw_width, stdin);
    else
      (*load_raw)();
    if (document_mode == 3) {
      top_margin = left_margin = fuji_width = 0;
      height = raw_height;
      width = raw_width;
    }
    iheight = (height + shrink) >> shrink;
    iwidth = (width + shrink) >> shrink;
    if (raw_image) {
      image = (ushort(*)[4])calloc(iheight, iwidth * sizeof *image);
      merror(image, "main()");
      crop_masked_pixels();
      free(raw_image);
    }
    if (zero_is_bad)
      remove_zeroes();
    bad_pixels(bpfile);
    if (dark_frame)
      subtract(dark_frame);
    quality = 2 + !fuji_width;
    if (user_qual >= 0)
      quality = user_qual;
    i = cblack[3];
    FORC3 if (i > cblack[c]) i = cblack[c];
    FORC4 cblack[c] -= i;
    black += i;
    i = cblack[6];
    FORC(cblack[4] * cblack[5])
    if (i > cblack[6 + c])
      i = cblack[6 + c];
    FORC(cblack[4] * cblack[5])
    cblack[6 + c] -= i;
    black += i;
    if (user_black >= 0)
      black = user_black;
    FORC4 cblack[c] += black;
    if (user_sat > 0)
      maximum = user_sat;
#ifdef COLORCHECK
    colorcheck();
#endif
    if (is_foveon) {
      if (document_mode || load_raw == &CLASS foveon_dp_load_raw) {
        for (i = 0; i < height * width * 4; i++)
          if ((short)image[0][i] < 0)
            image[0][i] = 0;
      } else
        foveon_interpolate();
    } else if (document_mode < 2)
      scale_colors();
    pre_interpolate();
    if (filters && !document_mode) {
      if (quality == 0)
        lin_interpolate();
      else if (quality == 1 || colors > 3)
        vng_interpolate();
      else if (quality == 2 && filters > 1000)
        ppg_interpolate();
      else if (filters == 9)
        xtrans_interpolate(quality * 2 - 3);
      else
        ahd_interpolate();
    }
    if (mix_green)
      for (colors = 3, i = 0; i < height * width; i++)
        image[i][1] = (image[i][1] + image[i][3]) >> 1;
    if (!is_foveon && colors == 3)
      median_filter();
    if (!is_foveon && highlight == 2)
      blend_highlights();
    if (!is_foveon && highlight > 2)
      recover_highlights();
    if (use_fuji_rotate)
      fuji_rotate();
#ifndef NO_LCMS
    if (cam_profile)
      apply_profile(cam_profile, out_profile);
#endif
    convert_to_rgb();
    if (use_fuji_rotate)
      stretch();
  thumbnail:
    if (write_fun == &CLASS jpeg_thumb)
      write_ext = ".jpg";
    else if (output_tiff && write_fun == nullptr)
      write_ext = ".tiff";
    else
      write_ext = &(".pgm\0.ppm\0.ppm\0.pam"[colors * 5 - 5]);
    ofname = (char *)malloc(strlen(ifname) + 64);
    merror(ofname, "main()");
    if (write_to_stdout)
      strcpy(ofname, _("standard output"));
    else {
      strcpy(ofname, ifname);
      if ((cp = strrchr(ofname, '.')))
        *cp = 0;
      if (multi_out)
        sprintf(ofname + strlen(ofname), "_%0*d",
                snprintf(0, 0, "%d", is_raw - 1), shot_select);
      if (thumbnail_only)
        strcat(ofname, ".thumb");
      strcat(ofname, write_ext);
      ofp = fopen(ofname, "wb");
      if (!ofp) {
        status = 1;
        perror(ofname);
        goto cleanup;
      }
    }
    if (verbose)
      fprintf(stderr, _("Writing data to %s ...\n"), ofname);
    //(*write_fun)();
    write_ppm_tiff(output);
    fclose(ifp);
    if (ofp != stdout)
      fclose(ofp);
  cleanup:
    if (meta_data)
      free(meta_data);
    if (ofname)
      free(ofname);
    if (oprof)
      free(oprof);
    if (image)
      free(image);
    if (multi_out) {
      if (++shot_select < is_raw)
        arg--;
      else
        shot_select = 0;
    }
  }
  return status;
}

#pragma warning(pop)
