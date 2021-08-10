/* srotm.f -- translated by f2c (version 20100827).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "datatypes.h"

/* Subroutine */ int srotm_(integer *n, real *sx, integer *incx, real *sy, 
	integer *incy, real *sparam)
{
    /* Initialized data */

    static real zero = 0.f;
    static real two = 2.f;

    /* System generated locals */
    integer i__1, i__2;

    /* Local variables */
    integer i__;
    real w, z__;
    integer kx, ky;
    real sh11, sh12, sh21, sh22, sflag;
    integer nsteps;

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*     APPLY THE MODIFIED GIVENS TRANSFORMATION, H, TO THE 2 BY N MATRIX */

/*     (SX**T) , WHERE **T INDICATES TRANSPOSE. THE ELEMENTS OF SX ARE IN */
/*     (DX**T) */

/*     SX(LX+I*INCX), I = 0 TO N-1, WHERE LX = 1 IF INCX .GE. 0, ELSE */
/*     LX = (-INCX)*N, AND SIMILARLY FOR SY USING USING LY AND INCY. */
/*     WITH SPARAM(1)=SFLAG, H HAS ONE OF THE FOLLOWING FORMS.. */

/*     SFLAG=-1.E0     SFLAG=0.E0        SFLAG=1.E0     SFLAG=-2.E0 */

/*       (SH11  SH12)    (1.E0  SH12)    (SH11  1.E0)    (1.E0  0.E0) */
/*     H=(          )    (          )    (          )    (          ) */
/*       (SH21  SH22),   (SH21  1.E0),   (-1.E0 SH22),   (0.E0  1.E0). */
/*     SEE  SROTMG FOR A DESCRIPTION OF DATA STORAGE IN SPARAM. */


/*  Arguments */
/*  ========= */

/*  N      (input) INTEGER */
/*         number of elements in input vector(s) */

/*  SX     (input/output) REAL array, dimension N */
/*         double precision vector with N elements */

/*  INCX   (input) INTEGER */
/*         storage spacing between elements of SX */

/*  SY     (input/output) REAL array, dimension N */
/*         double precision vector with N elements */

/*  INCY   (input) INTEGER */
/*         storage spacing between elements of SY */

/*  SPARAM (input/output)  REAL array, dimension 5 */
/*     SPARAM(1)=SFLAG */
/*     SPARAM(2)=SH11 */
/*     SPARAM(3)=SH21 */
/*     SPARAM(4)=SH12 */
/*     SPARAM(5)=SH22 */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. Data statements .. */
    /* Parameter adjustments */
    --sparam;
    --sy;
    --sx;

    /* Function Body */
/*     .. */

    sflag = sparam[1];
    if (*n <= 0 || sflag + two == zero) {
	goto L140;
    }
    if (! (*incx == *incy && *incx > 0)) {
	goto L70;
    }

    nsteps = *n * *incx;
    if (sflag < 0.f) {
	goto L50;
    } else if (sflag == 0) {
	goto L10;
    } else {
	goto L30;
    }
L10:
    sh12 = sparam[4];
    sh21 = sparam[3];
    i__1 = nsteps;
    i__2 = *incx;
    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	w = sx[i__];
	z__ = sy[i__];
	sx[i__] = w + z__ * sh12;
	sy[i__] = w * sh21 + z__;
/* L20: */
    }
    goto L140;
L30:
    sh11 = sparam[2];
    sh22 = sparam[5];
    i__2 = nsteps;
    i__1 = *incx;
    for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += i__1) {
	w = sx[i__];
	z__ = sy[i__];
	sx[i__] = w * sh11 + z__;
	sy[i__] = -w + sh22 * z__;
/* L40: */
    }
    goto L140;
L50:
    sh11 = sparam[2];
    sh12 = sparam[4];
    sh21 = sparam[3];
    sh22 = sparam[5];
    i__1 = nsteps;
    i__2 = *incx;
    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	w = sx[i__];
	z__ = sy[i__];
	sx[i__] = w * sh11 + z__ * sh12;
	sy[i__] = w * sh21 + z__ * sh22;
/* L60: */
    }
    goto L140;
L70:
    kx = 1;
    ky = 1;
    if (*incx < 0) {
	kx = (1 - *n) * *incx + 1;
    }
    if (*incy < 0) {
	ky = (1 - *n) * *incy + 1;
    }

    if (sflag < 0.f) {
	goto L120;
    } else if (sflag == 0) {
	goto L80;
    } else {
	goto L100;
    }
L80:
    sh12 = sparam[4];
    sh21 = sparam[3];
    i__2 = *n;
    for (i__ = 1; i__ <= i__2; ++i__) {
	w = sx[kx];
	z__ = sy[ky];
	sx[kx] = w + z__ * sh12;
	sy[ky] = w * sh21 + z__;
	kx += *incx;
	ky += *incy;
/* L90: */
    }
    goto L140;
L100:
    sh11 = sparam[2];
    sh22 = sparam[5];
    i__2 = *n;
    for (i__ = 1; i__ <= i__2; ++i__) {
	w = sx[kx];
	z__ = sy[ky];
	sx[kx] = w * sh11 + z__;
	sy[ky] = -w + sh22 * z__;
	kx += *incx;
	ky += *incy;
/* L110: */
    }
    goto L140;
L120:
    sh11 = sparam[2];
    sh12 = sparam[4];
    sh21 = sparam[3];
    sh22 = sparam[5];
    i__2 = *n;
    for (i__ = 1; i__ <= i__2; ++i__) {
	w = sx[kx];
	z__ = sy[ky];
	sx[kx] = w * sh11 + z__ * sh12;
	sy[ky] = w * sh21 + z__ * sh22;
	kx += *incx;
	ky += *incy;
/* L130: */
    }
L140:
    return 0;
} /* srotm_ */

