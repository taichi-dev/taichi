/* srotmg.f -- translated by f2c (version 20100827).
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

/* Subroutine */ int srotmg_(real *sd1, real *sd2, real *sx1, real *sy1, real 
	*sparam)
{
    /* Initialized data */

    static real zero = 0.f;
    static real one = 1.f;
    static real two = 2.f;
    static real gam = 4096.f;
    static real gamsq = 16777200.f;
    static real rgamsq = 5.96046e-8f;

    /* Format strings */
    static char fmt_120[] = "";
    static char fmt_150[] = "";
    static char fmt_180[] = "";
    static char fmt_210[] = "";

    /* System generated locals */
    real r__1;

    /* Local variables */
    real su, sp1, sp2, sq1, sq2, sh11, sh12, sh21, sh22;
    integer igo;
    real sflag, stemp;

    /* Assigned format variables */
    static char *igo_fmt;

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*     CONSTRUCT THE MODIFIED GIVENS TRANSFORMATION MATRIX H WHICH ZEROS */
/*     THE SECOND COMPONENT OF THE 2-VECTOR  (SQRT(SD1)*SX1,SQRT(SD2)* */
/*     SY2)**T. */
/*     WITH SPARAM(1)=SFLAG, H HAS ONE OF THE FOLLOWING FORMS.. */

/*     SFLAG=-1.E0     SFLAG=0.E0        SFLAG=1.E0     SFLAG=-2.E0 */

/*       (SH11  SH12)    (1.E0  SH12)    (SH11  1.E0)    (1.E0  0.E0) */
/*     H=(          )    (          )    (          )    (          ) */
/*       (SH21  SH22),   (SH21  1.E0),   (-1.E0 SH22),   (0.E0  1.E0). */
/*     LOCATIONS 2-4 OF SPARAM CONTAIN SH11,SH21,SH12, AND SH22 */
/*     RESPECTIVELY. (VALUES OF 1.E0, -1.E0, OR 0.E0 IMPLIED BY THE */
/*     VALUE OF SPARAM(1) ARE NOT STORED IN SPARAM.) */

/*     THE VALUES OF GAMSQ AND RGAMSQ SET IN THE DATA STATEMENT MAY BE */
/*     INEXACT.  THIS IS OK AS THEY ARE ONLY USED FOR TESTING THE SIZE */
/*     OF SD1 AND SD2.  ALL ACTUAL SCALING OF DATA IS DONE USING GAM. */


/*  Arguments */
/*  ========= */


/*  SD1    (input/output) REAL */

/*  SD2    (input/output) REAL */

/*  SX1    (input/output) REAL */

/*  SY1    (input) REAL */


/*  SPARAM (input/output)  REAL array, dimension 5 */
/*     SPARAM(1)=SFLAG */
/*     SPARAM(2)=SH11 */
/*     SPARAM(3)=SH21 */
/*     SPARAM(4)=SH12 */
/*     SPARAM(5)=SH22 */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Data statements .. */

    /* Parameter adjustments */
    --sparam;

    /* Function Body */
/*     .. */
    if (! (*sd1 < zero)) {
	goto L10;
    }
/*       GO ZERO-H-D-AND-SX1.. */
    goto L60;
L10:
/*     CASE-SD1-NONNEGATIVE */
    sp2 = *sd2 * *sy1;
    if (! (sp2 == zero)) {
	goto L20;
    }
    sflag = -two;
    goto L260;
/*     REGULAR-CASE.. */
L20:
    sp1 = *sd1 * *sx1;
    sq2 = sp2 * *sy1;
    sq1 = sp1 * *sx1;

    if (! (dabs(sq1) > dabs(sq2))) {
	goto L40;
    }
    sh21 = -(*sy1) / *sx1;
    sh12 = sp2 / sp1;

    su = one - sh12 * sh21;

    if (! (su <= zero)) {
	goto L30;
    }
/*         GO ZERO-H-D-AND-SX1.. */
    goto L60;
L30:
    sflag = zero;
    *sd1 /= su;
    *sd2 /= su;
    *sx1 *= su;
/*         GO SCALE-CHECK.. */
    goto L100;
L40:
    if (! (sq2 < zero)) {
	goto L50;
    }
/*         GO ZERO-H-D-AND-SX1.. */
    goto L60;
L50:
    sflag = one;
    sh11 = sp1 / sp2;
    sh22 = *sx1 / *sy1;
    su = one + sh11 * sh22;
    stemp = *sd2 / su;
    *sd2 = *sd1 / su;
    *sd1 = stemp;
    *sx1 = *sy1 * su;
/*         GO SCALE-CHECK */
    goto L100;
/*     PROCEDURE..ZERO-H-D-AND-SX1.. */
L60:
    sflag = -one;
    sh11 = zero;
    sh12 = zero;
    sh21 = zero;
    sh22 = zero;

    *sd1 = zero;
    *sd2 = zero;
    *sx1 = zero;
/*         RETURN.. */
    goto L220;
/*     PROCEDURE..FIX-H.. */
L70:
    if (! (sflag >= zero)) {
	goto L90;
    }

    if (! (sflag == zero)) {
	goto L80;
    }
    sh11 = one;
    sh22 = one;
    sflag = -one;
    goto L90;
L80:
    sh21 = -one;
    sh12 = one;
    sflag = -one;
L90:
    switch (igo) {
	case 0: goto L120;
	case 1: goto L150;
	case 2: goto L180;
	case 3: goto L210;
    }
/*     PROCEDURE..SCALE-CHECK */
L100:
L110:
    if (! (*sd1 <= rgamsq)) {
	goto L130;
    }
    if (*sd1 == zero) {
	goto L160;
    }
    igo = 0;
    igo_fmt = fmt_120;
/*              FIX-H.. */
    goto L70;
L120:
/* Computing 2nd power */
    r__1 = gam;
    *sd1 *= r__1 * r__1;
    *sx1 /= gam;
    sh11 /= gam;
    sh12 /= gam;
    goto L110;
L130:
L140:
    if (! (*sd1 >= gamsq)) {
	goto L160;
    }
    igo = 1;
    igo_fmt = fmt_150;
/*              FIX-H.. */
    goto L70;
L150:
/* Computing 2nd power */
    r__1 = gam;
    *sd1 /= r__1 * r__1;
    *sx1 *= gam;
    sh11 *= gam;
    sh12 *= gam;
    goto L140;
L160:
L170:
    if (! (dabs(*sd2) <= rgamsq)) {
	goto L190;
    }
    if (*sd2 == zero) {
	goto L220;
    }
    igo = 2;
    igo_fmt = fmt_180;
/*              FIX-H.. */
    goto L70;
L180:
/* Computing 2nd power */
    r__1 = gam;
    *sd2 *= r__1 * r__1;
    sh21 /= gam;
    sh22 /= gam;
    goto L170;
L190:
L200:
    if (! (dabs(*sd2) >= gamsq)) {
	goto L220;
    }
    igo = 3;
    igo_fmt = fmt_210;
/*              FIX-H.. */
    goto L70;
L210:
/* Computing 2nd power */
    r__1 = gam;
    *sd2 /= r__1 * r__1;
    sh21 *= gam;
    sh22 *= gam;
    goto L200;
L220:
    if (sflag < 0.f) {
	goto L250;
    } else if (sflag == 0) {
	goto L230;
    } else {
	goto L240;
    }
L230:
    sparam[3] = sh21;
    sparam[4] = sh12;
    goto L260;
L240:
    sparam[2] = sh11;
    sparam[5] = sh22;
    goto L260;
L250:
    sparam[2] = sh11;
    sparam[3] = sh21;
    sparam[4] = sh12;
    sparam[5] = sh22;
L260:
    sparam[1] = sflag;
    return 0;
} /* srotmg_ */

