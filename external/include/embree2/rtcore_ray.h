// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#ifndef __RTCORE_RAY_H__
#define __RTCORE_RAY_H__

#include "rtcore.h"

/*! \ingroup embree_kernel_api */
/*! \{ */

/*! \brief Ray structure for an individual ray */
#ifndef __RTCRay__
#define __RTCRay__
struct RTCORE_ALIGN(16)  RTCRay
{
  /* ray data */
public:
  float org[3];      //!< Ray origin
  float align0;
  
  float dir[3];      //!< Ray direction
  float align1;
  
  float tnear;       //!< Start of ray segment
  float tfar;        //!< End of ray segment (set to hit distance)

  float time;        //!< Time of this ray for motion blur
  unsigned mask;        //!< Used to mask out objects during traversal
  
  /* hit data */
public:
  float Ng[3];       //!< Unnormalized geometry normal
  float align2;
  
  float u;           //!< Barycentric u coordinate of hit
  float v;           //!< Barycentric v coordinate of hit

  unsigned geomID;        //!< geometry ID
  unsigned primID;        //!< primitive ID
  unsigned instID;        //!< instance ID
};
#endif

/*! Ray structure for packets of 4 rays. */
#ifndef __RTCRay4__
#define __RTCRay4__
struct RTCORE_ALIGN(16) RTCRay4
{
  /* ray data */
public:
  float orgx[4];  //!< x coordinate of ray origin
  float orgy[4];  //!< y coordinate of ray origin
  float orgz[4];  //!< z coordinate of ray origin
  
  float dirx[4];  //!< x coordinate of ray direction
  float diry[4];  //!< y coordinate of ray direction
  float dirz[4];  //!< z coordinate of ray direction
  
  float tnear[4]; //!< Start of ray segment 
  float tfar[4];  //!< End of ray segment (set to hit distance)

  float time[4];  //!< Time of this ray for motion blur
  unsigned mask[4];  //!< Used to mask out objects during traversal
  
  /* hit data */
public:
  float Ngx[4];   //!< x coordinate of geometry normal
  float Ngy[4];   //!< y coordinate of geometry normal
  float Ngz[4];   //!< z coordinate of geometry normal
  
  float u[4];     //!< Barycentric u coordinate of hit
  float v[4];     //!< Barycentric v coordinate of hit
  
  unsigned geomID[4];  //!< geometry ID
  unsigned primID[4];  //!< primitive ID
  unsigned instID[4];  //!< instance ID
};
#endif

/*! Ray structure for packets of 8 rays. */
#ifndef __RTCRay8__
#define __RTCRay8__
struct RTCORE_ALIGN(32) RTCRay8
{
  /* ray data */
public:
  float orgx[8];  //!< x coordinate of ray origin
  float orgy[8];  //!< y coordinate of ray origin
  float orgz[8];  //!< z coordinate of ray origin
  
  float dirx[8];  //!< x coordinate of ray direction
  float diry[8];  //!< y coordinate of ray direction
  float dirz[8];  //!< z coordinate of ray direction
  
  float tnear[8]; //!< Start of ray segment 
  float tfar[8];  //!< End of ray segment (set to hit distance)

  float time[8];  //!< Time of this ray for motion blur
  unsigned mask[8];  //!< Used to mask out objects during traversal
  
  /* hit data */
public:
  float Ngx[8];   //!< x coordinate of geometry normal
  float Ngy[8];   //!< y coordinate of geometry normal
  float Ngz[8];   //!< z coordinate of geometry normal
  
  float u[8];     //!< Barycentric u coordinate of hit
  float v[8];     //!< Barycentric v coordinate of hit
  
  unsigned geomID[8];  //!< geometry ID
  unsigned primID[8];  //!< primitive ID
  unsigned instID[8];  //!< instance ID
};
#endif

/*! \brief Ray structure for packets of 16 rays. */
#ifndef __RTCRay16__
#define __RTCRay16__
struct RTCORE_ALIGN(64) RTCRay16
{
  /* ray data */
public:
  float orgx[16];  //!< x coordinate of ray origin
  float orgy[16];  //!< y coordinate of ray origin
  float orgz[16];  //!< z coordinate of ray origin
  
  float dirx[16];  //!< x coordinate of ray direction
  float diry[16];  //!< y coordinate of ray direction
  float dirz[16];  //!< z coordinate of ray direction
  
  float tnear[16]; //!< Start of ray segment 
  float tfar[16];  //!< End of ray segment (set to hit distance)

  float time[16];  //!< Time of this ray for motion blur
  unsigned mask[16];  //!< Used to mask out objects during traversal
  
  /* hit data */
public:
  float Ngx[16];   //!< x coordinate of geometry normal
  float Ngy[16];   //!< y coordinate of geometry normal
  float Ngz[16];   //!< z coordinate of geometry normal
  
  float u[16];     //!< Barycentric u coordinate of hit
  float v[16];     //!< Barycentric v coordinate of hit
  
  unsigned geomID[16];  //!< geometry ID
  unsigned primID[16];  //!< primitive ID
  unsigned instID[16];  //!< instance ID
};
#endif

/* Helper functions to access ray packets of runtime size N */
#ifndef __RTCRayN__
#define __RTCRayN__
struct RTCRayN {};
RTCORE_FORCEINLINE float& RTCRayN_org_x(RTCRayN* ptr, size_t N, size_t i) { const size_t N1 = (size_t)(N == 1); return ((float*)ptr)[0*N+0*N1+i]; }  //!< x coordinate of ray origin
RTCORE_FORCEINLINE float& RTCRayN_org_y(RTCRayN* ptr, size_t N, size_t i) { const size_t N1 = (size_t)(N == 1); return ((float*)ptr)[1*N+0*N1+i]; }  //!< y coordinate of ray origin
RTCORE_FORCEINLINE float& RTCRayN_org_z(RTCRayN* ptr, size_t N, size_t i) { const size_t N1 = (size_t)(N == 1); return ((float*)ptr)[2*N+0*N1+i]; }  //!< z coordinate of ray origin

RTCORE_FORCEINLINE float& RTCRayN_dir_x(RTCRayN* ptr, size_t N, size_t i) { const size_t N1 = (size_t)(N == 1); return ((float*)ptr)[3*N+1*N1+i]; }  //!< x coordinate of ray direction
RTCORE_FORCEINLINE float& RTCRayN_dir_y(RTCRayN* ptr, size_t N, size_t i) { const size_t N1 = (size_t)(N == 1); return ((float*)ptr)[4*N+1*N1+i]; }  //!< y coordinate of ray direction
RTCORE_FORCEINLINE float& RTCRayN_dir_z(RTCRayN* ptr, size_t N, size_t i) { const size_t N1 = (size_t)(N == 1); return ((float*)ptr)[5*N+1*N1+i]; }  //!< z coordinate of ray direction

RTCORE_FORCEINLINE float& RTCRayN_tnear(RTCRayN* ptr, size_t N, size_t i) { const size_t N1 = (size_t)(N == 1); return ((float*)ptr)[6*N+2*N1+i]; }  //!< Start of ray segment 
RTCORE_FORCEINLINE float& RTCRayN_tfar (RTCRayN* ptr, size_t N, size_t i) { const size_t N1 = (size_t)(N == 1); return ((float*)ptr)[7*N+2*N1+i]; }  //!< End of ray segment (set to hit distance)

RTCORE_FORCEINLINE float&    RTCRayN_time(RTCRayN* ptr, size_t N, size_t i) { const size_t N1 = (size_t)(N == 1); return ((float*)   ptr)[8*N+2*N1+i]; }   //!< Time of this ray for motion blur 
RTCORE_FORCEINLINE unsigned& RTCRayN_mask(RTCRayN* ptr, size_t N, size_t i) { const size_t N1 = (size_t)(N == 1); return ((unsigned*)ptr)[9*N+2*N1+i]; }   //!< Used to mask out objects during traversal

RTCORE_FORCEINLINE float& RTCRayN_Ng_x(RTCRayN* ptr, size_t N, size_t i) { const size_t N1 = (size_t)(N == 1); return ((float*)ptr)[10*N+2*N1+i]; }  //!< x coordinate of geometry normal
RTCORE_FORCEINLINE float& RTCRayN_Ng_y(RTCRayN* ptr, size_t N, size_t i) { const size_t N1 = (size_t)(N == 1); return ((float*)ptr)[11*N+2*N1+i]; }  //!< y coordinate of geometry normal
RTCORE_FORCEINLINE float& RTCRayN_Ng_z(RTCRayN* ptr, size_t N, size_t i) { const size_t N1 = (size_t)(N == 1); return ((float*)ptr)[12*N+2*N1+i]; }  //!< z coordinate of geometry normal

RTCORE_FORCEINLINE float& RTCRayN_u   (RTCRayN* ptr, size_t N, size_t i) { const size_t N1 = (size_t)(N == 1); return ((float*)ptr)[13*N+3*N1+i]; }  //!< Barycentric u coordinate of hit
RTCORE_FORCEINLINE float& RTCRayN_v   (RTCRayN* ptr, size_t N, size_t i) { const size_t N1 = (size_t)(N == 1); return ((float*)ptr)[14*N+3*N1+i]; }  //!< Barycentric v coordinate of hit

RTCORE_FORCEINLINE unsigned& RTCRayN_geomID(RTCRayN* ptr, size_t N, size_t i) { const size_t N1 = (size_t)(N == 1); return ((unsigned*)ptr)[15*N+3*N1+i]; }  //!< geometry ID
RTCORE_FORCEINLINE unsigned& RTCRayN_primID(RTCRayN* ptr, size_t N, size_t i) { const size_t N1 = (size_t)(N == 1); return ((unsigned*)ptr)[16*N+3*N1+i]; }  //!< primitive ID
RTCORE_FORCEINLINE unsigned& RTCRayN_instID(RTCRayN* ptr, size_t N, size_t i) { const size_t N1 = (size_t)(N == 1); return ((unsigned*)ptr)[17*N+3*N1+i]; }  //!< instance ID
#endif

/* Helper structure to create a ray packet of compile time size N */
#ifndef __RTCRayNt__
#define __RTCRayNt__
template<int N>
struct RTCRayNt : public RTCRayN
{
  /* ray data */
public:
  float orgx[N];  //!< x coordinate of ray origin
  float orgy[N];  //!< y coordinate of ray origin
  float orgz[N];  //!< z coordinate of ray origin
  
  float dirx[N];  //!< x coordinate of ray direction
  float diry[N];  //!< y coordinate of ray direction
  float dirz[N];  //!< z coordinate of ray direction
  
  float tnear[N]; //!< Start of ray segment 
  float tfar[N];  //!< End of ray segment (set to hit distance)

  float time[N];  //!< Time of this ray for motion blur
  unsigned mask[N];  //!< Used to mask out objects during traversal
  
  /* hit data */
public:
  float Ngx[N];   //!< x coordinate of geometry normal
  float Ngy[N];   //!< y coordinate of geometry normal
  float Ngz[N];   //!< z coordinate of geometry normal
  
  float u[N];     //!< Barycentric u coordinate of hit
  float v[N];     //!< Barycentric v coordinate of hit
  
  unsigned geomID[N];  //!< geometry ID
  unsigned primID[N];  //!< primitive ID
  unsigned instID[N];  //!< instance ID
};
#endif

/*! \brief Ray structure template for packets of N rays in pointer SOA layout. */
#ifndef __RTCRayNp__
#define __RTCRayNp__
struct RTCRayNp
{
  /* ray data */
public:

  float* orgx;  //!< x coordinate of ray origin
  float* orgy;  //!< y coordinate of ray origin
  float* orgz;  //!< z coordinate of ray origin

  float* dirx;  //!< x coordinate of ray direction
  float* diry;  //!< y coordinate of ray direction
  float* dirz;  //!< z coordinate of ray direction
  
  float* tnear; //!< Start of ray segment (optional)
  float* tfar;  //!< End of ray segment (set to hit distance)
 
  float* time;  //!< Time of this ray for motion blur (optional)
  unsigned* mask;  //!< Used to mask out objects during traversal (optional)

  /* hit data */
public:

  float* Ngx;   //!< x coordinate of geometry normal (optional)
  float* Ngy;   //!< y coordinate of geometry normal (optional)
  float* Ngz;   //!< z coordinate of geometry normal (optional)

  float* u;     //!< Barycentric u coordinate of hit
  float* v;     //!< Barycentric v coordinate of hit
 
  unsigned* geomID;  //!< geometry ID
  unsigned* primID;  //!< primitive ID
  unsigned* instID;  //!< instance ID (optional)
};
#endif

/* Helper functions to access hit packets of size N */
#ifndef __RTCHitN__
#define __RTCHitN__
struct RTCHitN {};
RTCORE_FORCEINLINE float& RTCHitN_Ng_x(const RTCHitN* ptr, size_t N, size_t i) { return ((float*)ptr)[0*N+i]; }  //!< x coordinate of geometry normal
RTCORE_FORCEINLINE float& RTCHitN_Ng_y(const RTCHitN* ptr, size_t N, size_t i) { return ((float*)ptr)[1*N+i]; }  //!< y coordinate of geometry normal
RTCORE_FORCEINLINE float& RTCHitN_Ng_z(const RTCHitN* ptr, size_t N, size_t i) { return ((float*)ptr)[2*N+i]; }  //!< z coordinate of geometry normal

RTCORE_FORCEINLINE unsigned& RTCHitN_instID(const RTCHitN* ptr, size_t N, size_t i) { return ((unsigned*)ptr)[3*N+i]; }  //!< instance ID
RTCORE_FORCEINLINE unsigned& RTCHitN_geomID(const RTCHitN* ptr, size_t N, size_t i) { return ((unsigned*)ptr)[4*N+i]; }  //!< geometry ID
RTCORE_FORCEINLINE unsigned& RTCHitN_primID(const RTCHitN* ptr, size_t N, size_t i) { return ((unsigned*)ptr)[5*N+i]; }  //!< primitive ID

RTCORE_FORCEINLINE float& RTCHitN_u   (const RTCHitN* ptr, size_t N, size_t i) { return ((float*)ptr)[6*N+i]; } //!< Barycentric u coordinate of hit
RTCORE_FORCEINLINE float& RTCHitN_v   (const RTCHitN* ptr, size_t N, size_t i) { return ((float*)ptr)[7*N+i]; } //!< Barycentric v coordinate of hit
RTCORE_FORCEINLINE float& RTCHitN_t   (const RTCHitN* ptr, size_t N, size_t i) { return ((float*)ptr)[8*N+i]; } //!< hit distance
#endif

/* Helper structure to create a hit packet of compile time size N */
#ifndef __RTCHitNt__
#define __RTCHitNt__
template<int N>
struct RTCHitNt : public RTCHitN
{
  float Ngx[N];        //!< x coordinate of geometry normal
  float Ngy[N];        //!< y coordinate of geometry normal
  float Ngz[N];        //!< z coordinate of geometry normal

  unsigned instID[N];  //!< instance ID
  unsigned geomID[N];  //!< geometry ID
  unsigned primID[N];  //!< primitive ID

  float u[N];          //!< Barycentric u coordinate of hit
  float v[N];          //!< Barycentric v coordinate of hit
  float t[N];          //!< hit distance
};
#endif

/*! @} */

#endif
