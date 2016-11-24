// ======================================================================== //
// Copyright 2009-2015 Intel Corporation                                    //
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

#ifndef __RTCORE_USER_GEOMETRY_H__
#define __RTCORE_USER_GEOMETRY_H__

/*! \ingroup embree_kernel_api */
/*! \{ */

/*! Type of bounding function. */
typedef void (*RTCBoundsFunc)(void* ptr,              /*!< pointer to user data */
                              size_t item,            /*!< item to calculate bounds for */
                              RTCBounds& bounds_o     /*!< returns calculated bounds */);

/*! Type of bounding function. */
typedef void (*RTCBoundsFunc2)(void* userPtr,         /*!< pointer to user data */
                               void* geomUserPtr,     /*!< pointer to geometry user data */
                               size_t item,           /*!< item to calculate bounds for */
                               RTCBounds* bounds_o    /*!< returns calculated bounds */);

/*! Type of intersect function pointer for single rays. */
typedef void (*RTCIntersectFunc)(void* ptr,           /*!< pointer to user data */
                                 RTCRay& ray,         /*!< ray to intersect */
                                 size_t item          /*!< item to intersect */);

/*! Type of intersect function pointer for ray packets of size 4. */
typedef void (*RTCIntersectFunc4)(const void* valid,  /*!< pointer to valid mask */
                                  void* ptr,          /*!< pointer to user data */
                                  RTCRay4& ray,       /*!< ray packet to intersect */
                                  size_t item         /*!< item to intersect */);

/*! Type of intersect function pointer for ray packets of size 8. */
typedef void (*RTCIntersectFunc8)(const void* valid,  /*!< pointer to valid mask */
                                  void* ptr,          /*!< pointer to user data */
                                  RTCRay8& ray,       /*!< ray packet to intersect */
                                  size_t item         /*!< item to intersect */);

/*! Type of intersect function pointer for ray packets of size 16. */
typedef void (*RTCIntersectFunc16)(const void* valid, /*!< pointer to valid mask */
                                   void* ptr,         /*!< pointer to user data */
                                   RTCRay16& ray,     /*!< ray packet to intersect */
                                   size_t item        /*!< item to intersect */);

/*! Type of occlusion function pointer for single rays. */
typedef void (*RTCOccludedFunc) (void* ptr,           /*!< pointer to user data */ 
                                 RTCRay& ray,         /*!< ray to test occlusion */
                                 size_t item          /*!< item to test for occlusion */);

/*! Type of occlusion function pointer for ray packets of size 4. */
typedef void (*RTCOccludedFunc4) (const void* valid,  /*! pointer to valid mask */
                                  void* ptr,          /*!< pointer to user data */
                                  RTCRay4& ray,       /*!< Ray packet to test occlusion. */
                                  size_t item         /*!< item to test for occlusion */);

/*! Type of occlusion function pointer for ray packets of size 8. */
typedef void (*RTCOccludedFunc8) (const void* valid,  /*! pointer to valid mask */
                                  void* ptr,          /*!< pointer to user data */
                                  RTCRay8& ray,       /*!< Ray packet to test occlusion. */
                                  size_t item         /*!< item to test for occlusion */);

/*! Type of occlusion function pointer for ray packets of size 16. */
typedef void (*RTCOccludedFunc16) (const void* valid, /*! pointer to valid mask */
                                   void* ptr,         /*!< pointer to user data */
                                   RTCRay16& ray,     /*!< Ray packet to test occlusion. */
                                   size_t item        /*!< item to test for occlusion */);

/*! Creates a new user geometry object. This feature makes it possible
 *  to add arbitrary types of geometry to the scene by providing
 *  appropiate bounding, intersect and occluded functions. A user
 *  geometry object is a set of user geometries. As the rtcIntersect
 *  and rtcOccluded functions support different ray packet sizes, the
 *  user also has to provide different versions of intersect and
 *  occluded function pointers for these packet sizes. However, the
 *  ray packet size of the called function pointer always matches the
 *  packet size of the originally invoked rtcIntersect and rtcOccluded
 *  functions. A user data pointer, that points to a user specified
 *  representation of the geometry, is passed to each intersect and
 *  occluded function invokation, as well as the index of the geometry
 *  of the set to intersect. */
RTCORE_API unsigned rtcNewUserGeometry (RTCScene scene,           /*!< the scene the user geometry set is created in */
                                        size_t numGeometries      /*!< the number of geometries contained in the set */);

RTCORE_API unsigned rtcNewUserGeometry2 (RTCScene scene,          /*!< the scene the user geometry set is created in */
                                         size_t numGeometries,    /*!< the number of geometries contained in the set */
                                         size_t numTimeSteps = 1  /*!< number of motion blur time steps */);

/*! Sets the bounding function to calculate bounding boxes of the user
 *  geometry items when building spatial index structures. The
 *  calculated bounding box have to be conservative and should be
 *  tight. */
RTCORE_API void rtcSetBoundsFunction (RTCScene scene, unsigned geomID, RTCBoundsFunc bounds);

/*! Sets the bounding function to calculate bounding boxes of the user
 *  geometry items when building spatial index structures. The
 *  calculated bounding box have to be conservative and should be
 *  tight. */
RTCORE_API void rtcSetBoundsFunction2 (RTCScene scene, unsigned geomID, RTCBoundsFunc2 bounds, void* userPtr);

/*! Set intersect function for single rays. The rtcIntersect function
 *  will call the passed function for intersecting the user
 *  geometry. */
RTCORE_API void rtcSetIntersectFunction (RTCScene scene, unsigned geomID, RTCIntersectFunc intersect);

/*! Set intersect function for ray packets of size 4. The
 *  rtcIntersect4 function will call the passed function for
 *  intersecting the user geometry. */
RTCORE_API void rtcSetIntersectFunction4 (RTCScene scene, unsigned geomID, RTCIntersectFunc4 intersect4);

/*! Set intersect function for ray packets of size 8. The
 *  rtcIntersect8 function will call the passed function for
 *  intersecting the user geometry.*/
RTCORE_API void rtcSetIntersectFunction8 (RTCScene scene, unsigned geomID, RTCIntersectFunc8 intersect8);

/*! Set intersect function for ray packets of size 16. The
 *  rtcIntersect16 function will call the passed function for
 *  intersecting the user geometry. */
RTCORE_API void rtcSetIntersectFunction16 (RTCScene scene, unsigned geomID, RTCIntersectFunc16 intersect16);

/*! Set occlusion function for single rays. The rtcOccluded function
 *  will call the passed function for intersecting the user
 *  geometry. */
RTCORE_API void rtcSetOccludedFunction (RTCScene scene, unsigned geomID, RTCOccludedFunc occluded);

/*! Set occlusion function for ray packets of size 4. The rtcOccluded4
 *  function will call the passed function for intersecting the user
 *  geometry. */
RTCORE_API void rtcSetOccludedFunction4 (RTCScene scene, unsigned geomID, RTCOccludedFunc4 occluded4);

/*! Set occlusion function for ray packets of size 8. The rtcOccluded8
 *  function will call the passed function for intersecting the user
 *  geometry. */
RTCORE_API void rtcSetOccludedFunction8 (RTCScene scene, unsigned geomID, RTCOccludedFunc8 occluded8);

/*! Set occlusion function for ray packets of size 16. The
 *  rtcOccluded16 function will call the passed function for
 *  intersecting the user geometry. */
RTCORE_API void rtcSetOccludedFunction16 (RTCScene scene, unsigned geomID, RTCOccludedFunc16 occluded16);

/*! @} */

#endif
