project(EigenDoc)

set_directory_properties(PROPERTIES EXCLUDE_FROM_ALL TRUE)

project(EigenDoc)

if(CMAKE_COMPILER_IS_GNUCXX)
  if(CMAKE_SYSTEM_NAME MATCHES Linux)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -g1")
  endif(CMAKE_SYSTEM_NAME MATCHES Linux)
endif(CMAKE_COMPILER_IS_GNUCXX)

option(EIGEN_INTERNAL_DOCUMENTATION "Build internal documentation" OFF)
option(EIGEN_DOC_USE_MATHJAX "Use MathJax for rendering math in HTML docs" ON)

# Set some Doxygen flags
set(EIGEN_DOXY_PROJECT_NAME             "Eigen")
set(EIGEN_DOXY_OUTPUT_DIRECTORY_SUFFIX  "")
set(EIGEN_DOXY_INPUT                    "\"${Eigen_SOURCE_DIR}/Eigen\" \"${Eigen_SOURCE_DIR}/doc\"")
set(EIGEN_DOXY_HTML_COLORSTYLE_HUE      "220")
set(EIGEN_DOXY_TAGFILES                 "")

if(EIGEN_INTERNAL_DOCUMENTATION)
  set(EIGEN_DOXY_INTERNAL                 "YES")
else(EIGEN_INTERNAL_DOCUMENTATION)
  set(EIGEN_DOXY_INTERNAL                 "NO")
endif(EIGEN_INTERNAL_DOCUMENTATION)

if (EIGEN_DOC_USE_MATHJAX)
  set(EIGEN_DOXY_USE_MATHJAX "YES")
else ()
  set(EIGEN_DOXY_USE_MATHJAX "NO")
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
  ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
)

set(EIGEN_DOXY_PROJECT_NAME             "Eigen-unsupported")
set(EIGEN_DOXY_OUTPUT_DIRECTORY_SUFFIX  "/unsupported")
set(EIGEN_DOXY_INPUT                    "\"${Eigen_SOURCE_DIR}/unsupported/Eigen\" \"${Eigen_SOURCE_DIR}/unsupported/doc\"")
set(EIGEN_DOXY_HTML_COLORSTYLE_HUE      "0")
set(EIGEN_DOXY_TAGFILES                 "\"${Eigen_BINARY_DIR}/doc/Eigen.doxytags=..\"")
#set(EIGEN_DOXY_TAGFILES                 "")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
  ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile-unsupported
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/eigendoxy_header.html.in
  ${CMAKE_CURRENT_BINARY_DIR}/eigendoxy_header.html
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/eigendoxy_footer.html.in
  ${CMAKE_CURRENT_BINARY_DIR}/eigendoxy_footer.html
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/eigendoxy_layout.xml.in
  ${CMAKE_CURRENT_BINARY_DIR}/eigendoxy_layout.xml
)

configure_file(
  ${Eigen_SOURCE_DIR}/unsupported/doc/eigendoxy_layout.xml.in
  ${Eigen_BINARY_DIR}/doc/unsupported/eigendoxy_layout.xml
)

set(examples_targets "")
set(snippets_targets "")

add_definitions("-DEIGEN_MAKING_DOCS")
add_custom_target(all_examples)

add_subdirectory(examples)
add_subdirectory(special_examples)
add_subdirectory(snippets)

add_custom_target(
  doc-eigen-prerequisites
  ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/html/
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/eigen_navtree_hacks.js           ${CMAKE_CURRENT_BINARY_DIR}/html/
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Eigen_Silly_Professor_64x64.png  ${CMAKE_CURRENT_BINARY_DIR}/html/
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/ftv2pnode.png                    ${CMAKE_CURRENT_BINARY_DIR}/html/
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/ftv2node.png                     ${CMAKE_CURRENT_BINARY_DIR}/html/
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/AsciiQuickReference.txt          ${CMAKE_CURRENT_BINARY_DIR}/html/
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(
  doc-unsupported-prerequisites
  ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${Eigen_BINARY_DIR}/doc/html/unsupported
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/eigen_navtree_hacks.js           ${CMAKE_CURRENT_BINARY_DIR}/html/unsupported/
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Eigen_Silly_Professor_64x64.png  ${CMAKE_CURRENT_BINARY_DIR}/html/unsupported/
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/ftv2pnode.png                    ${CMAKE_CURRENT_BINARY_DIR}/html/unsupported/
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/ftv2node.png                     ${CMAKE_CURRENT_BINARY_DIR}/html/unsupported/
  WORKING_DIRECTORY ${Eigen_BINARY_DIR}/doc
)

add_dependencies(doc-eigen-prerequisites all_snippets all_examples)
add_dependencies(doc-unsupported-prerequisites unsupported_snippets unsupported_examples)

add_custom_target(doc ALL
  COMMAND doxygen
  COMMAND doxygen Doxyfile-unsupported
  COMMAND ${CMAKE_COMMAND} -E copy ${Eigen_BINARY_DIR}/doc/html/group__TopicUnalignedArrayAssert.html ${Eigen_BINARY_DIR}/doc/html/TopicUnalignedArrayAssert.html
  COMMAND ${CMAKE_COMMAND} -E rename html eigen-doc
  COMMAND ${CMAKE_COMMAND} -E remove eigen-doc/eigen-doc.tgz
  COMMAND ${CMAKE_COMMAND} -E tar cfz eigen-doc.tgz eigen-doc
  COMMAND ${CMAKE_COMMAND} -E rename eigen-doc.tgz eigen-doc/eigen-doc.tgz
  COMMAND ${CMAKE_COMMAND} -E rename eigen-doc html
  WORKING_DIRECTORY ${Eigen_BINARY_DIR}/doc)

add_dependencies(doc doc-eigen-prerequisites doc-unsupported-prerequisites)
