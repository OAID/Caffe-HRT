set(ACL_INC_PATHS
    /usr/include
    /usr/local/include
    /usr/local/acl
    $ENV{ACL_DIR}/include
    )

set(ACL_LIB_PATHS
    /lib
    /lib64
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    /usr/local/acl/lib
    /usr/local/acl/lib64
    $ENV{ACL_DIR}/lib
    )

find_path(ACL_INCLUDE NAMES arm_compute PATHS ${ACL_INC_PATHS})
find_library(ACL_LIBRARIES NAMES arm_compute-static PATHS ${ACL_LIB_PATHS})
find_library(ACL_CORE_LIBRARIES NAMES arm_compute_core-static PATHS ${ACL_LIB_PATHS})
SET(ACL_LIBRARIES "${ACL_CORE_LIBRARIES} ${ACL_LIBRARIES}")

if(ACL_INCS)
  SET(ACL_INCLUDE "${ACL_INCS}")
  SET(ACL_LIBRARIES "${ACL_LIBS}")
  SET(ACL_FOUND 1)
else  ()
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(ACL DEFAULT_MSG ACL_INCLUDE ACL_LIBRARIES)
endif ()

if (ACL_FOUND)
  message(STATUS "Found ACL    (include: ${ACL_INCLUDE}, library: ${ACL_LIBRARIES})")
  mark_as_advanced(ACL_INCLUDE ACL_LIBRARIES)
endif ()
