# Find SuperLU_dist
# SuperLU_dist_FOUND            - True if SuperLU_dist was found
# SuperLU_dist::SuperLU_dist    - interface target

find_path(SuperLU_dist_INCLUDE_DIR superlu_ddefs.h)
find_library(SuperLU_dist_LIBRARY NAMES superlu_dist)

mark_as_advanced(
  SuperLU_dist_INCLUDE_DIR
  SuperLU_dist_LIBRARY
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
  SuperLU_dist DEFAULT_MSG
  SuperLU_dist_LIBRARY SuperLU_dist_INCLUDE_DIR
)

if(SuperLU_dist_FOUND AND NOT TARGET SuperLU_dist::SuperLU_dist)
  add_library(SuperLU_dist::SuperLU_dist UNKNOWN IMPORTED)
  set_target_properties(SuperLU_dist::SuperLU_dist PROPERTIES
    IMPORTED_LOCATION "${SuperLU_dist_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${SuperLU_dist_INCLUDE_DIR}"
  )
  message("SuperLU_dist_LIBRARY: ${SuperLU_dist_LIBRARY}")
  message("SuperLU_dist_INCLUDE_DIR: ${SuperLU_dist_INCLUDE_DIR}")
endif()
