# Find gcovr
#
# GCOVR_EXECUTABLE - Path to gcovr executable
# GCOVR_FOUND      - True if the gcovr executable was found
# GCOVR_VERSION    - The version of gcovr found
#

find_program(GCOVR_EXECUTABLE
             NAMES gcovr
             DOC "gcovr executable")
mark_as_advanced(GCOVR_EXECUTABLE)

# Extract version from command "gcovr -version"
if(GCOVR_EXECUTABLE)
  execute_process(COMMAND ${GCOVR_EXECUTABLE} --version
                  OUTPUT_VARIABLE gcovr_version
                  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(gcovr_version MATCHES "^gcovr ([.0-9]+)")
    set(GCOVR_VERSION ${CMAKE_MATCH_1})
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GCOVR REQUIRED_VARS GCOVR_EXECUTABLE VERSION_VAR GCOVR_VERSION)
