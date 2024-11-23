# Find yapf
#
# YAPF_EXECUTABLE   - Path to python-format executable
# YAPF_FOUND        - True if the python-format executable was found.
# YAPF_VERSION      - The version of python-format found
#

find_program(YAPF_EXECUTABLE
             NAMES yapf
             DOC "Python formatter executable")
mark_as_advanced(YAPF_EXECUTABLE)

# Extract version from command "yapf -version"
if(YAPF_EXECUTABLE)
  execute_process(COMMAND ${YAPF_EXECUTABLE} -version
                  OUTPUT_VARIABLE yapf_version
                  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(yapf_version MATCHES "^yapf .*")
    # yapf_version sample: "yapf 0.32.0"
    string(REGEX
           REPLACE "yapf version ([.0-9]+).*"
                   "\\1"
                   YAPF_VERSION
                   "${yapf_version}")
    # YAPF_VERSION sample: "0.32.0"
  else()
    set(YAPF_VERSION 0.0)
  endif()
else()
  set(YAPF_VERSION 0.0)
endif()

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set YAPF_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(YAPF REQUIRED_VARS YAPF_EXECUTABLE VERSION_VAR YAPF_VERSION)
