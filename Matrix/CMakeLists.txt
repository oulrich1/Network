#  TO get the location of something:
#  get_target_property(VarToStorePath ExecName LOCATION )
#  get_target_property(varProp File, PROPERTY_TYPE_ID )

# set compiler flags

# Add the current directory to the include path
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

# Add the matrix library
set (MATRIX_SOURCE_FILES "matrix.cpp")
file (GLOB MATRIX_INCLUDE_FILES
  "*.h"
  # "../math/*.h"
)

# add library
add_library(Matrix STATIC ${MATRIX_SOURCE_FILES})

install (TARGETS Matrix DESTINATION bin)
install (FILES ${MATRIX_INCLUDE_FILES} DESTINATION include)
