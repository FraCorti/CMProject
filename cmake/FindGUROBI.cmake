set(GUROBI_HOME "/opt/gurobi902/linux64")
set(GUROBI_DIR ${GUROBI_HOME})
set(LD_LIBRARY_PATH ${GUROBI_HOME}/lib)

find_path(GUROBI_INCLUDE_DIRS
        NAMES gurobi_c.h
        HINTS ${GUROBI_DIR} $ENV{GUROBI_HOME}
        PATH_SUFFIXES include)

find_library(GUROBI_LIBRARY
        NAMES gurobi gurobi90 gurobi91
        HINTS ${GUROBI_DIR} $ENV{GUROBI_HOME}
        PATH_SUFFIXES lib)

find_library(GUROBI_CXX_LIBRARY
        NAMES gurobi_c++
        HINTS ${GUROBI_DIR} $ENV{GUROBI_HOME}
        PATH_SUFFIXES lib)

set(GUROBI_CXX_DEBUG_LIBRARY ${GUROBI_CXX_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GUROBI DEFAULT_MSG GUROBI_LIBRARY)
