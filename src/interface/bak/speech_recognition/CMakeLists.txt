cmake_minimum_required(VERSION 3.5.1)
project(speech_recognition)

## Compile as C++14, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set project dependencies
set(PROJECT_DEPS
    roscpp
    std_msgs
    rospy
    roslib
    roslint
    alfred_msgs)

# Set external dependencies
set(EXTERNAL_DEPS
    SDL2)

# Nlohmann JSON needs to be found manually
find_package(nlohmann_json 3.2.0 REQUIRED)

# Find external dependencies
find_package(${EXTERNAL_DEPS} REQUIRED)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  ${CATKIN_COMPONENTS}
  ${PROJECT_DEPS}
)

#################################################################
## Declare ROS tests for linting, unit testing and integration ##
#################################################################
# Disable whitespace/blank_line and build/namespaces, c++11
set(ROSLINT_CPP_OPTS "--extensions=cpp,h,hpp" "--filter=-whitespace/blank_line,-build/namespaces,-build/c++11,-runtime/references")

# Invoke roslint for files to be checked
roslint_cpp(src/main.cpp 
    src/Audio.cpp
    src/SpeechRecognitionHandle.cpp
    src/WhisperParams.cpp
    src/WhisperWrapper.cpp
    src/Util.cpp
    include/Audio.hpp
    include/SpeechRecognitionHandle.hpp 
    include/WhisperParams.hpp 
    include/WhisperWrapper.hpp
    include/Util.hpp)

# Add above checks to test
roslint_add_test()

# Whisper build configuration
# options

option(BUILD_SHARED_LIBS               "whisper: build shared libs" ${BUILD_SHARED_LIBS_DEFAULT})

option(WHISPER_ALL_WARNINGS            "whisper: enable all compiler warnings"                   ON)
option(WHISPER_ALL_WARNINGS_3RD_PARTY  "whisper: enable all compiler warnings in 3rd party libs" OFF)

option(WHISPER_SANITIZE_THREAD         "whisper: enable thread sanitizer"    OFF)
option(WHISPER_SANITIZE_ADDRESS        "whisper: enable address sanitizer"   OFF)
option(WHISPER_SANITIZE_UNDEFINED      "whisper: enable undefined sanitizer" OFF)

option(WHISPER_BUILD_TESTS             "whisper: build tests"    ${WHISPER_STANDALONE})
option(WHISPER_BUILD_EXAMPLES          "whisper: build examples" ${WHISPER_STANDALONE})

option(WHISPER_SUPPORT_SDL2            "whisper: support for libSDL2" ON)

if (APPLE)
    option(WHISPER_NO_ACCELERATE       "whisper: disable Accelerate framework" OFF)
    option(WHISPER_NO_AVX              "whisper: disable AVX" OFF)
    option(WHISPER_NO_AVX2             "whisper: disable AVX2" OFF)
    option(WHISPER_NO_FMA              "whisper: disable FMA" OFF)
else()
    option(WHISPER_SUPPORT_OPENBLAS    "whisper: support for OpenBLAS" OFF)
endif()

option(WHISPER_PERF                    "whisper: enable perf timings" OFF)

# sanitizers

if (NOT MSVC)
    if (WHISPER_SANITIZE_THREAD)
        set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -fsanitize=thread")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    endif()

    if (WHISPER_SANITIZE_ADDRESS)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}     -fsanitize=address -fno-omit-frame-pointer")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    endif()

    if (WHISPER_SANITIZE_UNDEFINED)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}     -fsanitize=undefined")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
    endif()
endif()

# dependencies

find_package(Threads REQUIRED)

# on APPLE - include Accelerate framework
if (APPLE AND NOT WHISPER_NO_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")

        set(WHISPER_EXTRA_LIBS  ${WHISPER_EXTRA_LIBS}  ${ACCELERATE_FRAMEWORK})
        set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DGGML_USE_ACCELERATE)
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()

if (WHISPER_SUPPORT_OPENBLAS)
    find_library(OPENBLAS_LIB
        NAMES openblas libopenblas
        )
    if (OPENBLAS_LIB)
        message(STATUS "OpenBLAS found")

        set(WHISPER_EXTRA_LIBS  ${WHISPER_EXTRA_LIBS}  ${OPENBLAS_LIB})
        set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DGGML_USE_OPENBLAS)
    else()
        message(WARNING "OpenBLAS not found")
    endif()
endif()

# compiler flags

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif ()

if (WHISPER_ALL_WARNINGS)
    if (NOT MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} \
            -Wall                           \
            -Wextra                         \
            -Wpedantic                      \
            -Wshadow                        \
            -Wcast-qual                     \
            -Wstrict-prototypes             \
            -Wpointer-arith                 \
            -Wno-unused-function            \
        ")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
            -Wall                           \
            -Wextra                         \
            -Wpedantic                      \
            -Wcast-qual                     \
            -fomit-frame-pointer            \
            -march=native                   \
            -flto                           \
            -ffloat-store                   \
            -fexcess-precision=fast         \
            -ffast-math                     \
            -fno-rounding-math              \
            -fno-signaling-nans             \
            -fcx-limited-range              \
            -fno-math-errno                 \
            -funsafe-math-optimizations     \
            -fassociative-math              \
            -freciprocal-math               \
            -ffinite-math-only              \
            -fno-signed-zeros               \
            -fno-trapping-math              \
            -frounding-math                 \
            -fsingle-precision-constant     \
            -fcx-fortran-rules              \
        ")
    else()
        # todo : msvc
    endif()
endif()

if (NOT MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror=vla")
    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno -ffinite-math-only -funsafe-math-optimizations")
endif()

message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
else()
    message(STATUS "x86 detected")
    if (MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX2")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX2")
    else()
        if (EMSCRIPTEN)
            set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -pthread")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
        else()
            if(NOT WHISPER_NO_AVX)
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx")
            endif()
            if(NOT WHISPER_NO_AVX2)
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
            endif()
            if(NOT WHISPER_NO_FMA)
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfma")
            endif()
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mf16c")
        endif()
    endif()
endif()

if (WHISPER_PERF)
    set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DGGML_PERF)
endif()

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     CMAKE_EXPORT_COMPILE_COMMANDSatkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS include
 LIBRARIES speech_recognition
 CATKIN_DEPENDS roscpp rospy std_msgs alfred_msgs
 DEPENDS
 ${PROJECT_DEPS}
 ${EXTERNAL_DEPS}
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(${SDL2_INCLUDE_DIRS})

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(lib${PROJECT_NAME}
  src/whisper.cpp
  src/ggml.c
  src/Util.cpp
  src/main.cpp
  src/Audio.cpp
  src/SpeechRecognitionHandle.cpp
  src/WhisperParams.cpp
  src/WhisperWrapper.cpp
)

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(${PROJECT_NAME} src/whisper.cpp src/ggml.c src/Util.cpp src/main.cpp src/Audio.cpp src/SpeechRecognitionHandle.cpp src/WhisperParams.cpp src/WhisperWrapper.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${SDL2_LIBRARIES}
  nlohmann_json::nlohmann_json
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

# Download whisper model
set(PATH "${CMAKE_CURRENT_SOURCE_DIR}/models")
set(FILE "${PATH}/ggml-base.en.bin")
message(STATUS "Checking and downloading whisper model to ${FILE}")
if (NOT EXISTS "${FILE}")
  message(STATUS "... file does not exist. Downloading whisper model to ${FILE}")
  execute_process(COMMAND mkdir -p ${PATH})
  execute_process(COMMAND wget https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin -P ${PATH})
endif()

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_speech_recognition.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
