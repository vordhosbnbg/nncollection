cmake_minimum_required (VERSION 3.1)

project (nncollection)

file(GLOB_RECURSE Project_SOURCES "src/*.cpp")
file(GLOB_RECURSE Project_HEADERS "inc/*.h")
file(GLOB_RECURSE Project_Test_SOURCES "tst/*.cpp")
file(GLOB_RECURSE Project_Test_HEADERS "tst/*.h")
file(GLOB_RECURSE External_SOURCES "ext/*.cpp")
file(GLOB_RECURSE External_HEADERS "ext/*.h")

set (Project_INCLUDE_DIRS "")
set (Project_Test_INCLUDE_DIRS "")

foreach (_headerFile ${Project_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list (APPEND Project_INCLUDE_DIRS ${_dir})
endforeach()

foreach (_headerFile ${External_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list (APPEND Project_INCLUDE_DIRS ${_dir})
endforeach()

foreach (_headerFile ${Project_Test_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list (APPEND Project_Test_INCLUDE_DIRS ${_dir})
endforeach()

list(REMOVE_DUPLICATES Project_INCLUDE_DIRS)
list(REMOVE_DUPLICATES Project_Test_INCLUDE_DIRS)

include_directories(${Project_INCLUDE_DIRS})

add_library (nncollection SHARED ${Project_SOURCES} ${External_SOURCES} )
set_property(TARGET nncollection PROPERTY CXX_STANDARD 17)
find_package (Threads)
target_link_libraries (nncollection ${CMAKE_THREAD_LIBS_INIT})

foreach(_test ${Project_Test_SOURCES})
    get_filename_component(_target_name ${_test} NAME_WE)
    add_executable (${_target_name} ${_test} )
    target_link_libraries(${_target_name} ${CMAKE_THREAD_LIBS_INIT})
    set_property(TARGET ${_target_name} PROPERTY CXX_STANDARD 17)
endforeach()

include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-march=native" HAS_FLAG_MARCH_NATIVE)
if (HAS_FLAG_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif()

CHECK_CXX_COMPILER_FLAG("-mtune=native" HAS_FLAG_MTUNE_NATIVE)
if (HAS_FLAG_MTUNE_NATIVE)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mtune=native")
endif()

CHECK_CXX_COMPILER_FLAG("-pthread" HAS_FLAG_PTHREAD)
if (HAS_FLAG_PTHREAD)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -pthread")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -Wextra")
