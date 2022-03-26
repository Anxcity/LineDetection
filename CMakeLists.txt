project(demo)

set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( main main.cpp include/Autocanny.h Autocanny.cpp include/kmeans.h kmeans.cpp)
target_link_libraries( main ${OpenCV_LIBS} )
