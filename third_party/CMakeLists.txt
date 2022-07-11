
cmake_minimum_required(VERSION 3.10)

project(third_party LANGUAGES C CXX)


# matio
SET(MATIO_WITH_ZLIB OFF)
SET(MATIO_SHARED OFF)
add_subdirectory(matio)
