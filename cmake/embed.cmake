# Writes OUTPUT, a header defining `inline constexpr char NAME[]` with the bytes of INPUT plus a
# terminating zero. Stands in for C++26 #embed until MSVC supports it.
# Usage: cmake -DINPUT=<file> -DOUTPUT=<header> -DNAME=<identifier> -P embed.cmake
file(READ "${INPUT}" content HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${content}")
string(REGEX REPLACE "(0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,)" "\\1\n    " bytes "${bytes}")
file(WRITE "${OUTPUT}" "// Generated from ${INPUT} by cmake/embed.cmake. Do not edit.\n#pragma once\ninline constexpr char ${NAME}[] = {\n    ${bytes}0x00};\n")
