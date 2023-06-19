if (WHISPER_SDL2)

    find_package(nlohmann_json 3.2.0 REQUIRED)

    # stream
    set(TARGET stream)
    add_executable(${TARGET} stream.cpp)

    include(DefaultTargetOptions)

    target_link_libraries(${TARGET} PRIVATE common common-sdl whisper nlohmann_json::nlohmann_json ${CMAKE_THREAD_LIBS_INIT})
endif ()
