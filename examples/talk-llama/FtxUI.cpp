// FtxUI.cpp
#include "FtxUI.h"

using namespace ftxui;

int event_count = 0;
int frame_count = 0;
int custom_loop_count = 0;
bool isVadSimple = false;
std::string current_task;
std::string text_heard;
int splitSize = 50;
std::string content_1;
std::string content_2;
auto textarea_1 = Input(&content_1);
auto textarea_2 = Input(&content_2);
ScreenInteractive screen = ScreenInteractive::Fullscreen();

Component componentVars() {
    return Renderer([&] {
        return vbox({
            text("ftxui event count: " + std::to_string(event_count)),
            text("ftxui frame count: " + std::to_string(frame_count)),
            text("Custom loop count: " + std::to_string(custom_loop_count)),
            text("isVadSimple: " + std::to_string(isVadSimple)),
            text("current_task: " + current_task),
        }) |
        border;
    });
}

Component conversation() {
    return Renderer([&] {
        return hbox(paragraph(content_1)) |
        border;
    });
}

Component layout = ResizableSplitLeft(conversation(), componentVars(), &splitSize);

Component component() {
    return Renderer(layout, [&] {
        return vbox({
            text("Input:" + text_heard),
            separator(),
            layout->Render() | flex,
        }) |
        border;
    });
}
