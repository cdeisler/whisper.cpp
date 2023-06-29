// FtxUI.h
#pragma once

#include <string>
#include "ftxui/component/captured_mouse.hpp"  // for ftxui
#include "ftxui/component/component.hpp"  // for Input, Renderer, ResizableSplitLeft
#include "ftxui/component/component_base.hpp"  // for ComponentBase, Component
#include "ftxui/component/screen_interactive.hpp"  // for ScreenInteractive
#include "ftxui/dom/elements.hpp"  // for operator|, separator, text, Element, flex, vbox, border
#include "ftxui/component/loop.hpp"
#include <ftxui/component/event.hpp> 

extern int event_count;
extern int frame_count;
extern int custom_loop_count;
extern bool isVadSimple;
extern std::string current_task;
extern std::string text_heard;
extern int splitSize;
extern std::string content_1;
extern std::string content_2;
extern ftxui::Component textarea_1;
extern ftxui::Component textarea_2;
extern ftxui::ScreenInteractive screen;
extern ftxui::Component componentVars();
extern ftxui::Component layout;
extern ftxui::Component component();