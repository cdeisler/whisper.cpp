#include "ftxui/component/captured_mouse.hpp"  // for ftxui
#include "ftxui/component/component.hpp"  // for Input, Renderer, ResizableSplitLeft
#include "ftxui/component/component_base.hpp"  // for ComponentBase, Component
#include "ftxui/component/screen_interactive.hpp"  // for ScreenInteractive
#include "ftxui/dom/elements.hpp"  // for operator|, separator, text, Element, flex, vbox, border
#include "ftxui/component/loop.hpp"
#include <ftxui/component/event.hpp>

using namespace ftxui;

class FtxuiUI {
private:
    // Data members
    int event_count;
    int frame_count;
    int custom_loop_count;
    bool isVadSimple;
    std::string current_task;
    std::string text_heard;

    // UI components
    int size = 50;
    std::string content_1;
    std::string content_2;
    ftxui::Component textarea_1{ Input(&content_1) };
    ftxui::Component textarea_2{ Input(&content_2) };
    ftxui::ScreenInteractive screen = ScreenInteractive::Fullscreen();
    ftxui::Component componentVars = Renderer([&] {
        return vbox({
            text("ftxui event count: " + std::to_string(event_count)),
            text("ftxui frame count: " + std::to_string(frame_count)),
            text("Custom loop count: " + std::to_string(custom_loop_count)),
            text("isVadSimple: " + std::to_string(isVadSimple)),
            text("current_task: " + current_task),
        }) |
        border;
    });
    ftxui::Component layout = ResizableSplitLeft(textarea_1, componentVars, &size);
    ftxui::Component component = Renderer(layout, [&] {
        return vbox({
            text("Input:" + text_heard),
            separator(),
            layout->Render() | flex,
        }) |
        border;
    });
    //std::unique_ptr<ftxui::Loop> loop; 

public:
    FtxuiUI() {

        //loop = std::make_unique<ftxui::Loop>(&screen, component);  // Use std::make_unique<ftxui::Loop>
    }

    void Refresh() {
        
        // if (!loop->HasQuitted()) {
        //     // custom_loop_count++;
        //     // screen.PostEvent(Event::Custom);
        //     loop->RunOnce();
        // }
        //custom_loop_count++;
        //screen.PostEvent(Event::Custom);
        //screen.Loop(component);
    }

    void SetCurrentTask(const std::string& task) {
        current_task = task;
    }

    void Write(const std::string& new_data) {
        content_1 += new_data;
    }
};