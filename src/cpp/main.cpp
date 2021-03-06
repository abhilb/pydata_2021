
// std includes
#include <array>
#include <string>
#include <algorithm>
#include <memory>

// onnx includes
#include <onnxruntime_cxx_api.h>

// sdl includes
#include <SDL.h>
#include <SDL_opengl.h>

// imgui includes
#include "imgui.h"
#include "imgui_impl_opengl2.h"
#include "imgui_impl_sdl.h"
#include "spdlog/spdlog.h"

// opencv includes
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using digits_input = std::array<float, 784>;

struct DigitsModel
{
    DigitsModel(Ort::Env& env, const ORTCHAR_T* model_path, std::vector<const char*>&& output_names) :
        _session(env, model_path, Ort::SessionOptions{ nullptr }),
        _output_names(std::move(output_names))
    {        
    }

    int infer(digits_input& input)
    {                
        Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(info, const_cast<float*>(input.data()), input.size(), _input_shape.data(), _input_shape.size());

        std::vector<Ort::Value> ort_outputs = _session.Run(Ort::RunOptions{ nullptr }, _input_names.data(), &input_tensor, 1, _output_names.data(), 2);
        
        auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
        auto data_length = ort_outputs[0].GetStringTensorDataLength();
        std::string result(data_length, '\0');
        std::vector<size_t> offsets(type_info.GetElementCount());
        ort_outputs[0].GetStringTensorContent((void*)result.data(), data_length, offsets.data(), offsets.size());

        return std::stoi(result);
    }

private:    
    std::array<int64_t, 2> _input_shape{ 1, 784 };
    Ort::Session _session;
    std::vector<const char*> _input_names = { "float_input" };
    std::vector<const char*> _output_names; 
};


int main(int argc, char **argv)
{
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) !=
        0) {
        spdlog::error("Error: {}\n", SDL_GetError());
        return -1;
    }

    Ort::Env env;
    //@todo change full paths to relative paths
    DigitsModel rf_model(env, L"C:/dev/pydata_2021/src/models/onnx_models/random_forest_model.onnx", { "output_label" , "output_probability" });
    DigitsModel lr_model(env, L"C:/dev/pydata_2021/src/models/onnx_models/log_reg_model.onnx", { "output_label" , "output_probability" });
    DigitsModel svm_model(env, L"C:/dev/pydata_2021/src/models/onnx_models/svm_model.onnx", { "label" , "probabilities" });

    // Setup window
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
    SDL_WindowFlags window_flags =
        (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE |
                          SDL_WINDOW_ALLOW_HIGHDPI);
    SDL_Window *window =
        SDL_CreateWindow("Digits", SDL_WINDOWPOS_CENTERED,
                         SDL_WINDOWPOS_CENTERED, 800, 800, window_flags);
    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, gl_context);
    SDL_GL_SetSwapInterval(1);  // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    ImFont* font1 = io.Fonts->AddFontDefault();
    ImFont* font2 = io.Fonts->AddFontDefault();
    font2->Scale *= 10;

    ImGui::StyleColorsDark();

    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL2_Init();
    
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    bool done = false;
    while (!done) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) done = true;
            if (event.type == SDL_WINDOWEVENT &&
                event.window.event == SDL_WINDOWEVENT_CLOSE &&
                event.window.windowID == SDL_GetWindowID(window))
                done = true;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        {
            static int rf_prediction = -1;
            static int lr_prediction = -1;
            static int svm_prediction = -1;

            ImGui::Begin("Canvas");
            {
                static ImVector<ImVec2> points;
                static bool predict, clear;

                predict = ImGui::Button("Predict");
                ImGui::SameLine();
                clear = ImGui::Button("Clear");
                
                // Using InvisibleButton() as a convenience 1) it will advance the layout cursor and 2) allows us to use IsItemHovered()/IsItemActive()
                ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();    // ImDrawList API uses screen coordinates!
                ImVec2 canvas_sz = ImGui::GetContentRegionAvail(); // Resize canvas to what's available
                if (canvas_sz.x < 50.0f)
                    canvas_sz.x = 50.0f;
                if (canvas_sz.y < 50.0f)
                    canvas_sz.y = 50.0f;
                ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);

                // Draw border and background color
                ImGuiIO &io = ImGui::GetIO();
                ImDrawList *draw_list = ImGui::GetWindowDrawList();
                draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
                draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

                // This will catch our interactions
                ImGui::InvisibleButton("canvas", canvas_sz, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
                const bool is_hovered = ImGui::IsItemHovered(); // Hovered
                const bool is_active = ImGui::IsItemActive();   // Held
                const ImVec2 origin(canvas_p0.x, canvas_p0.y);  // Lock scrolled origin
                const ImVec2 mouse_pos_in_canvas(io.MousePos.x - origin.x, io.MousePos.y - origin.y);
                                
                if (is_hovered && ImGui::IsMouseDown(ImGuiMouseButton_Left))
                    points.push_back(mouse_pos_in_canvas);

                draw_list->PushClipRect(canvas_p0, canvas_p1, true);
                
                for (int n = 0; n < points.Size; n++)
                    draw_list->AddCircle(ImVec2(origin.x + points[n].x, origin.y + points[n].y), 1.0, IM_COL32(0, 255, 0, 255), 0, 20.0f);

                draw_list->PopClipRect();                                                

                if(predict)
                {
                    cv::Mat image = cv::Mat(canvas_sz.y, canvas_sz.x, CV_8UC1, cv::Scalar(0));
                    spdlog::info("Image size: {}, {}", image.rows, image.cols);
                    for (int n = 0; n < points.Size; n++)
                        cv::circle(image, cv::Point(points[n].x, points[n].y), 20.0, cv::Scalar(255), -1);
                    cv::Mat scaled_image;
                    cv::resize(image, scaled_image, cv::Size(28, 28), cv::INTER_AREA);
                    
                    digits_input input_image;
                    int idx = 0;
                    for (auto it = scaled_image.begin<uchar>(); it != scaled_image.end<uchar>(); ++it, ++idx)
                        input_image[idx] = float(*it / 255.0);
                    
                    rf_prediction = rf_model.infer(input_image);
                    lr_prediction = lr_model.infer(input_image);
                    svm_prediction = svm_model.infer(input_image);
                    //cv::imwrite(std::to_string(prediction) + ".bmp", scaled_image);
                }

                if (clear)
                {
                    points.clear();
                    rf_prediction = -1;
                    lr_prediction = -1;
                    svm_prediction = -1;
                }

            }
            ImGui::End();

            ImGui::Begin("Prediction - ONNX Random Forest");
            {                
                ImGui::PushFont(font2);
                if (rf_prediction >= 0)
                    ImGui::TextColored(ImVec4{ 1.0, 1.0, 0.0, 1.0 }, "%i", rf_prediction);
                ImGui::PopFont();
            }
            ImGui::End();

            ImGui::Begin("Prediction - ONNX Logistic Regression");
            {
                ImGui::PushFont(font2);
                if (lr_prediction >= 0)
                    ImGui::TextColored(ImVec4{ 1.0, 0.0, 1.0, 1.0 }, "%i", lr_prediction);
                ImGui::PopFont();
            }
            ImGui::End();

            ImGui::Begin("Prediction - ONNX SVM");
            {
                ImGui::PushFont(font2);
                if (svm_prediction >= 0)
                    ImGui::TextColored(ImVec4{ 0.0, 1.0, 1.0, 1.0 }, "%i", svm_prediction);
                ImGui::PopFont();
            }
            ImGui::End();

        }

        // Rendering
        ImGui::Render();
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        glClearColor(clear_color.x * clear_color.w,
                     clear_color.y * clear_color.w,
                     clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
    }

    // Cleanup
    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplSDL2_Shutdown();

    ImGui::DestroyContext();

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
