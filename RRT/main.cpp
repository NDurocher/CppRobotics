#include <vector>
#include <ctime>

// ImGui and ImPlot includes
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"

// GLFW for window creation (assuming OpenGL backend)
#include <GLFW/glfw3.h>

#include "rrt.h"

// Error callback for GLFW
static void glfw_error_callback(const int error, const char* description) {
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int main() {
    // Setup GLFW window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    auto glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    auto glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.0 + GLSL 130
    auto glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ required
#endif

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(1280, 720, "RRT with ImPlot", nullptr, nullptr);
    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext(); // Create ImPlot context
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Our state
    constexpr auto clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    srand(time(nullptr));

    std::vector<float> plt_x;
    std::vector<float> plt_y;
    std::vector<float> solution_x;
    std::vector<float> solution_y;

    rrt::AreaBoundry bounds(-10, 10, 0, 20);
    rrt::RRT solver(-8.75, 1.25, 8.75, 18.75, 1.);

    bool rrt_done = false;

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // RRT logic updates
        if (!rrt_done) {
            solver.check_new_point(bounds);

            if (solver.check_done()) {
                rrt_done = true;
                // Get the solution path once RRT is done
                solution_x = solver.get_last_node().get_path_x();
                solution_y = solver.get_last_node().get_path_y();
            }
            plt_x.push_back(solver.get_last_node().x());
            plt_y.push_back(solver.get_last_node().y());
        }

        // ImGui window for plotting
        ImGui::Begin("RRT Path Planning");

        if (ImPlot::BeginPlot("RRT Tree and Solution", ImVec2(-1,0), ImPlotFlags_CanvasOnly)) {
            ImPlot::SetupAxesLimits(bounds.get_x_bounds()[0], bounds.get_x_bounds()[1], bounds.get_y_bounds()[0], bounds.get_y_bounds()[1], ImPlotCond_Always);

            // Plot the RRT tree points
            if (!plt_x.empty()) {
                ImPlot::PlotScatter("Tree", plt_x.data(), plt_y.data(), plt_x.size());
            }

            // Plot the solution path if RRT is done
            if (rrt_done && !solution_x.empty()) {
                ImPlot::PlotLine("Solution Path", solution_x.data(), solution_y.data(), solution_x.size());
            }
            ImPlot::EndPlot();
        }
        ImGui::End();

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext(); // Destroy ImPlot context
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}