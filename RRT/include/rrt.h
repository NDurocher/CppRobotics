#pragma once

#include <vector>

namespace rrt {

    class Node {
    public:
        Node(float x, float y);

        Node(float x, float y, Node *parent_node);

        std::vector<float> get_path_x();

        std::vector<float> get_path_y();

        float x();

        float y();

    private:
        float m_pos_x{0};
        float m_pos_y{0};
        std::vector<float> m_path_x{};
        std::vector<float> m_path_y{};
        Node *m_parent{nullptr};
    };


    class AreaBoundry {
    public:
        AreaBoundry(float x_min, float x_max, float y_min, float y_max);

        std::vector<float> get_x_bounds();

        std::vector<float> get_y_bounds();

    private:
        float m_x_min{0};
        float m_x_max{0};
        float m_y_min{0};
        float m_y_max{0};
    };

    class RRT {
        // Base Class for RRT path planning
    public:
        RRT(float start_x, float start_y, float finish_x, float finish_y, float epsilon);

        void check_new_point(AreaBoundry bounds);

        int find_closest_node_index(Node &sample_node);

        float distance_between_nodes(Node &n1, Node &n2);

        void make_new_node(Node &sample_node, Node &closest_node);

        bool check_done();

        Node get_last_node();

    private:
        std::vector<Node> m_node_list;
        Node m_finish_node;
        float m_epsilon;
    };

} // namespace rrt