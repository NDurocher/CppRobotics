#include "rrt.h"
#include <iostream>
#include <cmath>

namespace rrt{

Node::Node(float x, float y)  : m_pos_x{x}, m_pos_y{y}{};
Node::Node(float x, float y, Node* parent_node) : m_pos_x{x}, 
	m_pos_y{y}, m_parent{parent_node}{
		m_path_x = parent_node->get_path_x();
		m_path_x.push_back(m_pos_x);
		m_path_y = parent_node->get_path_y();
		m_path_y.push_back(m_pos_y);
};

std::vector<float> Node::get_path_x(){
	return m_path_x;
};

std::vector<float> Node::get_path_y(){
	return m_path_y;
};

float Node::x(){
	return m_pos_x;
}

float Node::y(){
	return m_pos_y;
}


//////////////////////////////////////////////

AreaBoundry::AreaBoundry(float x_min, float x_max, float y_min, float y_max) : 
	m_x_min{x_min}, m_x_max{x_max}, m_y_min{y_min}, m_y_max{y_max} {};

std::vector<float> AreaBoundry::get_x_bounds(){
	return std::vector<float>{m_x_min, m_x_max};
};

std::vector<float> AreaBoundry::get_y_bounds(){
	return std::vector<float>{m_y_min, m_y_max};	
};

//////////////////////////////////////////////

RRT::RRT(float start_x, float start_y, float finish_x, 
				float finish_y, float epsilon) : 
	m_finish_node{finish_x, finish_y}, m_epsilon{epsilon}
{
	m_node_list.push_back({start_x, start_y});
};

void RRT::check_new_point(AreaBoundry bounds){
	auto x_bound = bounds.get_x_bounds();
	auto y_bound = bounds.get_y_bounds();

	float sampled_x = x_bound[0] + static_cast <float>(rand()) / 
				( static_cast<float>(RAND_MAX/(x_bound[1]-x_bound[0])));
	float sampled_y = y_bound[0] + static_cast <float>(rand()) / 
				( static_cast<float>(RAND_MAX/(y_bound[1]-y_bound[0])));
	Node sample_node{sampled_x, sampled_y};

	make_new_node(sample_node, m_node_list[find_closest_node_index(sample_node)]);
};	

int RRT::find_closest_node_index(Node& sample_node){
	int min_index{-1};
	int min_distance{10000000};
	for (int i = 0; i < m_node_list.size(); i++){
		float distance = distance_between_nodes(sample_node, m_node_list[i]);
		if (distance < min_distance){
			min_distance = distance;
			min_index = i;
		}
	}
	return min_index;
};

float RRT::distance_between_nodes(Node& n1, Node& n2){
	return std::sqrt( std::pow(n1.x() - n2.x(), 2) + 
					  std::pow(n1.y() - n2.y(), 2));
};

void RRT::make_new_node(Node& sample_node, Node& closest_node){
	float unit_x = (sample_node.x() - closest_node.x()) / 
				distance_between_nodes(sample_node, closest_node);
	float unit_y = (sample_node.y() - closest_node.y()) / 
				distance_between_nodes(sample_node, closest_node);

	float new_x = unit_x * m_epsilon + closest_node.x();
	float new_y = unit_y * m_epsilon + closest_node.y();

	m_node_list.push_back({new_x, new_y, &closest_node}); 
};

bool RRT::check_done(){
	return distance_between_nodes(m_node_list.back(), m_finish_node) < 1;
};

Node RRT::get_last_node(){
	return m_node_list.back();
};

} // namespace rrt












