#include <cstdio>
#include <iostream>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include <cmath>
#include <ctime>

#include "gnuplot-iostream.h"
#include "../include/RRT/rrt.h"

int main() {
	
	srand(time(0));
	Gnuplot gp;
	gp << "set xrange [-10:10]\nset yrange [0:20]\n";

	std::vector<float> plt_x;
	std::vector<float> plt_y;

	rrt::AreaBoundry boundry(-10,10,0,20);

	rrt::RRT solver(-8.75, 1.25, 8.75, 18.75, 1.);

	while (true)
	{
		solver.check_new_point(boundry);
		
		if (solver.check_done()){
			break;
		}	
		plt_x.push_back(solver.get_last_node().x());
		plt_y.push_back(solver.get_last_node().y());
		gp << "plot '-' with points title 'Tree'\n";
		gp.send1d(boost::make_tuple(plt_x, plt_y));
	}

	gp << "plot '-' with points title 'Tree', \
		'-' with lines title 'Solution Tree'\n";
	gp.send1d(boost::make_tuple(plt_x, plt_y));
	gp.send1d(boost::make_tuple(solver.get_last_node().get_path_x(), solver.get_last_node().get_path_y()));

	return 0;
}