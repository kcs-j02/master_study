#pragma once

#include <taskflow/taskflow.hpp>
#include <vector>
#include <string>
#include <unordered_map>

#include "graph_types.hpp"

void make_nodeinfo(
  tf::Taskflow& tf,
  std::unordered_map<std::string, NodeInfo>& node_info
);

void add_level(
  std::vector<std::vector<std::string>>& level,
  int& count_level
);

void push_node_to_level(
  int count_level,
  tf::Taskflow& tf,
  std::unordered_map<std::string, NodeInfo>& node_info,
  std::vector<std::vector<std::string>>& level,
  int& node_count
);

void remove_node(
  int& count_level,
  std::vector<std::vector<std::string>>& level,
  std::unordered_map<std::string, NodeInfo>& node_info
);

void check_nodes(
  tf::Taskflow& tf,
  std::unordered_map<std::string, NodeInfo>& node_info
);

void check_level_all(
  const std::vector<std::vector<std::string>>& level
);

void DFG_based_task_leveling(
  tf::Taskflow& tf,
  std::unordered_map<std::string, NodeInfo>& node_info,
  std::vector<std::vector<std::string>>& level
);