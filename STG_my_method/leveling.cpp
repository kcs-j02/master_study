#include "leveling.hpp"

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <algorithm>

void make_nodeinfo(
  tf::Taskflow& tf,
  std::unordered_map<std::string, NodeInfo>& node_info
) {
  node_info.clear();

  tf.for_each_task([&](tf::Task t) {
    NodeInfo& ni = node_info[t.name()];
    ni.name = t.name();
    ni.indeg = 0;
    ni.preds.clear();
    ni.stream_id = -1;

    t.for_each_predecessor([&](tf::Task p) {
      ni.indeg++;
      ni.preds.push_back(p.name());
    });
  });
}

void add_level(
  std::vector<std::vector<std::string>>& level,
  int& count_level
) {
  if ((int)level.size() <= count_level) {
    level.resize(count_level + 1);
  }
  level[count_level].clear();
}

void push_node_to_level(
  int count_level,
  tf::Taskflow& tf,
  std::unordered_map<std::string, NodeInfo>& node_info,
  std::vector<std::vector<std::string>>& level,
  int& node_count
) {
  tf.for_each_task([&](tf::Task t) {
    auto& ni = node_info[t.name()];
    if (ni.indeg == 0) {
      level[count_level].push_back(t.name());
      ni.indeg = -1;
      node_count++;
    }
  });
}

void remove_node(
  int& count_level,
  std::vector<std::vector<std::string>>& level,
  std::unordered_map<std::string, NodeInfo>& node_info
) {
  std::unordered_set<std::string> pre(
    level[count_level].begin(),
    level[count_level].end()
  );

  for (auto& [name, ni] : node_info) {
    if (ni.indeg < 0) {
      continue;
    }

    int before = (int)ni.preds.size();

    ni.preds.erase(
      std::remove_if(
        ni.preds.begin(),
        ni.preds.end(),
        [&](const std::string& p) {
          return pre.count(p) != 0;
        }
      ),
      ni.preds.end()
    );

    int removed = before - (int)ni.preds.size();
    if (removed > 0) {
      ni.indeg -= removed;
      if (ni.indeg < 0) {
        ni.indeg = 0;
      }
    }
  }

  count_level++;
}

void check_nodes(
  tf::Taskflow& tf,
  std::unordered_map<std::string, NodeInfo>& node_info
) {
  tf.for_each_task([&](tf::Task t) {
    auto& ni = node_info[t.name()];
    std::cout << ni.name
              << " 入次数=" << ni.indeg
              << " 依存元=[";
    for (const auto& pn : ni.preds) {
      std::cout << pn << " ";
    }
    std::cout << "] stream=" << ni.stream_id << std::endl;
  });
}

void check_level_all(
  const std::vector<std::vector<std::string>>& level
) {
  for (size_t i = 0; i < level.size(); ++i) {
    std::cout << "level[" << i << "] = {";
    for (size_t j = 0; j < level[i].size(); ++j) {
      if (j) {
        std::cout << ", ";
      }
      std::cout << level[i][j];
    }
    std::cout << "}\n";
  }
}

void DFG_based_task_leveling(
  tf::Taskflow& tf,
  std::unordered_map<std::string, NodeInfo>& node_info,
  std::vector<std::vector<std::string>>& level
) {
  int count_level = 0;
  int node_count = 0;
  int flag = 1;

  level.clear();
  make_nodeinfo(tf, node_info);

  while (flag) {
    add_level(level, count_level);
    push_node_to_level(count_level, tf, node_info, level, node_count);
    remove_node(count_level, level, node_info);

    if (node_count == (int)node_info.size()) {
      flag = 0;
    }
  }
}