#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

static void normalize_rows(float* data, unsigned num, unsigned dim) {
  for (unsigned row = 0; row < num; ++row) {
    float* x = data + (size_t)row * dim;
    float norm = 0.0f;
    for (unsigned i = 0; i < dim; ++i) {
      norm += x[i] * x[i];
    }
    norm = std::sqrt(norm);
    if (norm <= std::numeric_limits<float>::epsilon()) continue;
    float inv_norm = 1.0f / norm;
    for (unsigned i = 0; i < dim; ++i) {
      x[i] *= inv_norm;
    }
  }
}

static float query_inv_norm(const float* query, unsigned dim) {
  float norm = 0.0f;
  for (unsigned i = 0; i < dim; ++i) {
    norm += query[i] * query[i];
  }
  norm = std::sqrt(norm);
  if (norm <= std::numeric_limits<float>::epsilon()) return 0.0f;
  return 1.0f / norm;
}

static float unit_cos_distance(const float* query, const float* unit_b,
                               unsigned dim, float inv_query_norm) {
  if (inv_query_norm <= 0.0f) {
    return std::numeric_limits<float>::infinity();
  }
  float dot = 0.0f;
  for (unsigned i = 0; i < dim; ++i) {
    dot += query[i] * unit_b[i];
  }
  return 1.0f - dot * inv_query_norm;
}

static float unit_cos_similarity(const float* query, const float* unit_b,
                                 unsigned dim, float inv_query_norm) {
  if (inv_query_norm <= 0.0f) {
    return -std::numeric_limits<float>::infinity();
  }
  float dot = 0.0f;
  for (unsigned i = 0; i < dim; ++i) {
    dot += query[i] * unit_b[i];
  }
  return dot * inv_query_norm;
}

static float unit_dot(const float* a, const float* b, unsigned dim) {
  float dot = 0.0f;
  for (unsigned i = 0; i < dim; ++i) {
    dot += a[i] * b[i];
  }
  return dot;
}

static std::vector<std::vector<unsigned> > build_hub_graph(
    const float* unit_embeddings, unsigned num, unsigned dim, unsigned degree) {
  degree = std::min(degree, num > 0 ? num - 1 : 0);
  std::vector<std::vector<unsigned> > graph(num);
  for (unsigned i = 0; i < num; ++i) {
    std::vector<std::pair<float, unsigned> > scores;
    scores.reserve(num > 0 ? num - 1 : 0);
    const float* xi = unit_embeddings + (size_t)i * dim;
    for (unsigned j = 0; j < num; ++j) {
      if (i == j) continue;
      const float* xj = unit_embeddings + (size_t)j * dim;
      scores.push_back({-unit_dot(xi, xj, dim), j});
    }
    if (degree < scores.size()) {
      std::partial_sort(scores.begin(), scores.begin() + degree, scores.end());
      scores.resize(degree);
    } else {
      std::sort(scores.begin(), scores.end());
    }
    graph[i].reserve(scores.size());
    for (auto &item : scores) graph[i].push_back(item.second);
  }
  return graph;
}

static std::vector<unsigned> select_hub_starts(
    const float* unit_embeddings, unsigned num, unsigned dim, unsigned start_count) {
  start_count = std::max(1u, std::min(start_count, num));
  std::vector<std::pair<float, unsigned> > centrality;
  centrality.reserve(num);
  for (unsigned i = 0; i < num; ++i) {
    const float* xi = unit_embeddings + (size_t)i * dim;
    float sum = 0.0f;
    for (unsigned j = 0; j < num; ++j) {
      if (i == j) continue;
      sum += unit_dot(xi, unit_embeddings + (size_t)j * dim, dim);
    }
    centrality.push_back({-sum, i});
  }
  std::partial_sort(centrality.begin(), centrality.begin() + start_count, centrality.end());
  std::vector<unsigned> starts;
  starts.reserve(start_count);
  for (unsigned i = 0; i < start_count; ++i) starts.push_back(centrality[i].second);
  return starts;
}

static std::vector<unsigned> select_hubs_by_scan(
    const float* query, const float* unit_embeddings, unsigned emb_num,
    unsigned dim, unsigned entry_count) {
  std::vector<std::pair<float, unsigned> > scores;
  scores.reserve(emb_num);
  float inv_norm = query_inv_norm(query, dim);
  for (unsigned j = 0; j < emb_num; ++j) {
    float dist = unit_cos_distance(query, unit_embeddings + (size_t)j * dim, dim, inv_norm);
    scores.push_back({dist, j});
  }
  unsigned take = std::min(entry_count, (unsigned)scores.size());
  std::partial_sort(scores.begin(), scores.begin() + take, scores.end());
  std::vector<unsigned> hubs;
  hubs.reserve(take);
  for (unsigned j = 0; j < take; ++j) hubs.push_back(scores[j].second);
  return hubs;
}

static std::vector<unsigned> select_hubs_by_navigation(
    const float* query, const float* unit_embeddings,
    const std::vector<std::vector<unsigned> > &hub_graph,
    const std::vector<unsigned> &starts, unsigned emb_num, unsigned dim,
    unsigned entry_count, unsigned hops) {
  std::vector<float> scores(emb_num, -std::numeric_limits<float>::infinity());
  std::vector<char> discovered(emb_num, 0);
  std::vector<char> expanded(emb_num, 0);
  std::vector<unsigned> candidates;
  candidates.reserve(starts.size() + hops * 16);
  float inv_norm = query_inv_norm(query, dim);
  for (unsigned start : starts) {
    if (start >= emb_num || discovered[start]) continue;
    scores[start] = unit_cos_similarity(query, unit_embeddings + (size_t)start * dim, dim, inv_norm);
    discovered[start] = 1;
    candidates.push_back(start);
  }

  for (unsigned step = 0; step < hops; ++step) {
    int best_pos = -1;
    float best_score = -std::numeric_limits<float>::infinity();
    for (unsigned pos = 0; pos < candidates.size(); ++pos) {
      unsigned hub = candidates[pos];
      if (!expanded[hub] && scores[hub] > best_score) {
        best_score = scores[hub];
        best_pos = (int)pos;
      }
    }
    if (best_pos < 0) break;
    unsigned hub = candidates[(unsigned)best_pos];
    expanded[hub] = 1;
    for (unsigned neigh : hub_graph[hub]) {
      if (neigh >= emb_num || discovered[neigh]) continue;
      scores[neigh] = unit_cos_similarity(query, unit_embeddings + (size_t)neigh * dim, dim, inv_norm);
      discovered[neigh] = 1;
      candidates.push_back(neigh);
    }
  }

  if (candidates.empty()) return {};
  std::vector<std::pair<float, unsigned> > ranked;
  ranked.reserve(candidates.size());
  for (unsigned hub : candidates) ranked.push_back({-scores[hub], hub});
  unsigned take = std::min(entry_count, (unsigned)ranked.size());
  std::partial_sort(ranked.begin(), ranked.begin() + take, ranked.end());
  std::vector<unsigned> hubs;
  hubs.reserve(take);
  for (unsigned i = 0; i < take; ++i) hubs.push_back(ranked[i].second);
  return hubs;
}

void load_data(const char* filename, float*& data, unsigned& num,
               unsigned& dim) {
  std::ifstream in(filename, std::ios::binary);
  std::cout << "opening: " << filename << '\n';
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[(size_t)num * (size_t)dim];
  std::cout << "data dimension: " << dim << ", num: " << num << std::endl;

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}

void save_result(const char* filename, std::vector<std::vector<unsigned> >& results) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < results.size(); i++) {
    unsigned GK = (unsigned)results[i].size();
    out.write((char*)&GK, sizeof(unsigned));
    out.write((char*)results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

int main(int argc, char** argv) {
  if (argc < 9 || argc > 14) {
    std::cout << argv[0]
              << " data_file query_file nsg_path search_L search_K hub_path embedding_path result_path"
              << " [entry_count] [selector_mode=scan|navigate] [gate_degree] [gate_hops] [gate_starts]"
              << std::endl;
    exit(-1);
  }
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);
  float* query_load = NULL;
  unsigned query_num, query_dim;
  load_data(argv[2], query_load, query_num, query_dim);
  assert(dim == query_dim);

  unsigned embadding_num, embadding_dim;
  float* embadding_data = nullptr;
  std::string embadding_path = argv[7];
  if (embadding_path != "nullptr") {
    load_data(argv[7], embadding_data, embadding_num, embadding_dim);
    assert(embadding_dim == query_dim);
    normalize_rows(embadding_data, embadding_num, embadding_dim);
  }

  unsigned L = (unsigned)atoi(argv[4]);
  unsigned K = (unsigned)atoi(argv[5]);
  unsigned entry_count = argc >= 10 ? (unsigned)atoi(argv[9]) : 3;
  std::string selector_mode = argc >= 11 ? argv[10] : "scan";
  unsigned gate_degree = argc >= 12 ? (unsigned)atoi(argv[11]) : 16;
  unsigned gate_hops = argc >= 13 ? (unsigned)atoi(argv[12]) : 5;
  unsigned gate_starts = argc >= 14 ? (unsigned)atoi(argv[13]) : 4;

  if (L < K) {
    std::cout << "search_L cannot be smaller than search_K!" << std::endl;
    exit(-1);
  }
  if (entry_count == 0) {
    std::cout << "entry_count cannot be zero!" << std::endl;
    exit(-1);
  }

  efanna2e::IndexNSG index(dim, points_num, efanna2e::FAST_L2, nullptr);
  index.Load(argv[3]);
  index.LoadKmeans(argv[6]);
  index.OptimizeGraph(data_load);

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);
  paras.Set<unsigned>("P_search", L);

  std::vector<std::vector<unsigned> > hub_graph;
  std::vector<unsigned> hub_starts;
  if (embadding_data != nullptr && selector_mode == "navigate") {
    hub_graph = build_hub_graph(embadding_data, embadding_num, embadding_dim, gate_degree);
    hub_starts = select_hub_starts(embadding_data, embadding_num, embadding_dim, gate_starts);
  }

  auto s = std::chrono::high_resolution_clock::now();
  double selector_seconds = 0.0;
  double graph_seconds = 0.0;
  std::vector<std::vector<unsigned> > res;
  res.reserve(query_num);
  for (unsigned i = 0; i < query_num; i++) {
    std::vector<unsigned> tmp(K);
    std::vector<unsigned> eps;

    auto selector_s = std::chrono::high_resolution_clock::now();
    if (embadding_data != nullptr) {
      const float* query = query_load + (size_t)i * dim;
      std::vector<unsigned> hub_ids;
      if (selector_mode == "navigate") {
        hub_ids = select_hubs_by_navigation(query, embadding_data, hub_graph, hub_starts, embadding_num, embadding_dim, entry_count, gate_hops);
      } else {
        hub_ids = select_hubs_by_scan(query, embadding_data, embadding_num, embadding_dim, entry_count);
      }
      for (unsigned ep_id : hub_ids) {
        unsigned ep = index.iter2_eps[ep_id / index.iter2_num][ep_id % index.iter2_num];
        eps.push_back(ep);
      }
    }
    auto selector_e = std::chrono::high_resolution_clock::now();
    selector_seconds += std::chrono::duration<double>(selector_e - selector_s).count();

    auto graph_s = std::chrono::high_resolution_clock::now();
    index.SearchWithOptGraphAndEntries(query_load + (size_t)i * dim, K, paras, tmp.data(), eps);
    auto graph_e = std::chrono::high_resolution_clock::now();
    graph_seconds += std::chrono::duration<double>(graph_e - graph_s).count();
    res.push_back(tmp);
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  std::cout << "search time: " << diff.count() << "\n";
  std::cout << "QPS: " << query_num / diff.count() << "\n";
  std::cout << "selector time: " << selector_seconds << "\n";
  std::cout << "graph time: " << graph_seconds << "\n";
  std::cout << "entry_count: " << entry_count << "\n";
  std::cout << "selector_mode: " << selector_mode << "\n";
  std::cout << "gate_degree: " << gate_degree << "\n";
  std::cout << "gate_hops: " << gate_hops << "\n";
  std::cout << "gate_starts: " << gate_starts << "\n";

  save_result(argv[8], res);

  delete[] data_load;
  delete[] query_load;
  delete[] embadding_data;
  return 0;
}
