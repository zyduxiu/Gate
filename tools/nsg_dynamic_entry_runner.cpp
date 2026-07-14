#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <sstream>
#include <unordered_set>
#include <numeric>
#include <vector>

namespace {

struct Matrix {
    uint32_t rows = 0;
    uint32_t dim = 0;
    std::vector<float> values;

    const float* row(uint32_t i) const {
        return values.data() + static_cast<size_t>(i) * dim;
    }
};

struct IdMatrix {
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<uint32_t> values;

    const uint32_t* row(uint32_t i) const {
        return values.data() + static_cast<size_t>(i) * cols;
    }
};

struct Graph {
    uint32_t width = 0;
    uint32_t ep = 0;
    std::vector<std::vector<uint32_t>> adj;
};

struct Args {
    std::string initial_fbin;
    std::string insert_fbin;
    std::string query_fbin;
    std::string truth_bin;
    std::string delete_u32;
    std::string initial_nsg;
    std::string all_knn_graph;
    std::string anchors_txt;
    std::string query_entries_ivecs;
    std::string hard_repair_ivecs;
    std::string mode = "nsg_ep";
    std::string out_json;
    std::string out_per_query;
    std::string out_anchors_txt;
    std::string out_anchor_costs_csv;
    std::string online_mlp_model;
    uint32_t search_l = 120;
    uint32_t topk = 10;
    uint32_t entries = 8;
    uint32_t insert_degree = 32;
    uint32_t max_degree = 64;
    uint32_t repair_degree = 0;
    uint32_t maintain_batch = 1000;
    uint32_t maintain_changes = 4;
    uint32_t maintain_sample = 256;
    uint32_t hard_repair_local_degree = 0;
    uint32_t hard_repair_bridge_degree = 0;
    uint32_t anchor_cost_query_limit = 0;
    std::array<double, 4> online_score_weights{{0.45, 0.20, 0.30, 0.05}};
    std::array<double, 3> online_retire_weights{{0.55, 0.25, 0.20}};
    double online_retire_load_weight = 0.0;
    double online_query_gain_weight = 0.0;
    double online_query_load_weight = 0.0;
};

struct OnlineMlpModel {
    uint32_t hidden = 0;
    std::vector<double> w1;
    std::vector<double> b1;
    std::vector<double> w2;
    double b2 = 0.0;

    bool enabled() const {
        return hidden > 0;
    }

    double score(const std::array<double, 4>& x) const {
        double out = b2;
        for (uint32_t h = 0; h < hidden; ++h) {
            double z = b1[h];
            for (uint32_t j = 0; j < 4; ++j) {
                z += w1[static_cast<size_t>(h) * 4 + j] * x[j];
            }
            const double relu = z > 0.0 ? z : 0.0;
            out += w2[h] * relu;
        }
        return out;
    }
};

struct MaintenanceStats {
    uint32_t initial_anchors = 0;
    uint32_t final_anchors = 0;
    uint32_t removed_dead = 0;
    uint32_t added = 0;
    uint32_t retired = 0;
    uint32_t rounds = 0;
    double maintenance_distance_computations = 0.0;
    double maintenance_seconds = 0.0;
};

struct Candidate {
    uint32_t id = 0;
    float distance = 0.0f;
    bool flag = true;
};

struct QueryWorkspace {
    std::vector<uint32_t> init_seen;
    std::vector<uint32_t> search_seen;
    std::vector<Candidate> retset;
    uint32_t init_stamp = 1;
    uint32_t search_stamp = 1;

    uint32_t next_init_stamp(size_t rows) {
        if (init_seen.size() < rows) init_seen.assign(rows, 0);
        ++init_stamp;
        if (init_stamp == 0) {
            std::fill(init_seen.begin(), init_seen.end(), 0);
            init_stamp = 1;
        }
        return init_stamp;
    }

    uint32_t next_search_stamp(size_t rows) {
        if (search_seen.size() < rows) search_seen.assign(rows, 0);
        ++search_stamp;
        if (search_stamp == 0) {
            std::fill(search_seen.begin(), search_seen.end(), 0);
            search_stamp = 1;
        }
        return search_stamp;
    }
};

[[noreturn]] void die(const std::string& message) {
    throw std::runtime_error(message);
}

std::string require_value(int& i, int argc, char** argv) {
    if (i + 1 >= argc) die(std::string("missing value for ") + argv[i]);
    ++i;
    return argv[i];
}

std::vector<double> parse_double_list(const std::string& value, size_t expected, const std::string& name) {
    std::vector<double> out;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) out.push_back(std::stod(item));
    }
    if (out.size() != expected) die(name + " expects " + std::to_string(expected) + " comma-separated values");
    return out;
}

OnlineMlpModel read_online_mlp_model(const std::string& path) {
    if (path.empty()) return OnlineMlpModel{};
    std::ifstream in(path);
    if (!in) die("cannot open online MLP model: " + path);
    OnlineMlpModel model;
    in >> model.hidden;
    if (!in || model.hidden == 0 || model.hidden > 256) die("invalid online MLP hidden size in: " + path);
    model.w1.resize(static_cast<size_t>(model.hidden) * 4);
    model.b1.resize(model.hidden);
    model.w2.resize(model.hidden);
    for (double& value : model.w1) in >> value;
    for (double& value : model.b1) in >> value;
    for (double& value : model.w2) in >> value;
    in >> model.b2;
    if (!in) die("truncated online MLP model: " + path);
    return model;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--initial-fbin") args.initial_fbin = require_value(i, argc, argv);
        else if (key == "--insert-fbin") args.insert_fbin = require_value(i, argc, argv);
        else if (key == "--query-fbin") args.query_fbin = require_value(i, argc, argv);
        else if (key == "--truth-bin") args.truth_bin = require_value(i, argc, argv);
        else if (key == "--delete-u32") args.delete_u32 = require_value(i, argc, argv);
        else if (key == "--initial-nsg") args.initial_nsg = require_value(i, argc, argv);
        else if (key == "--all-knn-graph") args.all_knn_graph = require_value(i, argc, argv);
        else if (key == "--anchors-txt") args.anchors_txt = require_value(i, argc, argv);
        else if (key == "--query-entries-ivecs") args.query_entries_ivecs = require_value(i, argc, argv);
        else if (key == "--hard-repair-ivecs") args.hard_repair_ivecs = require_value(i, argc, argv);
        else if (key == "--mode") args.mode = require_value(i, argc, argv);
        else if (key == "--out-json") args.out_json = require_value(i, argc, argv);
        else if (key == "--out-per-query") args.out_per_query = require_value(i, argc, argv);
        else if (key == "--out-anchors-txt") args.out_anchors_txt = require_value(i, argc, argv);
        else if (key == "--out-anchor-costs-csv") args.out_anchor_costs_csv = require_value(i, argc, argv);
        else if (key == "--online-mlp-model") args.online_mlp_model = require_value(i, argc, argv);
        else if (key == "--search-l") args.search_l = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--topk") args.topk = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--entries") args.entries = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--insert-degree") args.insert_degree = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--max-degree") args.max_degree = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--repair-degree") args.repair_degree = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--maintain-batch") args.maintain_batch = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--maintain-changes") args.maintain_changes = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--maintain-sample") args.maintain_sample = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--hard-repair-local-degree") args.hard_repair_local_degree = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--hard-repair-bridge-degree") args.hard_repair_bridge_degree = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--anchor-cost-query-limit") args.anchor_cost_query_limit = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--online-score-weights") {
            const std::vector<double> values = parse_double_list(require_value(i, argc, argv), 4, key);
            std::copy(values.begin(), values.end(), args.online_score_weights.begin());
        }
        else if (key == "--online-retire-weights") {
            const std::vector<double> values = parse_double_list(require_value(i, argc, argv), 3, key);
            std::copy(values.begin(), values.end(), args.online_retire_weights.begin());
        }
        else if (key == "--online-retire-load-weight") args.online_retire_load_weight = std::stod(require_value(i, argc, argv));
        else if (key == "--online-query-gain-weight") args.online_query_gain_weight = std::stod(require_value(i, argc, argv));
        else if (key == "--online-query-load-weight") args.online_query_load_weight = std::stod(require_value(i, argc, argv));
        else die("unknown argument: " + key);
    }
    if (args.initial_fbin.empty()) die("--initial-fbin is required");
    if (args.insert_fbin.empty()) die("--insert-fbin is required");
    if (args.query_fbin.empty()) die("--query-fbin is required");
    if (args.truth_bin.empty()) die("--truth-bin is required");
    if (args.delete_u32.empty()) die("--delete-u32 is required");
    if (args.initial_nsg.empty()) die("--initial-nsg is required");
    if (args.all_knn_graph.empty()) die("--all-knn-graph is required");
    const bool query_entries_mode = args.mode == "query_entries" || args.mode.rfind("query_entries_", 0) == 0;
    if (args.mode != "nsg_ep" && args.mode != "online_density" && args.mode != "online_hub_density" && !query_entries_mode && args.anchors_txt.empty()) die("--anchors-txt is required for anchor modes");
    if (query_entries_mode && args.query_entries_ivecs.empty()) die("--query-entries-ivecs is required for query_entries mode");
    if (args.out_json.empty()) die("--out-json is required");
    if (args.search_l < args.topk) die("--search-l must be >= --topk");
    if (args.maintain_batch == 0) die("--maintain-batch must be positive");
    if (args.maintain_sample == 0) die("--maintain-sample must be positive");
    return args;
}

Matrix read_fbin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) die("cannot open fbin: " + path);
    Matrix m;
    in.read(reinterpret_cast<char*>(&m.rows), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&m.dim), sizeof(uint32_t));
    if (!in || m.dim == 0) die("invalid fbin header: " + path);
    m.values.resize(static_cast<size_t>(m.rows) * m.dim);
    if (!m.values.empty()) {
        in.read(reinterpret_cast<char*>(m.values.data()), static_cast<std::streamsize>(m.values.size() * sizeof(float)));
        if (!in) die("truncated fbin: " + path);
    }
    return m;
}

IdMatrix read_id_bin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) die("cannot open id bin: " + path);
    IdMatrix m;
    in.read(reinterpret_cast<char*>(&m.rows), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&m.cols), sizeof(uint32_t));
    if (!in || m.rows == 0 || m.cols == 0) die("invalid id bin header: " + path);
    m.values.resize(static_cast<size_t>(m.rows) * m.cols);
    in.read(reinterpret_cast<char*>(m.values.data()), static_cast<std::streamsize>(m.values.size() * sizeof(uint32_t)));
    if (!in) die("truncated id bin: " + path);
    return m;
}

std::vector<uint32_t> read_u32_list(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) die("cannot open u32 list: " + path);
    uint32_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
    std::vector<uint32_t> values(count);
    if (count > 0) {
        in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(uint32_t)));
    }
    if (!in && count > 0) die("truncated u32 list: " + path);
    return values;
}

Graph read_nsg(const std::string& path, uint32_t expected_rows) {
    std::ifstream in(path, std::ios::binary);
    if (!in) die("cannot open NSG index: " + path);
    Graph graph;
    in.read(reinterpret_cast<char*>(&graph.width), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&graph.ep), sizeof(uint32_t));
    if (!in) die("invalid NSG header: " + path);
    graph.adj.resize(expected_rows);
    for (uint32_t i = 0; i < expected_rows; ++i) {
        uint32_t degree = 0;
        in.read(reinterpret_cast<char*>(&degree), sizeof(uint32_t));
        if (!in) die("truncated NSG graph degree: " + path);
        graph.adj[i].resize(degree);
        if (degree > 0) {
            in.read(reinterpret_cast<char*>(graph.adj[i].data()), degree * sizeof(uint32_t));
            if (!in) die("truncated NSG graph row: " + path);
        }
    }
    if (graph.ep >= expected_rows) die("invalid NSG entry point");
    return graph;
}

IdMatrix read_ivecs_graph(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) die("cannot open ivecs graph: " + path);
    int32_t dim_i = 0;
    in.read(reinterpret_cast<char*>(&dim_i), sizeof(int32_t));
    if (!in || dim_i <= 0) die("invalid ivecs graph header: " + path);
    const uint32_t dim = static_cast<uint32_t>(dim_i);
    in.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(in.tellg());
    const size_t row_bytes = sizeof(int32_t) + static_cast<size_t>(dim) * sizeof(uint32_t);
    if (file_size % row_bytes != 0) die("invalid ivecs graph size: " + path);
    const uint32_t rows = static_cast<uint32_t>(file_size / row_bytes);
    IdMatrix matrix;
    matrix.rows = rows;
    matrix.cols = dim;
    matrix.values.resize(static_cast<size_t>(rows) * dim);
    in.seekg(0, std::ios::beg);
    for (uint32_t i = 0; i < rows; ++i) {
        int32_t row_dim = 0;
        in.read(reinterpret_cast<char*>(&row_dim), sizeof(int32_t));
        if (row_dim != dim_i) die("inconsistent ivecs graph dim: " + path);
        in.read(reinterpret_cast<char*>(matrix.values.data() + static_cast<size_t>(i) * dim), dim * sizeof(uint32_t));
        if (!in) die("truncated ivecs graph: " + path);
    }
    return matrix;
}

std::vector<uint32_t> read_anchors(const std::string& path) {
    std::ifstream in(path);
    if (!in) die("cannot open anchors: " + path);
    std::vector<uint32_t> anchors;
    uint32_t value = 0;
    while (in >> value) anchors.push_back(value);
    return anchors;
}

void write_anchors(const std::string& path, const std::vector<uint32_t>& anchors) {
    if (path.empty()) return;
    std::ofstream out(path);
    if (!out) die("cannot write anchors: " + path);
    for (uint32_t anchor : anchors) {
        out << anchor << "\n";
    }
}

float l2(const float* a, const float* b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

Matrix merge_matrices(const Matrix& initial, const Matrix& inserts) {
    if (initial.dim != inserts.dim) die("initial/insert dimension mismatch");
    Matrix all;
    all.rows = initial.rows + inserts.rows;
    all.dim = initial.dim;
    all.values.reserve(static_cast<size_t>(all.rows) * all.dim);
    all.values.insert(all.values.end(), initial.values.begin(), initial.values.end());
    all.values.insert(all.values.end(), inserts.values.begin(), inserts.values.end());
    return all;
}

void prune_node(Graph& graph, const Matrix& all, const std::vector<uint8_t>& alive, uint32_t node, uint32_t max_degree) {
    if (node >= graph.adj.size() || !alive[node]) return;
    std::vector<uint32_t> cleaned;
    cleaned.reserve(graph.adj[node].size());
    std::unordered_set<uint32_t> seen;
    for (uint32_t other : graph.adj[node]) {
        if (other < graph.adj.size() && other != node && alive[other] && seen.insert(other).second) {
            cleaned.push_back(other);
        }
    }
    if (cleaned.size() > max_degree) {
        std::vector<std::pair<float, uint32_t>> scored;
        scored.reserve(cleaned.size());
        for (uint32_t other : cleaned) {
            scored.emplace_back(l2(all.row(node), all.row(other), all.dim), other);
        }
        std::partial_sort(scored.begin(), scored.begin() + max_degree, scored.end());
        cleaned.clear();
        for (uint32_t i = 0; i < max_degree; ++i) cleaned.push_back(scored[i].second);
    }
    graph.adj[node].swap(cleaned);
}

uint32_t repair_graph_from_knn(
    Graph& graph,
    const Matrix& all,
    const IdMatrix& all_knn,
    const std::vector<uint8_t>& alive,
    uint32_t initial_rows,
    uint32_t repair_degree,
    uint32_t max_degree) {
    if (repair_degree == 0) return 0;
    if (all_knn.rows != all.rows) die("all KNN graph row count mismatch");
    std::unordered_set<uint32_t> touched;
    for (uint32_t node = 0; node < graph.adj.size(); ++node) {
        if (!alive[node]) continue;
        bool needs_repair = node >= initial_rows;
        if (!needs_repair) {
            for (uint32_t other : graph.adj[node]) {
                if (other >= alive.size() || !alive[other]) {
                    needs_repair = true;
                    break;
                }
            }
        }
        if (!needs_repair) continue;
        std::unordered_set<uint32_t> seen;
        std::vector<uint32_t> cleaned;
        cleaned.reserve(graph.adj[node].size());
        for (uint32_t other : graph.adj[node]) {
            if (other < graph.adj.size() && other != node && alive[other] && seen.insert(other).second) {
                cleaned.push_back(other);
            }
        }
        graph.adj[node].swap(cleaned);
        if (graph.adj[node].size() >= repair_degree) continue;

        const uint32_t* row = all_knn.row(node);
        for (uint32_t j = 0; j < all_knn.cols && graph.adj[node].size() < repair_degree; ++j) {
            const uint32_t other = row[j];
            if (other >= graph.adj.size() || other == node || !alive[other]) continue;
            if (std::find(graph.adj[node].begin(), graph.adj[node].end(), other) != graph.adj[node].end()) continue;
            graph.adj[node].push_back(other);
            graph.adj[other].push_back(node);
            touched.insert(node);
            touched.insert(other);
        }
    }
    for (uint32_t node : touched) {
        prune_node(graph, all, alive, node, max_degree);
    }
    return static_cast<uint32_t>(touched.size());
}

bool add_edge(Graph& graph, const std::vector<uint8_t>& alive, uint32_t a, uint32_t b, std::unordered_set<uint32_t>& touched) {
    if (a >= graph.adj.size() || b >= graph.adj.size() || a == b || !alive[a] || !alive[b]) return false;
    if (std::find(graph.adj[a].begin(), graph.adj[a].end(), b) == graph.adj[a].end()) {
        graph.adj[a].push_back(b);
        touched.insert(a);
    }
    if (std::find(graph.adj[b].begin(), graph.adj[b].end(), a) == graph.adj[b].end()) {
        graph.adj[b].push_back(a);
        touched.insert(b);
    }
    return true;
}

uint32_t repair_hard_regions(
    Graph& graph,
    const Matrix& all,
    const IdMatrix& hard_regions,
    const std::vector<uint8_t>& alive,
    uint32_t local_degree,
    uint32_t bridge_degree,
    uint32_t max_degree,
    uint32_t entry_point) {
    if (hard_regions.rows == 0 || (local_degree == 0 && bridge_degree == 0)) return 0;
    std::unordered_set<uint32_t> touched;
    for (uint32_t row_id = 0; row_id < hard_regions.rows; ++row_id) {
        std::vector<uint32_t> region;
        region.reserve(hard_regions.cols);
        const uint32_t* row = hard_regions.row(row_id);
        for (uint32_t j = 0; j < hard_regions.cols; ++j) {
            const uint32_t node = row[j];
            if (node < graph.adj.size() && alive[node] && std::find(region.begin(), region.end(), node) == region.end()) {
                region.push_back(node);
            }
        }
        if (region.empty()) continue;

        const uint32_t bridges = std::min<uint32_t>(bridge_degree, static_cast<uint32_t>(region.size()));
        for (uint32_t i = 0; i < bridges; ++i) {
            add_edge(graph, alive, entry_point, region[i], touched);
        }

        if (local_degree == 0) continue;
        for (uint32_t node : region) {
            std::vector<std::pair<float, uint32_t>> scored;
            scored.reserve(region.size());
            for (uint32_t other : region) {
                if (other != node) scored.emplace_back(l2(all.row(node), all.row(other), all.dim), other);
            }
            const uint32_t take = std::min<uint32_t>(local_degree, static_cast<uint32_t>(scored.size()));
            if (take == 0) continue;
            std::partial_sort(scored.begin(), scored.begin() + take, scored.end());
            for (uint32_t i = 0; i < take; ++i) {
                add_edge(graph, alive, node, scored[i].second, touched);
            }
        }
    }
    for (uint32_t node : touched) {
        prune_node(graph, all, alive, node, max_degree);
    }
    return static_cast<uint32_t>(touched.size());
}

Graph build_dynamic_graph(
    const Graph& initial_graph,
    const Matrix& all,
    const IdMatrix& all_knn,
    uint32_t initial_rows,
    const std::vector<uint8_t>& alive,
    uint32_t insert_degree,
    uint32_t max_degree,
    uint32_t& touched_nodes) {
    if (all_knn.rows != all.rows) die("all KNN graph row count mismatch");
    Graph graph = initial_graph;
    graph.adj.resize(all.rows);
    std::unordered_set<uint32_t> touched;

    for (uint32_t node = initial_rows; node < all.rows; ++node) {
        if (!alive[node]) continue;
        const uint32_t* row = all_knn.row(node);
        uint32_t added = 0;
        for (uint32_t j = 0; j < all_knn.cols && added < insert_degree; ++j) {
            const uint32_t other = row[j];
            if (other >= all.rows || other == node || !alive[other]) continue;
            graph.adj[node].push_back(other);
            graph.adj[other].push_back(node);
            touched.insert(node);
            touched.insert(other);
            ++added;
        }
    }

    for (uint32_t node : touched) {
        prune_node(graph, all, alive, node, max_degree);
    }
    touched_nodes = static_cast<uint32_t>(touched.size());
    return graph;
}

bool by_distance(const Candidate& left, const Candidate& right) {
    if (left.distance != right.distance) return left.distance < right.distance;
    return left.id < right.id;
}

uint32_t insert_into_pool(std::vector<Candidate>& pool, uint32_t pool_size, const Candidate& candidate) {
    uint32_t left = 0;
    uint32_t right = pool_size;
    while (left < right) {
        const uint32_t mid = (left + right) / 2;
        if (by_distance(pool[mid], candidate)) left = mid + 1;
        else right = mid;
    }
    if (left >= pool_size) return pool_size;
    if (pool[left].id == candidate.id) return pool_size;
    for (uint32_t i = pool_size; i > left; --i) pool[i] = pool[i - 1];
    pool[left] = candidate;
    return left;
}

std::vector<uint32_t> closest_anchors(
    const Matrix& all,
    const std::vector<uint8_t>& alive,
    const std::vector<uint32_t>& anchors,
    const float* query,
    uint32_t entries) {
    std::vector<std::pair<float, uint32_t>> scored;
    scored.reserve(anchors.size());
    for (uint32_t anchor : anchors) {
        if (anchor < all.rows && alive[anchor]) {
            scored.emplace_back(l2(all.row(anchor), query, all.dim), anchor);
        }
    }
    if (scored.empty()) return {};
    const uint32_t take = std::min<uint32_t>(entries, static_cast<uint32_t>(scored.size()));
    std::partial_sort(scored.begin(), scored.begin() + take, scored.end());
    std::vector<uint32_t> result;
    result.reserve(take);
    for (uint32_t i = 0; i < take; ++i) result.push_back(scored[i].second);
    return result;
}

double nearest_anchor_distance(
    const Matrix& all,
    const std::vector<uint32_t>& anchors,
    uint32_t node,
    uint32_t skip,
    double& distance_computations) {
    double best = std::numeric_limits<double>::infinity();
    for (uint32_t anchor : anchors) {
        if (anchor == skip || anchor >= all.rows) continue;
        best = std::min(best, static_cast<double>(l2(all.row(node), all.row(anchor), all.dim)));
        distance_computations += 1.0;
    }
    return best;
}

double local_density_score(
    const Matrix& all,
    const IdMatrix& all_knn,
    const std::vector<uint8_t>& alive,
    uint32_t node,
    double& distance_computations) {
    if (node >= all_knn.rows) return 0.0;
    double total = 0.0;
    uint32_t count = 0;
    const uint32_t* row = all_knn.row(node);
    for (uint32_t j = 0; j < all_knn.cols && count < 16; ++j) {
        const uint32_t other = row[j];
        if (other >= all.rows || other == node || !alive[other]) continue;
        total += std::sqrt(std::max(0.0f, l2(all.row(node), all.row(other), all.dim)));
        distance_computations += 1.0;
        ++count;
    }
    if (count == 0) return 0.0;
    return 1.0 / (total / static_cast<double>(count) + 1e-6);
}

std::vector<double> graph_hubness_scores(const Graph& graph, const std::vector<uint8_t>& alive) {
    std::vector<double> indegree(graph.adj.size(), 0.0);
    for (uint32_t node = 0; node < graph.adj.size(); ++node) {
        if (node >= alive.size() || !alive[node]) continue;
        for (uint32_t neighbor : graph.adj[node]) {
            if (neighbor < graph.adj.size() && neighbor < alive.size() && alive[neighbor]) {
                indegree[neighbor] += 1.0;
            }
        }
    }
    const auto [lo_it, hi_it] = std::minmax_element(indegree.begin(), indegree.end());
    const double lo = *lo_it;
    const double hi = *hi_it;
    if (hi - lo < 1e-12) return indegree;
    for (double& value : indegree) value = (value - lo) / (hi - lo);
    return indegree;
}

uint32_t select_online_candidate(
    const Matrix& all,
    const IdMatrix& all_knn,
    const Matrix* workload_queries,
    const std::vector<uint8_t>& alive,
    const std::vector<uint32_t>& anchors,
    const std::vector<double>* hubness,
    const std::array<double, 4>& score_weights,
    const OnlineMlpModel* mlp_model,
    double query_gain_weight,
    double query_load_weight,
    uint32_t workload_query_limit,
    uint32_t initial_rows,
    uint32_t begin,
    uint32_t end,
    uint32_t max_candidates,
    double& distance_computations) {
    std::unordered_set<uint32_t> anchor_set(anchors.begin(), anchors.end());
    std::vector<uint32_t> pool;
    pool.reserve(end > begin ? end - begin : 0);
    for (uint32_t node = begin; node < end && node < all.rows; ++node) {
        if (alive[node] && anchor_set.find(node) == anchor_set.end()) pool.push_back(node);
    }
    std::vector<uint32_t> candidates;
    if (pool.size() <= max_candidates) {
        candidates.swap(pool);
    } else {
        candidates.reserve(max_candidates);
        for (uint32_t i = 0; i < max_candidates; ++i) {
            const size_t pos = (static_cast<size_t>(i) * pool.size()) / max_candidates;
            candidates.push_back(pool[pos]);
        }
    }
    if (candidates.empty()) return std::numeric_limits<uint32_t>::max();

    std::vector<double> far(candidates.size(), 0.0);
    std::vector<double> density(candidates.size(), 0.0);
    std::vector<double> hub(candidates.size(), 0.0);
    std::vector<double> query_gain(candidates.size(), 0.0);
    std::vector<double> query_load(candidates.size(), 0.0);
    for (size_t i = 0; i < candidates.size(); ++i) {
        far[i] = nearest_anchor_distance(all, anchors, candidates[i], std::numeric_limits<uint32_t>::max(), distance_computations);
        if (!std::isfinite(far[i])) far[i] = 0.0;
        density[i] = local_density_score(all, all_knn, alive, candidates[i], distance_computations);
        if (hubness != nullptr && candidates[i] < hubness->size()) hub[i] = (*hubness)[candidates[i]];
    }
    if (workload_queries != nullptr && !anchors.empty() && workload_query_limit > 0 && (query_gain_weight > 0.0 || query_load_weight > 0.0)) {
        const uint32_t rows = std::min<uint32_t>(workload_query_limit, workload_queries->rows);
        std::vector<double> base_best(rows, std::numeric_limits<double>::infinity());
        for (uint32_t qi = 0; qi < rows; ++qi) {
            const float* query = workload_queries->row(qi);
            for (uint32_t anchor : anchors) {
                if (anchor >= all.rows) continue;
                const double dist = l2(all.row(anchor), query, all.dim);
                ++distance_computations;
                if (dist < base_best[qi]) base_best[qi] = dist;
            }
        }
        for (size_t i = 0; i < candidates.size(); ++i) {
            const uint32_t candidate = candidates[i];
            double gain_sum = 0.0;
            double load_sum = 0.0;
            for (uint32_t qi = 0; qi < rows; ++qi) {
                const float* query = workload_queries->row(qi);
                const double dist = l2(all.row(candidate), query, all.dim);
                ++distance_computations;
                if (dist < base_best[qi]) {
                    gain_sum += base_best[qi] - dist;
                    load_sum += 1.0;
                }
            }
            query_gain[i] = rows > 0 ? gain_sum / static_cast<double>(rows) : 0.0;
            query_load[i] = rows > 0 ? load_sum / static_cast<double>(rows) : 0.0;
        }
    }
    auto normalize = [](const std::vector<double>& values, size_t i) -> double {
        const auto [lo_it, hi_it] = std::minmax_element(values.begin(), values.end());
        const double lo = *lo_it;
        const double hi = *hi_it;
        if (hi - lo < 1e-12) return 0.0;
        return (values[i] - lo) / (hi - lo);
    };

    double best_score = -std::numeric_limits<double>::infinity();
    uint32_t best = candidates[0];
    for (size_t i = 0; i < candidates.size(); ++i) {
        const std::array<double, 4> features{{
            normalize(far, i),
            normalize(density, i),
            hubness != nullptr ? normalize(hub, i) : 0.0,
            candidates[i] >= initial_rows ? 1.0 : 0.0,
        }};
        double score = 0.60 * features[0] + 0.40 * features[1];
        if (mlp_model != nullptr && mlp_model->enabled()) {
            score = mlp_model->score(features);
        } else if (hubness != nullptr) {
            score =
                score_weights[0] * features[0] +
                score_weights[1] * features[1] +
                score_weights[2] * features[2] +
                score_weights[3] * features[3];
        }
        score += query_gain_weight * normalize(query_gain, i);
        score += query_load_weight * normalize(query_load, i);
        if (score > best_score) {
            best_score = score;
            best = candidates[i];
        }
    }
    return best;
}

uint32_t retire_redundant_anchor(
    const Matrix& all,
    const Matrix* workload_queries,
    std::vector<uint32_t>& anchors,
    uint32_t protected_anchor,
    uint32_t initial_rows,
    const std::vector<double>* hubness,
    const std::array<double, 3>& retire_weights,
    double retire_load_weight,
    uint32_t workload_query_limit,
    double& distance_computations) {
    if (anchors.size() <= 1) return std::numeric_limits<uint32_t>::max();
    std::vector<double> nearest_values;
    std::vector<double> hub_values;
    std::vector<double> inserted_values;
    std::vector<double> load_values;
    std::vector<size_t> positions;
    nearest_values.reserve(anchors.size());
    hub_values.reserve(anchors.size());
    inserted_values.reserve(anchors.size());
    load_values.reserve(anchors.size());
    positions.reserve(anchors.size());
    for (size_t i = 0; i < anchors.size(); ++i) {
        const uint32_t anchor = anchors[i];
        if (anchor == protected_anchor) continue;
        const double nearest = nearest_anchor_distance(all, anchors, anchor, anchor, distance_computations);
        nearest_values.push_back(std::isfinite(nearest) ? nearest : 0.0);
        hub_values.push_back((hubness != nullptr && anchor < hubness->size()) ? (*hubness)[anchor] : 0.0);
        inserted_values.push_back(anchor >= initial_rows ? 1.0 : 0.0);
        load_values.push_back(0.0);
        positions.push_back(i);
    }
    if (positions.empty()) return std::numeric_limits<uint32_t>::max();

    if (workload_queries != nullptr && retire_load_weight > 0.0 && workload_query_limit > 0) {
        const uint32_t rows = std::min<uint32_t>(workload_query_limit, workload_queries->rows);
        for (uint32_t qi = 0; qi < rows; ++qi) {
            const float* query = workload_queries->row(qi);
            double best_dist = std::numeric_limits<double>::infinity();
            size_t best_anchor_pos = anchors.size();
            for (size_t ai = 0; ai < anchors.size(); ++ai) {
                const uint32_t anchor = anchors[ai];
                if (anchor >= all.rows) continue;
                const double dist = l2(all.row(anchor), query, all.dim);
                ++distance_computations;
                if (dist < best_dist) {
                    best_dist = dist;
                    best_anchor_pos = ai;
                }
            }
            for (size_t i = 0; i < positions.size(); ++i) {
                if (positions[i] == best_anchor_pos) {
                    load_values[i] += 1.0;
                    break;
                }
            }
        }
    }

    auto normalize_at = [](const std::vector<double>& values, size_t i) -> double {
        const auto [lo_it, hi_it] = std::minmax_element(values.begin(), values.end());
        const double lo = *lo_it;
        const double hi = *hi_it;
        if (hi - lo < 1e-12) return 0.0;
        return (values[i] - lo) / (hi - lo);
    };
    double best_score = std::numeric_limits<double>::infinity();
    size_t remove_pos = anchors.size();
    for (size_t i = 0; i < positions.size(); ++i) {
        double score = nearest_values[i];
        if (hubness != nullptr) {
            // Retire anchors that are redundant, weak graph hubs, and stale with respect to the insert stream.
            score =
                retire_weights[0] * normalize_at(nearest_values, i) +
                retire_weights[1] * normalize_at(hub_values, i) +
                retire_weights[2] * inserted_values[i];
        }
        if (retire_load_weight > 0.0) {
            score += retire_load_weight * normalize_at(load_values, i);
        }
        if (score < best_score) {
            best_score = score;
            remove_pos = positions[i];
        }
    }
    if (remove_pos == anchors.size()) return std::numeric_limits<uint32_t>::max();
    const uint32_t removed = anchors[remove_pos];
    anchors.erase(anchors.begin() + static_cast<std::ptrdiff_t>(remove_pos));
    return removed;
}

void dedup_alive_anchors(std::vector<uint32_t>& anchors, const std::vector<uint8_t>& alive, MaintenanceStats& stats) {
    std::vector<uint32_t> cleaned;
    cleaned.reserve(anchors.size());
    std::unordered_set<uint32_t> seen;
    for (uint32_t anchor : anchors) {
        if (anchor >= alive.size() || !alive[anchor]) {
            ++stats.removed_dead;
            continue;
        }
        if (seen.insert(anchor).second) cleaned.push_back(anchor);
    }
    anchors.swap(cleaned);
}

std::vector<uint32_t> maintain_online_density_anchors(
    const Matrix& all,
    const IdMatrix& all_knn,
    const Matrix* workload_queries,
    const std::vector<uint8_t>& alive,
    const std::vector<uint32_t>& seed_anchors,
    const std::vector<double>* hubness,
    const std::array<double, 4>& score_weights,
    const OnlineMlpModel* mlp_model,
    const std::array<double, 3>& retire_weights,
    double retire_load_weight,
    double query_gain_weight,
    double query_load_weight,
    uint32_t workload_query_limit,
    uint32_t initial_rows,
    uint32_t maintain_batch,
    uint32_t maintain_changes,
    uint32_t maintain_sample,
    MaintenanceStats& stats) {
    std::vector<uint32_t> maintained = seed_anchors;
    const uint32_t target = static_cast<uint32_t>(seed_anchors.size());
    stats.initial_anchors = target;
    dedup_alive_anchors(maintained, alive, stats);

    while (maintained.size() < target) {
        const uint32_t candidate = select_online_candidate(
            all,
            all_knn,
            workload_queries,
            alive,
            maintained,
            hubness,
            score_weights,
            mlp_model,
            query_gain_weight,
            query_load_weight,
            workload_query_limit,
            initial_rows,
            0,
            initial_rows,
            maintain_sample,
            stats.maintenance_distance_computations);
        if (candidate == std::numeric_limits<uint32_t>::max()) break;
        maintained.push_back(candidate);
        ++stats.added;
    }

    for (uint32_t begin = initial_rows; begin < all.rows; begin += maintain_batch) {
        const uint32_t end = std::min<uint32_t>(all.rows, begin + maintain_batch);
        ++stats.rounds;
        uint32_t changes = 0;
        while (changes < maintain_changes && !maintained.empty()) {
            const uint32_t candidate = select_online_candidate(
                all,
                all_knn,
                workload_queries,
                alive,
                maintained,
                hubness,
                score_weights,
                mlp_model,
                query_gain_weight,
                query_load_weight,
                workload_query_limit,
                initial_rows,
                begin,
                end,
                maintain_sample,
                stats.maintenance_distance_computations);
            if (candidate == std::numeric_limits<uint32_t>::max()) break;
            maintained.push_back(candidate);
            ++stats.added;
            ++changes;
            if (maintained.size() > target) {
                const uint32_t removed = retire_redundant_anchor(
                    all,
                    workload_queries,
                    maintained,
                    candidate,
                    initial_rows,
                    hubness,
                    retire_weights,
                    retire_load_weight,
                    workload_query_limit,
                    stats.maintenance_distance_computations);
                if (removed != std::numeric_limits<uint32_t>::max()) ++stats.retired;
            }
        }
        dedup_alive_anchors(maintained, alive, stats);
        while (maintained.size() > target) {
            double tmp = 0.0;
            const uint32_t removed = retire_redundant_anchor(
                all,
                workload_queries,
                maintained,
                std::numeric_limits<uint32_t>::max(),
                initial_rows,
                hubness,
                retire_weights,
                retire_load_weight,
                workload_query_limit,
                tmp);
            stats.maintenance_distance_computations += tmp;
            if (removed == std::numeric_limits<uint32_t>::max()) break;
            ++stats.retired;
        }
    }
    stats.final_anchors = static_cast<uint32_t>(maintained.size());
    return maintained;
}

std::vector<uint32_t> initial_ids_from_entries(
    const Graph& graph,
    const std::vector<uint8_t>& alive,
    const std::vector<uint32_t>& entries,
    uint32_t search_l,
    QueryWorkspace& workspace) {
    std::vector<uint32_t> init;
    init.reserve(search_l);
    const uint32_t stamp = workspace.next_init_stamp(graph.adj.size());
    auto add = [&](uint32_t id) {
        if (id < graph.adj.size() && alive[id] && workspace.init_seen[id] != stamp && init.size() < search_l) {
            workspace.init_seen[id] = stamp;
            init.push_back(id);
        }
    };

    for (uint32_t entry : entries) add(entry);
    bool progressed = true;
    size_t offset = 0;
    while (init.size() < search_l && progressed) {
        progressed = false;
        for (uint32_t entry : entries) {
            if (entry >= graph.adj.size()) continue;
            if (offset < graph.adj[entry].size()) {
                add(graph.adj[entry][offset]);
                progressed = true;
            }
            if (init.size() >= search_l) break;
        }
        ++offset;
    }
    for (uint32_t id = 0; init.size() < search_l && id < graph.adj.size(); ++id) add(id);
    return init;
}

std::vector<uint32_t> search_graph(
    const Matrix& all,
    const Graph& graph,
    const std::vector<uint8_t>& alive,
    const float* query,
    const std::vector<uint32_t>& init_ids,
    uint32_t search_l,
    uint32_t topk,
    uint32_t& expanded,
    uint32_t& distance_computations,
    QueryWorkspace& workspace) {
    const uint32_t stamp = workspace.next_search_stamp(all.rows);
    if (workspace.retset.size() < static_cast<size_t>(search_l) + 1) {
        workspace.retset.resize(static_cast<size_t>(search_l) + 1);
    }
    std::vector<Candidate>& retset = workspace.retset;
    uint32_t l = 0;
    distance_computations = 0;
    for (uint32_t id : init_ids) {
        if (id >= all.rows || !alive[id] || workspace.search_seen[id] == stamp || l >= search_l) continue;
        workspace.search_seen[id] = stamp;
        retset[l++] = Candidate{id, l2(all.row(id), query, all.dim), true};
        ++distance_computations;
    }
    if (l == 0) die("empty initial candidate set");
    std::sort(retset.begin(), retset.begin() + l, by_distance);

    expanded = 0;
    int k = 0;
    while (k < static_cast<int>(l)) {
        int nk = static_cast<int>(l);
        if (retset[static_cast<size_t>(k)].flag) {
            retset[static_cast<size_t>(k)].flag = false;
            const uint32_t node = retset[static_cast<size_t>(k)].id;
            ++expanded;
            for (uint32_t id : graph.adj[node]) {
                if (id >= all.rows || !alive[id] || workspace.search_seen[id] == stamp) continue;
                workspace.search_seen[id] = stamp;
                const float dist = l2(all.row(id), query, all.dim);
                ++distance_computations;
                if (l >= search_l && dist >= retset[l - 1].distance) continue;
                const uint32_t r = insert_into_pool(retset, l, Candidate{id, dist, true});
                if (l < search_l) ++l;
                if (r < static_cast<uint32_t>(nk)) nk = static_cast<int>(r);
            }
        }
        if (nk <= k) k = nk;
        else ++k;
    }

    std::vector<uint32_t> result;
    result.reserve(topk);
    for (uint32_t i = 0; i < topk && i < l; ++i) result.push_back(retset[i].id);
    return result;
}

double recall_at_k(const std::vector<uint32_t>& found, const uint32_t* truth, uint32_t k) {
    std::unordered_set<uint32_t> exact;
    exact.reserve(k * 2);
    for (uint32_t i = 0; i < k; ++i) exact.insert(truth[i]);
    uint32_t hits = 0;
    for (uint32_t i = 0; i < k && i < found.size(); ++i) {
        if (exact.find(found[i]) != exact.end()) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(k);
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double pos = (q / 100.0) * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) return values[lo];
    const double frac = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

void write_json(
    const std::string& path,
    const Args& args,
    uint32_t queries,
    uint32_t anchors,
    uint32_t initial_rows,
    uint32_t insert_rows,
    uint32_t deletes,
    uint32_t touched_nodes,
    uint32_t repair_touched_nodes,
    uint32_t hard_repair_touched_nodes,
    const MaintenanceStats& maintenance,
    double recall,
    double avg_expanded,
    double avg_distance_computations,
    double avg_latency_us,
    double p95_latency_us,
    double p99_latency_us,
    double qps) {
    std::ofstream out(path);
    if (!out) die("cannot write json: " + path);
    out << "{\n";
    out << "  \"backend\": \"nsg_dynamic_no_rebuild_entry_runner\",\n";
    out << "  \"mode\": \"" << args.mode << "\",\n";
    out << "  \"queries\": " << queries << ",\n";
    out << "  \"topk\": " << args.topk << ",\n";
    out << "  \"search_l\": " << args.search_l << ",\n";
    out << "  \"entries\": " << args.entries << ",\n";
    out << "  \"anchors\": " << anchors << ",\n";
    out << "  \"initial_rows\": " << initial_rows << ",\n";
    out << "  \"insert_rows\": " << insert_rows << ",\n";
    out << "  \"deleted_rows\": " << deletes << ",\n";
    out << "  \"touched_nodes\": " << touched_nodes << ",\n";
    out << "  \"repair_degree\": " << args.repair_degree << ",\n";
    out << "  \"repair_touched_nodes\": " << repair_touched_nodes << ",\n";
    out << "  \"hard_repair_local_degree\": " << args.hard_repair_local_degree << ",\n";
    out << "  \"hard_repair_bridge_degree\": " << args.hard_repair_bridge_degree << ",\n";
    out << "  \"hard_repair_touched_nodes\": " << hard_repair_touched_nodes << ",\n";
    out << "  \"insert_degree\": " << args.insert_degree << ",\n";
    out << "  \"max_degree\": " << args.max_degree << ",\n";
    out << "  \"online_score_weights\": ["
        << args.online_score_weights[0] << ", "
        << args.online_score_weights[1] << ", "
        << args.online_score_weights[2] << ", "
        << args.online_score_weights[3] << "],\n";
    out << "  \"online_retire_weights\": ["
        << args.online_retire_weights[0] << ", "
        << args.online_retire_weights[1] << ", "
        << args.online_retire_weights[2] << "],\n";
    out << "  \"online_retire_load_weight\": " << args.online_retire_load_weight << ",\n";
    out << "  \"online_query_gain_weight\": " << args.online_query_gain_weight << ",\n";
    out << "  \"online_query_load_weight\": " << args.online_query_load_weight << ",\n";
    out << "  \"online_mlp_model\": \"" << args.online_mlp_model << "\",\n";
    out << "  \"maintenance\": {\n";
    out << "    \"initial_anchors\": " << maintenance.initial_anchors << ",\n";
    out << "    \"final_anchors\": " << maintenance.final_anchors << ",\n";
    out << "    \"removed_dead\": " << maintenance.removed_dead << ",\n";
    out << "    \"added\": " << maintenance.added << ",\n";
    out << "    \"retired\": " << maintenance.retired << ",\n";
    out << "    \"rounds\": " << maintenance.rounds << ",\n";
    out << "    \"distance_computations\": " << maintenance.maintenance_distance_computations << ",\n";
    out << "    \"seconds\": " << maintenance.maintenance_seconds << "\n";
    out << "  },\n";
    out << "  \"recall_metric\": \"recall@" << args.topk << "\",\n";
    out << "  \"recall_at_k\": " << recall << ",\n";
    out << "  \"recall_at_10\": " << recall << ",\n";
    out << "  \"avg_expanded\": " << avg_expanded << ",\n";
    out << "  \"avg_distance_computations\": " << avg_distance_computations << ",\n";
    out << "  \"avg_latency_us\": " << avg_latency_us << ",\n";
    out << "  \"p95_latency_us\": " << p95_latency_us << ",\n";
    out << "  \"p99_latency_us\": " << p99_latency_us << ",\n";
    out << "  \"qps\": " << qps << "\n";
    out << "}\n";
}

void write_per_query_csv(
    const std::string& path,
    const std::vector<double>& recalls,
    const std::vector<uint32_t>& expanded,
    const std::vector<uint32_t>& distance_computations,
    const std::vector<double>& latencies_us) {
    if (path.empty()) return;
    std::ofstream out(path);
    if (!out) die("cannot write per-query csv: " + path);
    out << "query_id,recall_at_10,expanded,distance_computations,latency_us\n";
    for (size_t i = 0; i < recalls.size(); ++i) {
        out << i << ","
            << recalls[i] << ","
            << expanded[i] << ","
            << distance_computations[i] << ","
            << latencies_us[i] << "\n";
    }
}

void write_anchor_costs_csv(
    const std::string& path,
    const Matrix& all,
    const Graph& graph,
    const std::vector<uint8_t>& alive,
    const Matrix& queries,
    const IdMatrix& truth,
    const std::vector<uint32_t>& anchors,
    uint32_t search_l,
    uint32_t topk,
    uint32_t query_limit) {
    if (path.empty()) return;
    std::ofstream out(path);
    if (!out) die("cannot write anchor costs csv: " + path);
    out << "query_id,anchor_rank,anchor_id,recall_at_10,expanded,distance_computations\n";
    const uint32_t rows = query_limit == 0 ? queries.rows : std::min(query_limit, queries.rows);
    QueryWorkspace workspace;
    for (uint32_t qi = 0; qi < rows; ++qi) {
        const float* query = queries.row(qi);
        for (uint32_t ai = 0; ai < anchors.size(); ++ai) {
            const uint32_t anchor = anchors[ai];
            if (anchor >= all.rows || !alive[anchor]) {
                out << qi << "," << ai << "," << anchor << ",0,0,4294967295\n";
                continue;
            }
            std::vector<uint32_t> init_ids = initial_ids_from_entries(graph, alive, {anchor}, search_l, workspace);
            uint32_t expanded = 0;
            uint32_t distance_computations = 0;
            std::vector<uint32_t> found = search_graph(
                all,
                graph,
                alive,
                query,
                init_ids,
                search_l,
                topk,
                expanded,
                distance_computations,
                workspace);
            const double recall = recall_at_k(found, truth.row(qi), topk);
            out << qi << ","
                << ai << ","
                << anchor << ","
                << recall << ","
                << expanded << ","
                << distance_computations << "\n";
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const OnlineMlpModel online_mlp = read_online_mlp_model(args.online_mlp_model);
        const Matrix initial = read_fbin(args.initial_fbin);
        const Matrix inserts = read_fbin(args.insert_fbin);
        const Matrix queries = read_fbin(args.query_fbin);
        const IdMatrix truth = read_id_bin(args.truth_bin);
        const std::vector<uint32_t> deletes = read_u32_list(args.delete_u32);
        const Graph initial_graph = read_nsg(args.initial_nsg, initial.rows);
        const IdMatrix all_knn = read_ivecs_graph(args.all_knn_graph);
        const IdMatrix query_entries = args.query_entries_ivecs.empty() ? IdMatrix{} : read_ivecs_graph(args.query_entries_ivecs);
        const IdMatrix hard_regions = args.hard_repair_ivecs.empty() ? IdMatrix{} : read_ivecs_graph(args.hard_repair_ivecs);
        std::vector<uint32_t> anchors = args.anchors_txt.empty() ? std::vector<uint32_t>{} : read_anchors(args.anchors_txt);

        if (queries.rows != truth.rows) die("query/truth row mismatch");
        const bool query_entries_mode = args.mode == "query_entries" || args.mode.rfind("query_entries_", 0) == 0;
        if (query_entries_mode && query_entries.rows != queries.rows) die("query_entries row count mismatch");
        if (args.topk > truth.cols) die("topk exceeds truth width");
        if (initial.dim != queries.dim || inserts.dim != queries.dim) die("dimension mismatch");

        const Matrix all = merge_matrices(initial, inserts);
        std::vector<uint8_t> alive(all.rows, 1);
        for (uint32_t id : deletes) {
            if (id < alive.size()) alive[id] = 0;
        }
        uint32_t touched_nodes = 0;
        Graph graph = build_dynamic_graph(
            initial_graph,
            all,
            all_knn,
            initial.rows,
            alive,
            args.insert_degree,
            args.max_degree,
            touched_nodes);
        const uint32_t repair_touched_nodes = repair_graph_from_knn(
            graph,
            all,
            all_knn,
            alive,
            initial.rows,
            args.repair_degree,
            args.max_degree);
        const uint32_t hard_repair_touched_nodes = repair_hard_regions(
            graph,
            all,
            hard_regions,
            alive,
            args.hard_repair_local_degree,
            args.hard_repair_bridge_degree,
            args.max_degree,
            initial_graph.ep);

        MaintenanceStats maintenance;
        if (args.mode == "online_density" || args.mode == "online_hub_density") {
            if (anchors.empty()) die("online density modes require seed anchors in --anchors-txt");
            const std::vector<double> hubness = args.mode == "online_hub_density"
                ? graph_hubness_scores(graph, alive)
                : std::vector<double>{};
            const auto maintenance_start = std::chrono::steady_clock::now();
            anchors = maintain_online_density_anchors(
                all,
                all_knn,
                args.anchor_cost_query_limit > 0 ? &queries : nullptr,
                alive,
                anchors,
                args.mode == "online_hub_density" ? &hubness : nullptr,
                args.online_score_weights,
                online_mlp.enabled() ? &online_mlp : nullptr,
                args.online_retire_weights,
                args.online_retire_load_weight,
                args.online_query_gain_weight,
                args.online_query_load_weight,
                args.anchor_cost_query_limit,
                initial.rows,
                args.maintain_batch,
                args.maintain_changes,
                args.maintain_sample,
                maintenance);
            const auto maintenance_stop = std::chrono::steady_clock::now();
            maintenance.maintenance_seconds = std::chrono::duration<double>(maintenance_stop - maintenance_start).count();
        } else {
            maintenance.initial_anchors = static_cast<uint32_t>(anchors.size());
            maintenance.final_anchors = static_cast<uint32_t>(anchors.size());
        }
        write_anchors(args.out_anchors_txt, anchors);
        write_anchor_costs_csv(
            args.out_anchor_costs_csv,
            all,
            graph,
            alive,
            queries,
            truth,
            anchors,
            args.search_l,
            args.topk,
            args.anchor_cost_query_limit);

        std::vector<double> latencies_us;
        latencies_us.reserve(queries.rows);
        std::vector<double> per_query_recalls;
        std::vector<uint32_t> per_query_expanded;
        std::vector<uint32_t> per_query_distance_computations;
        if (!args.out_per_query.empty()) {
            per_query_recalls.reserve(queries.rows);
            per_query_expanded.reserve(queries.rows);
            per_query_distance_computations.reserve(queries.rows);
        }
        double recall_sum = 0.0;
        double expanded_sum = 0.0;
        double distance_sum = 0.0;
        QueryWorkspace workspace;

        const auto all_start = std::chrono::steady_clock::now();
        for (uint32_t qi = 0; qi < queries.rows; ++qi) {
            const auto q_start = std::chrono::steady_clock::now();
            const float* query = queries.row(qi);
            std::vector<uint32_t> entries;
            if (args.mode == "nsg_ep") {
                entries = {initial_graph.ep};
            } else if (query_entries_mode) {
                const uint32_t* row = query_entries.row(qi);
                for (uint32_t j = 0; j < query_entries.cols && entries.size() < args.entries; ++j) {
                    const uint32_t entry = row[j];
                    if (entry < all.rows && alive[entry]) entries.push_back(entry);
                }
            } else {
                entries = closest_anchors(all, alive, anchors, query, args.entries);
            }
            if (entries.empty()) entries = {initial_graph.ep};
            std::vector<uint32_t> init_ids = initial_ids_from_entries(graph, alive, entries, args.search_l, workspace);

            uint32_t expanded = 0;
            uint32_t distance_computations = 0;
            std::vector<uint32_t> found = search_graph(
                all,
                graph,
                alive,
                query,
                init_ids,
                args.search_l,
                args.topk,
                expanded,
                distance_computations,
                workspace);
            const auto q_stop = std::chrono::steady_clock::now();
            latencies_us.push_back(std::chrono::duration<double, std::micro>(q_stop - q_start).count());
            const double query_recall = recall_at_k(found, truth.row(qi), args.topk);
            recall_sum += query_recall;
            expanded_sum += static_cast<double>(expanded);
            distance_sum += static_cast<double>(distance_computations);
            if (!args.out_per_query.empty()) {
                per_query_recalls.push_back(query_recall);
                per_query_expanded.push_back(expanded);
                per_query_distance_computations.push_back(distance_computations);
            }
        }
        const auto all_stop = std::chrono::steady_clock::now();
        const double search_seconds = std::chrono::duration<double>(all_stop - all_start).count();
        const double q = static_cast<double>(queries.rows);
        double avg_latency = 0.0;
        for (double value : latencies_us) avg_latency += value;
        avg_latency /= q;

        write_json(
            args.out_json,
            args,
            queries.rows,
            static_cast<uint32_t>(anchors.size()),
            initial.rows,
            inserts.rows,
            static_cast<uint32_t>(deletes.size()),
            touched_nodes,
            repair_touched_nodes,
            hard_repair_touched_nodes,
            maintenance,
            recall_sum / q,
            expanded_sum / q,
            distance_sum / q,
            avg_latency,
            percentile(latencies_us, 95.0),
            percentile(latencies_us, 99.0),
            q / search_seconds);
        write_per_query_csv(
            args.out_per_query,
            per_query_recalls,
            per_query_expanded,
            per_query_distance_computations,
            latencies_us);

        std::cout << "mode=" << args.mode
                  << " L=" << args.search_l
                  << " recall=" << (recall_sum / q)
                  << " avg_dist=" << (distance_sum / q)
                  << " p99_us=" << percentile(latencies_us, 99.0)
                  << " qps=" << (q / search_seconds)
                  << " touched=" << touched_nodes
                  << " repair_touched=" << repair_touched_nodes
                  << " anchors=" << anchors.size()
                  << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "nsg_dynamic_entry_runner error: " << error.what() << std::endl;
        return 2;
    }
}
