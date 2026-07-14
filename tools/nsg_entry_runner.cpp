#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
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

struct NsgGraph {
    uint32_t width = 0;
    uint32_t ep = 0;
    std::vector<std::vector<uint32_t>> adj;
};

struct Args {
    std::string base_fvecs;
    std::string query_fvecs;
    std::string truth_ivecs;
    std::string nsg_path;
    std::string anchors_txt;
    std::string mode = "nsg_ep";
    std::string out_json;
    uint32_t search_l = 120;
    uint32_t topk = 10;
    uint32_t entries = 8;
};

struct Candidate {
    uint32_t id = 0;
    float distance = 0.0f;
    bool flag = true;
};

[[noreturn]] void die(const std::string& message) {
    throw std::runtime_error(message);
}

std::string require_value(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        die(std::string("missing value for ") + argv[i]);
    }
    ++i;
    return argv[i];
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--base-fvecs") args.base_fvecs = require_value(i, argc, argv);
        else if (key == "--query-fvecs") args.query_fvecs = require_value(i, argc, argv);
        else if (key == "--truth-ivecs") args.truth_ivecs = require_value(i, argc, argv);
        else if (key == "--nsg") args.nsg_path = require_value(i, argc, argv);
        else if (key == "--anchors-txt") args.anchors_txt = require_value(i, argc, argv);
        else if (key == "--mode") args.mode = require_value(i, argc, argv);
        else if (key == "--out-json") args.out_json = require_value(i, argc, argv);
        else if (key == "--search-l") args.search_l = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--topk") args.topk = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--entries") args.entries = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else die("unknown argument: " + key);
    }
    if (args.base_fvecs.empty()) die("--base-fvecs is required");
    if (args.query_fvecs.empty()) die("--query-fvecs is required");
    if (args.truth_ivecs.empty()) die("--truth-ivecs is required");
    if (args.nsg_path.empty()) die("--nsg is required");
    if (args.mode != "nsg_ep" && args.anchors_txt.empty()) die("--anchors-txt is required for anchor modes");
    if (args.out_json.empty()) die("--out-json is required");
    if (args.search_l < args.topk) die("--search-l must be >= --topk");
    return args;
}

Matrix read_fvecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) die("cannot open fvecs: " + path);
    int32_t dim_i = 0;
    in.read(reinterpret_cast<char*>(&dim_i), sizeof(int32_t));
    if (!in || dim_i <= 0) die("invalid fvecs header: " + path);
    const uint32_t dim = static_cast<uint32_t>(dim_i);
    in.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(in.tellg());
    const size_t row_bytes = sizeof(int32_t) + static_cast<size_t>(dim) * sizeof(float);
    if (file_size % row_bytes != 0) die("invalid fvecs size: " + path);
    const uint32_t rows = static_cast<uint32_t>(file_size / row_bytes);
    Matrix matrix;
    matrix.rows = rows;
    matrix.dim = dim;
    matrix.values.resize(static_cast<size_t>(rows) * dim);
    in.seekg(0, std::ios::beg);
    for (uint32_t i = 0; i < rows; ++i) {
        int32_t row_dim = 0;
        in.read(reinterpret_cast<char*>(&row_dim), sizeof(int32_t));
        if (row_dim != dim_i) die("inconsistent fvecs dim: " + path);
        in.read(reinterpret_cast<char*>(matrix.values.data() + static_cast<size_t>(i) * dim), dim * sizeof(float));
        if (!in) die("truncated fvecs: " + path);
    }
    return matrix;
}

IdMatrix read_ivecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) die("cannot open ivecs: " + path);
    int32_t dim_i = 0;
    in.read(reinterpret_cast<char*>(&dim_i), sizeof(int32_t));
    if (!in || dim_i <= 0) die("invalid ivecs header: " + path);
    const uint32_t dim = static_cast<uint32_t>(dim_i);
    in.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(in.tellg());
    const size_t row_bytes = sizeof(int32_t) + static_cast<size_t>(dim) * sizeof(uint32_t);
    if (file_size % row_bytes != 0) die("invalid ivecs size: " + path);
    const uint32_t rows = static_cast<uint32_t>(file_size / row_bytes);
    IdMatrix matrix;
    matrix.rows = rows;
    matrix.cols = dim;
    matrix.values.resize(static_cast<size_t>(rows) * dim);
    in.seekg(0, std::ios::beg);
    for (uint32_t i = 0; i < rows; ++i) {
        int32_t row_dim = 0;
        in.read(reinterpret_cast<char*>(&row_dim), sizeof(int32_t));
        if (row_dim != dim_i) die("inconsistent ivecs dim: " + path);
        in.read(reinterpret_cast<char*>(matrix.values.data() + static_cast<size_t>(i) * dim), dim * sizeof(uint32_t));
        if (!in) die("truncated ivecs: " + path);
    }
    return matrix;
}

NsgGraph read_nsg(const std::string& path, uint32_t expected_rows) {
    std::ifstream in(path, std::ios::binary);
    if (!in) die("cannot open NSG index: " + path);
    NsgGraph graph;
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

std::vector<uint32_t> read_anchors(const std::string& path) {
    std::ifstream in(path);
    if (!in) die("cannot open anchors: " + path);
    std::vector<uint32_t> anchors;
    uint32_t value = 0;
    while (in >> value) anchors.push_back(value);
    return anchors;
}

float l2(const float* a, const float* b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
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
    for (uint32_t i = pool_size; i > left; --i) {
        pool[i] = pool[i - 1];
    }
    pool[left] = candidate;
    return left;
}

std::vector<uint32_t> closest_anchors(
    const Matrix& base,
    const std::vector<uint32_t>& anchors,
    const float* query,
    uint32_t entries) {
    std::vector<std::pair<float, uint32_t>> scored;
    scored.reserve(anchors.size());
    for (uint32_t anchor : anchors) {
        if (anchor < base.rows) {
            scored.emplace_back(l2(base.row(anchor), query, base.dim), anchor);
        }
    }
    if (scored.empty()) return {};
    const uint32_t take = std::min<uint32_t>(entries, static_cast<uint32_t>(scored.size()));
    std::partial_sort(scored.begin(), scored.begin() + take, scored.end());
    std::vector<uint32_t> result;
    result.reserve(take);
    for (uint32_t i = 0; i < take; ++i) {
        result.push_back(scored[i].second);
    }
    return result;
}

std::vector<uint32_t> initial_ids_from_entries(
    const NsgGraph& graph,
    const std::vector<uint32_t>& entries,
    uint32_t search_l,
    uint32_t nd) {
    std::vector<uint32_t> init;
    init.reserve(search_l);
    std::vector<uint8_t> seen(nd, 0);

    auto add = [&](uint32_t id) {
        if (id < nd && !seen[id] && init.size() < search_l) {
            seen[id] = 1;
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
    for (uint32_t id = 0; init.size() < search_l && id < nd; ++id) add(id);
    return init;
}

std::vector<uint32_t> search_graph(
    const Matrix& base,
    const NsgGraph& graph,
    const float* query,
    const std::vector<uint32_t>& init_ids,
    uint32_t search_l,
    uint32_t topk,
    uint32_t& expanded,
    uint32_t& distance_computations) {
    std::vector<uint8_t> flags(base.rows, 0);
    std::vector<Candidate> retset(search_l + 1);
    uint32_t l = 0;
    distance_computations = 0;
    for (uint32_t id : init_ids) {
        if (id >= base.rows || flags[id] || l >= search_l) continue;
        flags[id] = 1;
        retset[l++] = Candidate{id, l2(base.row(id), query, base.dim), true};
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
                if (id >= base.rows || flags[id]) continue;
                flags[id] = 1;
                const float dist = l2(base.row(id), query, base.dim);
                ++distance_computations;
                if (l >= search_l && dist >= retset[l - 1].distance) continue;
                Candidate candidate{id, dist, true};
                const uint32_t r = insert_into_pool(retset, l, candidate);
                if (l < search_l) ++l;
                if (r < static_cast<uint32_t>(nk)) nk = static_cast<int>(r);
            }
        }
        if (nk <= k) k = nk;
        else ++k;
    }
    std::vector<uint32_t> result;
    result.reserve(topk);
    for (uint32_t i = 0; i < topk && i < l; ++i) {
        result.push_back(retset[i].id);
    }
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
    const NsgGraph& graph,
    uint32_t queries,
    uint32_t anchors,
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
    out << "  \"backend\": \"nsg_same_graph_entry_runner\",\n";
    out << "  \"mode\": \"" << args.mode << "\",\n";
    out << "  \"queries\": " << queries << ",\n";
    out << "  \"topk\": " << args.topk << ",\n";
    out << "  \"search_l\": " << args.search_l << ",\n";
    out << "  \"entries\": " << args.entries << ",\n";
    out << "  \"anchors\": " << anchors << ",\n";
    out << "  \"nsg_width\": " << graph.width << ",\n";
    out << "  \"nsg_ep\": " << graph.ep << ",\n";
    out << "  \"recall_at_10\": " << recall << ",\n";
    out << "  \"avg_expanded\": " << avg_expanded << ",\n";
    out << "  \"avg_distance_computations\": " << avg_distance_computations << ",\n";
    out << "  \"avg_latency_us\": " << avg_latency_us << ",\n";
    out << "  \"p95_latency_us\": " << p95_latency_us << ",\n";
    out << "  \"p99_latency_us\": " << p99_latency_us << ",\n";
    out << "  \"qps\": " << qps << "\n";
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const Matrix base = read_fvecs(args.base_fvecs);
        const Matrix queries = read_fvecs(args.query_fvecs);
        const IdMatrix truth = read_ivecs(args.truth_ivecs);
        if (base.dim != queries.dim) die("base/query dimension mismatch");
        if (queries.rows != truth.rows) die("query/truth row mismatch");
        if (args.topk > truth.cols) die("topk exceeds truth width");
        const NsgGraph graph = read_nsg(args.nsg_path, base.rows);
        const std::vector<uint32_t> anchors = args.anchors_txt.empty() ? std::vector<uint32_t>{} : read_anchors(args.anchors_txt);

        std::vector<double> latencies_us;
        latencies_us.reserve(queries.rows);
        double recall_sum = 0.0;
        double expanded_sum = 0.0;
        double distance_sum = 0.0;

        const auto all_start = std::chrono::steady_clock::now();
        for (uint32_t qi = 0; qi < queries.rows; ++qi) {
            const float* query = queries.row(qi);
            std::vector<uint32_t> entries;
            if (args.mode == "nsg_ep") {
                entries = {graph.ep};
            } else {
                entries = closest_anchors(base, anchors, query, args.entries);
            }
            if (entries.empty()) entries = {graph.ep};
            std::vector<uint32_t> init_ids = initial_ids_from_entries(graph, entries, args.search_l, base.rows);

            uint32_t expanded = 0;
            uint32_t distance_computations = 0;
            const auto q_start = std::chrono::steady_clock::now();
            std::vector<uint32_t> found = search_graph(
                base,
                graph,
                query,
                init_ids,
                args.search_l,
                args.topk,
                expanded,
                distance_computations);
            const auto q_stop = std::chrono::steady_clock::now();
            latencies_us.push_back(std::chrono::duration<double, std::micro>(q_stop - q_start).count());
            recall_sum += recall_at_k(found, truth.row(qi), args.topk);
            expanded_sum += static_cast<double>(expanded);
            distance_sum += static_cast<double>(distance_computations);
        }
        const auto all_stop = std::chrono::steady_clock::now();
        const double search_seconds = std::chrono::duration<double>(all_stop - all_start).count();
        const double q = static_cast<double>(queries.rows);
        double avg_latency = 0.0;
        for (double value : latencies_us) avg_latency += value;
        avg_latency /= q;
        const double recall = recall_sum / q;
        const double avg_expanded = expanded_sum / q;
        const double avg_distance = distance_sum / q;
        const double p95 = percentile(latencies_us, 95.0);
        const double p99 = percentile(latencies_us, 99.0);
        const double qps = q / search_seconds;

        write_json(
            args.out_json,
            args,
            graph,
            queries.rows,
            static_cast<uint32_t>(anchors.size()),
            recall,
            avg_expanded,
            avg_distance,
            avg_latency,
            p95,
            p99,
            qps);
        std::cout << "mode=" << args.mode
                  << " L=" << args.search_l
                  << " recall=" << recall
                  << " avg_dist=" << avg_distance
                  << " p99_us=" << p99
                  << " qps=" << qps
                  << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "nsg_entry_runner error: " << error.what() << std::endl;
        return 2;
    }
}
