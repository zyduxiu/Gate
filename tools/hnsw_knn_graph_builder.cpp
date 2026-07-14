#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "hnswlib/hnswlib.h"

namespace {

struct Matrix {
    uint32_t rows = 0;
    uint32_t dim = 0;
    std::vector<float> values;

    const float* row(uint32_t i) const {
        return values.data() + static_cast<size_t>(i) * dim;
    }
};

struct Args {
    std::string base_fbin;
    std::string out_graph;
    uint32_t k = 100;
    uint32_t m = 32;
    uint32_t ef_construction = 200;
    uint32_t ef_search = 240;
    uint32_t seed = 19;
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
        if (key == "--base-fbin") args.base_fbin = require_value(i, argc, argv);
        else if (key == "--out-graph") args.out_graph = require_value(i, argc, argv);
        else if (key == "--k") args.k = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--M") args.m = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--ef-construction") args.ef_construction = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--ef-search") args.ef_search = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--seed") args.seed = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else die("unknown argument: " + key);
    }
    if (args.base_fbin.empty()) die("--base-fbin is required");
    if (args.out_graph.empty()) die("--out-graph is required");
    if (args.k == 0) die("--k must be positive");
    return args;
}

Matrix read_fbin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) die("cannot open fbin: " + path);
    Matrix m;
    in.read(reinterpret_cast<char*>(&m.rows), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&m.dim), sizeof(uint32_t));
    if (!in || m.rows == 0 || m.dim == 0) die("invalid fbin header: " + path);
    m.values.resize(static_cast<size_t>(m.rows) * m.dim);
    in.read(reinterpret_cast<char*>(m.values.data()), static_cast<std::streamsize>(m.values.size() * sizeof(float)));
    if (!in) die("truncated fbin: " + path);
    return m;
}

std::vector<uint32_t> labels_from_queue(
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result,
    uint32_t self,
    uint32_t k) {
    std::vector<std::pair<float, uint32_t>> pairs;
    while (!result.empty()) {
        auto item = result.top();
        result.pop();
        const uint32_t label = static_cast<uint32_t>(item.second);
        if (label != self) {
            pairs.emplace_back(item.first, label);
        }
    }
    std::sort(pairs.begin(), pairs.end());
    std::vector<uint32_t> labels;
    labels.reserve(k);
    std::unordered_set<uint32_t> seen;
    for (const auto& item : pairs) {
        if (labels.size() >= k) break;
        if (seen.insert(item.second).second) {
            labels.push_back(item.second);
        }
    }
    return labels;
}

void write_ivecs(const std::string& path, const std::vector<std::vector<uint32_t>>& graph, uint32_t k) {
    std::ofstream out(path, std::ios::binary);
    if (!out) die("cannot write graph: " + path);
    const int32_t dim = static_cast<int32_t>(k);
    for (const auto& row : graph) {
        out.write(reinterpret_cast<const char*>(&dim), sizeof(int32_t));
        for (uint32_t j = 0; j < k; ++j) {
            const int32_t value = static_cast<int32_t>(row[j]);
            out.write(reinterpret_cast<const char*>(&value), sizeof(int32_t));
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const Matrix base = read_fbin(args.base_fbin);
        if (args.k >= base.rows) die("--k must be smaller than row count");

        hnswlib::L2Space space(base.dim);
        hnswlib::HierarchicalNSW<float> index(&space, base.rows, args.m, args.ef_construction, args.seed, false);

        const auto build_start = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < base.rows; ++i) {
            index.addPoint(base.row(i), static_cast<hnswlib::labeltype>(i));
        }
        index.setEf(std::max(args.ef_search, args.k + 1));
        const auto build_stop = std::chrono::steady_clock::now();

        std::vector<std::vector<uint32_t>> graph(base.rows);
        const uint32_t search_k = std::min<uint32_t>(base.rows, args.k + 1);

#pragma omp parallel for schedule(dynamic, 64)
        for (int64_t i = 0; i < static_cast<int64_t>(base.rows); ++i) {
            auto result = index.searchKnn(base.row(static_cast<uint32_t>(i)), search_k);
            auto labels = labels_from_queue(result, static_cast<uint32_t>(i), args.k);
            if (labels.size() < args.k) {
                for (uint32_t fallback = 0; labels.size() < args.k && fallback < base.rows; ++fallback) {
                    if (fallback != static_cast<uint32_t>(i) &&
                        std::find(labels.begin(), labels.end(), fallback) == labels.end()) {
                        labels.push_back(fallback);
                    }
                }
            }
            graph[static_cast<size_t>(i)] = std::move(labels);
        }

        write_ivecs(args.out_graph, graph, args.k);
        const auto stop = std::chrono::steady_clock::now();
        const double build_seconds = std::chrono::duration<double>(build_stop - build_start).count();
        const double total_seconds = std::chrono::duration<double>(stop - build_start).count();
        std::cout << "wrote " << args.out_graph
                  << " rows=" << base.rows
                  << " k=" << args.k
                  << " build_seconds=" << build_seconds
                  << " total_seconds=" << total_seconds
                  << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "hnsw_knn_graph_builder error: " << error.what() << std::endl;
        return 2;
    }
}
