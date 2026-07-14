#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

struct IdMatrix {
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<uint32_t> values;

    const uint32_t* row(uint32_t i) const {
        return values.data() + static_cast<size_t>(i) * cols;
    }
};

struct Args {
    std::string initial_fbin;
    std::string insert_fbin;
    std::string query_fbin;
    std::string truth_bin;
    std::string delete_u32;
    std::string anchors_txt;
    std::string mode = "anchors";
    std::string out_json;
    uint32_t topk = 10;
    uint32_t ef = 80;
    uint32_t entries = 8;
    uint32_t m = 16;
    uint32_t ef_construction = 200;
    uint32_t seed = 100;
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
        if (key == "--initial-fbin") args.initial_fbin = require_value(i, argc, argv);
        else if (key == "--insert-fbin") args.insert_fbin = require_value(i, argc, argv);
        else if (key == "--query-fbin") args.query_fbin = require_value(i, argc, argv);
        else if (key == "--truth-bin") args.truth_bin = require_value(i, argc, argv);
        else if (key == "--delete-u32") args.delete_u32 = require_value(i, argc, argv);
        else if (key == "--anchors-txt") args.anchors_txt = require_value(i, argc, argv);
        else if (key == "--mode") args.mode = require_value(i, argc, argv);
        else if (key == "--out-json") args.out_json = require_value(i, argc, argv);
        else if (key == "--topk") args.topk = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--ef") args.ef = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--entries") args.entries = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--M") args.m = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--ef-construction") args.ef_construction = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (key == "--seed") args.seed = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        else die("unknown argument: " + key);
    }
    if (args.initial_fbin.empty()) die("--initial-fbin is required");
    if (args.query_fbin.empty()) die("--query-fbin is required");
    if (args.truth_bin.empty()) die("--truth-bin is required");
    if (args.delete_u32.empty()) die("--delete-u32 is required");
    if (args.mode == "anchors" && args.anchors_txt.empty()) die("--anchors-txt is required for anchor mode");
    if (args.out_json.empty()) die("--out-json is required");
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
    std::vector<uint32_t> ids(count);
    if (count > 0) {
        in.read(reinterpret_cast<char*>(ids.data()), static_cast<std::streamsize>(ids.size() * sizeof(uint32_t)));
    }
    if (!in && count > 0) die("truncated u32 list: " + path);
    return ids;
}

std::vector<uint32_t> read_anchor_txt(const std::string& path) {
    std::ifstream in(path);
    if (!in) die("cannot open anchors: " + path);
    std::vector<uint32_t> anchors;
    uint32_t value = 0;
    while (in >> value) {
        anchors.push_back(value);
    }
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

std::vector<uint32_t> select_entries(
    const Matrix& all,
    const std::vector<uint8_t>& alive,
    const std::vector<uint32_t>& anchors,
    const float* query,
    uint32_t entries) {
    std::vector<std::pair<float, uint32_t>> scored;
    scored.reserve(anchors.size());
    for (uint32_t anchor : anchors) {
        if (anchor >= all.rows || !alive[anchor]) continue;
        scored.emplace_back(l2(query, all.row(anchor), all.dim), anchor);
    }
    if (scored.empty()) return {};
    const uint32_t take = std::min<uint32_t>(entries, static_cast<uint32_t>(scored.size()));
    std::partial_sort(scored.begin(), scored.begin() + take, scored.end());
    std::vector<uint32_t> selected;
    selected.reserve(take);
    for (uint32_t i = 0; i < take; ++i) {
        selected.push_back(scored[i].second);
    }
    return selected;
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

double recall_at_k(const std::vector<uint32_t>& found, const uint32_t* truth, uint32_t k) {
    std::unordered_set<uint32_t> truth_set;
    truth_set.reserve(k * 2);
    for (uint32_t i = 0; i < k; ++i) {
        truth_set.insert(truth[i]);
    }
    uint32_t hits = 0;
    for (uint32_t i = 0; i < std::min<uint32_t>(k, static_cast<uint32_t>(found.size())); ++i) {
        if (truth_set.find(found[i]) != truth_set.end()) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(k);
}

template <typename Queue>
std::vector<uint32_t> labels_from_queue(Queue result, hnswlib::HierarchicalNSW<float>& index, uint32_t topk) {
    std::vector<std::pair<float, uint32_t>> pairs;
    while (!result.empty()) {
        auto item = result.top();
        result.pop();
        pairs.emplace_back(item.first, static_cast<uint32_t>(item.second));
    }
    std::sort(pairs.begin(), pairs.end());
    std::vector<uint32_t> labels;
    labels.reserve(std::min<uint32_t>(topk, static_cast<uint32_t>(pairs.size())));
    for (uint32_t i = 0; i < pairs.size() && labels.size() < topk; ++i) {
        labels.push_back(pairs[i].second);
    }
    return labels;
}

std::vector<uint32_t> anchor_search(
    hnswlib::HierarchicalNSW<float>& index,
    const float* query,
    const std::vector<uint32_t>& entries,
    uint32_t ef_total,
    uint32_t topk,
    long& cmps,
    long& hops) {
    std::unordered_map<uint32_t, float> best;
    const uint32_t active_entries = std::max<uint32_t>(1, static_cast<uint32_t>(entries.size()));
    const uint32_t ef_per_entry = std::max<uint32_t>(topk, (ef_total + active_entries - 1) / active_entries);

    index.metric_distance_computations = 0;
    index.metric_hops = 0;
    for (uint32_t ep : entries) {
        auto candidates = index.searchBaseLayerST<false, true>(
            static_cast<hnswlib::tableint>(ep),
            query,
            ef_per_entry,
            nullptr);
        while (!candidates.empty()) {
            auto item = candidates.top();
            candidates.pop();
            const uint32_t label = static_cast<uint32_t>(index.getExternalLabel(item.second));
            auto found = best.find(label);
            if (found == best.end() || item.first < found->second) {
                best[label] = item.first;
            }
        }
    }
    cmps = index.metric_distance_computations.load();
    hops = index.metric_hops.load();

    std::vector<std::pair<float, uint32_t>> pairs;
    pairs.reserve(best.size());
    for (const auto& item : best) {
        pairs.emplace_back(item.second, item.first);
    }
    std::sort(pairs.begin(), pairs.end());

    std::vector<uint32_t> labels;
    labels.reserve(std::min<uint32_t>(topk, static_cast<uint32_t>(pairs.size())));
    for (uint32_t i = 0; i < pairs.size() && labels.size() < topk; ++i) {
        labels.push_back(pairs[i].second);
    }
    return labels;
}

void write_json(
    const std::string& path,
    const Args& args,
    uint32_t queries,
    uint32_t anchors,
    double recall,
    double avg_cmps,
    double avg_hops,
    double avg_latency_us,
    double p95_latency_us,
    double p99_latency_us,
    double qps,
    double build_seconds) {
    std::ofstream out(path);
    if (!out) die("cannot write json: " + path);
    out << "{\n";
    out << "  \"backend\": \"hnswlib_cpp\",\n";
    out << "  \"mode\": \"" << args.mode << "\",\n";
    out << "  \"queries\": " << queries << ",\n";
    out << "  \"topk\": " << args.topk << ",\n";
    out << "  \"ef\": " << args.ef << ",\n";
    out << "  \"entries\": " << args.entries << ",\n";
    out << "  \"anchors\": " << anchors << ",\n";
    out << "  \"M\": " << args.m << ",\n";
    out << "  \"ef_construction\": " << args.ef_construction << ",\n";
    out << "  \"recall_at_10\": " << recall << ",\n";
    out << "  \"avg_cmps\": " << avg_cmps << ",\n";
    out << "  \"avg_hops\": " << avg_hops << ",\n";
    out << "  \"avg_latency_us\": " << avg_latency_us << ",\n";
    out << "  \"p95_latency_us\": " << p95_latency_us << ",\n";
    out << "  \"p99_latency_us\": " << p99_latency_us << ",\n";
    out << "  \"qps\": " << qps << ",\n";
    out << "  \"build_seconds\": " << build_seconds << "\n";
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const Matrix initial = read_fbin(args.initial_fbin);
        const Matrix inserts = args.insert_fbin.empty() ? Matrix{} : read_fbin(args.insert_fbin);
        const Matrix queries = read_fbin(args.query_fbin);
        const IdMatrix truth = read_id_bin(args.truth_bin);
        const std::vector<uint32_t> deletes = read_u32_list(args.delete_u32);
        const std::vector<uint32_t> anchors = args.anchors_txt.empty() ? std::vector<uint32_t>{} : read_anchor_txt(args.anchors_txt);

        if (queries.rows != truth.rows) die("query/truth row mismatch");
        if (args.topk > truth.cols) die("topk exceeds truth width");
        if (initial.dim != queries.dim) die("initial/query dim mismatch");
        if (inserts.rows > 0 && inserts.dim != initial.dim) die("insert dim mismatch");

        Matrix all;
        all.rows = initial.rows + inserts.rows;
        all.dim = initial.dim;
        all.values.reserve(static_cast<size_t>(all.rows) * all.dim);
        all.values.insert(all.values.end(), initial.values.begin(), initial.values.end());
        all.values.insert(all.values.end(), inserts.values.begin(), inserts.values.end());

        std::vector<uint8_t> alive(all.rows, 1);
        for (uint32_t id : deletes) {
            if (id < alive.size()) alive[id] = 0;
        }

        hnswlib::L2Space space(initial.dim);
        hnswlib::HierarchicalNSW<float> index(&space, all.rows, args.m, args.ef_construction, args.seed, false);

        const auto build_start = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < initial.rows; ++i) {
            index.addPoint(initial.row(i), static_cast<hnswlib::labeltype>(i));
        }
        for (uint32_t i = 0; i < inserts.rows; ++i) {
            const uint32_t label = initial.rows + i;
            index.addPoint(inserts.row(i), static_cast<hnswlib::labeltype>(label));
        }
        for (uint32_t id : deletes) {
            if (id < all.rows) {
                index.markDelete(static_cast<hnswlib::labeltype>(id));
            }
        }
        index.setEf(args.ef);
        const auto build_stop = std::chrono::steady_clock::now();
        const double build_seconds = std::chrono::duration<double>(build_stop - build_start).count();

        std::vector<double> latencies_us;
        latencies_us.reserve(queries.rows);
        double recall_sum = 0.0;
        double cmps_sum = 0.0;
        double hops_sum = 0.0;

        const auto search_start_all = std::chrono::steady_clock::now();
        for (uint32_t qi = 0; qi < queries.rows; ++qi) {
            const float* query = queries.row(qi);
            std::vector<uint32_t> result_labels;
            long cmps = 0;
            long hops = 0;

            const auto q_start = std::chrono::steady_clock::now();
            if (args.mode == "builtin") {
                index.metric_distance_computations = 0;
                index.metric_hops = 0;
                auto result = index.searchKnn(query, args.topk);
                result_labels = labels_from_queue(result, index, args.topk);
                cmps = index.metric_distance_computations.load();
                hops = index.metric_hops.load();
            } else if (args.mode == "anchors") {
                auto selected = select_entries(all, alive, anchors, query, args.entries);
                if (selected.empty() && index.enterpoint_node_ < all.rows && alive[index.enterpoint_node_]) {
                    selected.push_back(index.enterpoint_node_);
                }
                result_labels = anchor_search(index, query, selected, args.ef, args.topk, cmps, hops);
            } else {
                die("mode must be builtin or anchors");
            }
            const auto q_stop = std::chrono::steady_clock::now();

            latencies_us.push_back(std::chrono::duration<double, std::micro>(q_stop - q_start).count());
            recall_sum += recall_at_k(result_labels, truth.row(qi), args.topk);
            cmps_sum += static_cast<double>(cmps);
            hops_sum += static_cast<double>(hops);
        }
        const auto search_stop_all = std::chrono::steady_clock::now();
        const double search_seconds = std::chrono::duration<double>(search_stop_all - search_start_all).count();

        const double q = static_cast<double>(queries.rows);
        const double recall = recall_sum / q;
        const double avg_cmps = cmps_sum / q;
        const double avg_hops = hops_sum / q;
        double avg_latency = 0.0;
        for (double value : latencies_us) avg_latency += value;
        avg_latency /= q;
        const double p95 = percentile(latencies_us, 95.0);
        const double p99 = percentile(latencies_us, 99.0);
        const double qps = q / search_seconds;

        write_json(
            args.out_json,
            args,
            queries.rows,
            static_cast<uint32_t>(anchors.size()),
            recall,
            avg_cmps,
            avg_hops,
            avg_latency,
            p95,
            p99,
            qps,
            build_seconds);

        std::cout << "mode=" << args.mode
                  << " ef=" << args.ef
                  << " recall=" << recall
                  << " p99_us=" << p99
                  << " qps=" << qps
                  << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "hnsw_anchor_runner error: " << error.what() << std::endl;
        return 2;
    }
}
