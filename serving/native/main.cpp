// Native hot path for the news-signal service: epoll sockets, WordPiece, fp16 FinBERT CUDA-graph replay.
// Built by serving.prepare against the installed torch. See serving/native.py for the glue.
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/jit/serialization/pickle.h>
#ifdef USE_ROCM
#include <ATen/hip/HIPGraph.h>
#include <ATen/cuda/tunable/Tunable.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime_api.h>
#define GPU(x) hip##x
namespace gs = c10::hip;
#else
#error "native runtime is only built for ROCm torch"
#endif

using std::string;
using std::vector;

[[noreturn]] static void die(const string& m) {
  std::fprintf(stderr, "serving-native: %s\n", m.c_str());
  std::fflush(stderr);
  std::_Exit(1);
}

// ------------------------------------------------------------------ JSON
struct JV {
  enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
  bool b = false;
  double n = 0;
  string s;
  vector<JV> a;
  vector<std::pair<string, JV>> o;
  const JV* get(const char* k) const {
    for (auto& kv : o)
      if (kv.first == k) return &kv.second;
    return nullptr;
  }
};

static void put_utf8(string& out, uint32_t c) {
  if (c < 0x80) out += (char)c;
  else if (c < 0x800) { out += (char)(0xC0 | (c >> 6)); out += (char)(0x80 | (c & 63)); }
  else if (c < 0x10000) { out += (char)(0xE0 | (c >> 12)); out += (char)(0x80 | ((c >> 6) & 63)); out += (char)(0x80 | (c & 63)); }
  else { out += (char)(0xF0 | (c >> 18)); out += (char)(0x80 | ((c >> 12) & 63)); out += (char)(0x80 | ((c >> 6) & 63)); out += (char)(0x80 | (c & 63)); }
}

struct JP {
  const char *p, *e;
  JP(const char* b, const char* en) : p(b), e(en) {}
  void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) ++p; }
  [[noreturn]] void fail(const char* m) { die(string("json: ") + m); }
  uint32_t hex4() {
    if (e - p < 4) fail("bad \\u");
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
      char c = *p++;
      v <<= 4;
      if (c >= '0' && c <= '9') v |= c - '0';
      else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
      else fail("bad hex");
    }
    return v;
  }
  string str() {  // p is after the opening quote
    string out;
    while (true) {
      if (p >= e) fail("unterminated string");
      char c = *p++;
      if (c == '"') return out;
      if (c != '\\') { out += c; continue; }
      if (p >= e) fail("bad escape");
      char x = *p++;
      switch (x) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          uint32_t u = hex4();
          if (u >= 0xD800 && u < 0xDC00) {
            if (e - p >= 6 && p[0] == '\\' && p[1] == 'u') {
              const char* save = p;
              p += 2;
              uint32_t lo = hex4();
              if (lo >= 0xDC00 && lo < 0xE000) u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
              else { p = save; u = 0xFFFD; }
            } else u = 0xFFFD;
          } else if (u >= 0xDC00 && u < 0xE000) u = 0xFFFD;
          put_utf8(out, u);
          break;
        }
        default: fail("bad escape");
      }
    }
  }
  JV val() {
    ws();
    if (p >= e) fail("eof");
    JV v;
    char c = *p;
    if (c == '{') {
      ++p; v.t = JV::OBJ; ws();
      if (p < e && *p == '}') { ++p; return v; }
      while (true) {
        ws();
        if (p >= e || *p != '"') fail("key");
        ++p;
        string k = str();
        ws();
        if (p >= e || *p != ':') fail(":");
        ++p;
        v.o.emplace_back(std::move(k), val());
        ws();
        if (p < e && *p == ',') { ++p; continue; }
        if (p < e && *p == '}') { ++p; return v; }
        fail("object");
      }
    }
    if (c == '[') {
      ++p; v.t = JV::ARR; ws();
      if (p < e && *p == ']') { ++p; return v; }
      while (true) {
        v.a.push_back(val());
        ws();
        if (p < e && *p == ',') { ++p; continue; }
        if (p < e && *p == ']') { ++p; return v; }
        fail("array");
      }
    }
    if (c == '"') { ++p; v.t = JV::STR; v.s = str(); return v; }
    if (e - p >= 4 && !std::strncmp(p, "true", 4)) { p += 4; v.t = JV::BOOL; v.b = true; return v; }
    if (e - p >= 5 && !std::strncmp(p, "false", 5)) { p += 5; v.t = JV::BOOL; return v; }
    if (e - p >= 4 && !std::strncmp(p, "null", 4)) { p += 4; return v; }
    char* end;
    string tmp(p, std::min<size_t>(e - p, 64));
    v.n = std::strtod(tmp.c_str(), &end);
    if (end == tmp.c_str()) fail("value");
    p += end - tmp.c_str();
    v.t = JV::NUM;
    return v;
  }
};
static JV parse_json(const string& s) {
  JP p(s.data(), s.data() + s.size());
  return p.val();
}
static void json_escape(string& out, const string& s) {
  out += '"';
  for (unsigned char c : s) {
    if (c == '"') out += "\\\"";
    else if (c == '\\') out += "\\\\";
    else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b; }
    else out += (char)c;
  }
  out += '"';
}

// ------------------------------------------------------------------ tokenizer
struct Tokenizer {
  std::unordered_map<string, int> vocab;
  int cls = 101, sep = 102, unk = 100, pad = 0;
  struct Range { uint32_t lo, hi; int cls; };
  vector<Range> ranges;  // sorted, class != 0, code points >= 128
  std::unordered_map<uint32_t, vector<uint32_t>> maps;  // NFD / strip-accent / lowercase for cp >= 128
  static constexpr int MAXLEN = 64;

  void load(const string& dir) {
    {
      std::ifstream f(dir + "/vocab.txt");
      if (!f) die("cannot open vocab.txt");
      string line;
      int i = 0;
      while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        vocab.emplace(line, i++);
      }
    }
    {
      std::ifstream mf(dir + "/meta.txt");
      string k, rest;
      while (mf >> k) {
        if (k == "cls") mf >> cls;
        else if (k == "sep") mf >> sep;
        else if (k == "unk") mf >> unk;
        else if (k == "pad") mf >> pad;
        else std::getline(mf, rest);
      }
    }
    std::ifstream f(dir + "/unicode.tab");
    if (!f) die("cannot open unicode.tab");
    string tag;
    while (f >> tag) {
      if (tag == "R") {
        Range r;
        f >> r.lo >> r.hi >> r.cls;
        ranges.push_back(r);
      } else if (tag == "M") {
        uint32_t cp; int n;
        f >> cp >> n;
        vector<uint32_t> v(n);
        for (auto& x : v) f >> x;
        maps.emplace(cp, std::move(v));
      }
    }
    std::sort(ranges.begin(), ranges.end(), [](const Range& a, const Range& b) { return a.lo < b.lo; });
  }
  // 0 other, 1 whitespace, 2 punctuation, 3 control, 4 CJK
  int cclass(uint32_t c) const {
    if (c < 128) {
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') return 1;
      if (c < 32 || c == 127) return 3;
      if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) || (c >= 123 && c <= 126)) return 2;
      return 0;
    }
    size_t lo = 0, hi = ranges.size();
    while (lo < hi) {
      size_t m = (lo + hi) / 2;
      if (ranges[m].hi < c) lo = m + 1; else hi = m;
    }
    return (lo < ranges.size() && ranges[lo].lo <= c) ? ranges[lo].cls : 0;
  }
  static void decode(const string& s, vector<uint32_t>& out) {
    size_t i = 0, n = s.size();
    while (i < n) {
      unsigned char c = s[i];
      uint32_t cp; int len;
      if (c < 0x80) { cp = c; len = 1; }
      else if ((c >> 5) == 6) { cp = c & 31; len = 2; }
      else if ((c >> 4) == 14) { cp = c & 15; len = 3; }
      else if ((c >> 3) == 30) { cp = c & 7; len = 4; }
      else { out.push_back(0xFFFD); ++i; continue; }
      if (i + len > n) { out.push_back(0xFFFD); break; }
      bool ok = true;
      for (int k = 1; k < len; k++) {
        unsigned char d = s[i + k];
        if ((d >> 6) != 2) { ok = false; break; }
        cp = (cp << 6) | (d & 63);
      }
      if (!ok) { out.push_back(0xFFFD); ++i; continue; }
      out.push_back(cp);
      i += len;
    }
  }
  void wordpiece(const vector<uint32_t>& w, vector<int>& out) const {
    if (w.size() > 100) { out.push_back(unk); return; }
    size_t start = 0, n = w.size(), base = out.size();
    string sub;
    while (start < n) {
      size_t end = n;
      int found = -1;
      while (end > start) {
        sub.clear();
        if (start > 0) sub = "##";
        for (size_t i = start; i < end; i++) put_utf8(sub, w[i]);
        auto it = vocab.find(sub);
        if (it != vocab.end()) { found = it->second; break; }
        --end;
      }
      if (found < 0) { out.resize(base); out.push_back(unk); return; }
      out.push_back(found);
      start = end;
    }
  }
  // headline -> ids with [CLS]/[SEP], truncated to MAXLEN
  void encode(const string& text, vector<int>& out) const {
    vector<uint32_t> raw, norm;
    decode(text, raw);
    norm.reserve(raw.size() + 8);
    auto push_mapped = [&](uint32_t c) {
      if (c < 128) { norm.push_back(c >= 'A' && c <= 'Z' ? c + 32 : c); return; }
      auto it = maps.find(c);
      if (it == maps.end()) norm.push_back(c);
      else for (uint32_t x : it->second) norm.push_back(x);
    };
    for (uint32_t c : raw) {
      if (c == 0 || c == 0xFFFD) continue;
      int k = cclass(c);
      if (k == 3) continue;
      if (k == 1) { norm.push_back(' '); continue; }
      if (k == 4) { norm.push_back(' '); push_mapped(c); norm.push_back(' '); continue; }
      push_mapped(c);
    }
    out.clear();
    out.push_back(cls);
    vector<uint32_t> word;
    auto flush = [&]() {
      if (!word.empty()) { wordpiece(word, out); word.clear(); }
    };
    for (uint32_t c : norm) {
      int k = cclass(c);
      if (k == 1 || c == 0) flush();
      else if (k == 2) { flush(); word.push_back(c); flush(); }
      else word.push_back(c);
      if (out.size() >= (size_t)MAXLEN - 1) break;
    }
    flush();
    if (out.size() > (size_t)MAXLEN - 1) out.resize(MAXLEN - 1);
    out.push_back(sep);
  }
};

// ------------------------------------------------------------------ model
static const int BATCH_BUCKETS[] = {1, 2, 4, 8, 16, 32, 64};
static const int SEQ_BUCKETS[] = {16, 32, 48, 64};
static const int NB = 7, NS = 4, MAX_BATCH = 64;

struct Meta {
  int heads = 12;
  double eps = 1e-12;
  int dirs[3] = {0, 0, 0};
  int ncls = 3;
};

struct Model {
  vector<at::Tensor> W;
  int layers = 0;
  Meta meta;
  at::Device dev = at::kCPU;
  at::ScalarType dt = at::kHalf;
  vector<at::Tensor> QW, QB;  // arm B: QKV weight/bias with 1/sqrt(d) folded into the Q rows (cuda only)

  void load(const string& dir, const string& device) {
    std::ifstream f(dir + "/meta.txt");
    if (!f) die("cannot open meta.txt");
    string k;
    while (f >> k) {
      if (k == "heads") f >> meta.heads;
      else if (k == "eps") f >> meta.eps;
      else if (k == "dirs") { meta.ncls = 3; for (int i = 0; i < 3; i++) f >> meta.dirs[i]; }
      else { string skip; std::getline(f, skip); }
    }
    std::ifstream wf(dir + "/weights.pt", std::ios::binary);
    if (!wf) die("cannot open weights.pt");
    vector<char> buf((std::istreambuf_iterator<char>(wf)), std::istreambuf_iterator<char>());
    auto iv = torch::pickle_load(buf);
    auto lst = iv.toList();
    bool cuda = device == "cuda";
    dev = cuda ? at::Device(at::kCUDA, 0) : at::Device(at::kCPU);
    dt = cuda ? at::kHalf : at::kFloat;
    for (size_t i = 0; i < lst.size(); i++) W.push_back(lst.get(i).toTensor().to(dev, dt));
    layers = ((int)W.size() - 9) / 12;
    if (cuda) {
      int64_t H = W[0].size(1);
      double scale = std::pow((double)(H / meta.heads), -0.5);
      for (int l = 0; l < layers; l++) {
        const at::Tensor* w = &W[5 + 12 * l];
        QW.push_back(w[0].clone());
        QB.push_back(w[1].clone());
        QW.back().slice(0, 0, H).mul_(scale);
        QB.back().slice(0, 0, H).mul_(scale);
      }
    }
  }
  // in: [2, B, S] int64 (ids, attention mask). armb: folded Q scale, fused bias+GELU, one contiguous q/k/v copy.
  at::Tensor forward(const at::Tensor& in, bool armb = false) const {
    int64_t B = in.size(1), S = in.size(2), H = W[0].size(1), nh = meta.heads, d = H / nh;
    auto ids = in[0], am = in[1];
    auto x = at::index_select(W[0], 0, ids.reshape({-1})).view({B, S, H}) + W[1].slice(0, 0, S) + W[2];
    x = at::layer_norm(x, {H}, W[3], W[4], meta.eps);
    auto amh = am.to(dt);
    auto mask = ((1 - amh) * -1e4).view({B, 1, 1, S});
    double scale = std::pow((double)d, -0.5);
    for (int l = 0; l < layers; l++) {
      const at::Tensor* w = &W[5 + 12 * l];
      at::Tensor qkv, p, att;
      if (armb) qkv = at::addmm(QB[l], x.view({B * S, H}), QW[l].t()).view({B, S, 3, nh, d});
      else qkv = at::addmm(w[1], x.view({B * S, H}), w[0].t()).view({B, S, 3, nh, d});
      p = qkv.permute({2, 0, 3, 1, 4});
      if (armb) p = p.contiguous();
      auto q = p[0], k = p[1], v = p[2];
      if (armb) att = at::softmax(at::matmul(q, k.transpose(-1, -2)) + mask, -1);
      else att = at::softmax(at::matmul(q, k.transpose(-1, -2)) * scale + mask, -1);
      auto a = at::matmul(att, v).transpose(1, 2).reshape({B * S, H});
      auto h = at::addmm(w[3], a, w[2].t()).view({B, S, H});
      x = at::layer_norm(x + h, {H}, w[4], w[5], meta.eps);
      auto f = armb ? at::_addmm_activation(w[7], x.view({B * S, H}), w[6].t(), 1, 1, /*use_gelu=*/true)
                    : at::gelu(at::addmm(w[7], x.view({B * S, H}), w[6].t()));
      h = at::addmm(w[9], f, w[8].t()).view({B, S, H});
      x = at::layer_norm(x + h, {H}, w[10], w[11], meta.eps);
    }
    size_t o = 5 + 12 * layers;
    auto pooled = at::tanh(at::addmm(W[o + 1], x.select(1, 0), W[o].t()));
    auto logits = at::addmm(W[o + 3], pooled, W[o + 2].t());
    return at::softmax(logits.to(at::kFloat), -1);
  }
};

struct Bucket {
  at::Tensor dev_in, host_in, out_dev, host_out;
  std::unique_ptr<at::cuda::CUDAGraph> g;
};

struct Runner {
  Model model;
  bool cuda = false;
  int pad_id = 0;
  std::unique_ptr<gs::HIPStream> st;
  Bucket bk[NB][NS];
  bool armb[NB][NS] = {};
  GPU(Event_t) ev{};

  static bool is_file(const string& f) { std::ifstream x(f); return (bool)x; }
  // Reads <dir>/arms.json (one bucket per line, written by --tune). Anything missing or odd means arm A, untuned.
  // Returns per-bucket arm names ("A", "B", "A+tuned", "B+tuned").
  static void read_arms(const string& dir, string arms[NB][NS]) {
    try {
      std::ifstream f(dir + "/arms.json");
      string line;
      while (std::getline(f, line)) {
        size_t a = line.find('"'), x, e;
        if (a == string::npos) continue;
        int b = std::atoi(line.c_str() + a + 1);
        x = line.find('x', a);
        if (x == string::npos) continue;
        int s = std::atoi(line.c_str() + x + 1);
        size_t k = line.find("\"arm\"");
        if (k == string::npos) continue;
        a = line.find('"', k + 5);
        if (a == string::npos) continue;
        e = line.find('"', a + 1);
        if (e == string::npos) continue;
        string arm = line.substr(a + 1, e - a - 1);
        if (arm != "A" && arm != "B" && arm != "A+tuned" && arm != "B+tuned") continue;
        for (int bi = 0; bi < NB; bi++)
          for (int si = 0; si < NS; si++)
            if (BATCH_BUCKETS[bi] == b && SEQ_BUCKETS[si] == s) arms[bi][si] = arm;
      }
    } catch (...) {
      for (int bi = 0; bi < NB; bi++)
        for (int si = 0; si < NS; si++) arms[bi][si] = "A";
    }
  }
  // Serve-time tunable-op configuration: only loads results, never tunes.
  static bool load_tunable(const string& dir) {
    try {
      string csv = dir + "/tunableop_results.csv";
      if (!is_file(csv)) return false;
      auto* ctx = at::cuda::tunable::getTuningContext();
      ctx->EnableTuning(false);
      ctx->WriteFileOnExit(false);
      ctx->SetFilename(csv, false);
      ctx->GetTuningResultsManager();  // first access reads the file
      return true;
    } catch (...) { return false; }
  }

  void init(const string& dir, const string& device, int pad) {
    cuda = device == "cuda";
    pad_id = pad;
    model.load(dir, device);
    c10::InferenceMode im;
    if (!cuda) return;
    st.reset(new gs::HIPStream(gs::getStreamFromPool()));
    gs::setCurrentHIPStream(*st);
    GPU(EventCreateWithFlags)(&ev, GPU(EventDisableTiming));
    auto pool = at::cuda::graph_pool_handle();
    auto lopt = at::TensorOptions().dtype(at::kLong);
    string arms[NB][NS];
    for (int bi = 0; bi < NB; bi++)
      for (int si = 0; si < NS; si++) arms[bi][si] = "A";
    read_arms(dir, arms);
    bool have_csv = load_tunable(dir);
    int nB = 0, nT = 0;
    for (int bi = 0; bi < NB; bi++)
      for (int si = 0; si < NS; si++) {
        int b = BATCH_BUCKETS[bi], s = SEQ_BUCKETS[si];
        Bucket& k = bk[bi][si];
        const string& arm = arms[bi][si];
        armb[bi][si] = arm[0] == 'B';
        bool tuned = have_csv && arm.size() > 1;
        nB += armb[bi][si];
        nT += tuned;
        try {
          at::cuda::tunable::getTuningContext()->EnableTunableOp(tuned);
        } catch (...) {}
        k.dev_in = at::zeros({2, b, s}, lopt.device(model.dev));
        k.dev_in[0].fill_(pad_id);
        k.dev_in[1].select(1, 0).fill_(1);
        k.host_in = at::empty({2, b, s}, lopt.pinned_memory(true));
        k.host_out = at::empty({b, 3}, at::TensorOptions().dtype(at::kFloat).pinned_memory(true));
        for (int i = 0; i < 3; i++) model.forward(k.dev_in, armb[bi][si]);
        GPU(StreamSynchronize)(st->stream());
        k.g.reset(new at::cuda::CUDAGraph());
        k.g->capture_begin(pool);
        k.out_dev = model.forward(k.dev_in, armb[bi][si]);
        k.g->capture_end();
        GPU(StreamSynchronize)(st->stream());
      }
    std::fprintf(stderr, "serving-native: arms: %d/%d buckets arm B, %d tuned\n", nB, NB * NS, nT);
  }
  static int pick(const int* t, int n, int v) {
    for (int i = 0; i < n; i++) if (t[i] >= v) return i;
    return -1;
  }
  // Launch a batch; returns the bucket (async on cuda, immediate on cpu).
  Bucket* launch(const vector<const vector<int>*>& rows, int& n_out) {
    int n = rows.size(), len = 0;
    for (auto r : rows) len = std::max<int>(len, r->size());
    int bi = pick(BATCH_BUCKETS, NB, n), si = pick(SEQ_BUCKETS, NS, len);
    if (bi < 0 || si < 0) die("no bucket");
    Bucket& k = bk[bi][si];
    int b = BATCH_BUCKETS[bi], s = SEQ_BUCKETS[si];
    n_out = n;
    c10::InferenceMode im;
    at::Tensor host = cuda ? k.host_in : at::empty({2, b, s}, at::kLong);
    int64_t* p = host.data_ptr<int64_t>();
    int64_t* ids = p;
    int64_t* mask = p + (int64_t)b * s;
    std::fill(ids, ids + (int64_t)b * s, (int64_t)pad_id);
    std::fill(mask, mask + (int64_t)b * s, 0);
    for (int i = 0; i < n; i++) {
      const vector<int>& r = *rows[i];
      for (size_t j = 0; j < r.size(); j++) { ids[(int64_t)i * s + j] = r[j]; mask[(int64_t)i * s + j] = 1; }
    }
    if (cuda) {
      GPU(MemcpyAsync)(k.dev_in.data_ptr(), p, 2 * (size_t)b * s * 8, GPU(MemcpyHostToDevice), st->stream());
      k.g->replay();
      GPU(MemcpyAsync)(k.host_out.data_ptr(), k.out_dev.data_ptr(), (size_t)b * 3 * 4, GPU(MemcpyDeviceToHost), st->stream());
      GPU(EventRecord)(ev, st->stream());
    } else {
      k.host_out = model.forward(host).contiguous();  // arm A
    }
    return &k;
  }
  bool done() const { return !cuda || GPU(EventQuery)(ev) == GPU(Success); }
  void wait() const { if (cuda) GPU(EventSynchronize)(ev); }
};

// ------------------------------------------------------------------ HTTP / WebSocket
struct Resp { int status = 0; string body; bool close = false; };

// 0 incomplete, 1 complete (used = bytes consumed), -1 error
static int parse_resp(const string& b, Resp& r, size_t& used, bool eof = false) {
  size_t he = b.find("\r\n\r\n");
  if (he == string::npos) return 0;
  if (b.size() < 12 || b.compare(0, 5, "HTTP/") != 0) return -1;
  r.status = std::atoi(b.c_str() + 9);
  string h = b.substr(0, he);
  for (auto& c : h) c = (char)std::tolower((unsigned char)c);
  r.close = h.find("connection: close") != string::npos;
  size_t bs = he + 4;
  size_t cl = h.find("content-length:");
  if (h.find("transfer-encoding: chunked") != string::npos) {
    size_t pos = bs;
    r.body.clear();
    while (true) {
      size_t le = b.find("\r\n", pos);
      if (le == string::npos) return 0;
      size_t sz = std::strtoul(b.c_str() + pos, nullptr, 16);
      if (sz == 0) {
        if (b.size() < le + 4) return 0;
        used = le + 4;
        return 1;
      }
      if (b.size() < le + 2 + sz + 2) return 0;
      r.body.append(b, le + 2, sz);
      pos = le + 2 + sz + 2;
    }
  }
  if (cl != string::npos) {
    size_t n = std::strtoul(h.c_str() + cl + 15, nullptr, 10);
    if (b.size() < bs + n) return 0;
    r.body.assign(b, bs, n);
    used = bs + n;
    return 1;
  }
  if (r.status == 204 || r.status == 304) { used = bs; return 1; }
  if (!eof) return 0;
  r.body.assign(b, bs, string::npos);
  used = b.size();
  return 1;
}

struct Addr { string host; int port = 80; sockaddr_in sa{}; };

static Addr parse_url(string u) {
  Addr a;
  size_t p = u.find("://");
  if (p != string::npos) u = u.substr(p + 3);
  while (!u.empty() && u.back() == '/') u.pop_back();
  size_t c = u.rfind(':');
  a.host = c == string::npos ? u : u.substr(0, c);
  if (c != string::npos) a.port = std::atoi(u.c_str() + c + 1);
  addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  if (getaddrinfo(a.host.c_str(), nullptr, &hints, &res) != 0 || !res) die("cannot resolve " + a.host);
  a.sa = *(sockaddr_in*)res->ai_addr;
  a.sa.sin_port = htons(a.port);
  freeaddrinfo(res);
  return a;
}
static int connect_to(const Addr& a) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) die("socket");
  if (connect(fd, (sockaddr*)&a.sa, sizeof a.sa) != 0) die("connect failed");
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  return fd;
}
static void send_all(int fd, const string& s) {
  size_t o = 0;
  while (o < s.size()) {
    ssize_t n = send(fd, s.data() + o, s.size() - o, MSG_NOSIGNAL);
    if (n <= 0) die("send failed");
    o += n;
  }
}
static string host_hdr(const Addr& a) { return a.host + ":" + std::to_string(a.port); }

// One blocking request on a fresh connection (startup only).
static Resp http_once(const Addr& a, const string& method, const string& path) {
  int fd = connect_to(a);
  string req = method + " " + path + " HTTP/1.1\r\nHost: " + host_hdr(a) + "\r\nConnection: close\r\n";
  if (method == "POST") req += "Content-Length: 0\r\n";
  req += "\r\n";
  send_all(fd, req);
  string buf;
  char tmp[8192];
  Resp r;
  size_t used = 0;
  bool eof = false;
  while (true) {
    int rc = parse_resp(buf, r, used, eof);
    if (rc == 1) break;
    if (rc < 0 || eof) die("bad response to " + path);
    ssize_t n = recv(fd, tmp, sizeof tmp, 0);
    if (n <= 0) eof = true; else buf.append(tmp, n);
  }
  close(fd);
  return r;
}

static string b64(const unsigned char* d, int n) {
  static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  string o;
  for (int i = 0; i < n; i += 3) {
    uint32_t v = d[i] << 16 | (i + 1 < n ? d[i + 1] << 8 : 0) | (i + 2 < n ? d[i + 2] : 0);
    o += t[v >> 18]; o += t[(v >> 12) & 63];
    o += i + 1 < n ? t[(v >> 6) & 63] : '='; o += i + 2 < n ? t[v & 63] : '=';
  }
  return o;
}

enum Kind { K_WS, K_POLL, K_POST };
struct Conn {
  int fd = -1;
  Kind kind = K_POST;
  int feed = -1;
  string rb, wb, frag;
  size_t wo = 0;
  bool busy = false;
};
struct Feed {
  string name, path;
  bool ws = false, ended = false;
  long after = 0;
  Conn* conn = nullptr;
};
struct Story { string id; vector<string> tickers; vector<int> ids; };

struct App {
  Addr addr;
  Tokenizer tok;
  Runner run;
  vector<Feed> feeds;
  int ep = -1;
  std::unordered_set<string> seen;
  std::deque<Story> pending;
  vector<Story> inflight;
  Bucket* inb = nullptr;
  vector<Conn*> posts;
  int posts_out = 0;
  bool stop = false;
  int errors = 0;
  Meta* meta = nullptr;

  void ctl(Conn* c, bool out) {
    epoll_event e{};
    e.events = EPOLLIN | (out ? EPOLLOUT : 0);
    e.data.ptr = c;
    epoll_ctl(ep, EPOLL_CTL_MOD, c->fd, &e);
  }
  void add(Conn* c) {
    int fl = fcntl(c->fd, F_GETFL, 0);
    fcntl(c->fd, F_SETFL, fl | O_NONBLOCK);
    epoll_event e{};
    e.events = EPOLLIN;
    e.data.ptr = c;
    epoll_ctl(ep, EPOLL_CTL_ADD, c->fd, &e);
  }
  void flush(Conn* c) {
    while (c->wo < c->wb.size()) {
      ssize_t n = send(c->fd, c->wb.data() + c->wo, c->wb.size() - c->wo, MSG_NOSIGNAL);
      if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) { ctl(c, true); return; }
        die("send error");
      }
      c->wo += n;
    }
    c->wb.clear();
    c->wo = 0;
    ctl(c, false);
  }
  void close_conn(Conn* c) {
    if (c->fd >= 0) { epoll_ctl(ep, EPOLL_CTL_DEL, c->fd, nullptr); close(c->fd); c->fd = -1; }
  }

  // ---- startup
  void connect_ws(int fi) {
    Feed& f = feeds[fi];
    int fd = connect_to(addr);
    unsigned char key[16];
    std::random_device rd;
    for (auto& k : key) k = (unsigned char)rd();
    string req = "GET " + f.path + " HTTP/1.1\r\nHost: " + host_hdr(addr) +
                 "\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: " + b64(key, 16) +
                 "\r\nSec-WebSocket-Version: 13\r\n\r\n";
    send_all(fd, req);
    string buf;
    char tmp[4096];
    size_t he;
    while ((he = buf.find("\r\n\r\n")) == string::npos) {
      ssize_t n = recv(fd, tmp, sizeof tmp, 0);
      if (n <= 0) die("ws handshake closed: " + f.name);
      buf.append(tmp, n);
    }
    if (buf.compare(0, 12, "HTTP/1.1 101") != 0) die("ws upgrade refused: " + f.name);
    Conn* c = new Conn();
    c->fd = fd; c->kind = K_WS; c->feed = fi;
    c->rb = buf.substr(he + 4);
    f.conn = c;
  }
  void poll_request(Conn* c) {
    Feed& f = feeds[c->feed];
    c->wb = "GET " + f.path + "?after=" + std::to_string(f.after) + "&wait_ms=1000&limit=500 HTTP/1.1\r\nHost: " +
            host_hdr(addr) + "\r\n\r\n";
    c->wo = 0;
    c->rb.clear();
    flush(c);
  }
  void start_poll(int fi) {
    Conn* c = new Conn();
    c->fd = connect_to(addr); c->kind = K_POLL; c->feed = fi;
    feeds[fi].conn = c;
    add(c);
    poll_request(c);
  }

  // ---- stories
  // returns true for an "end" message
  bool on_msg(const JV& m, int fi) {
    const JV* t = m.get("type");
    if (!t || t->t != JV::STR) die("message without type");
    if (t->s == "end") return true;
    if (t->s != "story") die("unknown message type " + t->s);
    const JV* id = m.get("story_id");
    const JV* hl = m.get("headline");
    const JV* tk = m.get("tickers");
    if (!id || !hl || !tk || id->t != JV::STR || hl->t != JV::STR || tk->t != JV::ARR) die("bad story message");
    if (!seen.insert(id->s).second) return false;
    Story s;
    s.id = id->s;
    for (auto& x : tk->a) s.tickers.push_back(x.s);
    tok.encode(hl->s, s.ids);
    pending.push_back(std::move(s));
    return false;
  }
  void feed_ended(int fi) {
    Feed& f = feeds[fi];
    f.ended = true;
    if (f.conn) { close_conn(f.conn); }
  }
  void ws_data(Conn* c) {
    size_t off = 0;
    const string& b = c->rb;
    while (b.size() - off >= 2 && c->fd >= 0) {
      unsigned char b0 = b[off], b1 = b[off + 1];
      uint64_t len = b1 & 0x7f;
      size_t hdr = 2;
      if (len == 126) { if (b.size() - off < 4) break; len = ((unsigned char)b[off + 2] << 8) | (unsigned char)b[off + 3]; hdr = 4; }
      else if (len == 127) {
        if (b.size() - off < 10) break;
        len = 0;
        for (int i = 0; i < 8; i++) len = (len << 8) | (unsigned char)b[off + 2 + i];
        hdr = 10;
      }
      bool masked = b1 & 0x80;
      if (masked) hdr += 4;
      if (b.size() - off < hdr + len) break;
      string payload(b, off + hdr, len);
      if (masked) for (size_t i = 0; i < len; i++) payload[i] ^= b[off + hdr - 4 + (i & 3)];
      off += hdr + len;
      int op = b0 & 0xf;
      bool fin = b0 & 0x80;
      if (op == 8) { if (!feeds[c->feed].ended) std::fprintf(stderr, "serving-native: ws %s closed without end\n", feeds[c->feed].name.c_str()); feed_ended(c->feed); return; }
      if (op == 9) {
        string pong;
        pong += (char)0x8A; pong += (char)(0x80 | payload.size()); pong.append(4, '\0'); pong += payload;
        c->wb += pong; flush(c);
        continue;
      }
      if (op == 10) continue;
      if (op == 1 || op == 2) c->frag = payload; else if (op == 0) c->frag += payload;
      if (!fin) continue;
      JV m = parse_json(c->frag);
      c->frag.clear();
      if (on_msg(m, c->feed)) { feed_ended(c->feed); return; }
    }
    c->rb.erase(0, off);
  }
  void poll_data(Conn* c) {
    Resp r;
    size_t used = 0;
    int rc = parse_resp(c->rb, r, used);
    if (rc == 0) return;
    if (rc < 0 || r.status != 200) die("poll failed: " + std::to_string(r.status) + " " + r.body.substr(0, 200));
    JV m = parse_json(r.body);
    int fi = c->feed;
    const JV* msgs = m.get("messages");
    const JV* nx = m.get("next");
    const JV* en = m.get("end");
    if (!msgs || !nx || !en) die("bad poll response");
    for (auto& x : msgs->a) on_msg(x, fi);
    feeds[fi].after = (long)nx->n;
    if (en->t == JV::BOOL && en->b) { feed_ended(fi); return; }
    if (r.close) {
      close_conn(c);
      c->fd = connect_to(addr);
      add(c);
    }
    poll_request(c);
  }
  void post_data(Conn* c) {
    Resp r;
    size_t used = 0;
    int rc = parse_resp(c->rb, r, used);
    if (rc == 0) return;
    if (rc < 0) die("bad POST /signals response");
    c->rb.erase(0, used);
    c->busy = false;
    posts_out--;
    if (r.status == 409) stop = true;
    else if (r.status != 200) { std::fprintf(stderr, "serving-native: POST /signals -> %d %s\n", r.status, r.body.substr(0, 200).c_str()); errors++; }
    else {
      JV j = parse_json(r.body);
      const JV* rj = j.get("rejected");
      if (rj && !rj->a.empty()) { std::fprintf(stderr, "serving-native: %zu pairs rejected\n", rj->a.size()); errors++; }
    }
    if (r.close) {
      close_conn(c);
      posts.erase(std::remove(posts.begin(), posts.end(), c), posts.end());
      delete c;
    }
  }
  void on_readable(Conn* c) {
    char buf[65536];
    while (c->fd >= 0) {
      ssize_t n = recv(c->fd, buf, sizeof buf, 0);
      if (n > 0) {
        c->rb.append(buf, n);
        if (c->kind == K_WS) ws_data(c);
        else if (c->kind == K_POLL) poll_data(c);
        else post_data(c);
        if (n < (ssize_t)sizeof buf) break;
        continue;
      }
      if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
      if (n < 0 && errno == EINTR) continue;
      // closed by peer
      if (c->kind == K_WS) {
        if (!feeds[c->feed].ended) std::fprintf(stderr, "serving-native: ws %s dropped without end\n", feeds[c->feed].name.c_str());
        feed_ended(c->feed);
      } else if (c->kind == K_POLL) {
        if (feeds[c->feed].ended) return;
        close_conn(c);
        c->fd = connect_to(addr);
        add(c);
        poll_request(c);
      } else {
        if (c->busy) die("POST connection closed with a request outstanding");
        close_conn(c);
        posts.erase(std::remove(posts.begin(), posts.end(), c), posts.end());
        delete c;
      }
      return;
    }
  }

  // ---- signals
  static void fmt_conf(string& out, float v) {
    char b[32];
    std::snprintf(b, sizeof b, "%.9g", (double)v);
    out += b;
    if (!std::strpbrk(b, ".eE")) out += ".0";
  }
  void send_post(const string& body) {
    Conn* c = nullptr;
    for (Conn* x : posts) if (!x->busy) { c = x; break; }
    if (!c) {
      c = new Conn();
      c->fd = connect_to(addr);
      c->kind = K_POST;
      add(c);
      posts.push_back(c);
    }
    c->busy = true;
    posts_out++;
    c->wb = "POST /signals HTTP/1.1\r\nHost: " + host_hdr(addr) + "\r\nContent-Type: application/json\r\nContent-Length: " +
            std::to_string(body.size()) + "\r\n\r\n" + body;
    c->wo = 0;
    flush(c);
  }
  void finish_batch() {
    const float* pr = inb->host_out.data_ptr<float>();
    string body = "{\"signals\":[";
    int cnt = 0;
    bool first = true;
    for (size_t i = 0; i < inflight.size(); i++) {
      const float* p = pr + i * 3;
      int idx = 0;
      for (int k = 1; k < 3; k++) if (p[k] > p[idx]) idx = k;
      int dir = meta->dirs[idx];
      for (auto& t : inflight[i].tickers) {
        if (cnt == 1024) {
          body += "]}";
          send_post(body);
          body = "{\"signals\":[";
          cnt = 0; first = true;
        }
        if (!first) body += ',';
        first = false;
        body += "{\"story_id\":";
        json_escape(body, inflight[i].id);
        body += ",\"ticker\":";
        json_escape(body, t);
        body += ",\"direction\":" + std::to_string(dir) + ",\"confidence\":";
        fmt_conf(body, p[idx]);
        body += '}';
        cnt++;
      }
    }
    if (cnt) { body += "]}"; send_post(body); }
    inflight.clear();
    inb = nullptr;
  }
  void launch() {
    vector<const vector<int>*> rows;
    while (!pending.empty() && (int)inflight.size() < MAX_BATCH) {
      inflight.push_back(std::move(pending.front()));
      pending.pop_front();
    }
    for (auto& s : inflight) rows.push_back(&s.ids);
    int n;
    inb = run.launch(rows, n);
  }

  int main_loop(const string& newsd, const string& dir, const string& device) {
    addr = parse_url(newsd);
    tok.load(dir);
    run.init(dir, device, tok.pad);
    meta = &run.model.meta;
    Resp fr = http_once(addr, "GET", "/feeds");
    if (fr.status != 200) die("GET /feeds failed");
    JV fj = parse_json(fr.body);
    for (auto& x : fj.get("feeds")->a) {
      Feed f;
      f.name = x.get("name")->s;
      f.path = x.get("path")->s;
      f.ws = x.get("transport")->s == "ws";
      feeds.push_back(f);
    }
    ep = epoll_create1(0);
    for (size_t i = 0; i < feeds.size(); i++) if (feeds[i].ws) { connect_ws(i); add(feeds[i].conn); }
    Resp rr = http_once(addr, "POST", "/ready");
    if (rr.status != 200) die("POST /ready failed");
    for (size_t i = 0; i < feeds.size(); i++) {
      if (feeds[i].ws) { if (!feeds[i].conn->rb.empty()) ws_data(feeds[i].conn); }
      else start_poll(i);
    }
    std::fprintf(stderr, "serving-native: ready, %zu feeds\n", feeds.size());
    epoll_event evs[64];
    while (!stop) {
      int n = epoll_wait(ep, evs, 64, inb ? 0 : -1);
      for (int i = 0; i < n; i++) {
        Conn* c = (Conn*)evs[i].data.ptr;
        if (c->fd < 0) continue;
        if (evs[i].events & EPOLLOUT) flush(c);
        if (evs[i].events & (EPOLLIN | EPOLLHUP | EPOLLERR)) on_readable(c);
      }
      if (inb && run.done()) finish_batch();
      if (!inb && !pending.empty()) launch();
      if (!inb && pending.empty() && posts_out == 0) {
        bool all = true;
        for (auto& f : feeds) all = all && f.ended;
        if (all) break;
      }
    }
    return errors ? 1 : 0;
  }
};

// ------------------------------------------------------------------ diagnostics
static int selftest(const string& dir, const string& device) {
  Tokenizer tok;
  tok.load(dir);
  vector<string> t = {"Stocks rally as earnings beat expectations", "Caf\xc3\xa9 \xe6\x97\xa5\xe6\x9c\xac $AAPL up 5% -- \"record\" high",
                      "", "supercalifragilisticexpialidocious!!", "tab\tsep\nline"};
  for (auto& s : t) {
    vector<int> ids;
    tok.encode(s, ids);
    if (ids.front() != tok.cls || ids.back() != tok.sep || ids.size() > 64) { std::fprintf(stderr, "selftest: bad ids\n"); return 1; }
  }
  std::fprintf(stderr, "selftest: tokenizer ok\n");
  return 0;
}
// Prepare-only: tune GEMMs, time arm A/B x untuned/tuned per bucket, verify numerics, write tunableop CSV + arms.json.
static int tune(const string& dir) {
  using Clock = std::chrono::steady_clock;
  auto t0 = Clock::now();
  auto elapsed = [&]() { return std::chrono::duration<double>(Clock::now() - t0).count(); };
  const double BUDGET_S = 420;  // tuning wall budget; remaining shapes keep default solutions
  Model model;
  model.load(dir, "cuda");
  c10::InferenceMode im;
  auto st = gs::getStreamFromPool();
  gs::setCurrentHIPStream(st);
  auto* ctx = at::cuda::tunable::getTuningContext();
  string csv = dir + "/tunableop_results.csv";
  ctx->SetFilename(csv, false);
  ctx->WriteFileOnExit(false);
  ctx->SetMaxTuningDurationMs(20);
  ctx->SetMaxTuningIterations(30);
  ctx->SetMaxWarmupDurationMs(5);
  ctx->SetMaxWarmupIterations(3);
  ctx->EnableTunableOp(true);
  ctx->EnableTuning(true);
  auto lopt = at::TensorOptions().dtype(at::kLong).device(model.dev);
  int64_t vocab = model.W[0].size(0);
  std::mt19937 rng(12345);
  auto make_in = [&](int b, int s) {
    auto h = at::zeros({2, b, s}, at::kLong);
    int64_t* p = h.data_ptr<int64_t>();
    for (int i = 0; i < b; i++) {
      int len = 2 + (int)(rng() % (unsigned)(s - 1));
      if (i == 0) len = s;
      for (int j = 0; j < s; j++) {
        p[(int64_t)i * s + j] = j < len ? 1 + (int64_t)(rng() % (unsigned)(vocab - 1)) : 0;
        p[(int64_t)b * s + (int64_t)i * s + j] = j < len;
      }
    }
    return h.to(model.dev);
  };
  vector<at::Tensor> inputs;
  // 1) tune eagerly over every bucket, both arms (small batches / sequences first)
  for (int bi = 0; bi < NB; bi++)
    for (int si = 0; si < NS; si++) {
      int b = BATCH_BUCKETS[bi], s = SEQ_BUCKETS[si];
      inputs.push_back(make_in(b, s));
      if (elapsed() > BUDGET_S) ctx->EnableTuning(false);
      for (int arm = 0; arm < 2; arm++)
        for (int r = 0; r < 2; r++) model.forward(inputs.back(), arm);
      GPU(StreamSynchronize)(st.stream());
    }
  ctx->EnableTuning(false);
  bool wrote = false;
  try { wrote = ctx->WriteFile(csv); } catch (...) {}
  std::fprintf(stderr, "tune: tuned in %.1fs, results %s\n", elapsed(), wrote ? "written" : "NOT written");
  // 2) per bucket: capture each variant, check against arm A untuned, time replays
  GPU(Event_t) e0, e1;
  GPU(EventCreate)(&e0);
  GPU(EventCreate)(&e1);
  const char* names[4] = {"A", "B", "A+tuned", "B+tuned"};
  string json = "{\n";
  int cnt[4] = {0, 0, 0, 0};
  bool first = true;
  for (int bi = 0; bi < NB; bi++)
    for (int si = 0; si < NS; si++) {
      int b = BATCH_BUCKETS[bi], s = SEQ_BUCKETS[si];
      auto& in = inputs[bi * NS + si];
      ctx->EnableTunableOp(false);
      at::Tensor ref = model.forward(in, false).to(at::kCPU);
      GPU(StreamSynchronize)(st.stream());
      double ms[4] = {1e30, 1e30, 1e30, 1e30}, dif[4] = {0, 0, 0, 0};
      bool ok[4] = {true, false, false, false};
      for (int v = 0; v < 4; v++) {
        bool armb = v & 1, tuned = v >= 2;
        if (tuned && !wrote) continue;
        try {
          ctx->EnableTunableOp(tuned);
          for (int i = 0; i < 3; i++) model.forward(in, armb);
          GPU(StreamSynchronize)(st.stream());
          at::cuda::CUDAGraph g;
          g.capture_begin();
          at::Tensor out = model.forward(in, armb);
          g.capture_end();
          GPU(StreamSynchronize)(st.stream());
          for (int i = 0; i < 10; i++) g.replay();
          GPU(StreamSynchronize)(st.stream());
          at::Tensor o = out.to(at::kCPU);
          double md = (o - ref).abs().max().item<double>();
          bool am = at::equal(o.argmax(-1), ref.argmax(-1));
          dif[v] = md;
          ok[v] = (md <= 2e-3) && am;
          vector<float> ts;
          for (int i = 0; i < 50; i++) {
            GPU(EventRecord)(e0, st.stream());
            g.replay();
            GPU(EventRecord)(e1, st.stream());
            GPU(EventSynchronize)(e1);
            float t = 0;
            GPU(EventElapsedTime)(&t, e0, e1);
            ts.push_back(t);
          }
          std::sort(ts.begin(), ts.end());
          ms[v] = ts[ts.size() / 2];
        } catch (const std::exception& ex) {
          std::fprintf(stderr, "tune: variant %s failed for %dx%d: %.200s\n", names[v], b, s, ex.what());
          ok[v] = false;
        }
      }
      int best = 0;
      for (int v = 1; v < 4; v++)
        if (ok[v] && ms[v] < 0.98 * ms[best]) best = v;
      cnt[best]++;
      char buf[512];
      std::snprintf(buf, sizeof buf,
                    "%s\"%dx%d\": {\"arm\": \"%s\", \"ms\": {\"A\": %.4f, \"B\": %.4f, \"A+tuned\": %.4f, \"B+tuned\": %.4f}, "
                    "\"maxdiff\": {\"B\": %.2e, \"A+tuned\": %.2e, \"B+tuned\": %.2e}, \"ok\": [%d, %d, %d, %d]}",
                    first ? "" : ",\n", b, s, names[best], ms[0], ms[1], ms[2], ms[3], dif[1], dif[2], dif[3], (int)ok[0], (int)ok[1],
                    (int)ok[2], (int)ok[3]);
      first = false;
      json += buf;
      std::fprintf(stderr, "tune: %dx%d -> %s (A %.3f B %.3f A+t %.3f B+t %.3f ms)\n", b, s, names[best], ms[0], ms[1], ms[2], ms[3]);
    }
  json += "\n}\n";
  std::ofstream(dir + "/arms.json") << json;
  std::printf("arms: A=%d B=%d A+tuned=%d B+tuned=%d (%.0fs)\n", cnt[0], cnt[1], cnt[2], cnt[3], elapsed());
  return 0;
}
static int tokenize_stdin(const string& dir) {
  Tokenizer tok;
  tok.load(dir);
  string line;
  vector<int> ids;
  while (std::getline(std::cin, line)) {
    JV j = parse_json(line);
    tok.encode(j.s, ids);
    string o;
    for (size_t i = 0; i < ids.size(); i++) { if (i) o += ' '; o += std::to_string(ids[i]); }
    std::puts(o.c_str());
  }
  return 0;
}

int main(int argc, char** argv) {
  string newsd, model, prepared, device = "cuda";
  bool st = false, ts = false, tn = false;
  for (int i = 1; i < argc; i++) {
    string a = argv[i];
    auto nxt = [&]() { if (i + 1 >= argc) die("missing value for " + a); return string(argv[++i]); };
    if (a == "--newsd") newsd = nxt();
    else if (a == "--model") model = nxt();
    else if (a == "--prepared") prepared = nxt();
    else if (a == "--device") device = nxt();
    else if (a == "--selftest") st = true;
    else if (a == "--tokenize-stdin") ts = true;
    else if (a == "--tune") tn = true;
    else die("unknown argument " + a);
  }
  if (prepared.empty()) die("--prepared required");
  string dir = prepared + "/native";
  if (ts) return tokenize_stdin(dir);
  if (st) return selftest(dir, device);
  if (tn) return tune(dir);
  if (newsd.empty()) die("--newsd required");
  App app;
  return app.main_loop(newsd, dir, device);
}
