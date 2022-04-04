#ifdef TI_WITH_DX11

#include "taichi/backends/dx/dx_info_queue.h"

namespace taichi {
namespace lang {
namespace directx11 {

void check_dx_error(HRESULT hr, const char *msg);

namespace {
inline std::string trim_string(const std::string &s) {
  int begin = 0, end = (int)s.size();
  while (begin < end && std::isspace(s[begin])) {
    begin++;
  }
  while (begin < end && std::isspace(s[end - 1])) {
    end--;
  }
  return std::string(s.begin() + begin, s.begin() + end);
}
}  // namespace

std::string munch_token(std::string &s) {
  if (s.empty()) {
    return "";
  }
  size_t idx = s.find(' ');
  std::string ret;
  ret = trim_string(s.substr(0, idx));
  s = s.substr(idx + 1);
  if (ret.empty() == false && ret.back() == ',')
    ret.pop_back();
  return ret;
}

std::vector<Dx11InfoQueue::Entry> Dx11InfoQueue::parse_reference_count(
    const std::vector<std::string> &messages) {
  std::vector<Dx11InfoQueue::Entry> ret;
  for (std::string line : messages) {
    Dx11InfoQueue::Entry entry;
    line = trim_string(line);
    std::string x;
    x = munch_token(line);

    // Example 1: "Live ID3D11Query at 0x0000018F64E81DA0, Refcount: 0, IntRef:
    // 1" Example 2: "Live ID3D11Buffer at 0x000001F8AF284370, Name: buffer
    // alloc#0 size=1048576, Refcount: 1, IntRef: 1"

    if (x != "Live")
      continue;

    x = munch_token(line);
    entry.type = x;

    x = munch_token(line);
    if (x != "at")
      continue;

    x = munch_token(line);
    if (x.empty())
      continue;

    entry.addr = reinterpret_cast<void *>(std::atoll(x.c_str()));

    while (true) {
      x = munch_token(line);
      if (x == "Refcount:") {
        x = munch_token(line);
        entry.refcount = std::atoi(x.c_str());
      } else if (x == "IntRef:") {
        x = munch_token(line);
        entry.intref = std::atoi(x.c_str());
      } else
        break;
    }
    ret.push_back(entry);
  }
  return ret;
}

Dx11InfoQueue::Dx11InfoQueue(ID3D11Device *device)
    : device_(device), last_message_count_(0) {
  init();
}

void Dx11InfoQueue::init() {
  typedef HRESULT(WINAPI * DXGIGetDebugInterface)(REFIID, void **);

  HRESULT hr;
  hr = device_->QueryInterface(__uuidof(ID3D11InfoQueue),
                               reinterpret_cast<void **>(&info_queue_));
  check_dx_error(hr, "Query ID3D11InfoQueue interface from the DX11 device");
  hr = device_->QueryInterface(__uuidof(ID3D11Debug),
                               reinterpret_cast<void **>(&debug_));
  check_dx_error(hr, "Query ID3D11Debug interface from the DX11 device");
}

std::vector<std::string> Dx11InfoQueue::get_updated_messages() {
  std::vector<std::string> ret;
  if (!info_queue_) {
    return ret;
  }
  const int num_messages = info_queue_->GetNumStoredMessages();
  const int n = num_messages - last_message_count_;
  ret.resize(n);
  for (int i = 0; i < n; i++) {
    D3D11_MESSAGE *msg;
    size_t len = 0;
    HRESULT hr =
        info_queue_->GetMessageW(i + last_message_count_, nullptr, &len);
    check_dx_error(hr, "Check D3D11 info queue message length");
    msg = (D3D11_MESSAGE *)malloc(len);
    hr = info_queue_->GetMessageW(i + last_message_count_, msg, &len);
    check_dx_error(hr, "Obtain D3D11 info queue message content");
    ret[i] = std::string(msg->pDescription);
    free(msg);
  }
  last_message_count_ = num_messages;
  return ret;
}

bool Dx11InfoQueue::has_updated_messages() {
  if (!info_queue_) {
    return false;
  }
  const int n = info_queue_->GetNumStoredMessages();
  return n > last_message_count_;
}

int Dx11InfoQueue::live_object_count() {
  get_updated_messages();  // Drain message queue
  debug_->ReportLiveDeviceObjects(D3D11_RLDO_DETAIL);
  if (has_updated_messages()) {
    live_objects_ = parse_reference_count(get_updated_messages());
  }
  return static_cast<int>(live_objects_.size());
}

}  // namespace directx11
}  // namespace lang
}  // namespace taichi

#endif