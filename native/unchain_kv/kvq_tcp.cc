#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" unsigned long crc32(unsigned long checksum,
                                const unsigned char* data,
                                unsigned int size);

namespace {

using FrameCallback = void (*)(unsigned char*, unsigned long long, void*);

std::mutex g_error_mu;
std::string g_last_error;
std::atomic<int> g_listen_fd{-1};
std::atomic<bool> g_stop{false};

constexpr char kBlockMagic[] = "KVB1";
constexpr char kNativeMagic[] = "KVN1";
constexpr char kKvMagic[] = "KVK1";
constexpr char kCompressedNativeMagic[] = "KVC1";

uint32_t Crc32(uint32_t checksum, const void* data, uint64_t size) {
  auto* current = static_cast<const unsigned char*>(data);
  while (size != 0) {
    auto chunk = static_cast<unsigned int>(std::min<uint64_t>(size, UINT_MAX));
    checksum = static_cast<uint32_t>(crc32(checksum, current, chunk));
    current += chunk;
    size -= chunk;
  }
  return checksum;
}

struct HostPort {
  std::string host;
  std::string port;
};

void SetError(const std::string& message) {
  std::lock_guard<std::mutex> lock(g_error_mu);
  g_last_error = message;
}

HostPort ParseHostPort(const char* value) {
  std::string text = value ? value : "";
  auto pos = text.rfind(':');
  if (pos == std::string::npos || pos + 1 == text.size()) {
    throw std::runtime_error("address must be HOST:PORT");
  }
  return {text.substr(0, pos), text.substr(pos + 1)};
}

void AppendU32(std::vector<uint8_t>& out, uint32_t value) {
  out.push_back((value >> 24) & 0xff);
  out.push_back((value >> 16) & 0xff);
  out.push_back((value >> 8) & 0xff);
  out.push_back(value & 0xff);
}

void AppendString(std::vector<uint8_t>& out, const char* value) {
  const char* text = value ? value : "";
  size_t len = std::strlen(text);
  if (len > 0xffffffffull) {
    throw std::runtime_error("string too large");
  }
  AppendU32(out, static_cast<uint32_t>(len));
  out.insert(out.end(), text, text + len);
}

std::vector<uint8_t> LayerHeader(const char* magic, const char* transfer_id,
                                 const char* request_id, uint32_t layer,
                                 uint32_t count, uint32_t unit_size) {
  std::vector<uint8_t> out;
  out.insert(out.end(), magic, magic + 4);
  AppendString(out, transfer_id);
  AppendString(out, request_id);
  AppendU32(out, layer);
  AppendU32(out, count);
  AppendU32(out, unit_size);
  return out;
}

int Connect(const char* peer) {
  HostPort hp = ParseHostPort(peer);
  addrinfo hints{};
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_family = AF_UNSPEC;
  addrinfo* result = nullptr;
  int rc = getaddrinfo(hp.host.c_str(), hp.port.c_str(), &hints, &result);
  if (rc != 0) {
    throw std::runtime_error(gai_strerror(rc));
  }
  int fd = -1;
  for (addrinfo* p = result; p != nullptr; p = p->ai_next) {
    fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd < 0) {
      continue;
    }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    if (connect(fd, p->ai_addr, p->ai_addrlen) == 0) {
      break;
    }
    close(fd);
    fd = -1;
  }
  freeaddrinfo(result);
  if (fd < 0) {
    throw std::runtime_error(std::strerror(errno));
  }
  return fd;
}

void WriteAll(int fd, std::vector<iovec> iov) {
  while (!iov.empty()) {
    ssize_t n = writev(fd, iov.data(), static_cast<int>(iov.size()));
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error(std::strerror(errno));
    }
    if (n == 0) {
      throw std::runtime_error("writev made no progress");
    }
    size_t sent = static_cast<size_t>(n);
    while (!iov.empty() && sent >= iov.front().iov_len) {
      sent -= iov.front().iov_len;
      iov.erase(iov.begin());
    }
    if (sent != 0 && !iov.empty()) {
      iov.front().iov_base =
          static_cast<char*>(iov.front().iov_base) + sent;
      iov.front().iov_len -= sent;
    }
  }
}

void SendFrameFd(int fd, std::vector<uint8_t>& header, const void* payload1,
                 uint64_t len1, const void* payload2 = nullptr,
                 uint64_t len2 = 0) {
  uint64_t total = header.size() + len1 + len2;
  if (total > 0xffffffffull) {
    throw std::runtime_error("frame too large");
  }
  uint32_t checksum = Crc32(0, header.data(), header.size());
  if (len1) {
    checksum = Crc32(checksum, payload1, len1);
  }
  if (len2) {
    checksum = Crc32(checksum, payload2, len2);
  }
  uint32_t prefix[] = {
      htonl(static_cast<uint32_t>(total)), htonl(static_cast<uint32_t>(checksum))};
  std::vector<iovec> iov;
  iov.push_back({prefix, sizeof(prefix)});
  iov.push_back({header.data(), header.size()});
  if (len1) {
    iov.push_back({const_cast<void*>(payload1), static_cast<size_t>(len1)});
  }
  if (len2) {
    iov.push_back({const_cast<void*>(payload2), static_cast<size_t>(len2)});
  }
  WriteAll(fd, iov);
}

void SendFrame(const char* peer, std::vector<uint8_t>& header,
               const void* payload1, uint64_t len1, const void* payload2 = nullptr,
               uint64_t len2 = 0) {
  int fd = Connect(peer);
  try {
    SendFrameFd(fd, header, payload1, len1, payload2, len2);
    close(fd);
  } catch (...) {
    close(fd);
    throw;
  }
}

void SendBlock(const char* peer, const char* magic, const char* transfer_id,
               const char* request_id, uint32_t layer, const void* data,
               uint64_t data_len, uint32_t block_size, uint32_t block_count) {
  if (block_size == 0 || block_count == 0) {
    return;
  }
  if (data_len != uint64_t(block_size) * block_count) {
    throw std::runtime_error("payload length does not match metadata");
  }
  auto header = LayerHeader(magic, transfer_id, request_id, layer, block_count,
                            block_size);
  SendFrame(peer, header, data, data_len);
}

void SendBlockFd(int fd, const char* magic, const char* transfer_id,
                 const char* request_id, uint32_t layer, const void* data,
                 uint64_t data_len, uint32_t block_size, uint32_t block_count) {
  if (block_size == 0 || block_count == 0) {
    return;
  }
  if (data_len != uint64_t(block_size) * block_count) {
    throw std::runtime_error("payload length does not match metadata");
  }
  auto header = LayerHeader(magic, transfer_id, request_id, layer, block_count,
                            block_size);
  SendFrameFd(fd, header, data, data_len);
}

std::vector<uint8_t> CompressedNativeHeader(
    const char* transfer_id, const char* request_id, uint32_t layer,
    uint32_t block_count, uint32_t raw_block_size, uint32_t raw_bytes,
    uint32_t codec_code, uint32_t encoded_bytes) {
  std::vector<uint8_t> out;
  out.insert(out.end(), kCompressedNativeMagic, kCompressedNativeMagic + 4);
  AppendString(out, transfer_id);
  AppendString(out, request_id);
  AppendU32(out, layer);
  AppendU32(out, block_count);
  AppendU32(out, raw_block_size);
  AppendU32(out, raw_bytes);
  AppendU32(out, codec_code);
  AppendU32(out, encoded_bytes);
  return out;
}

void SendCompressedNativeFd(int fd, const char* transfer_id,
                            const char* request_id, uint32_t layer,
                            const void* data, uint64_t data_len,
                            uint32_t raw_block_size, uint32_t block_count,
                            uint32_t raw_bytes, uint32_t codec_code) {
  if (raw_block_size == 0 || block_count == 0) {
    return;
  }
  if (data_len > 0xffffffffull) {
    throw std::runtime_error("compressed payload too large");
  }
  auto header = CompressedNativeHeader(
      transfer_id, request_id, layer, block_count, raw_block_size, raw_bytes,
      codec_code, static_cast<uint32_t>(data_len));
  SendFrameFd(fd, header, data, data_len);
}

void SendCompressedNative(const char* peer, const char* transfer_id,
                          const char* request_id, uint32_t layer,
                          const void* data, uint64_t data_len,
                          uint32_t raw_block_size, uint32_t block_count,
                          uint32_t raw_bytes, uint32_t codec_code) {
  int fd = Connect(peer);
  try {
    SendCompressedNativeFd(fd, transfer_id, request_id, layer, data, data_len,
                           raw_block_size, block_count, raw_bytes, codec_code);
    close(fd);
  } catch (...) {
    close(fd);
    throw;
  }
}

bool ReadExact(int fd, void* dst, size_t size, bool clean_eof = false) {
  auto* out = static_cast<uint8_t*>(dst);
  size_t offset = 0;
  while (offset < size) {
    ssize_t n = recv(fd, out + offset, size - offset, 0);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error(std::strerror(errno));
    }
    if (n == 0) {
      if (clean_eof && offset == 0) return false;
      throw std::runtime_error("truncated tcp frame");
    }
    offset += static_cast<size_t>(n);
  }
  return true;
}

int Listen(const char* bind_addr) {
  HostPort hp = ParseHostPort(bind_addr);
  addrinfo hints{};
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_family = AF_UNSPEC;
  hints.ai_flags = AI_PASSIVE;
  addrinfo* result = nullptr;
  int rc = getaddrinfo(hp.host.c_str(), hp.port.c_str(), &hints, &result);
  if (rc != 0) {
    throw std::runtime_error(gai_strerror(rc));
  }
  int fd = -1;
  for (addrinfo* p = result; p != nullptr; p = p->ai_next) {
    fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd < 0) {
      continue;
    }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (::bind(fd, p->ai_addr, p->ai_addrlen) == 0 && listen(fd, 128) == 0) {
      break;
    }
    close(fd);
    fd = -1;
  }
  freeaddrinfo(result);
  if (fd < 0) {
    throw std::runtime_error(std::strerror(errno));
  }
  return fd;
}

void ServeConnection(int fd, FrameCallback callback, void* context) {
  while (!g_stop.load()) {
    uint32_t prefix[2] = {};
    if (!ReadExact(fd, prefix, sizeof(prefix), true)) {
      return;
    }
    uint32_t size = ntohl(prefix[0]);
    uint32_t checksum = ntohl(prefix[1]);
    auto* frame = static_cast<unsigned char*>(std::malloc(size));
    if (frame == nullptr) {
      throw std::runtime_error("malloc failed");
    }
    try {
      ReadExact(fd, frame, size);
      if (Crc32(0, frame, size) != checksum) {
        throw std::runtime_error("tcp frame checksum mismatch");
      }
    } catch (...) {
      std::free(frame);
      throw;
    }
    callback(frame, size, context);
  }
}

int Wrap(const std::function<void()>& fn) {
  try {
    fn();
    return 0;
  } catch (const std::exception& e) {
    SetError(e.what());
    return -1;
  } catch (...) {
    SetError("unknown error");
    return -1;
  }
}

}  // namespace

extern "C" {

int kvq_tcp_send_block_layer(const char* peer, const char* transfer_id,
                             const char* request_id, unsigned layer,
                             const void* data, unsigned long long data_len,
                             unsigned block_size, unsigned block_count) {
  return Wrap([&] {
    SendBlock(peer, kBlockMagic, transfer_id, request_id, layer, data, data_len,
              block_size, block_count);
  });
}

int kvq_tcp_send_native_layer(const char* peer, const char* transfer_id,
                              const char* request_id, unsigned layer,
                              const void* data, unsigned long long data_len,
                              unsigned block_size, unsigned block_count) {
  return Wrap([&] {
    SendBlock(peer, kNativeMagic, transfer_id, request_id, layer, data, data_len,
              block_size, block_count);
  });
}

int kvq_tcp_send_compressed_native_layer(
    const char* peer, const char* transfer_id, const char* request_id,
    unsigned layer, const void* data, unsigned long long data_len,
    unsigned raw_block_size, unsigned block_count, unsigned raw_bytes,
    unsigned codec_code) {
  return Wrap([&] {
    SendCompressedNative(peer, transfer_id, request_id, layer, data, data_len,
                         raw_block_size, block_count, raw_bytes, codec_code);
  });
}

int kvq_tcp_connect(const char* peer) {
  try {
    return Connect(peer);
  } catch (const std::exception& e) {
    SetError(e.what());
    return -1;
  } catch (...) {
    SetError("unknown error");
    return -1;
  }
}

int kvq_tcp_close(int fd) {
  if (fd >= 0) {
    close(fd);
  }
  return 0;
}

int kvq_tcp_send_block_layer_fd(int fd, const char* transfer_id,
                                const char* request_id, unsigned layer,
                                const void* data,
                                unsigned long long data_len,
                                unsigned block_size,
                                unsigned block_count) {
  return Wrap([&] {
    SendBlockFd(fd, kBlockMagic, transfer_id, request_id, layer, data, data_len,
                block_size, block_count);
  });
}

int kvq_tcp_send_native_layer_fd(int fd, const char* transfer_id,
                                 const char* request_id, unsigned layer,
                                 const void* data,
                                 unsigned long long data_len,
                                 unsigned block_size,
                                 unsigned block_count) {
  return Wrap([&] {
    SendBlockFd(fd, kNativeMagic, transfer_id, request_id, layer, data, data_len,
                block_size, block_count);
  });
}

int kvq_tcp_send_compressed_native_layer_fd(
    int fd, const char* transfer_id, const char* request_id, unsigned layer,
    const void* data, unsigned long long data_len, unsigned raw_block_size,
    unsigned block_count, unsigned raw_bytes, unsigned codec_code) {
  return Wrap([&] {
    SendCompressedNativeFd(fd, transfer_id, request_id, layer, data, data_len,
                           raw_block_size, block_count, raw_bytes, codec_code);
  });
}

int kvq_tcp_send_kv_layer_fd(int fd, const char* transfer_id,
                             const char* request_id, unsigned layer,
                             const void* key_data,
                             unsigned long long key_len,
                             const void* value_data,
                             unsigned long long value_len,
                             unsigned part_size, unsigned block_count) {
  return Wrap([&] {
    if (part_size == 0 || block_count == 0) {
      return;
    }
    uint64_t expected = uint64_t(part_size) * block_count;
    if (key_len != expected || value_len != expected) {
      throw std::runtime_error("kv payload length does not match metadata");
    }
    auto header = LayerHeader(kKvMagic, transfer_id, request_id, layer,
                              block_count, part_size);
    SendFrameFd(fd, header, key_data, key_len, value_data, value_len);
  });
}

int kvq_tcp_send_kv_layer(const char* peer, const char* transfer_id,
                          const char* request_id, unsigned layer,
                          const void* key_data, unsigned long long key_len,
                          const void* value_data, unsigned long long value_len,
                          unsigned part_size, unsigned block_count) {
  return Wrap([&] {
    if (part_size == 0 || block_count == 0) {
      return;
    }
    uint64_t expected = uint64_t(part_size) * block_count;
    if (key_len != expected || value_len != expected) {
      throw std::runtime_error("kv payload length does not match metadata");
    }
    auto header = LayerHeader(kKvMagic, transfer_id, request_id, layer,
                              block_count, part_size);
    SendFrame(peer, header, key_data, key_len, value_data, value_len);
  });
}

int kvq_tcp_start_receiver(const char* bind, FrameCallback callback,
                           void* context) {
  return Wrap([&] {
    if (callback == nullptr) {
      throw std::runtime_error("callback is null");
    }
    signal(SIGPIPE, SIG_IGN);
    g_stop.store(false);
    int listener = Listen(bind);
    g_listen_fd.store(listener);
    while (!g_stop.load()) {
      int fd = accept(listener, nullptr, nullptr);
      if (fd < 0) {
        if (errno == EINTR) {
          continue;
        }
        if (g_stop.load()) {
          break;
        }
        throw std::runtime_error(std::strerror(errno));
      }
      try {
        ServeConnection(fd, callback, context);
        close(fd);
      } catch (...) {
        close(fd);
        throw;
      }
    }
    int expected = listener;
    if (g_listen_fd.compare_exchange_strong(expected, -1)) {
      close(listener);
    }
  });
}

int kvq_tcp_stop_receiver() {
  g_stop.store(true);
  int fd = g_listen_fd.exchange(-1);
  if (fd >= 0) {
    shutdown(fd, SHUT_RDWR);
    close(fd);
  }
  return 0;
}

void kvq_tcp_free(void* data) { std::free(data); }

const char* kvq_tcp_last_error() {
  std::lock_guard<std::mutex> lock(g_error_mu);
  return g_last_error.c_str();
}

}  // extern "C"
