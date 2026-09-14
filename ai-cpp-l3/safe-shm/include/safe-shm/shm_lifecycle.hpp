#pragma once
#include "safe-shm/flat_type.hpp"
#include "shm/shm.hpp"
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <signal.h>
#include <string>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace safe_shm
{
    // ── Query utilities ─────────────────────────────────────────────────

    /// Check if a shared memory segment exists.
    inline bool shm_exists(std::string const &name) noexcept
    {
        std::string path = "/dev/shm/" + name;
        struct stat st{};
        return stat(path.c_str(), &st) == 0;
    }

    /// Remove a shared memory segment. Returns true if removed.
    inline bool shm_remove(std::string const &name) noexcept
    {
        std::string path = "/dev/shm/" + name;
        return unlink(path.c_str()) == 0;
    }

    /// List shared memory segments matching an optional prefix.
    inline std::vector<std::string> shm_list(std::string const &prefix = "")
    {
        std::vector<std::string> result;
        DIR *dir = opendir("/dev/shm");
        if (!dir)
            return result;

        struct dirent *entry;
        while ((entry = readdir(dir)) != nullptr)
        {
            std::string name = entry->d_name;
            if (name == "." || name == "..")
                continue;
            if (prefix.empty() || name.starts_with(prefix))
                result.push_back(std::move(name));
        }
        closedir(dir);
        return result;
    }

    // ── SHM Header for liveness detection ───────────────────────────────

    /// Header placed at the beginning of owned shared memory segments.
    struct ShmHeader
    {
        uint32_t magic;       // 0x5348'4D48 ("SHMH")
        uint32_t version;     // protocol version
        uint64_t heartbeat_ns; // CLOCK_MONOTONIC, updated by writer
        int32_t writer_pid;   // PID of the writer process
        uint32_t reserved[3]; // future use, zero-initialized
    };
    static_assert(FlatType<ShmHeader>);
    static_assert(sizeof(ShmHeader) == 32);

    constexpr uint32_t SHM_HEADER_MAGIC = 0x5348'4D48u;
    constexpr uint32_t SHM_HEADER_VERSION = 1u;

    /// Check if the writer process is still alive.
    /// If max_stale_ns > 0, also checks heartbeat freshness.
    inline bool is_writer_alive(ShmHeader const &header,
                                uint64_t max_stale_ns = 0) noexcept
    {
        if (header.magic != SHM_HEADER_MAGIC)
            return false;

        // Check if PID exists
        if (kill(header.writer_pid, 0) != 0)
            return false;

        // Optional heartbeat freshness check
        if (max_stale_ns > 0)
        {
            struct timespec ts{};
            clock_gettime(CLOCK_MONOTONIC, &ts);
            auto now_ns = static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
                          static_cast<uint64_t>(ts.tv_nsec);
            if (now_ns - header.heartbeat_ns > max_stale_ns)
                return false;
        }

        return true;
    }

    // ── RAII Owned Shared Memory ────────────────────────────────────────

    /// Owned shared memory segment with header and automatic cleanup.
    /// The writer creates this; readers use regular SharedMemory/Seqlock/etc.
    ///
    /// Memory layout: [ShmHeader] [padding to alignof(T)] [T data]
    template <FlatType T>
    class OwnedShm
    {
    public:
        static constexpr std::size_t header_offset = 0;
        static constexpr std::size_t data_offset =
            (sizeof(ShmHeader) + alignof(T) - 1) & ~(alignof(T) - 1);
        static constexpr std::size_t total_size = data_offset + sizeof(T);

        explicit OwnedShm(std::string const &shm_name)
            : shm_(shm::path(shm_name), total_size), name_(shm_name)
        {
            // Initialize header
            auto &hdr = header();
            hdr.magic = SHM_HEADER_MAGIC;
            hdr.version = SHM_HEADER_VERSION;
            hdr.writer_pid = static_cast<int32_t>(getpid());
            std::memset(hdr.reserved, 0, sizeof(hdr.reserved));
            update_heartbeat();

            // Zero-initialize data
            std::memset(&data(), 0, sizeof(T));
        }

        OwnedShm(OwnedShm const &) = delete;
        OwnedShm &operator=(OwnedShm const &) = delete;

        OwnedShm(OwnedShm &&) = default;
        OwnedShm &operator=(OwnedShm &&) = default;

        ~OwnedShm() = default; // shm::Shm handles unlink

        ShmHeader &header() noexcept
        {
            return *static_cast<ShmHeader *>(shm_.get());
        }

        ShmHeader const &header() const noexcept
        {
            return *static_cast<ShmHeader const *>(shm_.get());
        }

        T &data() noexcept
        {
            return *reinterpret_cast<T *>(
                static_cast<char *>(shm_.get()) + data_offset);
        }

        T const &data() const noexcept
        {
            return *reinterpret_cast<T const *>(
                static_cast<char const *>(shm_.get()) + data_offset);
        }

        void update_heartbeat() noexcept
        {
            struct timespec ts{};
            clock_gettime(CLOCK_MONOTONIC, &ts);
            header().heartbeat_ns =
                static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
                static_cast<uint64_t>(ts.tv_nsec);
        }

        std::string const &name() const noexcept { return name_; }

    private:
        shm::Shm shm_;
        std::string name_;
    };

} // namespace safe_shm
