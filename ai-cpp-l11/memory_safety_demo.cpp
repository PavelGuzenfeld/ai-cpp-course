#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <stdexcept>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

namespace nb = nanobind;

// ============================================================================
// Simple types used by the demos
// ============================================================================

struct BBox {
    double x, y, w, h;

    double area() const { return w * h; }

    std::string repr() const {
        return "BBox(x=" + std::to_string(x) + ", y=" + std::to_string(y) +
               ", w=" + std::to_string(w) + ", h=" + std::to_string(h) + ")";
    }
};

struct Model {
    std::string name;
    int param_count;
    bool loaded;

    Model(std::string name_, int params)
        : name(std::move(name_)), param_count(params), loaded(true) {}

    std::string info() const {
        return "Model('" + name + "', params=" + std::to_string(param_count) +
               ", loaded=" + (loaded ? "true" : "false") + ")";
    }
};

struct Buffer {
    std::vector<double> data;
    std::string label;

    Buffer(size_t size, std::string label_)
        : data(size, 0.0), label(std::move(label_)) {}

    size_t size() const { return data.size(); }
};

// ============================================================================
// RAII — Resources Clean Themselves
// ============================================================================

class RAIIBuffer {
    std::unique_ptr<double[]> data_;
    size_t size_;
    static inline int active_count_ = 0;

public:
    RAIIBuffer(size_t n) : data_(std::make_unique<double[]>(n)), size_(n) {
        // Fill with zeros (make_unique already does this for arrays)
        ++active_count_;
    }

    // Destructor runs automatically when RAIIBuffer leaves scope
    ~RAIIBuffer() {
        --active_count_;
        // data_ is freed automatically by unique_ptr
    }

    // Non-copyable (single ownership)
    RAIIBuffer(const RAIIBuffer&) = delete;
    RAIIBuffer& operator=(const RAIIBuffer&) = delete;

    // Movable
    RAIIBuffer(RAIIBuffer&& other) noexcept
        : data_(std::move(other.data_)), size_(other.size_) {
        other.size_ = 0;
    }

    RAIIBuffer& operator=(RAIIBuffer&& other) noexcept {
        data_ = std::move(other.data_);
        size_ = other.size_;
        other.size_ = 0;
        return *this;
    }

    size_t size() const { return size_; }

    double get(size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("RAIIBuffer: index " +
                                     std::to_string(index) +
                                     " out of range for buffer of size " +
                                     std::to_string(size_));
        }
        return data_[index];
    }

    void set(size_t index, double value) {
        if (index >= size_) {
            throw std::out_of_range("RAIIBuffer: index " +
                                     std::to_string(index) +
                                     " out of range for buffer of size " +
                                     std::to_string(size_));
        }
        data_[index] = value;
    }

    static int active_count() { return active_count_; }
};

// ============================================================================
// std::optional — No More Null Pointers
// ============================================================================

class OptionalDetection {
    std::optional<BBox> detection_;

public:
    // Construct with a detection
    OptionalDetection(double x, double y, double w, double h)
        : detection_(BBox{x, y, w, h}) {}

    // Construct empty (no detection)
    OptionalDetection() : detection_(std::nullopt) {}

    bool has_value() const { return detection_.has_value(); }

    BBox value() const {
        if (!detection_.has_value()) {
            throw std::runtime_error(
                "OptionalDetection::value() called on empty detection");
        }
        return detection_.value();
    }

    BBox value_or(double x, double y, double w, double h) const {
        return detection_.value_or(BBox{x, y, w, h});
    }

    double area() const {
        if (!detection_.has_value()) {
            return 0.0;
        }
        return detection_->area();
    }

    std::string repr() const {
        if (detection_.has_value()) {
            return "OptionalDetection(" + detection_->repr() + ")";
        }
        return "OptionalDetection(empty)";
    }
};

// Factory function simulating a detector that sometimes finds nothing
static OptionalDetection detect_in_frame(int frame_id) {
    // Simulate: detection every 3 frames
    if (frame_id % 3 == 0) {
        return OptionalDetection();  // no detection
    }
    double x = 100.0 + (frame_id % 50);
    double y = 200.0 + (frame_id % 30);
    return OptionalDetection(x, y, 64.0, 48.0);
}

// ============================================================================
// std::unique_ptr — Clear Single Ownership
// ============================================================================

class UniqueModel {
    std::unique_ptr<Model> model_;

public:
    UniqueModel(const std::string& name, int params)
        : model_(std::make_unique<Model>(name, params)) {}

    // Move constructor — ownership transfer
    UniqueModel(UniqueModel&& other) noexcept = default;
    UniqueModel& operator=(UniqueModel&& other) noexcept = default;

    // Non-copyable
    UniqueModel(const UniqueModel&) = delete;
    UniqueModel& operator=(const UniqueModel&) = delete;

    bool is_valid() const { return model_ != nullptr; }

    std::string info() const {
        if (!model_) {
            return "UniqueModel(moved-from, empty)";
        }
        return "UniqueModel(" + model_->info() + ")";
    }

    std::string name() const {
        if (!model_) {
            throw std::runtime_error("UniqueModel: accessing moved-from model");
        }
        return model_->name;
    }

    int param_count() const {
        if (!model_) {
            throw std::runtime_error("UniqueModel: accessing moved-from model");
        }
        return model_->param_count;
    }
};

// Demonstrate ownership transfer: creates a model and returns it
static UniqueModel create_model(const std::string& name, int params) {
    return UniqueModel(name, params);
}

// ============================================================================
// std::shared_ptr — Reference-Counted Ownership
// ============================================================================

class SharedBuffer {
    std::shared_ptr<Buffer> buffer_;

public:
    SharedBuffer(size_t size, const std::string& label)
        : buffer_(std::make_shared<Buffer>(size, label)) {}

    // Shared copy — both SharedBuffer instances point to the same Buffer
    SharedBuffer(const SharedBuffer&) = default;
    SharedBuffer& operator=(const SharedBuffer&) = default;

    SharedBuffer(SharedBuffer&&) noexcept = default;
    SharedBuffer& operator=(SharedBuffer&&) noexcept = default;

    long use_count() const { return buffer_.use_count(); }

    size_t size() const { return buffer_->size(); }

    std::string label() const { return buffer_->label; }

    std::string repr() const {
        return "SharedBuffer(label='" + buffer_->label +
               "', size=" + std::to_string(buffer_->size()) +
               ", use_count=" + std::to_string(buffer_.use_count()) + ")";
    }

    // Create a shared copy (increments reference count)
    SharedBuffer share() const {
        return *this;  // copy constructor increments use_count
    }
};

// ============================================================================
// nanobind bindings
// ============================================================================

NB_MODULE(memory_safety_demo, m) {
    m.doc() = "Lesson 11: RAII, std::optional, and smart pointers";

    // --- BBox ---
    nb::class_<BBox>(m, "BBox")
        .def(nb::init<double, double, double, double>(),
             nb::arg("x"), nb::arg("y"), nb::arg("w"), nb::arg("h"))
        .def("area", &BBox::area)
        .def_ro("x", &BBox::x)
        .def_ro("y", &BBox::y)
        .def_ro("w", &BBox::w)
        .def_ro("h", &BBox::h)
        .def("__repr__", &BBox::repr);

    // --- RAIIBuffer ---
    nb::class_<RAIIBuffer>(m, "RAIIBuffer")
        .def(nb::init<size_t>(), nb::arg("size"))
        .def("size", &RAIIBuffer::size)
        .def("get", &RAIIBuffer::get, nb::arg("index"))
        .def("set", &RAIIBuffer::set, nb::arg("index"), nb::arg("value"))
        .def_static("active_count", &RAIIBuffer::active_count)
        .def("__repr__", [](const RAIIBuffer& b) {
            return "RAIIBuffer(size=" + std::to_string(b.size()) + ")";
        });

    // --- OptionalDetection ---
    nb::class_<OptionalDetection>(m, "OptionalDetection")
        .def(nb::init<double, double, double, double>(),
             nb::arg("x"), nb::arg("y"), nb::arg("w"), nb::arg("h"))
        .def(nb::init<>())
        .def("has_value", &OptionalDetection::has_value)
        .def("value", &OptionalDetection::value)
        .def("value_or", &OptionalDetection::value_or,
             nb::arg("x"), nb::arg("y"), nb::arg("w"), nb::arg("h"))
        .def("area", &OptionalDetection::area)
        .def("__repr__", &OptionalDetection::repr);

    m.def("detect_in_frame", &detect_in_frame,
          nb::arg("frame_id"),
          "Simulate detection — returns empty OptionalDetection every 3rd frame.");

    // --- UniqueModel ---
    nb::class_<UniqueModel>(m, "UniqueModel")
        .def(nb::init<const std::string&, int>(),
             nb::arg("name"), nb::arg("params"))
        .def("is_valid", &UniqueModel::is_valid)
        .def("info", &UniqueModel::info)
        .def("name", &UniqueModel::name)
        .def("param_count", &UniqueModel::param_count)
        .def("__repr__", &UniqueModel::info);

    m.def("create_model", &create_model,
          nb::arg("name"), nb::arg("params"),
          "Create a UniqueModel (demonstrates ownership transfer via return).");

    // --- SharedBuffer ---
    nb::class_<SharedBuffer>(m, "SharedBuffer")
        .def(nb::init<size_t, const std::string&>(),
             nb::arg("size"), nb::arg("label"))
        .def("use_count", &SharedBuffer::use_count)
        .def("size", &SharedBuffer::size)
        .def("label", &SharedBuffer::label)
        .def("share", &SharedBuffer::share)
        .def("__repr__", &SharedBuffer::repr);
}
