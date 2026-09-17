FROM ubuntu:22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get upgrade -y && \
    apt-get install -y tzdata && \
    echo "Etc/UTC" > /etc/timezone && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get install -y \
    libfmt-dev \
    git-lfs net-tools software-properties-common build-essential libtbb-dev \
    cmake git python3 python3-pip wget libjsoncpp-dev \
    python-is-python3 autoconf autoconf-archive automake libtool python3-dev \
    libjson-glib-dev nlohmann-json3-dev \
    gdb mold ccache libopencv-dev pybind11-dev \
    && rm -rf var/lib/apt/lists/*

RUN pip3 install meson ninja numpy opencv-python pyudev termcolor scipy pyulog pymavlink asyncio pydantic \
    pybind11 colcon-core colcon-common-extensions setuptools==58.2.0 empy==3.3.4 \
    pytest pytest-rerunfailures nanobind \
    && rm -rf /root/.cache/pip

RUN echo "alias ll='ls -l'" >> ~/.bashrc \
    && echo "alias lt='ls -ltr'" >> ~/.bashrc

RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y \
    && apt-get install -y g++-13 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 60 --slave /usr/bin/g++ g++ /usr/bin/g++-13 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 60 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 60

RUN git clone https://github.com/rui314/mold.git \
    && cd mold \
    && cmake -Bbuild -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j$(nproc) \
    && cmake --install build \
    && cd .. \
    && rm -rf mold

# -O a file, not a pipe: sh has no pipefail, so `wget | tee` reported tee's
# status and a dropped fetch surfaced as an unsigned-repo error four commands on.
RUN wget -q --tries=3 --retry-connrefused -O /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
      https://apt.llvm.org/llvm-snapshot.gpg.key \
    && gpg --show-keys /etc/apt/trusted.gpg.d/apt.llvm.org.asc > /dev/null \
    && echo "deb https://apt.llvm.org/jammy/ llvm-toolchain-jammy-19 main" > /etc/apt/sources.list.d/llvm.list \
    # llvm.sh's own distro-version check rejects point releases like
    # 22.04.5 that postdate the script; add the known-good jammy repo directly.
    && apt-get update \
    # clang-tidy ships separately from clang; L12 is built around it. It pulls
    # libclang-cpp19 and libllvm19, which clang-19 already brings: +55 MB.
    && apt-get install -y clang-19 clang-tidy-19 \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 100 \
    && rm -rf /var/lib/apt/lists/*

# Install xsimd  - generic SIMD types and functions for C++
RUN git clone https://github.com/xtensor-stack/xsimd \
    && cd xsimd \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install \
    && cd ../.. \
    && rm -rf xsimd

# Install sml - C++14 State Machine Library
RUN git clone https://github.com/boost-experimental/sml.git && \
    mkdir sml/build && \
    cd sml/build && \
    cmake .. && \
    make install && \
    cd ../.. && \
    rm -rf sml

# Install concurrentqueue - A fast multi-producer, multi-consumer lock-free concurrent queue for C++
RUN git clone https://github.com/cameron314/concurrentqueue && \
    mkdir concurrentqueue/cmake_build && \
    cd concurrentqueue/cmake_build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && \
    rm -rf concurrentqueue/

# Install GSL - Microsoft's Guidelines Support Library
RUN git clone https://github.com/microsoft/GSL.git \
    && mkdir -p GSL/cmake_build \
    && cd GSL/cmake_build \
    && cmake .. \
    && make -j$(nproc) \
    && make install \
    && cd ../.. \
    && rm -rf GSL

# Install Enoki - structured vectorization and differentiation on modern processor architectures
RUN git clone --recurse-submodules https://github.com/mitsuba-renderer/enoki.git \
    && cd enoki \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && cp -r ../include /usr/local/include/enoki \
    && cd ../.. \
    && rm -rf enoki

# Install nanobind - Faster bind to Python for C++
RUN git clone --recurse-submodules https://github.com/wjakob/nanobind.git \
    && cd nanobind \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DNB_CREATE_INSTALL_RULES=ON -DCMAKE_INSTALL_PREFIX=/usr/local .. \
    && make -j$(nproc) \
    && make install \
    && cd ../.. \
    && rm -rf nanobind \
    && ln -s /usr/local/nanobind/cmake /usr/local/lib/cmake/nanobind \
    && ln -s /usr/local/nanobind/include/nanobind /usr/local/include/nanobind 

WORKDIR /workspace

CMD [ "bash" ]