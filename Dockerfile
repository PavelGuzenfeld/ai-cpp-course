FROM ubuntu:22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get upgrade -y && \
    apt-get install -y tzdata && \
    echo "Etc/UTC" > /etc/timezone && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get install -y \
    libmpfr-dev libgmp3-dev libmpc-dev git-lfs net-tools software-properties-common build-essential libtbb-dev \
    libboost-all-dev cmake git python3 python3-pip wget libjsoncpp-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev python-is-python3 autoconf autoconf-archive automake libtool python3-dev python3-dbg \
    python-gi-dev libjson-glib-dev nlohmann-json3-dev python3-gi libgirepository1.0-dev gir1.2-gtk-3.0 libcairo2-dev \
    python3-gi-cairo python3-gst-1.0 gstreamer1.0-python3-plugin-loader gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-rtsp \
    libgstrtspserver-1.0-dev gdb valgrind mold ccache libopencv-dev pybind11-dev \
    && rm -rf var/lib/apt/lists/*

RUN pip3 install meson ninja numpy opencv-python pyudev termcolor scipy pyulog pymavlink asyncio pydantic \
    pybind11 colcon-core colcon-common-extensions setuptools==58.2.0 empy==3.3.4 \
    && rm -rf /root/.cache/pip

RUN echo "alias ll='ls -l'" >> ~/.bashrc \
    && echo "alias lt='ls -ltr'" >> ~/.bashrc

WORKDIR /workspace

CMD [ "bash" ]