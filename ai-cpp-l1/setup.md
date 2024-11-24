# **Setup Instructions for C++ Development Environment**

Follow the steps below to set up a complete development environment with essential tools, libraries, and dependencies.

---

## **Step 1: Install Basic Dependencies**
Install commonly required tools and libraries:
```bash
sudo apt-get install -y software-properties-common build-essential libboost-all-dev cmake git python3 python3-pip wget libjsoncpp-dev autoconf autoconf-archive automake libtool gdb valgrind mold ccache pybind11-dev
```

---

## **Step 2: Install Python Packages**
Install necessary Python packages:
```bash
pip3 install meson ninja numpy pyudev pybind11 colcon-core colcon-common-extensions setuptools==58.2.0 empy==3.3.4
sudo rm -rf /root/.cache/pip
```

---

## **Step 3: Install Microsoft GSL**
Clone and build Microsoft’s Guideline Support Library:
```bash
git clone https://github.com/microsoft/GSL.git
mkdir -p GSL/cmake_build
cd GSL/cmake_build
cmake ..
sudo make install
cd ../..
rm -rf GSL
```

---

## **Step 4: Install GCC 13**
Add the GCC 13 repository and set it as the default compiler:
```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get install -y g++-13
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 60 --slave /usr/bin/g++ g++ /usr/bin/g++-13
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*
```

---

## **Step 5: Install Clang 19**
Download and install Clang 19:
```bash
wget https://apt.llvm.org/llvm.sh
sudo chmod u+x llvm.sh
sudo ./llvm.sh 19
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 100
sudo rm llvm.sh
```

---

## **Final Verification**
Verify that the necessary tools are installed and accessible:
```bash
gcc --version
g++ --version
clang --version
clang++ --version
cmake --version
python3 --version
pip3 --version
```