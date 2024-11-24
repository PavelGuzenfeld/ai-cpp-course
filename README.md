# ai-cpp-course
CPP course intended for AI devs

## Building the docker image
```bash
    cd ai-cpp-course/
    docker build -t ai-cpp-course  -f ai-cpp-course/Dockerfile .
```

## Running the docker container
```bash
    docker run -it -v $(pwd):/workspace ai-cpp-course
```

## Building the lessons inside the container
```bash
    cd /workspace/
    colcon build
    source install/setup.bash
```
