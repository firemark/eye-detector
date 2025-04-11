ARG ROS_DISTRO=humble
 
FROM ros:${ROS_DISTRO} AS base
ENV ROS_DISTRO=${ROS_DISTRO}
SHELL ["/bin/bash", "-c"]

RUN apt-get update -y && apt-get install -y \
  libboost-python-dev \
  python3-numpy \
  python3-opencv \
  python3-skimage \
  python3-scipy \
  python3-pip \
  libdlib-data \
  libdlib-dev \
  ros-${ROS_DISTRO}-rviz2 \
  ros-${ROS_DISTRO}-plotjuggler \
  ros-${ROS_DISTRO}-realsense2-*

RUN python3 -m pip install \
  dlib \
  torch \
  torchvision \
  pyrealsense2
 
# Create Colcon workspace with external dependencies
RUN mkdir -p /ws/src
WORKDIR /ws/src
COPY ros/dependencies.repos .
RUN vcs import < dependencies.repos
 
# Build the base Colcon workspace, installing dependencies first.
WORKDIR /ws
RUN rosdep install --from-paths src --ignore-src --rosdistro ${ROS_DISTRO} -y
RUN colcon build --symlink-install

RUN mkdir -p /ws/src_py
COPY pyproject.toml /ws/src_py/pyproject.toml
COPY LICENSE /ws/src_py/LICENSE 
COPY README.md /ws/src_py/README.md
COPY src /ws/src_py/src
RUN pip install -e /ws/src_py