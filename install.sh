#!/bin/bash

case $(lsb_release -sc) in
  focal)
    ROS_VERSION=noetic
    ;;
  
  bionic)
    ROS_VERSION=melodic
    ;;

  *)
    echo "Currently only Ubuntu Bionic Beaver and Focal Fossa are supported for an automatic install. Please refer to the manual installation if you use any Linux release or version."
    exit 1
    ;;
esac

if test -n "$ZSH_VERSION"; then
  CURSHELL=zsh
elif test -n "$BASH_VERSION"; then
  CURSHELL=bash
else
  echo "Currently only Bash and ZSH are supported for an automatic install. Please refer to the manual installation if you use any other shell."
  exit 1
fi

mkdir -p catkin_ws/src && cd catkin_ws/src
git clone --depth 1 --branch MARL_noetic https://github.com/wittenator/arena-rosnav.git
cd arena-rosnav

sudo add-apt-repository universe
sudo add-apt-repository multiverse
sudo add-apt-repository restricted
sudo apt update

sudo apt-get install aptitude

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

sudo aptitude update
sudo aptitude -y install ros-${ROS_VERSION}-desktop-full

echo "source /opt/ros/${ROS_VERSION}/setup.${CURSHELL}" >> ~/.${CURSHELL}rc
source ~/.${CURSHELL}rc

sudo aptitude -y install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
sudo rosdep init
rosdep update

sudo aptitude update && sudo aptitude -y install \
libopencv-dev \
liblua5.2-dev \
screen \
python3-rospkg-modules \
ros-${ROS_VERSION}-navigation \
ros-${ROS_VERSION}-teb-local-planner \
ros-${ROS_VERSION}-mpc-local-planner \
libarmadillo-dev \
ros-${ROS_VERSION}-nlopt \

poetry install

if [ $ROS_VERSION = "noetic" ]; then
  sudo aptitude -y install ros-noetic-geometry2
else
  poetry run . ./geometry2_install.sh
fi

rosws update

echo "export PYTHONPATH=${PWD}:${PWD}/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl:\$PYTHONPATH" >> ~/.${CURSHELL}rc

source ~/.${CURSHELL}rc
poetry run catkin_make -C ../.. -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=${poetry env info -p}

echo "source $(readlink -f ${PWD}/../../devel/setup.sh)" >> ~/.${CURSHELL}rc
source ~/.${CURSHELL}rc
