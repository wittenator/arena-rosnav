#!/bin/bash

case $(lsb_release -sc) in
  focal)
    ROS_NAME_VERSION=noetic
    ;;
  
  bionic)
    ROS_NAME_VERSION=melodic
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

sudo apt-get install aptitude
sudo aptitude install curl git

mkdir -p catkin_ws/src && cd catkin_ws/src
git clone --depth 1 --branch MARL_noetic https://github.com/wittenator/arena-rosnav.git
cd arena-rosnav

sudo add-apt-repository universe
sudo add-apt-repository multiverse
sudo add-apt-repository restricted
sudo apt update

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

sudo aptitude update
sudo aptitude -y install ros-${ROS_NAME_VERSION}-desktop-full

echo "source /opt/ros/${ROS_NAME_VERSION}/setup.${CURSHELL}" >> ~/.${CURSHELL}rc
source ~/.${CURSHELL}rc

if [ $ROS_NAME_VERSION = "noetic" ]; then
  sudo aptitude -y install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
else
  sudo aptitude -y install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
fi


sudo rosdep init
rosdep update

sudo aptitude update && sudo aptitude -y install \
libopencv-dev \
liblua5.2-dev \
screen \
python3-rospkg-modules \
ros-${ROS_NAME_VERSION}-navigation \
ros-${ROS_NAME_VERSION}-teb-local-planner \
ros-${ROS_NAME_VERSION}-mpc-local-planner \
libarmadillo-dev \
ros-${ROS_NAME_VERSION}-nlopt \

poetry install

if [ $ROS_NAME_VERSION = "noetic" ]; then
  sudo aptitude -y install ros-noetic-geometry2
else
  poetry run PYTHON3_EXEC="$(which python3)"
  poetry run PYTHON3_INCLUDE="$(ls -d /usr/include/* | grep  python | sort -r| head -1)"
  poetry run PYTHON3_DLIB="$(ls -d /usr/lib/x86_64-linux-gnu/* | grep -P  "libpython3\S*.so"| sort | head -1)"
  if [ -z $PYTHON3_DLIB ] || [ -z $PYTHON3_INCLUDE ] || [ -z $PYTHON3_EXEC ] ; then
      echo "Can't find python library please install it with \" sudo apt-get python3-dev \" !" >&2
  fi

  PARENT_DIR_WS="$(cd "$(dirname $0)/../../.." >/dev/null 2>&1 && pwd)"
  mkdir -p ${PARENT_DIR_WS}/geometry2_ws/src && cd "$_"
  git clone --depth=1 https://github.com/ros/geometry2.git
  cd ..

  # compile geometry2 with python3 
  echo -n "compiling geometry2 with python3 ..."
  catkin_make -DPYTHON_EXECUTABLE=${PYTHON3_EXEC} -DPYTHON_INCLUDE_DIR=${PYTHON3_INCLUDE} -DPYTHON_LIBRARY=${PYTHON3_DLIB} > /dev/null 2>&1


  # add the lib path to the python environment 
  if [ $? -eq 0 ] ; then
      echo " done!"
      package_path="$(cd devel/lib/python3/dist-packages && pwd)"
      rc_info="export PYTHONPATH=${package_path}:\${PYTHONPATH}\n"
      if echo $SHELL | grep zsh > /dev/null
      then
          echo -e "$rc_info" >> ~/.zshrc
          echo "PYTHONPATH has been updated in your zshrc file."
      elif echo $SHELL | grep bash > /dev/null
      then
          echo -e "$rc_info" >> ~/.bashrc
          echo "PYTHONPATH has been updated in your bashrc file."
      else
          echo "Can't not determin which terminal you are using. Please manualy add the package path ${package_path} to you bashrc or zshrc file later"
      fi
  else
      echo "Fail to compile geometry2"
  fi
fi

rosws update

echo "export PYTHONPATH=${PWD}:${PWD}/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl:\$PYTHONPATH" >> ~/.${CURSHELL}rc

source ~/.${CURSHELL}rc
poetry run catkin_make -C ../.. -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=$(poetry env info -p)/bin/python3

echo "source $(readlink -f ${PWD}/../../devel/setup.sh)" >> ~/.${CURSHELL}rc
source ~/.${CURSHELL}rc
