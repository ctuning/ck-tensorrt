#! /bin/bash

#
# Installation script for jetson-inference.
#
# See CK LICENSE.txt for licensing details.
# See CK COPYRIGHT.txt for copyright details.
#
# Developer(s):
# - Daniil Efremov, daniil.efremov@xored.com, 2016
# - Anton Lokhmotov, anton@dividiti.com, 2016
#

# PACKAGE_DIR
# INSTALL_DIR

export JETSON_PKG_DIR=${PACKAGE_DIR}
export JETSON_SRC_DIR=${INSTALL_DIR}/src
export JETSON_BLD_DIR=${INSTALL_DIR}/bld

#export CUDA_CUDART_LIBRARY=/usr/local/cuda
#export CUDA_TOOLKIT_INCLUDE=/usr/local/cuda/include
#export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
#export PATH=$PATH:/usr/local/cuda/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib:/usr/local/lib
#export CPLUS_INCLUDE_PATH=/usr/local/cuda/include

export CK_ENV_COMPILER_CUDA=/usr/local/cuda

################################################################################
echo "Cleaning dir '${INSTALL_DIR}'"
rm -rf ${INSTALL_DIR}/*

################################################################################
echo "Creating dir '${JETSON_SRC_DIR}'"
mkdir ${JETSON_SRC_DIR}

echo "Creating dir '${JETSON_BLD_DIR}'"
mkdir ${JETSON_BLD_DIR}

################################################################################
echo ""
echo "Cloning jetson-inference from '${JETSON_URL}' ..."

echo "git clone ${JETSON_URL} --no-checkout"
git clone ${JETSON_URL} --no-checkout ${JETSON_SRC_DIR}
if [ "${?}" != "0" ] ; then
  echo "Error: Cloning jetson-inference from '${JETSON_URL}' failed!"
  exit 1
fi

################################################################################
echo ""
echo "Checking out the '${JETSON_BRANCH}' branch of jetson-inference ..."

cd ${JETSON_SRC_DIR}
git checkout ${JETSON_BRANCH}
if [ "${?}" != "0" ] ; then
  echo "Error: Checking out the '${JETSON_BRANCH}' branch of jetson-inference failed!"
  exit 1
fi

################################################################################
echo ""
echo "Configuring jetson-inference in '${JETSON_BLD_DIR}' ..."

cp ${ORIGINAL_PACKAGE_DIR}/CMakeLists.txt ${JETSON_SRC_DIR}/CMakeLists.txt

#  -DBUILD_DEPS=NO # YES - apt update/install, download nets, etc.

cd ${JETSON_BLD_DIR}
cmake ${JETSON_SRC_DIR} \
  -DCMAKE_BUILD_TYPE=${CK_ENV_CMAKE_BUILD_TYPE:-Release} \
  -DCUDA_TOOLKIT_ROOT_DIR="${CK_ENV_COMPILER_CUDA}" \
  -DCMAKE_CXX_COMPILER="${CK_CXX}" \
  -DCMAKE_C_COMPILER="${CK_CC}" \
  -DBUILD_DEPS=NO \
  -DGIE_PATH=${CK_ENV_LIB_TENSORRT_INCLUDE}/../ \
  -DNV_TENSORRT_MAJOR=2  

if [ "${?}" != "0" ] ; then
  echo "Error: Configuring jetson-inference failed!"
  exit 1
fi

################################################################################
echo ""
echo "Building jetson-inference in '${JETSON_BLD_DIR}' ..."

cd ${JETSON_BLD_DIR}
make imagenet-console
if [ "${?}" != "0" ] ; then
  echo "Error: Building jetson-inference failed!"
  exit 1
fi

################################################################################
echo ""
echo "Installing jetson-inference in '${INSTALL_DIR}' ..."

cp -r ${JETSON_BLD_DIR}/$(uname -m)/* ${INSTALL_DIR}
if [ "${?}" != "0" ] ; then
  echo "Error: Installing jetson-inference failed!"
  exit 1
fi

################################################################################
echo ""
echo "Installed jetson-inference in '${INSTALL_DIR}'."
exit 0
