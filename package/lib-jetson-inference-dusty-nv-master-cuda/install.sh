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

################################################################################
echo "Removing dir '${JETSON_SRC_DIR}'"
rm -rf ${JETSON_SRC_DIR}

echo "Removing dir '${JETSON_BLD_DIR}'"
rm -rf ${JETSON_BLD_DIR}

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

cd ${JETSON_BLD_DIR}
cmake ${JETSON_SRC_DIR} \
  -DCMAKE_BUILD_TYPE=${CK_ENV_CMAKE_BUILD_TYPE:-Release} \
  -DCUDA_TOOLKIT_ROOT_DIR="${CK_ENV_COMPILER_CUDA}" \
  -DCMAKE_CXX_COMPILER="${CK_CXX}" \
  -DCMAKE_C_COMPILER="${CK_CC}" \
  -DBUILD_DEPS=NO # YES - apt update/install, download nets, etc.

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
