#! /bin/bash

#
# Installation script for jetson-inference.
#
# See CK LICENSE.txt for licensing details.
# See CK COPYRIGHT.txt for copyright details.
#
# Developer(s):
# - Anton Lokhmotov, anton@dividiti.com, 2016
# - Grigori Fursin, grigori@dividiti.com, 2016

# PACKAGE_DIR
# INSTALL_DIR

export JETSON_PKG_DIR=${PACKAGE_DIR}
export JETSON_SRC_DIR=${INSTALL_DIR}
export JETSON_BLD_DIR=${JETSON_SRC_DIR}

################################################################################
echo ""
echo "Cloning jetson-inference from '${JETSON_URL}' to '${JETSON_SRC_DIR}'"

echo "Removing dir '${JETSON_SRC_DIR}'"
rm -rf ${JETSON_SRC_DIR}
echo "after removing "
ls -l ${JETSON_SRC_DIR}

mkdir ${JETSON_SRC_DIR}
cd ${JETSON_SRC_DIR}

#echo "Creating dir '${JETSON_SRC_DIR}'"
#mkdir ${JETSON_SRC_DIR}

echo "git clone ${JETSON_URL} --no-checkout"
git clone ${JETSON_URL} --no-checkout .

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
echo "Building jetson-inference in '${JETSON_BLD_DIR}' ..."

mkdir build
cd build
cmake ../
make

