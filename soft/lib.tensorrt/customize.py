#
# Collective Knowledge (individual environment - setup)
#
# See CK LICENSE.txt for licensing details
# See CK COPYRIGHT.txt for copyright details
#
# Developer: Grigori Fursin, Grigori.Fursin@cTuning.org, http://fursin.net
#

import os

##############################################################################
# get version from path

def version_cmd(i):
    libinfer_so=i['full_path']
    # On TX1, the full path is "/usr/lib/aarch64-linux-gnu/libnvinfer.so".
    # Otherwise, the path is "<ROOT>/lib/libnvinfer.so".
    libinfer_dir=os.path.dirname(libinfer_so)
    if os.path.basename(libinfer_dir)=='lib':
        lib_dir=libinfer_dir
        arch_os_name=''
    else:
        lib_dir=os.path.dirname(libinfer_dir)
        arch_os_name=os.path.basename(libinfer_dir)
    root_dir=os.path.dirname(lib_dir)

    # Undetected version: 0.0.0
    major='0'; minor='0'; patch='0'

    nvinferversion_h=os.path.join(root_dir, 'include', arch_os_name, 'NvInferVersion.h')
    nvinfer_h=os.path.join(root_dir, 'include', arch_os_name, 'NvInfer.h')
    version_file_path = None
    if os.path.exists(nvinferversion_h):
        # TensorRT v5-6.
        version_file_path = nvinferversion_h
    elif os.path.exists(nvinfer_h):
        # TensorRT v1-5 (?).
        version_file_path = nvinfer_h
    if version_file_path:
        with open(version_file_path, 'r') as version_file:
            lines=version_file.readlines()
            for line in lines:
                if line.startswith('#define NV_TENSORRT_MAJOR'): major=line.split()[2]
                if line.startswith('#define NV_TENSORRT_MINOR'): minor=line.split()[2]
                if line.startswith('#define NV_TENSORRT_PATCH'): patch=line.split()[2]

    version='%s.%s.%s' % (major,minor,patch)
    return {'return':0, 'cmd':'', 'version':version}

##############################################################################
# setup environment setup

def setup(i):
    """
    Input:  {
              cfg              - meta of this soft entry
              self_cfg         - meta of module soft
              ck_kernel        - import CK kernel module (to reuse functions)

              host_os_uoa      - host OS UOA
              host_os_uid      - host OS UID
              host_os_dict     - host OS meta

              target_os_uoa    - target OS UOA
              target_os_uid    - target OS UID
              target_os_dict   - target OS meta

              target_device_id - target device ID (if via ADB)

              tags             - list of tags used to search this entry

              env              - updated environment vars from meta
              customize        - updated customize vars from meta

              deps             - resolved dependencies for this soft

              interactive      - if 'yes', can ask questions, otherwise quiet
            }

    Output: {
              return       - return code =  0, if successful
                                         >  0, if error
              (error)      - error text if return > 0

              bat          - prepared string for bat file
            }

    """

    # Get variables
    ck=i['ck_kernel']
    s=''

    iv=i.get('interactive','')

    cus=i.get('customize',{})

    hosd=i['host_os_dict']
    tosd=i['target_os_dict']

    # Check platform
    hplat=hosd.get('ck_name','')

    hproc=hosd.get('processor','')
    tproc=tosd.get('processor','')

    remote=tosd.get('remote','')
    tbits=tosd.get('bits','')

    # Paths.
    fp=cus.get('full_path','')
    path_lib=os.path.dirname(fp)
    if not os.path.isdir(path_lib):
        return {'return':1, 'error':'can\'t find installation lib dir'}

    path_include=path_lib.replace('lib','include')
    if not os.path.isdir(path_include):
        return {'return':1, 'error':'can\'t find installation include dir'}

    path_bin=path_lib.replace('lib', 'bin')
    env=i['env']

    ep=cus['env_prefix']
    env[ep]=path_lib

    ############################################################
    # Setting environment depending on the platform
    if hplat=='win':
       # TBD
       return {'return':1, 'error':'not yet supported ...'}

    cus['dynamic_lib']=os.path.basename(fp)
    env[ep+'_DYNAMIC_NAME']=cus.get('dynamic_lib','')

    cus['path_lib']=path_lib
    cus['path_include']=path_include
    cus['path_bin']=path_bin

    r = ck.access({'action': 'lib_path_export_script', 'module_uoa': 'os', 'host_os_dict': hosd,
      'lib_path': cus.get('path_lib','')})
    if r['return']>0: return r
    s += r['script']
    s += 'PATH={}:$PATH\n\n'.format(path_bin)

    return {'return':0, 'bat':s}
