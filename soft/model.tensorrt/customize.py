#
# Collective Knowledge (individual environment - setup)
#
# See CK LICENSE.txt for licensing details
# See CK COPYRIGHT.txt for copyright details
#
# Developer: Leo Gordon, dividiti
#

import os

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

    import os

    ck              = i['ck_kernel']
    cus             = i.get('customize',{})
    full_path       = cus.get('full_path','')
    env             = i['env']
    install_root    = os.path.dirname(full_path)
    install_env     = cus.get('install_env', {})
    env_prefix      = cus['env_prefix']

    env[env_prefix + '_ROOT'] = install_root
    env[env_prefix + '_FILENAME'] = full_path

    # This group should end with _FILE prefix e.g. FLATLABELS_FILE
    # This suffix will be cut off and prefixed by cus['env_prefix']
    # so we'll get vars like CK_ENV_TENSORRT_MODEL_FLATLABELS_FILE
    for varname in install_env.keys():
        if varname.endswith('_FILE'):
            file_path = os.path.join(install_root, install_env[varname])
            if os.path.exists(file_path):
                env[env_prefix + '_' + varname] = file_path

    # Just copy those without any change in the name:
    #
    for varname in install_env.keys():
        if varname.startswith('ML_MODEL_'):
            env[varname] = install_env[varname]
    
    return {'return':0, 'bat':''}
