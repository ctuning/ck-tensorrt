#
# Convert raw output of the tensorrt-time
# program to the CK format.
#
# Developers:
#   - Anton Lokhmotov, dividiti, 2016
#

import json
import os
import re
import sys

def ck_postprocess(i):
    ck=i['ck_kernel']
    rt=i['run_time']
    env=i.get('env',{})
    deps=i.get('deps',{})

    d={}

    # Collect env vars of interest.
    d['REAL_ENV_CK_CAFFE_MODEL']=env.get('CK_CAFFE_MODEL','')

    # Load tensorrt-time cJSON output written to stderr.
    r=ck.load_json_file({'json_file':rt['run_cmd_out2']})
    if r['return']>0: return r

    # Update layer info similarly to Caffe output.
    d['per_layer_info'] = r['dict']['per_layer_info']
    for layer_info in d['per_layer_info']:
        layer_info['direction'] = 'forward'
        layer_info['time_s'] = layer_info['time_ms'] * 1e-3
        layer_info['label'] = '%02d: %s' % (layer_info['index'], layer_info['name'])
        # TODO: timestamp.
    
    d['post_processed'] = 'yes'
    d['execution_time'] = 0.0 # built-in CK key

    rr={}
    rr['return']=0
    if d.get('post_processed','')=='yes':
        r=ck.save_json_to_file({'json_file':'results.json', 'dict':d})
        if r['return']>0: return r
    else:
        rr['error']='failed to match best prediction in imagenet-console output!'
        rr['return']=1

    return rr

# Do not add anything here!
