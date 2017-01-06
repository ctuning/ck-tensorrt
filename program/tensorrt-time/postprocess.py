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

    # Load tensorrt-time profiling output.
    # TODO: Read from program meta run_vars['CK_TENSORRT_CJSON_PATH'].
    r=ck.load_json_file({'json_file':'profiler.json'})
    if r['return']>0: return r

    # Update layer info similarly to Caffe output.
    d['per_layer_info'] = r['dict']['per_layer_info']
    time_fw_ms = 0.0
    for layer_info in d['per_layer_info']:
        time_fw_ms += layer_info['time_ms']
        # Update optional keys for compatibility with CK-Caffe.
        layer_info['time_s'] = layer_info['time_ms'] * 1e-3
        layer_info['label'] = '%02d: %s' % (layer_info['index'], layer_info['name'])
        layer_info['timestamp'] = '0101 00:00:00.000000' # FIXME: Add proper timestamp.
        layer_info['direction'] = 'forward'

    # Execution time (ms).
    d['time_fw_ms'] = time_fw_ms
    d['time_bw_ms'] = 0.0
    d['time_fwbw_ms'] = d['time_fw_ms'] + d['time_bw_ms']
    d['time_total_ms'] = d['time_fwbw_ms']
    d['time_total_ms_kernel_0'] = d['time_total_ms']
    # Execution time (s).
    d['time_fw_s'] = d['time_fw_ms'] * 1e-3
    d['time_bw_s'] = d['time_bw_ms'] * 1e-3
    d['time_fwbw_s'] = d['time_fwbw_ms'] * 1e-3
    d['time_total_s'] = d['time_total_ms'] * 1e-3
    d['time_total_s_kernel_0'] = d['time_total_ms_kernel_0'] * 1e-3

    # FIXME: Add memory consumption.
    memory_bytes = 0
    d['memory_bytes']  = memory_bytes
    d['memory_kbytes'] = memory_bytes * 1e-3
    d['memory_mbytes'] = memory_bytes * 1e-6

    # Built-in CK keys.
    d['execution_time'] = d['time_total_s']
    d['post_processed'] = 'yes'

    rr={}
    rr['return']=0
    if d.get('post_processed','')=='yes':
        r=ck.save_json_to_file({'json_file':'results.json', 'dict':d})
        if r['return']>0: return r
    else:
        rr['error']='failed to match best prediction in tensorrt-time output!'
        rr['return']=1

    return rr

# Do not add anything here!
