#
# Convert raw output of the TensorRT imagenet-console
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
    deps=i['deps']

    d={}

    env=i.get('env',{})

    # Load stdout as list.
    r=ck.load_text_file({'text_file':'stdout.log','split_to_list':'yes'})
    if r['return']>0: return r

    d['REAL_ENV_CK_CAFFE_MODEL']=env.get('CK_CAFFE_MODEL','')
    d['all_predictions']=[]

    for line in r['lst']:
        # Match prediction info in e.g.:
        # "class 0287 - 0.049164  (lynx, catamount)"
        prediction_regex = \
            'class(\s+)(?P<class>\d{4})' + \
            '(\s+)-(\s+)' + \
            '(?P<probability>0.\d{6})' + \
            '(\s+)' + \
            '\((?P<synset>[\w\s,]*)\)'
        match = re.search(prediction_regex, line)
        if match:
            info = {}
            info['class'] = int(match.group('class'))
            info['probability'] = float(match.group('probability'))
            info['synset'] = match.group('synset')
            d['all_predictions'].append(info)

        # Match the most likely prediction in e.g.:
        # "imagenet-console: '<file path>' -> 33.05664% class #331 (hare)"
        best_prediction_regex = \
             'imagenet-console:(\s+)' + \
             '\'(?P<path>[\.\w/_-]*)\'' + \
             '(\s)*->(\s)*' + \
             '(?P<probability_pc>\d+\.\d*)%' + \
             '(\s)*class(\s)*#(?P<class>\d+)(\s*)' + \
             '\((?P<synset>[\w\s,]*)\)'            
        match = re.search(best_prediction_regex, line)
        if match:
            info = {}
            info['path'] = match.group('path')
            info['probability'] = float(match.group('probability_pc'))*0.01
            info['class'] = int(match.group('class'))
            info['synset'] = match.group('synset')
            d['best_prediction'] = info
            d['post_processed'] = 'yes'
            d['execution_time'] = 0.0 # built-in CK key

    rr={}
    rr['return']=0
    if d.get('post_processed','')=='yes':
       # Save to file.
       r=ck.save_json_to_file({'json_file':'results.json', 'dict':d})
       if r['return']>0: return r
    else:
       rr['return']=1
       rr['error']='failed to match best prediction in imagenet-console output!'

    return rr

# Do not add anything here!
