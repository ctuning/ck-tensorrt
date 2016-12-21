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
    env=i.get('env',{})

    d={}

    # Collect env vars of interest.
    d['REAL_ENV_CK_CAFFE_MODEL']=env.get('CK_CAFFE_MODEL','')
    # FIXME: Is not set?
    d['REAL_ENV_CK_CAFFE_IMAGENET_VAL_TXT']=env.get('CK_CAFFE_IMAGENET_VAL_TXT','')

    # Load ImageNet validation set labels.
#    image_to_synset_map = {}
#    with open(d['REAL_ENV_CK_CAFFE_IMAGENET_VAL_TXT']) as imagenet_val_txt:
#        for image_synset in imagenet_val_txt:
#            (image, synset) = image_synset.split()
#            image_to_synset_map[image] = synset

    # Load imagenet-console output as list.
    r=ck.load_text_file({'text_file':'stdout.log', 'split_to_list':'yes'})
    if r['return']>0: return r

    # Collect info about all and best predictions from imagenet-console output.
    d['all_predictions']=[]
    for line in r['lst']:
        # Match prediction info in e.g.:
        # "class 0287 - 0.049164  (lynx, catamount)"
        # "class 0673 - 1.000000  (mouse, computer mouse)" (yes, with GoogleNet!)
        prediction_regex = \
            'class(\s+)(?P<class>\d{4})' + \
            '(\s+)-(\s+)' + \
            '(?P<probability>\d+.\d+)' + \
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
            '\'(?P<file_path>[\.\w/_-]*)\'' + \
            '(\s)*->(\s)*' + \
            '(?P<probability_pc>\d+\.\d+)%' + \
            '(\s)*class(\s)*#(?P<class>\d+)(\s*)' + \
            '\((?P<synset>[\w\s,]*)\)'
        match = re.search(best_prediction_regex, line)
        if match:
            info = {}
            info['file_path'] = match.group('file_path')
            info['file_name'] = os.path.basename(info['file_path'])
            info['probability'] = float(match.group('probability_pc'))*0.01
            info['class'] = int(match.group('class'))
            info['synset'] = match.group('synset')
            d['best_prediction'] = info
            d['post_processed'] = 'yes'
            d['execution_time'] = 0.0 # built-in CK key

    d['all_predictions'] = sorted(d['all_predictions'], key=lambda k: k['probability'], reverse=True)
    # FIXME: A placeholder for things to come (see below).
    d['best_prediction']['is top?'] = 'yes' if d['all_predictions'][0]['class']==d['best_prediction']['class'] else 'no'
    # TODO: Check against the label assigned to this image (once we know what that is).
    # TODO: Set 'accuracy_top1' and 'accuracy_top5' to 'yes' or 'no' accordingly.

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
