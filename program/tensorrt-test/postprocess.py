#
# Convert raw output of the TensorRT tensorrt-test
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
    deps=i.get('deps',{})

    d={}

    # Collect env vars of interest.
    d['REAL_ENV_CK_CAFFE_MODEL']=env.get('CK_CAFFE_MODEL','')
    # FIXME: Is not set? (Hence, collecting the same var via the deps below.)
    d['REAL_ENV_CK_CAFFE_IMAGENET_VAL_TXT']=env.get('CK_CAFFE_IMAGENET_VAL_TXT','')

    # Collect deps of interest.
    imagenet_aux=deps.get('dataset-imagenet-aux',{})
    imagenet_aux_dict=imagenet_aux.get('dict',{})
    imagenet_aux_dict_env=imagenet_aux_dict.get('env',{})
    d['CK_CAFFE_IMAGENET_VAL_TXT']=imagenet_aux_dict_env.get('CK_CAFFE_IMAGENET_VAL_TXT','')
    d['CK_CAFFE_IMAGENET_SYNSET_WORDS_TXT']=imagenet_aux_dict_env.get('CK_CAFFE_IMAGENET_SYNSET_WORDS_TXT','')

    # Only defined for the imagenet-val cmd.
    imagenet_val=deps.get('dataset-imagenet-val',{})
    imagenet_val_dict=imagenet_val.get('dict',{})
    imagenet_val_dict_env=imagenet_val_dict.get('env',{})
    d['CK_ENV_DATASET_IMAGENET_VAL']=imagenet_val_dict_env.get('CK_ENV_DATASET_IMAGENET_VAL','')

    # For the imagenet-val command, load ImageNet validation set labels.
    if d['CK_ENV_DATASET_IMAGENET_VAL']!='':
        with open(d['CK_CAFFE_IMAGENET_VAL_TXT']) as imagenet_val_txt:
            image_to_synset_map = {}
            for image_synset in imagenet_val_txt:
                (image, synset) = image_synset.split()
                image_to_synset_map[image] = int(synset)
        with open(d['CK_CAFFE_IMAGENET_SYNSET_WORDS_TXT']) as imagenet_synset_words_txt:
            synset_list = []
            for n00000000_synset in imagenet_synset_words_txt:
                synset = n00000000_synset[10:-1]
                synset_list.append(synset)

    # Load tensorrt-test output as list.
    r=ck.load_text_file({'text_file':'stdout.log', 'split_to_list':'yes'})
    if r['return']>0: return r

    # Collect info from tensorrt-test output per image:
    # image properties, layer profiling, all predictions and the best one.
    d['info_per_image'] = []
    for line in r['lst']:
        # Match image properties info in e.g.:
        # "loaded image  /home/ilsvrc2012_val/ILSVRC2012_val_00012996.JPEG  (375 x 500)  3000000 bytes"
        image_properties_regex = \
            'loaded image(\s)*' + \
            '(?P<image_path>[\w\./_-]*)(\s)*' + \
            '\((?P<image_width>\d+) x (?P<image_height>\d+)\)(\s)*' + \
            '(?P<image_size_bytes>\d+) bytes'
        match = re.search(image_properties_regex, line)
        if match:
            info_per_image = {}
            info_per_image['properties'] = {}
            info_per_image['properties']['path'] = match.group('image_path')
            info_per_image['properties']['width'] = match.group('image_width')
            info_per_image['properties']['height'] = match.group('image_height')
            info_per_image['properties']['size_bytes'] = match.group('image_size_bytes')
            # Prepare to match layer profiling, all predictions and the best one.
            info_per_image['all_predictions'] = []
            info_per_image['profiling'] = []

        # Match layer profiling info in e.g.:
        # "[GIE]  layer inception_3a/3x3_reduce + inception_3a/relu_3x3_reduce||inception_3a/5x5_reduce + inception_3a/relu_5x5_reduce - 1.747789 ms"
        # "[GIE]  layer network time - 45.530819 ms"
        profiling_regex = \
            '\[GIE\]  layer ' + \
            '(?P<layer>[\ \w_+/|]*)' + \
            ' - ' + \
            '(?P<time_ms>\d+\.\d+)(\s)*ms'
        match = re.search(profiling_regex, line)
        if match:
            info = {}
            info['layer'] = match.group('layer')
            info['time_ms'] = match.group('time_ms')
            info_per_image['profiling'].append(info)

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
            info_per_image['all_predictions'].append(info)

        # Match the most likely prediction in e.g.:
        # "tensorrt-test: '<file path>' -> 33.05664% class #331 (hare)"
        best_prediction_regex = \
            '\[tensorrt-test\](\s+)' + \
            '\'(?P<file_path>[\w\./_-]*)\'' + \
            '(\s)*->(\s)*' + \
            '(?P<probability_pc>\d+\.\d+)%' + \
            '(\s)*class(\s)*#(?P<class>\d+)(\s*)' + \
            '\((?P<synset>[\w\s,\']*)\)'
        match = re.search(best_prediction_regex, line)
        if match:
            info = {}
            info['file_path'] = match.group('file_path')
            info['file_name'] = os.path.basename(info['file_path'])
            info['probability'] = float(match.group('probability_pc'))*0.01
            info['class'] = int(match.group('class'))
            info['synset'] = match.group('synset')
            info_per_image['best_prediction'] = info
            info_per_image['all_predictions'] = sorted(info_per_image['all_predictions'], key=lambda k: k['probability'], reverse=True)
            # For the imagenet-val command, set 'accuracy_top1' and 'accuracy_top5' to 'yes' or 'no'.
            if d['CK_ENV_DATASET_IMAGENET_VAL']!='':
                best_prediction = info_per_image['best_prediction']
                best_prediction['class_correct'] = image_to_synset_map.get(best_prediction['file_name'],-1)
                best_prediction['synset_correct'] = synset_list[best_prediction['class_correct']]
                for n in [1,5]:
                    top_n_accuracy = 'accuracy_top'+str(n)
                    top_n_predictions = info_per_image['all_predictions'][0:n]
                    best_prediction[top_n_accuracy] = 'no'
                    for prediction in top_n_predictions:
                        if prediction['class']==best_prediction['class_correct']:
                            best_prediction[top_n_accuracy] = 'yes'
                            break
            # Final match in per image info.
            d['info_per_image'].append(info_per_image)
            d['post_processed'] = 'yes'
            d['execution_time'] = 0.0 # built-in CK key

    rr={}
    rr['return']=0
    if d.get('post_processed','')=='yes':
        r=ck.save_json_to_file({'json_file':'results.json', 'dict':d})
        if r['return']>0: return r
    else:
        rr['error']='failed to match best prediction in tensorrt-test output!'
        rr['return']=1

    return rr

# Do not add anything here!
