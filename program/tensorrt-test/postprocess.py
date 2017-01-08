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
    rt=i['run_time']
    env=i.get('env',{})
    deps=i.get('deps',{})

    d={}
    d['execution_time']=0.0
    d['debug']=rt['params'].get('debug','no')

    # Collect env vars of interest.
    d['REAL_ENV_CK_CAFFE_MODEL']=env.get('CK_CAFFE_MODEL','')

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
        top_n_list = [1,5]
        for n in top_n_list:
            top_n_accuracy = 'accuracy_top'+str(n)
            d[top_n_accuracy] = 0

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
            if d['debug']=='yes':
                info_per_image['properties']['path'] = match.group('image_path')
                info_per_image['properties']['width'] = match.group('image_width')
                info_per_image['properties']['height'] = match.group('image_height')
                info_per_image['properties']['size_bytes'] = match.group('image_size_bytes')
            # Prepare to match layer profiling, all predictions and the best one.
            info_per_image['all_predictions'] = []
            info_per_image['per_layer_info'] = []
            # Reset timer. NB: 'time_fw_ms' is the sum of all per layer timings.
            time_fw_ms = 0.0
            # Reset layer index.
            index = 0

        # Match layer profiling info in e.g.:
        # "[GIE]  layer inception_3a/3x3_reduce + inception_3a/relu_3x3_reduce||inception_3a/5x5_reduce + inception_3a/relu_5x5_reduce - 1.747789 ms"
        # "[GIE]  layer network time - 45.530819 ms"
        profiling_regex = \
            '\[GIE\]  layer ' + \
            '(?P<name>[\ \w_+/|]*)' + \
            ' - ' + \
            '(?P<time_ms>\d+\.\d+)(\s)*ms'
        match = re.search(profiling_regex, line)
        if match:
            name = match.group('name')
            time_ms = float(match.group('time_ms'))
            if name=='network time':
                # NB: Unlike 'time_fw_ms', 'time_total_ms' is parsed from TensorRT's output.
                # They should obviously match (and normally do).
                info_per_image['time_total_ms'] = time_ms
                info_per_image['time_total_s'] = time_ms * 1e-3
            elif d['debug']=='yes':
                time_fw_ms += time_ms
                layer_info = {}
                layer_info['name'] = name
                layer_info['time_ms'] = time_ms
                layer_info['index'] = index; index += 1
                # Update optional keys for compatibility with CK-Caffe.
                layer_info['time_s'] = layer_info['time_ms'] * 1e-3
                layer_info['label'] = '%02d: %s' % (layer_info['index'], layer_info['name'])
                layer_info['timestamp'] = '0101 00:00:00.000000' # FIXME: Add proper timestamp?
                layer_info['direction'] = 'forward'
                info_per_image['per_layer_info'].append(layer_info)

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
            if d['debug']=='yes':
                info['synset'] = match.group('synset')
            info_per_image['all_predictions'].append(info)

        # Match the most likely prediction in e.g.:
        # ["tensorrt-test] '<file path>' -> 33.05664% class #331 (hare)"

        best_prediction_regex = \
            '\[tensorrt-test\](\s+)' + \
            '\'(?P<file_path>[\w\./_-]*)\'' + \
            '(\s)*->(\s)*' + \
            '(?P<probability_pc>\d+\.\d+)%' + \
            '(\s)*class(\s)*#(?P<class>\d+)(\s*)' + \
            '\((?P<synset>[\w\s,\'\-]*)\)'
        match = re.search(best_prediction_regex, line)
        if match:
            info = {}
            file_path = match.group('file_path')
            info['file_name'] = os.path.basename(file_path)
            info['class'] = int(match.group('class'))
            if d['debug']=='yes':
                info['file_path'] = file_path
                info['synset'] = match.group('synset')
                info['probability'] = float(match.group('probability_pc'))*0.01
            info_per_image['best_prediction'] = info
            info_per_image['all_predictions'] = sorted(info_per_image['all_predictions'], key=lambda k: k['probability'], reverse=True)
            # For the imagenet-val command, set 'accuracy_top1' and 'accuracy_top5' to 'yes' or 'no'.
            if d['CK_ENV_DATASET_IMAGENET_VAL']!='':
                best_prediction = info_per_image['best_prediction']
                class_correct = image_to_synset_map.get(best_prediction['file_name'],-1)
                for n in top_n_list:
                    top_n_accuracy = 'accuracy_top'+str(n)
                    top_n_predictions = info_per_image['all_predictions'][0:n]
                    best_prediction[top_n_accuracy] = 'no'
                    for prediction in top_n_predictions:
                        if prediction['class']==class_correct:
                            best_prediction[top_n_accuracy] = 'yes'
                            d[top_n_accuracy] += 1
                            break
                if d['debug']=='yes':
                    best_prediction['class_correct'] = class_correct
                    best_prediction['synset_correct'] = synset_list[best_prediction['class_correct']]

            # If we are here, it's the final match in per image info.
            if d['debug']=='yes':
                # Finalize the execution time info.
                # Execution time (ms).
                info_per_image['time_fw_ms'] = time_fw_ms
                info_per_image['time_bw_ms'] = 0.0
                info_per_image['time_fwbw_ms'] = info_per_image['time_fw_ms'] + info_per_image['time_bw_ms']
                info_per_image['time_total_ms_kernel_0'] = info_per_image['time_total_ms']
                # Execution time (s).
                info_per_image['time_fw_s'] = info_per_image['time_fw_ms'] * 1e-3
                info_per_image['time_bw_s'] = info_per_image['time_bw_ms'] * 1e-3
                info_per_image['time_fwbw_s'] = info_per_image['time_fwbw_ms'] * 1e-3
                info_per_image['time_total_s_kernel_0'] = info_per_image['time_total_ms_kernel_0'] * 1e-3
            else:
                # Remove all predictions info.
                info_per_image.pop('all_predictions', None)

            # Finalize the per image time info.
            d['info_per_image'].append(info_per_image)

            # Built-in CK keys.
            d['execution_time'] += info_per_image['time_total_s']
            d['post_processed'] = 'yes'

    rr={}
    rr['return']=0
    if d.get('post_processed','')=='yes':
        if d['CK_ENV_DATASET_IMAGENET_VAL']!='':
            num_images = len(d['info_per_image'])
            scaling = 1.0 / num_images
            for n in top_n_list:
                 top_n_accuracy = 'accuracy_top'+str(n)
                 d[top_n_accuracy] *= scaling
        r=ck.save_json_to_file({'json_file':'results.json', 'dict':d})
        if r['return']>0: return r
    else:
        rr['error']='failed to match best prediction in tensorrt-test output'
        rr['return']=1

    return rr

# Do not add anything here!
