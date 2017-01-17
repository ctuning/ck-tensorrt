import ck.kernel as ck
import copy
import re
import json

def do(i):
    # Detect basic platform info.
    ii={'action':'detect',
        'module_uoa':'platform',
        'out':'out'}
    r=ck.access(ii)
    if r['return']>0: return r

    # Host and target OS params.
    hos=r['host_os_uoa']
    hosd=r['host_os_dict']

    tos=r['os_uoa']
    tosd=r['os_dict']
    tdid=r['device_id']

    # Fix cmd key here since it may be used to get extra run-time deps.
    cmd_key='imagenet-val'

    # Load TensorRT-test program meta and desc to check deps.
    ii={'action':'load',
        'module_uoa':'program',
        'data_uoa':'tensorrt-test'}
    rx=ck.access(ii)
    if rx['return']>0: return rx
    mm=rx['dict']

    # Get compile-time and run-time deps.
    cdeps=mm.get('compile_deps',{})
    rdeps=mm.get('run_deps',{})

    # Merge rdeps with cdeps for setting up the pipeline (which uses
    # common deps), but tag them as "for_run_time".
    for k in rdeps:
        cdeps[k]=rdeps[k]
        cdeps[k]['for_run_time']='yes'

    # Limit the number of images (50,000 by default).
    max_num_images = 10 #50000

    # TensorRT engines.
    depl=copy.deepcopy(cdeps['lib-tensorrt'])

    ii={'action':'resolve',
        'module_uoa':'env',
        'host_os':hos,
        'target_os':tos,
        'device_id':tdid,
        'deps':{'lib-tensorrt':copy.deepcopy(depl)}
    }
    r=ck.access(ii)
    if r['return']>0: return r

    #udepl = ['tensorrt-1.0.0']
    udepl=r['deps']['lib-tensorrt'].get('choices',[]) # All UOAs of env for TensorRT engines.
    if len(udepl)==0:
        return {'return':1, 'error':'no registered TensorRT engines'}

    # Caffe models.
    depm=copy.deepcopy(rdeps['caffemodel'])

    ii={'action':'resolve',
        'module_uoa':'env',
        'host_os':hos,
        'target_os':tos,
        'device_id':tdid,
        'deps':{'caffemodel':copy.deepcopy(depm)}
    }
    r=ck.access(ii)
    if r['return']>0: return r

    udepm=r['deps']['caffemodel'].get('choices',[]) # All UOAs of env for Caffe models.
    if len(udepm)==0:
        return {'return':1, 'error':'no registered Caffe models'}

    # Prepare pipeline.
    cdeps['lib-tensorrt']['uoa']=udepl[0]
    cdeps['caffemodel']['uoa']=udepm[0]

    ii={'action':'pipeline',
        'prepare':'yes',

        'repo_uoa':'ck-tensorrt',
        'module_uoa':'program',
        'data_uoa':'tensorrt-test',
        'cmd_key':cmd_key,

        'dependencies': cdeps,

        'env':{
          'CK_TENSORRT_MAX_NUM_IMAGES': max_num_images
        },

        'no_compiler_description':'yes',
        'compile_only_once':'yes',

        'cpu_freq':'max',
        'gpu_freq':'max',

        'flags':'-O3',

        'speed':'no',
        'energy':'no',

        'no_state_check':'yes',
        'skip_calibration':'yes',

        'skip_print_timers':'yes',
        'out':'con',
    }

    r=ck.access(ii)
    if r['return']>0: return r

    fail=r.get('fail','')
    if fail=='yes':
        return {'return':10, 'error':'pipeline failed ('+r.get('fail_reason','')+')'}

    ready=r.get('ready','')
    if ready!='yes':
        return {'return':11, 'error':'pipeline not ready'}

    state=r['state']
    tmp_dir=state['tmp_dir']

    # Remember resolved deps for this benchmarking session.
    xcdeps=r.get('dependencies',{})

    # Clean pipeline.
    if 'ready' in r: del(r['ready'])
    if 'fail' in r: del(r['fail'])
    if 'return' in r: del(r['return'])

    pipeline=copy.deepcopy(r)

    # For each TensorRT engine.
    for lib_uoa in udepl:
        # Load TensorRT engine.
        ii={'action':'load',
            'module_uoa':'env',
            'data_uoa':lib_uoa}
        r=ck.access(ii)
        if r['return']>0: return r
        # Get the name e.g. 'TensorRT 1.0.0'
        lib_name='tensorrt-1.0.0' #r['data_name']
        lib_tags=lib_name
        # Skip some libs with "in [..]" or "not in [..]".
        if lib_name in []: continue

        # For each Caffe model.
        for model_uoa in udepm:
            # Load Caffe model.
            ii={'action':'load',
                'module_uoa':'env',
                'data_uoa':model_uoa}
            r=ck.access(ii)
            if r['return']>0: return r
            # Get the tags from e.g. 'Caffe model (net and weights) (deepscale, squeezenet, 1.1)'
            model_name=r['data_name']
            model_tags = re.match('Caffe model \(net and weights\) \((?P<tags>.*)\)', model_name)
            model_tags = model_tags.group('tags').replace(' ', '').replace(',', '-')
            # Skip some models with "in [..]" or "not in [..]".
            if model_tags not in ['nvidia-googlenet']: continue

            record_repo='local'
            record_uoa='imagenet-val-accuracy-'+model_tags+'-'+lib_tags

            # Prepare pipeline.
            ck.out('---------------------------------------------------------------------------------------')
            ck.out('%s - %s' % (lib_name, lib_uoa))
            ck.out('%s - %s' % (model_name, model_uoa))
            ck.out('Experiment - %s:%s' % (record_repo, record_uoa))

            # Prepare autotuning input.
            cpipeline=copy.deepcopy(pipeline)

            # Reset deps and change UOA.
            new_deps={#'lib-caffe':copy.deepcopy(depl),
                      'caffemodel':copy.deepcopy(depm)}

            #new_deps['lib-caffe']['uoa']=lib_uoa
            new_deps['caffemodel']['uoa']=model_uoa

            jj={'action':'resolve',
                'module_uoa':'env',
                'host_os':hos,
                'target_os':tos,
                'device_id':tdid,
                'deps':new_deps}
            r=ck.access(jj)
            if r['return']>0: return r

            cpipeline['dependencies'].update(new_deps)

            ii={'action':'autotune',

                'module_uoa':'pipeline',
                'data_uoa':'program',

                'choices_order':[
                    [
                        '##choices#env#CK_TENSORRT_ENABLE_FP16'
                    ]
                ],
                'choices_selection':[
                    {'type':'loop', 'start':0, 'stop':1, 'step':1, 'default':1}
                ],

                'features_keys_to_process':['##choices#env#CK_TENSORRT_ENABLE_FP16', '##choices#env#CK_TENSORRT_MAX_NUM_IMAGES'],

                'process_multi_keys':['##characteristics#compile#*'],

                'iterations':-1,
                'repetitions':1,

                'record':'yes',
                'record_failed':'yes',
                'record_params':{
                    'search_point_by_features':'yes'
                },
                'record_repo':record_repo,
                'record_uoa':record_uoa,

                'tags':['accuracy', 'imagenet-val', model_tags, lib_tags],

                'pipeline':cpipeline,
                'out':'con'}

            r=ck.access(ii)
            if r['return']>0: return r

            fail=r.get('fail','')
            if fail=='yes':
                return {'return':10, 'error':'pipeline failed ('+r.get('fail_reason','')+')'}

    return {'return':0}

r=do({})
if r['return']>0: ck.err(r)
