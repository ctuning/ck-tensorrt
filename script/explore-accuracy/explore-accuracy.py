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

    # Load TensorRT-test program meta and desc to check deps.
    ii={'action':'load',
        'module_uoa':'program',
        'data_uoa':'tensorrt-test'}
    rx=ck.access(ii)
    if rx['return']>0: return rx
    mm=rx['dict']

    # Update deps from GPGPU or ones remembered during autotuning.
    cdeps=mm.get('compile_deps',{})
    rdeps=mm.get('run_deps',{})

    # TODO: TensortRT libs (1.0, 2.0).
    udepl = ['tensorrt-1.0']

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
        return {'return':1, 'error':'no installed Caffe models'}

    # Prepare pipeline.
    #cdeps['lib-caffe']['uoa']=udepl[0]
    rdeps['caffemodel']['uoa']=udepm[0]

    ii={'action':'pipeline',
        'prepare':'yes',

        'repo_uoa':'ck-tensorrt',
        'module_uoa':'program',
        'data_uoa':'tensorrt-test',
        'cmd_key':'imagenet-val',

        'dependencies': cdeps,

        'no_compiler_description':'yes',
        'compile_only_once':'yes',

        'cpu_freq':'max',
        'gpu_freq':'max',

        'speed':'no',
        'energy':'no',

        'flags':'-O3',
        'env':{
           'CK_TENSORRT_MAX_IMAGES':10
         },

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

    # For each TensorRT lib.
    for lib_uoa in udepl:
        ## TODO: Load TensorRT lib.
        #ii={'action':'load',
        #    'module_uoa':'env',
        #    'data_uoa':lib_uoa}
        #r=ck.access(ii)
        #if r['return']>0: return r
        ## Get the name e.g. 'TensorRT 1.0'
        #lib_name=r['data_name']
        ## Skip some libs with "in [..]" or "not in [..]".
        #if lib_name in []: continue
        lib_name='TensorRT'
        lib_tags=lib_uoa

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
            if model_tags not in ['bvlc-alexnet', 'bvlc-googlenet']: continue

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
            pipeline_name = '%s.json' % record_uoa
            print ('Dumping pipeline to \'%s\'...' % pipeline_name)
            with open(pipeline_name, 'w') as f:
                json.dump(cpipeline, f, indent=2)

            ii={'action':'autotune',

                'module_uoa':'pipeline',
                'data_uoa':'program',

                'choices_order':[
                    [
                        '##env#CK_TENSORRT_ENABLE_FP16'
                    ]
                ],
                'choices_selection':[
                    {'type':'loop', 'start':0, 'stop':2, 'step':1, 'default':1}
                ],

                'features_keys_to_process':['##choices#*'],
                'process_multi_keys':['##characteristics#compile#*'],

                'iterations':1,
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
