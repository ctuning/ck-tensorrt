Assuming you have generated the NMS plugin on the system,

register it directly:

```bash
    ck detect soft --tags=tensorrt,plugin,nms --full_path=/datasets/xavier-zenodo/libnmsoptplugin.so
```

or search for it:

```bash
    ck detect soft --tags=tensorrt,plugin,nms --search_dirs=/datasets
```
