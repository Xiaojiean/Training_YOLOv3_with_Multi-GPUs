import copy

def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    net_sample = []
    down_sample_index = []
    net_sample_index = 0
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
            if module_defs[-1]['type'] == 'shortcut':
                net_sample.append({
                    'module_index': [net_sample_index-3, net_sample_index-2, net_sample_index-1],
                    'importance': 1.0,
                    'remove': False
                })
            net_sample_index += 1
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
            if module_defs[-1]['type'] =='convolutional' and key=='stride' and value=='2':
                down_sample_index.append(net_sample_index-2)

    return module_defs, net_sample, down_sample_index


def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options
