import yaml
import logging
import sys
from collections import OrderedDict

dic = dict()


def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):  # type: ignore

    class OrderedLoader(Loader):  # type: ignore
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,  # type: ignore
            construct_mapping)
    return yaml.load(stream, OrderedLoader)


def loadconfig(config_path=None):
    global dic
    if config_path is None:
        return
    else:
        logging.info('loading local config file {}'.format(config_path))
        with open(config_path) as stream:
            local_dict = ordered_load(stream, yaml.SafeLoader)  # type: ignore
            if sys.version_info[0] == '2':
                for k, v in local_dict.iteritems():
                    dic[k] = v
            else:
                for k, v in local_dict.items():
                    dic[k] = v


def write_config_to_file(filepath):
    global dic
    if len(dic) > 0:
        f = open(filepath, 'w')
        if sys.version_info[0] == '2':
            for k, v in dic.iteritems():
                f.write(str(k)+': '+str(v)+'\n')
        else:
            for k, v in dic.items():
                f.write(str(k)+': '+str(v)+'\n')
        f.close()
    else:
        logging.exception('Trying to write config to file, but no config has been loaded.')
        raise ValueError('Trying to write config to file, but no config has been loaded.')


def get_from_config(par):
    global dic
    if len(dic) > 0:
        try:
            val = dic[par]
            if isinstance(val, str):
                val = val[:val.rfind("#")] if "#" in val else val
                val = val.rstrip().lstrip()
                val = None if val == 'None' else val
            return val
        except:
            logging.exception("{} not in config file or commented out; returning -1".format(par))
            raise Exception("{} not in config file or commented out".format(par))
    else:
        logging.exception('No config has been loaded.')
        raise ValueError('No config has been loaded.')


if __name__ == '__main__':
    loadconfig()
