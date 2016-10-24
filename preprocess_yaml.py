#!/bin/env python3

#!/bin/env python3

from astropy.table import Table
from astropy.io import ascii
from astropy.units import Quantity
import ruamel.yaml as yaml
from collections import OrderedDict
from ruamel.yaml.loader      import Loader
from ruamel.yaml.dumper      import Dumper
from ruamel.yaml.constructor import Constructor
from ruamel.yaml.representer import Representer
import ruamel.yaml.comments as Comments
import numpy as np
import sys
from urllib.request import urlopen



class File(str):
    #yaml_tag = u'!File'
    def __init__(self, filename):
        self.filename = filename
    def read(self):
        with open(self.filename,"r") as fin:
            return(yaml.load(fin))
    def append(self,data):
        with open(self.filename,"a+") as fout:
            yaml.dump(data, fout, Dumper=Dumper)
    def write(self,data):
        with open(self.filename,"r+") as fout:
            yaml.dump(data, fout, Dumper=Dumper)
    def truncate(self):
        with open(self.filename,"w") as fout:
            fout.write("")
    def __new__(cls, a):
        return str.__new__(cls, a)
    def __repr__(self):
        return "File(%s)" % self

def extfile_representer(dumper, data):
    return dumper.represent_scalar(u'!file', data.filename)

def extfile_constructor(loader, node):
    from ruamel.yaml.nodes import ScalarNode
    if isinstance(node, ScalarNode):
        return File(node.value)

yaml.add_representer(File, extfile_representer)
yaml.add_constructor(u'!file', extfile_constructor)

### expand

with open("gct_telescope_output_test_v00.yaml") as fin:
    Telescope = yaml.load(fin)['GCT_Telescope_Prototype']

def recursive_build(item):
    for subitem in item:
        content = item[subitem]
        if type(content) is File:
            with open(content,'r') as fin:
                item[subitem] = content.read()
                item[subitem]['fromfilename'] = content.filename
                content = item[subitem]

        if type(content) is dict:
            content = recursive_build(content)

        item[subitem] = content

    return item
