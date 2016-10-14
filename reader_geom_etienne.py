#!/bin/env python3

from astropy.table import Table
from astropy.io import ascii
from collections import OrderedDict
from astropy.units import Quantity
import yaml
import numpy as np
import sys


templatef = "https://raw.githubusercontent.com/mireianievas/ctapipe_datamodel_config/master/array_config_compact_proto_v00.yaml"
infile = "https://drive.google.com/uc?export=download&id=0B4OIF0_Zm04WbFdfTzBuOTQ2em8"
content = ascii.read(infile)

















sys.exit(0)


######################### FULL #############3

def unique(items):
    return([k for k in list(set(items))])


with open(templatef) as fin:
    YamlTemplate = yaml.safe_load(fin)


Camera = dict(YamlTemplate['Camera'])
Camera["ID"] = "GCTProto_v00"
Camera["description"] = "GCT SST Prototype camera"
Camera["pixels"] = {}

Structs = [\
  "Can_master", "Can_node", "Module", \
  "MCR", "FADC_Board", "FADC_Quad", \
  "FADC_Chan" ]
Parents = Structs[:-1]
Childs  = Structs[1:]


def bool_representer(dumper, data):
    return dumper.represent_bool(data)
def int_representer(dumper, data):
    return dumper.represent_int(data)
def long_representer(dumper, data):
    return dumper.represent_long(data)
def float_representer(dumper, data):
    return dumper.represent_float(data)
def str_representer(dumper, data):
    return dumper.represent_str(data)
def array_representer(dumper, data):
    return dumper.represent_list(data)
def quantity_representer(dumper, data):
    item_key   = float(data.value)
    item_value = str(data.unit)
    value = "%s %s" %(item_key,item_value)
    
    return dumper.represent_str(value)


yaml.add_representer(np.bool, bool_representer)
yaml.add_representer(np.bool_, bool_representer)
yaml.add_representer(np.int16, int_representer)
yaml.add_representer(np.int32, int_representer)
yaml.add_representer(np.int64, int_representer)
yaml.add_representer(np.dtype(np.int32), int_representer)
yaml.add_representer(np.float16, float_representer)
yaml.add_representer(np.float32, float_representer)
yaml.add_representer(np.float64, float_representer)
yaml.add_representer(np.array, array_representer)
yaml.add_representer(Quantity, quantity_representer)


def get_flat_type(item):
    if 'int' in str(item.dtype):
        return(int(item))
    elif 'float' in str(item.dtype):
        return(float(item))
    else:
        print('Type not understood')
        return(None)


def check_id(box,idv):
    if box["ID"] is idv:
        return(True)
    else:
        return(False)

# remove the idXXX references in the dumper
class MyDumper(yaml.Dumper):
    def ignore_aliases(self, _data):
        return True

def quantity(value,unit=None):
    from astropy.units import Quantity
    if unit!=None:
        value = [value,unit]
    elif "[" and "]" in value:
        value = value.replace("]","[").split("[")
    elif "(" and ")" in value:
        value = value.replace(")","(").split("(")
    elif "<" and ">" in value:
        value = value.replace(">","<").split("<")
    else:
        value = value.split(" ")
    return(Quantity(value=float(value[0]),unit=value[1]))

### Create Drawers
criteria  = "Module"
critvalue = unique(content[criteria])

for cv in critvalue:
    Drawer = dict(YamlTemplate["Drawer"])
    Drawer["criteria"] = criteria
    Drawer["ID"]       = "%s=%s" %(criteria,cv)
    Drawer["active"]   = True
    Drawer["pixels"] = []
    if criteria not in Camera["drawers"]:
        Camera["drawers"][criteria] = {}
    Camera["drawers"][criteria][Drawer["ID"]] = Drawer

for k,pixel in enumerate(content):
    Pixel = dict(YamlTemplate["Pixel"])
    Pixel["ID"]     = "Pix_%d" %pixel["PixelID"]
    Pixel["posX"]   = quantity(pixel["x[mm]"],unit="mm")
    Pixel["posY"]   = quantity(pixel["y[mm]"],unit="mm")
    Pixel["number"] = pixel["PixelID"]
    Pixel["active"] = True

    # Write down the electronic information
    Electronics = Pixel["electronics"]
    for fields in [\
             "Can_master", "Can_node", "Module", "MCR", \
             "FADC_Board", "Pixel_number", "FADC_Quad", "FADC_Chan"]:
        Electronics[fields] = pixel[fields]

    CD = Camera["drawers"][criteria]
    for C in CD:
        if C == "%s=%s" %(criteria,pixel[criteria]):
            CD[C]["pixels"].append(Pixel["ID"])
            break
    
    Camera["pixels"][Pixel["ID"]] = Pixel

#content.write("test.ecsv",ascii.ecsv)

YamlCamOut = dict()

YamlCamOut["Camera"] = Camera

with open("gct_cam_output_test_v00.yaml","w+") as fout:
    yaml.dump(YamlCamOut, fout, Dumper=MyDumper, default_flow_style=False)


sys.exit(0)



################## ASTROPY TABLE #########3

# Convert the Astropy table to dict
YamlObject = dict()

### OrderedDict are not defined by default, so we have to register it in yaml.
def represent_ordereddict(dumper, data):
    value = []
    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)

        value.append((node_key, node_value))

    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)

yaml.add_representer(OrderedDict, represent_ordereddict)

YamlObject['CameraGeometry'] = OrderedDict()
YamlObject['CameraGeometry']['Header'] = content.columns.keys()
YamlObject['CameraGeometry']['Data'] = []

for pixel in content:
    Pixel = []
    for k,item in enumerate(content.columns.keys()):    
        Pixel.append(get_flat_type(pixel[item]))
    
    YamlObject['CameraGeometry']['Data'].append(Pixel) 

with open('test.yaml', 'w') as fout: 
    yaml.safe_dump(YamlObject, fout)#, default_flow_style=True)
    
#exit(0)

'''

for k,item in enumerate(content.columns.keys()):
    # Convert from numpys -> float and append as an array
    YamlObject['CameraGeometry'][item] = [get_flat_type(val) for val in content.columns[item]]

with open('test.yaml', 'w') as fout: 
    yaml.dump(YamlObject, fout)#, default_flow_style=True)
'''

# Read back the file and plot the array

with open('test.yaml', 'r') as fin: 
    YamlObject_read = yaml.load(fin)#, default_flow_style=True)

# The inline np.array(Data) is needed here, otherwise it raises an error.
CameraGeometry = Table(\
        np.array(YamlObject['CameraGeometry']['Data']), \
        names = YamlObject['CameraGeometry']['Header'])

print(CameraGeometry)

import matplotlib.pyplot as plt
x = CameraGeometry['x[mm]']
y = CameraGeometry['y[mm]']
plt.scatter(x=x, y=y, alpha=0.5)
plt.show()




