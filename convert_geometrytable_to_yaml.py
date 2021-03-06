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

templatef_local  = "array_config_compact_proto_v00.yaml"
templatef_remote = "https://raw.githubusercontent.com/mireianievas/ctapipe_datamodel_config/master/array_config_compact_proto_v00.yaml"
infile_camera  = "https://drive.google.com/uc?export=download&id=0B4OIF0_Zm04WbFdfTzBuOTQ2em8"
infile_mirrors = "https://raw.githubusercontent.com/mireianievas/ctapipe_datamodel_config/master/configOpticsGCT.yaml"

##### YAML Handlers
def quantity_representer(dumper, data):
    item_key   = float(data.value)
    item_value = str(data.unit)
    value = "%s %s" %(item_key,item_value)
    return yaml.representer_str(value)
def dict_representer(dumper, data):
    return dumper.represent_dict(data.items())
def dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))
def construct_yaml_odict(loader, node):
    from itertools import chain
    from ruamel.yaml.compat import ordereddict
    merge_map = loader.flatten_mapping(node)
    omap_gen=loader.construct_yaml_omap(node)
    omap = ordereddict((k,v) for (k,v) in omap_gen)
    if merge_map:
        omap.add_yaml_merge(merge_map)
    yield omap


class Loader_map_as_anydict( object):
    'inherit + Loader'
    anydict = None      #override
    @classmethod        #and call this
    def load_map_as_anydict( klas):
        yaml.add_constructor( 'tag:yaml.org,2002:map', klas.construct_yaml_map)

    'copied from constructor.BaseConstructor, replacing {} with self.anydict()'
    def construct_mapping(self, node, deep=False):
        if not isinstance(node, MappingNode):
            raise ConstructorError(None, None,
                    "expected a mapping node, but found %s" % node.id,
                    node.start_mark)
        mapping = self.anydict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError as exc:
                raise ConstructorError("while constructing a mapping", node.start_mark,
                        "found unacceptable key (%s)" % exc, key_node.start_mark)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping

    def construct_yaml_map( self, node):
        data = self.anydict()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

if np is not None:
    # Represent 1d ndarrays as lists in yaml files because it makes them much
    # prettier
    def ndarray_representer(dumper, data):
        if len(np.shape(data)) == 1:
            return dumper.represent_list([k for k in data])
        elif len(np.shape(data)) == 2:
            return dumper.represent_list([[l for l in k] for k in data])
        else:
            return dumper.represent_list(data.tolist())
    def complex_representer(dumper, data):
        return dumper.represent_scalar('!complex', repr(data.tolist()))
    def complex_constructor(loader, node):
        return complex(node.value)
    def numpy_float_representer(dumper, data):
        return dumper.represent_float(float(data))
    def numpy_int_representer(dumper, data):
        return dumper.represent_int(int(data))
    def numpy_dtype_representer(dumper, data):
        return dumper.represent_scalar('!dtype', data.name)
    def numpy_dtype_loader(loader, node):
        name = loader.construct_scalar(node)
        return np.dtype(name)

    Representer.add_representer(np.ndarray, ndarray_representer)
    Representer.add_representer(np.complex128, complex_representer)
    Representer.add_representer(np.complex, complex_representer)
    Representer.add_representer(np.float64, numpy_float_representer)
    Representer.add_representer(np.int64, numpy_int_representer)
    Representer.add_representer(np.int32, numpy_int_representer)
    Representer.add_representer(np.dtype, numpy_dtype_representer)
    Constructor.add_constructor('!complex', complex_constructor)
    Constructor.add_constructor('!dtype', numpy_dtype_loader)

Representer.add_representer(Quantity, quantity_representer)
Representer.add_representer(OrderedDict, dict_representer)
#Constructor.add_constructor(u'tag:yaml.org,2002:map',construct_yaml_odict)

# remove the idXXX references in the dumper
MyLoader = Loader
class MyDumper(Dumper):
    def ignore_aliases(self, _data):
        return True

def as_quantity(value,unit=None):
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

def unique(items):
    return([k for k in list(set(items))])


# Read the content and the template
with urlopen(templatef_remote) as fin:
    YamlTemplate = yaml.load(fin, Loader=MyLoader)
raw_camgeometry_file_content = ascii.read(infile_camera)
with urlopen(infile_mirrors) as fin:
    YamlOptics = yaml.load(fin, Loader=MyLoader)

### Create Drawers
Telescope   = OrderedDict(YamlTemplate['Telescope'])
Camera      = Telescope['camera']
Optics      = Telescope['optics']
Mount       = Telescope['mount']
MonitorUnit = Telescope['monitorunit']

Camera["ID"] = "GCTProto_v00"
Camera["Description"] = "GCT SST Prototype camera"
Camera["type"] = {"TelescopeCamera": "GCT"}

PixelTable = Camera["pixels"]
DrawerDict = Camera["drawers"]

DrawerDict['Module'] = OrderedDict(YamlTemplate['Drawer'])
DrawerModule = DrawerDict['Module']
DrawerModule["ID"] = "Drawer:Module"
DrawerModule["Description"] = "Cluster of pixels with the same module"
DrawerModule["Header"] = ["ID", "Can_master", "Can_node", "Module"]
DrawerModule["Units"]  = [None, None, None, None]

for drawer in DrawerDict:
    raw_camgeometry_file_content["ID"] = raw_camgeometry_file_content[drawer]
    data = [[c for c in l]\
        for l in raw_camgeometry_file_content[DrawerModule["Header"]]]
    # Join by ||, then find unique
    unique_rows = np.unique(['||'.join(str(l))\
        for l in data],return_index=True)[1]
    DrawerModule["Data"] = np.array(data)[unique_rows]

PixelTable["ID"] = "PixelTable"
PixelTable["Description"] = "Cluster of pixels with the same module"
PixelTable["Header"] = ["ID", "x", "y", "Pixel_number", "Drawer:Module"]
header_from = ["ID", "x[mm]", "y[mm]", "Pixel_number", "Module"]
PixelTable["Units"]  = [None, "mm", "mm", None]
raw_camgeometry_file_content["ID"] = raw_camgeometry_file_content[drawer]

PixelTable["Data"] = [[c for c in l] \
    for l in raw_camgeometry_file_content[header_from]]


camoutfile = "gct_cam_output_test_v00.yaml"
with open(camoutfile,"w+") as fout:
    yaml.round_trip_dump(Camera, fout, Dumper=MyDumper)

optoutfile = "gct_optics_output_test_v00.yaml"
Optics = YamlOptics['TelescopeOpticsGCT']
Optics["type"] = {"TelescopeOptics": "GCT"}

with open(optoutfile,"w+") as fout:
    yaml.round_trip_dump(Optics, fout, Dumper=MyDumper)

def extfile_representer(dumper, data):
    return dumper.represent_scalar(u'!file', data.filename)

def extfile_constructor(loader, node):
    return File(node)

class File(str):
    #yaml_tag = u'!File'
    def __init__(self, filename):
        self.filename = filename

    def read(self):
        with open(self.filename,"r") as fin:
            return(fin)
    def append(self,data):
        with open(self.filename,"a+") as fout:
            fout.write(data)
    def write(self,data):
        with open(self.filename,"r+") as fout:
            fout.write(data)
    def truncate(self):
        with open(self.filename,"w") as fout:
            fout.write("")

    def __new__(cls, a):
        return str.__new__(cls, a)
    def __repr__(self):
        return "File(%s)" % self

yaml.add_representer(File, extfile_representer)
yaml.add_constructor(u'!file', extfile_constructor)

def yaml_include_file(filename):
    return File(filename)

YamlObject = OrderedDict()
YamlObject['GCT_Telescope_Prototype'] = Telescope
Telescope["ID"]='GCT_Telescope_Prototype'
Telescope["camera"] = yaml_include_file(camoutfile)
Telescope["optics"] = yaml_include_file(optoutfile)
Telescope["type"] = {"Telescope": "GCT"}

'''
# To implement:
# How to expand this include_file format
optics = Yaml['GCT_Telescope_Prototype']['optics']
if 'file' in optics:
    with open(optics['file']) as fin:
        Yaml['GCT_Telescope_Prototype']['optics'] = yaml.load(fin)
loop over dicts (recursive function ??)
'''

teloutfile = "gct_telescope_output_test_v00.yaml"
with open(teloutfile,"w+") as fout:
    yaml.dump(YamlObject, fout, Dumper=MyDumper)





sys.exit(0)


















####### Find below some earlier attempts to generate YAML files #######

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

#YamlCamOut = dict()
#YamlCamOut["Camera"] = Camera

camoutfile = "gct_cam_output_test_v00.yaml"
with open(camoutfile,"w+") as fout:
    yaml.dump(Camera, fout, Dumper=MyDumper, default_flow_style=False)

optoutfile = "gct_optics_output_test_v00.yaml"
Optics = YamlOptics['TelescopeOpticsGCT']
with open(optoutfile,"w+") as fout:
    yaml.dump(Optics, fout, Dumper=MyDumper, default_flow_style=False)

Telescope = dict(YamlTemplate['Telescope'])
Telescope["camera"] = "include: %s" %camoutfile
Telescope["optics"] = "include: %s" %optoutfile

teloutfile = "gct_telescope_output_test_v00.yaml"
with open(teloutfile,"w+") as fout:
    yaml.dump(Telescope, fout, Dumper=MyDumper, default_flow_style=False)


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
