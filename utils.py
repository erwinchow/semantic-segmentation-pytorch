import os
import re
import functools
import fnmatch
import numpy as np

labels = {
0: "unknown",
1: "wall",
2: "building;edifice",
3: "sky",
4: "floor;flooring",
5: "tree",
6: "ceiling",
7: "road;route",
8: "bed",
9: "windowpane;window",
10: "grass",
11: "cabinet",
12: "sidewalk;pavement",
13: "person;individual;someone;somebody;mortal;soul",
14: "earth;ground",
15: "door;double;door",
16: "table",
17: "mountain;mount",
18: "plant;flora;plant;life",
19: "curtain;drape;drapery;mantle;pall",
20: "chair",
21: "car;auto;automobile;machine;motorcar",
22: "water",
23: "painting;picture",
24: "sofa;couch;lounge",
25: "shelf",
26: "house",
27: "sea",
28: "mirror",
29: "rug;carpet;carpeting",
30: "field",
31: "armchair",
32: "seat",
33: "fence;fencing",
34: "desk",
35: "rock;stone",
36: "wardrobe;closet;press",
37: "lamp",
38: "bathtub;bathing;tub;bath;tub",
39: "railing;rail",
40: "cushion",
41: "base;pedestal;stand",
42: "box",
43: "column;pillar",
44: "signboard;sign",
45: "chest;of;drawers;chest;bureau;dresser",
46: "counter",
47: "sand",
48: "sink",
49: "skyscraper",
50: "fireplace;hearth;open;fireplace",
51: "refrigerator;icebox",
52: "grandstand;covered;stand",
53: "path",
54: "stairs;steps",
55: "runway",
56: "case;display;case;showcase;vitrine",
57: "pool;table;billiard;table;snooker;table",
58: "pillow",
59: "screen;door;screen",
60: "stairway;staircase",
61: "river",
62: "bridge;span",
63: "bookcase",
64: "blind;screen",
65: "coffee;table;cocktail;table",
66: "toilet;can;commode;crapper;pot;potty;stool;throne",
67: "flower",
68: "book",
69: "hill",
70: "bench",
71: "countertop",
72: "stove;kitchen;stove;range;kitchen;range;cooking;stove",
73: "palm;palm;tree",
74: "kitchen;island",
75: "computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system",
76: "swivel;chair",
77: "boat",
78: "bar",
79: "arcade;machine",
80: "hovel;hut;hutch;shack;shanty",
81: "bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle",
82: "towel",
83: "light;light;source",
84: "truck;motortruck",
85: "tower",
86: "chandelier;pendant;pendent",
87: "awning;sunshade;sunblind",
88: "streetlight;street;lamp",
89: "booth;cubicle;stall;kiosk",
90: "television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box",
91: "airplane;aeroplane;plane",
92: "dirt;track",
93: "apparel;wearing;apparel;dress;clothes",
94: "pole",
95: "land;ground;soil",
96: "bannister;banister;balustrade;balusters;handrail",
97: "escalator;moving;staircase;moving;stairway",
98: "ottoman;pouf;pouffe;puff;hassock",
99: "bottle",
100: "buffet;counter;sideboard",
101: "poster;posting;placard;notice;bill;card",
102: "stage",
103: "van",
104: "ship",
105: "fountain",
106: "conveyer;belt;conveyor;belt;conveyer;conveyor;transporter",
107: "canopy",
108: "washer;automatic;washer;washing;machine",
109: "plaything;toy",
110: "swimming;pool;swimming;bath;natatorium",
111: "stool",
112: "barrel;cask",
113: "basket;handbasket",
114: "waterfall;falls",
115: "tent;collapsible;shelter",
116: "bag",
117: "minibike;motorbike",
118: "cradle",
119: "oven",
120: "ball",
121: "food;solid;food",
122: "step;stair",
123: "tank;storage;tank",
124: "trade;name;brand;name;brand;marque",
125: "microwave;microwave;oven",
126: "pot;flowerpot",
127: "animal;animate;being;beast;brute;creature;fauna",
128: "bicycle;bike;wheel;cycle",
129: "lake",
130: "dishwasher;dish;washer;dishwashing;machine",
131: "screen;silver;screen;projection;screen",
132: "blanket;cover",
133: "sculpture",
134: "hood;exhaust;hood",
135: "sconce",
136: "vase",
137: "traffic;light;traffic;signal;stoplight",
138: "tray",
139: "ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin",
140: "fan",
141: "pier;wharf;wharfage;dock",
142: "crt;screen",
143: "plate",
144: "monitor;monitoring;device",
145: "bulletin;board;notice;board",
146: "shower",
147: "radiator",
148: "glass;drinking;glass",
149: "clock",
150: "flag"}


def find_recursive(root_dir, ext='.jpg'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='BGR'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))
        print(label+1, labels[label+1])
        print(accuracy(labelmap, label))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


class NotSupportedCliException(Exception):
    pass


def process_range(xpu, inp):
    start, end = map(int, inp)
    if start > end:
        end, start = start, end
    return map(lambda x: '{}{}'.format(xpu, x), range(start, end+1))


REGEX = [
    (re.compile(r'^gpu(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^gpu(\d+)-(?:gpu)?(\d+)$'),
     functools.partial(process_range, 'gpu')),
    (re.compile(r'^(\d+)-(\d+)$'),
     functools.partial(process_range, 'gpu')),
]


def parse_devices(input_devices):

    """Parse user's devices input str to standard format.
    e.g. [gpu0, gpu1, ...]

    """
    ret = []
    for d in input_devices.split(','):
        for regex, func in REGEX:
            m = regex.match(d.lower().strip())
            if m:
                tmp = func(m.groups())
                # prevent duplicate
                for x in tmp:
                    if x not in ret:
                        ret.append(x)
                break
        else:
            raise NotSupportedCliException(
                'Can not recognize device: "{}"'.format(d))
    return ret
