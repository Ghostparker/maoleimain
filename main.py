# -*- coding: UTF-8 -*-
import cv2
import os, sys
import os.path as osp
import numpy as np
import json
import pyezi as ezi

cur_path = os.path.dirname(os.path.abspath(__file__))
if cur_path not in sys.path:
    sys.path.insert(0, cur_path)
from tools.preprocess_expre import PreprocessExpreModule
import postprocessfactory


class TagInfo:
    def __init__(self,line):
        data = line.strip().split()
        self.info = {}
        self.info['tagname'] = data[0]
        self.info['tagid'] = int(data[1])
        self.info['thlow'] = float(data[2])
        self.info['thhigh'] = float(data[3])
        self.info['outname'] = data[4]
        self.info['outindex'] = int(data[5])

    def show(self):
        print(self.info)

    def gettagname(self):
        return self.info['tagname']

    def gettagid(self):
        return self.info['tagid']

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, time):
            return obj.__str__()
        return json.JSONEncoder.default(self, obj)


# preprocess
def convert_image(image, box=None, preprocess_handle=None):
    usr_data = {}
    image = preprocess_handle.convert_img(image, box, usr_data)
    return image, usr_data


class ModelInfer:
    """
    gpuid: used for multi thread and multi card in pipeline
    """

    def __init__(self, ezm_path='model.ezm', mode='test', gpuid=0):
        # read output_names from model/model.json
        sample = json.load(open('{}/model/model.json'.format(cur_path)))
        self.out_layer_name = sample["output_names"]
        self.model = ezi.CVModel(sample["model_name"])
        self.mode = mode
        self.model_name = sample["model_name"]

        # preprocess init...
        self.preprocess_handle = PreprocessExpreModule('{}/data/pre_conf.json'.format(cur_path))
        # postprocess init...
        postprocessfactory.add_post_process(sample["model_name"], "{}/data/".format(cur_path))

        self.img_idx = 0
        if self.mode == 'save_pb':
            # self.model.PreserveInputs(True)
            if not os.path.isdir('engines/calib/'): os.makedirs('engines/calib/')
        else:
            if gpuid > 0:
                self.model.LoadModelFromFile(ezm_model, 'cfgs/trt_gpuid_%d.json' % gpuid)
            else:
                self.model.LoadModelFromFile(ezm_path)

        self.label_map = {0: '0', 1: '1', 2: '2', 3: '3'}
        self.set_taginfo()

    ''' self.mode=='save_pb': used to generate quantization files and only save pb files.
        self.mode=='dev': used to develop,return model outputs without threshold.
        self.mode==others : used to EZI test.
    '''


    def set_taginfo(self):
        self.taginfo = []
        self.tagcfgpath = osp.join(os.path.dirname(os.path.abspath(__file__)),'data','tag_uid.cfg')
        for line in open(self.tagcfgpath):
            if('#' in line or len(line.strip().split()) != 6):
                continue
            tmpbox = TagInfo(line)
            self.taginfo.append(tmpbox)


    def add_vegaout(self,vega_out):
        try:
            print(vega_out)
            print(type(vega_out))
            if(isinstance(vega_out[0],list) ):
                for idx,one_vegaout in enumerate(vega_out[0]):
                    print(vega_out[0][idx]['tagnameid'])
                    tmptagname = self.get_tagname_bytagid(vega_out[0][idx]['tagnameid'])
                    vega_out[0][idx]['tagname'] = tmptagname
            return vega_out
        except:
            print('except')
            return vega_out

    def get_tagname_bytagid(self,search_tagid):
        # print(search_tagid)
        # print(type(search_tagid))
        for i in self.taginfo:
            if(i.info['tagid'] == search_tagid):
                return i.info['tagname']
        return 'not exist'

    def get_infer_result(self, image, box=None):
        # preprocess image
        image, usr_data = convert_image(image, box, self.preprocess_handle)

        # only save pb files
        if self.mode == 'save_pb':
            self.img_idx += 1
            # input = self.model.RetrieveInput('data')
            # self.model.SaveInputsToFile('engines/calib/' + '%06d' % self.img_idx + '.pb')
            ezi.SaveNDArraysToBlobs({'data': image}, 'engines/calib/' + '%06d' % self.img_idx + '.pb')
            return None, None

        self.model.UploadInput('data', image)
        # model infer
        self.model.Infer()

        # get model outputs
        output = {}
        for name in self.out_layer_name:
            out = self.model.DownloadOutput(name).flatten()
            output[name] = out

        # NOTE call vega postprocess. if new postprocess specially and add new code here.
        outputs, vega_outputs = postprocessfactory.execute_post_process(self.model_name, [output], [usr_data])
        if self.mode == "dev":
            return outputs
        # print(type(vega_outputs[0][0]))
        return outputs, self.add_vegaout(vega_outputs)


if "__main__" == __name__:
    if len(sys.argv) == 5:
        mode = sys.argv[1]  # 'save_pb' #'dev'
        file_list = sys.argv[2]
        ezm_model = sys.argv[3]
        img_prefix = sys.argv[4]
    elif len(sys.argv) == 3 and sys.argv[1] == 'save_pb':
        mode = sys.argv[1]  # 'save_pb' #'dev'
        file_list = sys.argv[2]
        ezm_model = 'unused with save_pb'
        img_prefix = './'
    else:
        print("Usage: python main.py <MODE> <JSON> [EZM_MODEL]\n")
        print("Available MODE:\n")
        print("save_pb: only save pb files and EZM_MODEL is not need to pass parameters\n")
        print("others: test JSON files\n")
        exit(0)

    handle = ModelInfer(ezm_model, mode=mode)

    if file_list[-5:] != '.json':
        print("Input file must be json file...")
        exit(-1)
    save_name = os.path.splitext(os.path.basename(file_list))[0]
    # read json
    lines = open(file_list).readlines()

    save_list = []
    for idx, line in enumerate(lines):
        if idx % 1000 == 0: print(mode, idx)
        sample = json.loads(line.strip())

        roi_box = None
        for key in sample.keys():
            if key == 'image':
                img_path = sample[key]
            elif key == 'roi_box' or key == 'box':
                roi_box = sample[key]

        img = cv2.imread(os.path.join(img_prefix, img_path))
        if img is None:
            print("Error: can't load image " + os.path.join(img_prefix, img_path))
            exit(-1)

        height, width = img.shape[:2]
        roi_box = [0, 0, width, height] if roi_box is None else roi_box  # default is image size

        out, vega_out = handle.get_infer_result(img, roi_box)
        if out is None or mode == 'save_pb':
            print("The mode has't results", mode)
            continue

        # save vega output to json
        item = {"image": img_path, "result": vega_out[0]}
        with open("{}/{}_vega_results.json".format('./', save_name), "a") as dump_f:
            json.dump(item, dump_f, cls=NumpyEncoder)
            dump_f.write('\n')
