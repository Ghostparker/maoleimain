from main import ModelInfer
import cv2,os


if __name__ == '__main__':

    ezm_model = './/ezm/VehicleDanger_trt_int8_bs4.ezm'
    mode='test'
    handle = ModelInfer(ezm_model, mode=mode)

    #save_name = os.path.splitext(os.path.basename(file_list))[0]
    #read json
    file_list = './testimg/t1.txt'
    lines = [line.strip().split(' ')[0] for line in open(file_list)]

    save_list = []
    for idx, line in enumerate(lines):
        if idx % 1000 == 0: print(mode, idx)
        #sample = json.loads(line.strip())
        print(line)
        roi_box = None

        img = cv2.imread(line)
        if img is None:
            print("Error: can't load image " + os.path.join(img_prefix, img_path))
            exit(-1)

        height, width = img.shape[:2]
        print(img.shape)
        roi_box = [0, 0, width, height] if roi_box is None else roi_box #default is image size

        out = handle.get_infer_result(img, roi_box)
        print(out)
