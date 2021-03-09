import torch
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import torchvision.transforms as transforms

import sys

class Detector():
    def __init__(self):
        BASE_PATH = ''
        self.webcam = True
        self.weights= BASE_PATH + 'weights/last.pt'
        self.half = True
        self.cfg = BASE_PATH + 'cfg/yolov3-tiny.cfg'
        self.names= BASE_PATH + 'data/cones.names'
        self.image_size = 416
        self.classes = "+"

        self.conf_thresh = 0.05
        self.iou_thresh = 0.3

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Darknet(self.cfg, self.image_size)
        self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])

        # Eval mode
        self.model.to(self.device).eval()

        # Half precision
        self.half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()

    def detect(self, image):
        # color_image_RGB = img
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img = transforms.ToTensor()(img)

        pre_scale_shape = img.shape
        print(f"pre-dims: {img.shape}")

        def pad_to_square(img, pad_value):
            c, h, w = img.shape
            dim_diff = np.abs(h - w)
            # (upper / left) padding and (lower / right) padding
            pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
            # Determine padding
            pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
            # Add padding
            img = F.pad(img, pad, "constant", value=pad_value)

            return img, pad 

        def resize(image, size):
            image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
            return image

        square_img, pad = pad_to_square(img, 0)

        img = resize(square_img, self.image_size)
        after_scale_img = img
        print(f"post-dims: {img.shape}")
        img = img.unsqueeze(0)
        img = img.cuda()


        print(type(img))
        print(img.shape)

        # Run inference

        # img = torch.zeros((1, 3, self.image_size, self.image_size), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img.float()) if self.device.type != 'cpu' else None  # run once

        # for path, img, im0s, vid_cap in dataset:
            # img = torch.from_numpy(img).to(device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32

        # print(f"img: {img}")
        # print(f"max: {img.max()}")
        # print(f"mean: {img.mean()}")
        # 10/0
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = self.model(img, augment=False)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if self.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh,
                                   multi_label=False, classes=False, agnostic=False)[0]

        # rescale coords
        scale_coef = max(pre_scale_shape[1:3]) / 416;
        pred[:, 0:4] *= scale_coef;


        # resclale

        # remove padding
        pred[:, 1] -= 148;
        pred[:, 3] -= 148;
        print(f"pad: {pad}")

        return pred, square_img
        # sys.exit(0)

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            print(f"det: {det}")

        # if webcam:  # batch_size >= 1
            # p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
        # else:
            # p, s, im0 = path, '', im0s

        # save_path = str(Path(out) / Path(p).name)
        # s += '%gx%g ' % img.shape[2:]  # print string
        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            print(pred)
            if det is not None and len(det):
                print("HURRAAAAAAAAAAAY")
                # Rescale boxes from imgsz to im0 size
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], pre_scale_shape).round()

                print(f"det: {det}")
                return pred[0]


if __name__ == '__main__':
    detector = Detector()

    image = np.load("/home/snake/eforce/dv_ros/src/cone_detection/cone_detection/image2.npy")

    print(image.shape)

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import cv2

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    preds, img_new = detector.detect(image)
    print(f"preds: {preds}")

    # print(f"scaled: {scale_coords(img_new.shape, preds, image.shape)}")
    img_new = np.transpose(img_new, (1, 2, 0))
    print(img_new.shape)
    fig, ax = plt.subplots()
    plt.imshow(image_rgb)
    # plt.imshow(img_new)
    for i in range(preds.shape[0]):
        # print(preds[i])
        rect = patches.Rectangle((preds[i][0], preds[i][1]), -preds[i][0] + preds[i][2], -preds[i][1] + preds[i][3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # plt.plot((preds[i][0], preds[i][1]), (preds[i][2], preds[i][3]))
    plt.show()

