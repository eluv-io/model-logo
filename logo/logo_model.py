import os
import time
import json
from typing import List

import yaml
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F

from .yolov5_detect import detect as yolo_detect
from .yolo import load_model
from .resnext_with_AMSoftmaxLoss import ResNeXt_with_AMSoftmaxLoss
from .utils import logger

from dataclasses import dataclass
from typing import Union

from common_ml.model import FrameModel, FrameTag
from common_ml.types import Data
from config import config

import pathlib
pathlib.Path(__file__).parent.resolve()

INPUT_SIZE = (224, 224)
BATCH_SIZE = 64

class DictToClass:
    def __init__(self, dict):
        for k in dict:
            setattr(self, k, dict[k])

@dataclass
class RuntimeConfig(Data):
    fps: float
    allow_single_frame: bool
    
    @staticmethod
    def from_dict(data: dict) -> 'RuntimeConfig':
        return RuntimeConfig(**data)

class LogoRecognition(FrameModel):
    def __init__(self,
                 model_input_path,
                 runtime_config: Union[dict, RuntimeConfig]
                 ):
        if isinstance(runtime_config, dict):
            self.config = RuntimeConfig.from_dict(runtime_config)
        else:
            self.config = runtime_config
    
        self.model_input_path = model_input_path

        # specify device for torch
        self.device = 'cuda'
        # get args from command line to config this inference class
        self.args = self._add_args()
        # check value type
        if self.args.inference_type not in {'c', 'r'}:
            raise ValueError("Wrong inference_type!")
        if self.args.input_type not in {'image', 'video'}:
            raise ValueError("Wrong input_type!")
        # load feature pool
        with open(self.args.r_labels, 'r') as f:
            self.r_labels = json.load(f)
        with open(self.args.r_classes, 'r') as f:
            self.r_classes = json.load(f)
        with open(self.args.r_imagenames, 'r') as f:
            self.r_imagenames = json.load(f)
        self.r_feature_pool = np.load(self.args.r_feature_pool)
        self.r_feature_pool = torch.from_numpy(
            self.r_feature_pool).to(self.device)
        logger.info("---feature pool loaded!---")
        # load yolo
        self.yolo_cfg = self.args.yolo_cfg
        self.yolo_weights = self.args.yolo_weights
        self.yolo = self.load_yolo(self.yolo_cfg, self.yolo_weights)
        self.yolo.eval()
        logger.info("---YOLO model loaded!---")
        # load resnext
        self.num_classes = self.args.num_classes
        self.resnext_weights = self.args.resnext_weights
        self.resnext= self.load_resnext(self.resnext_weights, self.num_classes)
        self.resnext.eval()
        logger.info("---ResNeXt model loaded!---")
        # load the logo label map for retrieval and classification
        self.r_logo_label_map = {}
        self.r_label_logo_map = {}
        for i, logo in enumerate(self.r_classes):
            self.r_logo_label_map[logo] = i
            self.r_label_logo_map[i] = logo
        with open(self.args.c_logo_label_map) as f:
            self.c_logo_label_map = json.load(f)
        self.c_label_logo_map = {v: k for k,
                                 v in self.c_logo_label_map.items()}
        self.maskout = set(json.load(open(self.args.mask_file)))
        # config all kinds of the thres
        self.nms_iou_thres = self.args.nms_iou_thres
        self.c_thres = self.args.classify_thres
        self.eval_iou_thres = self.args.eval_iou_thres
        self.detect_thres = self.args.detect_thres
        self.r_thres = self.args.retrieval_thres
        # tag res
        self.tags = []

    def _add_args(self):
        args = DictToClass(config["inference"])

        data_path = self.model_input_path
        args.yolo_weights = os.path.join(data_path, args.yolo_weights)
        args.yolo_cfg = os.path.join(data_path, args.yolo_cfg)
        args.resnext_weights = os.path.join(
            data_path, args.resnext_weights)
        args.c_logo_label_map = os.path.join(data_path, args.c_logo_label_map)
        args.r_feature_pool = os.path.join(data_path, args.r_feature_pool)
        args.r_labels = os.path.join(data_path, args.r_labels)
        args.r_classes = os.path.join(data_path, args.r_classes)
        args.r_imagenames = os.path.join(data_path, args.r_imagenames)
        args.mask_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), args.mask_file)
        return args

    def load_resnext(self, resnext_weights_path, num_classes):
        net = ResNeXt_with_AMSoftmaxLoss(num_classes=num_classes)
        ckpt = torch.load(resnext_weights_path, map_location=self.device)
        net.load_state_dict(ckpt['net_state_dict'])
        net = net.to(self.device)
        net.eval()
        return net

    def load_yolo(self, yolo_cfg, yolo_weights_path):
        yolo = load_model(cfg=yolo_cfg, weights=yolo_weights_path, device=self.device)
        return yolo

    """
        calling the detect function in yolov5_detect to get the detect res and the cropped images
        the det res and the cropped images are on cpu
    """
    @torch.no_grad()
    def yolo_detect(self, image, detect_thres, nms_iou_thres, save=None):
        res = yolo_detect(
            image,
            model=self.yolo,
            conf_thres=self.detect_thres,
            iou_thres=nms_iou_thres,
            save=save,
            device=self.device)
        return res

    """
        given a list of cropped images, return the features.
        the images should be in RGB mode
    """
    @torch.no_grad()
    def feature_extractor(self, cropped_list):
        imgs = torch.zeros(len(cropped_list), 3, INPUT_SIZE[0], INPUT_SIZE[1])
        for i, img in enumerate(cropped_list):
            img = Image.fromarray(img, mode="RGB")
            img = transforms.Resize(INPUT_SIZE, Image.BILINEAR)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [
                                       0.229, 0.224, 0.225])(img)
            imgs[i] = img[...]
        with torch.no_grad():
            imgs = imgs.to(self.device)
            placeholder = torch.zeros(len(cropped_list), dtype=int).to(self.device)
            feature = self.resnext(imgs, placeholder, embed=True)
        features = feature / torch.norm(feature, dim=1, keepdim=True)
        return features

    """
        predict multiple images using retrieval. 
        a RGB, HWC images list must be given
        return rows, scores, logo_names
    """
    @torch.no_grad()
    def resnext_r(self, imgs):
        features = self.feature_extractor(imgs)
        logger.info(features.shape)
        rows, scores = self.cos_similarity(features)
        return rows, scores, [self.r_classes[self.r_labels[row]] for row in rows]

    """
        predict multiple images using classification. 
        a RGB, HWC images list must be given
        return logo_probs, logo_labels, logo_names
    """
    @torch.no_grad()
    def resnext_c(self, imgs):
        images = torch.zeros(len(imgs), 3, self.INPUT_SIZE[0], self.INPUT_SIZE[1])
        for i, img in enumerate(imgs):
            img = Image.fromarray(img, mode="RGB")
            img = transforms.Resize(self.INPUT_SIZE, Image.BILINEAR)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            images[i] = img[...]
        with torch.no_grad():
            images = images.to(self.device)
            placeholder = torch.zeros(len(imgs), dtype=int).to(self.device)
            _, preds, scores = self.resnext(images, placeholder)
        preds = preds.tolist()
        scores = scores.tolist()
        logo_names = []
        for i in range(len(imgs)):
            logo_names.append(self.c_label_logo_map[preds[i]])
            
        return scores, preds, logo_names

    def yolo_filter(self, frame, bbox_list, detect_thres, h, w, min_logo_size=(10, 10)):
        candidates = []
        new_bbox_list = []
        for xmin, ymin, xmax, ymax, num, score in bbox_list:
            # do not consider tiny logos
            if xmax-xmin > min_logo_size[1] and ymax-ymin > min_logo_size[0] and score > detect_thres:
                # enlarge each bbox predicted by YOLO
                enlarge_w = (xmax - xmin) * \
                    (self.args.enlarge_bbox_ratio - 1) / 2
                enlarge_h = (ymax - ymin) * \
                    (self.args.enlarge_bbox_ratio - 1) / 2
                xmin = max(0, int(xmin - enlarge_w))
                ymin = max(0, int(ymin - enlarge_h))
                xmax = min(w, int(xmax + enlarge_w))
                ymax = min(h, int(ymax + enlarge_h))
                candidates.append(frame[ymin:ymax, xmin:xmax])
                new_bbox_list.append((xmin, ymin, xmax, ymax, num, score))
        return candidates, new_bbox_list

    def cos_similarity(self, features_cand):

        cos_sim = torch.mm(self.r_feature_pool,
                           torch.transpose(features_cand, 0, 1))
        scores, rows = torch.max(cos_sim, 0)
        return rows.cpu().numpy(), np.round(scores.cpu().numpy(), 3)
    
    def tag(self, frame: np.ndarray) -> List[FrameTag]:
        # convert from rgb to bgr
        frame = frame[:, :, ::-1]
        frame = np.expand_dims(frame, axis=0)
        
        batch_yolo_preds = self.yolo_detect(frame, self.detect_thres, self.nms_iou_thres)
        batch_bboxes = []
        batch_cropped_images = []
        batch_idxes = []
        for j, [bboxes, cropped_images] in enumerate(batch_yolo_preds):
            batch_idxes.extend([j] * len(cropped_images))
            batch_bboxes.extend(bboxes)
            batch_cropped_images.extend(cropped_images)
        assert len(batch_idxes) == len(batch_bboxes)
        assert len(batch_bboxes) == len(batch_cropped_images)
        if len(batch_cropped_images) == 0:
            return []
        if self.args.inference_type == 'r':
            _, batch_scores, batch_logo_names = self.resnext_r(batch_cropped_images)
        else:
            batch_scores, _, batch_logo_names = self.resnext_c(batch_cropped_images)

        assert len(batch_scores) == len(batch_bboxes)

        res = []
        for idx in range(len(batch_scores)):
            score = batch_scores[idx]
            name = batch_logo_names[idx]
            bbox = batch_bboxes[idx]
            if score > self.r_thres and name not in self.maskout:
                h = frame.shape[1]
                w = frame.shape[2]
                curr_tag = {'text': name,
                            'confidence': float(score),
                            'box': {'x1': float(round(bbox[0]/w, 4)),
                                    'y1': float(round(bbox[1]/h, 4)),
                                    'x2': float(round(bbox[2]/w, 4)),
                                    'y2': float(round(bbox[3]/h, 4))
                                    }
                            }
                res.append(FrameTag.from_dict(curr_tag))
                
        return res