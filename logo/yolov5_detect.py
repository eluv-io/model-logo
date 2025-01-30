import numpy as np
import cv2
import torch
from loguru import logger
from .yolo_general import non_max_suppression, letterbox, scale_coords
from .plots import plot_one_box, save_one_box

batch_size = 64

"""
    doing some padding, cropping and resizing job
"""
def load_image(img0, img_size=640, stride=32):
    assert img0 is not None , 'Image Not Found'
    img = letterbox(img0, img_size, stride = stride)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    return img, img0
"""
    input a img_path, give the detection res
    img is required as a BGR, HWC array (like cv2 read img)
    det, cropped_img,
    the input imgs shape must be the same.
"""
@torch.no_grad()
def detect(images, model, imgsz=640, stride=32, conf_thres=0.1, iou_thres=0.4, save=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    # prepare torch tensor 
    if type(images) == list:
        # a list of images
        for image in images:
            assert len(image.shape) == 3
    elif len(images.shape) == 3:
        # a single images
        images = [images]
    # image is a batch of BGR
    # whatever a [B, H, W, C] numpy array or a list of [H, W, C] array
    imgs = []
    imgs0 = []
    res = []
    preds = []
    for img in images:
        img, img0 = load_image(img, imgsz, stride) # will return a GRB, CHW image array
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0 
        imgs.append(img)
        imgs0.append(img0)
    imgs = torch.stack(imgs)
    for i in range(0, len(images), batch_size):
        batch_imgs = imgs[i: min(i+batch_size, len(images))]
        batch_preds = model(batch_imgs, augment=False, visualize=False)[0]
        preds.extend(batch_preds)
    assert len(preds) == len(images)
    for i, pred in enumerate(preds):
        img = imgs[i]
        img0 = imgs0[i]
        h, w, _ = img0.shape
        # NMS  after NMS we got the dataformat xyxy conf cls
        pred = non_max_suppression(pred[None], conf_thres, iou_thres, classes=0, agnostic=False, max_det=50)
        det = pred[0]
        det[:, :4] = scale_coords(img.shape[1:], det[:, :4], img0.shape).round()
        cropped_images = []
        bboxes = []
        idxes = []
        im0 = img0.copy()  # im0 in BGR mode
        imc = im0.copy()  # for save_crop Rescale boxes from img_size to im0 size, imc in BGR mode
        
        for i, (*xyxy, conf, cls) in enumerate(det):
            #if xyxy[0].item() > 3 and xyxy[1].item() > 3 and xyxy[2].item() < w-3 and xyxy[3].item() < h-3:
            if save:
                plot_one_box(xyxy, im0, label=str(conf.cpu().data), line_thickness=2)
            xyxy, crop = save_one_box(xyxy, imc, BGR=True,  save=None)
            cropped_images.append(crop[:, :, ::-1]) # we need to convert the crop from BGR mode to RGB mode
            bboxes.append(xyxy.cpu())
            idxes.append(i)
        if save:
            cv2.imwrite(save, im0)

        res.append( [det[idxes].cpu().numpy(), cropped_images] )
    
    return res

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from .yolo import load_model
    yolo = load_model(fuse=False)
    yolo.eval()
    img_path = "/home/elv-zhounan/elv-ml/brand/v2/test/images/test_img.jpg"
    image = cv2.imread(img_path)
    detect_thres = 0.1
    nms_iou_thres = 0.5
    save="/home/elv-zhounan/elv-ml/brand/v2/test/yolo_res/" + img_path.split("/")[-1]
    res = detect(image, model=yolo, conf_thres=detect_thres, iou_thres=nms_iou_thres, save=save, device=device)
    logger.info(res[0][0])
