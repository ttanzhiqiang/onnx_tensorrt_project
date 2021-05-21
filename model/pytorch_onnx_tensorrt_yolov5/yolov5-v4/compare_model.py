from modelsori import *
from utils.general import *
import numpy as np
from copy import deepcopy
from test import test
from terminaltables import AsciiTable
import time
from utils.prune_utils import *
import argparse

from models.yolo import Model

import torchvision

def letterboxv5(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        # dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def non_max_suppressionv5(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def normalize_coordinate(box1,box2):
    x1, y1, x2, y2 = box1
    p1, q1, p2, q2 = box2
    min_x = min(x1, x2, p1, p2)
    max_x = max(x1, x2, p1, p2)
    min_y = min(y1, y2, q1, q2)
    max_y = max(y1, y2, q1, q2)
    norm_x1 = (x1 - min_x) / (max_x - min_x)
    norm_x2 = (x2 - min_x) / (max_x - min_x)
    norm_p1 = (p1 - min_x) / (max_x - min_x)
    norm_p2 = (p2 - min_x) / (max_x - min_x)
    norm_y1 = (y1 - min_y) / (max_y - min_y)
    norm_y2 = (y2 - min_y) / (max_y - min_y)
    norm_q1 = (q1 - min_y) / (max_y - min_y)
    norm_q2 = (q2 - min_y) / (max_y - min_y)
    return [norm_x1,norm_y1,norm_x2,norm_y2],[norm_p1,norm_q1,norm_p2,norm_q2]

def proximity_box(box1,box2):
    x1, y1, x2, y2 = box1
    p1, q1, p2, q2 = box2
    P=abs(x1-p1)+abs(y1-q1)+abs(x2-p2)+abs(y2-q2)
    return P

def non_max_suppression_confluence(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        Md = 0.6
        bf=confluenceV1(x,nc,Md)

        output[xi] = x[bf]

        # if (time.time() - t) > time_limit:
        #     break  # time limit exceeded

    return output

def confluence(det,class_num,Md):
    index = np.arange(0, len(det), 1).reshape(-1, 1)
    infos_tmp = torch.cat((det, torch.from_numpy(index).to(device)), 1)
    bf = []
    # 对所有类别遍历
    for box_class in range(class_num):  # line 2
        # 对单个类别的所有框遍历
        infos = infos_tmp[infos_tmp[:, 5] == box_class].cpu().detach().numpy()
        while len(infos):  # line 3
            optimalconfluence = 416 * 320  # imagesize line 5
            # the best id must be exist,because the optimalconfluence is so much big
            best_id = None
            for box_id, single_box in enumerate(infos):
                if len(infos)==1:
                    best_id=0
                    break
                confluence_bi = {}
                box1 = single_box[:4]
                for another_box_id, another_single_box in enumerate(infos):
                    if box_id == another_box_id:
                        continue
                    box2 = another_single_box[:4]
                    norm_box1, norm_box2 = normalize_coordinate(box1, box2)
                    P = proximity_box(norm_box1, norm_box2)
                    if P < 2:
                        confluence_bi[another_box_id] = P / single_box[4]

                if len(confluence_bi) == 0:
                    confluence_bi_min = 0
                else:
                    res_min = min(confluence_bi, key=lambda x: confluence_bi[x])
                    confluence_bi_min = confluence_bi[res_min]
                if confluence_bi_min < optimalconfluence:
                    optimalconfluence = confluence_bi_min
                    best_id = box_id

            bf.append(infos[best_id][-1])
            index_det = []
            box2 = infos[best_id][:4]
            for box_id, single_box in enumerate(infos):
                box1 = single_box[:4]
                norm_box1, norm_box2 = normalize_coordinate(box1, box2)
                P = proximity_box(norm_box1, norm_box2)
                if P < Md:
                    index_det.append(box_id)
            index_save = [i for i in range(infos.shape[0]) if i not in index_det]
            infos = infos[index_save]
    return bf

def confluenceV1(det,class_num,Md):
    index = np.arange(0, len(det), 1).reshape(-1, 1)
    infos_tmp = torch.cat((det, torch.from_numpy(index).to(device)), 1)
    bf = []
    # 对所有类别遍历
    for box_class in range(class_num):  # line 2
        # 对单个类别的所有框遍历
        infos = infos_tmp[infos_tmp[:, 5] == box_class].cpu().detach().numpy()
        while len(infos):  # line 3
            optimalconfluence = 416 * 320  # imagesize line 5
            # the best id must be exist,because the optimalconfluence is so much big
            best_id = None
            total_P=[]
            for box_id, single_box in enumerate(infos):
                if len(infos)==1:
                    best_id=0
                    break
                box1 = single_box[:4]

                box1 = np.tile(box1, (len(infos) - 1, 1))
                index_other=[i for i in range(len(infos)) if i != box_id]
                box2=infos[index_other,:4]
                box_total=np.concatenate((box1,box2),1)
                min_x=box_total[:,[0,2,4,6]].min(1).reshape(-1,1)
                max_x=box_total[:,[0,2,4,6]].max(1).reshape(-1,1)
                min_y=box_total[:,[1,3,5,7]].min(1).reshape(-1,1)
                max_y=box_total[:,[1,3,5,7]].max(1).reshape(-1,1)
                norm_box1=box1
                norm_box1[:,[0,2]]=(norm_box1[:,[0,2]]-min_x)/(max_x-min_x)
                norm_box1[:, [1, 3]] = (norm_box1[:, [1, 3]] - min_y) / (max_y - min_y)
                norm_box2 = box2
                norm_box2[:, [0, 2]] = (norm_box2[:, [0, 2]] - min_x) / (max_x - min_x)
                norm_box2[:, [1, 3]] = (norm_box2[:, [1, 3]] - min_y) / (max_y - min_y)
                x1=norm_box1[:,0]
                p1=norm_box2[:,0]
                x2=norm_box1[:,2]
                p2=norm_box2[:,2]
                y1=norm_box1[:,1]
                q1=norm_box2[:,1]
                y2=norm_box1[:,3]
                q2=norm_box2[:,3]
                P = abs(x1 - p1) + abs(y1 - q1) + abs(x2 - p2) + abs(y2 - q2)
                total_P.append(P)
                confluence_bi=P/single_box[4]
                confluence_bi=confluence_bi[P < 2]

                if len(confluence_bi)==0:
                    confluence_bi_min=0
                else:
                    confluence_bi_min=confluence_bi.min()

                if confluence_bi_min < optimalconfluence:
                    optimalconfluence = confluence_bi_min
                    best_id = box_id

            bf.append(infos[best_id][-1])
            if len(total_P)>0:
                best_p=total_P[best_id]
                index_del=np.where(best_p < Md)[0]
                #total_P is not include the best_id ,so when index is biger than best id ,it needs to add one place.
                index_del = [i if i < best_id else i + 1 for i in index_del]
            else:
                index_del=[]

            index_save = [j for j in range(len(infos)) if (j != best_id and j not in index_del)]
            infos = infos[index_save]
    return bf


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def copy_conv(conv_src,conv_dst):
    conv_dst.conv=conv_src[0]
    conv_dst.bn=conv_src[1]
    conv_dst.act=conv_src[2]

def copy_pt(modelyolov4tiny,model):
    conv1 = list(modelyolov4tiny.model.children())[0]
    copy_conv(model.module_list[0],conv1)
    conv2 = list(modelyolov4tiny.model.children())[1]
    copy_conv(model.module_list[1],conv2)
    conv3 = list(modelyolov4tiny.model.children())[2]
    copy_conv(model.module_list[2], conv3)
    ctiny1 = list(modelyolov4tiny.model.children())[3]
    copy_conv(model.module_list[4], ctiny1.cv1)
    copy_conv(model.module_list[5], ctiny1.cv2)
    copy_conv(model.module_list[7], ctiny1.cv3)
    conv4 = list(modelyolov4tiny.model.children())[6]
    copy_conv(model.module_list[10], conv4)
    ctiny2 = list(modelyolov4tiny.model.children())[7]
    copy_conv(model.module_list[12], ctiny2.cv1)
    copy_conv(model.module_list[13], ctiny2.cv2)
    copy_conv(model.module_list[15], ctiny2.cv3)
    conv5 = list(modelyolov4tiny.model.children())[10]
    copy_conv(model.module_list[18], conv5)
    ctiny3 = list(modelyolov4tiny.model.children())[11]
    copy_conv(model.module_list[20], ctiny3.cv1)
    copy_conv(model.module_list[21], ctiny3.cv2)
    copy_conv(model.module_list[23], ctiny3.cv3)
    conv6 = list(modelyolov4tiny.model.children())[14]
    copy_conv(model.module_list[26], conv6)
    conv7 = list(modelyolov4tiny.model.children())[15]
    copy_conv(model.module_list[27], conv7)
    conv8 = list(modelyolov4tiny.model.children())[16]
    copy_conv(model.module_list[28], conv8)
    conv9 = list(modelyolov4tiny.model.children())[17]
    copy_conv(model.module_list[32], conv9)
    upsample1 = list(modelyolov4tiny.model.children())[18]
    upsample1=model.module_list[33]
    conv10 = list(modelyolov4tiny.model.children())[20]
    copy_conv(model.module_list[35], conv10)
    detect = list(modelyolov4tiny.model.children())[21]
    detect.m[1]=model.module_list[29][0]
    detect.m[0] = model.module_list[36][0]

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def test_time():
    img_size=416
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # modelyolov4tiny = Model('models/yolov5s_ghostnet.yaml', nc=80).to(device)  #0.0062
    # modelyolov4tiny = Model('models/yolov4-tiny.yaml', nc=80).to(device)  #0.0029
    # modelyolov4tiny = Model('models/yolov5s.yaml', nc=80).to(device)  #0.0077
    modelyolov4tiny = Model('models/hub/yolov3-tiny.yaml', nc=80).to(device)  #0.0026
    # modelyolov4tiny = Darknet('cfg/yolov4-tiny.cfg', (img_size, img_size)) #0.0035
    # modelyolov4tiny = Darknet('cfg/yolov5s_v4.cfg', (img_size, img_size))  # 0.0081
    # modelyolov4tiny = Darknet('cfg/prune_0.5_keep_0.01_8x_yolov5s_v4.cfg', (img_size, img_size))  # 0.0081
    # initialize_weights(modelyolov4tiny)
    # modelyolov4tiny.module_list.to(device)

    # ckpt = {'epoch': -1,
    #         'model': modelyolov4tiny,
    #         'optimizer': None ,
    #         'wandb_id': None
    #         }
    # torch.save(ckpt, "yolov5s.pt")

    model = modelyolov4tiny

    random_input = torch.rand((1, 3, img_size, img_size)).to(device)

    def obtain_avg_forward_time(input, model, repeat=200):
        # model.to('cpu').fuse()
        # model.module_list.to(device)
        model.eval()
        start = time.time()
        with torch.no_grad():
            for i in range(repeat):
                output = model(input)[0]
        avg_infer_time = (time.time() - start) / repeat

        return avg_infer_time, output

    print('testing inference time...')
    pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, model)
    print(f'pruned_forward_time:{pruned_forward_time:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov4-tiny.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/fangweisui.data', help='*.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov4-tiny.weights', help='sparse model weights')
    parser.add_argument('--percent', type=float, default=0.6, help='channel prune percent')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)

    if 1:

        img_size = opt.img_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #the way of loading yolov5s
        # ckpt = torch.load('last_s.pt', map_location=device)  # load checkpoint
        # modelyolov5 = Model('cfg/yolov5s.yaml', nc=2).to(device)
        # exclude = ['anchor']  # exclude keys
        # ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
        #                  if k in modelyolov5.state_dict() and not any(x in k for x in exclude)
        #                  and modelyolov5.state_dict()[k].shape == v.shape}
        # modelyolov5.load_state_dict(ckpt['model'], strict=False)

        #another way of loading yolov5s
        # modelyolov5=torch.load(opt.weights, map_location=device)['model'].float().eval()
        # modelyolov5.model[24].export = False  # onnx export

        # model=modelyolov5

        #load prune model
        model_prune = Darknet(opt.cfg, (img_size, img_size))
        initialize_weights(model_prune)
        load_darknet_weights(model_prune, opt.weights)
        # model_prune.load_state_dict(torch.load(opt.weights)['model'].state_dict())
        # # model_prune.fuse()  #fuse
        model_prune.module_list.to(device)

        modelyolov4tiny = Model('models/yolov4-tiny.yaml', nc=80).to(device)
        copy_pt(modelyolov4tiny, model_prune)

        # model = modelyolov4tiny
        model=model_prune

        # ckpt = {'epoch': -1,
        #         'model': modelyolov4tiny,
        #         'optimizer': None ,
        #         'wandb_id': None
        #         }
        # torch.save(ckpt, "yolov4_tiny.pt")


        #load prune finetune model
        # model_prune=torch.load('last_prune.pt')['model'].float().eval()

        #load yolov5s from cfg
        # model = Darknet(opt.cfg, (img_size, img_size)).to(device)
        # copy_weight(modelyolov5,model)

        path='data/samples/bus.jpg'
        img0 = cv2.imread(path)  # BGR
        # Padded resize
        img = letterboxv5(img0, new_shape=416)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # modelyolov5.eval()


        model.eval()
        pred = model(img)[0]



        # model_prune.eval()
        # pred = model_prune(img)[0]

        pred = non_max_suppression_confluence(pred, 0.1, 0.5, classes=None,
                                   agnostic=False)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (str(int(cls)), conf)
                    plot_one_box(xyxy, img0, label=label, color=[random.randint(0, 255) for _ in range(3)], line_thickness=3)
                cv2.imwrite("v5.jpg", img0)
    else:
        test_time()



