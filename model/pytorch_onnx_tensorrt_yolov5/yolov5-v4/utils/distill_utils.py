import torch
import torch.nn as nn
# from utils.loss import build_targets

def wh_iou_cfg(box1, box2):
    # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
    box2 = box2.t()

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area  # iou

def build_targets_cfg(model, targets):
    # targets = [image, class, x, y, w, h]

    nt = len(targets)
    tcls, tbox, indices, av = [], [], [], []
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    for i in model.yolo_layers:
        # get number of grid points and anchor vec for this yolo layer
        if multi_gpu:
            ng, anchor_vec = model.module.module_list[i].ng, model.module.module_list[i].anchor_vec
        else:
            ng, anchor_vec = model.module_list[i].ng, model.module_list[i].anchor_vec

        # iou of targets-anchors
        t, a = targets, []
        gwh = t[:, 4:6] * ng
        if nt:
            iou = torch.stack([wh_iou_cfg(x, gwh) for x in anchor_vec], 0)

            use_best_anchor = False
            if use_best_anchor:
                iou, a = iou.max(0)  # best iou and anchor
            else:  # use all anchors
                na = len(anchor_vec)  # number of anchors
                a = torch.arange(na).view((-1, 1)).repeat([1, nt]).view(-1)
                t = targets.repeat([na, 1])
                gwh = gwh.repeat([na, 1])
                iou = iou.view(-1)  # use all ious

            # reject anchors below iou_thres (OPTIONAL, increases P, lowers R)
            reject = True
            if reject:
                j = iou > model.hyp['iou_t']  # iou threshold hyperparameter
                t, a, gwh = t[j], a[j], gwh[j]

        # Indices
        b, c = t[:, :2].long().t()  # target image, class
        gxy = t[:, 2:4] * ng  # grid x, y
        gi, gj = gxy.long().t()  # grid x, y indices
        indices.append((b, a, gj, gi))

        # GIoU
        gxy -= gxy.floor()  # xy
        tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)
        av.append(anchor_vec[a])  # anchor vec

        # Class
        tcls.append(c)
        if c.shape[0]:  # if any targets
            assert c.max() <= model.nc, 'Target classes exceed model classes'

    return tcls, tbox, indices, av

def distillation_loss1(output_s, output_t, num_classes, batch_size):
    T = 3.0
    Lambda_ST = 0.001
    criterion_st = torch.nn.KLDivLoss(reduction='sum')
    output_s = torch.cat([i.view(-1, num_classes + 5) for i in output_s])
    output_t = torch.cat([i.view(-1, num_classes + 5) for i in output_t])
    loss_st  = criterion_st(nn.functional.log_softmax(output_s/T, dim=1), nn.functional.softmax(output_t/T,dim=1))* (T*T) / batch_size
    return loss_st * Lambda_ST



def distillation_loss2(model, targets, output_s, output_t):
    reg_m = 0.0
    T = 3.0
    Lambda_cls, Lambda_box = 0.0001, 0.001

    criterion_st = torch.nn.KLDivLoss(reduction='sum')
    ft = torch.cuda.FloatTensor if output_s[0].is_cuda else torch.Tensor
    lcls, lbox = ft([0]), ft([0])

    tcls, tbox, indices, anchor_vec = build_targets_cfg(model,targets)
    reg_ratio, reg_num, reg_nb = 0, 0, 0
    for i, (ps, pt) in enumerate(zip(output_s, output_t)):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        ps=ps.float()
        nb = len(b)
        if nb:  # number of targets
            pss = ps[b, a, gj, gi]  # prediction subset corresponding to targets
            pts = pt[b, a, gj, gi]

            psxy = torch.sigmoid(pss[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            psbox = torch.cat((psxy, torch.exp(pss[:, 2:4]) * anchor_vec[i]), 1).view(-1, 4)  # predicted box

            ptxy = torch.sigmoid(pts[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            ptbox = torch.cat((ptxy, torch.exp(pts[:, 2:4]) * anchor_vec[i]), 1).view(-1, 4)  # predicted box


            l2_dis_s = (psbox - tbox[i]).pow(2).sum(1)
            l2_dis_s_m = l2_dis_s + reg_m
            l2_dis_t = (ptbox - tbox[i]).pow(2).sum(1)
            l2_num = l2_dis_s_m > l2_dis_t
            lbox += l2_dis_s[l2_num].sum()
            reg_num += l2_num.sum().item()
            reg_nb += nb

        output_s_i = ps[..., 4:].view(-1, model.nc + 1)
        output_t_i = pt[..., 4:].view(-1, model.nc + 1)
        lcls += criterion_st(nn.functional.log_softmax(output_s_i/T, dim=1), nn.functional.softmax(output_t_i/T,dim=1))* (T*T) / ps.size(0)

    if reg_nb:
        reg_ratio = reg_num / reg_nb

    return lcls * Lambda_cls + lbox * Lambda_box, reg_ratio