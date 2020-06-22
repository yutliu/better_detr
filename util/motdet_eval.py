import torch
import numpy as np
import time
import cv2

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # lists/pytorch to numpy
    tp, conf, pred_cls, target_cls = np.array(tp), np.array(conf), np.array(pred_cls), np.array(target_cls)

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls == c)  # Number of ground truth objects
        n_p = sum(i)  # Number of predicted objects

        if (n_p == 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p)

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=False):
    """
    Returns the IoU of two bounding boxes
    """
    N, M = len(box1), len(box2)
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).view(-1,1).expand(N,M)
    b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).view(1,-1).expand(N,M)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y

@torch.no_grad()
def motdet_evaluate(model, data_loader, iou_thres=0.5, print_interval=10):
    model.eval()
    mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    outputs, mAPs, mR, mP, TP, confidence, pred_class, target_class, jdict = \
        [], [], [], [], [], [], [], [], []
    AP_accum, AP_accum_count = np.zeros(1), np.zeros(1)
    for batch_i, data in enumerate(data_loader):
        seen += 1
        if(batch_i > 100):
            break
        # [batch_size x 3 x H x W]
        imgs, _ = data[0].decompose()
        #dict{'boxes':cxcywh_norm 'labels', size, orig_size}
        targets = data[1][0]
        # img_path = data[2]
        height, width = targets['orig_size'].cpu().numpy().tolist()
        t = time.time()
        output = model(imgs.cuda())
        outputs_class = output['pred_logits'].squeeze()
        outputs_boxes = output['pred_boxes'].squeeze()
        target_boxes = targets['boxes']

        # Compute average precision
        if target_boxes is None:
            # If there are labels but no detections mark as zero AP
            if target_boxes.size(0) != 0:
                mAPs.append(0), mR.append(0), mP.append(0)
            continue

        # If no labels add number of detections as incorrect
        correct = []
        if target_boxes.size(0) == 0:
            # correct.extend([0 for _ in range(len(detections))])
            mAPs.append(0), mR.append(0), mP.append(0)
            continue
        else:
            target_cls = targets['labels']
            # Extract target boxes as (x1, y1, x2, y2)
            target_boxes = xywh2xyxy(target_boxes)
            target_boxes[:, 0] *= width
            target_boxes[:, 2] *= width
            target_boxes[:, 1] *= height
            target_boxes[:, 3] *= height

            outputs_boxes = xywh2xyxy(outputs_boxes)
            outputs_boxes[:, 0] *= width
            outputs_boxes[:, 2] *= width
            outputs_boxes[:, 1] *= height
            outputs_boxes[:, 3] *= height


            # img0 = cv2.imread(str(img_path[0]))
            # img1 = cv2.imread(str(img_path[0]))
            # for t in range(len(target_boxes)):
            #     x1 = target_boxes[t, 0].cpu().numpy()
            #     y1 = target_boxes[t, 1].cpu().numpy()
            #     x2 = target_boxes[t, 2].cpu().numpy()
            #     y2 = target_boxes[t, 3].cpu().numpy()
            #     cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 4)
            # cv2.imwrite('gt.jpg', img0)
            #
            # for t in range(len(outputs_boxes)):
            #     if outputs_class[t, 0] < iou_thres:
            #         continue
            #     x1 = outputs_boxes[t, 0].cpu().numpy()
            #     y1 = outputs_boxes[t, 1].cpu().numpy()
            #     x2 = outputs_boxes[t, 2].cpu().numpy()
            #     y2 = outputs_boxes[t, 3].cpu().numpy()
            #     cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 4)
            # cv2.imwrite('pred.jpg', img1)

            detected = []
            for *pred_bbox, conf in zip(outputs_boxes, outputs_class):
                obj_pred = 0
                pred_bbox = torch.FloatTensor(pred_bbox[0]).view(1, -1)
                # Compute iou with target boxes
                iou = bbox_iou(pred_bbox, target_boxes, x1y1x2y2=True)[0]
                # Extract index of largest overlap
                best_i = np.argmax(iou)
                # If overlap exceeds threshold and classification is correct mark as correct
                if iou[best_i] > iou_thres and obj_pred == int(target_cls[best_i]) and best_i not in detected:
                    correct.append(1)
                    detected.append(best_i)
                else:
                    correct.append(0)

        # Compute Average Precision (AP) per class
        AP, AP_class, R, P = ap_per_class(tp=correct,
                                          conf=outputs_class[:, 0].cpu(),
                                          pred_cls=np.zeros_like(outputs_class[:, 0].cpu()),
                                          target_cls=target_cls)

        # Accumulate AP per class
        AP_accum_count += np.bincount(AP_class, minlength=1)
        AP_accum += np.bincount(AP_class, minlength=1, weights=AP)

        # Compute mean AP across all classes in this image, and append to image list
        mAPs.append(AP.mean())
        mR.append(R.mean())
        mP.append(P.mean())

        # Means of all images
        mean_mAP = np.sum(mAPs) / (AP_accum_count + 1E-16)
        mean_R = np.sum(mR) / (AP_accum_count + 1E-16)
        mean_P = np.sum(mP) / (AP_accum_count + 1E-16)

        if batch_i % print_interval == 0:
            # Print image mAP and running mean mAP
            print(('%11s%11s' + '%11.3g' * 4 + 's') %
                  (seen, 100, mean_P, mean_R, mean_mAP, time.time() - t))
    # Print mAP per class
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))

    print('AP: %-.4f\n\n' % (AP_accum[0] / (AP_accum_count[0] + 1E-16)))

    # Return mAP
    return mean_mAP, mean_R, mean_P