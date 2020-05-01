import onnxruntime
import numpy as np
import cv2


Model_Name = "Smart_V1"   #Smart_V1, Smart_V4
score_threshold= 0.3
iou_threshold = 0.4
H=640
W=640
std_box = np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)

def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """
    if ratios is None:
        ratios = np.array([1, 1, 1])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, 1)).T

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors

def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K * A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors    

def py_cpu_nms(dets, scores, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    #scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def RegressionTransform(anchors,bbox_deltas):
    widths  = anchors[:, :, 2] - anchors[:, :, 0]
    heights = anchors[:, :, 3] - anchors[:, :, 1]
    ctr_x   = anchors[:, :, 0] + 0.5 * widths
    ctr_y   = anchors[:, :, 1] + 0.5 * heights

    # Rescale
    bbox_deltas = bbox_deltas * std_box

    bbox_dx = bbox_deltas[:, :, 0] 
    bbox_dy = bbox_deltas[:, :, 1] 
    bbox_dw = bbox_deltas[:, :, 2]
    bbox_dh = bbox_deltas[:, :, 3]

    # get predicted boxes
    pred_ctr_x = ctr_x + bbox_dx * widths
    pred_ctr_y = ctr_y + bbox_dy * heights
    pred_w     = np.exp(bbox_dw) * widths
    pred_h     = np.exp(bbox_dh) * heights

    pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
    pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
    pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
    pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

    pred_boxes = np.vstack((pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2))
    pred_boxes = np.transpose(pred_boxes, (1,0))

    pred_boxes[:,::2] = np.clip(pred_boxes[:,::2], 0, W)
    pred_boxes[:,1::2] = np.clip(pred_boxes[:,1::2], 0, H)

    return pred_boxes

pyramid_levels = [3, 4, 5]
strides = [2 ** x for x in pyramid_levels]
sizes = [2 ** 4.0, 2 ** 6.0, 2 ** 8.0]
ratios = np.array([1, 1, 1])
scales = np.array([1, 2])

image_shape = (W,H)
image_shape = np.array(image_shape)
image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]

# compute anchors over all pyramid levels
all_anchors = np.zeros((0, 4)).astype(np.float32)

for idx, p in enumerate(pyramid_levels):
    anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
    shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
    all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

anchors = np.expand_dims(all_anchors, axis=0)

mean = (104, 117, 123)
img_ori = cv2.imread('./image.jpg')
scale_x = float(img_ori.shape[1]) / W
scale_y = float(img_ori.shape[0]) / H
img = cv2.resize(img_ori,(W,H)).astype(np.float32)
img -= mean
if Model_Name == "Smart_V4":
    img = img / 255.0
img = np.transpose(img,(2,0,1))
img = np.expand_dims(img, axis=0)

session = onnxruntime.InferenceSession("./models/"+Model_Name+".onnx")
prediction = session.run(None, {"input0":img})

classifications = prediction[0]
bbox_regressions = prediction[1]
bboxes = RegressionTransform(anchors, bbox_regressions)
classification = classifications[0,:,:]

scores = classification[:,1]
scores_indice = scores>score_threshold
positive_indices = scores_indice
scores = scores[positive_indices]
bbox = bboxes[positive_indices]

# keep top-K before NMS
order = scores.argsort()[::-1][:5000]
bbox = bbox[order]
scores = scores[order]

keep = py_cpu_nms(bbox, scores, iou_threshold)
keep_boxes = bbox[keep]
keep_boxes[:,::2] *= scale_x
keep_boxes[:,1::2] *= scale_y
keep_scores = scores[keep]
keep_scores = np.expand_dims(keep_scores, axis=1)
keep_boxes = keep_boxes.astype('int')
print(len(keep_boxes))

font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(len(keep_scores)):
    cv2.rectangle(img_ori,(keep_boxes[i,0],keep_boxes[i,1]),(keep_boxes[i,2],keep_boxes[i,3]),(0,0,255),thickness=2)
    cv2.putText(img_ori, text=str(keep_scores[i]), org=(keep_boxes[i,0],keep_boxes[i,1]), fontFace=font, fontScale=0.2,
                            thickness=1, lineType=cv2.LINE_AA, color=(255, 255, 255))
cv2.imwrite('./output.jpg',img_ori)
