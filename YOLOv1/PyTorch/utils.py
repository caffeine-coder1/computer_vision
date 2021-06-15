import torch
import numpy as np
from torch.functional import Tensor
from general import check_type
from timeit import timeit


def check_shape(t, last_dim_shape):
    if check_type(t, [torch.Tensor, np.ndarray]) and check_type(last_dim_shape, [int]):
        assert t.shape[-1] == last_dim_shape, (
            f'tensor last dimension be {last_dim_shape},' +
            f'tensor last dimension is {t.shape[-1]}')
    else:
        raise TypeError(
            f'the tensor type is {type(t)}. accepted types are torch.Tensor and numpy.ndarray')


def check_dim(t, dim):
    if check_type(t, [torch.Tensor, np.ndarray]) and check_type(dim, [int]):
        assert t.dim(
        ) == dim, f'tensor should be {dim} dimensional, tensor dimension is {t.dim()}'
    else:
        raise TypeError(
            f'the tensor type is {type(t)}. accepted types are torch.Tensor and numpy.ndarray')


def midpoint_to_corner(box):
    # [N,x,y,w,h]
    x1 = box[..., 0:1]-(box[..., 2:3]/2)
    y1 = box[..., 1:2]-(box[..., 3:]/2)
    x2 = box[..., 0:1]+(box[..., 2:3]/2)
    y2 = box[..., 1:2]+(box[..., 3:]/2)
    return torch.cat([x1, y1, x2, y2], dim=-1)


def box_area(b):
    return (b[..., 2:3]-b[..., 0:1]) * (b[..., 3:]-b[..., 1:2])


def calculate_intersection(b1, b2):
    x1 = torch.max(b1[..., 0:1], b2[..., 0:1])
    y1 = torch.max(b1[..., 1:2], b2[..., 1:2])
    x2 = torch.min(b1[..., 2:3], b2[..., 2:3])
    y2 = torch.min(b1[..., 3:], b2[..., 3:])

    return (x2-x1).clamp(0) * (y2-y1).clamp(0)


def iou(b1, b2, f='c'):
    """calculate the `IOU` of 2 boxes namely b1 and b2.

        Args:
            b1: first box
            b2: second box
            f: box format. `c` for corner, `m` for midpoint.

        both b1 and b2 should be of type `torch.Tensor` or `numpy.ndarray`.
    """

# ~~~~~~~~~~~~~~~~~~~~~ type checking ~~~~~~~~~~~~~~~~~~~~~ #
    known_types = [torch.Tensor, np.ndarray]
    if not check_type(b1, known_types):
        raise TypeError(
            f'the b1 has a type of {type(b1)}. accepted types are torch.Tensor and numpy.ndarray')

    if not check_type(b2, known_types):
        raise TypeError(
            f'the b1 has a type of {type(b2)}. accepted types are torch.Tensor and numpy.ndarray')

    if isinstance(b1, np.ndarray):
        b1 = torch.from_numpy(b1)
    if isinstance(b2, np.ndarray):
        b2 = torch.from_numpy(b2)

# ~~~~~~~~~~~~~~~~~~~~~ coordinate conversion ~~~~~~~~~~~~~~~~~~~~~ #

    if f == 'm':
        b1 = midpoint_to_corner(b1)
        b2 = midpoint_to_corner(b2)

# ~~~~~~~~~~~~~~~~~~~~~ IOU calculation ~~~~~~~~~~~~~~~~~~~~~ #

    b1_area = box_area(b1)
    b2_area = box_area(b2)
    intersection = calculate_intersection(b1, b2)
    return intersection/(b1_area+b2_area-intersection+1e-6)


# ~~~~~~~~~~~~~~~~~~~~~ Non max supression ~~~~~~~~~~~~~~~~~~~~~ #

def nms(boxes, iou_threshold, prob_threshold):
    # input format [[class,probability,x1,y1,x2,y2]] shape is (N,6)

    # ~~~~~~~~~~~~~~~~~~~~~ type checking ~~~~~~~~~~~~~~~~~~~~~ #
    known_types = [torch.Tensor, np.ndarray]
    if not check_type(boxes, known_types):
        raise TypeError(
            f'the b1 has a type of {type(boxes)}. accepted types are torch.Tensor and numpy.ndarray')
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)

    check_shape(boxes, 6)
    # ~~~~~~~~~~~~~~~~~~~~~ nms ~~~~~~~~~~~~~~~~~~~~~ #
    # remove all the boxes below prob_threshold
    boxes = boxes[boxes[..., 1] > prob_threshold]
    # get the all unique classes
    classes = torch.unique(boxes[..., 0])
    check_dim(classes, 1)
    bounding_boxes = []
    # for each classes calculate nms
    for c in classes:

        b = boxes[boxes[..., 0] == c]
        check_dim(b, 2)
        if b.shape[0] > 1:

            # sort the boxes based on the probability
            torch.sort(b, 0)

            # taking the max box out
            max_box = b[0].unsqueeze(0)
            b = b[1:]
            check_dim(max_box, 2)
            bounding_boxes.append(max_box)

            # calculate the iou
            ious = iou(max_box[..., 2:], b[..., 2:]).squeeze(-1)
            check_dim(ious, 1)

            # apply nms
            b = b[ious > iou_threshold]
            if b.shape[0] != 0:
                bounding_boxes.append(b)

        else:
            bounding_boxes.append(b)
    bounding_boxes = torch.cat(bounding_boxes, dim=0)
    check_dim(bounding_boxes, 2)
    check_shape(bounding_boxes, 6)
    return bounding_boxes


if __name__ == '__main__':
    pass
