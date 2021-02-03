# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import inspect
import numpy as np
from numpy import random
import cv2

import vedacore.image as image
from vedacore.misc import is_list_of, is_str, registry
from vedadet.misc.bbox import bbox_overlaps

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


@registry.register_module('pipeline')
class OverlapVideoCrop(object):
    def __init__(self,
                 window_size=256,
                 overlap_ratio=0.5,
                 test_mode=False):
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.test_mode = test_mode
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }

    def __call__(self, results):

        if 'window_size' not in results:
            pass
        else:
            self.window_size = results['window_size']

        imgs = results['img']
        duration = results['img_info']['duration']

        stride = int(self.window_size * (1 - self.overlap_ratio))
        time = max(0, int(np.ceil(
            (duration - self.window_size) / stride))) + 1

        if self.test_mode:
            crop_imgs = []
            for i in range(time):
                crop_img = imgs[i * stride: i * stride + self.window_size]
                crop_imgs.append(crop_img)

            results['img'] = np.concatenate(crop_imgs, axis=0)

            return results

        else:
            while True:
                # print('=='* 40)
                c = np.random.randint(time)
                patch = np.array([c * stride, min(duration, c * stride + self.window_size)])
                # print(c, patch, duration)

                flag = False
                for key in results.get('bbox_fields', []):
                    boxes = results[key].copy()
                    boxes[:, 0] = boxes[:, 0].clip(min=patch[0])
                    boxes[:, 1] = boxes[:, 1].clip(max=patch[1])
                    boxes -= patch[0]
                    mask = boxes[:, 1] > boxes[:, 0]

                    if key == 'gt_bboxes' and sum(mask) == 0:
                        flag = True
                        break

                    results[key] = boxes[mask]
                    # labels
                    label_key = self.bbox2label.get(key)
                    if label_key in results:
                        results[label_key] = results[label_key][mask]

                if flag:
                    continue

                # adjust the img no matter whether the gt is empty before crop
                imgs = imgs[patch[0]:patch[1]]
                results['img'] = imgs
                results['img_shape'] = imgs.shape

                # print(imgs.shape)
                # import cv2
                # bbox = results['gt_bboxes']
                # for i in range(len(bbox)):
                #     x, y = bbox[i]
                #     for j in range(int(x), int(y + 1)):
                #         cv2.imwrite(f'{j}.jpg', imgs[j])

                assert len(results['gt_bboxes']) > 0
                # for k in results.keys():
                #     if k != 'img':
                #         print(k, results[k])

                return results


@registry.register_module('pipeline')
class VideoRandomCrop(object):
    def __init__(self,
                 window_size=256,
                 patch_iof_th=0,
                 gt_iof_th=0):
        self.window_size = window_size
        self.patch_iof_th = patch_iof_th
        self.gt_iof_th = gt_iof_th
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }

    def __call__(self, results):

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        assert 'bbox_fields' in results
        bboxes = [results[key] for key in results['bbox_fields']]
        bboxes = np.concatenate(bboxes, 0)

        imgs = results['img']
        duration = results['img_info']['duration']
        while True:
            sample_position = int(max(1, duration - self.window_size))
            start_idx = random.randint(0, sample_position)

            patch = np.array([start_idx, min(duration, start_idx + self.window_size)])

            def filter_bboxes(bboxes, patch):
                lens = bboxes[:, 1] - bboxes[:, 0]

                x1s = bboxes[:, 0]
                x2s = bboxes[:, 1]
                x1, x2 = patch

                ss = np.maximum(x1s, x1)
                es = np.minimum(x2s, x2)

                unions = es - ss
                gt_iofs = np.maximum(0, unions) / lens
                patch_iofs = np.maximum(0, unions) / (x2 - x1)

                # print(iofs, unions)
                mask = np.logical_or((gt_iofs > self.gt_iof_th),
                                     (patch_iofs > self.patch_iof_th))
                # TODO: ignore
                # ignore_mask = (iofs < self.gt_iof_th) * (iofs > 0)

                return mask

            mask = filter_bboxes(bboxes, patch)
            if mask.sum() == 0:
                continue

            for key in results.get('bbox_fields', []):
                boxes = results[key].copy()
                mask = filter_bboxes(boxes, patch)
                boxes = boxes[mask]

                boxes[:, 0] = boxes[:, 0].clip(min=patch[0])
                boxes[:, 1] = boxes[:, 1].clip(max=patch[1])
                boxes -= patch[0]

                results[key] = boxes
                # labels
                label_key = self.bbox2label.get(key)
                if label_key in results:
                    results[label_key] = results[label_key][mask]

            # adjust the img no matter whether the gt is empty before crop
            imgs = imgs[patch[0]:patch[1]]
            results['img'] = imgs
            results['img_shape'] = imgs.shape

            # import cv2
            # for i in range(256):
            #     cv2.imwrite(f'{i}c.jpg', imgs[i])

            return results


@registry.register_module('pipeline')
class VideoRandomCropBeforeLoading(object):
    def __init__(self,
                 window_size=256,
                 patch_iof_th=0,
                 gt_iof_th=0,
                 interval=1):
        self.window_size = window_size
        self.patch_iof_th = patch_iof_th
        self.gt_iof_th = gt_iof_th
        self.interval = interval
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }

    def __call__(self, results):

        assert 'bbox_fields' in results
        bboxes = [results[key] for key in results['bbox_fields']]
        bboxes = np.concatenate(bboxes, 0)

        duration = results['img_info']['duration']
        while True:

            interval = random.randint(1, self.interval + 1)
            results['interval'] = interval

            sample_position = int(max(1, duration - self.window_size * interval))
            start_idx = random.randint(0, sample_position)

            patch = np.array([start_idx, min(
                duration, start_idx + self.window_size * interval)])
            results['patch'] = list(patch)

            def filter_bboxes(bboxes, patch):
                lens = bboxes[:, 1] - bboxes[:, 0]

                x1s = bboxes[:, 0]
                x2s = bboxes[:, 1]
                x1, x2 = patch

                ss = np.maximum(x1s, x1)
                es = np.minimum(x2s, x2)

                unions = es - ss
                gt_iofs = np.maximum(0, unions) / lens
                patch_iofs = np.maximum(0, unions) / (x2 - x1)

                # print(iofs, unions)
                mask = np.logical_or((gt_iofs > self.gt_iof_th),
                                     (patch_iofs > self.patch_iof_th))
                # TODO: ignore
                # ignore_mask = (iofs < self.gt_iof_th) * (iofs > 0)

                return mask

            mask = filter_bboxes(bboxes, patch)
            if mask.sum() == 0:
                continue

            for key in results.get('bbox_fields', []):
                boxes = results[key].copy()
                mask = filter_bboxes(boxes, patch)
                boxes = boxes[mask]

                boxes[:, 0] = boxes[:, 0].clip(min=patch[0])
                boxes[:, 1] = boxes[:, 1].clip(max=patch[1])
                boxes -= patch[0]
                boxes /= interval

                results[key] = boxes
                # labels
                label_key = self.bbox2label.get(key)
                if label_key in results:
                    results[label_key] = results[label_key][mask]

            return results


@registry.register_module('pipeline')
class Pad3d(object):
    """Pad the image.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size_divisor=None, pad_val=0):
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size_divisor is not None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        for key in results.get('img_fields', ['img']):
            img = results[key]
            pad_t = int(np.ceil(
                img.shape[0] / self.size_divisor)) * self.size_divisor - img.shape[0]
            if pad_t > 0:
                padded_img = np.pad(
                    img, ((0, pad_t), (0, 0), (0, 0), (0, 0)),
                    'constant', constant_values=self.pad_val)
            else:
                padded_img = img
            results[key] = padded_img

        results['pad_shape'] = padded_img.shape
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@registry.register_module('pipeline')
class Rotate3d(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, max_angle, p=0.5, 
                 interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101):
        self.max_angle = max_angle
        self.p = p
        self.interpolation = interpolation
        self.border_mode = border_mode

    def __call__(self, results):

        if random.uniform() < self.p:
            for key in results.get('img_fields', ['img']):
                img = results[key]
                t, h, w, c = img.shape
                angle = random.uniform(-self.max_angle, self.max_angle)
                matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

                new_img = []
                for i in range(t):
                    _img = img[i]
                    _img = cv2.warpAffine(
                        _img, M=matrix, dsize=(w, h),
                        flags=self.interpolation, borderMode=self.border_mode)
                    new_img.append(_img)
                results[key] = np.array(new_img)

        return results


@registry.register_module('pipeline')
class Normalize3d(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.channel = len(mean)

    def __call__(self, results):

        for key in results.get('img_fields', ['img']):
            img = results[key]
            mean = np.reshape(np.array(self.mean, dtype=img.dtype),
                              [self.channel])
            std = np.reshape(np.array(self.std, dtype=img.dtype), [self.channel])
            denominator = np.reciprocal(std, dtype=img.dtype)

            new_img = (img - mean) * denominator
            results[key] = new_img

        # import cv2
        # for i in range(256):
        #     cv2.imwrite(f'{i}.jpg', new_img[i].astype(np.uint8))

        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


@registry.register_module('pipeline')
class Resize3d(object):
    """Random crop the image & bboxes.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        allow_negative_crop (bool): Whether to allow a crop that does not
            contain any bbox area. Default to False.

    Notes:
        - If the image is smaller than the crop size, return the original image
        - The keys for bboxes, labels must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self, size):
        assert size[0] > 0 and size[1] > 0
        self.size = size
        # The key correspondence from bboxes to labels.

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key]
            t, h, w, c = img.shape
            # print(img.shape)

            new_img = []
            for i in range(t):
                _img = img[i]
                _img = cv2.resize(_img, self.size)
                new_img.append(_img)

            img = np.array(new_img)
            img_shape = img.shape
            results[key] = img

        #     print(results)

        #     for i in range(768):
        #         cv2.imwrite(f'{i}.jpg', img[i].astype(np.uint8))

        # raise

        results['img_shape'] = img_shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@registry.register_module('pipeline')
class RandomCrop3d(object):
    """Random crop the image & bboxes.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        allow_negative_crop (bool): Whether to allow a crop that does not
            contain any bbox area. Default to False.

    Notes:
        - If the image is smaller than the crop size, return the original image
        - The keys for bboxes, labels must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self, crop_size):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        # The key correspondence from bboxes to labels.

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        for key in results.get('img_fields', ['img']):
            img = results[key]
            t, h, w, c = img.shape

            margin_h = max(h - self.crop_size[0], 0)
            margin_w = max(w - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            # crop the image
            img = img[:, crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img

        results['img_shape'] = img_shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@registry.register_module('pipeline')
class CenterCrop3d(object):
    """Random crop the image & bboxes.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        allow_negative_crop (bool): Whether to allow a crop that does not
            contain any bbox area. Default to False.

    Notes:
        - If the image is smaller than the crop size, return the original image
        - The keys for bboxes, labels must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self, crop_size):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        # The key correspondence from bboxes to labels.

    def _get_center_crop_coords(self, height, width, crop_height, crop_width):
        y1 = (height - crop_height) // 2
        y2 = y1 + crop_height
        x1 = (width - crop_width) // 2
        x2 = x1 + crop_width
        return x1, y1, x2, y2

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        for key in results.get('img_fields', ['img']):
            img = results[key]
            t, h, w, c = img.shape

            assert h > self.crop_size[0] and w > self.crop_size[1]

            crop_x1, crop_y1, crop_x2, crop_y2 = self._get_center_crop_coords(
                h, w, self.crop_size[0], self.crop_size[1])

            # crop the image
            img = img[:, crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img

        results['img_shape'] = img_shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@registry.register_module('pipeline')
class RandomFlip3d(object):
    """Flip the image & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, flip_ratio=0, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        """Call function to flip bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                assert self.direction in ['horizontal', 'vertical']
                if self.direction == 'horizontal':
                    results[key] = np.flip(results[key], axis=2)
                else:
                    results[key] = np.flip(results[key], axis=1)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'


@registry.register_module('pipeline')
class PhotoMetricDistortion3D(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 p=0.5,
                 mode=1):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.p = p
        self.mode = mode

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        assert img.dtype == np.float32, \
            'PhotoMetricDistortion needs the input image of dtype np.float32,'\
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'

        def _filter(img):
            img[img < 0] = 0
            img[img > 255] = 255
            return img

        t, h, w, c = img.shape

        if random.uniform(0, 1) <= self.p:

            if self.mode == 0:
                # random brightness
                new_img = []
                for i in range(t):
                    _img = img[i]

                    if random.randint(2):
                        delta = random.uniform(-self.brightness_delta,
                                               self.brightness_delta)
                        _img += delta
                        _img = _filter(_img)

                    # mode == 0 --> do random contrast first
                    # mode == 1 --> do random contrast last
                    mode = random.randint(2)
                    if mode == 1:
                        if random.randint(2):
                            alpha = random.uniform(self.contrast_lower,
                                                   self.contrast_upper)
                            _img *= alpha
                            _img = _filter(_img)

                    # convert color from BGR to HSV
                    _img = image.bgr2hsv(_img)

                    # random saturation
                    if random.randint(2):
                        _img[..., 1] *= random.uniform(self.saturation_lower,
                                                       self.saturation_upper)

                    # random hue
                    if random.randint(2):
                        _img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                        _img[..., 0][_img[..., 0] > 360] -= 360
                        _img[..., 0][_img[..., 0] < 0] += 360

                    # convert color from HSV to BGR
                    _img = image.hsv2bgr(_img)
                    _img = _filter(_img)

                    # random contrast
                    if mode == 0:
                        if random.randint(2):
                            alpha = random.uniform(self.contrast_lower,
                                                   self.contrast_upper)
                            _img *= alpha
                            _img = _filter(_img)

                    new_img.append(_img)

                # randomly swap channels
                # if random.randint(2):
                #     img = img[..., random.permutation(3)]

                img = np.array(new_img)

            elif self.mode == 1:
                # random brightness
                if random.randint(2):
                    delta = random.uniform(-self.brightness_delta,
                                           self.brightness_delta)
                    img += delta
                    img = _filter(img)

                # mode == 0 --> do random contrast first
                # mode == 1 --> do random contrast last
                mode = random.randint(2)
                if mode == 1:
                    if random.randint(2):
                        alpha = random.uniform(self.contrast_lower,
                                               self.contrast_upper)
                        img *= alpha
                        img = _filter(img)

                # convert color from BGR to HSV
                new_img = []
                saturation = random.uniform(self.saturation_lower, self.saturation_upper) if random.randint(2) else 1
                hue = random.uniform(-self.hue_delta, self.hue_delta) if random.randint(2) else 0

                for i in range(t):
                    _img = img[i]

                    _img = image.bgr2hsv(_img)

                    # random saturation
                    _img[..., 1] *= saturation

                    # random hue
                    _img[..., 0] += hue
                    _img[..., 0][_img[..., 0] > 360] -= 360
                    _img[..., 0][_img[..., 0] < 0] += 360

                    # convert color from HSV to BGR
                    _img = image.hsv2bgr(_img)
                    new_img.append(_img)

                img = np.array(new_img)
                img = _filter(img)

                # random contrast
                if mode == 0:
                    if random.randint(2):
                        alpha = random.uniform(self.contrast_lower,
                                               self.contrast_upper)
                        img *= alpha
                        img = _filter(img)
            else:
                raise NotImplementedError

            results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@registry.register_module('pipeline')
class Resize(object):
    """Resize images & bbox.

    This transform resizes the input image to some scale. Bboxes are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio range
      and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                img, scale_factor = image.imrescale(
                    results[key], results['scale'], return_scale=True)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = image.imresize(
                    results[key], results['scale'], return_scale=True)
            results[key] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def __call__(self, results):
        """Call function to resize images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            assert 'scale_factor' not in results, (
                'scale and scale_factor cannot be both set.')

        self._resize_img(results)
        self._resize_bboxes(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio})'
        return repr_str


@registry.register_module('pipeline')
class RandomFlip(object):
    """Flip the image & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1
        assert direction in ['horizontal', 'vertical']

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """

        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def __call__(self, results):
        """Call function to flip bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                results[key] = image.imflip(
                    results[key], direction=results['flip_direction'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'],
                                              results['flip_direction'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'


@registry.register_module('pipeline')
class Pad(object):
    """Pad the image.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        for key in results.get('img_fields', ['img']):
            if self.size is not None:
                padded_img = image.impad(
                    results[key], shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                padded_img = image.impad_to_multiple(
                    results[key], self.size_divisor, pad_val=self.pad_val)
            results[key] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@registry.register_module('pipeline')
class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img']):
            results[key] = image.imnormalize(results[key], self.mean, self.std,
                                             self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@registry.register_module('pipeline')
class RandomCrop(object):
    """Random crop the image & bboxes.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        allow_negative_crop (bool): Whether to allow a crop that does not
            contain any bbox area. Default to False.

    Notes:
        - If the image is smaller than the crop size, return the original image
        - The keys for bboxes, labels must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self, crop_size, allow_negative_crop=False):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.allow_negative_crop = allow_negative_crop
        # The key correspondence from bboxes to labels.
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # self.allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not self.allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@registry.register_module('pipeline')
class RandomSquareCrop(object):
    """Random crop the square patch of image & bboxes with a size from Args
        (crop_ratio_range or crop_choice) of the short edge of image and
        keep the overlapped part of box if its center is within the cropped patch.

    Args:
        crop_ratio_range (list): a list of two elements (min, max)
        crop_choice (list): a list of crop ratio.

    Note:
        The keys for bboxes, labels and masks should be paired. That is, \
        `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and \
        `gt_bboxes_ignore` to `gt_labels_ignore` and `gt_masks_ignore`.
    """

    def __init__(self, crop_ratio_range=None, crop_choice=None):

        self.crop_ratio_range = crop_ratio_range
        self.crop_choice = crop_choice

        assert (self.crop_ratio_range is None) ^ (self.crop_choice is None)
        if self.crop_ratio_range is not None:
            self.crop_ratio_min, self.crop_ratio_max = self.crop_ratio_range

        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def __call__(self, results):
        """Call function to crop images and bounding boxes with minimum IoU
        constraint.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images and bounding boxes cropped, \
                'img_shape' key is updated.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        assert 'bbox_fields' in results
        boxes = [results[key] for key in results['bbox_fields']]
        boxes = np.concatenate(boxes, 0)
        h, w, c = img.shape

        while True:

            if self.crop_ratio_range is not None:
                scale = np.random.uniform(self.crop_ratio_min,
                                          self.crop_ratio_max)
            elif self.crop_choice is not None:
                scale = np.random.choice(self.crop_choice)

            # print(scale, img.shape[:2], boxes)
            # import cv2
            # cv2.imwrite('aaa.png', img)

            for i in range(250):
                short_side = min(w, h)
                cw = int(scale * short_side)
                ch = cw

                # TODO +1
                left = random.uniform(w - cw)
                top = random.uniform(h - ch)

                patch = np.array(
                    (int(left), int(top), int(left + cw), int(top + ch)))

                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                # adjust boxes
                def is_center_of_bboxes_in_patch(boxes, patch):
                    # TODO >=
                    center = (boxes[:, :2] + boxes[:, 2:]) / 2
                    mask = ((center[:, 0] > patch[0]) *
                            (center[:, 1] > patch[1]) *
                            (center[:, 0] < patch[2]) *
                            (center[:, 1] < patch[3]))
                    return mask

                mask = is_center_of_bboxes_in_patch(boxes, patch)
                if not mask.any():
                    continue
                for key in results.get('bbox_fields', []):
                    boxes = results[key].copy()
                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    boxes = boxes[mask]
                    boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                    boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                    boxes -= np.tile(patch[:2], 2)

                    results[key] = boxes
                    # labels
                    label_key = self.bbox2label.get(key)
                    if label_key in results:
                        results[label_key] = results[label_key][mask]

                    # mask fields
                    mask_key = self.bbox2mask.get(key)
                    if mask_key in results:
                        results[mask_key] = results[mask_key][mask.nonzero()
                                                              [0]].crop(patch)

                # adjust the img no matter whether the gt is empty before crop
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                results['img'] = img
                results['img_shape'] = img.shape

                # seg fields
                for key in results.get('seg_fields', []):
                    results[key] = results[key][patch[1]:patch[3],
                                                patch[0]:patch[2]]
                return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_ious={self.min_iou}, '
        repr_str += f'crop_size={self.crop_size})'
        return repr_str


@registry.register_module('pipeline')
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 p=0.5):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.p = p

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        assert img.dtype == np.float32, \
            'PhotoMetricDistortion needs the input image of dtype np.float32,'\
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'

        def _filter(img):
            img[img < 0] = 0
            img[img > 255] = 255
            return img

        if random.uniform(0, 1) <= self.p:

            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                       self.brightness_delta)
                img += delta
                img = _filter(img)

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    img *= alpha
                    img = _filter(img)

            # convert color from BGR to HSV
            img = image.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                              self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = image.hsv2bgr(img)
            img = _filter(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    img *= alpha
                    img = _filter(img)

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]

            results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@registry.register_module('pipeline')
class Expand(object):
    """Random expand the image & bboxes.

    Randomly place the original image on a canvas of 'ratio' x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    Args:
        mean (tuple): mean value of dataset.
        to_rgb (bool): if need to convert the order of mean to align with RGB.
        ratio_range (tuple): range of expand ratio.
        prob (float): probability of applying this transformation
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 to_rgb=True,
                 ratio_range=(1, 4),
                 prob=0.5):
        self.to_rgb = to_rgb
        self.ratio_range = ratio_range
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range
        self.prob = prob

    def __call__(self, results):
        """Call function to expand images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images, bounding boxes expanded
        """

        if random.uniform(0, 1) > self.prob:
            return results

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean,
                             dtype=img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img

        results['img'] = expand_img
        # expand bboxes
        for key in results.get('bbox_fields', []):
            results[key] = results[key] + np.tile(
                (left, top), 2).astype(results[key].dtype)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, to_rgb={self.to_rgb}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        return repr_str


@registry.register_module('pipeline')
class MinIoURandomCrop(object):
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold for all intersections with
        bounding boxes
        min_crop_size (float): minimum crop's size (i.e. h,w := a*h, a*w,
        where a >= min_crop_size).

    Notes:
        The keys for bboxes, labels should be paired. That is,
        `gt_bboxes` corresponds to `gt_labels`, and
        `gt_bboxes_ignore` to `gt_labels_ignore`.
    """

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }

    def __call__(self, results):
        """Call function to crop images and bounding boxes with minimum IoU
        constraint.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images and bounding boxes cropped,
                'img_shape' key is updated.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        assert 'bbox_fields' in results
        boxes = [results[key] for key in results['bbox_fields']]
        boxes = np.concatenate(boxes, 0)
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            self.mode = mode
            if mode == 1:
                return results

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                # Line or point crop is not allowed
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                # only adjust boxes when the gt is not empty
                if len(overlaps) > 0:
                    # adjust boxes
                    def is_center_of_bboxes_in_patch(boxes, patch):
                        center = (boxes[:, :2] + boxes[:, 2:]) / 2
                        mask = ((center[:, 0] > patch[0]) *
                                (center[:, 1] > patch[1]) *
                                (center[:, 0] < patch[2]) *
                                (center[:, 1] < patch[3]))
                        return mask

                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if not mask.any():
                        continue
                    for key in results.get('bbox_fields', []):
                        boxes = results[key].copy()
                        mask = is_center_of_bboxes_in_patch(boxes, patch)
                        boxes = boxes[mask]
                        boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                        boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                        boxes -= np.tile(patch[:2], 2)

                        results[key] = boxes
                        # labels
                        label_key = self.bbox2label.get(key)
                        if label_key in results:
                            results[label_key] = results[label_key][mask]

                # adjust the img no matter whether the gt is empty before crop
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                results['img'] = img
                results['img_shape'] = img.shape

                return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_ious={self.min_ious}, '
        repr_str += f'min_crop_size={self.min_crop_size})'
        return repr_str


@registry.register_module('pipeline')
class Corrupt(object):
    """Corruption augmentation.

    Corruption transforms implemented based on
    `imagecorruptions <https://github.com/bethgelab/imagecorruptions>`_.

    Args:
        corruption (str): Corruption name.
        severity (int, optional): The severity of corruption. Default: 1.
    """

    def __init__(self, corruption, severity=1):
        self.corruption = corruption
        self.severity = severity

    def __call__(self, results):
        """Call function to corrupt image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images corrupted.
        """

        if corrupt is None:
            raise RuntimeError('imagecorruptions is not installed')
        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        results['img'] = corrupt(
            results['img'].astype(np.uint8),
            corruption_name=self.corruption,
            severity=self.severity)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(corruption={self.corruption}, '
        repr_str += f'severity={self.severity})'
        return repr_str


@registry.register_module('pipeline')
class Albu(object):
    """Albumentation augmentation.

    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.

    An example of ``transforms`` is as followed:

    .. code-block::

        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]

    Args:
        transforms (list[dict]): A list of albu transformations
        bbox_params (dict): Bbox_params for albumentation `Compose`
        keymap (dict): Contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): Whether to skip the image if no ann left
            after aug
    """

    def __init__(self,
                 transforms,
                 bbox_params=None,
                 keymap=None,
                 update_pad_shape=False,
                 skip_img_without_anno=False):
        if Compose is None:
            raise RuntimeError('albumentations is not installed')

        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape
        self.skip_img_without_anno = skip_img_without_anno

        # A simple workaround to remove masks without boxes
        if (isinstance(bbox_params, dict) and 'label_fields' in bbox_params
                and 'filter_lost_elements' in bbox_params):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params['label_fields']
            bbox_params['label_fields'] = ['idx_mapper']
            del bbox_params['filter_lost_elements']

        self.bbox_params = (
            self.albu_builder(bbox_params) if bbox_params else None)
        self.aug = Compose([self.albu_builder(t) for t in self.transforms],
                           bbox_params=self.bbox_params)

        if not keymap:
            self.keymap_to_albu = {'img': 'image', 'gt_bboxes': 'bboxes'}
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.

        It inherits some of :func:`build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper. Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """

        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)
        # TODO: add bbox_fields
        if 'bboxes' in results:
            # to list of boxes
            if isinstance(results['bboxes'], np.ndarray):
                results['bboxes'] = [x for x in results['bboxes']]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        results = self.aug(**results)

        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)
            results['bboxes'] = results['bboxes'].reshape(-1, 4)

            # filter label_fields
            if self.filter_lost_elements:

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])

                if (not len(results['idx_mapper'])
                        and self.skip_img_without_anno):
                    return None

        if 'gt_labels' in results:
            if isinstance(results['gt_labels'], list):
                results['gt_labels'] = np.array(results['gt_labels'])
            results['gt_labels'] = results['gt_labels'].astype(np.int64)

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


@registry.register_module('pipeline')
class RandomCenterCropPad(object):
    """Random center crop and random around padding for CornerNet.

    This operation generates randomly cropped image from the original image and
    pads it simultaneously. Different from `RandomCrop`, the output shape may
    not equal to `crop_size` strictly. We choose a random value from `ratios`
    and the output shape could be larger or smaller than `crop_size`. Also the
    pad in this operation is different from `Pad`, actually we use around
    padding instead of right-bottom padding.

    The relation between output image (padding image) and original image:

    .. code-block: text

                        output image
            +----------------------------+
            |          padded area       |
        +------|----------------------------|----------+
        |      |         cropped area       |          |
        |      |         +---------------+  |          |
        |      |         |    .   center |  |          | original image
        |      |         |        range  |  |          |
        |      |         +---------------+  |          |
        +------|----------------------------|----------+
            |          padded area       |
            +----------------------------+

    There are 5 main areas in the figure:
        - output image: output image of this operation, also called padding
            image in following instruction.
        - original image: input image of this operation.
        - padded area: non-intersect area of output image and original image.
        - cropped area: the overlap of output image and original image.
        - center range: a smaller area where random center chosen from.
            center range is computed by `border` and original image's shape
            to avoid our random center is too close to original image's border.

    Also this operation act differently in train and test mode, the summary
    pipeline is listed below.

    Train pipeline:
        1. Choose a `random_ratio` from `ratios`, the shape of padding image
            will be `random_ratio * crop_size`.
        2. Choose a `random_center` in `center range`.
        3. Generate padding image with center matches the `random_center`.
        4. Initialize the padding image with pixel value equals to `mean`.
        5. Copy the `cropped area` to padding image.
        6. Refine annotations.

    Test pipeline:
        1. Compute output shape according to `test_pad_mode`.
        2. Generate padding image with center matches the original image
            center.
        3. Initialize the padding image with pixel value equals to `mean`.
        4. Copy the `cropped area` to padding image.

    Args:
        crop_size (tuple | None): expected size after crop, final size will
            computed according to ratio. Requires (h, w) in train mode, and
            None in test mode.
        ratios (tuple): random select a ratio from tuple and crop image to
            (crop_size[0] * ratio) * (crop_size[1] * ratio).
            Only available in train mode.
        border (int): max distance from center select area to image border.
            Only available in train mode.
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB.
        test_mode (bool): whether involve random variables in transform.
            In train mode, crop_size is fixed, center coords and ratio is
            random selected from predefined lists. In test mode, crop_size
            is image's original shape, center coords and ratio is fixed.
        test_pad_mode (tuple): padding method and padding shape value, only
            available in test mode. Default is using 'logical_or' with
            127 as padding shape value.

            - 'logical_or': final_shape = input_shape | padding_shape_value
            - 'size_divisor': final_shape = int(
                ceil(input_shape / padding_shape_value) * padding_shape_value)
    """

    def __init__(self,
                 crop_size=None,
                 ratios=(0.9, 1.0, 1.1),
                 border=128,
                 mean=None,
                 std=None,
                 to_rgb=None,
                 test_mode=False,
                 test_pad_mode=('logical_or', 127)):
        if test_mode:
            assert crop_size is None, 'crop_size must be None in test mode'
            assert ratios is None, 'ratios must be None in test mode'
            assert border is None, 'border must be None in test mode'
            assert isinstance(test_pad_mode, (list, tuple))
            assert test_pad_mode[0] in ['logical_or', 'size_divisor']
        else:
            assert isinstance(crop_size, (list, tuple))
            assert crop_size[0] > 0 and crop_size[1] > 0, (
                'crop_size must > 0 in train mode')
            assert isinstance(ratios, (list, tuple))
            assert test_pad_mode is None, (
                'test_pad_mode must be None in train mode')

        self.crop_size = crop_size
        self.ratios = ratios
        self.border = border
        # We do not set default value to mean, std and to_rgb because these
        # hyper-parameters are easy to forget but could affect the performance.
        # Please use the same setting as Normalize for performance assurance.
        assert mean is not None and std is not None and to_rgb is not None
        self.to_rgb = to_rgb
        self.input_mean = mean
        self.input_std = std
        if to_rgb:
            self.mean = mean[::-1]
            self.std = std[::-1]
        else:
            self.mean = mean
            self.std = std
        self.test_mode = test_mode
        self.test_pad_mode = test_pad_mode

    def _get_border(self, border, size):
        """Get final border for the target size.

        This function generates a `final_border` according to image's shape.
        The area between `final_border` and `size - final_border` is the
        `center range`. We randomly choose center from the `center range`
        to avoid our random center is too close to original image's border.

        Args:
            border (int): The initial border, default is 128.
            size (int): The width or height of original image.
        Returns:
            int: The final border.
        """
        i = pow(2, np.ceil(np.log2(np.ceil(2 * border / size))))
        return border // i

    def _filter_boxes(self, patch, boxes):
        """Check whether the center of each box is in the patch.

        Args:
            patch (list[int]): The cropped area, [left, top, right, bottom].
            boxes (numpy array, (N x 4)): Ground truth boxes.
        Returns:
            mask (numpy array, (N,)): Each box is inside or outside the patch.
        """
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask = (center[:, 0] > patch[0]) * (center[:, 1] > patch[1]) * (
            center[:, 0] < patch[2]) * (
                center[:, 1] < patch[3])
        return mask

    def _crop_image_and_paste(self, image, center, size):
        """Crop image with a given center and size, then paste the cropped
        image to a blank image with two centers align.

        This function is equivalent to generating a blank image with `size` as
        its shape. Then cover it on the original image with two centers (
        the center of blank image and the random center of original image)
        aligned. The overlap area is paste from the original image and the
        outside area is filled with `mean pixel`.

        Args:
            image (np array, H x W x C): Original image.
            center (list[int]): Target crop center coord.
            size (list[int]): Target crop size. [target_h, target_w]
        Returns:
            cropped_img (np array, target_h x target_w x C): Cropped image.
            border (np array, 4): The distance of four border of `cropped_img`
                to the original image area, [top, bottom, left, right]
            patch (list[int]): The cropped area, [left, top, right, bottom].
        """
        center_y, center_x = center
        target_h, target_w = size
        img_h, img_w, img_c = image.shape

        x0 = max(0, center_x - target_w // 2)
        x1 = min(center_x + target_w // 2, img_w)
        y0 = max(0, center_y - target_h // 2)
        y1 = min(center_y + target_h // 2, img_h)
        patch = np.array((int(x0), int(y0), int(x1), int(y1)))

        left, right = center_x - x0, x1 - center_x
        top, bottom = center_y - y0, y1 - center_y

        cropped_center_y, cropped_center_x = target_h // 2, target_w // 2
        cropped_img = np.zeros((target_h, target_w, img_c), dtype=image.dtype)
        for i in range(img_c):
            cropped_img[:, :, i] += self.mean[i]
        y_slice = slice(cropped_center_y - top, cropped_center_y + bottom)
        x_slice = slice(cropped_center_x - left, cropped_center_x + right)
        cropped_img[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

        border = np.array([
            cropped_center_y - top, cropped_center_y + bottom,
            cropped_center_x - left, cropped_center_x + right
        ],
                          dtype=np.float32)

        return cropped_img, border, patch

    def _train_aug(self, results):
        """Random crop and around padding the original image.

        Args:
            results (dict): Image infomations in the augment pipeline.
        Returns:
            results (dict): The updated dict.
        """
        img = results['img']
        h, w, c = img.shape
        boxes = results['gt_bboxes']
        while True:
            scale = random.choice(self.ratios)
            new_h = int(self.crop_size[0] * scale)
            new_w = int(self.crop_size[1] * scale)
            h_border = self._get_border(self.border, h)
            w_border = self._get_border(self.border, w)

            for i in range(50):
                center_x = random.randint(low=w_border, high=w - w_border)
                center_y = random.randint(low=h_border, high=h - h_border)

                cropped_img, border, patch = self._crop_image_and_paste(
                    img, [center_y, center_x], [new_h, new_w])

                mask = self._filter_boxes(patch, boxes)
                # if image do not have valid bbox, any crop patch is valid.
                if not mask.any() and len(boxes) > 0:
                    continue

                results['img'] = cropped_img
                results['img_shape'] = cropped_img.shape
                results['pad_shape'] = cropped_img.shape

                x0, y0, x1, y1 = patch

                left_w, top_h = center_x - x0, center_y - y0
                cropped_center_x, cropped_center_y = new_w // 2, new_h // 2

                # crop bboxes accordingly and clip to the image boundary
                for key in results.get('bbox_fields', []):
                    mask = self._filter_boxes(patch, results[key])
                    bboxes = results[key][mask]
                    bboxes[:, 0:4:2] += cropped_center_x - left_w - x0
                    bboxes[:, 1:4:2] += cropped_center_y - top_h - y0
                    bboxes[:, 0:4:2] = np.clip(bboxes[:, 0:4:2], 0, new_w)
                    bboxes[:, 1:4:2] = np.clip(bboxes[:, 1:4:2], 0, new_h)
                    keep = (bboxes[:, 2] > bboxes[:, 0]) & (
                        bboxes[:, 3] > bboxes[:, 1])
                    bboxes = bboxes[keep]
                    results[key] = bboxes
                    if key in ['gt_bboxes']:
                        if 'gt_labels' in results:
                            labels = results['gt_labels'][mask]
                            labels = labels[keep]
                            results['gt_labels'] = labels

                return results

    def _test_aug(self, results):
        """Around padding the original image without cropping.

        The padding mode and value are from `test_pad_mode`.

        Args:
            results (dict): Image infomations in the augment pipeline.
        Returns:
            results (dict): The updated dict.
        """
        img = results['img']
        h, w, c = img.shape
        results['img_shape'] = img.shape
        if self.test_pad_mode[0] in ['logical_or']:
            target_h = h | self.test_pad_mode[1]
            target_w = w | self.test_pad_mode[1]
        elif self.test_pad_mode[0] in ['size_divisor']:
            divisor = self.test_pad_mode[1]
            target_h = int(np.ceil(h / divisor)) * divisor
            target_w = int(np.ceil(w / divisor)) * divisor
        else:
            raise NotImplementedError(
                'RandomCenterCropPad only support two testing pad mode:'
                'logical-or and size_divisor.')

        cropped_img, border, _ = self._crop_image_and_paste(
            img, [h // 2, w // 2], [target_h, target_w])
        results['img'] = cropped_img
        results['pad_shape'] = cropped_img.shape
        results['border'] = border
        return results

    def __call__(self, results):
        img = results['img']
        assert img.dtype == np.float32, (
            'RandomCenterCropPad needs the input image of dtype np.float32,'
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline')
        h, w, c = img.shape
        assert c == len(self.mean)
        if self.test_mode:
            return self._test_aug(results)
        else:
            return self._train_aug(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'ratios={self.ratios}, '
        repr_str += f'border={self.border}, '
        repr_str += f'mean={self.input_mean}, '
        repr_str += f'std={self.input_std}, '
        repr_str += f'to_rgb={self.to_rgb}, '
        repr_str += f'test_mode={self.test_mode}, '
        repr_str += f'test_pad_mode={self.test_pad_mode})'
        return repr_str
