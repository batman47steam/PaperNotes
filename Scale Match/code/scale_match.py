import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
import json
import warnings
from copy import deepcopy
from math import inf
import random
from mmdet.core.bbox.coder.bouding_box import BoxList
from ..builder import PIPELINES
from .transforms import Resize
PIL_RESIZE_MODE = {'bilinear': Image.BILINEAR, 'nearest': Image.NEAREST}


@PIPELINES.register_module()
class ScaleMatchResize(Resize):
    def __init__(self, scale_match_type,
                 # img_scale=None,
                 # multiscale_mode='range',
                 # ratio_range=None,
                 # keep_ratio=True,
                 backend='cv2',
                 # scale_factor=None,
                 filter_box_size_th=2,
                 *args, **kwargs):
        super(ScaleMatchResize, self).__init__(keep_ratio=True, backend=backend)
        if scale_match_type == 'ScaleMatch':
            self.scale_match = ScaleMatch(*args, **kwargs)
        elif scale_match_type == 'MonotonicityScaleMatch':
            self.scale_match = MonotonicityScaleMatch(*args, **kwargs)
        elif scale_match_type == 'GaussianScaleMatch':
            self.scale_match = GaussianScaleMatch(*args, **kwargs)
        else:
            raise ValueError("type must be chose in ['ScaleMatch', 'MonotonicityScaleMatch'"
                             "'GaussianScaleMatch'], but {} got".format(type))
        self.filter_box_size_th = filter_box_size_th

    def filter_small_bbox(self, results):
        if 'gt_bboxes' in results:
            bbox = results['gt_bboxes']
            keep = np.logical_and((bbox[:, 2] - bbox[:, 0] + 1) >= self.filter_box_size_th,
                                  (bbox[:, 3] - bbox[:, 1] + 1) >= self.filter_box_size_th)
            for key in ['gt_bboxes', 'gt_labels']:
                results[key] = results[key][keep]

            gt_bboxes_ignore = results['gt_bboxes_ignore']
            if len(gt_bboxes_ignore) > 0:
                bbox = gt_bboxes_ignore
                keep = np.logical_and((bbox[:, 2] - bbox[:, 0] + 1) >= self.filter_box_size_th,
                                      (bbox[:, 3] - bbox[:, 1] + 1) >= self.filter_box_size_th)
                results['gt_bboxes_ignore'] = results['gt_bboxes_ignore'][keep]
        elif len(results['mask_fields']) > 0:
            raise NotImplementedError('not implement for mask')

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """
        assert 'scale_factor' not in results and 'scale' not in results,\
            'scale_factor or scale can not been specified in results for ScaleMatch'
        new_image_HW = self.scale_match.get_new_size(results['img_shape'][:2], results['gt_bboxes'], 'xyxy')
        results['scale'] = new_image_HW

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)

        self.filter_small_bbox(results)
        return results


class ScaleMatch(object):
    """
        ScaleMatch face two problem when using:
            1) May generate too small scale, it will lead loss to NaN.
            2) May generate too big scale, it will lead out of memory.
            we find bigger batch size can ease 1) problem.
        there are four way to handle these problem:
            1) clip scale constraint to a specified scale_range
            2) change SM target distribute by scale mean and var
            3) use MonotonicityScaleMatch
            4) use chose scale as warm up scale
    """

    def __init__(self, distribute=None, sizes=None,  # param group 1
                 anno_file=None, bins=100, except_rate=-1.,  # param group 2
                 scale_range=(0., 2.), default_scale=1.0, max_sample_try=5, out_scale_deal='clip', use_log_bins=False,
                 mode='bilinear', debug_no_image_resize=False, debug_close_record=True):
        assert anno_file is not None or (distribute is not None and sizes is not None)
        if anno_file is not None:
            if except_rate < 0:
                except_rate = 1. / bins * 2
            distribute, sizes = ScaleMatch._get_distribute(json.load(open(anno_file))['annotations'], bins,
                                                           except_rate, use_log_bins)
        self.distri_cumsum = np.cumsum(distribute)
        self.sizes = sizes
        self.mode = PIL_RESIZE_MODE[mode]

        self.scale_range = scale_range  # scale_range[1] to avoid out of memory
        self.out_scale_deal = out_scale_deal
        assert out_scale_deal in ['clip', 'use_default_scale']
        self.max_sample_try = max_sample_try
        self.default_scale = default_scale

        self.fail_time = 0
        self.debug_no_image_resize = debug_no_image_resize
        self.debug_record = DebugScaleRecord(debug_close_record)

    @staticmethod
    def _get_distribute(annotations, bins=100, except_rate=0.1, use_log_bins=False, mu_sigma=(-1, -1)):
        """
        except_rate: to except len(annotations)*except_rate/2 abnormal points as head and tial bin
        """
        #这样来看好像真的是把所有的标记图片都拿来算直方图了
        annos = [anno for anno in annotations if not anno['iscrowd']]
        if len(annos) > 0 and 'ignore' in annos[0]:
            annos = [anno for anno in annos if not anno['ignore']]
        sizes = np.sqrt(np.array([anno['bbox'][2] * anno['bbox'][3] for anno in annos]))
        sizes = sizes[sizes > 0]

        if mu_sigma[0] > 0 and mu_sigma[1] > 0:
            print('distribute(mu, sigma): ', np.mean(sizes), np.std(sizes), end='->')
            sizes = (sizes - np.mean(sizes)) / np.std(sizes)
            sizes = sizes * mu_sigma[1] + mu_sigma[0]
            print(np.mean(sizes), np.std(sizes))
            sizes = sizes.clip(1)

        if use_log_bins:
            sizes = np.log(sizes)
        #对所有的size进行了一个排序
        sizes = np.sort(sizes)
        N = len(sizes)
        #应该是选取了不太大也不太小的size来创建直方图
        hist_sizes = sizes[int(N * except_rate / 2): int(N * (1 - except_rate / 2))]
        if except_rate > 0:
            #这个就已经创建好直方图了
            #s应该是帮我分配好了的bin的每个bin的上下边界？
            #c落在每个bin范围上值的数目
            c, s = np.histogram(hist_sizes, bins=bins - 2)
            #tolist()就是简单的把array给转换成列表，这里应该是又加上了头尾对应的数目，解决长尾分布的
            c = np.array([int(N * except_rate / 2)] + c.tolist() + [N - int(N * (1 - except_rate / 2))])
            #一开始只算了中间区间的，现在又把头尾给加上了，这里的思路就是我分布在头尾的bin对应的count可能很小，
            #于是他就把多个bin给合在一起，这样每个bin对应的count就又比较大了
            s = [sizes[0]] + s.tolist() + [sizes[-1]]
            s = np.array(s)
        else:
            c, s = np.histogram(hist_sizes, bins=bins)
        #这个得到的就是概率啊
        c = c / len(sizes)
        if use_log_bins:
            s = np.exp(s)
        return c, s

    def _sample_by_distribute(self):
        #就是从均匀分布（0，1）中随机采样
        #均匀分布就在这个区间每个位置取值的概率是一样的
        # r取每个值的概率都是相同的，但是根据直方图的不同，到底是落在哪个idx上会有所差异
        r = np.random.uniform()
        #self.distri_cumsum是一个函数里算出来的c沿着一个轴的累加
        #如果c是array([0.        , 0.66666667, 0.33333333])
        #c.cumsum()的结果是array([0.        , 0.66666667, 1.        ])
        #nonzero返回的是一个元组，所以我才先取元组中的第一个元素，然后再把真正的idx取出来
        # 这个其实就是判断你采样的结果是落在哪个bin里的
        idx = np.nonzero(r <= self.distri_cumsum + 1e-6)[0][0]
        #就由这个index决定了是落在哪个bin内
        mins, maxs = self.sizes[idx], self.sizes[idx + 1]
        ir = np.random.uniform()
        return (maxs - mins) * ir + mins

    def default_scale_deal(self, image_hw):
        scale = self.default_scale
        size = int(round(scale * image_hw[0])), int(round(scale * image_hw[1]))
        return size

    def get_new_size(self, image_hw, bbox, mode='xyxy'):
        """
            image_hw: (h, w) of image
            bbox: bbox of gt, [[x1, y1, x2, y2], ...]
        return:
            new image size: (h, w) of new image
        """
        target = BoxList(deepcopy(bbox), image_hw[::-1], mode)
        if len(target.bbox) == 0:
            return self.default_scale_deal(image_hw)

        # record old target info
        old_mode = target.mode

        # cal mean size of image's bbox
        boxes = target.convert('xywh').bbox.cpu().numpy()
        sizes = np.sqrt(boxes[:, 2] * boxes[:, 3])
        sizes = sizes[sizes > 0]
        src_size = np.exp(np.log(sizes).mean())
        # src_size = sizes.mean()

        # sample a size respect to target distribute
        # For memory stable, we set scale range.
        scale = self.default_scale
        for try_i in range(self.max_sample_try):
            # 主要也就是这里在发挥作用
            dst_size = self._sample_by_distribute()
            _scale = dst_size / src_size
            if self.scale_range[1] > _scale > self.scale_range[0]:
                scale = _scale
                break  # last time forget
        self.debug_record(_scale)
        if self.out_scale_deal == 'clip':
            if _scale >= self.scale_range[1]:
                scale = self.scale_range[1]  # note 1
            elif _scale <= self.scale_range[0]:
                scale = self.scale_range[0]  # note 1

        # resize bbox mean size to our want size
        size = int(round(scale * image_hw[0])), int(round(scale * image_hw[1]))
        target = target.convert(old_mode)
        target = target.resize((size[1], size[0]))

        # remove too tiny size to avoid unstable loss
        if len(target.bbox) > 0:
            target = target[(target.bbox[:, 2] - target.bbox[:, 0] + 1) >= 2]
            target = target[(target.bbox[:, 3] - target.bbox[:, 1] + 1) >= 2]

        if len(target.bbox) == 0:
            self.fail_time += 1
            if self.fail_time % 1 == 0:
                warnings.warn("Scale Matching failed for {} times, you may need to change the mean to min. "
                              "dst_size is {}, src_size is {}, sizes".format(self.fail_time, dst_size, src_size, sizes))
            return self.default_scale_deal(image_hw)
        return size

    def __call__(self, image: Image, target: BoxList):
        """
            transform format of facebook maskrcnn_benchmark
        """
        size = self.get_new_size([image.height, image.width], target.bbox, target.mode)
        target = target.resize((size[1], size[0]))
        if len(target.bbox) > 0:
            target = target[(target.bbox[:, 2] - target.bbox[:, 0] + 1) >= 2]
            target = target[(target.bbox[:, 3] - target.bbox[:, 1] + 1) >= 2]
        if not self.debug_no_image_resize:
            image = F.resize(image, size, self.mode)
        return image, target

# 这个的目的是保证他不产生过大或者过小的尺度
class MonotonicityScaleMatch(object):
    def __init__(self, src_anno_file, dst_anno_file, bins=100, except_rate=-1.,
                 scale_range=(0., 2.), default_scale=1.0, out_scale_deal='clip',
                 use_log_bins=False, mode='bilinear', mu_sigma=(-1, -1),
                 debug_no_image_resize=False, debug_close_record=False):
        if except_rate < 0:
            except_rate = 1. / bins * 2
        # 到底哪个是dst的，哪个是src的 --> 原始正常大小的，应该是src的，scale Match以后的应该是dst的 （或者说TinyPerson的就是dst的）
        dst_distri, dst_sizes = ScaleMatch._get_distribute(json.load(open(dst_anno_file))['annotations'],
                                                           bins, except_rate, use_log_bins, mu_sigma)
        dst_distri_cumsum = np.cumsum(dst_distri)
        src_sizes = MonotonicityScaleMatch.match_distribute(json.load(open(src_anno_file))['annotations'],
                                                            dst_distri_cumsum)
        self.src_sizes = src_sizes
        self.dst_sizes = dst_sizes

        self.default_scale = default_scale
        self.mode = PIL_RESIZE_MODE[mode]
        self.fail_time = 0
        self.scale_range = scale_range  # scale_range[1] to avoid out of memory
        self.out_scale_deal = out_scale_deal
        assert out_scale_deal in ['clip', 'use_default_scale']

        self.debug_no_image_resize = debug_no_image_resize
        self.debug_record = DebugScaleRecord(debug_close_record)

    @staticmethod
    def match_distribute(src_annotations, dst_distri_cumsum):
        annos = [anno for anno in src_annotations if not anno['iscrowd']]
        if len(annos) > 0 and 'ignore' in annos[0]:
            annos = [anno for anno in annos if not anno['ignore']]
        sizes = np.sqrt(np.array([anno['bbox'][2] * anno['bbox'][3] for anno in annos]))
        sizes = sizes[sizes > 0]
        sizes = np.sort(sizes)
        N = len(sizes)
        # 最小的size就是我这个src里面原本最小的size
        src_sizes = [sizes[0]]
        # dst_distri_cumsum:的形式应该是
        # 直方图 [0.1, 0.3, 0.2, 0.4]
        # cumsum [0.1, 0.4, 0.6, 1]
        # p_sum * N 还算是有深意的吧，就是在这个size前面的size总共有这么多
        # 通过这个方法就是把src的这个直方图，肯定变得和dst的形状是一样的了
        # 他就是为了得到这个比例而这样去截取每个bin的取值
        for p_sum in dst_distri_cumsum:
            src_sizes.append(sizes[min(int(p_sum * N), N - 1)])
        if src_sizes[-1] < sizes[-1]:
            src_sizes[-1] = sizes[-1]
        return np.array(src_sizes)

    # 从这里来看这个单调的确实是有点不一样，之前的sample_by_distribute那里都和src_size间没什么关系
    # 这个应该比那个每次都用随机采样的靠谱多了
    def _sample_by_distribute(self, src_size):
        bin_i = np.nonzero(src_size <= self.src_sizes[1:] + 1e-6)[0][0]
        dst_bin_d = self.dst_sizes[bin_i + 1] - self.dst_sizes[bin_i]
        src_bin_d = self.src_sizes[bin_i + 1] - self.src_sizes[bin_i]
        dst_size = (src_size - self.src_sizes[bin_i]) / src_bin_d * dst_bin_d + self.dst_sizes[bin_i]
        return dst_size

    def default_scale_deal(self, image_hw):
        scale = self.default_scaleS
        # resize bbox mean size to our want size
        size = int(round(scale * image_hw[0])), int(round(scale * image_hw[1]))
        return size

    def get_new_size(self, image_hw, bbox, mode='xyxy'):
        """
            image_hw: (h, w) of image
            bbox: bbox of gt, [[x1, y1, x2, y2], ...]
        return:
            new image size: (h, w) of new image
        """
        target = BoxList(deepcopy(bbox), image_hw[::-1], mode)
        if len(target.bbox) == 0:
            return self.default_scale_deal(image_hw)

        # record old target info
        old_mode = target.mode

        # cal mean size of image's bbox
        boxes = target.convert('xywh').bbox.cpu().numpy()
        sizes = np.sqrt(boxes[:, 2] * boxes[:, 3])
        sizes = sizes[sizes > 0]
        src_size = np.exp(np.log(sizes).mean())
        # src_size = sizes.mean()

        # sample a size respect to target distribute
        # For memory stable, we set scale range.
        dst_size = self._sample_by_distribute(src_size)
        scale = dst_size / src_size
        self.debug_record(scale)
        if self.out_scale_deal == 'clip':
            if scale >= self.scale_range[1]:
                scale = self.scale_range[1]  # note 1
            elif scale <= self.scale_range[0]:
                scale = self.scale_range[0]  # note 1
        else:
            if scale >= self.scale_range[1] or scale <= self.scale_range[0]:
                scale = self.default_scale

        # resize bbox mean size to our want size
        size = int(round(scale * image_hw[0])), int(round(scale * image_hw[1]))
        target = target.convert(old_mode)
        target = target.resize((size[1], size[0]))

        # remove too tiny size to avoid unstable loss
        if len(target.bbox) > 0:
            target = target[(target.bbox[:, 2] - target.bbox[:, 0] + 1) >= 2]
            target = target[(target.bbox[:, 3] - target.bbox[:, 1] + 1) >= 2]

        if len(target.bbox) == 0:
            self.fail_time += 1
            if self.fail_time % 1 == 0:
                warnings.warn("Scale Matching failed for {} times, you may need to change the mean to min. "
                              "dst_size is {}, src_size is {}, sizes".format(self.fail_time, dst_size, src_size, sizes))
            return self.default_scale_deal(image_hw)
        return size

    def __call__(self, image: Image, target: BoxList):
        """
            transform format of facebook maskrcnn_benchmark
        """
        size = self.get_new_size([image.height, image.width], target.bbox, target.mode)
        target = target.resize((size[1], size[0]))
        if len(target.bbox) > 0:
            target = target[(target.bbox[:, 2] - target.bbox[:, 0] + 1) >= 2]
            target = target[(target.bbox[:, 3] - target.bbox[:, 1] + 1) >= 2]
        if not self.debug_no_image_resize:
            image = F.resize(image, size, self.mode)
        return image, target


class ReAspect(object):
    def __init__(self, aspects: tuple):
        """
        :param aspects: (h/w, ...)
        """
        self.aspects = aspects

    def __call__(self, image, target):
        target_aspect = random.choice(self.aspects)

        boxes = target.convert('xywh').bbox.cpu().numpy()
        mean_boxes_aspect = np.exp(np.log(boxes[:, 3] / boxes[:, 2]).mean())

        s = (target_aspect / mean_boxes_aspect) ** 0.5

        w, h = image.size
        w = int(round(w / s))
        h = int(round(h * s))
        image = F.resize(image, (h, w))
        target = target.resize(image.size)
        return image, target


class GaussianScaleMatch(MonotonicityScaleMatch):
    def __init__(self, src_anno_file, mu_sigma, bins=100, except_rate=-1.,
                 scale_range=(0., 2.), default_scale=1.0, out_scale_deal='clip',
                 use_log_bins=False, mode='bilinear', standard_gaussian_sample_file=None,
                 use_size_in_image=True, min_size=0,
                 debug_no_image_resize=False, debug_close_record=False):
        """
        1. GaussianScaleMatch use equal area histogram to split bin, not equal x-distance, so [except_rate] are removed.
        2. _get_gaussain_distribute can get simulate gaussian distribute, can can set [min_size] to constraint it.
        3. use log mean size of objects in each image as src distribute, not log size of each object
        :param src_anno_file:
        :param mu_sigma:
        :param bins:
        :param except_rate:
        :param scale_range:
        :param default_scale:
        :param out_scale_deal:
        :param use_log_bins:
        :param mode:
        :param standard_gaussian_sample_file:
        :param debug_no_image_resize:
        :param debug_close_record:
        """
        assert out_scale_deal in ['clip', 'use_default_scale']
        assert use_log_bins, "GaussianScaleMatch need USE_LOG_BINS set to True."
        assert except_rate <= 0, 'GaussianScaleMatch need except_rate < 0'

        if except_rate < 0:
            except_rate = 1. / bins * 2
        mu, sigma = mu_sigma
        dst_distri, dst_sizes = GaussianScaleMatch._get_gaussain_distribute(mu, sigma, bins, except_rate, use_log_bins,
                                                                            standard_gaussian_sample_file, min_size)
        dst_distri_cumsum = np.cumsum(dst_distri)
        src_sizes = GaussianScaleMatch.match_distribute(json.load(open(src_anno_file))['annotations'],
                                                        dst_distri_cumsum, use_size_in_image)
        self.src_sizes = src_sizes
        self.dst_sizes = dst_sizes

        self.default_scale = default_scale
        self.mode = PIL_RESIZE_MODE[mode]
        self.fail_time = 0
        self.scale_range = scale_range  # scale_range[1] to avoid out of memory
        self.out_scale_deal = out_scale_deal

        self.debug_no_image_resize = debug_no_image_resize
        self.debug_record = DebugScaleRecord(debug_close_record)

    @staticmethod
    def _get_gaussain_distribute(mu, sigma, bins=100, except_rate=0.1, use_log_bins=False,
                                 standard_gaussian_sample_file=None, min_size=0):
        """
        except_rate: to except len(annotations)*except_rate/2 abnormal points as head and tial bin
        """
        x = np.load(standard_gaussian_sample_file)
        sizes = x * sigma + mu
        if min_size >= 0:
            sizes = sizes[sizes > min_size]
        sizes = np.sort(sizes)
        N = len(sizes)
        # hist_sizes = sizes[int(N * except_rate / 2): int(N * (1 - except_rate / 2))]
        # if except_rate > 0:
        #     c, s = np.histogram(hist_sizes, bins=bins-2)
        #     c = np.array([int(N * except_rate / 2)] + c.tolist() + [N - int(N * (1 - except_rate / 2))])
        #     s = [sizes[0]] + s.tolist() + [sizes[-1]]
        #     s = np.array(s)
        # else:
        #     c, s = np.histogram(hist_sizes, bins=bins)
        from math import ceil
        step = int(ceil(N / bins))
        last_c = N - step * (bins - 1)
        s = np.array(sizes[::step].tolist() + [sizes[-1]])
        c = np.array([step] * (bins - 1) + [last_c])

        c = c / len(sizes)
        if use_log_bins:
            s = np.exp(s)
        return c, s

    def _sample_by_distribute(self, src_size):
        bin_i = np.nonzero(src_size <= self.src_sizes[1:] + 1e-6)[0][0]
        dst_bin_d = np.log(self.dst_sizes[bin_i + 1]) - np.log(self.dst_sizes[bin_i])
        src_bin_d = np.log(self.src_sizes[bin_i + 1]) - np.log(self.src_sizes[bin_i])
        dst_size = np.exp(
            (np.log(src_size) - np.log(self.src_sizes[bin_i])) / src_bin_d * dst_bin_d + np.log(self.dst_sizes[bin_i])
        )
        return dst_size

    @staticmethod
    def match_distribute(src_annotations, dst_distri_cumsum, use_size_in_image=True):
        def get_json_sizes(annos):
            annos = [anno for anno in annos if not anno['iscrowd']]
            if len(annos) > 0 and 'ignore' in annos[0]:
                annos = [anno for anno in annos if not anno['ignore']]
            sizes = np.sqrt(np.array([anno['bbox'][2] * anno['bbox'][3] for anno in annos]))
            sizes = sizes[sizes > 0]
            return sizes

        def get_im2annos(annotations):
            im2annos = {}
            for anno in annotations:
                iid = anno['image_id']
                if iid in im2annos:
                    im2annos[iid].append(anno)
                else:
                    im2annos[iid] = [anno]
            return im2annos

        def get_json_sizes_in_image(annotations):
            im2annos = get_im2annos(annotations)
            _sizes = []
            for iid, annos in im2annos.items():
                annos = [anno for anno in annos if not anno['iscrowd']]
                if len(annos) > 0 and 'ignore' in annos[0]:
                    annos = [anno for anno in annos if not anno['ignore']]
                sizes = np.sqrt(np.array([anno['bbox'][2] * anno['bbox'][3] for anno in annos]))
                sizes = sizes[sizes > 0]
                size = np.exp(np.log(sizes).mean())
                _sizes.append(size)
            return _sizes

        if use_size_in_image:
            sizes = get_json_sizes_in_image(src_annotations)
        else:
            sizes = get_json_sizes(src_annotations)
        sizes = np.sort(sizes)
        N = len(sizes)
        src_sizes = [sizes[0]]
        for p_sum in dst_distri_cumsum:
            src_sizes.append(sizes[min(int(p_sum * N), N - 1)])
        if src_sizes[-1] < sizes[-1]:
            src_sizes[-1] = sizes[-1]
        return np.array(src_sizes)


class DebugScaleRecord(object):
    def __init__(self, close=False):
        self.debug_record_scales = [inf, -inf]
        self.iters = 0
        self.close = close

    def __call__(self, scale):
        if self.close: return
        self.iters += 1
        last_record_scales = deepcopy(self.debug_record_scales)
        update = False
        if scale > self.debug_record_scales[1] + 1e-2:
            self.debug_record_scales[1] = scale
            update = True
        if scale < self.debug_record_scales[0] - 1e-2:
            self.debug_record_scales[0] = scale
            update = True
        if (update and self.iters > 1000) or self.iters == 1000:
            warnings.warn('update record scale {} -> {}'.format(last_record_scales, self.debug_record_scales))
