import torch as th
import numpy as np
import random
from PIL import Image
# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# pip install opencv-python
import cv2
import torchvision.transforms as transforms


# Based on https://github.com/ArrowLuo/CLIP4Clip
class RawVideoExtractorCV2():
    """Implementation of the raw video preprocessing.
        Params:
            size: Processed image's width and height, in pixel. If param transform_type = 0 and
                the original image is greater than this value, it will be resized and center cropped. Default: 224
            framerate: sampling rate in second. Default: 1.0
            type:
                0: default transformation;
                1: transformation for objects, iou, temporal, action;
                2: transformation for i3d;. Default: 0
            sample_tpye:
                'fps': sampling in order
                'k_split':k segment
        """

    def __init__(self, size=224, framerate=-1, transform_type=0,
                 sample_type='k_split',
                 k_segments=20,
                 extract_loc='start',
                 random_seed=42):
        self.size = size
        self.framerate = framerate
        self.transform_type = transform_type  # 选择不同的transform方法转换成tensor
        self.transform = self._transform(self.size)
        self.sample_type = sample_type
        self.k_segments = k_segments
        self.extract_loc = extract_loc
        self.random_seed = random_seed
        
    def _transform(self, n_px):
        if self.transform_type == 0:
            return Compose([
                Resize(n_px, interpolation=Image.BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        # objects, iou, temporal, action
        elif self.transform_type == 1:
            return Compose([transforms.ToTensor()])
        # i3d
        elif self.transform_type == 2:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

            return Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None, patch=0,
                        overlapped=0):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        # Samples a frame sample_fp X frames.
        cap = cv2.VideoCapture(video_file)
        # 视频总帧数
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.sample_type == 'fps':
            # fps帧率，fps帧/s
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # 这个公式用于计算视频总时长（秒数），其核心原理是通过整数运算实现向上取整的效果
            # 如 25帧率 fps为24帧/s 那么如果25//24= 1剩下的1s只有1帧但是也要算进去，fps-1是为了向上取整
            # print("fps=",fps) debug
            # print('video_file=',video_file) debug
            total_duration = (frameCount + fps - 1) // fps
            # 开始时间，结束时间
            start_sec, end_sec = 0, total_duration

            if start_time is not None:
                # 确保结束时间不超过总total_duration市场
                start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
                # 如果开始时间不None，则start_time * fps就是start_time的第一帧
                # 如果stat_time == 0 fps == 24 则，开始帧的索引为 0
                # 如果stat_time == 1 fps == 24 则，开始帧的索引为 24,也就是第二秒的第一帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

            interval = 1  # 间隔为1
            if sample_fp > 0:  # 如果采样率 > 0
                # 采样帧间隔 为 视频帧率 fps // 采样帧 sample_fp
                # 为什么呢？因为fps代表一秒内的帧数
                # 视频fps，代表1s内有fps个帧
                # sample_fp代表1s内要采样多少帧
                # 比如视频fps = 8
                # sample_fp = 4
                # 则采样间隔为2#
                interval = fps // sample_fp
            else:  # 如果采样sample_fp<=0 ，则以原视频帧率进行采样
                sample_fp = fps
            # 如果interval == 0，则还是以原视频帧率进行采样，
            # 这种情况是因为sample_fp > fps导致interval == 0
            if interval == 0:
                interval = 1  # fps//sample_fp 说明采样率和视频帧率相同

            # np.arrange(0,fps,interval) [0,fps) 一秒内的帧的id，interval为间隔+1  如interval = 2
            # 每interval个进行一次采样
            # 0 2 4 8
            # 这是一秒内的所有帧的id
            inds = [ind for ind in np.arange(0, fps, interval)]
            assert len(inds) >= sample_fp  # 什么样会出现这个情况呢？
            # 一秒内采样的帧数id个数，也就是帧数>1s内采样的帧数
            # 如果sample_fp=3
            # fps = 10
            # interval = 3
            # inds = [0,3,6,9]
            # 但是实际上我最多一秒内只需要采样sample_fp个帧
            inds = inds[:sample_fp]

            # cv2读取到帧的判断
            ret = True
            # 图像帧，
            images, included = [], []

            for sec in np.arange(start_sec, end_sec + 1):  # 遍历视频的每一秒
                if not ret: break  # 读取失败就退出
                sec_base = int(sec * fps)  # 每一秒第一帧的初始位置
                for ind in inds:
                    # 每一秒要采样的帧的位置sec_base + ind
                    cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
                    ret, frame = cap.read()  # 读取
                    if not ret:  # 读取失败，原本是if not ret:break，因为发现710第一帧读不进来，所以多改了一帧
                        ret, frame = cap.read()
                        if not ret:
                            break
                    # [H,W,C]
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换成RGB格式

                    smaller_edge = min(frame_rgb.shape[0], frame_rgb.shape[1])  # 选取一个小的边
                    frame_rgb = np.array(CenterCrop(smaller_edge)(Image.fromarray(frame_rgb).convert("RGB")))
                    patches = []

                    if patch > 1:  # patch表示的是，长宽均匀分成几块，如patch=3，那么就是一个九宫格
                        for i in range(patch):  # 遍历patch的行
                            x_crop_start = i * int(frame_rgb.shape[0] / patch)  # patch_i行的起始位置
                            if overlapped > 0 and i != 0:  # 有重叠的话并且i!=0也就是不在第一行，就往左边移动重叠比例的patch大小
                                x_crop_start -= int((frame_rgb.shape[0] / (patch)) * overlapped)

                            x_crop_end = (i + 1) * int(frame_rgb.shape[0] / patch)  # patch_i 行的结束位置
                            if overlapped > 0:  # 这里不考虑最后一航，如果不是最后一行，就加上重叠大小
                                x_crop_end += int((frame_rgb.shape[0] / (patch)) * overlapped)
                            if i == patch - 1:  # 如果是最后一航的话，end就是底边
                                x_crop_end = frame_rgb.shape[0]

                            for j in range(patch):  # 遍历patch的列
                                y_crop_start = j * int(frame_rgb.shape[1] / patch)  # 这个同理，开始位置就是默认不重叠的位置
                                if overlapped > 0 and j != 0:  # 不是第一列，并且有重叠，就往前移动
                                    y_crop_start -= int((frame_rgb.shape[1] / (patch)) * overlapped)

                                y_crop_end = (j + 1) * int(frame_rgb.shape[1] / patch)  # 默认无重叠，就是对应的列底边
                                if overlapped > 0:  # 如果有重叠，这个overlap是重叠率
                                    # 前提是这个不是最后一列的右边，但是呢如果是也没关系，因为下面那个会单独判断矫正
                                    y_crop_end += int((frame_rgb.shape[1] / (patch)) * overlapped)  # 就向右扩展重叠边
                                if j == patch - 1:  # 如果是最后以列的右边，则赋值为最后一列
                                    y_crop_end = frame_rgb.shape[1]

                                cropped_frame = frame_rgb[x_crop_start:x_crop_end, y_crop_start:y_crop_end, :]  # 把patch裁下来
                                # 裁下来然后
                                patches.append(preprocess(Image.fromarray(cropped_frame).convert("RGB")))
                                # 在维度0新增，然后原本的[patches_size,patches_size,c]
                                # 处理玩一张图像以后
                                # 变成了[patches_nums,patches_size,patches_size,c]
                                # 然后images的shape就是[n, #n个图片
                                #                   patches_nums, #每个图片的patches个数
                                #                   patches_size,patches_size,3] # 每个图片的patches的shape
                        images.append(np.stack(patches))
                    else:
                        # 否则就是不需要patch，直接进行preprocess
                        # [n,h,w,c]
                        # 这里的preprocess是transform转换成tensor,不是clip的preprocess
                        images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

            cap.release()  # 释放cap对象，不需要读取frame

            if len(images) > 0:  # 如果抽取的帧>1
                # images是个列表
                # 里面每个元素都是tensor
                # np.stack就把它转化成numpy array的形式，然后才能转化成tensor
                # 返回的video_data如果是一整张图片stack的话就是
                # [n, h, w, c]
                # 返回的video_data如果是patches的话就是
                # [n, patches_nums, patches_size, patches_size, c]
                video_data = th.tensor(np.stack(images))
            else:
                video_data = th.zeros(1)  # 读取失败
            return {'video': video_data}, video_data.shape
        elif self.sample_type == 'k_split':
            k = self.k_segments
            segment_size = frameCount // k
            images = []
            for i in range(k):
                start_frame = i * segment_size
                end_frame = (i + 1) * segment_size if i < k - 1 else frameCount

                # 根据抽帧位置计算具体帧
                target_frame = start_frame
                if self.extract_loc == "mid":
                    target_frame = (start_frame + end_frame) // 2
                elif self.extract_loc == "end":
                    target_frame = max(start_frame, end_frame - 1)
                elif self.extract_loc == "tsn":
                    # TSN模式：在每个段内随机采样一帧
                    # 设置随机种子确保结果可复现
                    random.seed(self.random_seed + i)  # 每个段使用不同的种子
                    np.random.seed(self.random_seed + i)
                    if start_frame < end_frame - 1:
                        target_frame = random.randint(start_frame, end_frame - 1)
                    else:
                        target_frame = start_frame

                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()
                if not ret:  # 读取失败，原本是if not ret:break，因为发现710第一帧读不进来，所以多改了一帧
                    ret, frame = cap.read()
                    if not ret:
                        break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                smaller_edge = min(frame_rgb.shape[0], frame_rgb.shape[1])
                frame_rgb = np.array(CenterCrop(smaller_edge)(Image.fromarray(frame_rgb).convert("RGB")))

                # 添加分块处理逻辑
                patches = []
                if patch > 1:  # patch表示的是，长宽均匀分成几块，如patch=3，那么就是一个九宫格
                    for i in range(patch):  # 遍历patch的行
                        x_crop_start = i * int(frame_rgb.shape[0] / patch)  # patch_i行的起始位置
                        if overlapped > 0 and i != 0:  # 有重叠的话并且i!=0也就是不在第一行，就往左边移动重叠比例的patch大小
                            x_crop_start -= int((frame_rgb.shape[0] / (patch)) * overlapped)

                        x_crop_end = (i + 1) * int(frame_rgb.shape[0] / patch)  # patch_i 行的结束位置
                        if overlapped > 0:  # 这里不考虑最后一航，如果不是最后一行，就加上重叠大小
                            x_crop_end += int((frame_rgb.shape[0] / (patch)) * overlapped)
                        if i == patch - 1:  # 如果是最后一航的话，end就是底边
                            x_crop_end = frame_rgb.shape[0]

                        for j in range(patch):  # 遍历patch的列
                            y_crop_start = j * int(frame_rgb.shape[1] / patch)  # 这个同理，开始位置就是默认不重叠的位置
                            if overlapped > 0 and j != 0:  # 不是第一列，并且有重叠，就往前移动
                                y_crop_start -= int((frame_rgb.shape[1] / (patch)) * overlapped)

                            y_crop_end = (j + 1) * int(frame_rgb.shape[1] / patch)  # 默认无重叠，就是对应的列底边
                            if overlapped > 0:  # 如果有重叠，这个overlap是重叠率
                                # 前提是这个不是最后一列的右边，但是呢如果是也没关系，因为下面那个会单独判断矫正
                                y_crop_end += int((frame_rgb.shape[1] / (patch)) * overlapped)  # 就向右扩展重叠边
                            if j == patch - 1:  # 如果是最后以列的右边，则赋值为最后一列
                                y_crop_end = frame_rgb.shape[1]

                            cropped_frame = frame_rgb[x_crop_start:x_crop_end, y_crop_start:y_crop_end, :]  # 把patch裁下来
                            # 裁下来然后
                            patches.append(preprocess(Image.fromarray(cropped_frame).convert("RGB")))
                            # 在维度0新增，然后原本的[patches_size,patches_size,c]
                            # 处理玩一张图像以后
                            # 变成了[patches_nums,patches_size,patches_size,c]
                            # 然后images的shape就是[n, #n个图片
                            #                   patches_nums, #每个图片的patches个数
                            #                   patches_size,patches_size,3] # 每个图片的patches的shape
                    images.append(np.stack(patches))
                else:
                    images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))
            cap.release()  # 释放cap对象，不需要读取frame

            if len(images) > 0:  # 如果抽取的帧>1
                # images是个列表
                # 里面每个元素都是tensor
                # np.stack就把它转化成numpy array的形式，然后才能转化成tensor
                # 返回的video_data如果是一整张图片stack的话就是
                # [n, h, w, c]
                # 返回的video_data如果是patches的话就是
                # [n, patches_nums, patches_size, patches_size, c]
                video_data = th.tensor(np.stack(images))
            else:
                video_data = th.zeros(1)  # 读取失败
            return {'video': video_data}, video_data.shape
        else:
            raise ValueError('Unknown sample type: {}'.format(self.sample_type))

    def get_video_data(self, video_path, start_time=None, end_time=None, patch=0, overlapped=0):
        # image_input 是 {'video',video_frames_tensor的字典}
        # video_path是视频文件路径
        # self.transform
        image_input, shapes = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate,
                                                   start_time=start_time, end_time=end_time, patch=patch,
                                                   overlapped=overlapped)
        # image_input
        # 如果是patch则为[frames, patch_nums,patch_size,patch_size,channel]
        # 如果不进行patch则为[frames, h, w, channel]
        # shape就只有上面两种情况
        return image_input, shapes

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        # tensor_size只有两种情况
        # 1: [frames, h, w, c]
        # 2: [frames, patch_nums, patch_size, patch_size, c]

        # 这里大概是第一种情况
        # 也就是需要变成   [frames, 1, h, w, c]
        # 如果是第二种情况 [frames*patches, 1, ps, ps, c]
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 这个raw_video_data是video tensor
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:  # 正常顺序
            pass
        elif frame_order == 1:  # 反顺序
            # size0是总帧数
            # size0-1~0就是所有帧的order
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:  # 随机顺序
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data


# An ordinary video frame extractor based CV2
RawVideoExtractor = RawVideoExtractorCV2