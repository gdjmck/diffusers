import os
import glob
import numpy as np
import json
from PIL import Image
from util.condition import Condition

# 描述模板
caption_template = "A layout plan of residential community, with a total of {} buildings, a floor area ratio of {:.2f}, a building density of {:.1f}%, and an average number of floors of {:.1f}"


def full_caption(num_buildings: int, floor_area_ratio: float, density: float, average_floors: float):
    return caption_template.format(int(num_buildings), floor_area_ratio, density * 100, average_floors)


# 条件解析工具
class Options:
    condition_size: int = 4
    condition_order: str = 'nvdfs'
    condition_norm: str = './util/mean&stdvar.json'
    field_max_size: int = 500
    condition_json = ''


condition_parser = Condition(Options)


def parse_condition(image_file: str, mask_file: str = ''):
    """通过排布图片+掩码图片精确计算条件"""
    assert os.path.exists(image_file)

    try:
        condition_nvdf = condition_parser.cal_condition(image_file, file_type='real')
    except ZeroDivisionError:
        print(image_file)
        return [0, 0, 0, 0]

    # 若有建筑掩码，计算更精确的建筑数量
    if mask_file and os.path.exists(mask_file):
        try:
            num_building_precise = condition_parser.cal_condition(mask_file)[0]
            condition_nvdf[0] = num_building_precise
        except ZeroDivisionError:
            print(f'Failed to parse building count from mask file: {mask_file}')

    return condition_nvdf


def caption_for_file(image_file: str, mask_file: str = ''):
    condition = parse_condition(image_file, mask_file=mask_file)
    caption = full_caption(*condition)
    return caption


class DataFormer:
    def __init__(self, root: str, image_folder: str, condition_folder: str):
        """
        root:
            - subfolder1:
                - image_folder:
                    [list of image files]
                - condition_folder:
                    [list of image files]

        Args:
            image_folder: 图片文件夹的名称
            condition_folder: 条件图片文件夹的名称
        """
        # 为方便后续批量操作，把数据存放与字典
        self.root = root
        self.image_key = image_folder
        self.condition_key = condition_folder
        self.caption_key = 'caption'
        self.data = {self.image_key: [], self.condition_key: [], self.caption_key: []}

        self.cities = os.listdir(root)
        # 分别找出所有目标图片、条件图片、描述图片
        for subfolder in self.cities:
            image_files = glob.glob(os.path.join(root, subfolder, image_folder, '*.png'))
            condition_image_files = glob.glob(os.path.join(root, subfolder, condition_folder, '*.png'))

            assert len(image_files) == len(condition_image_files)

            # 把图片数据分别存入self.image_list, self.condition_image_list, self.caption_list
            self.data[self.image_key].extend(image_files)
            self.data[self.condition_key].extend(condition_image_files)

    def make_caption(self):
        # 根据image data生成文字描述
        raise NotImplementedError

    def split_train_test(self, train_ratio: float):
        """将self.data分为{'train': subset(self.data), 'test': subset(self.data)}"""
        if train_ratio > 1:
            train_ratio /= 100
        assert len(np.unique([len(data) for data in self.data.values()])) == 1, '{}数据长度不一致'.format(
            list(self.data.keys()))
        data_size = len(self.data[self.image_key])
        train_idxs = np.random.choice(range(data_size), int(np.floor(data_size * train_ratio)))
        test_idxs = [idx for idx in range(data_size) if idx not in train_idxs]

        train_data = {key: [val[idx] for idx in train_idxs] for key, val in self.data.items()}
        test_data = {key: [val[idx] for idx in test_idxs] for key, val in self.data.items()}

        return {'train': train_data, 'test': test_data}


class BuildingLayoutDataFormer(DataFormer):
    def __init__(self, root: str, image_folder: str, condition_folder: str, image_mask_folder, condition_json: str=''):
        """
        root:
            - subfolder1:
                - image_folder:
                    [list of image files]
                - condition_folder:
                    [list of image files]
                - image_mask_folder: (optional)
                    [list of image files]

        Args:
            image_folder: 图片文件夹的名称
            condition_folder: 条件图片文件夹的名称
            image_mask_folder: 建筑掩码图片的文件夹名称
            condition_json: 预先计算好的条件字典 {pattern: condition}
        """
        super().__init__(root, image_folder, condition_folder)
        self.image_mask_key = image_mask_folder
        """
        self.data[self.image_mask_key] = []
        # 加载建筑掩码图片
        for subfolder in self.cities:
            image_mask_files = glob.glob(os.path.join(root, subfolder, image_mask_folder, '*.png'))
            self.data[self.image_mask_key].extend(image_mask_files)
        try:
            assert len(self.data[self.image_key]) == len(self.data[
                                                             self.image_mask_key]), f'image: {len(self.data[self.image_key])} \tmask: {len(self.data[self.image_mask_key])}'
        except AssertionError:
            more_key = max([self.image_key, self.image_mask_key], key=lambda x: len(self.data[x]))
            less_key = min([self.image_key, self.image_mask_key], key=lambda x: len(self.data[x]))
            for file in self.data[more_key]:
                if file.replace(more_key, less_key) not in self.data[less_key]:
                    print(f'no paired data: {file}')
        """

        if os.path.exists(condition_json):
            with open(condition_json, 'r') as f:
                self.condition_json = json.load(f)
        else:
            self.condition_json = {}
        self.data[self.caption_key] = self.make_caption()

    """
    def make_caption(self):
        caption_list = []
        for file_image, file_mask in zip(self.data[self.image_key], self.data[self.image_mask_key]):
            pattern = file_image.replace(self.root, '')
            if pattern in self.condition_json:
                # 从json读取
                caption = self.condition_json[pattern]
            else:
                # 重新计算
                caption = caption_for_file(file_image, file_mask)
            caption_list.append(caption)
            # try:
            #     caption_list.append(caption_for_file(file_image, file_mask))
            # except:
            #     print(file_image)
        return caption_list
    """
    
    def make_caption(self):
        caption_list = []
        for file_image in self.data[self.image_key]:
            pattern = file_image.replace(self.root, '')
            # ▒~Njson读▒~O~V
            caption = self.condition_json[pattern]
            caption_list.append(caption)
        return caption_list

