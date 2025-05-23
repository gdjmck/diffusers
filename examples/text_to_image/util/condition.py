import random
import traceback
import cv2
import math
import os
import json
import glob
import shutil
import pyproj
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import CRS
from shapely.geometry import Polygon, LineString, Point
from shapely.affinity import scale, translate
from geopandas import GeoSeries
from PIL import Image

DEBUG = False


def longtitude_to_coord(lon, lat):
    """

    :param lon: 经度
    :param lat: 纬度
    :return: x, y坐标
    """
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    return transformer.transform(lon, lat)


def coord_to_longtitude(x, y):
    """

    :param x: 坐标x值
    :param y: 坐标y值
    :return: 经度, 纬度
    """
    transformer = pyproj.Transformer.from_crs('EPSG:3857', 'EPSG:4326')
    return transformer.transform(x, y)


def extract_outloop(img):
    def get_area(pts):
        try:
            return Polygon(pts).area
        except ValueError:
            return 0

    contour, hierachy = cv2.findContours(cv2.threshold(img, 250, 1, cv2.THRESH_BINARY_INV)[1],
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if hierachy.shape[1] > 1:  # 有多个轮廓
        contour = [c for c in contour if c.shape[0] > 2]  # 去除 点与线
        if len(contour) > 1:  # 只保留面积最大的
            contour = max(contour, key=lambda pts: get_area(pts))
    #     assert hierachy.shape[1] == 1, "找轮廓时可能由于外轮廓不连续，导致有多个"
    contour = np.array(contour).reshape(-1, 2).tolist()
    if contour[0] != contour[-1]:
        contour.append(contour[0])
    return contour


def open_op(img_mask):
    return cv2.dilate(cv2.erode(img_mask, np.ones((3, 3)), 3), np.ones((3, 3)), 3)


class PostProcess:
    def __init__(self, max_size, tolerance=1.0, field_size=300):
        """

        :param max_size:
        :param tolerance:
        :param field_size: 图片对应真实地块的尺寸（米）
        """
        self.tolerance = tolerance
        self.field_size = field_size
        self.max_size = max_size  # 用于y轴反转
        self.minBuildArea = 25
        self.segment_to_color = {0: 200,
                                 1: 180,
                                 2: 160,
                                 3: 140,
                                 4: 120,
                                 5: 100,
                                 6: 80,
                                 7: 60,
                                 8: 40,
                                 9: 20,
                                 10: 0}
        self.init()

    def init(self):
        self.clear()

    def clear(self):
        self.field_loop = None
        self.img = None
        self.mask = None
        self.num_comp = -1
        self.bg_id = -1
        self.building_list = []
        self.floor_list = []
        self.build2maskId = {}
        self.scale = 0  # 一个像素代表的面积

    def color2floor(self, color_val: int):
        # 高度映射颜色 color_val = 220 - 2 * floor
        # floor height = 3.0
        return max(1, math.ceil((220 - color_val) / 6))

    def set_scale(self):
        h, w = self.img.shape[:2]
        assert h == w, f"h={h} \t w={w}"
        self.scale = (self.field_size / h) ** 2

    def process(self, img, type='real'):
        """
        从图片解析建筑，得到建筑轮廓与层数（color2floor）
        结果:
            轮廓list：self.building_list
            层数list: self.floor_list
        """
        self.clear()
        self.img = img
        self.set_scale()

        outloop = extract_outloop(img)
        self.field_loop = LineString(outloop)
        mask_outloop = np.zeros_like(img)
        mask_outloop = cv2.drawContours(mask_outloop, np.array(outloop).reshape(-1, 1, 2), -1, (255,), 1)
        mask_outloop = cv2.dilate(mask_outloop, np.ones((3, 3)), 1)
        img[mask_outloop > 0] = 255

        if type == 'real':
            img_bin = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)[1]
        else:
            img_bin = cv2.adaptiveThreshold(img, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                            thresholdType=cv2.THRESH_BINARY_INV, blockSize=25, C=5)

        flat_area = open_op(img_bin) & img_bin
        # todo： flat_area减掉地块轮廓区域
        self.num_comp, self.mask = cv2.connectedComponents(flat_area, connectivity=4)
        self.bg_id = self.mask[0, 0]

        # 提取每个楼栋的轮廓
        for i in range(self.num_comp):
            if i == self.bg_id:
                continue
            mask = self.mask == i
            cover_area = mask.sum()
            # 过滤面积小于25㎡
            if cover_area * self.scale < self.minBuildArea:
                continue
            if self.contain_multiple_object(mask):
                pass
                # print('轮廓索引{}包含多个建筑'.format(len(self.building_list)))

            self.extract_build(self.mask == i)

    def __is_valid_geom(self, poly: Polygon):
        return poly.area * self.scale >= self.minBuildArea

    def contain_multiple_object(self, mask):
        color_vals = self.img[mask]
        color_stats = Image.fromarray(color_vals).getcolors(color_vals.shape[0])
        if len(color_stats) < 2:
            return False
        # 按频数高到低排序
        color_stats = sorted(color_stats, key=lambda item: item[0], reverse=True)
        # 颜色数量比例不超过5: 1 && 颜色差值大于20
        color_most, color_second_most = color_stats[:2]
        if color_most[0] / color_second_most[0] <= 5 and abs(color_most[1] - color_second_most[1]) > 20:
            # 暂且认为只会有
            return True
        else:
            return False, None

    def add_build(self, build_loop: list, mask):
        mask_val = np.median(self.mask[mask])
        self.build2maskId[len(self.building_list)] = mask_val  # 用于调试时查看楼栋对应的mask，self.mask == build.index
        self.building_list.append(build_loop)
        self.floor_list.append(self.color2floor(np.median(self.img[mask])))

    def extract_build(self, mask):
        # val = np.median(self.img[mask])
        loop, contours_info = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        loop = self.simplify(loop[0])
        # 楼栋轮廓不能与红线相交
        loops = self.seperate(loop)
        """
        # 判断楼栋轮廓内是否只有一种楼
        multi_build, colors = self.is_multipart(mask)
        multi_build = False  # 先屏蔽裙楼
        if multi_build:
            # 区分不同楼栋
            # 假设存在不同楼栋的情况只有内部包含另外一种楼
            mask1 = mask & (self.img == colors[0][1])
            mask2 = mask & (self.img == colors[1][1])
            hole1 = PostProcess.find_hole(mask1)
            hole2 = PostProcess.find_hole(mask2)
            assert any([len(hole1), len(hole2)])

            mask_w_hole = mask1
            mask_wo_hole = mask2
        else:
            self.add_build(loop, mask=mask)
        """
        for loop in loops:
            self.add_build(loop, mask=mask)

        return loops

    def to_output(self, real_center, standard_size, coord2longlat=False):
        """
        构造返回数据
        """
        outloop = self.building_list
        field_size = (self.field_loop.bounds[2] - self.field_loop.bounds[0],
                      self.field_loop.bounds[3] - self.field_loop.bounds[1])
        scaler = min([s_real / s_gen for s_gen, s_real in zip(field_size, standard_size)], key=lambda v: abs(v - 1))
        field_poly = Polygon(self.field_loop)
        bias_dict = {'xoff': real_center[0] - field_poly.centroid.x,
                     'yoff': real_center[1] - field_poly.centroid.y}
        # 恢复偏移
        for i in range(len(outloop)):
            outloop_ls = LineString(np.array(outloop[i]))
            outloop_ls = scale(translate(outloop_ls, **bias_dict),
                               scaler, scaler, origin=list(real_center) + [0.0])
            outloop[i] = np.array(outloop_ls.coords).tolist()
            if coord2longlat:
                # 坐标转回经纬度
                outloop[i] = [coord_to_longtitude(*coord) for coord in outloop[i]]
        # # 把生成的地块轮廓也加上
        # field_outloop = scale(translate(self.field_loop, **bias_dict), scaler, scaler, origin=list(real_center)+[0.0])
        # print('生成的地块轮廓重心:{}'.format(Polygon(field_outloop).centroid))
        # outloop.append(np.array(field_outloop.coords).tolist())
        return {'outloop': outloop, 'floor': self.floor_list}

    ################# 工具函数 #############################

    def is_multipart(self, mask):
        """
        判断图像img的掩码mask中是否有多种颜色（多种建筑）
        """
        color_list = self.img[mask]
        colors = Image.fromarray(color_list).getcolors(color_list.shape[0])
        # 取数量最多的2个颜色
        colors = sorted(colors, key=lambda item: item[0], reverse=True)
        color_most, color_2nd_most = colors[:2]
        # 最多颜色数量 / 第二多颜色数量 < 5 && abs(最多颜色 - 第二多颜色) > 30 (跨度超过1个色域)
        if color_most[0] / color_2nd_most[0] < 5 and abs(color_most[1] - color_2nd_most[1]) > 30:
            return True, colors
        else:
            return False, colors

    @classmethod
    def find_hole(cls, mask):
        hole_list = []
        try:
            pts, hierachy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        except TypeError:
            mask = mask.astype(np.uint8)
            pts, hierachy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        hierachy = hierachy.reshape(-1, 4)
        children_index = np.where(hierachy[:, -1] != -1)[0]
        for child_index in children_index:
            child = pts[child_index].reshape(-1, 2)
            if child.shape[0] < 3:
                continue
            child_area = Polygon(child).area

            parent_index = hierachy[child_index][3]
            if hierachy[parent_index][-1] != -1:  # 只取最外层的子轮廓
                print('非外层的子轮廓')
                continue
            parent = pts[parent_index].reshape(-1, 2)
            parent_area = Polygon(parent).area
            child_area_over_parent = child_area / parent_area
            print('child面积占比{:.2f}'.format(child_area_over_parent))
            if child_area_over_parent > 0.2:
                child = child.tolist()
                if child[0] != child[-1]:
                    child.append(child[0])
                hole_list.append(child)
        print('内部空洞个数:{}'.format(len(hole_list)))
        return hole_list

    def seperate(self, loop: list):
        """
        使楼栋轮廓与地块红线分离开
        Returns:
            List[outloop points]
        """
        loop_poly = Polygon(loop).buffer(-0.01)
        if max([Point(coord).distance(self.field_loop) for coord in loop_poly.exterior.coords]) < 3:
            # 当前轮廓是红线的一部分
            return []

        if not loop_poly.intersects(self.field_loop):
            return [loop]
        while loop_poly.distance(self.field_loop) <= 0.01:
            loop_poly = loop_poly.buffer(-0.01)
        if isinstance(loop_poly, Polygon):
            return [list(loop_poly.exterior.simplify(2 * self.tolerance).coords)]
        else:
            # 从多个Polygon中选大小合适的
            return [list(poly.exterior.simplify(2 * self.tolerance).coords) for poly in loop_poly.geoms
                    if isinstance(poly, Polygon) and self.__is_valid_geom(poly)]

    def simplify(self, contour):
        contour = contour.reshape(-1, 2).tolist()
        if contour[0] != contour[-1]:
            contour.append(contour[0])
        # 最小距离为toerlance
        contour = LineString(contour).simplify(self.tolerance)
        contour = list(contour.coords)

        return contour

    def plot(self):
        ax = plt.gca()
        ax.invert_yaxis()
        GeoSeries([self.field_loop] + [LineString(build) for build in self.building_list]).plot(ax=ax)


def to_geojson(data: dict):
    """
    排楼结果转成shp文件
    :param layout_result: json 路径
    :param shpfiles_dir: 存储shp文件位置
    :param shpfiles_name:shp名字
    :return:
    """
    geo = {
        'geometry': [Polygon(outloop) for outloop in data['outloop']],
        'floor': data['floor']
    }

    s = gpd.GeoDataFrame(geo, crs="EPSG:3857")
    # 定义目标坐标系
    target_crs = CRS.from_epsg(4326)
    # 将原始数据集的坐标系转换为目标坐标系
    s = s.to_crs(target_crs)
    return json.loads(s.to_json())


def probe(val, stdvar, max_range: float = 0.1):
    seed = random.randint(1, 10) / 10
    sign = random.randint(0, 1) == 1
    return val + sign * seed * max_range / stdvar


class Condition:
    condition_dict = {'s': 'fieldSize', 'v': 'volRat', 'n': 'buildNum', 'f': 'avgFloors', 'd': 'density'}
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, opt):
        if self._initialized:
            return

        self.condition_history = {}
        self.opt = opt
        self.condition_size = opt.condition_size
        # 所有条件label必须在condition_dict中
        assert all([c in Condition.condition_dict for c in list(opt.condition_order)])
        self.condition_name = np.array([Condition.condition_dict[c] for c in list(opt.condition_order)])
        self.condition_mask = np.ones(len(Condition.condition_dict)).astype(bool)
        self.condition_mask[self.condition_size:] = False
        try:
            with open(opt.condition_norm, 'r', encoding='utf-8') as f:
                condition_norm = json.load(f)
            self.condition_mean = np.array(condition_norm['mean'])
            self.condition_stdvar = np.array(condition_norm['stdvar'])
        except:
            self.condition_mean = np.array([0] * self.condition_size)
            self.condition_stdvar = np.array([1] * self.condition_size)
        # 条件的均值标准差按condition_order进行重新排序
        self.reorder_index_list = self.condition_order_mask()
        self.condition_mean = self.reorder(self.condition_mean)
        self.condition_stdvar = self.reorder(self.condition_stdvar)
        # 后处理模块的解析器
        self.parser = PostProcess(256, field_size=self.opt.field_max_size)

        # 原始json数据计算条件
        self.condition_json = None
        try:
            if os.path.exists(opt.condition_json):
                with open(opt.condition_json, 'r') as f:
                    condition_json = json.load(f)
                # 将列表转换为_id为key的字典
                self.condition_json = {}  # id: [boundary, buildings]
                for item in condition_json:
                    self.condition_json[item['_id']] = {key: item[key] for key in ['boundary', 'buildings']}
        except:
            print(traceback.format_exc())

        # 过滤不需要的条件
        self.condition_mean = self.condition_mean[self.condition_mask]
        self.condition_stdvar = self.condition_stdvar[self.condition_mask]
        self.MAX_AREA = 90000
        self.COLOR_MAP = {i: 220 - 2 * i for i in range(1, 101)}  # 高度与颜色的映射
        self.floor_choice = [h // 3 for h in self.COLOR_MAP.keys() if h % 3 == 0]
        self.STANDARD_SIZE = 512

    def reorder(self, condition: np.array):
        return condition[self.reorder_index_list]

    def condition_order_mask(self):
        """
        [地块大小, 平均建筑层数, 地块密度, 建筑数量, 容积率]
        :param condition_size:
        :return: opt.condition_order对默认条件顺序的索引
        """
        default_order = list('sfdnv')
        return [default_order.index(c) for c in list(self.opt.condition_order)]

    def update_mean_and_stdvar(self):
        """
        根据已记录在案的数据计算均值与标准差
        """
        data = np.empty((len(self.condition_history), self.condition_size), dtype=np.float32)
        for i, condition in enumerate(self.condition_history.values()):
            data[i] = np.array(condition) * self.condition_stdvar + self.condition_mean

        # print('旧均值:{}\n旧标准差为:{}'.format(self.condition_mean.tolist(),
        #                                       self.condition_stdvar.tolist()))
        mean_update = data.mean(0)
        var_update = data.std(0)
        # print('更新的均值为:{}\n标准差为:{}'.format(mean_update, var_update))

    def get_mask(self, mask_all, floor: int):
        """
        按层数获取相应的掩码
        :param mask_all:
        :param floor:
        :return:
        """
        height_lb = math.ceil((floor - 0.5) * 3)
        height_ub = math.ceil((floor + 0.5) * 3) - 1
        color_range = (self.COLOR_MAP[height_ub], self.COLOR_MAP[height_lb])
        return ((color_range[0] <= mask_all) & (mask_all <= color_range[1])).astype(np.uint8)

    def open_op(self, img_mask, kernel_size=3):
        return cv2.dilate(cv2.erode(img_mask, np.ones((kernel_size, kernel_size)), 3),
                          np.ones((kernel_size, kernel_size)), 3)

    def mask_dt_distance(self, mask, mask_dt):
        # 计算mask的点中距离最小值
        return mask_dt[mask].min()

    def parse_image(self, img):
        """
        Return:
            {
                1: [outloop_pts1, ..., outloop_ptsN],
                ...
            }
        """
        # 楼栋面积阈值
        area_thr = (5 / 300 * img.shape[0]) ** 2  # 对应真实地块中的25㎡
        # 记录每种层高楼栋的轮廓列表
        floor_obj_map = {}
        # 红线的距离变换矩阵，用于去除地块内部的颜色异常点
        field_outloop = self.extract_outloop(img)
        mask_field = np.ones_like(img, dtype=np.uint8)
        for pt_idx in range(len(field_outloop) - 1):
            pt_0, pt_1 = field_outloop[pt_idx], field_outloop[pt_idx + 1]
            cv2.line(mask_field, pt_0, pt_1, 0, lineType=cv2.LINE_AA)
            mask_field[pt_0[1], pt_0[0]] = 0
            mask_field[pt_1[1], pt_1[0]] = 0
        if DEBUG:
            plt.imsave('field.png', cv2.bitwise_not(mask_field))
        dtm = cv2.distanceTransform(mask_field * 255, cv2.DIST_L1, cv2.DIST_MASK_PRECISE)

        mask_realloc = np.zeros_like(img)
        floor_dict = {}
        floor_dt_dict = {}
        for floor in self.floor_choice:
            floor_obj_map[floor] = []
            # 获取所有层高为floor的楼栋
            mask_floor = self.get_mask(img, floor)
            mask_floor_clean = self.open_op(mask_floor)
            if DEBUG:
                plt.imsave('floor_{}.png'.format(floor), mask_floor_clean)
            floor_dict[floor] = mask_floor_clean
            floor_dt_dict[floor] = cv2.distanceTransform((1 - mask_floor_clean) * 255, cv2.DIST_L1,
                                                         cv2.DIST_MASK_PRECISE)
            mask_free = (mask_floor_clean == 0) & (mask_floor > 0)  # 被清掉的点
            # 判断是否为红线（距离红线小于3），不是红线则去除
            num_clear_comp, mask_clear_comp = cv2.connectedComponents(mask_free.astype(np.uint8), connectivity=4)
            for i in range(1, num_clear_comp):
                if self.mask_dt_distance(mask=mask_clear_comp == i, mask_dt=dtm) < 2:  # 到红线最短距离小于2，认为是红线的一部分
                    mask_free[mask_clear_comp == i] = 0
            mask_realloc |= mask_free

        # 对偏离的坐标重新赋值
        num_comp, label_mask_realloc = cv2.connectedComponents(mask_realloc.astype(np.uint8), connectivity=4)
        for realloc_id in range(1, num_comp):
            mask = label_mask_realloc == realloc_id
            # 找距离最近的楼栋进行分配
            target_floor_type, min_dist = None, np.inf
            for floor_type in floor_dt_dict.keys():
                d = self.mask_dt_distance(mask_dt=floor_dt_dict[floor_type], mask=mask)
                if d < min_dist:
                    min_dist = d
                    target_floor_type = floor_type

            if target_floor_type is not None:
                mask_realloc[mask] = 0
                floor_dict[target_floor_type][mask] = 1

        # 清空其余非前景区域
        mask_non_building = np.ones_like(img)
        # 去除红线区域
        mask_field_buffer = cv2.dilate(1 - mask_field, kernel=np.ones((5, 5)))
        mask_non_building[mask_field_buffer > 0] = 0
        # 去除建筑
        for build_type_mask in floor_dict.values():
            mask_non_building[build_type_mask > 0] = 0
        # 清空图片的非前景区域
        img[mask_non_building > 0] = 255
        if DEBUG:
            plt.imsave('p1_mask.png', mask_non_building)
            plt.imsave('p1.png', img)

        for floor in floor_dict:
            # seperate 每个楼栋
            components = cv2.connectedComponents(floor_dict[floor], connectivity=4)

            for label in range(1, components[0]):
                mask_build = (components[1] == label).astype(np.uint8)
                if mask_build.sum() > area_thr:
                    contour, _ = cv2.findContours(mask_build, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contour = np.array(contour).reshape(-1, 2).tolist()
                    if contour[0] != contour[-1]:
                        contour.append(contour[0])
                    floor_obj_map[floor].append(contour)
                else:
                    # print('建筑面积太小{}被过滤'.format(mask_build.sum()))
                    mask_build[mask_field_buffer > 0] = 0
                    img[mask_build > 0] = 255

        return img, floor_obj_map

    def extract_outloop(self, img, type_flag='real'):
        if type_flag == 'real':
            img_binary = cv2.threshold(img, 250, 1, cv2.THRESH_BINARY_INV)[1]
        else:
            img_binary = cv2.adaptiveThreshold(img, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                               thresholdType=cv2.THRESH_BINARY_INV, blockSize=25, C=5)
        contour, hierachy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if hierachy.shape[1] > 1:  # 有多个轮廓
            contour = [c for c in contour if c.shape[0] > 2]  # 去除 点与线
            if len(contour) > 1:  # 只保留面积最大的
                contour = max(contour, key=lambda pts: cv2.contourArea(np.array(pts)))
        #     assert hierachy.shape[1] == 1, "找轮廓时可能由于外轮廓不连续，导致有多个"
        contour = np.array(contour).reshape(-1, 2).tolist()
        if contour[0] != contour[-1]:
            contour.append(contour[0])
        return contour

    def read_condition_from_json(self, file):
        split_char = '/' if '/' in file else '\\'
        fid = file.rsplit(split_char, 1)[-1].rsplit('.', 1)[0]
        if self.condition_json and fid in self.condition_json:
            raw_data = self.condition_json[fid]
            field_size = cv2.contourArea(np.array(raw_data['boundary'], dtype=np.float32))
            condition_values = []
            for c in self.opt.condition_order[:self.opt.condition_size]:
                if c == 's':  # 地块大小
                    condition_values.append(field_size)
                elif c == 'f':  # 平均层数
                    avg_floor = np.array([build['floor'] for build in raw_data['buildings']]).mean()
                    condition_values.append(avg_floor)
                elif c == 'd':  # 密度
                    cover_area = sum([cv2.contourArea(np.array(build['coords'], dtype=np.float32)) for build in
                                      raw_data['buildings']]) \
                                 / field_size
                    condition_values.append(cover_area)
                elif c == 'n':  # 建筑数量
                    num_builds = len(raw_data['buildings'])
                    condition_values.append(num_builds)
                elif c == 'v':  # 容积率
                    volume_ratio = sum([cv2.contourArea(np.array(build['coords'], dtype=np.float32)) * build['floor']
                                        for build in raw_data['buildings']]) / field_size
                    condition_values.append(volume_ratio)
            return condition_values
        else:
            return False

    """
    def cal_condition(self, file, real_flag=True):
        '''
        计算原始条件 [地块大小, 平均建筑层数, 地块密度, 建筑数量, 容积率]
        '''
        img = cv2.imread(file)
        if len(img.shape) == 3:
            img = img[..., 0]
        if img.shape[0] != self.STANDARD_SIZE:
            fx = fy = self.STANDARD_SIZE / img.shape[0]
            img = cv2.resize(img, dsize=None, fx=fx, fy=fy)
        img, build_info = self.parse_image(img)
        # 根据解析结果计算容积率
        field_area = cv2.contourArea(np.array(self.extract_outloop(img)).reshape(-1, 2))
        volume_area = 0  # 计容面积
        cover_area = 0  # 占地面积
        num_builds = 0  # 建筑数量
        floor_list = []
        for floor, outloop_list in build_info.items():
            floor_list += [floor] * len(outloop_list)
            num_builds += len(outloop_list)
            for outloop in outloop_list:
                outloop_cover = cv2.contourArea(np.array(outloop))
                volume_area += outloop_cover * floor
                cover_area += outloop_cover

        volume_rate = volume_area / field_area
        density = cover_area / field_area
        floor_avg = float(np.mean(floor_list)) if floor_list else 0

        condition_array = np.array([field_area, floor_avg, density, num_builds, volume_rate])
        condition_array = self.reorder(condition_array)

        return condition_array[self.condition_mask].tolist()
    """

    def cal_condition(self, file, file_type='fake'):
        """
        后处理图片后解析条件值
        return:
            dict('n': 建筑数量, 'v': 容积率, 'd': 密度, 'f': 平均层数, 's': 地块大小)
        """
        img = cv2.imread(file)
        if len(img.shape) == 3:
            img = img[..., 0]
        self.parser.process(img, file_type)
        # 建筑基底面积列表
        base_area_lst = [Polygon(build_loop).area for build_loop in self.parser.building_list]
        # 层数列表
        floor_lst = self.parser.floor_list
        # 地块轮廓LineString
        field_ls = self.parser.field_loop

        # 栋数
        num_build = len(base_area_lst)
        # 地块大小
        field_size = Polygon(field_ls).area
        assert field_size > 0
        # 建筑密度
        density = sum(base_area_lst) / field_size
        # 容积率
        volume_rate = sum(map(lambda item: item[0] * item[1], zip(base_area_lst, floor_lst))) / field_size
        # 平均层数
        avg_floor = sum(floor_lst) / num_build

        condition_array = np.array([field_size * self.parser.scale, avg_floor, density, num_build, volume_rate])
        condition_array = self.reorder(condition_array)

        return condition_array[self.condition_mask].tolist()

    def get_volume_rate(self, file):
        condition = self.get(file)
        vr = condition[-1]
        return (vr * self.condition_stdvar[-1]) + self.condition_mean[-1]

    def get(self, file, file_type='fake'):
        """
        计算图片文件的容积率
        :param file:
        :return: 1-D numpy.array
        """
        if not os.path.exists(file):
            return 0
        elif file in self.condition_history:
            return self.condition_history[file]
        else:
            # 生成并记录
            condition = None
            if self.condition_json:  # 从json中读取
                condition = self.read_condition_from_json(file)
            if not condition:  # 从图片中读取
                condition = self.cal_condition(file, file_type)
            # z-score
            condition = (np.array(condition) - self.condition_mean) / self.condition_stdvar
            self.condition_history[file] = condition
            print('add one condition', len(self.condition_history))
            return condition

    def read_condition(self, input_list):
        return (np.array(input_list) * self.condition_stdvar + self.condition_mean).astype(np.float32)
