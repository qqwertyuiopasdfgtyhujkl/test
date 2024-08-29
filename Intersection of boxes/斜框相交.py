import numpy as np

class CommonFun():
    @staticmethod
    def judgeIntersectionOfDiagonalBoxesSuper(obj1, obj2, iou=0.0, box_type=None, iou_oth=None):
        '''
        判断两输入的斜框是否在偏差为iou的阈值下处于相对相交状态
        对于有特殊扩框要求的标签，可以分类放入box_type内，对应的扩框需求放入iou_oth内，会进行对应处理
        :param obj1:
        :param obj2:
        :param iou:
        :param box_type:
        :param iou_oth:
        :return:
        '''
        box1, box2 = obj1[2][:4], obj2[2][:4]
        # box1, box2 = obj1, obj2
        if not (box1 and box2):
            return False

        if len(box1) < 4 or len(box2) < 4:
            return False

        if (len(box1[0]) < 2 or len(box1[1]) < 2 or len(box1[2]) < 2 or len(box1[3]) < 2 or
                len(box2[0]) < 2 or len(box2[1]) < 2 or len(box2[2]) < 2 or len(box2[3]) < 2):
            return False

        box1, point1_center = CommonFun.points_sort(box1)
        box2, point2_center = CommonFun.points_sort(box2)

        flag_obj1, flag_obj2 = 1, 1
        if box_type:
            if len(iou_oth) < len(box_type):
                iou_oth += [0] * (len(box_type) - len(iou_oth))
            for idx in range(len(box_type)):
                if obj1[0] in box_type[idx]:
                    flag_obj1 = 0
                    # 判断对象1标签框四个顶点是否存在至少1个被对象2包含，有则返回True
                    for point in box1:
                        if CommonFun.judgePointAndRectangle(point, box2 + [point2_center], iou_oth[idx]):
                            return True
                if obj2[0] in box_type[idx]:
                    flag_obj2 = 0
                    # 判断对象2标签框四个顶点是否存在至少1个被对象1包含，有则返回True
                    for point in box2:
                        if CommonFun.judgePointAndRectangle(point, box1 + [point1_center], iou_oth[idx]):
                            return True
        if flag_obj1:
            # 判断对象1标签框四个顶点是否存在至少1个被对象2包含，有则返回True
            for point in box1:
                if CommonFun.judgePointAndRectangle(point, box2 + [point2_center], iou):
                    return True
        if flag_obj2:
            # 判断对象2标签框四个顶点是否存在至少1个被对象1包含，有则返回True
            for point in box2:
                if CommonFun.judgePointAndRectangle(point, box1 + [point1_center], iou):
                    return True
        return False

    @staticmethod
    def judgePointAndRectangle(point, box, iou=0.0):
        '''
        判断点与矩形对象的位置关系（包含还是不包含）， iou表示包含时所容许的边界放大阈值
        求值公式：
            假设两顶点为（a1, a2）,(b1, b2);中心坐标为（o1, o2）;任意点坐标为（x0, y0）
            则在非两点重合或两连线平行情况下，交点坐标为：
            n = (y0-o2)(a1*b2-a2*b1)+(b2-a2)(o2*x0-o1*y0)
            m = (x0-o1)(b2-a2)-(b1-a1)(y0-o2)
            y = n/m
            x = (y-o2)(x0-o1)/(y0-o2)+o1 = (y-a2)(b1-a1)/(b2-a2)+a1
        :param point:
        :param box:内包含矩形四角顶点坐标及其对称中心的坐标，中心坐标为第五个元素值
        :param iou:
        :return:
        '''
        # print('初始输入数据', point, box)
        count = 0
        point_center = box[4]
        box = box[:4]
        for idx in range(2):
            point_a, point_b = box[idx], box[idx - 1]
            if abs(point[0] - point_center[0]) <= 5 and abs(
                    point[1] - point_center[1]) <= 5:  # TODO在中心五个像素范围内的点默认被包含（在同侧），可调整
                # print('当前顶点与中心重合', point_center, point)
                count += 1
                continue
            elif abs(round((point[0] - point_center[0]) * (point_a[1] - point_b[1]) -
                           (point[1] - point_center[1]) * (
                                   point_a[0] - point_b[0]))) <= 0.1:  # TODO点与中心连线与当前检查边平行时，点与中心在同侧，阈值可调
                # print('当前顶点与中心连线平行一边', point_center, point, point_a, point_b, )
                count += 1
                continue
            else:
                # print('判断相关点坐标：', point_center, point, point_a, point_b)
                n = ((point[1] - point_center[1]) * (point_a[0] * point_b[1] - point_a[1] * point_b[0]) +
                     (point_b[1] - point_a[1]) * (point_center[1] * point[0] - point_center[0] * point[1]))
                m = ((point[0] - point_center[0]) * (point_b[1] - point_a[1]) -
                     (point_b[0] - point_a[0]) * (point[1] - point_center[1]))
                y = n / m
                if point_b[1] - point_a[1] == 0:
                    x = ((y - point_center[1]) * (point[0] - point_center[0]) / (point[1] - point_center[1]) +
                         point_center[0])
                else:
                    x = ((y - point_a[1]) * (point_b[0] - point_a[0]) / (point_b[1] - point_a[1]) +
                         point_a[0])

                # print('交点坐标：', (x, y), n, m)
                distance_OP = CommonFun.euclideanDistance(point_center, point)
                distance_OQ = CommonFun.euclideanDistance(point_center, (x, y))
                if distance_OP <= distance_OQ * (1 + iou):
                    count += 1
                    continue
        if count == 2:
            return True
        return False

    @staticmethod
    def euclideanDistance(point1, point2):
        '''
        二位平面欧氏距离计算
        :param point1:
        :param point2:
        :return:
        '''
        if not (point1 and point2):
            return None

        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    @staticmethod
    def points_sort(box):
        """
        将斜框四顶角点按一定相邻的顺序排序, 并给出对角线交点
        :return:
        """
        # 确定对象1右上左下四个顶点
        point1_r, point1_u, point1_l, point1_d = box[0], None, box[0], None
        for point in box[1:4]:
            # 获取最右侧顶点
            if point[0] > point1_r[0]:
                point1_r = point
            elif point[0] == point1_r[0]:
                if point[1] < point1_r[1]: point1_r = point
            # 获取最左侧顶点
            if point[0] < point1_l[0]:
                point1_l = point
            elif point[0] == point1_l[0]:
                if point[1] > point1_l[1]: point1_l = point
        for point in box:
            if point in [point1_r, point1_l]:
                continue
            if not point1_u:
                point1_u = point
                point1_d = point
                continue
            # 获取最上侧顶点
            if point[1] < point1_u[1]:
                point1_u = point
            elif point[1] == point1_u[1]:
                if point[0] < point1_u[0]: point1_u = point
            # 获取最下侧顶点
            if point[1] > point1_d[1]:
                point1_d = point
            elif point[1] == point1_d[1]:
                if point[0] > point1_d[0]: point1_d = point
        point1_center = ((point1_r[0] + point1_l[0]) / 2, (point1_r[1] + point1_l[1]) / 2)
        return [point1_r, point1_u, point1_l, point1_d], point1_center

if "__main__" == __name__:
    box1 = ["lable1", 1, [(10, 100), (130, 40), (150, 80), (30, 140)]]
    box2 = ["lable2", 2, [(80, 50), (200, 50), (200, 110), (80, 110)]]
    res = CommonFun.judgeIntersectionOfDiagonalBoxesSuper(box1, box2)
    print("res : ", res)
    pass