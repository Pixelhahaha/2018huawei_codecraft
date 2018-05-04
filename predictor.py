# coding=utf-8
from __future__ import division
import copy
import datetime
import math
import random
# import matplotlib.pyplot as plt


def predict_vm(ecs_lines, input_lines):
    # Do your work from here#
    result = []
    if ecs_lines is None:
        print "ecs information is none"
        return result
    if input_lines is None:
        print "input file information is none"
        return result

    Train_data, train_end = convert_Train_data(ecs_lines)   # 数据转换成列表
    input_data, predict_start, predict_end = convert_Input_data(input_lines)  # input文件读入
    del(ecs_lines, input_lines)
    Flavor = Everyday_flavor_num(Train_data)  # 各种flavor每天出现次数

    Flavor0 = copy.deepcopy(Flavor)


    all_num = {}
    for i in range(len(Flavor0)):
        myset = set(Flavor0[i])
        cout_num = []
        cout = 0
        for item in myset:
            if item <= 1.6*cout:
                cout_num.append([item, Flavor0[i].count(item)])
            cout += 1
        if len(cout_num) > 2:
            if cout_num[-1][0] > 2*cout_num[-2][0] and cout_num[-2][0] > 5:
                all_num[i] = cout_num[-2][0] + cout_num[-2][1]
                if cout_num[-2][0] > 1.5*(cout_num[-3][0])+1:
                    all_num[i] = cout_num[-3][0]
            else:
                all_num[i] = cout_num[-1][0] + cout_num[-1][1]
        else:
            all_num[i] = cout_num[-1][0] + 2
    # filtering_CFAR(Flavor)   # CFAR滤波
    Flavor = Filtering(Flavor, all_num)

    global coe
    coe = 1.2
    # flavor_input0 = Predict_HoltWinters(Flavor0, input_data)  # 指数平滑 HoltWinters预测
    # flavor_predict0 = dict(flavor_input0)
    day_All_data = len(Flavor[0])
    gap = (predict_start-train_end).days-1
    day_Test = input_data[-1]
    flavor_input = []
    host_num = input_data[0]
    day_start = predict_start.day

    # 一天一天预测
    # if gap%7==0 and day_Test%7==0:
    #     for i in range(host_num + 2, host_num + input_data[host_num + 1] + 2):
    #         a = [Flavor[input_data[i][0] - 1][day_All_data-14:day_All_data-7],
    #              Flavor[input_data[i][0] - 1][day_All_data-7:day_All_data]]
    #         a1 = 0
    #         for one_week in range(int(gap/7+day_Test/7)):
    #             temp = int(0.8*sum(a[-1]) + 0.4*sum(a[-2]))
    #             a.append([temp])
    #             if one_week >= gap/7:
    #                 a1+=temp
    #         flavor_input.append([input_data[i][0], a1])
    # else:

    # gap = 0 时的系数
    alpha_gap0 = []
    for i in range(gap + day_Test):
        if i < 7:
            alpha_gap0.append([0.8, 0.4])
        else:
            alpha_gap0.append([1.1, 0.2])

    # gap + day_test < 21 时的系数
    alpha_gap21 = []
    for i in range(gap + day_Test):
        if i < 4:
            alpha_gap21.append([0.8, 0.4])
        elif i < 17:
            alpha_gap21.append([1.1, 0.2])
        else:
            alpha_gap21.append([1.45, 0.25])

    # gap + day_test < 42 时的系数
    alpha_gap42 = []
    for i in range(gap + day_Test):
        if i < 14:
            alpha_gap42.append([0.8, 0.4])
        else:
            alpha_gap42.append([1.1, 0.2])

    if gap == 0:
        # pass
        for i in range(host_num + 2, host_num + input_data[host_num + 1] + 2):
            for one_day in range(gap + day_Test):
                alpha1 = alpha_gap0[one_day][0]
                alpha2 = alpha_gap0[one_day][1]
                temp = int(round(
                    alpha1 * sum(Flavor[input_data[i][0] - 1][day_All_data - 13 + one_day:day_All_data - 6 + one_day]) +
                    alpha2 * sum(Flavor[input_data[i][0] - 1][day_All_data - 20 + one_day:day_All_data - 13 + one_day])
                    - 1.0 * sum(Flavor[input_data[i][0] - 1][day_All_data - 6 + one_day:day_All_data + one_day])))
                if temp < 0:
                    temp = 0
                Flavor[input_data[i][0] - 1].append(temp)
            a1 = sum(Flavor[input_data[i][0] - 1][day_All_data + gap:day_All_data + gap + day_Test])
            a1 = int(math.ceil(a1 / (alpha1 + alpha2)))
            flavor_input.append([input_data[i][0], a1])
    elif day_Test + gap <= 21:
        for i in range(host_num + 2, host_num + input_data[host_num + 1] + 2):
            for one_day in range(gap + day_Test):
                alpha1 = alpha_gap21[one_day][0]
                alpha2 = alpha_gap21[one_day][1]
                temp = int(round(
                    alpha1 * sum(Flavor[input_data[i][0] - 1][day_All_data - 13 + one_day:day_All_data - 6 + one_day]) +
                    alpha2 * sum(Flavor[input_data[i][0] - 1][day_All_data - 20 + one_day:day_All_data - 13 + one_day])
                    - 1.0 * sum(Flavor[input_data[i][0] - 1][day_All_data - 6 + one_day:day_All_data + one_day])))
                if temp < 0:
                    temp = 0
                # if one_day < 17:
                #     temp = math.ceil(temp / (alpha1 + alpha2))
                # if temp > all_num[input_data[i][0] - 1]:
                #     temp += math.ceil(all_num[input_data[i][0] - 1]/5)
                # temp = int(math.ceil(temp / (alpha1 + alpha2)))
                Flavor[input_data[i][0] - 1].append(temp)
            a1 = sum(Flavor[input_data[i][0] - 1][day_All_data + gap:day_All_data + gap + day_Test])
            # a1 = int(math.ceil(a1 / 1.2))
            flavor_input.append([input_data[i][0], a1])
    else:
        for i in range(host_num + 2, host_num + input_data[host_num + 1] + 2):
            for one_day in range(gap + day_Test):
                alpha1 = alpha_gap42[one_day][0]
                alpha2 = alpha_gap42[one_day][1]
                temp = int(round(
                    alpha1 * sum(Flavor[input_data[i][0] - 1][day_All_data - 13 + one_day:day_All_data - 6 + one_day]) +
                    alpha2 * sum(Flavor[input_data[i][0] - 1][day_All_data - 20 + one_day:day_All_data - 13 + one_day])
                    - 1.0 * sum(Flavor[input_data[i][0] - 1][day_All_data - 6 + one_day:day_All_data + one_day])))
                if temp < 0:
                    temp = 0
                # if one_day < 17:
                #     temp = math.ceil(temp / (alpha1 + alpha2))
                # if temp > all_num[input_data[i][0] - 1]:
                #     temp += math.ceil(all_num[input_data[i][0] - 1]/5)
                Flavor[input_data[i][0] - 1].append(temp)
            a1 = sum(Flavor[input_data[i][0] - 1][day_All_data + gap:day_All_data + gap + day_Test])
            a1 = int(math.ceil(a1 / (alpha1 + alpha2)))
            flavor_input.append([input_data[i][0], a1])



    flavor_predict = dict(flavor_input)

    for keys in flavor_predict:    # 模型等值加权求和
        flavor_predict[keys] = int(1*flavor_predict[keys]+0.0*flavor_predict[keys])

    output, num_ = MyPacking(flavor_predict, input_data)  # 最坏适应 Worst-Fit装箱分配
    flavor_output = []  # 构造输出文件需要的列表
    for index in range(len(flavor_input)):
        flavor_output.append([flavor_input[index][0], flavor_predict[flavor_input[index][0]]])
    result = [flavor_output, output]

    return result, input_data, num_


# # # 统一预测
# if train_end.year % 4 == 0:
#     days_month = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# else:
#     days_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
#
# days_train_month = days_month[train_end.month]
#
# if day_Test + day_start > days_train_month:
#     for i in range(host_num + 2, host_num + input_data[host_num + 1] + 2):
#         a1 = int(
#             2.0 * (sum(Flavor[input_data[i][0] - 1][day_All_data - days_train_month + day_start:day_All_data]) +
#                    sum(Flavor[input_data[i][0] - 1][day_All_data - days_train_month:
#                                                     day_All_data - days_train_month +
#                                                     day_start + day_Test - days_train_month])))
#         flavor_input.append([input_data[i][0], a1])
# else:
#     for i in range(host_num + 2, host_num + input_data[host_num + 1] + 2):
#         a1 = int(2.0 * sum(Flavor[input_data[i][0] - 1][day_All_data - days_train_month + day_start:
#                                                         day_All_data - days_train_month + day_start + day_Test]))
#         flavor_input.append([input_data[i][0], a1])

def Everyday_flavor_num(Train_data):
    Flavor = []  # vm每天出现次数
    for flavor_num in range(18):  # 将input_data转换为每天vm出现次数
        num_list = 0
        d1 = datetime.date(Train_data[0][1], Train_data[0][2], Train_data[0][3])
        d2 = datetime.date(Train_data[-1][1], Train_data[-1][2], Train_data[-1][3])
        day_num = (d2 - d1).days + 1
        Flavor_list = []
        for index in range(day_num):
            Flavor_list.append(0)

        for index in range(len(Train_data) - 1):

            day_now = datetime.date(Train_data[index][1], Train_data[index][2], Train_data[index][3])

            day_num_now = (day_now - d1).days

            if Train_data[index][0] == flavor_num + 1:
                num_list += 1

            if Train_data[index + 1][1:4] != Train_data[index][1:4]:
                Flavor_list[day_num_now] = num_list
                num_list = 0
            else:
                Flavor_list[day_num_now] = num_list

        if Train_data[-1][0] == flavor_num + 1:
            Flavor_list[-1] += 1
        Flavor.append(Flavor_list)
    return Flavor


def filtering_CFAR(Flavor):
    N = len(Flavor[0])
    r = 2
    p = 3
    x = []
    [x.append(i) for i in range(N)]

    for i in range(len(Flavor)):
        average = 0
        for j in range(N):
            average += Flavor[i][j]
        average = average/N
        Z = []
        for j in range(0, r + p):
            Z.append(sum(Flavor[i][j + p + 1:j + p + r]))
        for j in range(r + p, N - r - p):
            Z.append(sum(Flavor[i][j - r - p:j - p - 1] + Flavor[i][j + p + 1:j + r + p]))
        for j in range(N - r - p, N):
            Z.append(sum(Flavor[i][j - r - p:j - p - 1]))

        for j in range(N):
            Z[j] += 2*average
            if Flavor[i][j] > Z[j]:
                Flavor[i][j] = int(Flavor[i][j]/2)
        # plt.plot(x,Z,'r',x,Flavor[i],'g')
        # plt.show()
    return Flavor


def Filtering(Flavor, all_num):
    for i in range(len(Flavor)):
        ave = sum(Flavor[i])/len(Flavor[i])*2.4
        for j in range(len(Flavor[i])-1):
            if Flavor[i][j] > all_num[i]:
                Flavor[i][j] = int(round(ave))
            # if Flavor[i][j] > ave:
            #     Flavor[i][j] = int(ave)
    return Flavor


def Predict_HoltWinters(Flavor, input_data):
    global coe
    def exponential_smoothing(alpha, s):
        s2 = []
        [s2.append(0) for i in range(len(s))]
        s2[0] = (s[0]+s[1]+s[2])/3
        for i in range(1, len(s2)):
            s2[i] = alpha * s[i] + (1 - alpha) * s2[i - 1]
        return s2

    alpha = 0.1  # 设置alphe，即平滑系数
    day_all = len(Flavor[0])
    day_test = input_data[-1]
    flavor_input = []
    for index in range(input_data[1]):
        num = input_data[2 + index][0]-1
        number = []
        for i in range(day_all - day_test + 1):
            number.append(sum(Flavor[num][day_all - day_test - i:day_all - i]))
        number.reverse()
        # times = day_all//day_test
        # for i in range(times):
        #     number.append(sum(Flavor[num][day_all - day_test*(i+1):day_all - day_test*i]))
        # number.reverse()
        s_single = exponential_smoothing(alpha, number)  # 计算一次指数平滑

        s_double = exponential_smoothing(alpha, s_single)  # 计算二次平滑字数，二次平滑指数是在一次指数平滑的基础上进行的，三次指数平滑以此类推
        a_double = [2 * s_single[i] - s_double[i] for i in range(len(s_single))]  # 计算二次指数平滑的a
        b_double = [(alpha / (1 - alpha)) * (s_single[i] - s_double[i]) for i in range(len(s_single))]  # 计算二次指数平滑的b
        pre_next_seven_day1 = a_double[-1] + b_double[-1] * day_test  # 预测未来

        if pre_next_seven_day1 < 0:
            pre_next_seven_day1 = 0
        x1 = int(round(coe*pre_next_seven_day1))
        flavor_input.append([input_data[2 + index][0], x1])
        # plt.plot(number)
        # plt.show()
    return flavor_input


def MyPacking(flavor_predict, input_data):
    dict_ = {1: [1, 1], 2: [1, 2], 3: [1, 4], 4: [2, 2], 5: [2, 4], 6: [2, 8], 7: [4, 4], 8: [4, 8], 9: [4, 16],
             10: [8, 8], 11: [8, 16], 12: [8, 32], 13: [16, 16], 14: [16, 32], 15: [16, 64], 16: [32, 32], 17: [32, 64],
             18: [32, 128]}
    flavor_order = flavor_predict.keys()
    flavor_order.reverse()  # 对所有flavor种类排序，从小到大

    vec_flavors = []
    for keys in flavor_predict:
        for i in range(flavor_predict[keys]):
            vec_flavors.append(keys)
    random.shuffle(vec_flavors)
    CPU_need = 0
    MEM_need = 0
    for i in flavor_order:
        CPU_need += flavor_predict[i] * dict_[i][0]  # flavor 数量×cpu规格
        MEM_need += flavor_predict[i] * dict_[i][1]
    rate = CPU_need / MEM_need
    host_format = {}
    host = {'G': 0, 'L': 0, 'H': 0}
    for i in range(input_data[0]):
        if input_data[i + 1][0] == 'General':
            host_format['G'] = [input_data[i + 1][1], input_data[i + 1][2]]
            host['G'] = int(math.ceil(max(CPU_need / input_data[i + 1][1], CPU_need / input_data[i + 1][2])))
        if input_data[i + 1][0] == 'Large-Memory':
            host_format['L'] = [input_data[i + 1][1], input_data[i + 1][2]]
            host['L'] = int(math.ceil(max(CPU_need / input_data[i + 1][1], CPU_need / input_data[i + 1][2])))
        if input_data[i + 1][0] == 'High-Performance':
            host_format['H'] = [input_data[i + 1][1], input_data[i + 1][2]]
            host['H'] = int(math.ceil(max(CPU_need / input_data[i + 1][1], CPU_need / input_data[i + 1][2])))

    if input_data[0] == 3:  # 只有两种物理服务器规格
        host_ = []
        [host_.append([input_data[i + 1][0], input_data[i + 1][1] / input_data[i + 1][2]]) for i in
         range(input_data[0])]
        length = len(host_)
        while length > 0:
            length -= 1
            cur = 0
            while cur < length:  # 拿到当前元素
                if host_[cur][1] > host_[cur + 1][1]:  # rate从小到大排序
                    host_[cur], host_[cur + 1] = host_[cur + 1], host_[cur]
                cur += 1
        if rate <= host_[0][1]:
            temp1 = host_[1][0]
            temp2 = host_[2][0]
            host[temp1[0]] = 0
            host[temp2[0]] = 0
        elif rate >= host_[0][1] and rate <= host_[1][1]:
            temp1 = host_[2][0]
            host[temp1[0]] = 0
        elif rate >= host_[1][1] and rate <= host_[2][1]:
            temp1 = host_[0][0]
            host[temp1[0]] = 0
        else:
            temp1 = host_[0][0]
            temp2 = host_[1][0]
            host[temp1[0]] = 0
            host[temp2[0]] = 0
    f_x = (CPU_need) ** 2 + (MEM_need) ** 2
    for g_ in range(host['G'] + 1):
        for l_ in range(host['L'] + 1):
            for h_ in range(host['H'] + 1):
                temp = (56 * g_ + 84 * l_ + 112 * h_ - CPU_need) ** 2 + (128 * g_ + 256 * l_ + 192 * h_ - MEM_need) ** 2
                if temp < f_x:  # 56*g_+84*l_+112*h_-CPU_need>=0 and 128*g_+256*l_+192*h_-MEM_need>=0 and temp < f_x:
                    f_x = temp
                    num_ = [g_, l_, h_]
    # num_t
    RH = []  #
    [RH.append(host_format['G']) for i in range(num_[0])]
    [RH.append(host_format['L']) for i in range(num_[1])]
    [RH.append(host_format['H']) for i in range(num_[2])]
    CPU_ = 0
    MEM_ = 0
    for i in range(len(RH)):
        CPU_ += RH[i][0]  # flavor 数量×cpu规格
        MEM_ += RH[i][1]
    Rh = []
    for i in range(len(RH)):
        Rh.append(copy.deepcopy(RH[i]))
    h_ = len(RH)  # 需要的物理服务器最少数量
    h = []
    [h.append([]) for i in range(h_)]
    for i in flavor_order:  # flavor从大到小开始放置
        Rv = [dict_[i][0], dict_[i][1]]  # 每种flavor虚拟机规格

        for j in range(flavor_predict[i]):  # 每种flavor中所有数目
            Rh_local, fig = Packing(Rv, Rh)  # Rh是否有地方可放（flag=1为可放），以及放置位置
            if fig == 1:  # 虚拟机能够放置进去
                Rh[Rh_local][0] = Rh[Rh_local][0] - Rv[0]
                Rh[Rh_local][1] = Rh[Rh_local][1] - Rv[1]
                h[Rh_local].append(i)
            else:
                if RH[-1] == host_format['G']:
                    num_[0] += 1
                elif RH[-1] == host_format['L']:
                    num_[1] += 1
                else:
                    num_[2] += 1
                RH.append(RH[-1])
                temp_Rh = copy.deepcopy(RH[-1])
                Rh.append(temp_Rh)

                h.append([])
                Rh[-1][0] = Rh[-1][0] - Rv[0]
                Rh[-1][1] = Rh[-1][1] - Rv[1]
                h[-1].append(i)
    # # 模拟退火 #######
    vec_flavors = []
    for i in range(len(h)):
        for j in range(len(h[i])):
            vec_flavors.append(h[i][j])
    N = len(vec_flavors)
    h_min = len(vec_flavors)
    Rh_old = copy.deepcopy(Rh)
    del h, Rh
    # RH = []  #
    # [RH.append(host_format['G']) for i in range(num_[0])]
    # [RH.append(host_format['L']) for i in range(num_[1])]
    # [RH.append(host_format['H']) for i in range(num_[2])]

    if input_data[-1] <= 7:
        T = 3000
    elif input_data[-1] <= 14:
        T = 300
    else:
        T = 100

    dice = []
    [dice.append(i) for i in range(N)]

    while T > 0:
        num_ = [0, 0, 0]
        random.shuffle(dice)
        new_vec_flavors = copy.deepcopy(vec_flavors)
        new_vec_flavors[dice[0]], new_vec_flavors[dice[1]] = new_vec_flavors[dice[1]], new_vec_flavors[dice[0]]
        # new_vec_flavors[dice[2]], new_vec_flavors[dice[3]] = new_vec_flavors[dice[3]], new_vec_flavors[dice[2]]
        temp_RH = copy.deepcopy([RH[0]])
        temp_Rh = copy.deepcopy(RH[0])

        if temp_Rh == host_format['G']:
            num_[0] += 1
        elif temp_Rh == host_format['L']:
            num_[1] += 1
        else:
            num_[2] += 1

        Rh = [temp_Rh]  # 开辟物理服务器，物理服务器规格（CPU：个，MEM：G）
        h = [[]]
        order_RH = 1
        for i in range(N):
            Rv = dict_[new_vec_flavors[i]]
            Rh_local, fig = Mylittle(Rv, Rh)  # 比较Rv是否比Rh中某个值小，并找出位置
            if fig:
                Rh[Rh_local][0] = Rh[Rh_local][0] - Rv[0]
                Rh[Rh_local][1] = Rh[Rh_local][1] - Rv[1]
                h[Rh_local].append(new_vec_flavors[i])
            else:
                if order_RH < len(RH):
                    temp_RH.append(RH[order_RH])
                    temp_Rh = copy.deepcopy(temp_RH[order_RH])

                    if temp_Rh == host_format['G']:
                        num_[0] += 1
                    elif temp_Rh == host_format['L']:
                        num_[1] += 1
                    else:
                        num_[2] += 1

                    Rh.append(temp_Rh)
                else:
                    temp_RH.append(RH[0])
                    if RH[0] == host_format['G']:
                        num_[0] += 1
                    elif RH[0] == host_format['L']:
                        num_[1] += 1
                    else:
                        num_[2] += 1
                    temp_Rh = copy.deepcopy(RH[0])
                    Rh.append(temp_Rh)
                order_RH += 1
                h.append([])
                Rh[-1][0] = Rh[-1][0] - Rv[0]
                Rh[-1][1] = Rh[-1][1] - Rv[1]
                h[-1].append(new_vec_flavors[i])
        h_num_cpu = 0
        for i in range(len(Rh) - 1):
            h_num_cpu += Rh[i][0]
        h_num_mem = 0
        for i in range(len(Rh) - 1):
            h_num_mem += Rh[i][1]
        h_num = (h_num_cpu + h_num_mem)  # 除最后一个物理服务器外的碎片CPU，MEM和

        if (h_num < h_min):  # or random.random() < math.exp(-(h_num - h_min) / T):
            h_min = h_num
            vec_flavors = copy.deepcopy(new_vec_flavors)
            res_h = copy.deepcopy(h)  # res_h较优的放置结果
            res_Rh = copy.deepcopy(Rh)  # new_Rh 较优的物理服务器剩余情况
            res_num = copy.deepcopy(num_)
        T -= 1
    del Rh, h, num_
    Rh = res_Rh  # 更新物理服务器剩余
    h = res_h  # 更新虚拟机放置结果
    num_ = res_num
    #####填充最后一个物理服务器#######
    h_last_rate_cpu = Rh[-1][0] / RH[-1][0]
    h_last_rate_mem = Rh[-1][1] / RH[-1][1]

    if h_last_rate_cpu + h_last_rate_mem <= 1.7 or len(h[-1]) > 3:  # 填充
        h_predict_last = {}  # 最后一个host需要的最少每种flavor数目
        for keys in flavor_predict:
            h_predict_last[keys] = int(math.floor(min(Rh[-1][0] / dict_[keys][0], Rh[-1][1] / dict_[keys][1])))
        vec_flavor_last = []  # 需要的每种flavor放置成列表
        for keys in h_predict_last:  # h_predict_last中所有vm放入一个列表下
            for i in range(h_predict_last[keys]):
                vec_flavor_last.append(keys)
        vec_flavor_last.reverse()  # flavor从大到小放置
        N1 = len(vec_flavor_last)
        new_vec_flavors_last = copy.deepcopy(vec_flavor_last)
        temp_h = copy.deepcopy(h)
        for i in range(N1):
            if Rh[-1][0] - dict_[new_vec_flavors_last[i]][0] >= 0 and \
                    Rh[-1][1] - dict_[new_vec_flavors_last[i]][1] >= 0:
                Rh[-1][0] = Rh[-1][0] - dict_[new_vec_flavors_last[i]][0]
                Rh[-1][1] = Rh[-1][1] - dict_[new_vec_flavors_last[i]][1]
                temp_h[-1].append(new_vec_flavors_last[i])
        new_h = copy.deepcopy(temp_h)
        for i in range(len(h[-1]), len(new_h[-1])):
            flavor_predict[new_h[-1][i]] += 1
    else:  # 舍弃
        for item in h[-1]:
            flavor_predict[item] -= 1
        if RH[-1] == host_format['G']:
            num_[0] -= 1
        elif RH[-1] == host_format['L']:
            num_[1] -= 1
        else:
            num_[2] -= 1
        del h[-1]
        del Rh[-1]
        new_h = copy.deepcopy(h)
    hh = []
    for index in range(len(h)):
        temp = new_h[index]
        temp.sort(reverse=True)
        hh.append([[temp[0], 1]])

        for ii in range(len(new_h[index]) - 1):
            if new_h[index][ii + 1] == new_h[index][ii]:
                hh[index][-1][1] += 1
            else:
                hh[index].append([new_h[index][ii + 1], 1])
    return hh, num_


def Packing(Rv, Rh):
    fig = 0
    rate = []
    for i in range(len(Rh)):
        if Rh[i][0]-Rv[0] > 0 and Rh[i][1]-Rv[1] > 0:
            rate.append([i, abs(float(Rh[i][0] / Rh[i][1]) - float(Rv[0] / Rv[1])), Rh[i][0] - Rv[0]])
        elif (Rh[i][0] - Rv[0] == 0 and Rh[i][1] - Rv[1] == 0)\
                or (Rh[i][0] - Rv[0] >= 0 and Rh[i][1] - Rv[1] >= 0 and Rv == [1, 1]) \
                or (Rh[i][0] - Rv[0] >= 0 and Rh[i][1] - Rv[1] >= 0 and Rv == [1, 2]) \
                or (Rh[i][0] - Rv[0] >= 0 and Rh[i][1] - Rv[1] >= 0 and Rv == [1, 4]) \
                or (Rh[i][0] - Rv[0] >= 0 and Rh[i][1] - Rv[1] >= 0 and Rv == [2, 2]):
            return i, 1
        else:
            rate.append([i, float("inf"), 0])  # rate 无穷大，剩余空间为0
    Rate = copy.deepcopy(rate)
    length = len(Rate)
    while length > 0:
        length -= 1
        cur = 0
        while cur < length:  # 拿到当前元素
            if Rate[cur][1] > Rate[cur + 1][1]:  # rate从小到大排序
                Rate[cur], Rate[cur + 1] = Rate[cur + 1], Rate[cur]
            cur += 1
    length = len(Rate)
    while length > 0:
        length -= 1
        cur = 0
        while cur < length:  # 拿到当前元素
            if Rate[cur][1] == Rate[cur + 1][1] and Rate[cur][2] < Rate[cur + 1][2]:  # 剩余空间从大到小排序
                Rate[cur], Rate[cur + 1] = Rate[cur + 1], Rate[cur]
            cur += 1
    if Rate[0][1] != float("inf"):
        return Rate[0][0], 1
    else:
        return 0, 0


def Mylittle(Rv, Rh):
    fig = 0
    for i in range(len(Rh)):
        if Rv[0] <= Rh[i][0] and Rv[1] <= Rh[i][1]:
            fig = 1
            break
    return i, fig


def convert_Train_data(ecs_lines):
    Train_data = []
    for index, item in zip(range(len(ecs_lines)), ecs_lines):
        values0 = item.split(" ")[0]
        values = values0.split('\t')
        temp = []
        temp.append(values[1].split('flavor')[1])
        temp.append(values[2].split('-')[0])
        temp.append(values[2].split('-')[1])
        temp.append(values[2].split('-')[2])
        temp = map(int, temp)
        Train_data.append(temp)
    return Train_data, datetime.date(Train_data[-1][1], Train_data[-1][2], Train_data[-1][3])


def convert_Input_data(input_lines):
    Input_data = []
    num = int(input_lines[0].split('\n')[-2])
    Input_data.append(num)
    for i in range(num):
        temp = input_lines[i+1].split('\n')[-2]
        temp1 = temp.split(' ')
        temp2 = map(int, temp1[1:])
        temp2.insert(0, temp1[0])
        # temp2 = [].extend(temp1)
        Input_data.append(temp2)  # 硬件

    num0 = input_lines[num+2]
    num1 = int(num0.split('\n')[-2])
    Input_data.append(num1)   # 虚拟机数量

    for i in range(num1):
        line0 = input_lines[i+num+3]
        line1 = line0.split('\n')[-2]
        line2 = line1.split('flavor')[1]
        line3 = map(int, line2.split(' '))
        Input_data.append(line3)        # 虚拟机类别

        # 读取预测时间
    line0 = input_lines[num1+num+4]
    year_begin = int(line0[0:4])
    month_begin = int(line0[5:7])
    day_begin = int(line0[8:10])
    d1 = datetime.date(year_begin, month_begin, day_begin)

    line1 = input_lines[num1+num+5]
    year_end = int(line1[0:4])
    month_end = int(line1[5:7])
    day_end = int(line1[8:10])
    d2 = datetime.date(year_end, month_end, day_end)
    all_day_num = (d2 - d1).days+1
    Input_data.append(all_day_num)
    return Input_data, d1, d2