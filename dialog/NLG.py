#!/usr/bin/env python3
# coding=utf-8
"""
给定系统动作，输出自然语言
"""
import random

def list2str(list):
    sent = ""
    for ii in list:
        if ii !="今年" and ii !="去年" and ii !="前年":
            sent = sent + ii + "、"
    sent = sent.rstrip("、")
    return sent



def rule_NLG(system_acts, KB_pointer):
    """
    基于规则的自然语言生成
    :param system_acts: 例如 [
                    {
                        "diaact": "inform",
                        "requestable_slots": [
                            "简介"
                        ]
                    }
                ]
    :param KB_pointer: 用户当前锁定的一部电影
    :return: 一段话
    """
    for act in system_acts:
        if act["diaact"] == "inform":
            rand = random.random()
            sent = KB_pointer["片名"]
            # if rand < 0.4:
            #     sent = "该电影"
            # elif rand > 0.6:
            #     sent = KB_pointer["片名"] + " "
            # else:
            #     sent = ""
            for slot in act["requestable_slots"]:
                if slot == "导演":
                    if random.random() >0.5:
                        sent = sent + "由 " + KB_pointer["导演"] + "执导，"
                    else:
                        sent = sent + "导演是 " + KB_pointer["导演"] + "，"
                elif slot == "主演":
                    if random.random() >0.5:
                        sent = sent + "主演阵容是 " + list2str(KB_pointer["主演"]) + "，"
                    else:
                        sent = sent + "主演有" + list2str(KB_pointer["主演"]) + "，"
                elif slot == "地区":
                    if random.random() > 0.5:
                        sent = sent + "在" + list2str(KB_pointer["地区"]) + " 等地区拍摄，"
                    else:
                        sent = sent + "是" + list2str(KB_pointer["地区"]) + " 等地区的，"
                elif slot == "类型":
                    if random.random() > 0.5:
                        sent = sent + "类型为" + list2str(KB_pointer["类型"]) + "，"
                    else:
                        sent = sent + "是" + list2str(KB_pointer["类型"]) + "类型的，"
                elif slot == "年代":
                    if random.random() > 0.5:
                        sent = sent + "是" + list2str(KB_pointer["年代"]) + "年代的电影，"
                    else:
                        sent = sent + "年代为" + list2str(KB_pointer["年代"]) + "，"
                elif slot == "资费":
                    if list2str(KB_pointer["资费"]) == "免费":
                        sent = sent + "是免费电影，"
                    else:
                        sent = sent + "需要付费，"
                elif slot == "片长":
                    sent = sent + "片长为" + KB_pointer["片长"] + "分钟，"
                elif slot == "评分":
                    sent = sent + "评分为" + KB_pointer["评分"] + "，"
                elif slot == "简介":
                    rand = random.random()
                    if  rand < 0.4:
                        sent = sent + "讲述的是:" + KB_pointer["简介"]
                    elif rand > 0.7:
                        sent= sent + "剧情梗概是:" + KB_pointer["简介"]
                    else:
                        sent = sent + "简介是:" + KB_pointer["简介"]

                elif slot == "上映日期":
                    if random.random() >0.5:
                        sent = sent + "上映日期是" + KB_pointer["上映日期"] + "，"
                    else:
                        sent = sent + "于" + KB_pointer["上映日期"] + "上映，"
            sent = sent.rstrip("，")

        elif act["diaact"] == "request":
            sent = ""
            if act["informable_slots"][0] == "主演":
                rand = random.random()
                if rand < 0.3:
                    sent += "想看由谁主演的？"
                elif rand > 0.7:
                    sent += "有喜欢的主演吗？"
                else:
                    sent += "对主演有什么要求呀？"
            elif act["informable_slots"][0] == "导演":
                rand = random.random()
                if rand < 0.3:
                    sent += "想看由谁导演的电影？"
                elif rand > 0.7:
                    sent += "有喜欢的导演吗？"
                else:
                    sent += "对导演有木有要求？"
            elif act["informable_slots"][0] == "类型":
                rand = random.random()
                if rand < 0.3:
                    sent += "想看什么类型的电影？"
                elif rand > 0.7:
                    sent += "请问有喜欢的电影类型吗？"
                else:
                    sent += "对电影的类型有什么偏好"
            elif act["informable_slots"][0] == "地区":
                rand = random.random()
                if rand < 0.5:
                    sent += "想看哪个地区的？"
                else:
                    sent += "对地区有要求没？"
            elif act["informable_slots"][0] == "年代":
                rand = random.random()
                if rand < 0.3:
                    sent += "哪个年代的？"
                elif rand > 0.7:
                    sent += "有喜欢的电影年代吗？"
                else:
                    sent += "那您对年代有啥要求？"
            elif act["informable_slots"][0] == "资费":
                rand = random.random()
                if rand < 0.3:
                    sent += "想看付费的还是免费的？"
                elif rand > 0.7:
                    sent += "那请问对资费有什么要求"
                else:
                    sent += "喜欢付费还是免费？"

        elif act["diaact"] == "inform_one_match":
            sent = "找到一部电影：" + list2str(act["片名"]) + "，您想知道该电影的哪些信息？"

        elif act["diaact"] == "inform_two_match":
            sent = "为您找到两部电影：" + list2str(act["片名"]) + "，您想看哪一部？"

        elif act["diaact"] == "inform_three_match":
            sent = "有三部电影符合要求：" + list2str(act["片名"]) + "，您想看哪一部？"

        elif act["diaact"] == "inform_many_match":
            sent = "有多部电影符合要求：" + list2str(act["片名"]) + "，您想看哪一部？"
        elif act["diaact"] == "inform_no_match":
            sent = "对不起没有找到符合您要求的电影"

        elif act["diaact"] == "repeat":
            sent = "抱歉，能再重述一遍么？"
    return sent



if __name__ == '__main__':
    system_acts = [
                    {
                        "diaact": "repeat"
                    }
                ]
    KB_pointer = {
        "id": "10",
        "上映日期": "2015-8-7",
        "主演": [
            "彭于晏",
            "窦骁",
            "王珞丹",
            "陈家乐",
            "欧阳娜娜"
        ],
        "地区": [
            "香港",
            "中国大陆"
        ],
        "导演": "林超贤",
        "年代": [
            "一零",
            "前年"
        ],
        "片名": "破风",
        "片长": "90",
        "简介": "《破风》将讲述了4个年轻人加入单车队参加公路自行车赛的故事，而所谓破风，指的是在高速骑行下，自行车手需要通过前面一位领骑自行车手的配合下，减小空气阻力，节省后面自行车手的体力，是比赛中团队精神的体现.",
        "类型": [
            "爱情",
            "剧情"
        ],
        "评分": "八点四",
        "资费": "免费"
    }
    print(rule_NLG(system_acts,KB_pointer))


