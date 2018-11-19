#!/usr/bin/env python3
# coding=utf-8
"""
对话策略：给定DST，搜索数据库，生成对话操作
"belief_states": {
                    "年代": {
                        "年代": "NOT_MENTIONED"
                    },
                    "资费": {
                        "资费": "NOT_MENTIONED"
                    },
                    "片名": {
                        "片名": "NOT_MENTIONED"
                    },
                    "主演": {
                        "主演": "NOT_MENTIONED"
                    },
                    "地区": {
                        "地区": "DONT_CARE"
                    },
                    "导演": {
                        "导演": "NOT_MENTIONED"
                    },
                    "类型": {
                        "悬疑": "DISLIKE",
                        "类型": "MENTIONED"
                    }
                }
            }
"""
import  json,pprint


def search_DB(belief_states, DB):
    """
    查找数据库
    :param belief_states: EDST 结果
    :param DB: 数据库，list
    :return: searching results 例如：
    "KB_results": {
                    "片名": [
                        "湄公河行动",
                        "让子弹飞",
                        "三傻大闹宝莱坞"
                    ],
                    "数量": 54
                },
    """
    all_movies = set()
    for movie in DB:
        all_movies.add(movie["片名"])
    choose_movie = {
        "年代": {
            "LIKE": set(),
            "DISLIKE": set()
        },
        "资费": {
            "LIKE": set(),
            "DISLIKE": set()
        },
        "片名": {
            "LIKE": set(),
            "DISLIKE": set()
        },
        "主演": {
            "LIKE": set(),
            "DISLIKE": set()
        },
        "地区": {
            "LIKE": set(),
            "DISLIKE": set()
        },
        "导演": {
            "LIKE": set(),
            "DISLIKE": set()
        },
        "类型": {
            "LIKE": set(),
            "DISLIKE": set()
        }
    }
    for slot in belief_states:
        if belief_states[slot][slot] == "MENTIONED":
            for value in  belief_states[slot]:
                if belief_states[slot][value] == "DISLIKE":
                    choose_movie[slot]["DISLIKE"] =  choose_movie[slot]["DISLIKE"]|find_moive(slot,value,DB)
                elif belief_states[slot][value] == "LIKE":
                    choose_movie[slot]["LIKE"] = choose_movie[slot]["LIKE"]|find_moive(slot, value, DB)
        elif belief_states[slot][slot] == "DONT_CARE":
            choose_movie[slot]["LIKE"] = all_movies

    # pprint.pprint(choose_movie, width=50)
    LIKED = all_movies
    DISLIKED = set()
    if len(choose_movie["片名"]["LIKE"]) >0:
        LIKED = choose_movie["片名"]["LIKE"]
    else:
        for slot in choose_movie:
            if len(choose_movie[slot]["LIKE"]) > 0:
                LIKED = LIKED & choose_movie[slot]["LIKE"]
            DISLIKED = DISLIKED | choose_movie[slot]["DISLIKE"]
    print(LIKED-DISLIKED)
    return LIKED - DISLIKED

def find_moive(slot, value, DB):
    """
    找出满足 slot=value 的所有电影
    :param slot: 槽
    :param value: 值
    :return: 电影的set
    """
    movie_set = set()
    for movie in DB:
        if type(movie[slot]) == list:
            if value in movie[slot]:
                movie_set.add(movie["片名"])
        else:
            if value == movie[slot]:
                movie_set.add(movie["片名"])
    return movie_set

def rule_method(belief_states,turn_num, requested_slots, KB_pointer, DB):
    """
    基于规则的policy，顺着问NOT_MENTIONED 的informable slots，KB_poiter 填充后可问 requestable slot
    :param belief_states: EDST
    :param turn_num: 轮数
    :param requested_slots: requestable slots 是否被询问
    :param KB_pointer: 用户当前锁定的一部电影
    :param DB:数据库
    :return: system_acts  例如：
    "system_acts": [
                    {
                        "diaact": "inform",
                        "requestable_slots": [
                            "简介"
                        ]
                    }
                ]
    """
    system_acts = []
    inquire_slots = ["类型","主演","年代","资费","地区","导演"]
    KB_results = search_DB(belief_states, DB)
    if len(KB_results) == 1:
        for item in DB:
            if item["片名"] in KB_results:
                KB_pointer = item
        system_acts.append({
                            "diaact": "inform_one_match",
                            "片名": list(KB_results)
                            })
        system_acts.append({
                    "diaact": "reqmore"
                })
    elif len(KB_results) == 2:
        system_acts.append({
            "diaact": "inform_two_match",
            "片名": list(KB_results)
        })
    elif len(KB_results) == 3:
        system_acts.append({
            "diaact": "inform_three_match",
            "片名": list(KB_results)
        })
    elif len(KB_results) == 0:
        system_acts.append({
            "diaact": "inform_no_match"
        })

    if len(system_acts) == 0:
        # 处理 informable slots
        for slot in belief_states:
            if belief_states[slot][slot] != "NOT_MENTIONED" and slot !="片名":
                inquire_slots.remove(slot)
        if len(inquire_slots)> 2:
            system_acts.append({
                "diaact": "request",
                "informable_slots": [
                    inquire_slots[0]
                ]
            })
        else:
            system_acts.append({
                "diaact": "inform_many_match",
                "片名": list(KB_results)[:3]
            })

    # 处理 requestable slots
    if len(requested_slots) >0 and KB_pointer != None:
        if len(system_acts) >0:
            system_acts.clear()
        system_acts.append({"diaact": "inform",
                    "requestable_slots": list(requested_slots.keys())
                    })
    if len(system_acts) == 0:
        system_acts.append({"diaact": "repeat"})
    return system_acts,KB_pointer



if __name__ == '__main__':
    with open('F:\daiyp\Project_RLforDialogue\Experiment\myblog\data/Iqiyi_movie_DB.json', encoding='utf-8') as f:
        DB = json.load(f)
    belief_states = {
        "片名": {
            "片名": "NOT_MENTIONED",
        },
                    "导演": {
                        "导演": "NOT_MENTIONED"
                    },
                    "主演": {
                        "主演": "MENTIONED",
                        "成龙": "LIKE"
                    },
                    "类型": {
                        "类型": "NOT_MENTIONED"
                    },
                    "地区": {
                        "地区": "NOT_MENTIONED",
                    },
                    "年代": {
                        "年代": "MENTIONED",
                        "九十":"LIKE"
                    },
                    "资费": {
                        "资费": "NOT_MENTIONED"
                    }
                }
    print(len(search_DB(belief_states, DB)))
    # print(find_moive("类型", "喜剧", DB))
    # print(rule_method(belief_states,True, {}, None, DB))

