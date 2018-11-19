'''
This file is to provide batch training data.
We dont use it when we test our model.
'''
import json,copy
import random
import numpy as np
import jieba
jieba.load_userdict("data/New_words.txt")
embedding_dim = 25
np.random.seed(1234)

# embedding table
actor = [
            "克里斯汀韦格",
            "艾伦鲍斯汀",
            "杰克布莱克",
            "杰森斯坦森",
            "约翰考伯特",
            "桑迪牛顿",
            "杨秀措",
            "梁家辉",
            "马特达蒙",
            "元华",
            "泰伦斯霍华德",
            "罗伯特杜瓦尔",
            "李宗盛",
            "薇诺娜瑞德",
            "米歇尔莫娜汉",
            "西恩潘",
            "周冬雨",
            "比尔奈伊",
            "戎祥",
            "佟大为",
            "艾莉丝布拉加",
            "金巴",
            "林子聪",
            "徐子珊",
            "阿曼达普拉莫",
            "后藤久美子",
            "朱丽娅斯蒂尔斯",
            "詹妮弗康纳利",
            "袁泉",
            "尼基凯特",
            "帕特里克麦高汉",
            "梁旋",
            "王宝强",
            "曾江",
            "瑞切尔格里菲斯",
            "梅婷",
            "乌玛瑟曼",
            "张家辉",
            "张静初",
            "奥兰多布鲁姆",
            "周润发",
            "沙尔曼乔什",
            "吉恩哈克曼",
            "查理塔汉",
            "杰克吉伦哈尔",
            "邓超",
            "黄晓明",
            "邓家佳",
            "罗伯布朗",
            "楚原",
            "窦骁",
            "徐峥",
            "金妮弗古德温",
            "周星驰",
            "强沃特",
            "汤姆汉克斯",
            "史蒂夫卡瑞尔",
            "迈克尔伊雷",
            "卢克布雷西",
            "葛优",
            "阿尔帕西诺",
            "杰伊赫尔南德兹",
            "郭采洁",
            "李晨",
            "贾登史密斯",
            "陈慧琳",
            "约翰赖利",
            "肖央",
            "马思纯",
            "方力申",
            "约翰雷吉扎莫",
            "陈道明",
            "马克鲁弗洛",
            "关之琳",
            "黛安韦斯特",
            "沙鲁巴舒克拉",
            "余男",
            "董骠",
            "安妮海瑟薇",
            "王珞丹",
            "詹妮弗加纳",
            "陈坤",
            "凯文詹姆斯",
            "安努舒卡莎玛",
            "徐正曦",
            "甄子丹",
            "布莱恩豪威",
            "琳达费奥伦蒂诺",
            "杰夫盖尔品",
            "詹森艾萨克",
            "王庆祥",
            "王双宝",
            "谭咏麟",
            "林蛟",
            "罗伯特德尼罗",
            "徐静蕾",
            "孙红雷",
            "安德鲁加菲尔德",
            "任达华",
            "布拉德皮特",
            "郑伊健",
            "乔伊金",
            "杰西艾森伯格",
            "钟楚红",
            "吴孟达",
            "张国荣",
            "威尔史密斯",
            "陈冠希",
            "九孔",
            "佟丽娅",
            "陈家乐",
            "姜文",
            "桑杰达特",
            "马龙韦恩斯",
            "张雨绮",
            "欧阳娜娜",
            "郭富城",
            "查宁塔图姆",
            "程野",
            "伍迪哈里森",
            "莉莎博内特",
            "马龙白兰度",
            "胡枫",
            "黄觉",
            "海伦娜邦汉卡特",
            "元彪",
            "约翰特拉沃塔",
            "刘信义",
            "瑞克冈萨雷斯",
            "宋晓峰",
            "高捷",
            "邵美琪",
            "郑欣宜",
            "何浩文",
            "李小璐",
            "张学友",
            "刘若英",
            "米兰达卡斯格拉夫",
            "麦克怀特",
            "西德尼玛",
            "吴旭东",
            "苏菲玛索",
            "张曼玉",
            "朱莉理查德森",
            "威廉菲德内尔",
            "比尔普尔曼",
            "赛琳娜戈麦斯",
            "泰瑞斯吉布森",
            "徐帆",
            "梅艳芳",
            "沃维克戴维斯",
            "文森特普莱斯",
            "陈赫",
            "梅拉尼罗兰",
            "吴镇宇",
            "道恩强森",
            "邱淑贞",
            "克利斯丁格拉夫",
            "克里斯托弗瓦尔兹",
            "六小龄童",
            "琼艾伦",
            "林雪",
            "萨姆沃辛顿",
            "冯文娟",
            "杰弗里拉什",
            "古铭瀚",
            "埃里瓦拉赫",
            "艾伦图代克",
            "文斯沃恩",
            "黄渤",
            "迪伦贝克",
            "周秀娜",
            "卡琳娜卡普",
            "陆琦蔚",
            "本阿弗莱克",
            "吴京",
            "黄秋生",
            "卡罗琳古道",
            "姜春琦",
            "帕迪康斯戴恩",
            "艾拉菲舍尔",
            "林保怡",
            "万梓良",
            "凯瑟琳麦克马克",
            "塞缪尔杰克逊",
            "李程彬",
            "凯拉奈特莉",
            "本金斯利",
            "洪金宝",
            "拉尔夫费因斯",
            "丁嘉丽",
            "卢惠光",
            "詹姆斯凯恩",
            "曹达华",
            "埃德加拉米雷兹",
            "伊娃门德斯",
            "尼尔帕特里克哈里斯",
            "王馥荔",
            "曾志伟",
            "刘昊然",
            "秦沛",
            "张子枫",
            "小沈阳",
            "刘嘉玲",
            "弗兰克约翰休斯",
            "苏小桐",
            "文森特诺费奥",
            "马德哈万",
            "约翰尼德普",
            "艾伯丝戴维兹",
            "吴彦祖",
            "狄龙",
            "佩内洛普安米勒",
            "艾伦阿金",
            "斯科特麦克纳里",
            "赵立新",
            "刘承俊",
            "雷普汤恩",
            "陈宝国",
            "维拉法梅加",
            "罗莎里奥道森",
            "刘青云",
            "保罗沃克",
            "叶德娴",
            "安贝瓦莱塔",
            "瞿颖",
            "单立文",
            "普路特泰勒文斯",
            "马克斯冯叙多",
            "余文乐",
            "李馨巧",
            "塔莉娅夏尔",
            "连姆尼森",
            "亚香缇",
            "李冰冰",
            "古天乐",
            "本斯蒂勒",
            "李小龙",
            "帕特里克波查",
            "路易斯古兹曼",
            "彭于晏",
            "吉田洁",
            "陈奕迅",
            "杜鹃",
            "波曼伊拉妮",
            "巴里佩珀",
            "梁天",
            "王祖贤",
            "伊莱罗斯",
            "黛安基顿",
            "尼古拉斯凯奇",
            "刘德华",
            "玛丽麦克唐纳",
            "梅尔吉布森",
            "詹姆斯布洛林",
            "布莱恩考克斯",
            "露丝威尔森",
            "林青霞",
            "波伊德霍布鲁克",
            "米歇尔威廉姆斯",
            "柴浩伟",
            "雨果维文",
            "梁朝伟",
            "姚星彤",
            "阿米尔汗",
            "丹尼斯奎德",
            "迈克尔马德森",
            "布丽姬穆娜",
            "希斯莱杰",
            "乔丹娜布鲁斯特",
            "迈克尔阿登",
            "郑业成",
            "汤米李琼斯",
            "杰森李",
            "布鲁斯威利斯",
            "郭笑",
            "成奎安",
            "娜奥米哈里斯",
            "郭涛",
            "成龙",
            "罗莎曼德派克",
            "莱昂纳多迪卡普里奥",
            "范迪塞尔",
            "萨拉丝沃曼",
            "安迪加西亚",
            "扎西",
            "莎莉理查德森",
            "张涵予",
            "理查德希夫",
            "周韵",
            "廖启智",
            "迈克尔法斯宾德",
            "李保田",
            "林正英",
            "马苏",
            "李连杰",
            "卢海鹏",
            "周迅",
            "詹姆斯贝吉戴尔",
            "杰弗里怀特",
            "约翰库萨克"
        ]
actor.sort(key=lambda x: len(x), reverse=True)
director = [
    "林超贤",
    "加布里尔穆奇诺",
    "姜文",
    "梅尔吉布森",
    "拉吉库马尔希拉尼",
    "成龙",
    "冯小刚",
    "戈尔维宾斯基",
    "刘伟强",
    "布莱恩德帕尔玛",
    "安迪坦纳特",
    "弗朗西斯劳伦斯",
    "罗兰艾默里奇",
    "托尼斯科特",
    "张艺谋",
    "大卫芬奇",
    "史蒂文斯皮尔伯格",
    "昆汀塔伦蒂诺",
    "斯蒂文斯皮尔伯格",
    "马丁斯科塞斯",
    "邓肯琼斯",
    "约翰李汉考克",
    "曹保平",
    "陈可辛",
    "王晶",
    "洪金宝",
    "唐季礼",
    "徐峥",
    "叶伟民",
    "宁浩",
    "理查德林克莱特",
    "蒂姆波顿",
    "陈思诚",
    "曾国祥",
    "万玛才旦",
    "周德元",
    "王家卫",
    "阚家伟",
    "阮世生",
    "弗朗西斯福特科波拉",
    "迈克内威尔",
    "叶伟信",
    "王一淳",
    "非行",
    "游乃海",
    "吴宇森",
    "爱德华兹威克",
    "达伦阿罗诺夫斯基",
    "巴里索南菲尔德",
    "泰勒海克福德",
    "伊丽莎白艾伦",
    "梁旋",
    "黄健中",
    "徐静蕾",
    "陈木胜",
    "许鞍华",
    "林诣彬",
    "托马斯卡特",
    "保罗格林格拉斯",
    "卡洛斯沙尔丹哈",
    "路易斯莱特里尔",
    "哈罗德雷米斯",
    "罗杰米歇尔",
    "奥利弗斯通",
    "安德鲁尼科尔",
    "盖里奇",
    "大卫阿耶",
    "莱塞霍尔斯道姆",
    "詹姆斯曼高德",
    "西蒙韦斯特",
    "周星驰",
    "徐克",
    "罗维",
    "罗伯特高洛斯",
    "蔺水净",
    "埃里克达尼尔",
    "本斯蒂勒",
    "皮埃尔科芬",
    "巴兹鲁赫曼",
    "罗伯明可夫",
    "李仁港",
    "管虎",
    "陈正道",
    "拜恩霍华德",
    "罗伯特泽米吉斯",
    "瑞奇摩尔",
    "林伟伦",
    "李力持",
    "比尔帕克斯顿",
    "诺拉艾芙隆",
    "霍华德齐耶夫",
    "韩三平",
    "佩波丹科瓦特",
    "黄建新",
    "麦兆辉",
    "彼得杰克逊",
    "乔赖特",
    "奥利维耶霍莱",
    "雷德利斯科特",
    "李杨",
    "陈嘉上",
    "关喆",
    "崔俊杰",
    "郑保瑞",
    "李惠民"
]
director.sort(key=lambda x: len(x), reverse=True)
title = [
    "湄公河行动",
    "当幸福来敲门",
    "让子弹飞",
    "血战钢锯岭",
    "三傻大闹宝莱坞",
    "十二生肖",
    "唐山大地震",
    "加勒比海盗二",
    "无间道一",
    "情枭的黎明",
    "破风",
    "激战",
    "七磅",
    "全民情敌",
    "我是传奇",
    "独立日",
    "全民公敌",
    "有话好好说",
    "勇敢的心",
    "社交网络",
    "辛德勒的名单",
    "无耻混蛋",
    "猫鼠游戏",
    "禁闭岛",
    "源代码",
    "心灵投手",
    "李米的猜想",
    "中国合伙人",
    "龙兄虎弟",
    "城市猎人",
    "福星高照",
    "红番区",
    "人再囧途之泰囧",
    "人在囧途",
    "心花路放",
    "疯狂的赛车",
    "我的个神啊",
    "摇滚校园",
    "加勒比海盗一",
    "加勒比海盗三",
    "独行侠",
    "剪刀手爱德华",
    "唐人街探案",
    "天下无贼",
    "无间道二",
    "七月与安生",
    "塔洛",
    "王牌御史之猎妖教室",
    "阿飞正传",
    "我的极品女神",
    "巴黎假期",
    "旺角卡门",
    "教父三",
    "教父二",
    "忠奸人",
    "教父",
    "杀破狼",
    "黑处有什么",
    "金钱帝国",
    "全民目击",
    "跟踪",
    "纵横四海",
    "英雄本色一",
    "血钻",
    "梦之安魂曲",
    "黑衣人一",
    "灵魂歌王",
    "蕾蒙娜和姐姐",
    "大鱼海棠",
    "过年",
    "一个陌生女人的来信",
    "扫毒",
    "桃姐",
    "速度与激情五",
    "低俗小说",
    "卡特教练",
    "谍影重重三",
    "爱国者",
    "消失的爱人",
    "里约大冒险",
    "惊天魔盗团",
    "偷天情缘",
    "诺丁山",
    "华尔街",
    "圆梦巨人",
    "千钧一发",
    "偷拐抢骗",
    "狂怒",
    "不一样的天空",
    "致命ID",
    "空中监狱",
    "魔兽",
    "月球",
    "弱点",
    "亲爱的",
    "财神到",
    "功夫",
    "青蛇",
    "警察故事三",
    "精武门",
    "落水狗",
    "死亡游戏",
    "警察故事四",
    "警察故事",
    "飞鹰计划",
    "警察故事二",
    "斗战胜佛",
    "马达加斯加三",
    "白日梦想家",
    "神偷奶爸",
    "神偷奶爸二",
    "了不起的盖茨比",
    "捕蝇纸",
    "两杆大烟枪",
    "黑侠",
    "无人区",
    "斗牛",
    "一零一次求婚",
    "疯狂动物城",
    "阿甘正传",
    "无敌破坏王",
    "小飞侠",
    "行运一条龙",
    "天若有情",
    "那些最伟大的比赛",
    "西雅图夜未眠",
    "小丫头",
    "建国大业",
    "流浪的尤莱克",
    "背靠背，脸对脸",
    "窃听风云一",
    "指环王一",
    "傲慢与偏见",
    "爱丽丝梦游奇境",
    "浓情巧克力",
    "变脸",
    "男孩与鹈鹕",
    "大鱼",
    "天国王朝",
    "盲井",
    "逃学威龙一",
    "神笔马娘",
    "山炮进城",
    "山炮进城二",
    "狗咬狗",
    "新龙门客栈",
    "重庆森林"
]
title.sort(key=lambda x: len(x), reverse=True)
filter_slots = [actor, director, title]
with open('data/Iqiyi_ONTO.json', 'r', encoding='utf-8') as f:
    OTGY = json.load(f)
informable_slots = ["片名","导演","主演","类型", "地区","年代","资费"]
requestable_slots = OTGY["requestable"]
value_label = ["LIKE","DISLIKE","NOT_MENTIONED"]
slot_label = ["DONT_CARE","MENTIONED","NOT_MENTIONED"]
request_label = ["MENTIONED","NOT_MENTIONED"]

# build semantic dictionary
def build_semantic_dict(OTGY):
    semantic_dict = {"requestable":{}, "informable":{}}
    for slot in OTGY["requestable"]:
        semantic_dict["requestable"][slot] = [slot]
    for slot,value in OTGY["informable"].items():
        values = [[v] for v in value]
        semantic_dict["informable"][slot] = dict(zip(value, values))
    with open('data/semantic_dict_simple.json', 'w',encoding='utf-8') as f:
        json.dump(semantic_dict,f,indent=4,ensure_ascii=False)


class DataSet(object):
    """next_batch_informable[slot][slot/value_specific]: (a, u, value/slot name, label)"""
    def __init__(self, raw_data_set):
        self._OTGY = OTGY
        self._raw_data_set = raw_data_set
        self._value_specific_data, self._slot_specific_data, self._requestable_data = self.build_dataset_for_training()

        # Shuffle the data
        for slot in self._value_specific_data:
            for label in self._value_specific_data[slot]:
                random.shuffle(self._value_specific_data[slot][label])
                # print(slot, label, len(self._value_specific_data[slot][label]))
        for slot in self._slot_specific_data:
            for label in self._slot_specific_data[slot]:
                random.shuffle(self._slot_specific_data[slot][label])
                # print(slot, label, len(self._slot_specific_data[slot][label]))
        for slot in self._requestable_data:
            for label in self._requestable_data[slot]:
                random.shuffle(self._requestable_data[slot][label])
                # print(slot, label, len(self._requestable_data[slot][label]))
        # index for minibatch next batch
        self._value_specific_informable_index = {}
        for slot in informable_slots:
            self._value_specific_informable_index[slot] = {}
            for label in value_label:
                self._value_specific_informable_index[slot][label] = 0
        self._slot_specific_informable_index = {}
        for slot in informable_slots:
            self._slot_specific_informable_index[slot] = {}
            for label in slot_label:
                self._slot_specific_informable_index[slot][label] = 0
        self._requestable_index = {}
        for slot in requestable_slots:
            self._requestable_index[slot] = {}
            for label in request_label:
                self._requestable_index[slot][label] = 0


    def next_batch_informable(self):
        value_specific_data = {}
        for slot in informable_slots:
            value_specific_data[slot] = []

        slot_specific_data = {}
        for slot in informable_slots:
            slot_specific_data[slot] = []

        # for value-specific tracker, minibatch 256(22/10/224) LIKE/DISLIKE/NOT_MENTIONED
        value_train_num = ("LIKE",22),("DISLIKE",10),("NOT_MENTIONED",224)
        start_value_specific = copy.deepcopy(self._value_specific_informable_index)
        end_value_specific = copy.deepcopy(self._value_specific_informable_index)
        for slot in self._value_specific_informable_index:
            for label,num in value_train_num:
                self._value_specific_informable_index[slot][label] += num
                if self._value_specific_informable_index[slot][label] > len(
                        self._value_specific_data[slot][label]):
                    # Shuffle the data
                    random.shuffle(self._value_specific_data[slot][label])
                    # Start next epoch
                    start_value_specific[slot][label] = 0
                    self._value_specific_informable_index[slot][label] = num
                end_value_specific[slot][label] = self._value_specific_informable_index[slot][label]

                value_specific_data[slot].extend(copy.deepcopy(self._value_specific_data[slot][label][
                                                         start_value_specific[slot][label]:
                                                         end_value_specific[slot][label]]))
        # for slot-specific tracker, minibatch
        slot_train_num = ("DONT_CARE",8), ("MENTIONED",16), ("NOT_MENTIONED",40)
        start_slot_specific = copy.deepcopy(self._slot_specific_informable_index)
        end_slot_specific = copy.deepcopy(self._slot_specific_informable_index)
        for slot in self._slot_specific_informable_index:
            for label, num in slot_train_num:
                self._slot_specific_informable_index[slot][label] += num
                if self._slot_specific_informable_index[slot][label] > len(
                        self._slot_specific_data[slot][label]):
                    # Shuffle the data
                    random.shuffle(self._slot_specific_data[slot][label])
                    # Start next epoch
                    start_slot_specific[slot][label] = 0
                    self._slot_specific_informable_index[slot][label] = num
                end_slot_specific[slot][label] = self._slot_specific_informable_index[slot][label]

                slot_specific_data[slot].extend(copy.deepcopy(self._slot_specific_data[slot][label][
                                                               start_slot_specific[slot][label]:
                                                               end_slot_specific[slot][label]]))

        return {"slot-specific":slot_specific_data,"value-specific":value_specific_data}

    def next_batch_requestable(self):
        request_specific_data = {}
        for slot in requestable_slots:
            request_specific_data[slot] = []

        # minibatch 64(8/ 56)
        start_request = copy.deepcopy(self._requestable_index)
        end_request = copy.deepcopy(self._requestable_index)

        for slot in request_specific_data:
            self._requestable_index[slot]["MENTIONED"] += 8
            if self._requestable_index[slot]["MENTIONED"] > len(
                    self._requestable_data[slot]["MENTIONED"]):
                # Shuffle the data
                random.shuffle(self._requestable_data[slot]["MENTIONED"])
                # Start next epoch
                start_request[slot]["MENTIONED"] = 0
                self._requestable_index[slot]["MENTIONED"] = 8
            end_request[slot]["MENTIONED"] = self._requestable_index[slot]["MENTIONED"]

            request_specific_data[slot].extend(copy.deepcopy(self._requestable_data[slot]["MENTIONED"][
                                            start_request[slot]["MENTIONED"]:
                                            end_request[slot]["MENTIONED"]]))

            self._requestable_index[slot]["NOT_MENTIONED"] += 56
            if self._requestable_index[slot]["NOT_MENTIONED"] > len(
                    self._requestable_data[slot]["NOT_MENTIONED"]):
                # Shuffle the data
                random.shuffle(self._requestable_data[slot]["NOT_MENTIONED"])
                # Start next epoch
                start_request[slot]["NOT_MENTIONED"] = 0
                self._requestable_index[slot]["NOT_MENTIONED"] = 56
            end_request[slot]["NOT_MENTIONED"] = self._requestable_index[slot]["NOT_MENTIONED"]

            request_specific_data[slot].extend(copy.deepcopy(self._requestable_data[slot]["NOT_MENTIONED"][
                                                start_request[slot]["NOT_MENTIONED"]:
                                                end_request[slot]["NOT_MENTIONED"]]))


        return request_specific_data

    def extract_transcripts(self, line):
        for slot in filter_slots:
            for value in slot:
                if value in line:
                    line = line.replace(value, " " + value + " ")
                    break
        return ' '.join(jieba.cut(line + '。')).split()

    def extract_sys_act(self, acts):
        sys_act = {"request": [], "confirm_positive": [],"confirm_negative":[], "inform_one_match":None}
        for act in acts:
            if act["diaact"] == "expl_conf":
                for slot in act["positive_slots"]:
                    for value in act["positive_slots"][slot]:
                        sys_act["confirm_positive"].append((slot,value))
                for slot in act["negative_slots"]:
                    for value in act["negative_slots"][slot]:
                        sys_act["confirm_negative"].append((slot,value))
            elif act["diaact"] == "request":
                sys_act["request"] = copy.deepcopy(act["informable_slots"])
            elif act["diaact"] == "inform_one_match":
                sys_act["inform_one_match"] = act["片名"][0]
        return sys_act

    def extract_one_turn(self, one_turn, last_belief_states):
        """one turn for all the informable slots training data"""

        usr = self.extract_transcripts(one_turn["user_transcript"])
        sys_act = self.extract_sys_act(one_turn["system_acts"])
        value_specific_data = {}
        for slot in informable_slots:
            value_specific_data[slot] = {}
            for label in value_label:
                value_specific_data[slot][label] = []
        slot_specific_data ={}
        for slot in informable_slots:
            slot_specific_data[slot] = {}
            for label in slot_label:
                slot_specific_data[slot][label] = []
        request_specific_data = {}
        for slot in requestable_slots:
            request_specific_data[slot] = {}
            for label in request_label:
                request_specific_data[slot][label] = []

        # informable
        for belief_slot,belief_tracking in one_turn["belief_states"].items():
            if belief_tracking[belief_slot] == "NOT_MENTIONED":
                not_mentioned_data = [(sys_act, usr, belief_slot, v,
                                       last_belief_states[belief_slot][belief_slot],
                                       last_belief_states[belief_slot][v])
                                      for v in self._OTGY["informable"][belief_slot] if v in last_belief_states[belief_slot]] +\
                                     [(sys_act, usr, belief_slot, v,
                                       last_belief_states[belief_slot][belief_slot],
                                       "NOT_MENTIONED")
                                      for v in self._OTGY["informable"][belief_slot] if v not in last_belief_states[belief_slot]]
                value_specific_data[belief_slot]["NOT_MENTIONED"].extend(
                    copy.deepcopy(not_mentioned_data))
                slot_specific_data[belief_slot]["NOT_MENTIONED"].append(
                    (sys_act, usr, belief_slot, last_belief_states[belief_slot][belief_slot]))
            elif belief_tracking[belief_slot] == "DONT_CARE":
                slot_specific_data[belief_slot]["DONT_CARE"].append(
                    (sys_act, usr, belief_slot, last_belief_states[belief_slot][belief_slot]))
                not_mentioned_data = [(sys_act, usr, belief_slot, v,
                                       last_belief_states[belief_slot][belief_slot],
                                       last_belief_states[belief_slot][v])
                                      for v in self._OTGY["informable"][belief_slot] if v in last_belief_states[belief_slot]] +\
                                     [(sys_act, usr, belief_slot, v,
                                       last_belief_states[belief_slot][belief_slot],
                                       "NOT_MENTIONED")
                                      for v in self._OTGY["informable"][belief_slot] if v not in last_belief_states[belief_slot]]

                value_specific_data[belief_slot]["NOT_MENTIONED"].extend(
                    copy.deepcopy(not_mentioned_data))
            else:
                for value in belief_tracking:
                    if value != belief_slot and value in last_belief_states[belief_slot]:
                        value_specific_data[belief_slot][belief_tracking[value]].append(
                            (sys_act, usr, belief_slot, value,
                            last_belief_states[belief_slot][belief_slot],
                            last_belief_states[belief_slot][value])
                        )
                    elif value != belief_slot and value not in last_belief_states[belief_slot]:
                        value_specific_data[belief_slot][belief_tracking[value]].append(
                            (sys_act, usr, belief_slot, value,
                             last_belief_states[belief_slot][belief_slot],
                             "NOT_MENTIONED")
                        )
                slot_specific_data[belief_slot]["MENTIONED"].append(
                    (sys_act, usr, belief_slot, last_belief_states[belief_slot][belief_slot]))

        # requestable
        for slot in requestable_slots:
            request_specific_data[slot][one_turn["requested_slots"][slot]].append((sys_act, usr, slot))

        return slot_specific_data, value_specific_data, request_specific_data

    def build_dataset_for_training(self):
        """transform all the train set into training data"""
        value_specific_data = {}
        for slot in informable_slots:
            value_specific_data[slot] = {}
            for label in value_label:
                value_specific_data[slot][label] = []
        slot_specific_data = {}
        for slot in informable_slots:
            slot_specific_data[slot] = {}
            for label in slot_label:
                slot_specific_data[slot][label] = []
        request_specific_data = {}
        for slot in requestable_slots:
            request_specific_data[slot] = {}
            for label in request_label:
                request_specific_data[slot][label] = []

        for dial in self._raw_data_set:
            last_belief_states = {
                    "片名": {
                        "片名": "NOT_MENTIONED"
                    },
                    "导演": {
                        "导演": "NOT_MENTIONED"
                    },
                    "主演": {
                        "主演": "NOT_MENTIONED"
                    },
                    "类型": {
                        "类型": "NOT_MENTIONED"
                    },
                    "地区": {
                        "地区": "NOT_MENTIONED"
                    },
                    "年代": {
                        "年代": "NOT_MENTIONED"
                    },
                    "资费": {
                        "资费": "NOT_MENTIONED"
                    }
                }
            for turn in dial["dialog"]:
                oneturn_slot_specific_data, oneturn_value_specific_data, oneturn_requestable_data = \
                    self.extract_one_turn(turn, last_belief_states)
                for slot in value_specific_data:
                    for label in value_specific_data[slot]:
                        value_specific_data[slot][label].extend(oneturn_value_specific_data[slot][label])
                for slot in slot_specific_data:
                    for label in slot_specific_data[slot]:
                        slot_specific_data[slot][label].extend(oneturn_slot_specific_data[slot][label])
                for slot in request_specific_data:
                    for label in request_specific_data[slot]:
                        request_specific_data[slot][label].extend(oneturn_requestable_data[slot][label])
                last_belief_states = turn["belief_states"]

        return value_specific_data, slot_specific_data, request_specific_data

    def build_dataset_for_test(self):
        """ turn level"""
        dataset_for_test = []
        for dial in self._raw_data_set:
            last_belief_states = {
                "片名": {
                    "片名": "NOT_MENTIONED"
                },
                "导演": {
                    "导演": "NOT_MENTIONED"
                },
                "主演": {
                    "主演": "NOT_MENTIONED"
                },
                "类型": {
                    "类型": "NOT_MENTIONED"
                },
                "地区": {
                    "地区": "NOT_MENTIONED"
                },
                "年代": {
                    "年代": "NOT_MENTIONED"
                },
                "资费": {
                    "资费": "NOT_MENTIONED"
                }
            }
            for turn in dial["dialog"]:
                positive_onedata = {"system_act": self.extract_sys_act(turn["system_acts"]),
                                    "usr_transcript": self.extract_transcripts(turn["user_transcript"]),
                                    "last_belief_states": last_belief_states,
                                    "belief_states": turn["belief_states"],
                                    "requested_slots": turn["requested_slots"]}
                dataset_for_test.append(positive_onedata)
                last_belief_states = turn["belief_states"]
        return dataset_for_test

    def build_dataset_for_belief_states(self):
        dialog_data = []
        for dial in self._raw_data_set:
            one_dialog = []
            for turn in dial["dialog"]:
                one_turn = {"system_act": self.extract_sys_act(turn["system_acts"]),
                            "usr_transcript": self.extract_transcripts(turn["user_transcript"]),
                            "belief_states": turn["belief_states"],
                            "requested_slots": turn["requested_slots"]
                            }
                one_dialog.append(copy.deepcopy(one_turn))
            dialog_data.append(copy.deepcopy(one_dialog))
        return dialog_data


def read_data_sets():
    class DataSets(object):
        pass
    data_sets = DataSets()
    with open('data/Iqiyi_800.json','r',encoding='utf-8') as f:
        rawdata = json.load(f)
    random.shuffle(rawdata)
    data_sets.train = DataSet(rawdata[300:])
    # with open('data/train_valid_set.json','w',encoding='utf-8') as f:
    #     json.dump(rawdata[300:],f,ensure_ascii=False,indent=4)
    # data_sets.test = DataSet(rawdata[:300])
    # with open('data/test_set.json','w',encoding='utf-8') as f:
    #     json.dump(rawdata[:300],f,ensure_ascii=False,indent=4)

    with open('data/train_valid_set.json','r',encoding='utf-8') as f:
        train_valid_set = json.load(f)
    data_sets.train = DataSet(train_valid_set)
    with open('data/test_set.json','r',encoding='utf-8') as f:
        test_set = json.load(f)
    data_sets.test = DataSet(test_set)
    return data_sets

if __name__ == '__main__':
    with open('data/Iqiyi_800.json','r',encoding='utf-8') as f:
        rawdata = json.load(f)
    dataset = DataSet(rawdata)

    value_specific_data_valid, slot_specific_data_valid, requestable_data_valid = \
        dataset.build_dataset_for_training()
    for ii in informable_slots:
        for dd in slot_specific_data_valid[ii]["MENTIONED"]:
            if dd[3] == "DONT_CARE":
                print(dd)