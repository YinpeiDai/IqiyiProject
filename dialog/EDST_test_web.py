'''
This is the core file for testing EDST dialog system.
You can change this file slightly to train EDST dialog system.
'''
import tensorflow as tf
import numpy as np
import pickle, json,copy
import jieba
from dialog.policy import rule_method
from dialog.NLG import rule_NLG
jieba.load_userdict("data/New_words.txt")
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
max_length = 32
hidden_size = 100
learning_rate = 0.0003

# input file
# print('load the model ...')
embedding_dim = 25
with open('data/vocab_dict.json', 'r', encoding='utf-8') as f:
    vocab_dict = json.load(f)
with open('data/vocab_norm.25d.pkl', 'rb') as f:
    embedding_table = pickle.load(f)
with open('data/Iqiyi_ONTO.json', 'r',encoding='utf-8') as f:
    OTGY = json.load(f)
with open('data/semantic_dict.json', 'r',encoding='utf-8') as f:
    semantic_dict = json.load(f)
with open('data/Iqiyi_movie_DB.json', encoding='utf-8') as f:
    DB = json.load(f)
informable_slots = ["类型", "地区","资费","片名","导演","主演","年代"]
requestable_slots = OTGY["requestable"]


# value-specific tracker
class Value_Specific_tracker(object):
    def __init__(self, slot_name):
        """
        对每个 informable slot 都维护一个 value-specific tracker
        :param slot_name: 名字
        """
        self._slot_name = slot_name
        # tf Graph input
        self._x_last = tf.placeholder(tf.float32, [None, 3]) # LIKE/DISLIKE/NOT_MENTIONED
        self._x_usr = tf.placeholder(tf.float32, [None, max_length, embedding_dim])
        self._x_usr_len = tf.placeholder(tf.int32, [None])
        self._x_sys_acts = tf.placeholder(tf.float32, [None, 5])
        # request/confirm_slot/confirm_value+/confirm_value-/inform_one_match
        self._x_stringmatch = tf.placeholder(tf.float32, [None,max_length,1]) # whether MENTIONED
        self._x_value = tf.placeholder(tf.float32, [None, embedding_dim])
        self._is_training = tf.placeholder(tf.bool)
        self._y = tf.placeholder(tf.float32, [None, 3])
        self._lr = tf.Variable(0.001, trainable=False)
        self._batchsize_value = tf.shape(self._x_value)[0]

        # self._W1 = tf.Variable(10.0, dtype=tf.float32)
        # self._b1 = tf.Variable(-0.8, dtype=tf.float32)
        # self._b2 = tf.Variable(0.0, dtype=tf.float32)
        # self._x_value_m = tf.expand_dims(self._x_value, 2)
        # self._value_sequence  = tf.nn.sigmoid(tf.multiply(tf.matmul(self._x_usr, self._x_value_m)+self._b1, self._W1) + self._b2)
        # value_specific usr sentence
        self._value_usr = tf.concat([self._x_usr, self._x_stringmatch],2)  # -> (?, max_length, embedding_size+1)


        with tf.variable_scope("last_state"):
            # last belief state
            self._conv1 = tf.layers.conv1d(
                inputs=self._value_usr,
                filters=20,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool1 = tf.layers.max_pooling1d(
                self._conv1,
                pool_size=max_length,
                strides=1,
            )
            self._pool1 = tf.squeeze(self._pool1, [1])
            # bigram filters
            self._conv2 = tf.layers.conv1d(
                inputs=self._value_usr,
                filters=20,
                kernel_size=2,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool2 = tf.layers.max_pooling1d(
                self._conv2,
                pool_size=max_length,
                strides=1,
            )
            self._pool2 = tf.squeeze(self._pool2, [1])

            # trigram
            self._conv3 = tf.layers.conv1d(
                inputs=self._value_usr,
                filters=20,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool3 = tf.layers.max_pooling1d(
                self._conv3,
                pool_size=max_length,
                strides=1,
            )



            self._pool3 = tf.squeeze(self._pool3, [1])
            self._output_pool = tf.concat([self._pool1, self._pool2, self._pool3], 1)

            self._is_match = tf.squeeze(tf.layers.max_pooling1d(
                self._x_stringmatch,
                pool_size=max_length,
                strides=1,
            ), [1])

            self._input = tf.concat([self._output_pool,
                                     self._is_match],
                                    1)

            self._dnn_hiddenlayer = tf.layers.dense(self._input,
                                                    self._input.shape[1]+self._input.shape[1],
                                                    activation=tf.nn.sigmoid,
                                                    kernel_initializer=tf.random_normal_initializer(0, 0.5))

            self._last_state_W = tf.layers.dense(self._dnn_hiddenlayer, 3,
                                                   activation=tf.nn.sigmoid,
                                                   kernel_initializer=tf.random_normal_initializer(0, 0.5), )

            self._last_state = tf.multiply(self._last_state_W, self._x_last)

        # feed value-specific usr sentence to LSTM
        with tf.variable_scope("x_usr"):
            self._conv1 = tf.layers.conv1d(
                inputs=self._value_usr,
                filters=100,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool1 = tf.layers.max_pooling1d(
                self._conv1,
                pool_size=max_length,
                strides=1,
            )
            self._pool1 = tf.squeeze(self._pool1, [1])
            # bigram filters
            self._conv2 = tf.layers.conv1d(
                inputs=self._value_usr,
                filters=100,
                kernel_size=2,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool2 = tf.layers.max_pooling1d(
                self._conv2,
                pool_size=max_length,
                strides=1,
            )
            self._pool2 = tf.squeeze(self._pool2, [1])

            # trigram
            self._conv3 = tf.layers.conv1d(
                inputs=self._value_usr,
                filters=100,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool3 = tf.layers.max_pooling1d(
                self._conv3,
                pool_size=max_length,
                strides=1,
            )

            self._pool3 = tf.squeeze(self._pool3, [1])
            self._output_pool = tf.concat([self._pool1, self._pool2, self._pool3], 1)
            self._input = tf.concat([self._x_sys_acts],
                                     1)

            self._dnn_hiddenlayer = tf.layers.dense(self._input,
                                                    self._input.shape[1]+self._input.shape[1],
                                                    activation=tf.nn.sigmoid,
                                                    kernel_initializer=tf.random_normal_initializer(0, 0.5))

            self._output_final_W = tf.layers.dense(self._dnn_hiddenlayer, 1,
                                                 activation=tf.nn.sigmoid,
                                                 kernel_initializer=tf.random_normal_initializer(0, 0.5), )
            self._output_finals = tf.multiply(self._output_final_W, self._output_pool)
        with tf.variable_scope("x_usr_request"):
            # feed value-specific usr sentence to CNN
            # unigram filters
            self._conv1 = tf.layers.conv1d(
                inputs=self._value_usr,
                filters=100,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool1 = tf.layers.max_pooling1d(
                self._conv1,
                pool_size=max_length,
                strides=1,
            )
            self._pool1 = tf.squeeze(self._pool1, [1])
            # bigram filters
            self._conv2 = tf.layers.conv1d(
                inputs=self._value_usr,
                filters=100,
                kernel_size=2,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool2 = tf.layers.max_pooling1d(
                self._conv2,
                pool_size=max_length,
                strides=1,
            )
            self._pool2 = tf.squeeze(self._pool2, [1])

            # trigram
            self._conv3 = tf.layers.conv1d(
                inputs=self._value_usr,
                filters=100,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool3 = tf.layers.max_pooling1d(
                self._conv3,
                pool_size=max_length,
                strides=1,
            )

            self._pool3 = tf.squeeze(self._pool3, [1])
            self._output_pool = tf.concat([self._pool1, self._pool2, self._pool3], 1)
            # context modelling
            self._context_request = tf.multiply(self._output_pool,tf.expand_dims(self._x_sys_acts[:,0],1))
        with tf.variable_scope("x_usr_confirm_slot"):
            # unigram filters
            self._conv1 = tf.layers.conv1d(
                inputs=self._value_usr,
                filters=20,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool1 = tf.layers.max_pooling1d(
                self._conv1,
                pool_size=max_length,
                strides=1,
            )
            self._pool1 = tf.squeeze(self._pool1, [1])
            # bigram filters
            self._conv2 = tf.layers.conv1d(
                inputs=self._value_usr,
                filters=20,
                kernel_size=2,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool2 = tf.layers.max_pooling1d(
                self._conv2,
                pool_size=max_length,
                strides=1,
            )
            self._pool2 = tf.squeeze(self._pool2, [1])

            # trigram
            self._conv3 = tf.layers.conv1d(
                inputs=self._value_usr,
                filters=20,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool3 = tf.layers.max_pooling1d(
                self._conv3,
                pool_size=max_length,
                strides=1,
            )

            self._pool3 = tf.squeeze(self._pool3, [1])
            self._output_pool = tf.concat([self._pool1, self._pool2, self._pool3], 1)
            # context modelling
            self._context_confirm_slot = tf.multiply(self._output_pool, tf.expand_dims(self._x_sys_acts[:,1],1))
        with tf.variable_scope("x_usr_confirm_value_positive"):
            # unigram filters
            self._conv1 = tf.layers.conv1d(
                inputs=self._value_usr,
                filters=20,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool1 = tf.layers.max_pooling1d(
                self._conv1,
                pool_size=max_length,
                strides=1,
            )
            self._pool1 = tf.squeeze(self._pool1, [1])
            # bigram filters
            self._conv2 = tf.layers.conv1d(
                inputs=self._value_usr,
                filters=20,
                kernel_size=2,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool2 = tf.layers.max_pooling1d(
                self._conv2,
                pool_size=max_length,
                strides=1,
            )
            self._pool2 = tf.squeeze(self._pool2, [1])

            # trigram
            self._conv3 = tf.layers.conv1d(
                inputs=self._value_usr,
                filters=20,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool3 = tf.layers.max_pooling1d(
                self._conv3,
                pool_size=max_length,
                strides=1,
            )

            self._pool3 = tf.squeeze(self._pool3, [1])
            self._output_pool = tf.concat([self._pool1, self._pool2, self._pool3], 1)
            self._context_confirm_value_positive = tf.multiply(self._output_pool, tf.expand_dims(self._x_sys_acts[:,2],1))
        with tf.variable_scope("x_usr_confirm_value_negative"):
            # unigram filters
            self._conv1 = tf.layers.conv1d(
                inputs=self._value_usr,
                filters=20,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool1 = tf.layers.max_pooling1d(
                self._conv1,
                pool_size=max_length,
                strides=1,
            )
            self._pool1 = tf.squeeze(self._pool1, [1])
            # bigram filters
            self._conv2 = tf.layers.conv1d(
                inputs=self._value_usr,
                filters=20,
                kernel_size=2,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool2 = tf.layers.max_pooling1d(
                self._conv2,
                pool_size=max_length,
                strides=1,
            )
            self._pool2 = tf.squeeze(self._pool2, [1])

            # trigram
            self._conv3 = tf.layers.conv1d(
                inputs=self._value_usr,
                filters=20,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool3 = tf.layers.max_pooling1d(
                self._conv3,
                pool_size=max_length,
                strides=1,
            )

            self._pool3 = tf.squeeze(self._pool3, [1])
            self._output_pool = tf.concat([self._pool1, self._pool2, self._pool3], 1)
            self._context_confirm_value_negative = tf.multiply(self._output_pool, tf.expand_dims(self._x_sys_acts[:,3],1))

        # assemble
        self._inputs = tf.concat([self._output_finals,
                                  self._context_request,
                                  self._context_confirm_slot,
                                  self._context_confirm_value_positive,
                                  self._context_confirm_value_negative,
                                  tf.expand_dims(self._x_sys_acts[:,4],1),
                                  self._last_state],
                                 1)


        self._dnn_hiddenlayer = tf.layers.dense(self._inputs,
                                                500,
                                                activation=tf.nn.relu,
                                                kernel_initializer=tf.random_normal_initializer(0, 0.1))
        self._dnn_hiddenlayer = tf.layers.dropout(self._dnn_hiddenlayer, rate=0.5, training=self._is_training)

        self._pred = tf.layers.dense(self._dnn_hiddenlayer, 3, kernel_initializer=tf.random_normal_initializer(0, 0.1),)

        self._loss = loss = tf.losses.softmax_cross_entropy(onehot_labels=self._y, logits=self._pred)  # compute cost
        # self._train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

        # gradient clipping
        self._tvars = tf.trainable_variables()
        self._grads, _ = tf.clip_by_global_norm(tf.gradients(loss, self._tvars), clip_norm=5)
        self._optimizer = tf.train.AdamOptimizer(self._lr)
        self._train_op = self._optimizer.apply_gradients(
            zip(self._grads, self._tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)


        # Evaluate model
        # separately evaluate the slot-value
        self._results = tf.argmax(self._pred, 1)
        self._correct_pred = tf.equal(self._results, tf.argmax(self._y, 1))
        self._probability = tf.nn.softmax(self._pred)
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_pred, tf.float32))

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def accuracy(self):
        return self._accuracy

    def train_op(self):
        return self._train_op

    def loss(self):
        return self._loss

# slot-specific tracker
class Slot_Specific_tracker(object):
    def __init__(self, slot_name):
        """
        对每个informable slot 都维护一个 slot specific tracker
        :param slot_name: 名字
        """
        self._slot_name = slot_name
        # tf Graph input
        self._x_last = tf.placeholder(tf.float32, [None,3])
        self._x_usr = tf.placeholder(tf.float32, [None, max_length, embedding_dim])
        self._x_usr_len = tf.placeholder(tf.int32, [None])
        self._x_sys_acts = tf.placeholder(tf.float32, [None, 4])
        # request/confirm_slot/confirm_value+/confirm_value-
        self._x_stringmatch_DONTCARE = tf.placeholder(tf.float32, [None, 1])  # whether DONT_CARE
        self._x_slot = tf.placeholder(tf.float32, [None, embedding_dim])
        self._is_training = tf.placeholder(tf.bool)
        self._y = tf.placeholder(tf.float32, [None, 3])
        self._batchsize_slot = tf.shape(self._x_slot)[0]

        self._lr = tf.Variable(0.001, trainable=False)
        # self._x_usr = tf.layers.dropout(self._x_usr, rate=0.5, training=self._is_training)

        # # slot-specific maxpooling
        # self._x_slot_m = tf.expand_dims(self._x_slot, 2)
        # self._W1 = tf.Variable(0.1, dtype=tf.float32)
        # self._b1 = tf.Variable(0, dtype=tf.float32)
        # self._slot_sequence = tf.multiply(self._W1,tf.matmul(self._x_usr, self._x_slot_m))+ self._b1
        # self._maxpool = tf.squeeze(tf.layers.max_pooling1d(self._slot_sequence, pool_size=max_length, strides=1),
        #                            2)  # ->(?, 1)

        # string matching results
        self._W5 = tf.Variable(5.0, dtype=tf.float32)
        self._b5 = tf.Variable(0.0, dtype=tf.float32)
        self._stringmatch_DONTCARE = tf.multiply(self._W5, self._x_stringmatch_DONTCARE) + self._b5

        with tf.variable_scope("last_state"):
            self._conv1 = tf.layers.conv1d(
                inputs=self._x_usr,
                filters=20,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool1 = tf.layers.max_pooling1d(
                self._conv1,
                pool_size=max_length,
                strides=1,
            )
            self._pool1 = tf.squeeze(self._pool1, [1])
            # bigram filters
            self._conv2 = tf.layers.conv1d(
                inputs=self._x_usr,
                filters=20,
                kernel_size=2,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool2 = tf.layers.max_pooling1d(
                self._conv2,
                pool_size=max_length,
                strides=1,
            )
            self._pool2 = tf.squeeze(self._pool2, [1])

            # trigram
            self._conv3 = tf.layers.conv1d(
                inputs=self._x_usr,
                filters=20,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool3 = tf.layers.max_pooling1d(
                self._conv3,
                pool_size=max_length,
                strides=1,
            )

            self._pool3 = tf.squeeze(self._pool3, [1])
            self._output_pool = tf.concat([self._pool1, self._pool2, self._pool3], 1)
            # last belief state
            self._input = tf.concat([self._x_sys_acts,
                                      self._x_stringmatch_DONTCARE,
                                     self._output_pool],
                                     1)

            self._dnn_hiddenlayer = tf.layers.dense(self._input,
                                                    self._input.shape[1] + self._input.shape[1],
                                                    activation=tf.nn.sigmoid,
                                                    kernel_initializer=tf.random_normal_initializer(0, 0.5))

            self._last_state_W = tf.layers.dense(self._dnn_hiddenlayer, 3,
                                         activation=tf.nn.sigmoid,
                                         kernel_initializer=tf.random_normal_initializer(0, 0.5), )

            self._last_state = tf.multiply(self._last_state_W, self._x_last)

        # feed value-specific usr sentence to LSTM
        with tf.variable_scope("x_usr"):
            self._conv1 = tf.layers.conv1d(
                inputs=self._x_usr,
                filters=100,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool1 = tf.layers.max_pooling1d(
                self._conv1,
                pool_size=max_length,
                strides=1,
            )
            self._pool1 = tf.squeeze(self._pool1, [1])
            # bigram filters
            self._conv2 = tf.layers.conv1d(
                inputs=self._x_usr,
                filters=100,
                kernel_size=2,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool2 = tf.layers.max_pooling1d(
                self._conv2,
                pool_size=max_length,
                strides=1,
            )
            self._pool2 = tf.squeeze(self._pool2, [1])

            # trigram
            self._conv3 = tf.layers.conv1d(
                inputs=self._x_usr,
                filters=100,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool3 = tf.layers.max_pooling1d(
                self._conv3,
                pool_size=max_length,
                strides=1,
            )

            self._pool3 = tf.squeeze(self._pool3, [1])
            self._output_pool = tf.concat([self._pool1, self._pool2, self._pool3], 1)
            self._input = tf.concat([self._x_sys_acts],
                                     1)

            self._dnn_hiddenlayer = tf.layers.dense(self._input,
                                                    self._input.shape[1] + self._input.shape[1],
                                                    activation=tf.nn.sigmoid,
                                                    kernel_initializer=tf.random_normal_initializer(0, 0.5))

            self._output_final_W = tf.layers.dense(self._dnn_hiddenlayer, 1,
                                                 activation=tf.nn.sigmoid,
                                                 kernel_initializer=tf.random_normal_initializer(0, 0.5), )
            self._output_finals = tf.multiply(self._output_final_W, self._output_pool)
        with tf.variable_scope("x_usr_request"):
            # feed value-specific usr sentence to CNN
            # unigram filters
            self._conv1 = tf.layers.conv1d(
                inputs=self._x_usr,
                filters=20,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool1 = tf.layers.max_pooling1d(
                self._conv1,
                pool_size=max_length,
                strides=1,
            )
            self._pool1 = tf.squeeze(self._pool1, [1])
            # bigram filters
            self._conv2 = tf.layers.conv1d(
                inputs=self._x_usr,
                filters=20,
                kernel_size=2,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool2 = tf.layers.max_pooling1d(
                self._conv2,
                pool_size=max_length,
                strides=1,
            )
            self._pool2 = tf.squeeze(self._pool2, [1])

            # trigram
            self._conv3 = tf.layers.conv1d(
                inputs=self._x_usr,
                filters=20,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool3 = tf.layers.max_pooling1d(
                self._conv3,
                pool_size=max_length,
                strides=1,
            )

            self._pool3 = tf.squeeze(self._pool3, [1])
            self._output_pool = tf.concat([self._pool1, self._pool2, self._pool3], 1)
            # context modelling
            self._context_request = tf.multiply(self._output_pool,tf.expand_dims(self._x_sys_acts[:,0],1))
        with tf.variable_scope("x_usr_confirm_slot"):
            # unigram filters
            self._conv1 = tf.layers.conv1d(
                inputs=self._x_usr,
                filters=20,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool1 = tf.layers.max_pooling1d(
                self._conv1,
                pool_size=max_length,
                strides=1,
            )
            self._pool1 = tf.squeeze(self._pool1, [1])
            # bigram filters
            self._conv2 = tf.layers.conv1d(
                inputs=self._x_usr,
                filters=20,
                kernel_size=2,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool2 = tf.layers.max_pooling1d(
                self._conv2,
                pool_size=max_length,
                strides=1,
            )
            self._pool2 = tf.squeeze(self._pool2, [1])

            # trigram
            self._conv3 = tf.layers.conv1d(
                inputs=self._x_usr,
                filters=20,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool3 = tf.layers.max_pooling1d(
                self._conv3,
                pool_size=max_length,
                strides=1,
            )

            self._pool3 = tf.squeeze(self._pool3, [1])
            self._output_pool = tf.concat([self._pool1, self._pool2, self._pool3], 1)
            # context modelling
            self._context_confirm_slot = tf.multiply(self._output_pool, tf.expand_dims(self._x_sys_acts[:,1],1))
        with tf.variable_scope("x_usr_confirm_value_positive"):
            # unigram filters
            self._conv1 = tf.layers.conv1d(
                inputs=self._x_usr,
                filters=20,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool1 = tf.layers.max_pooling1d(
                self._conv1,
                pool_size=max_length,
                strides=1,
            )
            self._pool1 = tf.squeeze(self._pool1, [1])
            # bigram filters
            self._conv2 = tf.layers.conv1d(
                inputs=self._x_usr,
                filters=20,
                kernel_size=2,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool2 = tf.layers.max_pooling1d(
                self._conv2,
                pool_size=max_length,
                strides=1,
            )
            self._pool2 = tf.squeeze(self._pool2, [1])

            # trigram
            self._conv3 = tf.layers.conv1d(
                inputs=self._x_usr,
                filters=20,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool3 = tf.layers.max_pooling1d(
                self._conv3,
                pool_size=max_length,
                strides=1,
            )

            self._pool3 = tf.squeeze(self._pool3, [1])
            self._output_pool = tf.concat([self._pool1, self._pool2, self._pool3], 1)
            self._context_confirm_value_positive = tf.multiply(self._output_pool, tf.expand_dims(self._x_sys_acts[:,2],1))
        with tf.variable_scope("x_usr_confirm_value_negative"):
            # unigram filters
            self._conv1 = tf.layers.conv1d(
                inputs=self._x_usr,
                filters=20,
                kernel_size=1,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool1 = tf.layers.max_pooling1d(
                self._conv1,
                pool_size=max_length,
                strides=1,
            )
            self._pool1 = tf.squeeze(self._pool1, [1])
            # bigram filters
            self._conv2 = tf.layers.conv1d(
                inputs=self._x_usr,
                filters=20,
                kernel_size=2,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool2 = tf.layers.max_pooling1d(
                self._conv2,
                pool_size=max_length,
                strides=1,
            )
            self._pool2 = tf.squeeze(self._pool2, [1])

            # trigram
            self._conv3 = tf.layers.conv1d(
                inputs=self._x_usr,
                filters=20,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            self._pool3 = tf.layers.max_pooling1d(
                self._conv3,
                pool_size=max_length,
                strides=1,
            )

            self._pool3 = tf.squeeze(self._pool3, [1])
            self._output_pool = tf.concat([self._pool1, self._pool2, self._pool3], 1)
            self._context_confirm_value_negative = tf.multiply(self._output_pool, tf.expand_dims(self._x_sys_acts[:,3],1))

        # assemble
        self._inputs = tf.concat([self._output_finals,
                                  self._context_request,
                                  self._context_confirm_slot,
                                  self._context_confirm_value_positive,
                                  self._context_confirm_value_negative,
                                  self._last_state],
                                 1)
        self._dnn_hiddenlayer = tf.layers.dense(self._inputs,
                                                500,
                                                activation=tf.nn.relu,
                                                kernel_initializer=tf.random_normal_initializer(0, 0.1))
        self._dnn_hiddenlayer = tf.layers.dropout(self._dnn_hiddenlayer, rate=0.5, training=self._is_training)

        self._pred = tf.layers.dense(self._dnn_hiddenlayer, 3,
                                     kernel_initializer=tf.random_normal_initializer(0, 0.1))


        self._loss = loss = tf.losses.softmax_cross_entropy(onehot_labels=self._y, logits=self._pred)  # compute cost
        # self._train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # gradient clipping
        self._tvars = tf.trainable_variables()
        self._grads, _ = tf.clip_by_global_norm(tf.gradients(loss, self._tvars), clip_norm=5)
        self._optimizer = tf.train.AdamOptimizer(self._lr)
        self._train_op = self._optimizer.apply_gradients(
            zip(self._grads, self._tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

        # Evaluate model
        # separately evaluate the slot-value
        self._results = tf.argmax(self._pred, 1)
        self._probability = tf.nn.softmax(self._pred)
        self._correct_pred = tf.equal(self._results, tf.argmax(self._y, 1))
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_pred, tf.float32))

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def accuracy(self):
        return self._accuracy

    def train_op(self):
        return self._train_op

    def loss(self):
        return self._loss

# requestable slot-specific tracker
class Request_Specific_tracker(object):
    def __init__(self, slot_name):
        """
        对每个 requestable slot 维护一个 tracker
        :param slot_name: 名字
        """
        self._slot_name = slot_name
        # tf Graph input
        self._x_usr = tf.placeholder(tf.float32, [None, max_length, embedding_dim])
        self._x_usr_len = tf.placeholder(tf.int32, [None])
        self._x_slot = tf.placeholder(tf.float32, [None, embedding_dim])
        self._is_training = tf.placeholder(tf.bool)
        self._y = tf.placeholder(tf.float32, [None, 2])
        self._batchsize_slot = tf.shape(self._x_slot)[0]
        # value-specific maxpooling
        self._x_slot_m = tf.expand_dims(self._x_slot, 2)
        self._W = tf.Variable(0.1, dtype=tf.float32)
        self._b = tf.Variable(0, dtype=tf.float32)
        self._slot_sequence = tf.matmul(self._x_usr, self._x_slot_m)
        self._slot_sequence_ = tf.multiply(self._W, self._slot_sequence) + self._b # ->(?, 37, 1)

        self._maxpool = tf.squeeze(tf.layers.max_pooling1d(self._slot_sequence_, pool_size=max_length, strides=1),
                                   2)  # ->(?, 1)

        # slot_specific usr sentence
        self._slot_usr = self._slot_sequence_ * self._x_usr  # -> (?, max_length, embedding_size)
        #
        # feed slot-specific usr sentence to LSTM
        self._lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
        self._outputs, _ = tf.nn.dynamic_rnn(self._lstm_cell,
                                             self._slot_usr,
                                             sequence_length=self._x_usr_len,
                                             dtype=tf.float32)

        # Hack to build the indexing and retrieve the right output.
        # Start indices for each sample
        self._index = tf.range(0, self._batchsize_slot) * max_length + (self._x_usr_len - 1)
        # Indexing
        self._output_final = tf.gather(tf.reshape(self._outputs, [-1, hidden_size]), self._index)  # ->(?, hidden_size)

        # assemble
        # self._inputs = tf.concat([self._output_final], 1)

        self._dnn_hiddenlayer = tf.layers.dense(self._output_final,
                                                50,
                                                activation=tf.nn.relu,
                                                kernel_initializer=tf.random_normal_initializer(0, 0.1))

        self._dnn_hiddenlayer = tf.layers.dropout(self._dnn_hiddenlayer, 0.5, training=self._is_training)
        self._pred = tf.layers.dense(self._dnn_hiddenlayer, 2, kernel_initializer=tf.random_normal_initializer(0, 0.1))

        self._loss = loss = tf.losses.softmax_cross_entropy(onehot_labels=self._y, logits=self._pred)  # compute cost
        # self._train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # gradient clipping
        self._tvars = tf.trainable_variables()
        self._grads, _ = tf.clip_by_global_norm(tf.gradients(loss, self._tvars), clip_norm=5)
        self._optimizer = tf.train.AdamOptimizer(learning_rate)
        self._train_op = self._optimizer.apply_gradients(
            zip(self._grads, self._tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        # Evaluate model
        # separately evaluate the slot-value
        self._results = tf.argmax(self._pred, 1)
        self._probability = tf.nn.softmax(self._pred)[:,0]
        self._correct_pred = tf.equal(tf.argmax(self._pred, 1), tf.argmax(self._y, 1))
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_pred, tf.float32))

    def accuracy(self):
        return self._accuracy

    def train_op(self):
        return self._train_op

    def loss(self):
        return self._loss

def value_specific_data2num(batch_data):
    """
    input: next_batch_informable["value-specific"][slot].
    change the data into processable type for tensorflow
    :param batch_data: 一个 batch 的 训练数据
    :return: 直接输入value-specific tracker 模型计算的数据
    """
    batchsize_value = len(batch_data)
    x_last = np.zeros((batchsize_value, 3))
    x_usr = np.zeros((batchsize_value, max_length, embedding_dim))
    x_usr_len = np.zeros((batchsize_value), dtype='int32')
    x_sys_requset = np.zeros((batchsize_value, 1))
    x_sys_confirm_slot = np.zeros((batchsize_value, 1))
    x_sys_confirm_value_positive = np.zeros((batchsize_value, 1))
    x_sys_confirm_value_negative = np.zeros((batchsize_value, 1))
    x_sys_inform_one_match = np.zeros((batchsize_value, 1))
    x_stringmatch = np.zeros((batchsize_value,  max_length, 1))
    x_value = np.zeros((batchsize_value, embedding_dim))

    for batch_id, data in enumerate(batch_data):
        # data:[act, usr, s,v,last_state_s, last_state_v]
        for confirm_dict in data[0]['confirm_positive']:
            if confirm_dict[0] == data[2] and confirm_dict[1] == data[3]:
                x_sys_confirm_value_positive[batch_id,0] = 1
            if confirm_dict[0] == data[2] and confirm_dict[1] != data[3]:
                x_sys_confirm_slot[batch_id, 0] = 1
        for confirm_dict in data[0]['confirm_negative']:
            if confirm_dict[0] == data[2] and confirm_dict[1] == data[3]:
                x_sys_confirm_value_negative[batch_id,0] = 1
            if confirm_dict[0] == data[2] and confirm_dict[1] != data[3]:
                x_sys_confirm_slot[batch_id, 0] = 1
        for request_item in data[0]['request']:
            if request_item == data[2]:
                x_sys_requset[batch_id,0] = 1
        if data[0]['inform_one_match'] == data[3]:
            x_sys_inform_one_match[batch_id,0] = 1

        for word_id, word in enumerate(data[1]):
            if word in vocab_dict:
                x_usr[batch_id, word_id, :] = embedding_table[word]
            else:
                x_usr[batch_id, word_id, :] = embedding_table['unk']
            for candidate in semantic_dict["informable"][data[2]][data[3]]:
                if word == candidate:
                    x_stringmatch[batch_id, word_id,0] = 1

        x_usr_len[batch_id] = len(data[1])
        if data[5] == "LIKE":
            x_last[batch_id, 0] = 1
        elif data[5] == "DISLIKE":
            x_last[batch_id, 1] = 1
        else:
            x_last[batch_id, 2] = 1
        x_value[batch_id, :] = embedding_table[data[3]]
    x_sys_act = np.concatenate((x_sys_requset,
                                x_sys_confirm_slot,
                                x_sys_confirm_value_positive,
                                x_sys_confirm_value_negative,
                                x_sys_inform_one_match)
                               , 1)
    return x_sys_act, x_usr, x_usr_len, x_stringmatch, x_value, x_last

def slot_specific_data2num(batch_data):
    """
    input: next_batch_informable["slot-specific"][slot].
    change the data into processable type for tensorflow
    :param batch_data: 一个 batch 的训练数据
    :return: 直接输入slot-specific tracker 模型计算的数据
    """
    batchsize_slot = len(batch_data)
    x_last = np.zeros((batchsize_slot, 3))
    x_usr = np.zeros((batchsize_slot, max_length, embedding_dim))
    x_usr_len = np.zeros((batchsize_slot), dtype='int32')
    x_sys_requset = np.zeros((batchsize_slot, 1))
    x_sys_confirm_slot = np.zeros((batchsize_slot, 1))
    x_sys_confirm_value_positive = np.zeros((batchsize_slot, 1))
    x_sys_confirm_value_negative = np.zeros((batchsize_slot, 1))
    x_slot = np.zeros((batchsize_slot, embedding_dim))
    x_stringmatch_DONTCARE = np.zeros((batchsize_slot, 1))

    for batch_id, data in enumerate(batch_data):
    # data : [act, usr, s, last_state]
        for confirm_dict in data[0]['confirm_positive']:
            if confirm_dict[0] == data[2] and confirm_dict[1] == "DONTCARE":
                x_sys_confirm_value_positive[batch_id, 0] = 1
            if confirm_dict[0] == data[2] and confirm_dict[1] != "DONTCARE":
                x_sys_confirm_slot[batch_id, 0] = 1
        for confirm_dict in data[0]['confirm_negative']:
            if confirm_dict[0] == data[2] and confirm_dict[1] == "DONTCARE":
                x_sys_confirm_value_negative[batch_id, 0] = 1
            if confirm_dict[0] == data[2] and confirm_dict[1] != "DONTCARE":
                x_sys_confirm_slot[batch_id, 0] = 1
        for request_item in data[0]['request']:
            if request_item == data[2]:
                x_sys_requset[batch_id, 0] = 1

        for word_id, word in enumerate(data[1]):
            if word in vocab_dict:
                x_usr[batch_id, word_id, :] = embedding_table[word]
            else:
                x_usr[batch_id, word_id, :] = embedding_table['unk']
        x_usr_len[batch_id] = len(data[1])
        if data[3] == "DONT_CARE":
            x_last[batch_id, 0] = 1
        elif data[3] == "MENTIONED":
            x_last[batch_id, 1] = 1
        else:
            x_last[batch_id, 2] = 1
        x_slot[batch_id, :] = embedding_table[data[2]]
        flag_DONTCARE = 0
        sent = ' ' + ' '.join(data[1]) + ' '
        for candidate in [' 随意 '," 都可以 ",' 都行 ',' 无所谓 '," 没所谓 "," 随便 "," 不限 ",
                          " 不太关心 "," 不关心 ", " 任意 ", " 没关系 "]:
            if candidate in sent:
                flag_DONTCARE = 1
            break
        x_stringmatch_DONTCARE[batch_id,0] = flag_DONTCARE

    x_sys_act = np.concatenate((x_sys_requset,
                                x_sys_confirm_slot,
                                x_sys_confirm_value_positive,
                                x_sys_confirm_value_negative)
                               , 1)

    return x_sys_act, x_usr, x_usr_len,  x_stringmatch_DONTCARE, x_slot, x_last

def request_specific_data2num(batch_data):
    """
    input: next_batch_requestable  request_specific_data[slot].
    change the data into processable type for tensorflow
    :param batch_data: 一个 batch 的训练数据
    :return: 直接输入request-specific tracker 模型计算的数据
    """
    batchsize_request = len(batch_data)
    x_usr = np.zeros((batchsize_request, max_length, embedding_dim))
    x_usr_len = np.zeros((batchsize_request), dtype='int32')
    x_slot = np.zeros((batchsize_request, embedding_dim))

    for batch_id, data in enumerate(batch_data):
        for word_id, word in enumerate(data[1]):
            if word in vocab_dict:
                x_usr[batch_id, word_id, :] = embedding_table[word]
            else:
                x_usr[batch_id, word_id, :] = embedding_table['unk']
        x_usr_len[batch_id] = len(data[1])
        x_slot[batch_id, :] = embedding_table[data[2]]

    return x_usr, x_usr_len, x_slot

def extract_transcripts(line):
    """
    预处理句子
    :param line: 输出自然语句
    :return: 分好词的句子, 类型为list
    """
    for slot in filter_slots:
        for value in slot:
            if value in line:
                line = line.replace(value, " " + value + " ")
                break
    return ' '.join(jieba.cut(line + '。')).split()

def extract_sys_act(acts):
    """
    预处理 sys acts
    :param acts: 原始 acts
    :return: sys_act 字典类型
    """
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


class DialogSystem:
    def __init__(self):
        self.turn_num = 0
        self.dialog_history = []
        self.turn_num = 0
        self.last_belief_states = {
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
        self.system_acts = [{"diaact": "hello"}]
        self.is_belief_states_differ = None
        self.KB_pointer = None
        with tf.Graph().as_default():
            with tf.name_scope("Value_Specific"):
                with tf.variable_scope("vname"):
                    self.value_tracker_name = Value_Specific_tracker("name")
                with tf.variable_scope("vdirector"):
                    self.value_tracker_director = Value_Specific_tracker("director")
                with tf.variable_scope("vactor"):
                    self.value_tracker_actor = Value_Specific_tracker("actor")
                with tf.variable_scope("vtype"):
                    self.value_tracker_type = Value_Specific_tracker("type")
                with tf.variable_scope("varea"):
                    self.value_tracker_area = Value_Specific_tracker("area")
                with tf.variable_scope("vpayment"):
                    self.value_tracker_payment = Value_Specific_tracker("payment")
                with tf.variable_scope("vera"):
                    self.value_tracker_era = Value_Specific_tracker("era")

            with tf.name_scope("Slot_Specific"):
                with tf.variable_scope("sdirector"):
                    self.slot_tracker_director = Slot_Specific_tracker("director")
                with tf.variable_scope("sactor"):
                    self.slot_tracker_actor = Slot_Specific_tracker("actor")
                with tf.variable_scope("stype"):
                    self.slot_tracker_type = Slot_Specific_tracker("type")
                with tf.variable_scope("sarea"):
                    self.slot_tracker_area = Slot_Specific_tracker("area")
                with tf.variable_scope("spayment"):
                    self.slot_tracker_payment = Slot_Specific_tracker("payment")
                with tf.variable_scope("sera"):
                    self.slot_tracker_era = Slot_Specific_tracker("era")

            with tf.name_scope("Requestable"):
                with tf.variable_scope("rdirector"):
                    self.request_tracker_director = Request_Specific_tracker("director")
                with tf.variable_scope("ractor"):
                    self.request_tracker_actor = Request_Specific_tracker("actor")
                with tf.variable_scope("rtype"):
                    self.request_tracker_type = Request_Specific_tracker("type")
                with tf.variable_scope("rarea"):
                    self.request_tracker_area = Request_Specific_tracker("area")
                with tf.variable_scope("rera"):
                    self.request_tracker_era = Request_Specific_tracker("era")
                with tf.variable_scope("rpayment"):
                    self.request_tracker_payment = Request_Specific_tracker("payment")
                with tf.variable_scope("rdate"):
                    self.request_tracker_date = Request_Specific_tracker("date")
                with tf.variable_scope("rscore"):
                    self.request_tracker_score = Request_Specific_tracker("score")
                with tf.variable_scope("rintro"):
                    self.request_tracker_intro = Request_Specific_tracker("intro")
                with tf.variable_scope("rlength"):
                    self.request_tracker_length = Request_Specific_tracker("length")



            self.sess = tf.Session()
            self.sess.run(tf.group(tf.global_variables_initializer()))
            saver = tf.train.Saver()

            saver.restore(self.sess, "./model_ckpt/model.ckpt")
            self.value_tracker = {
                "主演": self.value_tracker_actor,
                "导演": self.value_tracker_director,
                "类型": self.value_tracker_type,
                "地区": self.value_tracker_area,
                "年代": self.value_tracker_era,
                "资费": self.value_tracker_payment,
                "片名": self.value_tracker_name
            }
            self.slot_tracker = {
                "主演": self.slot_tracker_actor,
                "导演": self.slot_tracker_director,
                "类型": self.slot_tracker_type,
                "地区": self.slot_tracker_area,
                "年代": self.slot_tracker_era,
                "资费": self.slot_tracker_payment
            }
            self.request_tracker = {
                    "导演": self.request_tracker_director,
                    "主演": self.request_tracker_actor,
                    "类型": self.request_tracker_type,
                    "地区": self.request_tracker_area,
                    "年代": self.request_tracker_era,
                    "资费": self.request_tracker_payment,
                    "上映日期": self.request_tracker_date,
                    "评分": self.request_tracker_score,
                    "简介": self.request_tracker_intro,
                    "片长": self.request_tracker_length
                }

    def evaluate_belief_states(self, turn):
        """
        计算 current belief states
        :param turn: 字典包含 system_act, usr_transcript, last_belief_states
        :return: current belief states
        """
        turn_label_pred = {
            "片名": {
            },
            "导演": {
            },
            "主演": {
            },
            "类型": {
            },
            "地区": {
            },
            "年代": {
            },
            "资费": {
            }
        }

        for slot in informable_slots:
            data = []
            data.append(turn['system_act'])
            data.append(turn['usr_transcript'])
            data.append(slot)
            all_value_data = []
            for val in OTGY["informable"][slot]:
                data.append(val)
                data.append(turn["last_belief_states"][slot][slot])
                if val in turn["last_belief_states"][slot]:
                    data.append(turn["last_belief_states"][slot][val])
                else:
                    data.append("NOT_MENTIONED")
                all_value_data.append(copy.deepcopy(data))
                data.pop()
                data.pop()
                data.pop()

            value_num = value_specific_data2num(all_value_data)
            del all_value_data

            value_prediction = self.sess.run(self.value_tracker[slot]._probability,
                                        feed_dict={
                                            self.value_tracker[slot]._x_sys_acts: value_num[0],
                                            self.value_tracker[slot]._x_usr: value_num[1],
                                            self.value_tracker[slot]._x_usr_len: value_num[2],
                                            self.value_tracker[slot]._x_stringmatch: value_num[3],
                                            self.value_tracker[slot]._x_value: value_num[4],
                                            self.value_tracker[slot]._x_last: value_num[5],
                                            self.value_tracker[slot]._is_training: False
                                        }
                                        )
            del value_num

            if slot != "片名":
                # 片名 has no  DONT_CARE in this dataset
                data = []
                data.append(turn['system_act'])
                data.append(turn['usr_transcript'])
                data.append(slot)
                data.append(turn["last_belief_states"][slot][slot])
                slot_num = slot_specific_data2num([data])
                slot_prediction = self.sess.run(
                    self.slot_tracker[slot]._probability,
                    feed_dict={
                        self.slot_tracker[slot]._x_sys_acts: slot_num[0],
                        self.slot_tracker[slot]._x_usr: slot_num[1],
                        self.slot_tracker[slot]._x_usr_len: slot_num[2],
                        self.slot_tracker[slot]._x_stringmatch_DONTCARE: slot_num[3],
                        self.slot_tracker[slot]._x_slot: slot_num[4],
                        self.slot_tracker[slot]._x_last: slot_num[5],
                        self.slot_tracker[slot]._is_training: False
                    }
                )
                del slot_num

            # 资费 片名 单独处理
            if slot != "片名" and slot != "资费":
                sflag = 0
                sp = np.argmax(slot_prediction, 1)
                if sp[0] == 0:
                    turn_label_pred[slot][slot] = "DONT_CARE"
                elif sp[0] == 1:
                    turn_label_pred[slot][slot] = "MENTIONED"
                    sflag = 1
                else:
                    turn_label_pred[slot][slot] = "NOT_MENTIONED"

                if sflag == 1:
                    vflag = 0
                    vi = np.argmax(value_prediction, 1)
                    for ii, pp in enumerate(vi):
                        if pp == 0:
                            MENTIONED_value = OTGY["informable"][slot][ii]
                            turn_label_pred[slot][MENTIONED_value] = "LIKE"
                            vflag = 1
                        elif pp == 1:
                            MENTIONED_value = OTGY["informable"][slot][ii]
                            turn_label_pred[slot][MENTIONED_value] = "DISLIKE"
                            vflag = 1
                    if vflag == 0:
                        turn_label_pred[slot][slot] = "NOT_MENTIONED"
            elif slot == "片名":
                flag = 0
                vi = np.argmax(value_prediction, 1)
                # 只取一个value LIKE
                pro = 0
                idx = 0
                for ii, pp in enumerate(vi):
                    if pp == 0:
                        if value_prediction[ii][0] > pro:
                            pro = value_prediction[ii][0]
                            idx = ii
                        flag = 1
                    elif pp == 1:
                        MENTIONED_value = OTGY["informable"][slot][ii]
                        turn_label_pred[slot][MENTIONED_value] = "DISLIKE"
                        flag = 1
                if pro > 1 / 3:
                    turn_label_pred[slot][OTGY["informable"][slot][idx]] = "LIKE"
                if flag == 1:
                    turn_label_pred[slot][slot] = "MENTIONED"
                else:
                    turn_label_pred[slot][slot] = "NOT_MENTIONED"
            else:
                sflag = 0
                sp = np.argmax(slot_prediction, 1)
                if sp[0] == 0:
                    turn_label_pred[slot][slot] = "DONT_CARE"
                elif sp[0] == 1:
                    turn_label_pred[slot][slot] = "MENTIONED"
                    sflag = 1
                else:
                    turn_label_pred[slot][slot] = "NOT_MENTIONED"

                if sflag == 1:
                    vflag = 0
                    vi = np.argmax(value_prediction, 1)
                    for ii, pp in enumerate(vi):
                        if pp == 0:
                            vflag = 1
                            MENTIONED_value = OTGY["informable"][slot][ii]
                            turn_label_pred[slot][MENTIONED_value] = "LIKE"
                            if MENTIONED_value == "付费":
                                turn_label_pred[slot]["免费"] = "DISLIKE"
                            else:
                                turn_label_pred[slot]["付费"] = "DISLIKE"
                    if vflag == 0:
                        turn_label_pred[slot][slot] = "NOT_MENTIONED"

        return turn_label_pred

    def evaluate_metric_requestable(self, turn):
        """
        we find requestable slots MENTIONED
        :param turn: 当前轮次的数据
        :return: requests evaluation
        """
        input_asr = turn["usr_transcript"]
        turn_label_pred = {}
        x_usr = np.zeros((1, max_length, embedding_dim))
        x_usr_len = np.zeros((1,), dtype='int32')

        for i, word in enumerate(input_asr):
            if word in vocab_dict:
                x_usr[0, i] = embedding_table[word]
            else:
                x_usr[0, i] = embedding_table['unk']
        x_usr_len[0] = len(input_asr)
        for slot in requestable_slots:
            x_slot = np.zeros((1, embedding_dim))
            x_slot[0, :] = embedding_table[slot]
            prediction = self.sess.run(self.request_tracker[slot]._probability,
                                  feed_dict={self.request_tracker[slot]._x_usr: x_usr,
                                             self.request_tracker[slot]._x_usr_len: x_usr_len,
                                             self.request_tracker[slot]._x_slot: x_slot,
                                             self.request_tracker[slot]._is_training: False})

            if prediction[0] > 0.5:
                turn_label_pred[slot] = True
        return turn_label_pred

    def update(self, input_usr):
        if "重来" in input_usr or "谢谢" in input_usr or "结束" in input_usr:  # 用户重来
            self.turn_num = 0
            self.last_belief_states = {
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
            self.system_acts = [{"diaact": "hello"}]
            self.is_belief_states_differ = None
            self.KB_pointer = None
            self.dialog_history = []
            return self.dialog_history

        elif "好的" in input_usr or "那就看" in input_usr:  # 播放电影
            self.dialog_history.append(("usr: "+input_usr, "sys: 正在为您播放,请稍后...(输入\"重来\"开启新对话)"))
            return self.dialog_history

        elif "换" in input_usr or "都不要" in input_usr or "其他" in input_usr or "其它" in input_usr:
            self.turn_num += 1
            self.belief_states = copy.deepcopy(self.last_belief_states)
            if self.system_acts[0]["diaact"] == "inform_many_match":
                self.belief_states["片名"]["片名"] = "MENTIONED"
                for name in self.system_acts[0]["片名"]:
                    self.belief_states["片名"][name] = "DISLIKE"
            self.last_belief_states = copy.deepcopy(self.belief_states)
            # 生成 system_acts
            self.system_acts, self.KB_pointer = rule_method(self.belief_states, self.is_belief_states_differ,
                                                                   {}, self.KB_pointer, DB)

            # 输出中间结果
            print("  belief_states: ", end='')
            for slot in self.belief_states:
                if self.belief_states[slot][slot] != "NOT_MENTIONED":
                    print(self.belief_states[slot], end='；')
            print()
            print("  requested_slots: ", {})

            # 生成回复
            reply = rule_NLG(self.system_acts, self.KB_pointer)
            self.dialog_history.append(("usr: " + input_usr, "sys: " + reply))
            return self.dialog_history

        else:  # 用户正常对话
            self.turn_num += 1
            current_turn = {}
            current_turn["usr_transcript"] = extract_transcripts(input_usr)
            print(current_turn["usr_transcript"])
            current_turn["last_belief_states"] = self.last_belief_states
            current_turn["system_act"] = extract_sys_act(self.system_acts)

            # 计算 belief states
            self.belief_states = self.evaluate_belief_states(current_turn)
            if self.last_belief_states == self.belief_states:
                self.is_belief_states_differ = False
            else:
                self.is_belief_states_differ = True
                self.last_belief_states = copy.deepcopy(self.belief_states)
            # 计算 requested slots
            requested_slots = self.evaluate_metric_requestable(current_turn)

            # 生成 system_acts
            self.system_acts, self.KB_pointer = rule_method(self.belief_states, self.is_belief_states_differ,
                                                  requested_slots, self.KB_pointer, DB)

            # 输出中间结果
            print("  belief_states: ", end='')
            for slot in self.belief_states:
                if self.belief_states[slot][slot] != "NOT_MENTIONED":
                    print(self.belief_states[slot], end='；')
            print()
            print("  requested_slots: ", requested_slots)

            # 生成回复
            reply = rule_NLG(self.system_acts, self.KB_pointer)
            self.dialog_history.append(("usr: " + input_usr,"sys: " + reply))
            return self.dialog_history

            # except Exception as e:
        #     print("=== 出现异常重新开始 ===")
        #     turn_num = 0
        #     last_belief_states = {
        #         "片名": {
        #             "片名": "NOT_MENTIONED"
        #         },
        #         "导演": {
        #             "导演": "NOT_MENTIONED"
        #         },
        #         "主演": {
        #             "主演": "NOT_MENTIONED"
        #         },
        #         "类型": {
        #             "类型": "NOT_MENTIONED"
        #         },
        #         "地区": {
        #             "地区": "NOT_MENTIONED"
        #         },
        #         "年代": {
        #             "年代": "NOT_MENTIONED"
        #         },
        #         "资费": {
        #             "资费": "NOT_MENTIONED"
        #         }
        #     }
        #     system_acts = [{"diaact": "hello"}]
        #     is_belief_states_differ = None
        #     KB_pointer = None


if __name__ == '__main__':
    ss = DialogSystem()
    while True:
        input_asr = input("输入：")
        returns = ss.update(input_asr)
        if len(returns) == 0:
            print()
        else:
            print(returns[-1])







