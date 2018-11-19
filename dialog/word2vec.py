'''
This file is to train word vectors.
'''
import collections
import math
import jieba
import pickle
import copy
import random
import numpy as np
import tensorflow as tf
import jieba,json

with open('data/Iqiyi_ONTO.json', 'r',encoding='utf-8') as f:
    OTGY = json.load(f)

actor = OTGY["informable"]["主演"]
actor.sort(key=lambda x:len(x),reverse=True)
area = OTGY["informable"]["地区"]
area.sort(key=lambda x:len(x),reverse=True)
director = OTGY["informable"]["导演"]
director.sort(key=lambda x:len(x),reverse=True)
year = OTGY["informable"]["年代"]
title = OTGY["informable"]["片名"]
title.sort(key=lambda x:len(x),reverse=True)
movietype = OTGY["informable"]["类型"]
payment = ["付费", "免费"]
requestable_slots = OTGY["requestable"]
ontology = [actor, area, director, year, title, movietype, payment, requestable_slots]
# Step 1: Build the rawdata

# def add_new_words():
#     # 添加新词
#     with open('New_words.txt','w+',encoding='utf-8') as f:
#         for i in ontology:
#             for j in i:
#                 f.write(j+'\n')
#         for i in movietype:
#             f.write(i + '片\n')

jieba.load_userdict("data/New_words.txt")

def artificial_data():
    """人工造一些对话数据"""
    datas = []
    for ii in actor:
        line = "有没有 "+ii+" 主演的电影"
        datas.append(line)
        line = "我想看 "+ii+" 主演的影片"
        datas.append(line)
        line = "有 " + ii + " 演的电影吗？"
        datas.append(line)
    for ii in area:
        line = "有没有 " + ii + " 上映的电影"
        datas.append(line)
        line = "我想看 " + ii + " 地区的影片"
        datas.append(line)
    for ii in director:
        line = "有 " + ii + " 导演的电影吗"
        datas.append(line)
        line = "喜欢 " + ii + " 执导的影片"
        datas.append(line)
    for ii in year:
        line = "我想看 " + ii + " 年代的电影"
        datas.append(line)
        line = "有 " + ii + " 年代的电影吗？"
        datas.append(line)
        line = "有 " + ii + " 上映的的电影吗？"
        datas.append(line)
    for ii in title:
        line = "我想看 " + ii + "，有吗？"
        datas.append(line)
        line = "有没有 " + ii + "？"
        datas.append(line)
    for ii in movietype:
        line = "来一部 " + ii + " 片吧"
        datas.append(line)
        line = "我想看 " + ii + " 类的电影"
        datas.append(line)
        line = "有没有 " + ii + " 片？"
        datas.append(line)
    random.shuffle(datas)
    return datas

def filter(line):
    """
    过滤出命名实体
    :param line: 字符串句子
    :return: 分割出命名实体的句子
    """
    informable_slots = [actor, director, title]
    for slot in informable_slots:
        for value in slot:
            if value in line:
                line = line.replace(value, " " + value + " ")
                break
    return line

def generate_raw_data():
    """生成经过分词后的数据集"""
    raw_data = []
    # 原始数据 + 人工数据
    linelist = []
    with open('data/Iqiyi_800.json','r',encoding='utf-8') as f:
        Iqiyi_data = json.load(f)
    for dial in Iqiyi_data:
        for turn in dial["dialog"]:
            linelist.append(turn["user_transcript"])

    linelist.extend(artificial_data())
    # 打乱顺序
    random.shuffle(linelist)
    maxlength = 0
    for line in linelist:
        cc = ' '.join(jieba.cut(filter(line) + '。')).split()
        maxlength = max(maxlength, len(cc))
        raw_data.extend(cc)

    with open("data/rawdata.json",'w',encoding='utf-8') as f0:
        json.dump(raw_data,f0,ensure_ascii=False)
    return maxlength
maxlen = generate_raw_data()
print("max len of all sentence: " , maxlen)  #  31


# Step 2: Build the dictionary and replace rare words with UNK token.
with open("data/rawdata.json", 'r', encoding='utf-8') as f:
    rawdata = json.load(f)
def build_dataset(words):
    """
    Process raw inputs into a dataset
    :param words: rawdata
    :return: data, count, dictionary, reversed_dictionary
    """
    count = [['unk', -1]]
    count.extend(collections.Counter(words).most_common())
    dictionary = dict()
    for word, _ in count:
          dictionary[word] = len(dictionary)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    data = []
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(copy.deepcopy(index))
    count[0][1] = unk_count
    return data, count, dictionary, reversed_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(rawdata)
data_index = 0
vocabulary_size = len(dictionary) #  1527 words
print("vocab size:",vocabulary_size)
with open('data/vocab_dict.json','w',encoding='utf-8') as f:
    json.dump(dictionary,f,ensure_ascii=False)

# 处理近义词，用于后续词向量的学习
synonym_inputs_ = []
with open('data/Synonyms.txt','r',encoding='utf-8') as f:
    for line in f:
        line = line.lstrip('\ufeff')
        words = line.split()
        if words[0] in dictionary and words[1] in dictionary:
            synonym_inputs_.append([dictionary[words[0]], dictionary[words[1]]])


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  """
  生成训练数据
  :param batch_size: minibatch大小
  :param num_skips: 跳词数目
  :param skip_window: 窗长
  :return: 数据
  """
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

# # Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 25  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_words = [
        "片长",
        "主演",
        "导演",
        "类型",
        "地区",
        "年代",
        "资费",
        "片长",
        "评分",
        "简介",
        "上映日期",
        "免费",
        "付费",
        "美国",
        "中国大陆",
        "惊悚",
        "恐怖",
        "不",
        "喜欢"
    ]

valid_size = len(valid_words)    # Random set of words to evaluate similarity on.
valid_window = 50  # Only pick dev samples in the head of the distribution.
valid_sample = [dictionary[i]  for i in valid_words]
valid_examples = np.array(valid_sample)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.

  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  synonym_inputs = tf.placeholder(tf.int32, shape=[None,2])
  embed0 = tf.nn.embedding_lookup(embeddings, synonym_inputs[:, 0])
  embed1 = tf.nn.embedding_lookup(embeddings, synonym_inputs[:, 1])
  synonym_loss = tf.sqrt(tf.reduce_mean(tf.square(embed0-embed1)))


  final_loss = loss + 0.035 * synonym_loss

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(final_loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = tf.div(embeddings, norm)
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in range(num_steps):
    batch_inputs_, batch_labels_ = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs_,
                 train_labels: batch_labels_,
                 synonym_inputs: synonym_inputs_}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, final_loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[:top_k]
        log_str = 'Nearest to %s:' % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s %.4f,' % (log_str, close_word,sim[i,nearest[k]])
        print(log_str)
  final_embeddings_norm = normalized_embeddings.eval()
  final_embeddings = embeddings.eval()
  vocab_norm = {}
  for ii,embedding in enumerate(final_embeddings_norm):
      vocab_norm[reverse_dictionary[ii]] = embedding
  vocab ={}
  for ii,embedding in enumerate(final_embeddings):
      vocab[reverse_dictionary[ii]] = embedding
  with open('data/vocab_norm.25d.pkl','wb') as f:
      pickle.dump(vocab_norm,f)
  with open('data/vocab.25d.pkl','wb') as f:
      pickle.dump(vocab,f)



# Step 6: Visualize the embeddings.
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom',)

  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in range(plot_only)]
  plot_with_labels(low_dim_embs, labels, 'tsne.png')

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings_norm[:plot_only, :])
  labels = [reverse_dictionary[i] for i in range(plot_only)]
  plot_with_labels(low_dim_embs, labels, 'tsne_norm.png')

except ImportError:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
