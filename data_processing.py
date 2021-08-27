import emoji
import re
import random
import yaml
import matplotlib.pyplot as plt


def remo_emoji_symbol(data_path):
    """
        remove the emojis, '@' symbol and '#' symbol from original dataset
        imput: data_path
        return: a list including examples whose lenght larger than 3 words
    """
    examples_list = []
    with open(data_path, 'r', encoding='utf-8') as file:
        filecontent = file.readlines()
    for line in filecontent:
        examples_list.append(line)

    clean_data = []
    for r in range(len(examples_list)):
        example = examples_list[r]
        text = emoji.demojize(example)
        re_emoji = re.sub(':\S+?:', '', text)
        re_symbol = re_emoji.replace('@', '')
        re_hashtag = re_symbol.replace('#', '')
        words = re_hashtag.split(' ')
        if len(words) > 3:
            clean_data.append(re_hashtag)

    return clean_data


def getRandomSet(bits):
    """
        Generate a random ID with characteristic and number by given length
        Input: ID length
        Return: Random ID
    """
    num_set = [chr(i) for i in range(48,58)]
    char_set = [chr(i) for i in range(97,123)]
    total_set = num_set + char_set
    value_set = "".join(random.sample(total_set, bits))

    return value_set


def replace_brand(brands_path, data_path):
    """
    Replace the Brands name showing up in the dataset by random ID to avoid the classification bias
    Input: brand name list; chitchat dataset;
    Return: dataset(list)
    """
    with open(brands_path ,'r',encoding='utf-8') as file:
        brands_list = []
        filecontent = file.readlines()
        for line in filecontent:
            text = line.replace('- ', '')
            text = text.replace('\n', '')
            brands_list.append(text)

    with open(data_path ,'r',encoding='utf-8') as file:
        dataset = []
        filecontent = file.readlines()
        for line in filecontent:
            text = line.replace('- ', '')
            text = text.replace('\n', '')
            dataset.append(text)

    dataset_random_id = []

    for i in range(len(dataset)):
        example = dataset[i]

        for j in range(len(brands_list)):
            brand = brands_list[j]
            brand_upper = brand.upper()
            c = example.count(brand)  # if exist the brand name, c>=1
            c_upper = example.count(brand_upper)

            if c>0:
                random_len = random.randint(3,12)  # creat random id between 3-12 letter or number
                random_id = getRandomSet(random_len)
                example = example.replace(brand,random_id)

            if c_upper>0:
                random_len = random.randint(3,12)  # creat random id between 3-12 letter or number
                random_id = getRandomSet(random_len)
                example = example.replace(brand_upper,random_id)

        dataset_random_id.append(example)
    return dataset_random_id


def calculate_chitchat_length(data_path):
    """
    Calculate the length of data samples(chitchat)
    Input: chitchat(text file) data_path
    Return: a dict of length ranges with corresponding example number
    """

    with open(data_path, 'r', encoding='utf-8') as file:
        chitchat_train = []
        filecontent = file.readlines()
    for line in filecontent:
        text = line.replace('- ', '')
        text = text.replace('\n', '')
        chitchat_train.append(text)

    length_list = []
    for i in range(len(chitchat_train)):
        text_word = chitchat_train[i].split(' ')
        length = len(text_word)
        length_list.append(length)

    chitchat_dict = {}
    from itertools import groupby
    for k, g in groupby(sorted(length_list), key=lambda x: x // 10):
        chitchat_dict['{}-{}'.format(k * 10, (k + 1) * 10 - 1)] = len(list(g))

    return chitchat_dict


def calculate_faq_length(data_path):
    """
        Calculate the length of data samples(faq)
        Input: chitchat(yaml file) data_path
        Return: a dict of length ranges with corresponding example number
    """

    faq_train = []
    with open(data_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    for i in data['nlu']:
        for j in i['examples']:  # get the clean data from yaml
            input = j.replace('- ', '')
            if input not in faq_train:
                faq_train.append(input)

    length_list2 = []
    for i in range(len(faq_train)):
        text_word = faq_train[i].split(' ')
        length = len(text_word)
        length_list2.append(length)

    faq_dict = {}
    from itertools import groupby
    for k, g in groupby(sorted(length_list2), key=lambda x: x // 10):
        faq_dict['{}-{}'.format(k * 10, (k + 1) * 10 - 1)] = len(list(g))
    del faq_dict['70-79']
    del faq_dict['80-89']
    del faq_dict['90-99']
    return faq_dict


def plot_length_distribution(faq_dict,chitchat_dict):
    """
    Plot the length distribution of FAQ and Chitchat dataset
    Input: dict of FAQ and Chitchat
    Return: Distribution Plot
    """

    plt.figure(figsize=(10,8))
    y1=list(chitchat_dict.values())
    y2=list(faq_dict.values())

    name = list(chitchat_dict.keys())
    width = 0.4
    name = ['10','20','30','40','50','60','70']
    x1=list(range(len(name)))
    x2=list(range(len(y2)))

    for i in range(len(x1)):
        x1[i]=x1[i]+0.2

    plt.bar(x1,y1,width=width,label='Chit-chat', tick_label=name,fc='dodgerblue')

    for i in range(len(x2)):
        x2[i]=x2[i]-0.2

    plt.bar(x2,y2,width=width,label='FAQ',fc='darkorange')
    plt.xlabel('Length of Messages')
    plt.ylabel('Frequency')

    plt.legend()
    plt.savefig('length statistic.jpg')
    plt.show()


def count_words(sentence, top_n):
    """
        Count the top nth frequency of word in a sentence
        Input: sentence; nth number
        Return: A tuple of words with their corresponding frequency
  """
    sentence_list = sentence.lower().split(' ')
    top_n_dict = {}
    for word in sentence_list:
        if word in top_n_dict:
            top_n_dict[word] += 1
        else:
            top_n_dict[word] = 1

    word_frequency = []
    values = sorted(list(set(top_n_dict.values())), reverse=True)
    for w in values:
        word_list = []
        for k, v in top_n_dict.items():
            if v == w:
                word_list.append((k, v))
        word_frequency.extend(sorted(word_list))
    return word_frequency[:top_n]


def plot_frequency(faq_path, chitchat_path, top_n):
    """
    Plot the top nth frequency words
    :param faq_path:
    :param chitchat_path:
    :param top_n:
    :return:
    """
    faq_train = []
    with open(faq_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    for i in data['nlu']:
        for j in i['examples']:  # get the clean data from yaml
            input = j.replace('- ', '')
            if input not in faq_train:
                faq_train.append(input)

    faq_together = faq_train[0]
    for i in range(len(faq_train) - 1):
        faq_together = faq_together + ' ' + faq_train[i + 1]

    with open(chitchat_path, 'r', encoding='utf-8') as file:
        chitchat_train = []
        filecontent = file.readlines()
        for line in filecontent:
            text = line.replace('- ', '')
            text = text.replace('\n', '')
            chitchat_train.append(text)

    chitchat_together = chitchat_train[0]
    for i in range(len(chitchat_train) - 1):
        chitchat_together = chitchat_together + ' ' + chitchat_train[i + 1]

    faq_counts = count_words(faq_together, top_n)
    chitchat_counts = count_words(chitchat_together, top_n + 1)[1:]

    figure, ax = plt.subplots(2, 1, figsize=(12, 8))

    x1 = [tup[0] for tup in chitchat_counts]
    y1 = [tup[1] for tup in chitchat_counts]

    x2 = [tup[0] for tup in faq_counts]
    y2 = [tup[1] for tup in faq_counts]

    ax[0].bar(x1, y1, label='chitchat', fc='dodgerblue')
    ax[0].set_xticklabels(x1, rotation=90)
    ax[0].legend()
    ax[1].bar(x2, y2, label='faq', fc='darkorange')
    ax[1].set_xticklabels(x2, rotation=90)
    ax[1].legend()
    plt.savefig('faq_train_word_frequency2.jpg')
    plt.show()
