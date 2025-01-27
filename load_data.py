__author__ = 'PC-LiNing'

import codecs
from gensim import corpora
import numpy
import random


def generate_dic():
    entitys = []
    relations = []
    entity_ids = codecs.open('data/FB15K/entity2id.txt', 'r', encoding='utf-8')
    relation_ids = codecs.open('data/FB15K/relation2id.txt', 'r', encoding='utf-8')
    for line in entity_ids.readlines():
        line = line.strip('\n').strip()
        temp = line.split('\t')
        entitys.append([temp[0]])
    print(len(entitys))
    dic_entitys = corpora.Dictionary(entitys)
    dic_entitys.save('entitys.dict')
    for line in relation_ids.readlines():
        line = line.strip('\n').strip()
        temp = line.split('\t')
        relations.append([temp[0]])
    print(len(relations))
    dic_relations = corpora.Dictionary(relations)
    dic_relations.save('relations.dict')


def invert_dict(d):
    return dict([(v,k) for k,v in d.items()])


def load_train_test():
    entity_dic = invert_dict(corpora.Dictionary.load('entitys.dict'))
    relation_dic = invert_dict(corpora.Dictionary.load('relations.dict'))
    train_size = 483142
    test_size = 59071
    train_data = numpy.zeros(shape=(train_size, 3), dtype=numpy.int32)
    test_data = numpy.zeros(shape=(test_size, 3), dtype=numpy.int32)
    train_f = codecs.open('data/FB15K/train.txt', 'r', encoding='utf-8')
    test_f = codecs.open('data/FB15K/test.txt', 'r', encoding='utf-8')

    count = 0
    for line in train_f.readlines():
        line = line.strip('\n').strip()
        example = line.split('\t')
        example_id = [entity_dic[example[0]], entity_dic[example[1]], relation_dic[example[2]]]
        train_data[count] = numpy.asarray(example_id, dtype=numpy.int32)
        count += 1

    count = 0
    for line in test_f.readlines():
        line = line.strip('\n').strip()
        example = line.split('\t')
        example_id = [entity_dic[example[0]], entity_dic[example[1]], relation_dic[example[2]]]
        test_data[count] = numpy.asarray(example_id, dtype=numpy.int32)
        count += 1
    return test_data, train_data


# train_data = [batch_size,3]
# return [batch_size,2*(entity_size-1),3] or [batch_size,2*num_sample,3]
def generate_neg_data(train_data, num_sample=None):
    entity_size = 14951
    n,m = train_data.shape
    if num_sample is None:
        neg_train_data = numpy.zeros(shape=(n, 2*(entity_size-1), 3), dtype=numpy.int32)
        for k in range(n):
            example = train_data[k].copy()
            example2 = train_data[k].copy()
            neg_example = numpy.zeros(shape=(2*(entity_size-1), 3), dtype=numpy.int32)
            j = 0
            # h
            h_id = example[0]
            for i in range(entity_size):
                if i != h_id:
                    example[0] = i
                    neg_example[j] = example
                    j += 1
            # t
            t_id = example2[1]
            for i in range(entity_size):
                if i != t_id:
                    example2[1] = i
                    neg_example[j] = example2
                    j += 1

            neg_train_data[k] = neg_example
            k += 1
        return neg_train_data
    else:
        neg_train_data = numpy.zeros(shape=(n, 2*num_sample, 3), dtype=numpy.int32)
        for k in range(n):
            example = train_data[k].copy()
            example2 = train_data[k].copy()
            neg_example = numpy.zeros(shape=(2*num_sample, 3), dtype=numpy.int32)
            # h
            h_id = example[0]
            t_id = example[1]
            ids = list(range(entity_size))
            if h_id in ids:
                ids.remove(h_id)
            if t_id in ids:
                ids.remove(t_id)
            shuffle_indices = numpy.random.permutation(numpy.asarray(ids, dtype=numpy.int32))
            sample_ids = shuffle_indices[:num_sample].tolist()
            j = 0
            for item in sample_ids:
                example[0] = item
                neg_example[j] = example
                j += 1
            # t
            shuffle_2 = numpy.random.permutation(numpy.asarray(ids, dtype=numpy.int32))
            sample_ids_2 = shuffle_2[:num_sample].tolist()
            for item in sample_ids_2:
                example2[1] = item
                neg_example[j] = example2
                j += 1
            neg_train_data[k] = neg_example
            k += 1
        return neg_train_data
