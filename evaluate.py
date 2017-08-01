__author__ = 'PC-LiNing'

import numpy as np
import multiprocessing as mp
from  multiprocessing import Pool


def unit(h, t, r, entity_embedding):
    k = 10
    entity_size = 14951
    # scores = [entity_size,]
    h_scores = np.sum((entity_embedding + r -t)**2, axis=1)
    t_scores = np.sum((h + r - entity_embedding) ** 2, axis=1)
    my_score = np.sum((h + r - t) ** 2)
    h_flag = 0.0
    t_flag = 0.0
    # get rank
    h_sort_scores = np.sort(h_scores)
    h_rank = np.argwhere(h_sort_scores > my_score).squeeze(axis=1).tolist()
    h_rank = (h_rank[0] + 1) if len(h_rank) > 0 else entity_size
    t_sort_scores = np.sort(t_scores)
    t_rank = np.argwhere(t_sort_scores > my_score).squeeze(axis=1).tolist()
    t_rank = (t_rank[0] + 1) if len(t_rank) > 0 else entity_size
    if h_rank <= k:
        h_flag = 1.0
    if t_rank <= k:
        t_flag = 1.0
    return (h_flag, h_rank, t_flag, t_rank)


def compute_acc(h_embed, t_embed, r_embed, entity_embedding):
    Test_size = 5000
    pool = Pool(mp.cpu_count())
    res_list = []
    for i in range(Test_size):
        res_list.append(pool.apply_async(func=unit, args=(h_embed[i],t_embed[i],r_embed[i],entity_embedding,)))
    pool.close()
    pool.join()
    h_flag = []
    h_rank = []
    t_flag = []
    t_rank = []
    for res in res_list:
        result = res.get()
        h_flag.append(result[0])
        h_rank.append(result[1])
        t_flag.append(result[2])
        t_rank.append(result[3])

    # compute acc
    h_acc = float(100*sum(h_flag)/Test_size)
    t_acc = float(100*sum(t_flag)/Test_size)
    # compute mean-rank
    h_mean_rank = int(sum(h_rank)/Test_size)
    t_mean_rank = int(sum(t_rank)/Test_size)
    return h_acc, t_acc, h_mean_rank, t_mean_rank

"""
if __name__ == '__main__':
    Test_size = 5000
    entity_size = 14951
    embedding_size = 50
    h_embed = np.random.uniform(-1.0, 1.0, size=(Test_size,embedding_size))
    t_embed = np.random.uniform(-1.0, 1.0, size=(Test_size,embedding_size))
    r_embed = np.random.uniform(-1.0, 1.0, size=(Test_size,embedding_size))
    entity_embed = np.random.uniform(-1.0, 1.0, size=(entity_size,embedding_size))
    import datetime
    time_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(time_start)
    h_acc, t_acc, h_mean_rank, t_mean_rank = compute_acc(h_embed, t_embed, r_embed, entity_embed)
    print(h_acc)
    print(t_acc)
    print(h_mean_rank)
    print(t_mean_rank)
    time_end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(time_end)
"""

