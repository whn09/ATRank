import random
import pickle
import numpy as np

random.seed(1234)

with open('../raw_data/funny_remap.pkl', 'rb') as f:
    # reviews_df = pickle.load(f)
    train_reviews_df = pickle.load(f)
    test_reviews_df = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count, example_count = pickle.load(f)
    # print('reviews_df:', reviews_df.shape, reviews_df.head())
    print('train_reviews_df:', train_reviews_df.shape, train_reviews_df.head())
    print('test_reviews_df:', test_reviews_df.shape, test_reviews_df.head())
    print('cate_list:', len(cate_list))
    print('user_count, item_count, cate_count, example_count:', user_count, item_count, cate_count, example_count)
    # asin_key, cate_key, revi_key = pickle.load(f)
    # print('asin_key, cate_key, revi_key:', len(asin_key), len(cate_key), len(revi_key))

# [1, 2) = 0, [2, 4) = 1, [4, 8) = 2, [8, 16) = 3...    need len(gap) hot
gap = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
# gap = [2, 7, 15, 30, 60,]
# gap.extend( range(90, 4000, 200) )
# gap = np.array(gap)
print(gap.shape[0])


def proc_time_emb(hist_t, cur_t):
    print('cur_t:', cur_t)
    print('hist_t0:', hist_t)
    hist_t = [cur_t - i + 1 for i in hist_t]
    print('hist_t1:', hist_t)
    hist_t = [np.sum(i >= gap) for i in hist_t]
    print('hist_t2:', hist_t)
    return hist_t


def generate_dataset(df):
    dataset = []
    cnt = 0
    user_df = df.groupby('reviewerID')
    for reviewerID, hist in user_df:
        if cnt % 1000 == 0:
            print('cnt:', cnt)
        cnt += 1
        # print('hist:', hist)
        pos_list = hist[hist.rating == 2]['asin'].tolist()
        neg_list = hist[hist.rating == 1]['asin'].tolist()
        print('pos_list:', len(pos_list), pos_list[:5])
        print('neg_list:', len(neg_list), neg_list[:5])
        tim_list = hist['unixReviewTime'].tolist()
        # print('tim_list1:', len(tim_list), tim_list[:5])
        tim_list = [i // 3600 // 24 for i in tim_list]
        # print('tim_list2:', len(tim_list), tim_list[:5])

        for i in range(0, len(pos_list)):  # TODO whn?
            hist_i = pos_list[:i+1]
            hist_t = proc_time_emb(tim_list[:i+1], tim_list[i])
            # print('hist_i:', hist_i, 'hist_t:', hist_t)
            dataset.append((reviewerID, hist_i, hist_t, pos_list[i], 1))

        for i in range(0, len(neg_list)):  # TODO whn?
            hist_i = neg_list[:i+1]
            hist_t = proc_time_emb(tim_list[:i+1], tim_list[i])
            # print('hist_i:', hist_i, 'hist_t:', hist_t)
            dataset.append((reviewerID, hist_i, hist_t, neg_list[i], 0))
    return dataset


train_set = generate_dataset(train_reviews_df)
print('train_set:', len(train_set), train_set[:5])
test_set = generate_dataset(test_reviews_df)
print('test_set:', len(test_set), test_set[:5])

random.shuffle(train_set)
random.shuffle(test_set)

# assert len(test_set) == user_count
# assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])

with open('funny_dataset.pkl', 'wb') as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
