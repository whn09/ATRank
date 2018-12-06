import os
import random
import pickle
import numpy as np
import pandas as pd

random.seed(1234)

DIR = '../../dataset/log_preprocess/20181128/'
CACHE_DIR = '../../algo_recommend_server/script/cache_ph/'


def load_train_muid_and_content_id(muids_map_file, content_ids_map_file):
    index_muid_map = {}
    muid_index_map = {}
    with open(muids_map_file, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            params = line.strip().split('\t')
            index_muid_map[int(params[1])] = params[0]
            muid_index_map[params[0]] = int(params[1])
    index_content_id_map = {}
    content_id_index_map = {}
    with open(content_ids_map_file, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            params = line.strip().split('\t')
            index_content_id_map[int(params[1])] = params[0]
            content_id_index_map[params[0]] = int(params[1])
    return index_muid_map, muid_index_map, index_content_id_map, content_id_index_map


def get_tags(item_portraits):
    tags = {}
    train_tags_list = [[] for _ in range(len(item_portraits))]
    for content_id, item_portrait in item_portraits.items():
        # print('content_id:', content_id)
        # print('item_portrait:', item_portrait)
        try:
            tag_portrait = item_portrait['tag_portrait']
        except:
            continue
        content_tags = []
        for tag, tscore in tag_portrait.items():
            content_tags.append(str(tag))
            if tag not in tags:
                tags[tag] = len(tags)
        train_tags_list.append(content_tags)
    print('tags:', len(tags))
    print('train_tags_list:', len(train_tags_list))
    return tags, train_tags_list


def _parse(data):
    for line in data:
        if not line:
            continue
        uid, iid, rating, timestamp = [int(x) for x in line.split('\t')]
        yield uid, iid, rating, timestamp


def _build_pd(rows, cols, data):
    uids = []
    iids = []
    ratings = []
    timestamps = []
    for uid, iid, rating, timestamp in data:
        uids.append(uid)
        iids.append(iid)
        ratings.append(rating)
        timestamps.append(timestamp)
    mat = pd.DataFrame({'reviewerID': uids, 'asin': iids, 'unixReviewTime': timestamps})
    return mat


def prepare_data(dir, cache_dir):
    with open(dir + 'train_ph.txt', 'r') as fin:
        # muid content_id rating timestamp
        train_raw = fin.readlines()
    with open(dir + 'test_ph.txt', 'r') as fin:
        # muid content_id rating timestamp
        test_raw = fin.readlines()

    index_muid_map, muid_index_map, index_content_id_map, content_id_index_map = load_train_muid_and_content_id(
        dir + 'train_muids_map.txt', dir + 'train_content_ids_map.txt')

    num_users, num_items = len(index_muid_map), len(index_content_id_map)

    # Load train interactions
    train = _build_pd(num_users, num_items, _parse(train_raw))
    # Load test interactions
    test = _build_pd(num_users, num_items, _parse(test_raw))

    cache_path = cache_dir + 'item_portraits.pik'
    print('cache_path:', cache_path, 'file_exists:', os.path.exists(cache_path))
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            item_portraits = pickle.load(data_f)
    else:
        print('ERROR!', 'cache_path:', cache_path)
        exit(-1)
    print('item_portraits:', len(item_portraits))

    tags, train_tags_list = get_tags(item_portraits)

    item_features = np.zeros(shape=(num_items,))
    for content_id_index, content_id in index_content_id_map.items():
        if content_id in item_portraits:
            item_portrait = item_portraits[content_id]
            tag_portrait = item_portrait['tag_portrait']
            item_features[content_id_index] = tags[list(tag_portrait.keys())[0]]
            # item_features[content_id_index] = []
            # for tag, tscore in tag_portrait.items():
            #     item_features[content_id_index].append(str(tags[tag]) + ':' + str(tscore))
        else:
            item_features[content_id_index] = tags['']
    item_features = item_features.tolist()

    data = {'train': train,
            'test': test,
            'item_features': item_features,
            'tags': tags,
            'user_count': num_users,
            'item_count': num_items,
            'cate_count': len(tags),
            'example_count': train.shape[0]}

    return data


if __name__ == '__main__':
    data = prepare_data(DIR, CACHE_DIR)
    print('data:', data)

    reviews_df = data['train']
    cate_list = data['item_features']
    user_count = data['user_count']
    item_count = data['item_count']
    cate_count = data['cate_count']
    example_count = data['example_count']

    with open('../raw_data/funny_remap.pkl', 'wb') as f:
        pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL) # uid, iid
        pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL) # cid of iid line
        pickle.dump((user_count, item_count, cate_count, example_count), f, pickle.HIGHEST_PROTOCOL)
        # pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)
