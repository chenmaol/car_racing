import pandas as pd
import os
pd.options.mode.chained_assignment = None  # default='warn'


def split_seq(df_image_single_seq, seq_id):
    last_row = None
    for index, row in df_image_single_seq.iterrows():
        if index >= len(df_image):
            break
        if index % interval != 0:
            continue

        key_pressing_time = {}
        for key in target_keys:
            key_pressing_time[key] = 0.0
        frame_name = row['frame_name'].strip()
        timestamp = row['record_time']

        last_timestamp = timestamp - interval / fps if last_row is None else last_row['record_time']

        filtered_df = df_key[(df_key['start_time'] < timestamp) & (df_key['end_time'] > last_timestamp)]

        for _, key_row in filtered_df.iterrows():
            if timestamp == last_timestamp:
                print(index)
            t_ratio = (min(timestamp, key_row['end_time']) - max(last_timestamp, key_row['start_time'])) / (
                        timestamp - last_timestamp)
            key = key_row['key_name']
            key_pressing_time[key] = t_ratio

        df_image_single_seq.loc[index, 'w'] = key_pressing_time['w'] > 0.5
        df_image_single_seq.loc[index, 's'] = key_pressing_time['s'] > 0.5
        df_image_single_seq.loc[index, 'a'] = key_pressing_time['a'] > 0.5
        df_image_single_seq.loc[index, 'd'] = key_pressing_time['d'] > 0.5
        df_image_single_seq.loc[index, 'if_dirty'] = key_pressing_time['r'] > 0

        if df_image_single_seq.loc[index, 'if_dirty'] is True:
            start_index = max(0, index - int(fps * dirty_data_filter_time))
            df_image_single_seq.loc[start_index:index, 'if_dirty'] = True
        last_row = row
    df_image_single_seq = df_image_single_seq.dropna()
    df_image_single_seq['if_dirty'] = df_image_single_seq['if_dirty'].astype(bool)
    df_image_single_seq_clean = df_image_single_seq[~df_image_single_seq['if_dirty']]

    last_index = None
    for index, row in df_image_single_seq_clean.iterrows():
        if last_index is not None and index - last_index != interval:
            seq_id += 1
        df_image_single_seq_clean.loc[index, 'seq'] = seq_id
        last_index = index
    del df_image_single_seq_clean['record_time']
    del df_image_single_seq_clean['if_dirty']
    return df_image_single_seq_clean, seq_id + 1


def extend_path(name):
    return os.path.join(image_save_path, name)


if __name__ == '__main__':
    # 每隔多少张图像取一张图作为训练数据
    interval = 1
    # 图像录制间隔
    fps = 10
    # 脏数据过滤追溯时间 / 秒
    dirty_data_filter_time = 5.0
    data_path = "../../IDM/data"
    save_path = "../data"
    target_keys = ['w', 'a', 's', 'd', 'r']

    sub_folders = os.listdir(data_path)
    seq_count_total = 0
    df_processed_total = []
    for sub_folder in sub_folders:
        if sub_folder == 'images':
            continue
        if not os.path.isdir(os.path.join(data_path, sub_folder)):
            continue
        print(sub_folder)
        image_save_path = os.path.join(data_path, sub_folder, 'images')
        # 读取CSV文件
        df_image = pd.read_csv(os.path.join(data_path, sub_folder, "images.csv"))
        df_key = pd.read_csv(os.path.join(data_path, sub_folder, "keys.csv"))
        df_image['frame_name'] = df_image['frame_name'].apply(extend_path)
        seq_count = df_image['seq'].iloc[-1]

        for seq_id in range(seq_count + 1):
            if len(df_image[df_image['seq'] == seq_id]) == 0:
                continue
            df_processed, seq_count_total = split_seq(df_image[df_image['seq'] == seq_id].copy(), seq_count_total)
            df_processed_total.append(df_processed)

    df_all = pd.concat(df_processed_total)

    out_path = os.path.join(save_path, 'labels_interval-{}_dirty-{}.csv'.format(interval, dirty_data_filter_time))
    df_all.to_csv(out_path, index=False)

