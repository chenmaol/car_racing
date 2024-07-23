# Car_Racing - IDM - Post Processing

## 数据结构
--data
    --chenmao
        --images
        --images.csv
        --keys.csv
    --quanhao
        --images
        --images.csv
        --keys.csv
    -- ...
    --labels_xxx.csv (运行后生成)
--post_processing
    --process.py

## 运行  
```
python process.py
```
1. 设置py中的interval来控制两个相邻图像相隔多少秒  
2. 设置py中的dirty_data_filter_time来控制，丢弃多少秒按下R之前的数据
ps:原本的record script每隔0.1秒录制一帧画面，但训练时不一定用到每一帧画面，因此处理后的相邻图像实际上相隔0.1 * interval秒
