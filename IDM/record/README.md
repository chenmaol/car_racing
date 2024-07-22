# Car_Racing - IDM - Record

## 游戏设置
1. Options -> Video -> Display  
· Windows Style: Full Screen  
· Resolution: 1280x800  
2. Options -> Gameplay Setting -> Configure  
· Damage Effect: Visual only  
· Broken down vehicles on the roadside: off  
· Display Ghost: off  
3. Options -> Gameplay Setting -> HUD  
· All: on  

## 环境设置  
```
pip install -r requirements.txt
```

## 录制  
```
python record.py
```
0. 运行record代码
1. 游戏主界面，点击QuickPlay
2. 选择sheet上对应国家的地图
3. 按ctrl随机选择后续选项（天气如果选择到了Night，重新随机）
4. 进入到赛道，并可以自由控制车后，按“B”开始录制（避免出现菜单画面）
5. 要到终点时，按“P”停止录制（避免出现菜单画面）
6. 切换到下一个赛道时，按“B”开始新的录制

## IMPORTANT ！
1. 第一次录制时，点击B跑几秒后按P暂停，切出来看看数据保存是否正确  
· 在../data路径，会有：images文件夹，images.csv，keys.csv  
· images文件夹中会有若干图像，检查每个图像分辨率是否正确，是否包含了游戏所有的画面  
· images.csv中有三个col，分别是图像名称，对应时间，以及对应seq_id。确保相隔图像的时间差在0.1s左右，否则可能是因为电脑配置不够导致取图帧率不够  
· keys.csv中有三个col，分别是key名称，按下时间，松开时间  
2. 录制过程中尽量避免跑出赛道，撞击到障碍物等会导致画面突变的操作，开慢点没关系。
3. 录制过程中不要录制到开始和结束时无法操作的场景（菜单等），因为动作和画面不对应，用P暂停来避免



