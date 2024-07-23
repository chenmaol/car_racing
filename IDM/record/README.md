# Car_Racing - IDM - Record

## 游戏设置
1. Options -> Video -> Display  
· Windows Style: Full Screen  
· Resolution: 1280x800  （根据自己的屏幕比例选择，分辨率尽可能小且比例等于屏幕比例，同时修改record.py中第15行的分辨率为游戏设置的像素）
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
7. 录制完成后使用visual.py验证是否录制正确
```
python record.py
```

## IMPORTANT ！
1. 录制前需要先调整游戏视角（按c），调到能够看到车的全身的那个视角，可参考示例图像
2. 第一次录制时，点击B跑几秒后按P暂停，切出来看看数据保存是否正确  
· 在../data路径，会有：images文件夹，images.csv，keys.csv  
· images文件夹中会有若干图像，检查每个图像分辨率是否正确，是否包含了游戏所有的画面，是否录制到游戏外边框 
· images.csv中有三个col，分别是图像名称，对应时间，以及对应seq_id。确保相隔图像的时间差在0.1s左右，否则可能是因为电脑配置不够导致取图帧率不够  
· keys.csv中有三个col，分别是key名称，按下时间，松开时间  
3. 录制过程中尽量避免跑出赛道，撞击到障碍物等会导致画面突变的操作，开慢点没关系。
4. 录制过程中不要录制到开始和结束时无法操作的场景（菜单等），因为动作和画面不对应，用P暂停来避免
5. 如果不小心录入了脏数据：车子飞出赛道被动重置、车子卡在一个地方动不了、不小心录入了菜单等可能会导致动作和画面不一致的数据，长按两秒R。这可以帮助车子重置，同时我们也会记录下R的按键，在后处理中过滤数据，按下R就意味着丢弃掉按R前若干秒的数据！



