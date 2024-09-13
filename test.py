import time
starttime = time.strftime("%Y-%m-%d_%H:%M:%S")#时间格式可以自定义，如果需要定义到分钟记得改下冒号，否则输入logdir时候会出问题
print("Start experiment:", starttime)#定义实验时间
writer = SummaryWriter(log_dir="./log/",comment=starttime[:13],flush_secs=60)