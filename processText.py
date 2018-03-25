import random

alltrain = open('alltraindata_mutil.txt', 'r')
val = open('val_mutil.txt', 'w')
train = open('train_mutil.txt', 'w')

list = []
for line in alltrain:
    list.append(line)
    print(line)
random.shuffle(list)

for count in range(len(list)):
    # print(list[count])
    if(count<len(list)*0.8):
        train.write(list[count])
    else:
        val.write(list[count])



alltrain.close()
train.close()
val.close()