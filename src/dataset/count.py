def count_classes(dataset):
    'Count each type of class in the dataset.'
    # Example Usage:
    # print("Start Counting Class Frequencies")
    # print(count_classes(train_dataset))
    # print(count_classes(val_dataset))
    # print(count_classes(test_dataset))
    class0 = 0
    class1 = 0
    class2 = 0
    class3 = 0
    for data in dataset:
        y = data['label']
        if y[0] == 1:
            class0 += 1
        if y[1] == 1:
            class1 += 1
        if y[2] == 1:
            class2 += 1
        if y[3] == 1:
            class3 += 1
    return class0, class1, class2, class3

def count_labels(dataset):
    'Count each type of label in the dataset.'
    type0 = (1., 0., 1., 0.)
    type1 = (1., 0., 0., 1.)
    type2 = (0., 1., 1., 0.)
    type3 = (0., 1., 0., 1.)
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    for data in dataset:
        y = data['label']
        if y[0] == 1:
            if y[2] == 1:
                count0 += 1
            else:
                count1 += 1
        else:
            if y[2] == 1:
                count2 += 1
            else:
                count3 += 1
    return {type0: count0, type1: count1, type2: count2, type3: count3}
