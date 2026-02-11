from net_train2 import DNN_SAT_BF, HyperDop_SAT_BF, ZF_BF2, ZF_BF

###################################波束赋形########################################
def BF_test():
    DNN_SAT_BF(64, 15, 2, 2, 2, 1024, 1, 5000)
    HyperDop_SAT_BF(64, 15, 2, 2, 1, 2, 1024, 6, 1, 5000)


def ZF_test():
    ZF_BF2(64, 15, 2, 2, 2, 1024, 1, 5000)





