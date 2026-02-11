from net_train import DNN_SAT_CE, HyperDop_SAT_CE

###################################信道估计########################################
def CE_test():
    DNN_SAT_CE(64,25,4,4,1024,1,5000)
    HyperDop_SAT_CE(64,25,4,4,1,1024,4,1,5000)




