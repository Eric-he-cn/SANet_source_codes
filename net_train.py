import numpy as np
import tensorflow as tf
from scipy.io import savemat
import helper as hp
import math
"""
net_train.py
============
Training and evaluation routines for SANet experiments (TensorFlow 1.x graph mode).
This file mainly contains: channel generation/sampling, loss construction, training loops,
and saving results (.mat).

Main functions
--------------
Channel Estimation (CE)
- DNN_SAT_CE(...):
    Baseline CE training/evaluation using helper.DNN_CE.
- HyperDop_SAT_CE(...):   [FINAL for CE]
    Sensing-aided CE training/evaluation using helper.HyperDop_CE. Hyper-input encodes
    Doppler/velocity-related features.
"""
def DNN_SAT_CE(M=64, B=25, Lp=2, L=2, batch_size=1024, mini_batch=1,total_epoch=10000):
    # M transmit annennas
    # K=1 number of users
    # B bits for each user;
    # L pilot length
    # Lp number of paths

    np.random.seed(12)
    K = 1
    SNRdb = 10
    Mx = np.sqrt(M).astype(int)
    My = np.sqrt(M).astype(int)
    file_name = 'DNN_SAT_CE B (' + str(B) + ') L(' + str(L) + ') Lp(' + str(Lp) + ')'
    print(file_name)

    #构建神经网络
    input_number = M            #输入特征数
    output_number = M * 2       #输出特征数
    tf.reset_default_graph()    #创建新的图
    X = tf.placeholder("complex64", [batch_size, input_number])         #X为真实的信道信息h（复数形式）
    y_true = tf.placeholder("float32", [batch_size, output_number])     #y_true为真实的信道信息h（实数形式）
    alpha_para = tf.placeholder("float32", [])                          #神经网络的激活函数tanh的一个参数
    N0_dnn = tf.placeholder("float32", [])

    #调用函数hp.DNN_CE，输出为y_pred, test1
    y_pred, test1 = hp.DNN_CE(X, N0_dnn, batch_size, M, B, L, alpha_para)  # B* M1

    cross = tf.reduce_mean(tf.square(tf.abs(y_pred - y_true)))/tf.reduce_mean(tf.square(y_true))    #计算均一化均方误差NMSE

    learning_rate = tf.placeholder(tf.float32, shape=[])                            #学习率
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross) #优化器为adam
    init = tf.global_variables_initializer()                                        #初始化全局变量

    avg_cost = 0.
    cost_array = np.zeros((int(total_epoch / 10), 1), dtype=float)            #存储每隔10个周期的损失值
    cost_array_dB = np.zeros((int(total_epoch / 10), 1), dtype=float)
    cost_index = 0

    with tf.Session() as sess:                              #创建了一个 TensorFlow 会话（Session），会话是执行计算图的环境
        # Training
        sess.run(init)                                      #使用之前定义的init变量初始化sess
        for epoch in range(total_epoch):                 #开始循环

            if epoch < total_epoch / 3:                 #动态调整学习率
                learning_R = 0.001
            elif epoch < (3.0 * total_epoch / 4):
                learning_R = 0.0001
            else:
                learning_R = 0.00001

            sigma2 = 1 / (10 ** (SNRdb / 10.0))                            #sigma2:噪声方差σ^2
            N0_input = sigma2  # 噪声输入
            alpha_para_input = 1.0                      #α是固定值

            for index_m in range(mini_batch):           #循环mini_batch次
                gain_dl_img = np.random.normal(loc=0.707, scale=2, size=(batch_size, K, Lp))
                gain_dl_real = np.random.normal(loc=0.707, scale=2, size=(batch_size, K, Lp))
                gain_dl = gain_dl_real + 1j * gain_dl_img   #初始化路径增益                    #[batch_size, K, Lp]

                KK = 10     #莱斯因子定为10dB
                f_dop = 3000 + 4000 * np.random.rand(batch_size, K)                         #多普勒频移:3000~7000Hz
                tau = (10.0 * np.random.rand(batch_size, K) + 40.0) * 0.001             #40ms~50ms
                theta = (4.0 * np.pi / 9) * np.random.rand(batch_size, K) + np.pi / 18  # π/18 ~ π/2
                phi = 2 * np.pi * np.random.rand(batch_size, K)                         #0~2π
                h = np.zeros([batch_size, M * 2], dtype=float)      #信道（实数形式）
                h_complex = np.zeros([batch_size, M], dtype=complex)      #信道（复数形式）
                arv_x = np.zeros([Mx], dtype=complex)       #array response vector_x
                arv_y = np.zeros([My], dtype=complex)       #array response vector_y
                for size in range(batch_size):
                    # 计算array response vector
                    for mx in range(Mx):
                        arv_x[mx] = np.exp(-1j * np.pi * mx * np.cos(theta[size, 0]) * np.sin(phi[size, 0]))
                    for my in range(My):
                        arv_y[my] = np.exp(-1j * np.pi * my * np.cos(theta[size, 0]) * np.sin(phi[size, 0]))
                    arv = np.kron(arv_x , arv_y)

                    h_los = np.exp(1j * 2 * np.pi * (f_dop[size, 0] * 10000 - 2 * 1e9 * tau[size, 0])) * arv
                    h_nlos = np.zeros([M], dtype=complex)
                    for p in range(Lp):
                        h_nlos = h_nlos + np.sqrt(1/Lp) * gain_dl[size, 0, p] \
                            * np.exp(1j * 2 * np.pi * (f_dop[size, 0] * 10000 - 2 * 1e9 * tau[size, 0])) * arv

                    h_complex[size,:]= np.sqrt(KK/(KK+1)) * h_los + np.sqrt(1/(KK+1)) * h_nlos    #计算信道h
                h[:, 0:M] = np.real(h_complex)        #x_h中的实部部分赋值给h的前M列
                h[:, M:2 * M] = np.imag(h_complex)    #x_h中的虚部部分赋值给h的后M列（其实就是复数x_h转成实数h）



                #### sess.run()意味执行了一次神经网络的训练迭代，通过反向传播和优化器更新模型参数，同时计算了损失和测试结果
                _, cs, test11 = sess.run([optimizer, cross, test1],
                                         feed_dict={X: h_complex, y_true: h, N0_dnn: N0_input, alpha_para: alpha_para_input,
                                                    learning_rate: learning_R})
                avg_cost += cs

            if (epoch + 1) % 10 == 0:           #每10个周期打印NMSE
                # print(test11)
                print("Epoch:", '%04d' % (epoch + 1), "train_cost=", "{:.9f}".format(avg_cost / 10),
                      "  train_cost_dB=","{:.3f}".format(-10 * math.log10(avg_cost/10)))

                cost_array[cost_index, 0] = avg_cost / 10
                cost_array_dB[cost_index, 0] = 10 * math.log10(cost_array[cost_index, 0])
                cost_index += 1
                avg_cost = 0.

            if (epoch + 1) % 1000 == 0:         #每1000个周期打印信息并保存数据
                store_dic = {'cost_array': cost_array,'cost_array_dB': cost_array_dB}
                savemat(file_name + ".mat", store_dic)  # 保存损失数组





def HyperDop_SAT_CE( M=64, B=25, Lp=2, L=2,time_doppler_opt=1 , batch_size=1024, S=4 , mini_batch=1, total_epoch=20000):
    # M transmit annennas
    # K number of users
    # B bits for each user;
    # L pilot length
    # Lp number of paths

    np.random.seed(0)
    SNRdb = 10
    K = 1
    delta = 100
    file_name = 'HyperDop_SAT_CE  B(' + str(B) + ') L(' + str(L) + ') Lp(' + str(Lp) + ') S(' + str(S) + ')'
    print(file_name)

    tf.reset_default_graph()
    X1 = tf.placeholder("complex64", [batch_size, K, S, M])         #y_real 复数形式
    hyper_input = tf.placeholder("float32", [batch_size, K, 1])     #超网络输入
    y_true = tf.placeholder("complex64", [batch_size, K, S, M])     #y_real 复数形式
    N0_dnn = tf.placeholder("float32", [])
    y_pred = hp.HyperDop_CE(X1, K, N0_dnn, batch_size, M, B, L, S, Lp, hyper_input)  # B* M1

    cross = tf.reduce_mean(tf.square(tf.abs(y_true - y_pred)))  /  tf.reduce_mean(tf.square(tf.abs(y_true)))    #计算均一化均方误差NMSE

    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross)
    init = tf.global_variables_initializer()

    avg_cost = 0.
    cost_array = np.zeros((int(total_epoch / 10), 1), dtype=float)
    cost_array_dB = np.zeros((int(total_epoch / 10), 1), dtype=float)
    cost_index = 0

    if time_doppler_opt==1:#0.5e-4
        dl_cor=0.9998
    if time_doppler_opt==2:#0.5e-3
        dl_cor=0.9829
    if time_doppler_opt==3:#0.8e-3
        dl_cor=0.9566
    if time_doppler_opt==4:#1e-3
        dl_cor=0.9326
    if time_doppler_opt==5:#0.2e-2
        dl_cor=0.7441
    if time_doppler_opt==6:#0.3e-2
        dl_cor=0.4720
    if time_doppler_opt==7:#0.4e-2
        dl_cor=0.1698

    with tf.Session() as sess:
        # Training
        sess.run(init)
        for epoch in range(total_epoch):

            if epoch < total_epoch / 3:
                learning_R = 0.001
            elif epoch < (3.0 * total_epoch / 4):
                learning_R = 0.0001
            else:
                learning_R = 0.00001
            SNR = 10 ** (SNRdb / 10.0)
            sigma2 = 1 / SNR
            for index_m in range(mini_batch):
                ##初始化路径增益
                gain_dl0 = np.sqrt(0.5)*(np.random.standard_normal([batch_size,K,Lp])+1j*np.random.standard_normal([batch_size,K,Lp]))
                gain_dl = np.sqrt(0.5)*(np.random.standard_normal([batch_size,K,S,Lp]) + 1j*np.random.standard_normal([batch_size,K,S,Lp]))
                gain_dl[:,:,0,:] = gain_dl0
                for s in range(S):
                    if s > 0:
                        noise = np.sqrt(0.5) * (np.random.standard_normal([batch_size,K,Lp]) + 1j * np.random.standard_normal(
                                    [batch_size,K,Lp]))
                        gain_dl[:,:,s,:] = gain_dl[:,:,s-1,:] * dl_cor + noise * np.sqrt(1 - np.square(dl_cor))



                Mx = np.sqrt(M).astype(int)     #X方向的的天线数
                My = np.sqrt(M).astype(int)     #Y方向的的天线数
                #delta_time = 0.4 * np.random.rand(batch_size, K) + 0.1  #上下行信道的时间延迟
                fc_dl = 2 * 1e9   #下行信道载频
                KK = 10                     #莱斯因子10dB

                f_dop2 = 60000 + 80000 * np.random.rand(batch_size, K)                         #多普勒频移:60k~140kHz
                #f_dop2 = np.expand_dims(f_dop,0).repeat(batch_size,axis=0)
                v_sat = np.zeros([batch_size, K, 1], dtype=float)
                v_sat[:, :, 0] = f_dop2 * 3e8 / (2 * fc_dl)                     #速度：4.5k~10.5km/s
                v_sat_input = v_sat/20-10


                tau = (10.0 * np.random.rand(batch_size, K) + 40.0) * 0.001     #40ms~50ms
                theta = (4 * np.pi / 9) * np.random.rand(batch_size, K) + np.pi / 18    #18/π ~ 2/π
                phi = 2 * np.pi * np.random.rand(batch_size, K)                 # 0~2π

                h_los = np.zeros([batch_size, K, S, M], dtype=complex)
                h_nlos = np.zeros([batch_size, K, S, M], dtype=complex)
                h = np.zeros([batch_size, K, S, M], dtype=complex)

                arv_x = np.zeros([Mx], dtype=complex)   #array response vector_x
                arv_y = np.zeros([My], dtype=complex)   #array response vector_y
                for size in range(batch_size):
                    for k in range(K):
                        for mx in range(Mx):
                            arv_x[mx] = np.exp(-1j * np.pi * mx * np.cos(theta[size, k]) * np.sin(phi[size, k]))
                        for my in range(My):
                            arv_y[my] = np.exp(-1j * np.pi * my * np.cos(theta[size, k]) * np.sin(phi[size, k]))
                        arv = np.kron(arv_x, arv_y)     #array response vector
                        for s in range(S):
                            h_los[size, k , s, :] = np.exp(1j * 2 * np.pi *(f_dop2[size, k] * 10000 - fc_dl * tau[size, k]))* arv
                            for p in range(Lp):
                                h_nlos[size, k, s, :] = h_nlos[size, k, s, :] + np.sqrt(1/Lp) * gain_dl[size, k, s, p] * np.exp(
                                    1j * 2 * np.pi * (f_dop2[size, k] * 10000 - fc_dl * tau[size, k])) * arv
                            h[size, k, s, :] = np.sqrt(KK / (KK + 1)) * h_los[size, k, s, :] + np.sqrt(
                                1 / (KK + 1)) * h_nlos[size, k, s, :]
                N0_input = sigma2

                _, cs = sess.run([optimizer, cross], feed_dict={X1: h, y_true: h,  hyper_input: (v_sat_input), N0_dnn: N0_input,
                                                                learning_rate: learning_R})
                avg_cost += cs


            if (epoch + 1) % 10 == 0:           #每10个周期打印NMSE
                print("Epoch:",'%04d' % (epoch+1), "  train_cost=","{:.9f}".format(avg_cost/10),
                      "  train_cost_dB=","{:.3f}".format(-10 * math.log10(avg_cost/10)))

                cost_array[cost_index, 0] = avg_cost / 10
                cost_array_dB[cost_index, 0] = 10 * math.log10(cost_array[cost_index, 0])
                cost_index += 1
                avg_cost = 0.


            if (epoch + 1) % 200 == 0:         #每200个周期打印信息并保存数据
                store_dic = {'cost_array': cost_array,'cost_array_dB':cost_array_dB}
                savemat(file_name + ".mat", store_dic)  # 保存损失数组










