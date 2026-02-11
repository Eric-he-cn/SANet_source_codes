import numpy as np
import tensorflow as tf
from scipy.io import savemat
import helper as hp
from tensorflow import set_random_seed
"""
net_train2.py
=============
Consolidated training/evaluation utilities for the *final BF pipeline* and *ZF baselines*
(TensorFlow 1.x graph mode). This file is intended to be the cleaner, publication-facing
counterpart of net_train.py for beamforming-related experiments.

Main functions
--------------
Beamforming (BF)
- DNN_SAT_BF(...):
    Baseline BF training/evaluation using helper.DNN_BF.
- HyperDop_SAT_BF(...):   [FINAL for BF]
    Sensing-aided hypernetwork BF training/evaluation using helper.HyperDop_BF with the
    standard expected sum-rate objective (sample-wise log(1+SINR) then average).

ZF Baselines
- ZF_Procoder(...):
    Computes ZF precoder from estimated channel and applies power normalization.
- ZF_BF(...):
    ZF baseline evaluation pipeline (typically used when sum-rate curves are needed).
- ZF_BF2(...):
    ZF baseline variant focused on NMSE / saving intermediate CSI (used by some sweeps).
"""


def DNN_SAT_BF(M=64, B=5, Lp=2, L=2, K=2, batch_size=1024, mini_batch=1,total_epoch=10000):
    # M transmit annennas
    # K number of users
    # B bits for each user;
    # L pilot length
    # Lp number of paths

    np.random.seed(42)
    set_random_seed(42)
    SNRdb = 10
    sigma2 = 1 / (10 ** (SNRdb / 10.0))  # sigma2:噪声方差σ^2
    Mx = np.sqrt(M).astype(int)
    My = np.sqrt(M).astype(int)
    file_name = 'DNN_BF L(' + str(L) + ') K(' + str(K) + ') B(' + str(B) + ') Lp(' + str(Lp) + ')'
    print(file_name)

    #构建神经网络
    tf.reset_default_graph()    #创建新的图
    X = tf.placeholder("complex64", [batch_size, K, M])         #X为真实的信道信息h（复数形式）
    alpha_para = tf.placeholder("float32", [])                          #神经网络的激活函数tanh的一个参数
    N0_dnn = tf.placeholder("float32", [])

    #调用函数hp.DNN_BF，输出为v_BF, test1
    precode,test1 = hp.DNN_BF(X, N0_dnn, batch_size, M, B, L, K, alpha_para)  # [batch_size, K, M]

    #计算sum_rate
    ach_sum_rate = 0.
    for k in range(K):
        sr_temp1 = 0.
        sr_temp2 = 0.
        sum_rate = 0.
        precode_k = precode[:, k, :]  # 只算第S个时隙,[batch_size, M]
        for j in range(K):
            X_j = X[:, j, :]                       #[batch_size, M]
            if j == k:
                # sr_temp1_k = tf.square(tf.real(tf.reduce_sum(tf.multiply(precode_k, X_j), axis=1))) + tf.square(tf.imag(tf.reduce_sum(tf.multiply(precode_k, X_j), axis=1))) # [batch_size]
                sr_temp1 = tf.square(  tf.abs(  tf.reduce_sum(tf.multiply(precode_k, X_j), axis=1)  )  ) # [batch_size]
            else:
                # sr_temp2_k += tf.square(tf.real(tf.reduce_sum(tf.multiply(precode_k, X_j), axis=1))) + tf.square(tf.imag(tf.reduce_sum(tf.multiply(precode_k, X_j), axis=1)))  # [batch_size]
                sr_temp2 += tf.square(  tf.abs(  tf.reduce_sum(tf.multiply(precode_k, X_j), axis=1)  )  )   # [batch_size]
        sum_rate = tf.reduce_mean(tf.math.log(1 + tf.divide(sr_temp1, (sr_temp2 + sigma2))))/ 0.6931
        ach_sum_rate += sum_rate / K
    cross = -ach_sum_rate


    learning_rate = tf.placeholder(tf.float32, shape=[])                            #学习率
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross) #优化器为adam
    init = tf.global_variables_initializer()                                        #初始化全局变量

    avg_rate = 0.
    rate_array = np.zeros((int(total_epoch / 10), 1), dtype=float)            #存储每隔10个周期的损失值
    rate_index = 0


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

            N0_input = sigma2  # 噪声输入
            alpha_para_input = 1.0                      #α是固定值

            for index_m in range(mini_batch):           #循环mini_batch次
                gain_dl_real = np.random.normal(loc=0.707, scale=2, size=(batch_size, K, Lp))
                gain_dl_img = np.random.normal(loc=0.707, scale=2, size=(batch_size, K, Lp))
                gain_dl = gain_dl_real + 1j * gain_dl_img

                fc_dl = 2 * 1e9  # 下行信道载频


                ############关于星地信道的部分###################
                KK = 10  # 莱斯因子10dB
                f_dop2 = 20000 + 40000 * np.random.rand(batch_size, K)                         #多普勒频移:20k~60kHz
                #f_dop2 = np.expand_dims(f_dop, 0).repeat(batch_size, axis=0)
                tau = (10.0 * np.random.rand(batch_size, K) + 10.0) * 0.001  # 10ms~20ms
                theta = (4 * np.pi / 9) * np.random.rand(batch_size, K) + np.pi / 18  # 18/π ~ 2/π
                phi = 2 * np.pi * np.random.rand(batch_size, K)  # 0~2π

                h_los_dl = np.zeros([batch_size, K, M], dtype=complex)
                h_nlos_dl = np.zeros([batch_size, K, M], dtype=complex)
                h_dl = np.zeros([batch_size, K, M], dtype=complex)

                arv_x = np.zeros([Mx], dtype=complex)  # array response vector_x
                arv_y = np.zeros([My], dtype=complex)  # array response vector_y
                for size in range(batch_size):

                    for k in range(K):
                        for mx in range(Mx):
                            arv_x[mx] = np.exp(-1j * np.pi * mx * np.cos(theta[size, k]) * np.sin(phi[size, k]))
                        for my in range(My):
                            arv_y[my] = np.exp(-1j * np.pi * my * np.cos(theta[size, k]) * np.sin(phi[size, k]))
                        arv = np.kron(arv_x, arv_y)  # array response vector

                        h_los_dl[size, k, :] = np.exp(1j * 2 * np.pi * (4 * f_dop2[size, k]- fc_dl * tau[size, k])) * arv
                        for p in range(Lp):
                            h_nlos_dl[size, k, :] = h_nlos_dl[size, k, :] + np.sqrt(1 / Lp) * gain_dl[
                                size, k, p] * np.exp(
                                1j * 2 * np.pi * (4 * f_dop2[size, k] - fc_dl * tau[size, k])) * arv

                        h_dl[size, k, :] = np.sqrt(KK / (KK + 1)) * h_los_dl[size, k, :] + np.sqrt(
                            1 / (KK + 1)) * h_nlos_dl[size, k, :]



                #### sess.run()意味执行了一次神经网络的训练迭代，通过反向传播和优化器更新模型参数，同时计算了损失和测试结果
                _, cs, test11 = sess.run([optimizer, cross, test1],
                                         feed_dict={X: h_dl, N0_dnn: N0_input, alpha_para: alpha_para_input,
                                                    learning_rate: learning_R})
                avg_rate += -cs

            if (epoch + 1) % 200 == 0:           #每10个周期打印NMSE
                print("Epoch:", '%04d' % (epoch + 1), "sum_rate=", "{:9f}".format(avg_rate / 200))
                rate_array[rate_index, 0] = avg_rate / 200
                rate_index += 1
                avg_rate = 0.

            if (epoch + 1) % 1000 == 0:         #每1000个周期打印信息并保存数据
                store_dic = {'rate_array': rate_array}
                savemat(file_name + ".mat", store_dic)  # 保存损失数组


def HyperDop_SAT_BF(M=64, B=25, Lp=2, L=2, time_doppler_opt=1, K=2, batch_size=1024, S=4, mini_batch=1, total_epoch=10000):
    ############################
    # M transmit annennas
    # K number of users
    # B bits for each user;
    # L pilot length
    # Lp number of paths
    np.random.seed(42)
    set_random_seed(42)
    SNRdb = 10
    sigma2 = 1 / (10 ** (SNRdb / 10.0))  # sigma2:噪声方差σ^2
    Mx = np.sqrt(M).astype(int)
    My = np.sqrt(M).astype(int)
    file_name = 'Hyp_BF L(' + str(L) + ') K(' + str(K) + ') B(' + str(B) + ') Lp(' + str(Lp) + ')   opt(' + str(time_doppler_opt) + ') S(' + str(S) + ') '
    print(file_name)

    # 构建神经网络
    tf.reset_default_graph()                        # 创建新的图
    X = tf.placeholder("complex64", [batch_size, K, S, M])          # X为真实的信道信息h（复数形式）
    hyper_input = tf.placeholder("float32", [batch_size, K, 1])
    N0_dnn = tf.placeholder("float32", [])

    # 调用函数hp.DNN_BF，输出为v_BF, test1
    precode = hp.HyperDop_BF(X, N0_dnn, batch_size, M, B, L, K, S, hyper_input)  # [batch_size, K,S, M]


    #计算sum_rate
    ach_sum_rate = 0.
    # ach_sum_rate = tf.Variable(tf.zeros(shape=[batch_size], dtype=tf.float32))
    for k in range(K):
        sr_temp1_k = 0.
        sr_temp2_k = 0.
        precode_k = precode[:, k, S - 1, :]  # 只算第S个时隙,[batch_size, M]
        for j in range(K):
            X_j = X[:, j, S-1, :]                       #[batch_size, M]
            if j == k:
                # sr_temp1_k = tf.square(tf.real(tf.reduce_sum(tf.multiply(precode_k, X_j), axis=1))) + tf.square(tf.imag(tf.reduce_sum(tf.multiply(precode_k, X_j), axis=1))) # [batch_size]
                sr_temp1_k = tf.square(  tf.abs(  tf.reduce_sum(tf.multiply(precode_k, X_j), axis=1)  )  ) # [batch_size]
            else:
                # sr_temp2_k += tf.square(tf.real(tf.reduce_sum(tf.multiply(precode_k, X_j), axis=1))) + tf.square(tf.imag(tf.reduce_sum(tf.multiply(precode_k, X_j), axis=1)))  # [batch_size]
                sr_temp2_k += tf.square(  tf.abs(  tf.reduce_sum(tf.multiply(precode_k, X_j), axis=1)  )  )   # [batch_size]
        sum_rate_k = tf.reduce_mean(tf.math.log(1 + tf.divide(sr_temp1_k, (sr_temp2_k + sigma2))))/ 0.6931
        ach_sum_rate += sum_rate_k / K
    cross = - ach_sum_rate

    learning_rate = tf.placeholder(tf.float32, shape=[])  # 学习率
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross)  # 优化器为adam
    init = tf.global_variables_initializer()  # 初始化全局变量

    avg_rate = 0.
    rate_array = np.zeros((int(total_epoch / 10), 1), dtype=float)  # 存储每隔10个周期的损失值
    rate_index = 0
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

    with tf.Session() as sess:  # 创建了一个 TensorFlow 会话（Session），会话是执行计算图的环境
        # Training
        sess.run(init)  # 使用之前定义的init变量初始化sess
        for epoch in range(total_epoch):  # 开始循环

            if epoch < total_epoch / 3:  # 动态调整学习率
                learning_R = 0.001
            elif epoch < (3.0 * total_epoch / 4):
                learning_R = 0.0001
            else:
                learning_R = 0.00001

            N0_input = sigma2  # 噪声输入

            for index_m in range(mini_batch):  # 循环mini_batch次
                gain_dl0 = np.sqrt(0.5)*(np.random.standard_normal([batch_size,K,Lp])+1j*np.random.standard_normal([batch_size,K,Lp]))
                gain_dl = np.sqrt(0.5)*(np.random.standard_normal([batch_size,K,S,Lp]) + 1j*np.random.standard_normal([batch_size,K,S,Lp]))
                gain_dl[:,:,0,:] = gain_dl0
                for s in range(S):
                    if s > 0:
                        noise = np.sqrt(0.5) * (np.random.standard_normal([batch_size,K,Lp]) + 1j * np.random.standard_normal(
                                    [batch_size,K,Lp]))
                        gain_dl[:,:,s,:] = gain_dl[:,:,s-1,:] * dl_cor + noise * np.sqrt(1 - np.square(dl_cor))


                # delta_time = 0.4 * np.random.rand(batch_size, K) + 0.1  # 上下行信道的时间延迟
                fc_dl = 2 * 1e9  # 下行信道载频
                KK = 10  # 莱斯因子10dB

                f_dop2 = 20000 + 40000 * np.random.rand(batch_size, K)  # 多普勒频移: 20k~60kHz
                # f_dop2 = np.expand_dims(f_dop,0).repeat(batch_size,axis=0)
                v_sat = np.zeros([batch_size, K, 1], dtype=float)
                v_sat[:, :, 0] = f_dop2 * 3e8 / (2 * fc_dl)  # 速度：1500~4500m/s
                v_sat_input = v_sat / 300 - 4             #############################################   有修改

                tau = (10.0 * np.random.rand(batch_size, K) + 10.0) * 0.001  # 时延  10ms~20ms
                theta = (4 * np.pi / 9) * np.random.rand(batch_size, K) + np.pi / 18  # 18/π ~ 2/π
                phi = 2 * np.pi * np.random.rand(batch_size, K)  # 0~2π

                h_los_dl = np.zeros([batch_size, K, S, M], dtype=complex)
                h_nlos_dl = np.zeros([batch_size, K, S, M], dtype=complex)
                h_dl = np.zeros([batch_size, K, S, M], dtype=complex)

                arv_x = np.zeros([Mx], dtype=complex)  # array response vector_x
                arv_y = np.zeros([My], dtype=complex)  # array response vector_y
                for size in range(batch_size):
                    for k in range(K):
                        for mx in range(Mx):
                            arv_x[mx] = np.exp(-1j * np.pi * mx * np.cos(theta[size, k]) * np.sin(phi[size, k]))
                        for my in range(My):
                            arv_y[my] = np.exp(-1j * np.pi * my * np.cos(theta[size, k]) * np.sin(phi[size, k]))
                        arv = np.kron(arv_x, arv_y)  # array response vector
                        for s in range(S):
                            h_los_dl[size, k , s, :] = np.exp(1j * 2 * np.pi *(s * f_dop2[size, k] - fc_dl * tau[size, k])) * arv
                            for p in range(Lp):
                                h_nlos_dl[size, k , s, :] = h_nlos_dl[size, k , s, :] + np.sqrt(1 / Lp) * gain_dl[size, k, s, p] * np.exp(1j * 2 * np.pi * (s  * f_dop2[size, k]- fc_dl * tau[size, k])) * arv
                                                                #######################     有修改   S*f_dop2
                            h_dl[size, k , s, :] = np.sqrt(KK / (KK + 1)) * h_los_dl[size, k , s, :] + np.sqrt(
                                1 / (KK + 1)) * h_nlos_dl[size, k , s, :]
                # print(h_dl[0,:,:,0])
                #### sess.run()意味执行了一次神经网络的训练迭代，通过反向传播和优化器更新模型参数，同时计算了损失和测试结果
                _, cs = sess.run([optimizer, cross],
                                         feed_dict={X: h_dl, hyper_input: (v_sat_input), N0_dnn: N0_input,
                                                    learning_rate: learning_R})
                avg_rate += -cs

            if (epoch + 1) % 200 == 0:  # 每10个周期打印NMSE
                print("Epoch:", '%04d' % (epoch + 1), "sum_rate=", "{:9f}".format(avg_rate / 200))
                rate_array[rate_index, 0] = avg_rate / 200
                rate_index += 1
                avg_rate = 0.

            if (epoch + 1) % 1000 == 0:  # 每1000个周期打印信息并保存数据
                store_dic = {'rate_array': rate_array}
                savemat(file_name + ".mat", store_dic)  # 保存损失数组


def ZF_Procoder(h_est,batch_size,K,M):
    h_est_complex_reshaped = np.reshape(h_est, [batch_size * K, M])
    h_est_complex_H = np.conj(np.transpose(h_est_complex_reshaped))
    h_product_reg = np.matmul(h_est_complex_reshaped, h_est_complex_H) + 0.01 * np.eye(batch_size * K, dtype=np.complex64)
    h_product_inv = np.linalg.inv(h_product_reg)
    W = np.matmul(h_est_complex_H, h_product_inv)
    W = np.reshape(W, [batch_size, K, M])
    W_power = np.sqrt(np.sum(np.sum(np.square(np.abs(W)),axis=1,keepdims=True),axis=2,keepdims=True))
    W_normalized = W / W_power
    return W_normalized

def ZF_BF(M=64, B=5, Lp=2, L=2, K=2, batch_size=1024, mini_batch=1,total_epoch=10000):
    # M transmit annennas
    # K number of users
    # B bits for each user;
    # L pilot length
    # Lp number of paths

    np.random.seed(42)
    set_random_seed(42)
    SNRdb = 10
    sigma2 = 1 / (10 ** (SNRdb / 10.0))  # sigma2:噪声方差σ^2
    Mx = np.sqrt(M).astype(int)
    My = np.sqrt(M).astype(int)
    file_name = 'ZF_BF L(' + str(L) + ') K(' + str(K) + ') B(' + str(B) + ') Lp(' + str(Lp) + ')'
    print(file_name)

    #构建神经网络
    tf.reset_default_graph()    #创建新的图
    h_true = tf.placeholder("complex64", [batch_size, K, M])         #X为真实的信道信息h（复数形式）
    alpha_para = tf.placeholder("float32", [])                          #神经网络的激活函数tanh的一个参数
    N0_dnn = tf.placeholder("float32", [])

    #调用函数hp.DNN_BF，输出为v_BF, test1
    h_est = hp.ZF_BF(h_true, N0_dnn, batch_size, M, B, L, K, alpha_para)  # [batch_size, K, M]

    cross = tf.reduce_mean(tf.square(tf.abs(h_true - h_est)))  /  tf.reduce_mean(tf.square(tf.abs(h_true)))    #计算均一化均方误差NMSE
    learning_rate = tf.placeholder(tf.float32, shape=[])                            #学习率
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross) #优化器为adam
    init = tf.global_variables_initializer()                                        #初始化全局变量
    avg_nmse = 0.
    avg_rate = 0.
    rate_array = np.zeros((int(total_epoch / 10), 1), dtype=float)            #存储每隔10个周期的损失值
    rate_index = 0


    with tf.Session() as sess:
        # Training
        sess.run(init)
        for epoch in range(total_epoch):                 #开始循环

            if epoch < total_epoch / 3:                 #动态调整学习率
                learning_R = 0.001
            elif epoch < (3.0 * total_epoch / 4):
                learning_R = 0.0001
            else:
                learning_R = 0.00001

            N0_input = sigma2  # 噪声输入
            alpha_para_input = 1.0                      #α是固定值

            for index_m in range(mini_batch):           #循环mini_batch次
                gain_dl_real = np.random.normal(loc=0.707, scale=2, size=(batch_size, K, Lp))
                gain_dl_img = np.random.normal(loc=0.707, scale=2, size=(batch_size, K, Lp))
                gain_dl = gain_dl_real + 1j * gain_dl_img

                fc_dl = 2 * 1e9  # 下行信道载频

                ############关于星地信道的部分###################
                KK = 10  # 莱斯因子10dB
                f_dop2 = 20000 + 40000 * np.random.rand(batch_size, K)                         #多普勒频移:20k~60kHz
                #f_dop2 = np.expand_dims(f_dop, 0).repeat(batch_size, axis=0)
                tau = (10.0 * np.random.rand(batch_size, K) + 10.0) * 0.001  # 10ms~20ms
                theta = (4 * np.pi / 9) * np.random.rand(batch_size, K) + np.pi / 18  # 18/π ~ 2/π
                phi = 2 * np.pi * np.random.rand(batch_size, K)  # 0~2π

                h_los_dl = np.zeros([batch_size, K, M], dtype=complex)
                h_nlos_dl = np.zeros([batch_size, K, M], dtype=complex)
                h_dl = np.zeros([batch_size, K, M], dtype=complex)

                arv_x = np.zeros([Mx], dtype=complex)  # array response vector_x
                arv_y = np.zeros([My], dtype=complex)  # array response vector_y
                for size in range(batch_size):

                    for k in range(K):
                        for mx in range(Mx):
                            arv_x[mx] = np.exp(-1j * np.pi * mx * np.cos(theta[size, k]) * np.sin(phi[size, k]))
                        for my in range(My):
                            arv_y[my] = np.exp(-1j * np.pi * my * np.cos(theta[size, k]) * np.sin(phi[size, k]))
                        arv = np.kron(arv_x, arv_y)  # array response vector

                        h_los_dl[size, k, :] = np.exp(1j * 2 * np.pi * (4 * f_dop2[size, k]- fc_dl * tau[size, k])) * arv
                        for p in range(Lp):
                            h_nlos_dl[size, k, :] = h_nlos_dl[size, k, :] + np.sqrt(1 / Lp) * gain_dl[
                                size, k, p] * np.exp(
                                1j * 2 * np.pi * (4 * f_dop2[size, k] - fc_dl * tau[size, k])) * arv

                        h_dl[size, k, :] = np.sqrt(KK / (KK + 1)) * h_los_dl[size, k, :] + np.sqrt(
                            1 / (KK + 1)) * h_nlos_dl[size, k, :]

                #### sess.run()意味执行了一次神经网络的训练迭代，通过反向传播和优化器更新模型参数，同时计算了损失和测试结果
                _, cs, h_est2, h_true2 = sess.run([optimizer, cross, h_est,h_true],
                                         feed_dict={h_true: h_dl, N0_dnn: N0_input, alpha_para: alpha_para_input,
                                                    learning_rate: learning_R})
                avg_nmse += cs

                w = ZF_Procoder(h_est2,batch_size,K,M)
                ach_sum_rate = 0.
                for k in range(K):
                    sr_temp1 = 0.
                    sr_temp2 = 0.
                    sum_rate = 0.
                    precode_k = w[:, k, :]  # 只算第S个时隙,[batch_size, M]
                    for j in range(K):
                        X_j = h_true2[:, j, :]                       #[batch_size, M]
                        if j == k:
                            sr_temp1 = np.square(np.abs(np.sum(np.multiply(precode_k, X_j), axis=1))) # [batch_size]
                        else:
                            sr_temp2 += np.square(np.abs(np.sum(np.multiply(precode_k, X_j), axis=1)))   # [batch_size]
                    sum_rate = np.mean(np.log(1 + np.divide(sr_temp1, (sr_temp2 + sigma2))))/ 0.6931
                    ach_sum_rate += sum_rate / K

                avg_rate += ach_sum_rate

            if (epoch + 1) % 10 == 0:           #每10个周期打印NMSE
                print("Epoch:", '%04d' % (epoch + 1), "nmse=", "{:9f}".format(avg_nmse / 10), "sum_rate=", "{:9f}".format(avg_rate / 10))
                rate_array[rate_index, 0] = avg_rate / 10
                rate_index += 1
                avg_rate = 0.
                avg_nmse = 0.

            if (epoch + 1) % 1000 == 0:         #每1000个周期打印信息并保存数据
                store_dic = {'rate_array': rate_array}
                savemat(file_name + ".mat", store_dic)  # 保存损失数组

def ZF_BF2(M=64, B=5, Lp=2, L=2, K=2, batch_size=1024, mini_batch=1,total_epoch=10000):
    # M transmit annennas
    # K number of users
    # B bits for each user;
    # L pilot length
    # Lp number of paths

    np.random.seed(42)
    set_random_seed(42)
    SNRdb = 10
    sigma2 = 1 / (10 ** (SNRdb / 10.0))  # sigma2:噪声方差σ^2
    Mx = np.sqrt(M).astype(int)
    My = np.sqrt(M).astype(int)
    file_name = 'ZF_BF L(' + str(L) + ') K(' + str(K) + ') B(' + str(B) + ') Lp(' + str(Lp) + ')'
    print(file_name)

    #构建神经网络
    tf.reset_default_graph()    #创建新的图
    h_true = tf.placeholder("complex64", [batch_size, K, M])         #X为真实的信道信息h（复数形式）
    alpha_para = tf.placeholder("float32", [])                          #神经网络的激活函数tanh的一个参数
    N0_dnn = tf.placeholder("float32", [])

    #调用函数hp.DNN_BF，输出为v_BF, test1
    h_est = hp.ZF_BF(h_true, N0_dnn, batch_size, M, B, L, K, alpha_para)  # [batch_size, K, M]

    cross = tf.reduce_mean(tf.square(tf.abs(h_true - h_est)))  /  tf.reduce_mean(tf.square(tf.abs(h_true)))    #计算均一化均方误差NMSE



    learning_rate = tf.placeholder(tf.float32, shape=[])                            #学习率
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross) #优化器为adam
    init = tf.global_variables_initializer()                                        #初始化全局变量

    avg_nmse = 0.

    h_est_array = np.zeros((batch_size, K, M), dtype=complex)
    h_real_array = np.zeros((batch_size, K, M), dtype=complex)

    nmse_array = np.zeros((int(total_epoch / 10), 1), dtype=float)            #存储每隔10个周期的损失值
    nmse_index = 0


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

            N0_input = sigma2  # 噪声输入
            alpha_para_input = 1.0                      #α是固定值

            for index_m in range(mini_batch):           #循环mini_batch次
                gain_dl_real = np.random.normal(loc=0.707, scale=2, size=(batch_size, K, Lp))
                gain_dl_img = np.random.normal(loc=0.707, scale=2, size=(batch_size, K, Lp))
                gain_dl = gain_dl_real + 1j * gain_dl_img

                fc_dl = 2 * 1e9  # 下行信道载频


                ############关于星地信道的部分###################
                KK = 10  # 莱斯因子10dB
                f_dop2 = 20000 + 40000 * np.random.rand(batch_size, K)                         #多普勒频移:20k~60kHz
                #f_dop2 = np.expand_dims(f_dop, 0).repeat(batch_size, axis=0)
                tau = (10.0 * np.random.rand(batch_size, K) + 10.0) * 0.001  # 10ms~20ms
                theta = (4 * np.pi / 9) * np.random.rand(batch_size, K) + np.pi / 18  # 18/π ~ 2/π
                phi = 2 * np.pi * np.random.rand(batch_size, K)  # 0~2π

                h_los_dl = np.zeros([batch_size, K, M], dtype=complex)
                h_nlos_dl = np.zeros([batch_size, K, M], dtype=complex)
                h_dl = np.zeros([batch_size, K, M], dtype=complex)

                arv_x = np.zeros([Mx], dtype=complex)  # array response vector_x
                arv_y = np.zeros([My], dtype=complex)  # array response vector_y
                for size in range(batch_size):

                    for k in range(K):
                        for mx in range(Mx):
                            arv_x[mx] = np.exp(-1j * np.pi * mx * np.cos(theta[size, k]) * np.sin(phi[size, k]))
                        for my in range(My):
                            arv_y[my] = np.exp(-1j * np.pi * my * np.cos(theta[size, k]) * np.sin(phi[size, k]))
                        arv = np.kron(arv_x, arv_y)  # array response vector

                        h_los_dl[size, k, :] = np.exp(1j * 2 * np.pi * (4 * f_dop2[size, k]- fc_dl * tau[size, k])) * arv
                        for p in range(Lp):
                            h_nlos_dl[size, k, :] = h_nlos_dl[size, k, :] + np.sqrt(1 / Lp) * gain_dl[
                                size, k, p] * np.exp(
                                1j * 2 * np.pi * (4 * f_dop2[size, k] - fc_dl * tau[size, k])) * arv

                        h_dl[size, k, :] = np.sqrt(KK / (KK + 1)) * h_los_dl[size, k, :] + np.sqrt(
                            1 / (KK + 1)) * h_nlos_dl[size, k, :]

                #### sess.run()意味执行了一次神经网络的训练迭代，通过反向传播和优化器更新模型参数，同时计算了损失和测试结果
                _, cs, h_est2, h_true2 = sess.run([optimizer, cross, h_est,h_true],
                                         feed_dict={h_true: h_dl, N0_dnn: N0_input, alpha_para: alpha_para_input,
                                                    learning_rate: learning_R})
                avg_nmse += cs

            if (epoch + 1) % 10 == 0:           #每10个周期打印NMSE
                print("Epoch:", '%04d' % (epoch + 1), "nmse=", "{:9f}".format(avg_nmse / 10))
                nmse_array[nmse_index, 0] = avg_nmse / 10
                nmse_index += 1
                avg_nmse = 0.
            if (epoch + 1) % 1000 == 0:         #每1000个周期打印信息并保存数据
                store_dic = {'rate_array': nmse_array, 'h_true': h_true2, 'h_est': h_est2}
                savemat(file_name + ".mat", store_dic)  # 保存损失数组









