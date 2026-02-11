import numpy as np
import tensorflow as tf
from tensorflow import set_random_seed
from tensorflow.contrib.layers import xavier_initializer

"""
helper.py
=========
This file defines the neural network building blocks (graph-level forward models) used in the
GLOBECOM 2025 SANet codebase. All models are implemented in TensorFlow 1.x graph mode.

Main contents
-------------
1) Channel Estimation (CE) networks
   - DNN_CE(...):
       Baseline CE network using pilots/received signals only.
   - HyperDop_CE(...):
       Sensing-aided (velocity/Doppler) hypernetwork-based CE model. The hypernetwork takes
       sensing features (e.g., velocity) as input to adapt/condition CE inference.

2) Beamforming (BF) networks
   - DNN_BF(...):
       Baseline BF network that outputs the precoder/beamforming matrix (data-driven BF).
   - HyperDop_BF(...):
       Sensing-aided hypernetwork BF model. The hypernetwork conditions the BF network on
       Doppler/velocity-related features to improve robustness under LEO time variation.

3) Classical baseline helper
   - ZF_BF(...):
       Utility module used by ZF baseline pipelines (e.g., mapping estimated CSI to ZF precoder,
       or related intermediate components). Final ZF evaluation is implemented in net_train2.py.

Notes
-----
- This file focuses on *network architecture / forward graph* only.
- Training loops, channel generation, loss definitions, logging, and saving .mat outputs are
  implemented in net_train.py / net_train2.py.
"""


np.random.seed(12)
set_random_seed(12)


def init_weights(shape, name):          #初始化权重
    return tf.get_variable(str(name) + '_w', shape, tf.float32, xavier_initializer())

def init_bias(shape, name):             ##初始化偏置
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals, name=name + '_b')

def normal_full_layer(input_layer, size, name):         #定义一个全连接层（有偏置），返回值为输出
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size], name)
    b = init_bias([size], name)
    return tf.matmul(input_layer, W) + b

def normal_full_layer2(input_size, size, name):         #定义一个全连接层（有偏置），返回值为W和b
    W = init_weights([input_size, size], name)
    b = init_bias([size], name)
    return W, b

def normal_full_layer2_nobias(input_size, size, name):  #定义一个全连接层（无偏置），返回值为W
    W = init_weights([input_size, size], name)
    return W

def full_layer_no_bias(input_layer, size, name):        #定义一个全连接层（无偏置），返回值为输出
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size], name)
    return tf.matmul(input_layer, W)

def normal_full_layer_withWandB(input_layer, W, b):     #定义一个全连接层（有偏置），用户手动提供W和b
    return tf.matmul(input_layer, W) + b

def standart_gaussian_noise_layer(shape):               #定义一个标准高斯噪声层
    noise = tf.random_normal(shape=shape)
    return noise

def frange(x, y, jump):                                 #range升级版，可以浮点数range
    while x <= y:
        yield x
        x += jump


def DNN_CE(X,  N0, batch_size, M,  B, L, alpha_para):
    #X:真实信道信息，[batch_size, M]

    #导频信号初始化
    X_tilde_ini = tf.Variable(tf.sqrt(1 / M) * standart_gaussian_noise_layer([M, 2 * L]), trainable=True, name='x')     #[M, 2 * L]
    power_normal = tf.sqrt(tf.reduce_sum(tf.square(X_tilde_ini[:, 0:L]) + tf.square(X_tilde_ini[:, L:2 * L]), axis=0))  #[L]
    X_tilde = X_tilde_ini / (tf.concat([power_normal, power_normal], axis=0))           #归一化，[M, 2 * L]
    X_tilde_complex = tf.complex(X_tilde[:, 0:L], X_tilde[:, L:2 * L])                  #待训练的导频信号[M,L]

    #计算接收信号y
    y = tf.matmul(X, X_tilde_complex)                                                   #y=h*x， y:[batch_size,L]
    y_real = tf.concat([tf.real(y), tf.imag(y)], axis=1)                                #[batch_size, 2*L]
    noise = tf.sqrt(N0 / 2) * standart_gaussian_noise_layer((batch_size, L * 2))
    y_noise = y_real + noise                                                    #[batch_size,2*L]

    #量化DNN，输入y_noise，输出q
    u1 = (normal_full_layer(y_noise, 1024, 'u1'))                     #第一个全连接层，u1是全连接层输出
    n_u1 = tf.nn.relu(tf.layers.batch_normalization(u1, name='u1_normalization'))       #n_u1是u1加上relu激活函数
    u2 = (normal_full_layer(n_u1, 512, 'u2'))                         #第二个全连接层
    n_u2 = tf.nn.relu(tf.layers.batch_normalization(u2, name='u2_normalization'))
    u3 = (normal_full_layer(n_u2, 256, 'u3'))                         #第三个全连接层
    n_u3 = tf.nn.relu(tf.layers.batch_normalization(u3, name='u3_normalization'))
    q = tf.sign(normal_full_layer(n_u3, B, 'u4'))          #第四个全连接层，激活函数不是tanh而是sign，输出q[batch_size,B]

    #信道估计DNN，输入q，输出h_est
    r1 = (normal_full_layer(q, 1024, 'r1'))                             #第五个全连接层（下半部分开始）
    n_r1 = tf.nn.relu(tf.layers.batch_normalization(r1, name='r1_normalization'))
    r2 = (normal_full_layer(n_r1, 512, 'r2'))                           #第六个全连接层
    n_r2 = tf.nn.relu(tf.layers.batch_normalization(r2, name='r2_normalization'))
    r3 = (normal_full_layer(n_r2, 512, 'r3'))                           #第七个全连接层
    n_r3 = tf.nn.relu(tf.layers.batch_normalization(r3, name='r3_normalization'))
    h_est = normal_full_layer(n_r3, M * 2, 'r4')                             #第八个全连接层，没有激活函数
    return h_est, X_tilde               #h_est:[batch_size,2*M],   X_tilde:[M, 2*L],  均为实数

def HyperDop_CE(X1, K, N0, batch_size, M, B, L, S, Lp, v_sat):

    #v_sat:[batch_size, K, 1]

    ##################################     HYPER  DNN     ####################################
    U1, B1 = normal_full_layer2(1, 32, 'R1')
    U2, B2 = normal_full_layer2(32, 64, 'R2')
    U3, B3 = normal_full_layer2(64, B, 'R4')
    hyper_temp1 = tf.nn.relu(tf.matmul(v_sat, U1) + B1)  # batch_size, K, 64
    hyper_temp2 = tf.nn.relu(tf.matmul(hyper_temp1, U2) + B2)  # batch_size, K, 64
    hyper_y = tf.matmul(hyper_temp2, U3) + B3  # batch_size, K, B


    ###################################          MAIN           #################################

    X_tilde_ini = tf.Variable(tf.sqrt(1 / M) * standart_gaussian_noise_layer([M, 2 * L]), trainable=True, name='x1')
    power_normal = tf.sqrt(
        tf.reduce_sum(tf.square(X_tilde_ini[:, 0:L]) + tf.square(X_tilde_ini[:, L:2 * L]), axis=0))
    X_tilde = X_tilde_ini / (tf.concat([power_normal, power_normal], axis=0))
    X_tilde_complex = tf.reshape(tf.complex(X_tilde[:, 0:L], X_tilde[:, L:2 * L]), [1,1, M, L])  # M,L
    X_tilde_complex1_full = tf.tile(X_tilde_complex, [batch_size,K, 1, 1])  # (batch_size,batch_size,K,M,L)
    y = tf.matmul(X1, X_tilde_complex1_full)                # (batch_size,K,S,M) (batch_size,K,M,L) =(batch_size, K, S, L)
    y_real = tf.concat([tf.real(y), tf.imag(y)], axis=3)    #(batch_size, K, S, 2*L)
    noise = tf.sqrt(N0 / 2) * standart_gaussian_noise_layer((batch_size, K, S, L * 2))
    y_noise = y_real + noise  # (batch_size, K, S, 2*L)

    #量化DNN，输入y_noise，输出q：(batch_size, K, B)
    u1, ub1 = normal_full_layer2(2 * L, 256, 'r1')
    u2, ub2 = normal_full_layer2(256, 256, 'r2')
    u3, ub3 = normal_full_layer2(256, 128, 'r3')
    u4, ub4 = normal_full_layer2(128, B, 'r4')
    nu1 = tf.nn.relu(tf.matmul(y_noise[:, :, 0, :], u1) + ub1)
    nu2 = tf.nn.relu(tf.matmul(nu1, u2) + ub2)
    nu3 = tf.nn.relu(tf.matmul(nu2, u3) + ub3)
    q = tf.sign(tf.matmul(nu3, u4) + ub4)
    q2 = q * hyper_y
    #信道估计RNN，输入q2(batch_size, K, B)，输出h_est(batch_size, K, 2M)
    U2, B2 = normal_full_layer2(B, 256, 'U2')
    W2 = normal_full_layer2_nobias(256, 256, 'H2')
    V2, C2 = normal_full_layer2(256, M * 2, 'V2')
    H2 = tf.nn.relu(tf.matmul(q2, U2) + B2)
    h_est = tf.reshape(tf.matmul(H2, V2) + C2,[batch_size, K, 1, M * 2])

    for s in range(S-1):
        nu1 = tf.nn.relu(tf.matmul(y_noise[:, :, s + 1, :], u1) + ub1)
        nu2 = tf.nn.relu(tf.matmul(nu1, u2) + ub2)
        nu3 = tf.nn.relu(tf.matmul(nu2, u3) + ub3)
        q = tf.sign(tf.matmul(nu3, u4) + ub4)
        q2 = q * hyper_y
        H2 = tf.nn.relu(tf.matmul(q2, U2) + B2 + tf.matmul(H2, W2))
        y_temp = tf.reshape(tf.matmul(H2, V2) + C2, [batch_size, K, 1, M * 2])
        h_est = tf.concat([h_est, y_temp], axis=2)

    h_est_complext = tf.complex(h_est[:, :, :, 0:M], h_est[:, :, :, M:2 * M])       #[batch_size, K, S, M * 2])
    return h_est_complext

def DNN_BF(X,  N0, batch_size, M,  B, L, K, alpha_para):
    #X:真实信道信息，[batch_size, K, M]

    #导频信号初始化
    X_tilde_ini = tf.Variable(tf.sqrt(1 / M) * standart_gaussian_noise_layer([M, 2 * L]), trainable=True, name='x')     #[M, 2 * L]
    power_normal = tf.sqrt(tf.reduce_sum(tf.square(X_tilde_ini[:, 0:L]) + tf.square(X_tilde_ini[:, L:2 * L]), axis=0))  #[L]
    X_tilde = X_tilde_ini / (tf.concat([power_normal, power_normal], axis=0))           #归一化，[M, 2 * L]
    X_tilde_complex = tf.complex(X_tilde[:, 0:L], X_tilde[:, L:2 * L])                  #待训练的导频信号[M,L]

    #计算接收信号y
    y = tf.matmul(X, X_tilde_complex)                                                   #y=h*x， y:[batch_size, K, L]
    y_real = tf.concat([tf.real(y), tf.imag(y)], axis=2)                                #[batch_size, K, 2*L]
    noise = tf.sqrt(N0 / 2) * standart_gaussian_noise_layer((batch_size, K ,L * 2))
    y_noise = y_real + noise                                                    #[batch_size, K, 2*L]
    y_noise2 = tf.reshape(y_noise,[batch_size, 2 * K * L])

    #量化DNN，输入y_noise，输出q
    u1 = (normal_full_layer(y_noise2, 1024, 'u1'))                     #第一个全连接层，u1是全连接层输出
    n_u1 = tf.nn.relu(tf.layers.batch_normalization(u1, name='u1_normalization'))       #n_u1是u1加上relu激活函数
    u2 = (normal_full_layer(n_u1, 512, 'u2'))                         #第二个全连接层
    n_u2 = tf.nn.relu(tf.layers.batch_normalization(u2, name='u2_normalization'))
    u3 = (normal_full_layer(n_u2, 256, 'u3'))                         #第三个全连接层
    n_u3 = tf.nn.relu(tf.layers.batch_normalization(u3, name='u3_normalization'))
    q = tf.sign(alpha_para * normal_full_layer(n_u3, K * B, 'u4'))          #第四个全连接层，激活函数为sign，输出q[batch_size, K, B]

    #信道估计DNN，输入q，输出v_BF
    r1 = (normal_full_layer(q, 1024, 'r1'))                             #第五个全连接层（下半部分开始）
    n_r1 = tf.nn.relu(tf.layers.batch_normalization(r1, name='r1_normalization'))
    r2 = (normal_full_layer(n_r1, 512, 'r2'))                           #第六个全连接层
    n_r2 = tf.nn.relu(tf.layers.batch_normalization(r2, name='r2_normalization'))
    r3 = (normal_full_layer(n_r2, 512, 'r3'))                           #第七个全连接层
    n_r3 = tf.nn.relu(tf.layers.batch_normalization(r3, name='r3_normalization'))
    v_BF = normal_full_layer(n_r3, K * M * 2, 'r4')                             #第八个全连接层，没有激活函数
    v_BF2 = tf.reshape(v_BF, [batch_size, K, 2 * M])
    norm_v0 = tf.square(v_BF2[:, :, M:2 * M]) + tf.square(v_BF2[:,:, 0:M])
    norm_v = tf.sqrt(tf.reduce_sum(tf.reduce_sum(norm_v0, axis=1, keepdims=True), axis=2, keepdims=True))
    v_BF_complex = tf.complex(v_BF2[:, :, 0:M]/norm_v, v_BF2[:, :, M:2 * M]/norm_v)
    return v_BF_complex, X_tilde               #v_BF_complex:[batch_size, K, M],   X_tilde:[M, 2*L],  均为实数


def HyperDop_BF(X,  N0, batch_size, M,  B, L, K, S, v_sat):
    #X:真实信道信息，[batch_size, K, M]
    # v_sat:[batch_size, K, 1]

    ##################################     HYPER  DNN     ####################################
    U1, B1 = normal_full_layer2(1, 32, 'R1')
    U2, B2 = normal_full_layer2(32, 64, 'R2')
    U3, B3 = normal_full_layer2(64, B, 'R4')
    hyper_temp1 = tf.nn.relu(tf.matmul(v_sat, U1) + B1)  # batch_size, K, 32
    hyper_temp2 = tf.nn.relu(tf.matmul(hyper_temp1, U2) + B2)  # batch_size, K, 64
    hyper_y = tf.reshape(tf.matmul(hyper_temp2, U3) + B3, [batch_size,K * B])  # batch_size, K, B

    ###################################          MAIN           #################################
    #导频信号初始化
    X_tilde_ini = tf.Variable(tf.sqrt(1 / M) * standart_gaussian_noise_layer([M, 2 * L]), trainable=True, name='x')     #[M, 2 * L]
    power_normal = tf.sqrt(tf.reduce_sum(tf.square(X_tilde_ini[:, 0:L]) + tf.square(X_tilde_ini[:, L:2 * L]), axis=0))  #[L]
    X_tilde = X_tilde_ini / (tf.concat([power_normal, power_normal], axis=0))           #归一化，[M, 2 * L]
    X_tilde_complex = tf.reshape(tf.complex(X_tilde[:, 0:L], X_tilde[:, L:2 * L]), [1,1, M, L])  # M,L

    X_tilde_complex1_full = tf.tile(X_tilde_complex, [batch_size,K, 1, 1])  # (batch_size,K,M,L)

    #计算接收信号y
    # (batch_size,K,S,M) (batch_size,K,M,L) =(batch_size, K, S, L)
    y = tf.matmul(X, X_tilde_complex1_full)                                          #y=h*x， y:(batch_size, K, S, L)
    y_real = tf.concat([tf.real(y), tf.imag(y)], axis=3)                                #(batch_size, K, S, 2*L)
    noise = tf.sqrt(N0 / 2) * standart_gaussian_noise_layer((batch_size, K, S, L * 2))
    y_noise = y_real + noise                                                    #(batch_size, K, S, 2*L)
    y_noise2 = tf.reshape(y_noise,[batch_size, S , 2 * K * L])

    #量化DNN，输入y_noise，输出q
    u1, ub1 = normal_full_layer2(2 * K * L, 256, 'r1')
    u2, ub2 = normal_full_layer2(256, 256, 'r2')
    u3, ub3 = normal_full_layer2(256, 128, 'r3')
    u4, ub4 = normal_full_layer2(128, K * B, 'r4')
    nu1 = tf.nn.relu(tf.matmul(y_noise2[:, 0, :], u1) + ub1)
    nu2 = tf.nn.relu(tf.matmul(nu1, u2) + ub2)
    nu3 = tf.nn.relu(tf.matmul(nu2, u3) + ub3)
    q = tf.sign(tf.matmul(nu3, u4) + ub4)       #(batch_size, K*B)
    q2 = (2 * q - 1) * hyper_y
    #波束赋形DNN，输入q，输出v_BF
    U2, B2 = normal_full_layer2(K*B, 256, 'U2')
    W2 = normal_full_layer2_nobias(256, 256, 'H2')
    V2, C2 = normal_full_layer2(256, K * M * 2, 'V2')
    H2 = tf.nn.relu(tf.matmul(q2, U2) + B2)
    v_BF = tf.reshape(tf.matmul(H2, V2) + C2, [batch_size, K, 1, M * 2])

    for s in range(S-1):
        nu1 = tf.nn.relu(tf.matmul(y_noise2[:, s + 1, :], u1) + ub1)
        nu2 = tf.nn.relu(tf.matmul(nu1, u2) + ub2)
        nu3 = tf.nn.relu(tf.matmul(nu2, u3) + ub3)
        q = tf.sign(tf.matmul(nu3, u4) + ub4)
        q2 = (2 * q - 1) * hyper_y
        H2 = tf.nn.relu(tf.matmul(q2, U2) + B2 + tf.matmul(H2, W2))
        v_BF_temp = tf.reshape(tf.matmul(H2, V2) + C2, [batch_size, K, 1, M * 2])
        v_BF = tf.concat([v_BF, v_BF_temp], axis=2)
    norm_v0 = tf.square(v_BF[:, :,:, M:2 * M]) + tf.square(v_BF[:, :,:, 0:M])  #[batch_size, K, S, M * 2]
    norm_v = tf.sqrt(tf.reduce_sum(tf.reduce_sum(norm_v0, axis=1, keepdims=True), axis=3, keepdims=True))
    v_BF_complex = tf.complex(v_BF[:, :,:, 0:M]/norm_v , v_BF[:, :,:, M:2 * M]/norm_v)
    return v_BF_complex              #v_BF_complex:[batch_size, K, S, M ])

def ZF_BF(X,  N0, batch_size, M,  B, L, K, alpha_para):
    #X:真实信道信息，[batch_size, K, M]

    #导频信号初始化
    X_tilde_ini = tf.Variable(tf.sqrt(1 / M) * standart_gaussian_noise_layer([M, 2 * L]), trainable=True, name='x')     #[M, 2 * L]
    power_normal = tf.sqrt(tf.reduce_sum(tf.square(X_tilde_ini[:, 0:L]) + tf.square(X_tilde_ini[:, L:2 * L]), axis=0))  #[L]
    X_tilde = X_tilde_ini / (tf.concat([power_normal, power_normal], axis=0))           #归一化，[M, 2 * L]
    X_tilde_complex = tf.complex(X_tilde[:, 0:L], X_tilde[:, L:2 * L])                  #待训练的导频信号[M,L]

    #计算接收信号y
    y = tf.matmul(X, X_tilde_complex)                                                   #y=h*x， y:[batch_size, K, L]
    y_real = tf.concat([tf.real(y), tf.imag(y)], axis=2)                                #[batch_size, K, 2*L]
    noise = tf.sqrt(N0 / 2) * standart_gaussian_noise_layer((batch_size, K ,L * 2))
    y_noise = y_real + noise                                                    #[batch_size, K, 2*L]
    y_noise2 = tf.reshape(y_noise,[batch_size, 2 * K * L])

    #量化DNN，输入y_noise，输出q
    u1 = (normal_full_layer(y_noise2, 1024, 'u1'))                     #第一个全连接层，u1是全连接层输出
    n_u1 = tf.nn.relu(tf.layers.batch_normalization(u1, name='u1_normalization'))       #n_u1是u1加上relu激活函数
    u2 = (normal_full_layer(n_u1, 512, 'u2'))                         #第二个全连接层
    n_u2 = tf.nn.relu(tf.layers.batch_normalization(u2, name='u2_normalization'))
    u3 = (normal_full_layer(n_u2, 256, 'u3'))                         #第三个全连接层
    n_u3 = tf.nn.relu(tf.layers.batch_normalization(u3, name='u3_normalization'))
    q = tf.sign(alpha_para * normal_full_layer(n_u3, K * B, 'u4'))          #第四个全连接层，激活函数为sign，输出q[batch_size, K, B]


    #DNN，输入q，输出h_est2
    r1 = (normal_full_layer(q, 1024, 'r1'))                             #第五个全连接层（下半部分开始）
    n_r1 = tf.nn.relu(tf.layers.batch_normalization(r1, name='r1_normalization'))
    r2 = (normal_full_layer(n_r1, 512, 'r2'))                           #第六个全连接层
    n_r2 = tf.nn.relu(tf.layers.batch_normalization(r2, name='r2_normalization'))
    r3 = (normal_full_layer(n_r2, 512, 'r3'))                           #第七个全连接层
    n_r3 = tf.nn.relu(tf.layers.batch_normalization(r3, name='r3_normalization'))
    h_est0 = normal_full_layer(n_r3, K * M * 2, 'r4')                             #第八个全连接层，没有激活函数
    h_est = tf.reshape(h_est0, [batch_size, K, 2 * M])
    h_est_complex = tf.complex(h_est[:,:,0:M] , h_est[:,:,M:2*M])

    return h_est_complex               #h_est_complex:[batch_size, K, M], W:(batch_size, K, M)



