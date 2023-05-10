import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.keras.layers import BatchNormalization, Dense
import os
import scipy.io as sio

tf.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, dims, name):
        super(MLPBlock, self).__init__()
        self.layers = []
        self.num_layers = num_layers
        for ii in range(num_layers - 1):
            self.layers.append(Dense(units=dims[ii], activation='relu', name=name + '_relu_' + str(ii)))
            self.layers.append(BatchNormalization())
        self.layers.append(Dense(units=dims[-1], activation='linear', name=name + '_linear'))

    def call(self, inputs, **kwargs):
        x = inputs
        for ii in range(len(self.layers)):
            if ii == 0:
                x = self.layers[ii](inputs)
            else:
                x = self.layers[ii](x)
        return x


'System Information'
N1 = 64  # Number of BS's antennas
N2 = 32  # Number of UE's antennas
tau1 = 4
tau2 = 3
L = 3  # number of path

'Channel Information'
phi_min = -60 * (np.pi / 180)  # Lower-bound of AoAs
phi_max = 60 * (np.pi / 180)  # Upper-bound of AoAs
snrdB = 0  # SNRs
P_snr = 10 ** (snrdB / 10)  # Considered TX powers
mean_true_alpha = 0.0 + 0.0j  # Mean of the fading coefficient
std_per_dim_alpha = np.sqrt(0.5)  # STD of the Gaussian fading coefficient per real dim.
noiseSTD_per_dim = np.sqrt(0.5)  # STD of the Gaussian noise per real dim.
#####################################################
'Learning Parameters'
initial_run = 0  # 0: Continue training; 1: Starts from the scratch
n_epochs = 0  # Num of epochs
learning_rate = 0.0001  # Learning rate
batch_per_epoch = 400  # Number of mini batches per epoch
batch_size_train = 1024  # Mini_batch_size = batch_size_order*delta_inv
batch_size_val = 10000  # Scaling the number of tests
model_path = '../params/DNN_trainable_w_v3_L_SNR_tau1_tau2_N1_N2_' + str((L, snrdB, tau1, tau2, N1, N2))
######################################################
tf.reset_default_graph()  # Reseting the graph
he_init = tf.variance_scaling_initializer()  # Define initialization method
######################################## Place Holders
alpha_input = tf.placeholder(tf.complex64, shape=(None, L), name="alpha_input")
phi_input_1 = tf.placeholder(tf.float32, shape=(None, L), name="phi_input1")
phi_input_2 = tf.placeholder(tf.float32, shape=(None, L), name="phi_input2")
batch_size = tf.shape(alpha_input)[0]
##################### NETWORK
with tf.name_scope("array_response_construction"):
    lay = {}
    lay['P'] = tf.constant(1.0)
    for ii in range(L):
        phi_1 = tf.reshape(phi_input_1[:, ii], [-1, 1])
        from0toN = tf.cast(tf.range(0, N1, 1), tf.float32)
        phi_expanded = tf.tile(phi_1, (1, N1))
        a_phi_1 = (tf.exp(1j * np.pi * tf.cast(tf.multiply(tf.sin(phi_expanded), from0toN), tf.complex64)))
        a_phi_1 = tf.expand_dims(a_phi_1, axis=-1)

        phi_2 = tf.reshape(phi_input_2[:, ii], [-1, 1])
        from0toN = tf.cast(tf.range(0, N2, 1), tf.float32)
        phi_expanded = tf.tile(phi_2, (1, N2))
        a_phi_2 = (tf.exp(1j * np.pi * tf.cast(tf.multiply(tf.sin(phi_expanded), from0toN), tf.complex64)))
        a_phi_2 = tf.expand_dims(a_phi_2, axis=-1)

        G_i = a_phi_1 @ tf.transpose(tf.conj(a_phi_2), perm=(0, 2, 1))
        if ii == 0:
            G = tf.reshape(alpha_input[:, ii], [-1, 1, 1]) * G_i
        else:
            G = G + tf.reshape(alpha_input[:, ii], [-1, 1, 1]) * G_i
    G = G / np.sqrt(L)

    s, u, v = tf.linalg.svd(G)
    v_opt = tf.expand_dims(u[:, :, 0], axis=2)
    v_opt = v_opt / tf.norm(v_opt, axis=1, keepdims=True)
    w_opt = tf.expand_dims(v[:, :, 0], axis=2)
    w_opt = w_opt / tf.norm(w_opt, axis=1, keepdims=True)
    tmp_opt = tf.transpose(tf.conj(v_opt), perm=[0, 2, 1]) @ G @ w_opt
    bf_gain_opt = tf.reduce_mean(tf.abs(tmp_opt) ** 2)
    bf_gain_opt = tf.real(bf_gain_opt)

with tf.name_scope("channel_sensing"):
    V_real = tf.get_variable('V_real', shape=[1, N1, tau1])
    V_imag = tf.get_variable('V_imag', shape=[1, N1, tau1])
    V_tf = tf.complex(V_real, V_imag)
    V_tf = V_tf / tf.norm(V_tf, axis=1, keepdims=True)

    W_real = tf.get_variable('W_real', shape=[1, N2, tau2])
    W_imag = tf.get_variable('W_imag', shape=[1, N2, tau2])
    W_tf = tf.complex(W_real, W_imag)
    W_tf = W_tf / tf.norm(W_tf, axis=1, keepdims=True)

    Y_noiseless = tf.transpose(tf.conj(W_tf), perm=[0, 2, 1]) @ tf.transpose(tf.conj(G), perm=[0, 2, 1]) @ V_tf
    noise = tf.complex(tf.random_normal(tf.shape(Y_noiseless), mean=0.0, stddev=noiseSTD_per_dim), \
                       tf.random_normal(tf.shape(Y_noiseless), mean=0.0, stddev=noiseSTD_per_dim))
    Y_complex = tf.complex(tf.sqrt(lay['P']), 0.0) * Y_noiseless + noise
    y_complex_vec = tf.reshape(Y_complex, [-1, tau1 * tau2])
    y_real = tf.concat([tf.real(y_complex_vec), tf.imag(y_complex_vec)], axis=1)

    'caculate bf_gain'
    MLP_bf1 = MLPBlock(3, [1024, 1024, 2 * N1], name='mlp_bf1')
    MLP_bf2 = MLPBlock(3, [1024, 1024, 2 * N2], name='mlp_bf2')

    v_tmp = MLP_bf1(y_real)
    v_norm = tf.reshape(tf.norm(v_tmp, axis=1), (-1, 1))
    v_tmp = tf.divide(v_tmp, v_norm)
    v_complex = tf.complex(v_tmp[:, 0:N1], v_tmp[:, N1:2 * N1])
    v_complex = tf.reshape(v_complex, [-1, N1, 1])

    w_tmp = MLP_bf2(y_real)
    w_norm = tf.reshape(tf.norm(w_tmp, axis=1), (-1, 1))
    w_tmp = tf.divide(w_tmp, w_norm)
    w_complex = tf.complex(w_tmp[:, 0:N2], w_tmp[:, N2:2 * N2])
    w_complex = tf.reshape(w_complex, [-1, N2, 1])

    bf_gain = tf.reduce_mean(tf.abs(tf.transpose(tf.conj(v_complex), perm=[0, 2, 1]) @ G @ w_complex) ** 2)

####################################################################################
####### Loss Function
loss = -bf_gain
####### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()
#########################################################################
###########  Validation Set
# alpha_val = np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_val, L]) \
#             + 1j * np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_val, L])
# phi_1_val = np.random.uniform(low=phi_min, high=phi_max, size=[batch_size_val, L])
# phi_2_val = np.random.uniform(low=phi_min, high=phi_max, size=[batch_size_val, L])

test_data = sio.loadmat('../test_data.mat')
alpha_val = test_data['alpha_val']
phi_1_val = test_data['phi_1_val']
phi_2_val = test_data['phi_2_val']

feed_dict_val = {alpha_input: alpha_val,
                 phi_input_1: phi_1_val,
                 phi_input_2: phi_2_val,
                 lay['P']: P_snr}
###########  Training
with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, model_path)
    best_loss, opt_loss, w_Tx_final,w_Rx_final,G_np = sess.run([loss, bf_gain_opt, v_complex,w_complex ,G], feed_dict=feed_dict_val)
    print(-best_loss, opt_loss)
    print(tf.test.is_gpu_available())  # Prints whether or not GPU is on

    sio.savemat('DNN_trainable_interpretation_tau_' + str(tau1)+str(tau2) + '.mat',
                {'bf_gain_dB': (10 * np.log10(-best_loss)), 'bf_gain_opt_dB': (10 * np.log10(opt_loss)),
                 'N1_N2_tau_L': (N1, N2, tau1,tau2, L),
                 'phi_min_max': (phi_min, phi_max), 'snrdB': snrdB,
                 'phi_1_val': phi_1_val, 'phi_2_val': phi_2_val, 'alpha_val': alpha_val,
                 'w_Tx_final': w_Tx_final,'w_Rx_final':w_Rx_final, 'G': G_np})
