import os

import numpy as np
import scipy.io as sio
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import BatchNormalization, Dense

tf.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
Tx and Rx controls RIS. Beamforming vectors and reflection coefficients are trainable parameters. 
'''


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
M1 = 64  # Number of BS's antennas
M2 = 32  # Number of UE's antennas
N1, N2 = 8, 8
N = N1 * N2  # Number of RIS elements
tau = 4  # Communication rounds
L1 = 2  # number of path between BS and RIS
L2 = 3  # number of path between RIS and UE

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
initial_run = 1  # 0: Continue training; 1: Starts from the scratch
n_epochs = 10  # Num of epochs
learning_rate = 0.0001  # Learning rate
batch_per_epoch = 400  # Number of mini batches per epoch
batch_size_train = 1024
batch_size_val = 10000
model_path = './params/DNN_trainable_sensing_v2_L1_L2_SNR_tau_N1_N2_' + str((L1, L2, snrdB, tau, M1, M2))
######################################################
tf.reset_default_graph()  # Reseting the graph
he_init = tf.variance_scaling_initializer()  # Define initialization method
######################################## Place Holders
alpha_input1 = tf.placeholder(tf.complex64, shape=(None, L1), name="alpha_input1")  # fading coefficients
alpha_input2 = tf.placeholder(tf.complex64, shape=(None, L2), name="alpha_input2")  # fading coefficients
phi_input_1 = tf.placeholder(tf.float32, shape=(None, L1, 4), name="phi_input1")  # AoAs and AoDs between RIS and Tx
phi_input_2 = tf.placeholder(tf.float32, shape=(None, L2, 4), name="phi_input2")  # AoAs and AoDs between RIS and Rx
batch_size = tf.shape(alpha_input1)[0]
##################### NETWORK
with tf.name_scope("array_response_construction"):
    lay = {}
    lay['P'] = tf.constant(1.0)
    for ii in range(L1):
        phi_bs = tf.reshape(phi_input_1[:, ii, 0], [-1, 1])
        theta_bs = tf.reshape(phi_input_1[:, ii, 1], [-1, 1])
        phi_ris = tf.reshape(phi_input_1[:, ii, 2], [-1, 1])
        theta_ris = tf.reshape(phi_input_1[:, ii, 3], [-1, 1])

        a_bs = tf.exp(
            1j * np.pi * tf.cast(tf.cos(phi_bs) * tf.cos(theta_bs) * tf.range(M1, dtype=tf.float32), tf.complex64))
        a_bs = tf.expand_dims(a_bs, axis=-1)

        i1 = tf.mod(tf.range(N1 * N2, dtype=tf.float32), N1)
        i2 = tf.floor(tf.range(N1 * N2, dtype=tf.float32) / N1)
        a_ris = tf.exp(
            1j * np.pi * tf.cast(i1 * tf.sin(phi_ris) * tf.cos(theta_ris) + i2 * tf.sin(theta_ris), tf.complex64))
        a_ris = tf.expand_dims(a_ris, axis=-1)

        A_i = a_bs @ tf.transpose(tf.conj(a_ris), perm=(0, 2, 1))
        if ii == 0:
            A = tf.reshape(alpha_input1[:, ii], [-1, 1, 1]) * A_i
        else:
            A = A + tf.reshape(alpha_input1[:, ii], [-1, 1, 1]) * A_i
    A = A / np.sqrt(L1)

    for ii in range(L2):
        phi_bs = tf.reshape(phi_input_2[:, ii, 0], [-1, 1])
        theta_bs = tf.reshape(phi_input_2[:, ii, 1], [-1, 1])
        phi_ris = tf.reshape(phi_input_2[:, ii, 2], [-1, 1])
        theta_ris = tf.reshape(phi_input_2[:, ii, 3], [-1, 1])

        a_ue = tf.exp(
            1j * np.pi * tf.cast(tf.cos(phi_bs) * tf.cos(theta_bs) * tf.range(M2, dtype=tf.float32), tf.complex64))
        a_ue = tf.expand_dims(a_ue, axis=-1)

        i1 = tf.mod(tf.range(N1 * N2, dtype=tf.float32), N1)
        i2 = tf.floor(tf.range(N1 * N2, dtype=tf.float32) / N1)
        a_ris = tf.exp(
            1j * np.pi * tf.cast(i1 * tf.sin(phi_ris) * tf.cos(theta_ris) + i2 * tf.sin(theta_ris), tf.complex64))
        a_ris = tf.expand_dims(a_ris, axis=-1)

        B_i = a_ris @ tf.transpose(tf.conj(a_ue), perm=(0, 2, 1))
        if ii == 0:
            B = tf.reshape(alpha_input2[:, ii], [-1, 1, 1]) * B_i
        else:
            B = B + tf.reshape(alpha_input2[:, ii], [-1, 1, 1]) * B_i
    B = B / np.sqrt(L2)

    #####################optimal beamforming vectors with random reflection################
    v = tf.complex(tf.random_normal(shape=[batch_size, N]), tf.random_normal(shape=[batch_size, N]))
    v = v / tf.cast(tf.abs(v), tf.complex64)
    G = A @ tf.linalg.diag(v) @ B
    s, u_svd, v_svd = tf.linalg.svd(G)
    v_opt = tf.expand_dims(u_svd[:, :, 0], axis=2)
    v_opt = v_opt / tf.norm(v_opt, axis=1, keepdims=True)
    w_opt = tf.expand_dims(v_svd[:, :, 0], axis=2)
    w_opt = w_opt / tf.norm(w_opt, axis=1, keepdims=True)
    tmp_opt = tf.transpose(tf.conj(v_opt), perm=[0, 2, 1]) @ G @ w_opt
    bf_gain_opt = tf.reduce_mean(tf.abs(tmp_opt) ** 2)
    bf_gain_opt = tf.real(bf_gain_opt)

with tf.name_scope("channel_sensing"):
    MLP_user1_transmit = MLPBlock(3, [512, 512, 2 * M1], name='mlp_user1_transmit')
    MLP_user1_receive = MLPBlock(3, [512, 512, 2 * M1], name='mlp_user1_receive')
    MLP_user2_transmit = MLPBlock(3, [512, 512, 2 * M2], name='mlp_user2_transmit')
    MLP_user2_receive = MLPBlock(3, [512, 512, 2 * M2], name='mlp_user2_receive')
    MLP_RIS = MLPBlock(3, [512, 512, 2 * N1 * N2], name='mlp_RIS')

    for t in range(tau):
        'sensing vectors w_a2 and w_b1'
        w_a_uplink_real = tf.get_variable("wa_uplink_real2" + str(t), shape=(1, M1, 1), trainable=True)
        w_a_uplink_imag = tf.get_variable("wa_uplink_imag2" + str(t), shape=(1, M1, 1), trainable=True)
        w_a_complex = tf.complex(w_a_uplink_real, w_a_uplink_imag)
        w_a2 = w_a_complex / tf.norm(w_a_complex, axis=1, keepdims=True)

        w_b_uplink_real = tf.get_variable("wb_uplink_real1" + str(t), shape=(1, M2, 1), trainable=True)
        w_b_uplink_imag = tf.get_variable("wb_uplink_imag1" + str(t), shape=(1, M2, 1), trainable=True)
        w_b_complex = tf.complex(w_b_uplink_real, w_b_uplink_imag)
        w_b1 = w_b_complex / tf.norm(w_b_complex, axis=1, keepdims=True)

        'sensing vectors v'
        v_uplink_real = tf.get_variable("v_uplink_real1" + str(t), shape=(1, N), trainable=True)
        v_uplink_imag = tf.get_variable("v_uplink_imag1" + str(t), shape=(1, N), trainable=True)
        v_complex = tf.complex(v_uplink_real, v_uplink_imag)
        v = v_complex / tf.cast(tf.abs(v_complex), tf.complex64)
        G = A @ tf.linalg.diag(v) @ B

        'Rx observes the next measurement'
        y_noiseless2 = tf.transpose(tf.conj(w_b1), perm=[0, 2, 1]) @ tf.transpose(tf.conj(G), perm=[0, 2, 1]) @ w_a2
        noise2 = tf.complex(tf.random_normal(tf.shape(y_noiseless2), mean=0.0, stddev=noiseSTD_per_dim), \
                            tf.random_normal(tf.shape(y_noiseless2), mean=0.0, stddev=noiseSTD_per_dim))
        y_complex2 = tf.complex(tf.sqrt(lay['P']), 0.0) * y_noiseless2 + noise2
        y_complex2 = tf.reshape(y_complex2, [-1, 1])
        y_real2 = tf.concat([tf.real(y_complex2), tf.imag(y_complex2)], axis=1) / tf.sqrt(lay['P'])
        if t == 0:
            y2_all = y_real2
        else:
            y2_all = tf.concat([y2_all, y_real2], axis=-1)

        'sensing vectors w_a1 and w_b2'
        w_a_uplink_real = tf.get_variable("wa_uplink_real1" + str(t), shape=(1, M1, 1), trainable=True)
        w_a_uplink_imag = tf.get_variable("wa_uplink_imag1" + str(t), shape=(1, M1, 1), trainable=True)
        w_a_complex = tf.complex(w_a_uplink_real, w_a_uplink_imag)
        w_a1 = w_a_complex / tf.norm(w_a_complex, axis=1, keepdims=True)

        w_b_uplink_real = tf.get_variable("wb_uplink_real2" + str(t), shape=(1, M2, 1), trainable=True)
        w_b_uplink_imag = tf.get_variable("wb_uplink_imag2" + str(t), shape=(1, M2, 1), trainable=True)
        w_b_complex = tf.complex(w_b_uplink_real, w_b_uplink_imag)
        w_b2 = w_b_complex / tf.norm(w_b_complex, axis=1, keepdims=True)

        'sensing vectors v'
        v_uplink_real = tf.get_variable("v_uplink_real2" + str(t), shape=(1, N), trainable=True)
        v_uplink_imag = tf.get_variable("v_uplink_imag2" + str(t), shape=(1, N), trainable=True)
        v_complex = tf.complex(v_uplink_real, v_uplink_imag)
        v = v_complex / tf.cast(tf.abs(v_complex), tf.complex64)
        G = A @ tf.linalg.diag(v) @ B

        'Tx observes the next measurement'
        y_noiseless1 = tf.transpose(tf.conj(w_a1), perm=[0, 2, 1]) @ G @ w_b2
        noise1 = tf.complex(tf.random_normal(tf.shape(y_noiseless1), mean=0.0, stddev=noiseSTD_per_dim), \
                            tf.random_normal(tf.shape(y_noiseless1), mean=0.0, stddev=noiseSTD_per_dim))
        y_complex1 = tf.complex(tf.sqrt(lay['P']), 0.0) * y_noiseless1 + noise1
        y_complex1 = tf.reshape(y_complex1, [-1, 1])
        y_real1 = tf.concat([tf.real(y_complex1), tf.imag(y_complex1)], axis=1) / tf.sqrt(lay['P'])

        if t == 0:
            y1_all = y_real1
        else:
            y1_all = tf.concat([y1_all, y_real1], axis=-1)

    'caculate bf_gain'
    MLP_bf1 = MLPBlock(3, [1024, 1024, 2 * M1], name='mlp_bf1')
    MLP_bf2 = MLPBlock(3, [1024, 1024, 2 * M2], name='mlp_bf2')
    MLP_bf3 = MLPBlock(3, [1024, 1024, 2 * N], name='mlp_bf3')

    w_a = MLP_bf1(y1_all)
    w_a_norm = tf.reshape(tf.norm(w_a, axis=1), (-1, 1))
    w_a = tf.divide(w_a, w_a_norm)
    w_a_complex = tf.complex(w_a[:, 0:M1], w_a[:, M1:2 * M1])
    w_a_complex = tf.reshape(w_a_complex, [-1, M1, 1])

    v_tmp = MLP_bf3(tf.concat([y1_all, y2_all], axis=1))
    v_complex = tf.complex(v_tmp[:, 0:N], v_tmp[:, N:2 * N])
    v_complex = v_complex / tf.cast(tf.abs(v_complex), tf.complex64)
    v_complex = tf.reshape(v_complex, [-1, N])

    G = A @ tf.linalg.diag(v_complex) @ B

    w_b = MLP_bf2(y2_all)
    w_b_norm = tf.reshape(tf.norm(w_b, axis=1), (-1, 1))
    w_b = tf.divide(w_b, w_b_norm)
    w_b_complex = tf.complex(w_b[:, 0:M2], w_b[:, M2:2 * M2])
    w_b_complex = tf.reshape(w_b_complex, [-1, M2, 1])

    bf_gain = tf.reduce_mean(tf.abs(tf.transpose(tf.conj(w_a_complex), perm=[0, 2, 1]) @ G @ w_b_complex) ** 2)

####### Loss Function
loss = -bf_gain
####### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()
#########################################################################
###########  Validation Set
alpha1_val = np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_val, L1]) \
             + 1j * np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_val, L1])
alpha2_val = np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_val, L2]) \
             + 1j * np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_val, L2])
phi_1_val = np.random.uniform(low=phi_min, high=phi_max, size=[batch_size_val, L1, 4])
phi_2_val = np.random.uniform(low=phi_min, high=phi_max, size=[batch_size_val, L2, 4])

# test_data = sio.loadmat('test_data.mat')
# alpha1_val = test_data['alpha1_val']
# alpha2_val = test_data['alpha2_val']
# phi_1_val = test_data['phi_1_val']
# phi_2_val = test_data['phi_2_val']

feed_dict_val = {alpha_input1: alpha1_val, alpha_input2: alpha2_val, phi_input_1: phi_1_val, phi_input_2: phi_2_val,
                 lay['P']: P_snr}
###########  Training
with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, model_path)
    best_loss, opt_loss = sess.run([loss, bf_gain_opt], feed_dict=feed_dict_val)
    print(10 * np.log10(-best_loss), 10 * np.log10(opt_loss))
    print(tf.test.is_gpu_available())  # Prints whether or not GPU is on
    no_increase = 0
    for epoch in range(n_epochs):
        batch_iter = 0
        for rnd_indices in range(batch_per_epoch):
            alpha1_train = np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha,
                                            size=[batch_size_train, L1]) \
                           + 1j * np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha,
                                                   size=[batch_size_train, L1])
            alpha2_train = np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha,
                                            size=[batch_size_train, L2]) \
                           + 1j * np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha,
                                                   size=[batch_size_train, L2])
            phi_1_train = np.random.uniform(low=phi_min, high=phi_max, size=[batch_size_train, L1, 4])
            phi_2_train = np.random.uniform(low=phi_min, high=phi_max, size=[batch_size_train, L2, 4])
            feed_dict_train = {alpha_input1: alpha1_train, alpha_input2: alpha2_train, phi_input_1: phi_1_train,
                               phi_input_2: phi_2_train, lay['P']: P_snr}

            sess.run(training_op, feed_dict=feed_dict_train)
            batch_iter += 1
        loss_val = sess.run(loss, feed_dict=feed_dict_val)
        print('epoch', epoch, '  loss_test:%2.5f' % (10 * np.log10(-loss_val)),
              ' dB:%2.3f' % (10 * np.log10(-best_loss)),
              '  opt_test:%2.3f' % (10 * np.log10(opt_loss)), 'no_increase:', no_increase)
        if epoch % 10 == 0:  # Every 10 iterations it checks if the validation performace is improved, then saves parameters
            if loss_val < best_loss:
                save_path = saver.save(sess, model_path)
                best_loss = loss_val
                no_increase = 0
            else:
                no_increase = no_increase + 10
        if no_increase > 50:
            break
    #
    # sio.savemat('./results/DNN_trainable_sensing_both_control_RIS' + str(tau) + '.mat',
    #             {'bf_gain_opt_dB': 10 * np.log10(-best_loss),
    #              'bf_gain_random_RIS_optimal_beamform_dB': 10 * np.log10(opt_loss),
    #              'M1_M2_N1_N2_L1_L2_phi_min_max': (M1, M2, N1, N2, L1, L2, phi_min, phi_max),
    #              'snr_dB': snrdB,
    #              'communication_rounds': tau})
