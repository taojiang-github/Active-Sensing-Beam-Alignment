import os
import numpy as np
import scipy.io as sio
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import BatchNormalization, Dense

tf.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

'''
Tx and Rx control RIS. Beamforming vectors and reflection coefficients are actively updated. The RIS is updated based 
on the state information from both side.
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


class RNN(tf.keras.layers.Layer):
    def __init__(self, hidden_size, name):
        super(RNN, self).__init__()
        self.layer_Ui = Dense(units=hidden_size, activation='linear', name='Ui' + name)
        self.layer_Wi = Dense(units=hidden_size, activation='linear', name='Wi' + name)
        self.layer_Uf = Dense(units=hidden_size, activation='linear', name='Uf' + name)
        self.layer_Wf = Dense(units=hidden_size, activation='linear', name='Wf' + name)
        self.layer_Uo = Dense(units=hidden_size, activation='linear', name='Uo' + name)
        self.layer_Wo = Dense(units=hidden_size, activation='linear', name='Wo' + name)
        self.layer_Uc = Dense(units=hidden_size, activation='linear', name='Uc' + name)
        self.layer_Wc = Dense(units=hidden_size, activation='linear', name='Wc' + name)

    def call(self, inputs, **kwargs):
        (input_x, h_old, c_old) = inputs
        i_t = tf.sigmoid(self.layer_Ui(input_x) + self.layer_Wi(h_old))
        f_t = tf.sigmoid(self.layer_Uf(input_x) + self.layer_Wf(h_old))
        o_t = tf.sigmoid(self.layer_Uo(input_x) + self.layer_Wo(h_old))
        c_t = tf.tanh(self.layer_Uc(input_x) + self.layer_Wc(h_old))
        c = i_t * c_t + f_t * c_old
        h_new = o_t * tf.tanh(c)
        return h_new, c


'System Information'
M1 = 64  # Number of BS's antennas
M2 = 32  # Number of UE's antennas
N1, N2 = 8, 8
N = N1 * N2  # Number of RIS elements
tau = 8  # Communication rounds
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
initial_run = 0  # 0: Continue training; 1: Starts from the scratch
n_epochs = 0  # Num of epochs
learning_rate = 0.00001  # Learning rate
batch_per_epoch = 400  # Number of mini batches per epoch
batch_size_train = 1024
batch_size_val = 20
model_path = '../params/active_learning_RNN_v4_L1_L2_SNR_tau_N1_N2_' + str((L1, L2, snrdB, tau, M1, M2))
######################################################
tf.reset_default_graph()  # Reseting the graph
he_init = tf.variance_scaling_initializer()  # Define initialization method
######################################## Place Holders
alpha_input1 = tf.placeholder(tf.complex64, shape=(None, L1), name="alpha_input1")  # fading coefficients
alpha_input2 = tf.placeholder(tf.complex64, shape=(None, L2), name="alpha_input2")  # fading coefficients
phi_input_1 = tf.placeholder(tf.float32, shape=(None, L1, 4), name="phi_input1")  # AoAs and AoDs between RIS and BS
phi_input_2 = tf.placeholder(tf.float32, shape=(None, L2, 4), name="phi_input2")  # AoAs and AoDs between RIS and UE
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
    hidden_size1, hidden_size2 = 512, 256
    RNN1 = RNN(hidden_size1, 'rnn_1')
    RNN2 = RNN(hidden_size2, 'rnn_2')
    MLP_user1_transmit = MLPBlock(3, [512, 512, 2 * M1], name='mlp_user1_transmit')
    MLP_user1_receive = MLPBlock(3, [512, 512, 2 * M1], name='mlp_user1_receive')
    MLP_user2_transmit = MLPBlock(3, [512, 512, 2 * M2], name='mlp_user2_transmit')
    MLP_user2_receive = MLPBlock(3, [512, 512, 2 * M2], name='mlp_user2_receive')
    MLP_RIS1 = MLPBlock(3, [512, 512, 2 * N1 * N2], name='mlp_RIS1')
    MLP_RIS2 = MLPBlock(3, [512, 512, 2 * N1 * N2], name='mlp_RIS2')

    beamformer_list = [[], [], [], []]
    ris_list = [[],[]]
    ris_final = []
    beamformer_final = [[], []]
    G_list = [[],[],[],[]]
    G_list[2].append(G)
    for t in range(tau):
        'initailization'
        if t == 0:
            h_old1 = tf.zeros([batch_size, hidden_size1])
            c_old1 = tf.zeros([batch_size, hidden_size1])
            h_old2 = tf.zeros([batch_size, hidden_size2])
            c_old2 = tf.zeros([batch_size, hidden_size2])

            w_uplink_real = tf.get_variable("w_uplink_real_a1" + str(t), shape=(1, M1, 1), trainable=True)
            w_uplink_imag = tf.get_variable("w_uplink_imag_a1" + str(t), shape=(1, M1, 1), trainable=True)
            w_complex = tf.complex(w_uplink_real, w_uplink_imag)
            w_a1 = w_complex / tf.norm(w_complex, axis=1, keepdims=True)
            beamformer_list[0].append(tf.repeat(w_a1,batch_size_val,axis=0))

            w_uplink_real = tf.get_variable("w_uplink_real_a2" + str(t), shape=(1, M1, 1), trainable=True)
            w_uplink_imag = tf.get_variable("w_uplink_imag_a2" + str(t), shape=(1, M1, 1), trainable=True)
            w_complex = tf.complex(w_uplink_real, w_uplink_imag)
            w_a2 = w_complex / tf.norm(w_complex, axis=1, keepdims=True)
            beamformer_list[1].append(tf.repeat(w_a2,batch_size_val,axis=0))

            w_uplink_real = tf.get_variable("w_uplink_real_b1" + str(t), shape=(1, M2, 1), trainable=True)
            w_uplink_imag = tf.get_variable("w_uplink_imag_b1" + str(t), shape=(1, M2, 1), trainable=True)
            w_complex = tf.complex(w_uplink_real, w_uplink_imag)
            w_b1 = w_complex / tf.norm(w_complex, axis=1, keepdims=True)
            beamformer_list[2].append(tf.repeat(w_b1,batch_size_val,axis=0))

            v_uplink_real = tf.get_variable("v_uplink_real" + str(t), shape=(1, N1 * N2), trainable=True)
            v_uplink_imag = tf.get_variable("v_uplink_imag" + str(t), shape=(1, N1 * N2), trainable=True)
            v_complex = tf.complex(v_uplink_real, v_uplink_imag)
            v = v_complex / tf.cast(tf.abs(v_complex), tf.complex64)
            ris_list[0].append(tf.repeat(v,batch_size_val,axis=0))

            G = A @ tf.linalg.diag(v) @ B
            G_list[0].append(G)
            bf_gain_rnd = tf.reduce_mean(tf.abs(tf.transpose(tf.conj(w_a2), perm=[0, 2, 1]) @ G @ w_b1) ** 2)

        'Rx observes the next measurement'
        y_noiseless2 = tf.transpose(tf.conj(w_b1), perm=[0, 2, 1]) @ tf.transpose(tf.conj(G), perm=[0, 2, 1]) @ w_a2
        noise2 = tf.complex(tf.random_normal(tf.shape(y_noiseless2), mean=0.0, stddev=noiseSTD_per_dim), \
                            tf.random_normal(tf.shape(y_noiseless2), mean=0.0, stddev=noiseSTD_per_dim))
        y_complex2 = tf.complex(tf.sqrt(lay['P']), 0.0) * y_noiseless2 + noise2
        y_complex2 = tf.reshape(y_complex2, [-1, 1])
        y_real2 = tf.concat([tf.real(y_complex2), tf.imag(y_complex2)], axis=1) / tf.sqrt(lay['P'])

        'Rx design next receive beamformer based on h_old2'
        h_old2, c_old2 = RNN2((y_real2, h_old2, c_old2))
        w_her = MLP_user2_receive(h_old2)
        w_norm = tf.reshape(tf.norm(w_her, axis=1), (-1, 1))
        w_her = tf.divide(w_her, w_norm)
        w_b1 = tf.complex(w_her[:, 0:M2], w_her[:, M2:2 * M2])
        w_b1 = tf.reshape(w_b1, [-1, M2, 1])
        if t<tau-1: beamformer_list[2].append(w_b1)

        'Rx design next transimit beamformer based on  h_old2'
        w_her = MLP_user2_transmit(h_old2)
        w_norm = tf.reshape(tf.norm(w_her, axis=1), (-1, 1))
        w_her = tf.divide(w_her, w_norm)
        w_b2 = tf.complex(w_her[:, 0:M2], w_her[:, M2:2 * M2])
        w_b2 = tf.reshape(w_b2, [-1, M2, 1])
        beamformer_list[3].append(w_b2)

        'Rx update RIS reflection coefficients'
        h_old = tf.concat([h_old2, h_old1], axis=1)
        v_her = MLP_RIS2(h_old)
        v_complex = tf.complex(v_her[:, 0:N], v_her[:, N:2 * N])
        v = v_complex / tf.cast(tf.abs(v_complex), tf.complex64)
        ris_list[1].append(v)

        'Update effective channel'
        G = A @ tf.linalg.diag(v) @ B
        G_list[1].append(G)

        'Tx observes the next measurement'
        y_noiseless1 = tf.transpose(tf.conj(w_a1), perm=[0, 2, 1]) @ G @ w_b2
        noise1 = tf.complex(tf.random_normal(tf.shape(y_noiseless1), mean=0.0, stddev=noiseSTD_per_dim), \
                            tf.random_normal(tf.shape(y_noiseless1), mean=0.0, stddev=noiseSTD_per_dim))
        y_complex1 = tf.complex(tf.sqrt(lay['P']), 0.0) * y_noiseless1 + noise1
        y_complex1 = tf.reshape(y_complex1, [-1, 1])
        y_real1 = tf.concat([tf.real(y_complex1), tf.imag(y_complex1)], axis=1) / tf.sqrt(lay['P'])

        'Tx designs next receive beamformer based on  h_old1'
        h_old1, c_old1 = RNN1((y_real1, h_old1, c_old1))
        v_her = MLP_user1_receive(h_old1)
        v_norm = tf.reshape(tf.norm(v_her, axis=1), (-1, 1))
        v_her = tf.divide(v_her, v_norm)
        w_a1 = tf.complex(v_her[:, 0:M1], v_her[:, M1:2 * M1])
        w_a1 = tf.reshape(w_a1, [-1, M1, 1])
        if t<tau-1: beamformer_list[0].append(w_a1)

        'Tx designs next transimit beamformer based on  h_old1'
        v_her = MLP_user1_transmit(h_old1)
        v_norm = tf.reshape(tf.norm(v_her, axis=1), (-1, 1))
        v_her = tf.divide(v_her, v_norm)
        w_a2 = tf.complex(v_her[:, 0:M1], v_her[:, M1:2 * M1])
        w_a2 = tf.reshape(w_a2, [-1, M1, 1])
        if t<tau-1: beamformer_list[1].append(w_a2)

        'Tx update RIS reflection coefficients'
        h_old = tf.concat([h_old1, h_old2], axis=1)
        v_her = MLP_RIS1(h_old)
        v_complex = tf.complex(v_her[:, 0:N], v_her[:, N:2 * N])
        v = v_complex / tf.cast(tf.abs(v_complex), tf.complex64)
        if t<tau-1: ris_list[0].append(v)

        'Update effective channel'
        G = A @ tf.linalg.diag(v) @ B
        if t<tau-1: G_list[0].append(G)

    'Caculate bf_gain'
    MLP_bf1 = MLPBlock(3, [1024, 1024, 2 * M1], name='mlp_bf1')
    MLP_bf2 = MLPBlock(3, [1024, 1024, 2 * M2], name='mlp_bf2')
    MLP_bf3 = MLPBlock(3, [1024, 1024, 2 * N], name='mlp_bf3')

    w_a = MLP_bf1(c_old1)
    w_a_norm = tf.reshape(tf.norm(w_a, axis=1), (-1, 1))
    w_a = tf.divide(w_a, w_a_norm)
    w_a_complex = tf.complex(w_a[:, 0:M1], w_a[:, M1:2 * M1])
    w_a_complex = tf.reshape(w_a_complex, [-1, M1, 1])
    beamformer_final[0].append(w_a_complex)

    c_old = tf.concat([c_old1, c_old2], axis=1)
    v_tmp = MLP_bf3(c_old)
    v_complex = tf.complex(v_tmp[:, 0:N], v_tmp[:, N:2 * N])
    v_complex = v_complex / tf.cast(tf.abs(v_complex), tf.complex64)
    v_complex = tf.reshape(v_complex, [-1, N])
    ris_final.append(v_complex)
    G = A @ tf.linalg.diag(v_complex) @ B
    G_list[3].append(G)

    w_b = MLP_bf2(c_old2)
    w_b_norm = tf.reshape(tf.norm(w_b, axis=1), (-1, 1))
    w_b = tf.divide(w_b, w_b_norm)
    w_b_complex = tf.complex(w_b[:, 0:M2], w_b[:, M2:2 * M2])
    w_b_complex = tf.reshape(w_b_complex, [-1, M2, 1])
    beamformer_final[1].append(w_b_complex)

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
# alpha1_val = np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_val, L1]) \
#             + 1j * np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_val, L1])
# alpha2_val = np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_val, L2]) \
#             + 1j * np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_val, L2])
# phi_1_val = np.random.uniform(low=phi_min, high=phi_max, size=[batch_size_val, L1,4])
# phi_2_val = np.random.uniform(low=phi_min, high=phi_max, size=[batch_size_val, L2,4])

test_data = sio.loadmat('test_data.mat')
alpha1_val = test_data['alpha1_val'][0:batch_size_val]
alpha2_val = test_data['alpha2_val'][0:batch_size_val]
phi_1_val = test_data['phi_1_val'][0:batch_size_val]
phi_2_val = test_data['phi_2_val'][0:batch_size_val]

feed_dict_val = {alpha_input1: alpha1_val, alpha_input2: alpha2_val, phi_input_1: phi_1_val, phi_input_2: phi_2_val,
                 lay['P']: P_snr}
###########  Training
with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, model_path)
    best_loss, opt_loss, rnd_loss, beamformer_list_np, beamformer_final_np, G_list_np, ris_list_np, ris_final_np, A_np, B_np \
        = sess.run([loss, bf_gain_opt, bf_gain_rnd,beamformer_list, beamformer_final, G_list,
                    ris_list, ris_final,A,B], feed_dict=feed_dict_val)
    print(10 * np.log10(-best_loss), 10 * np.log10(opt_loss), 10 * np.log10(rnd_loss))
    print(tf.test.is_gpu_available())  # Prints whether or not GPU is on

    sio.savemat('RNN_interpretation' + str(tau) + '.mat',
                {'bf_gain_opt_dB': 10 * np.log10(-best_loss),
                 'bf_gain_random_RIS_optimal_beamform_dB': 10 * np.log10(opt_loss),
                 'M1_M2_N1_N2_L1_L2_phi_min_max': (M1, M2, N1, N2, L1, L2, phi_min, phi_max),
                 'snr_dB': snrdB,'communication_rounds': tau,'w_Tx_r': beamformer_list_np[0],
                 'w_Tx_t': beamformer_list_np[1],'w_Rx_r': beamformer_list_np[2],'w_Rx_t': beamformer_list_np[3],
                 'w_Tx_final': beamformer_final_np[0],'w_Rx_final': beamformer_final_np[1],
                 'ris_sensing_Tx':ris_list_np[0],'ris_sensing_rx':ris_list_np[1],'ris_final':ris_final_np,
                 'G_random_ris':G_list_np[2],'G_final':G_list_np[3],'G_Tx':G_list_np[0],'G_Rx':G_list_np[1],
                 'A':A_np,'B':B_np})
