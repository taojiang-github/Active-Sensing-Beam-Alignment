import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
from tensorflow.keras.layers import BatchNormalization, Dense
import os
import scipy.io as sio

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
N1 = 64  # Number of BS's antennas
N2 = 32  # Number of UE's antennas
tau = 6  # Communication rounds
L = 3  # number of path

'Channel Information'
phi_min = -60 * (np.pi / 180)  # Lower-bound of AoAs
phi_max = 60 * (np.pi / 180)  # Upper-bound of AoAs
snrdB = 0
P_snr = 10 ** (snrdB / 10)  # Considered TX powers
mean_true_alpha = 0.0 + 0.0j  # Mean of the fading coefficient
std_per_dim_alpha = np.sqrt(0.5)  # STD of the Gaussian fading coefficient per real dim.
noiseSTD_per_dim = np.sqrt(0.5)  # STD of the Gaussian noise per real dim.
#####################################################
'Learning Parameters'
initial_run = 0  # 0: Continue training; 1: Starts from the scratch
n_epochs = 0  # Num of epochs
learning_rate = 0.0001
batch_per_epoch = 400
batch_size_train = 1024
batch_size_val = 10000
model_path = '../params/active_learning_RNN_L_SNR_tau_N1_N2_' + str((L, snrdB, tau, N1, N2))

'Build the graph'
tf.reset_default_graph()  # Reseting the graph
he_init = tf.variance_scaling_initializer()  # Define initialization method

alpha_input = tf.placeholder(tf.complex64, shape=(None, L), name="alpha_input")
phi_input_1 = tf.placeholder(tf.float32, shape=(None, L), name="phi_input1")
phi_input_2 = tf.placeholder(tf.float32, shape=(None, L), name="phi_input2")
batch_size = tf.shape(alpha_input)[0]

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

    # optimal beamforming vectors
    s, u, v = tf.linalg.svd(G)
    v_opt = tf.expand_dims(u[:, :, 0], axis=2)
    v_opt = v_opt / tf.norm(v_opt, axis=1, keepdims=True)
    w_opt = tf.expand_dims(v[:, :, 0], axis=2)
    w_opt = w_opt / tf.norm(w_opt, axis=1, keepdims=True)
    tmp_opt = tf.transpose(tf.conj(v_opt), perm=[0, 2, 1]) @ G @ w_opt
    bf_gain_opt = tf.reduce_mean(tf.abs(tmp_opt) ** 2)
    bf_gain_opt = tf.real(bf_gain_opt)

with tf.name_scope("channel_sensing"):
    hidden_size1, hidden_size2 = 512, 256
    RNN1 = RNN(hidden_size1, 'rnn_1')
    RNN2 = RNN(hidden_size2, 'rnn_2')
    MLP_user1_transmit = MLPBlock(3, [512, 512, 2 * N1], name='mlp_user1_transmit')
    MLP_user1_receive = MLPBlock(3, [512, 512, 2 * N1], name='mlp_user1_receive')
    MLP_user2_transmit = MLPBlock(3, [512, 512, 2 * N2], name='mlp_user2_transmit')
    MLP_user2_receive = MLPBlock(3, [512, 512, 2 * N2], name='mlp_user2_receive')

    beamformer_list = [[], [], [], []]
    beamformer_final = [[], []]
    for t in range(tau):
        'initailization'
        if t == 0:
            h_old1 = tf.zeros([batch_size, hidden_size1])
            c_old1 = tf.zeros([batch_size, hidden_size1])
            h_old2 = tf.zeros([batch_size, hidden_size2])
            c_old2 = tf.zeros([batch_size, hidden_size2])

            v_uplink_real = tf.get_variable("v_uplink_real1" + str(t), shape=(1, N1, 1), trainable=True)
            v_uplink_imag = tf.get_variable("v_uplink_imag1" + str(t), shape=(1, N1, 1), trainable=True)
            v_complex = tf.complex(v_uplink_real, v_uplink_imag)
            v1 = v_complex / tf.norm(v_complex, axis=1, keepdims=True)
            beamformer_list[0].append(tf.repeat(v1,batch_size_val,axis=0))

            v_uplink_real = tf.get_variable("v_uplink_real2" + str(t), shape=(1, N1, 1), trainable=True)
            v_uplink_imag = tf.get_variable("v_uplink_imag2" + str(t), shape=(1, N1, 1), trainable=True)
            v_complex = tf.complex(v_uplink_real, v_uplink_imag)
            v2 = v_complex / tf.norm(v_complex, axis=1, keepdims=True)
            beamformer_list[1].append(tf.repeat(v2,batch_size_val,axis=0))

            w_uplink_real = tf.get_variable("w_uplink_real1" + str(t), shape=(1, N2, 1), trainable=True)
            w_uplink_imag = tf.get_variable("w_uplink_imag1" + str(t), shape=(1, N2, 1), trainable=True)
            w_complex = tf.complex(w_uplink_real, w_uplink_imag)
            w1 = w_complex / tf.norm(w_complex, axis=1, keepdims=True)
            beamformer_list[2].append(tf.repeat(w1,batch_size_val,axis=0))

            bf_gain_rnd = tf.reduce_mean(tf.abs(tf.transpose(tf.conj(v2), perm=[0, 2, 1]) @ G @ w1) ** 2)

        'User 2 observes the next measurement'
        y_noiseless2 = tf.transpose(tf.conj(w1), perm=[0, 2, 1]) @ tf.transpose(tf.conj(G), perm=[0, 2, 1]) @ v2
        noise2 = tf.complex(tf.random_normal(tf.shape(y_noiseless2), mean=0.0, stddev=noiseSTD_per_dim), \
                            tf.random_normal(tf.shape(y_noiseless2), mean=0.0, stddev=noiseSTD_per_dim))
        y_complex2 = tf.complex(tf.sqrt(lay['P']), 0.0) * y_noiseless2 + noise2
        y_complex2 = tf.reshape(y_complex2, [-1, 1])
        y_real2 = tf.concat([tf.real(y_complex2), tf.imag(y_complex2)], axis=1) / tf.sqrt(lay['P'])

        'user 2 design next receive beamformer based on h_old2'
        h_old2, c_old2 = RNN2((y_real2, h_old2, c_old2))
        w_her = MLP_user2_receive(h_old2)
        w_norm = tf.reshape(tf.norm(w_her, axis=1), (-1, 1))
        w_her = tf.divide(w_her, w_norm)
        w1 = tf.complex(w_her[:, 0:N2], w_her[:, N2:2 * N2])
        w1 = tf.reshape(w1, [-1, N2, 1])
        if t<tau-1: beamformer_list[2].append(w1)

        'user 2 design next transimit beamformer based on  h_old2'
        w_her = MLP_user2_transmit(h_old2)
        w_norm = tf.reshape(tf.norm(w_her, axis=1), (-1, 1))
        w_her = tf.divide(w_her, w_norm)
        w2 = tf.complex(w_her[:, 0:N2], w_her[:, N2:2 * N2])
        w2 = tf.reshape(w2, [-1, N2, 1])
        beamformer_list[3].append(w2)

        'user 1 observes the next measurement'
        y_noiseless1 = tf.transpose(tf.conj(v1), perm=[0, 2, 1]) @ G @ w2
        noise1 = tf.complex(tf.random_normal(tf.shape(y_noiseless1), mean=0.0, stddev=noiseSTD_per_dim), \
                            tf.random_normal(tf.shape(y_noiseless1), mean=0.0, stddev=noiseSTD_per_dim))
        y_complex1 = tf.complex(tf.sqrt(lay['P']), 0.0) * y_noiseless1 + noise1
        y_complex1 = tf.reshape(y_complex1, [-1, 1])
        y_real1 = tf.concat([tf.real(y_complex1), tf.imag(y_complex1)], axis=1) / tf.sqrt(lay['P'])

        'user 1 design next receive beamformer based on  h_old1'
        h_old1, c_old1 = RNN1((y_real1, h_old1, c_old1))
        v_her = MLP_user1_transmit(h_old1)  # this is actually the MLP for receive beamformer
        v_norm = tf.reshape(tf.norm(v_her, axis=1), (-1, 1))
        v_her = tf.divide(v_her, v_norm)
        v1 = tf.complex(v_her[:, 0:N1], v_her[:, N1:2 * N1])
        v1 = tf.reshape(v1, [-1, N1, 1])
        if t<tau-1: beamformer_list[0].append(v1)

        'user 1 design next transimit beamformer based on  h_old1'
        v_her = MLP_user1_receive(h_old1)  # this is actually the MLP for transimit beamformer
        v_norm = tf.reshape(tf.norm(v_her, axis=1), (-1, 1))
        v_her = tf.divide(v_her, v_norm)
        v2 = tf.complex(v_her[:, 0:N1], v_her[:, N1:2 * N1])
        v2 = tf.reshape(v2, [-1, N1, 1])
        if t<tau-1: beamformer_list[1].append(v2)

    'caculate bf_gain'
    MLP_bf1 = MLPBlock(3, [1024, 1024, 2 * N1], name='mlp_bf1')
    MLP_bf2 = MLPBlock(3, [1024, 1024, 2 * N2], name='mlp_bf2')

    v_tmp = MLP_bf1(c_old1)
    v_norm = tf.reshape(tf.norm(v_tmp, axis=1), (-1, 1))
    v_tmp = tf.divide(v_tmp, v_norm)
    v_complex = tf.complex(v_tmp[:, 0:N1], v_tmp[:, N1:2 * N1])
    v_complex = tf.reshape(v_complex, [-1, N1, 1])
    beamformer_final[0].append(v_complex)

    w_tmp = MLP_bf2(c_old2)
    w_norm = tf.reshape(tf.norm(w_tmp, axis=1), (-1, 1))
    w_tmp = tf.divide(w_tmp, w_norm)
    w_complex = tf.complex(w_tmp[:, 0:N2], w_tmp[:, N2:2 * N2])
    w_complex = tf.reshape(w_complex, [-1, N2, 1])
    beamformer_final[1].append(w_complex)

    bf_gain = tf.reduce_mean(tf.abs(tf.transpose(tf.conj(v_complex), perm=[0, 2, 1]) @ G @ w_complex) ** 2)

####### Loss Function
loss = -bf_gain
####### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()

###########  Validation Set
# alpha_val = np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_val, L]) \
#             + 1j * np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_val, L])
# phi_1_val = np.random.uniform(low=phi_min, high=phi_max, size=[batch_size_val, L])
# phi_2_val = np.random.uniform(low=phi_min, high=phi_max, size=[batch_size_val, L])
##### sio.savemat('test_data.mat',{'alpha_val': alpha_val,'phi_1_val':phi_1_val,'phi_2_val':phi_2_val})

# load test_data
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
    best_loss, opt_loss, rnd_loss, beamformer_list_np,beamformer_final_np, G_np  = \
        sess.run([loss, bf_gain_opt, bf_gain_rnd, beamformer_list, beamformer_final, G], feed_dict_val)
    print(-best_loss, opt_loss, rnd_loss)
    print(tf.test.is_gpu_available())  # Prints whether or not GPU is on

    sio.savemat('RNN_interpretation_tau_' + str(tau) + '.mat',
                {'bf_gain_dB': (10 * np.log10(-best_loss)), 'bf_gain_opt_dB': (10 * np.log10(opt_loss)),
                 'bf_gain_rnd_dB': (10 * np.log10(rnd_loss)), 'N1_N2_tau_L': (N1, N2, tau, L),
                 'phi_min_max': (phi_min, phi_max), 'snrdB': snrdB, 'w_Tx_r': beamformer_list_np[0],
                 'w_Tx_t': beamformer_list_np[1],'w_Rx_r': beamformer_list_np[2],'w_Rx_t': beamformer_list_np[3],
                 'phi_1_val': phi_1_val, 'phi_2_val': phi_2_val, 'alpha_val': alpha_val,
                 'w_Tx_final': beamformer_final_np[0],'w_Rx_final': beamformer_final_np[1], 'G': G_np})
