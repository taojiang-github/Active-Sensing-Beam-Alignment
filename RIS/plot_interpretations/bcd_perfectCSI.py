import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio

tf.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'System Information'
M1 = 64  # Number of BS's antennas
M2 = 32  # Number of UE's antennas
N1, N2 = 8, 8
N = N1 * N2  # Number of RIS elements
L1 = 2  # number of path between BS and RIS
L2 = 3  # number of path between RIS and UE

'Channel Information'
phi_min = -60 * (np.pi / 180)  # Lower-bound of AoAs
phi_max = 60 * (np.pi / 180)   # Upper-bound of AoAs
mean_true_alpha = 0.0 + 0.0j      # Mean of the fading coefficient
std_per_dim_alpha = np.sqrt(0.5)  # STD of the Gaussian fading coefficient per real dim.

'Algorithm Parameters'
max_iter = 20
batch_size_val = 10000

'BCD Algorithm'
tf.reset_default_graph()  # Reseting the graph
alpha_input1 = tf.placeholder(tf.complex64, shape=(None, L1), name="alpha_input1")  # fading coefficients
alpha_input2 = tf.placeholder(tf.complex64, shape=(None, L2), name="alpha_input2")  # fading coefficients
phi_input_1 = tf.placeholder(tf.float32, shape=(None, L1, 4), name="phi_input1")  # AoAs and AoDs between RIS and BS
phi_input_2 = tf.placeholder(tf.float32, shape=(None, L2, 4), name="phi_input2")  # AoAs and AoDs between RIS and UE
batch_size = tf.shape(alpha_input1)[0]
with tf.name_scope("array_response_construction"):
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

    #####################optimal beamforming vectors with random reflection coefficients################
    v = tf.complex(tf.random_normal(shape=[batch_size, N]), tf.random_normal(shape=[batch_size, N]))
    v = v / tf.cast(tf.abs(v), tf.complex64)
    G = A @ tf.linalg.diag(v) @ B
    s, u_tmp, v_tmp = tf.linalg.svd(G)
    w_a_opt = tf.expand_dims(u_tmp[:, :, 0], axis=2)
    w_a_opt = w_a_opt / tf.norm(w_a_opt, axis=1, keepdims=True)
    w_b_opt = tf.expand_dims(v_tmp[:, :, 0], axis=2)
    w_b_opt = w_b_opt / tf.norm(w_b_opt, axis=1, keepdims=True)
    tmp_opt = tf.transpose(tf.conj(w_a_opt), perm=[0, 2, 1]) @ G @ w_b_opt
    bf_gain_opt = tf.abs(tmp_opt) ** 2
    bf_gain_opt_rand_RIS = tf.real(bf_gain_opt)

with tf.name_scope("Alternating_optimization"):
    w_a = w_a_opt
    w_b = w_b_opt
    bf_gain_list = [bf_gain_opt_rand_RIS]
    for ii in range(max_iter):
        #####optimize reflection coefficients given w_a and w_b
        tmp1 = tf.reshape(tf.transpose(tf.conj(w_a), perm=[0, 2, 1]) @ A, [-1, N])
        tmp2 = tf.reshape(B @ w_b, [-1, N])
        opt_angle = -tf.math.angle(tmp1 * tmp2)
        v = tf.complex(tf.cos(opt_angle), tf.sin(opt_angle))

        #########record the bf_gain
        G = A @ tf.linalg.diag(v) @ B
        tmp_opt = tf.transpose(tf.conj(w_a), perm=[0, 2, 1]) @ G @ w_b
        bf_gain_opt = tf.abs(tmp_opt) ** 2
        bf_gain_list.append(bf_gain_opt)

        #######optimize the w_a and w_b given reflection coefficients v
        s, u_tmp, v_tmp = tf.linalg.svd(G)
        w_a = tf.expand_dims(u_tmp[:, :, 0], axis=2)
        w_a = w_a / tf.norm(w_a, axis=1, keepdims=True)
        w_b = tf.expand_dims(v_tmp[:, :, 0], axis=2)
        w_b = w_b / tf.norm(w_b, axis=1, keepdims=True)

        #########record the bf_gain
        tmp_opt = tf.transpose(tf.conj(w_a), perm=[0, 2, 1]) @ G @ w_b
        bf_gain_opt = tf.abs(tmp_opt) ** 2
        bf_gain_list.append(bf_gain_opt)



###########  Validation Set
# alpha1_val = np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_val, L1]) \
#             + 1j * np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_val, L1])
# alpha2_val = np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_val, L2]) \
#             + 1j * np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_val, L2])
# phi_1_val = np.random.uniform(low=phi_min, high=phi_max, size=[batch_size_val, L1,4])
# phi_2_val = np.random.uniform(low=phi_min, high=phi_max, size=[batch_size_val, L2,4])
#
# sio.savemat('test_data.mat',{'alpha1_val': alpha1_val,'alpha2_val':alpha2_val,'phi_1_val':phi_1_val,'phi_2_val':phi_2_val})
test_size = 20
test_data = sio.loadmat('test_data.mat')
alpha1_val = test_data['alpha1_val'][0:test_size]
alpha2_val = test_data['alpha2_val'][0:test_size]
phi_1_val = test_data['phi_1_val'][0:test_size]
phi_2_val = test_data['phi_2_val'][0:test_size]

feed_dict_val = {alpha_input1: alpha1_val, alpha_input2: alpha2_val, phi_input_1: phi_1_val, phi_input_2: phi_2_val}
with tf.Session() as sess:
    bf_gain_iter, v_np, w_tx, w_rx = sess.run([bf_gain_list,v,w_a,w_b], feed_dict=feed_dict_val)
    bf_gain_iter_dB = 10 * np.log10(np.mean(np.squeeze(bf_gain_iter), axis=1))

    print(bf_gain_iter_dB)

    plt.plot(bf_gain_iter_dB, 'o-')
    plt.xlabel('Number of iterations')
    plt.ylabel('Beamforming Gain (dB)')
    plt.show()

    sio.savemat('bcd_perfectCSI.mat', {'bf_gain_bcd_iterations_all_samples': np.squeeze(bf_gain_iter),
                                     'bf_gain_dB':bf_gain_iter_dB,
                                     'bf_gain_random_RIS_optimal_beamform_dB': bf_gain_iter_dB[0],
                                     'bf_gain_opt_dB': bf_gain_iter_dB[-1],
                                     'M1_M2_N1_N2_L1_L2_phi_min_max': (M1,M2,N1,N2,L1,L2,phi_min,phi_max),
                                     'BCD_max_iter':max_iter,
                                      'v_np':v_np,
                                      'w_tx':w_tx,
                                      'w_rx':w_rx})
