import numpy as np
import scipy.io as sio

def func_codedesign(delta_inv, phi_min, phi_max, N1, N2):
    phi = np.linspace(start=phi_min, stop=phi_max, num=delta_inv)
    for i in range(delta_inv):
        a_phi1 = np.exp(1j * np.pi * np.arange(N1) * np.sin(phi[i]))
        A_tmp = np.zeros([N1 * N2, delta_inv], dtype=np.complex128)
        for j in range(delta_inv):
            a_phi2 = np.exp(1j * np.pi * np.arange(N2) * np.sin(phi[j]))
            A_tmp[:, j] = np.kron(a_phi1.conj(), a_phi2)
        if i == 0:
            A_dic = A_tmp
        else:
            A_dic = np.concatenate([A_dic, A_tmp], axis=1)
    return A_dic, phi


'System Information'
N1 = 64  # Number of BS's antennas
N2 = 32
tau1 = 4  # Pilot length
tau2 = 3  # Pilot length
L = 3  # number of path

'Channel Information'
phi_min = -60 * (np.pi / 180)  # Lower-bound of AoAs
phi_max = 60 * (np.pi / 180)  # Upper-bound of AoAs
snrdB = 0  # SNRs
P_snr = 10 ** (snrdB / 10)  # Considered TX powers
mean_true_alpha = 0.0 + 0.0j  # Mean of the fading coefficient
std_per_dim_alpha = np.sqrt(0.5)  # STD of the Gaussian fading coefficient per real dim.
noiseSTD_per_dim = np.sqrt(0.5)  # STD of the Gaussian noise per real dim.
delta_inv = 64
A_dic, phi_all = func_codedesign(delta_inv, phi_min, phi_max, N1, N2)

'Generate channels'
batch_size_test = 10000
alpha_val = np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_test, L]) \
            + 1j * np.random.normal(loc=np.real(mean_true_alpha), scale=std_per_dim_alpha, size=[batch_size_test, L])
phi_1_idx_val = np.ones((batch_size_test, L))
phi_2_idx_val = np.ones((batch_size_test, L))
for ii in range(batch_size_test):
    phi_1_idx_val[ii] = np.random.choice(delta_inv, replace=False, size=[L])
    phi_2_idx_val[ii] = np.random.choice(delta_inv, replace=False, size=[L])
# idx_vec_true = phi_1_idx_val * delta_inv + phi_2_idx_val
# idx_vec_true = np.squeeze(idx_vec_true.astype(int))
phi_1_val = phi_all[phi_1_idx_val.astype(int)]
phi_2_val = phi_all[phi_2_idx_val.astype(int)]

from0toN1 = np.reshape(np.arange(N1), (-1, N1, 1))
from0toN2 = np.reshape(np.arange(N2), (-1, N2, 1))
for ii in range(L):
    phi_1 = (phi_1_val[:, ii]).reshape(-1, 1, 1)
    a_phi1 = np.exp(1j * np.pi * from0toN1 * np.sin(phi_1))
    phi_2 = (phi_2_val[:, ii]).reshape(-1, 1, 1)
    a_phi2 = np.exp(1j * np.pi * from0toN2 * np.sin(phi_2))
    G_i = a_phi1 @ np.transpose(np.conj(a_phi2), axes=(0, 2, 1))
    if ii == 0:
        G = np.reshape(alpha_val[:, ii], [-1, 1, 1]) * G_i
    else:
        G = G + np.reshape(alpha_val[:, ii], [-1, 1, 1]) * G_i
G = G / np.sqrt(L)

'Optimal beamforming under 2-norm constraints'
u, s, vh = np.linalg.svd(G, full_matrices=False)
v_opt = np.reshape(u[:, :, 0], [-1, N1, 1])
wh_opt = np.reshape(vh[:, 0, :], [-1, 1, N2])
tmp_opt = wh_opt @ np.transpose(np.conj(G), axes=[0, 2, 1]) @ v_opt
bf_gain_opt = np.mean(np.abs(tmp_opt) ** 2)

'System model'
V = np.random.normal(loc=0, scale=1, size=[batch_size_test, N1, tau1]) \
    + 1j * np.random.normal(loc=0, scale=1, size=[batch_size_test, N1, tau1])
V = V / np.linalg.norm(V, axis=1, keepdims=True)
W = np.random.normal(loc=0, scale=1, size=[batch_size_test, N2, tau2]) \
    + 1j * np.random.normal(loc=0, scale=1, size=[batch_size_test, N2, tau2])
W = W / np.linalg.norm(W, axis=1, keepdims=True)

Y_noiseless = np.transpose(np.conj(W), axes=(0, 2, 1)) @ np.transpose(np.conj(G), axes=(0, 2, 1)) @ V
noise = np.random.normal(loc=0, scale=noiseSTD_per_dim, size=np.shape(Y_noiseless)) \
        + 1j * np.random.normal(loc=0, scale=noiseSTD_per_dim, size=np.shape(Y_noiseless))
Y = np.complex(np.sqrt(P_snr)) * Y_noiseless + noise
y_vec = np.reshape(np.transpose(Y, (0, 2, 1)), (batch_size_test, -1, 1))

'OMP algorithm'
bf_gain_omp_list = []
mse_aoa, mse_G = [], []
for ii in range(batch_size_test):
    A = np.kron(np.transpose(V[ii]), np.transpose(np.conj(W[ii])))
    A = np.complex(np.sqrt(P_snr)) * A @ A_dic
    ### for debugging
    # z = np.zeros((delta_inv * delta_inv, 1), dtype=np.complex128)
    # z[idx_vec_true[ii]] = np.conj(np.squeeze(alpha_val[ii, 0]))
    # y = A @ z
    # print(np.linalg.norm(y - y_vec[ii]))
    idx = np.zeros(L, dtype=int)
    idx1 = np.zeros(L, dtype=int)
    idx2 = np.zeros(L, dtype=int)
    r = y_vec[ii]
    for tt in range(L):
        lamda_t = np.argmax(np.abs(np.transpose(np.conj(r)) @ A))
        idx[tt] = lamda_t
        idx1_hat, idx2_hat = lamda_t // delta_inv, np.mod(lamda_t, delta_inv)
        idx1[tt], idx2[tt] = idx1_hat, idx2_hat
        phi_A = np.reshape(A[:, idx[0:tt + 1]], (-1, tt + 1))
        alpha_hat = np.linalg.inv(np.transpose(np.conj(phi_A)) @ phi_A) @ np.transpose(np.conj(phi_A)) @ y_vec[ii]
        r = y_vec[ii] - phi_A @ alpha_hat
    alpha_hat = alpha_hat * np.sqrt(L)
    aoa1_hat, aoa2_hat = np.sort(phi_all[idx1]), np.sort(phi_all[idx2])
    aoa1_true, aoa2_true = np.sort(phi_1_val[ii]), np.sort(phi_2_val[ii])
    mse_aoa.append(np.sum((aoa1_hat - aoa1_true) ** 2) + np.sum((aoa2_hat - aoa2_true) ** 2))

    # construt the estimated G_hat
    for tt in range(L):
        a_phi1 = np.exp(1j * np.pi * np.arange(N1) * np.sin(phi_all[idx1[tt]]))
        a_phi2 = np.exp(1j * np.pi * np.arange(N2) * np.sin(phi_all[idx2[tt]]))
        G_t = np.reshape(a_phi1, [N1, 1]) @ np.transpose(np.conj(np.reshape(a_phi2, [N2, 1])))
        if tt == 0:
            G_hat = alpha_hat[tt].conj() * G_t
        else:
            G_hat = G_hat + alpha_hat[tt].conj() * G_t
    G_hat = G_hat / np.sqrt(L)
    mse_G.append(np.linalg.norm(G_hat - G[ii]) ** 2 / np.linalg.norm(G[ii]) ** 2)

    u, s, vh = np.linalg.svd(G_hat, full_matrices=False)
    v_omp = np.reshape(u[:, 0], [N1, 1])
    wh_omp = np.reshape(vh[0, :], [1, N2])
    tmp_omp = wh_omp @ np.transpose(np.conj(G[ii])) @ v_omp
    bf_gain_omp = np.abs(tmp_omp) ** 2
    bf_gain_omp_list.append(bf_gain_omp)
    if ii % 10 == 0:
        print('ii:%3d' % ii, ' SNR:', P_snr, 'mse_aoa:%2.5f' % (np.mean(mse_aoa)), ' mse_G:%2.5f' % (np.mean(mse_G)),
              '  bf_gain_omp:%2.5f' % np.mean(bf_gain_omp_list), '  bf_gain_opt:%2.5f' % bf_gain_opt)

# results_path = './results/OMP_L_SNR_tau1_tau2_N1_N2_' + str((L, snrdB, tau1, tau2, N1, N2)) + '.mat'
# sio.savemat(results_path,
#             {'mse_aoa': np.mean(mse_aoa), 'mse_G': np.mean(mse_G),
#              'bf_gain_omp_dB': 10*np.log10(np.mean(bf_gain_omp_list)),
#              'bf_gain_opt_dB': 10*np.log10(bf_gain_opt), 'batch_size_test': batch_size_test})
