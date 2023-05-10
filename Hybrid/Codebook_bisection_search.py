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

#
def generate_beamformer(Na, k, ii, NRF):
    Ma = NRF*2**(k-1)
    if ii>=Ma: raise Exception("ii <= Ma-1")
    phi_a_i = np.pi*(1-(2*ii+1)/Ma)
    w = np.zeros(Na,dtype=np.complex)
    w[0:Ma] = np.exp(-1j*phi_a_i*np.arange(Ma))
    w = w/np.linalg.norm(w)
    return w[:,None]

def generate_beamformer_sets(Na, k, NRF):
    for ii in range(NRF):
        if ii==0:
            w = generate_beamformer(Na, k, ii, NRF)
        else:
            w = np.concatenate([w,generate_beamformer(Na, k, ii, NRF)],axis=1)
    return w

'System Information'
N1 = 64  # Number of BS's antennas
N2 = 32
tau = 2  # Pingpong rounds
L = 3  # number of path
NRF = 2
Ka = np.log2(N1/NRF)+1
Kb = np.log2(N2/NRF)+1
# w = generate_beamformer(N1, 3, 10)
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

def update_analog_precoder(F,p,k,i,K):
    N = F.shape[0]
    pI = np.argsort(p)
    bI, bL = [], []
    n= 0
    while n<NRF:
        bI.append(pI[n])
        p[pI[n]] = 0
        if k[pI[n]]==K or n==NRF-1:
            bL.append(1)
            n = n+1
        else:
            bL.append(2)
            n = n+2
    F = []
    k_new = []
    i_new = []
    for t in range(len(bI)):
        m = bL[t]
        if m == 2:
            w1 = generate_beamformer(N,k[bI[t]]+1,i[bI[t]]*2,NRF)
            w2 = generate_beamformer(N,k[bI[t]]+1,i[bI[t]]*2+1,NRF)
            F.append(w1[:,0])
            F.append(w2[:,0])
            k_new.append(k[bI[t]]+1)
            k_new.append(k[bI[t]]+1)
            i_new.append(i[bI[t]]*2)
            i_new.append(i[bI[t]]*2+1)
            # F = np.concatenate([F,w1,w2],axis=1)
        else:
            w = generate_beamformer(N,k[bI[t]],i[bI[t]],NRF)
            F.append(w[:,0])
            k_new.append(k[bI[t]])
            i_new.append(i[bI[t]])
            # F = np.concatenate([F,w],axis=1)
    return np.array(F).T, k_new, i_new, p

bf_gain_all = 0
for ii in range(batch_size_test):
    v_j = np.random.normal(loc=0, scale=1, size=[N1, 1]) \
        + 1j * np.random.normal(loc=0, scale=1, size=[N1, 1])
    v_j_t = v_j/np.linalg.norm(v_j,axis=0,keepdims=True)
    v_j_r = generate_beamformer_sets(N1, 1, NRF)
    w_j_r = generate_beamformer_sets(N2, 1, NRF)
    pa, ka, ia = np.array(0), [1]*NRF, list(np.arange(NRF))
    pb, kb, ib = np.array(0), [1]*NRF, list(np.arange(NRF))
    bf_gain_pingpong = []
    for jj in range(tau):
        yb_noiseless = np.transpose(np.conj(w_j_r)) @ np.transpose(np.conj(G[ii])) @ v_j_t
        noise = np.random.normal(loc=0, scale=noiseSTD_per_dim, size=np.shape(yb_noiseless)) \
            + 1j * np.random.normal(loc=0, scale=noiseSTD_per_dim, size=np.shape(yb_noiseless))
        yb = np.complex(np.sqrt(P_snr))*yb_noiseless+noise
        yb_abs = np.abs(yb)
        if jj==0:
            pb = -1*np.abs(yb)[:,0]
        else:
            pb = np.concatenate([pb,-1*np.abs(yb)[:,0]])
        w_j_r, k_new, i_new, p_new = update_analog_precoder(w_j_r,pb,kb,ib,Kb)
        w_j_t = w_j_r@yb
        w_j_t = w_j_t/np.linalg.norm(w_j_t,axis=0,keepdims=True)
        kb = kb+k_new
        ib = ib+i_new


        ya_noiseless = np.transpose(np.conj(v_j_r)) @ G[ii] @ w_j_t
        noise = np.random.normal(loc=0, scale=noiseSTD_per_dim, size=np.shape(ya_noiseless)) \
            + 1j * np.random.normal(loc=0, scale=noiseSTD_per_dim, size=np.shape(ya_noiseless))
        ya = np.complex(np.sqrt(P_snr))*ya_noiseless+noise
        ya_abs = np.abs(ya)
        if jj==0:
            pa = -1*np.abs(ya)[:,0]
        else:
            pa = np.concatenate([pa,-1*np.abs(ya)[:,0]])
        v_j_r, k_new, i_new, p_new = update_analog_precoder(v_j_r,pa,ka,ia,Ka)
        v_j_t = v_j_r@ya
        v_j_t = v_j_t/np.linalg.norm(v_j_t,axis=0,keepdims=True)
        ka = ka+k_new
        ia = ia+i_new

        # v_j_t = v_j_r@ya
        # v_j_t = v_j_t/np.linalg.norm(v_j_t,axis=0,keepdims=True)

        tmp_omp = np.transpose(np.conj(w_j_r)) @ np.transpose(np.conj(G[ii])) @ v_j_r
        bf_gain = np.max(np.abs(tmp_omp) ** 2)
        bf_gain_pingpong.append(bf_gain)
        #
        # print('jj:%d' % jj, ' SNR:', P_snr, 'bf_gain_pingpong:%2.5f' % bf_gain,
        #       '  bf_gain_opt:%2.5f' % bf_gain_opt)

    bf_gain_all = bf_gain_all+np.array(bf_gain_pingpong)
    if ii%100==0 and ii>0:
        bf_gain_all_mean = bf_gain_all/(ii+1)
        print('ii=%d, bf_gain=%.4f, bf_gain_opt=%.4f'%(ii,bf_gain_all_mean[-1],bf_gain_opt))
        results_path = './results/pingpong_L_SNR_tau_N1_N2_' + str((L, snrdB, tau, N1, N2)) + '.mat'
        sio.savemat(results_path,
                    {'bf_gain_pingpong_dB': 10*np.log10(bf_gain_all_mean),
                     'bf_gain_opt_dB': 10*np.log10(bf_gain_opt), 'batch_size_test': batch_size_test})
