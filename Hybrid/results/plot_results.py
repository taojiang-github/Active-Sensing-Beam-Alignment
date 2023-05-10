import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})

# plt.rcParams.update({
#     "text.usetex": True,
#     'text.latex.preamble':[
#     r'\usepackage{amsmath}',r'\usepackage{bm}',
#     r'\usepackage{amssymb}'],
#     "font.family": "serif",
#     "font.serif": ["New Century Schoolbook"],
# })

'OMP results'
L, snrdB, N1, N2 = 3, 0, 64, 32
tau_tuple = [(2, 2), (4, 2), (4, 3), (4, 4), (4, 5)]
bf_gain_omp = []
for idx in tau_tuple:
    results_path = 'OMP_L_SNR_tau1_tau2_N1_N2_' + str((L, snrdB, idx[0], idx[1], N1, N2)) + '.mat'
    data = sio.loadmat(results_path)
    bf_gain_omp.append(data['bf_gain_omp_dB'][0][0])

'DNN random sensing results'
tau_tuple = [(2, 2), (4, 2), (4, 3), (4, 4), (5, 4)]
bf_gain_DNN_random_sensing = []
for idx in tau_tuple:
    results_path = 'DNN_random_tau' + str((idx[0], idx[1])) + '.mat'
    data = sio.loadmat(results_path)
    bf_gain_DNN_random_sensing.append(data['bf_gain_dB'][0][0])

'DNN trainable sensing results'
tau_tuple = [(2, 2), (4, 2), (4, 3), (4, 4), (5, 4)]
bf_gain_DNN_trainable_sensing = []
for idx in tau_tuple:
    results_path = 'DNN_trainable_sensing_wi_feedback_omp_model_tau' + str((idx[0], idx[1])) + '.mat'
    data = sio.loadmat(results_path)
    bf_gain_DNN_trainable_sensing.append(data['bf_gain_dB'][0][0])

'RNN network results'
taus = range(2, 12, 2)
bf_gain_RNN = []
bf_gain_opt = []
for tau in taus:
    data = sio.loadmat('RNN_tau_' + str(tau) + '.mat')
    bf_gain_RNN.append(data['bf_gain_dB'][0][0])
    bf_gain_opt.append(data['bf_gain_opt_dB'][0][0])

'Pingpong codebook'
tau=10
results_path = 'pingpong_L_SNR_tau_N1_N2_' + str((L, snrdB, tau, N1, N2)) + '.mat'
data = sio.loadmat(results_path)
bf_gain_codebook = data['bf_gain_pingpong_dB'][0]

'Plot results'
plt.figure(1)
taus = np.array(taus) * 2
plt.plot(taus, bf_gain_opt, '-', label='SVD w/ Perfect CSI')
plt.plot(taus, bf_gain_RNN, 's-', label='Proposed active sensing method')
plt.plot(taus, bf_gain_DNN_trainable_sensing, '<-', label='DNN-based design (learned sensing vectors)')
plt.plot(taus, bf_gain_DNN_random_sensing, 'v-', label='DNN-based design (random sensing vectors)')
plt.plot(taus, bf_gain_omp, 'o-', label='SVD w/ OMP channel estimation')
plt.plot(2*(np.arange(1,10)+1),bf_gain_codebook[1:],'*-',
         label='Codebook bisection search')
plt.xlabel('Pilot transmission overhead')
plt.ylabel('Average beamforming gain (dB)')
plt.ylim([7,32])
plt.grid()
plt.legend()
plt.savefig('bf_gain_hybrid.pdf', format='pdf', bbox_inches='tight')
plt.show()
