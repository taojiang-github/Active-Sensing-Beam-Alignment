import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["New Century Schoolbook"],
# })

'OMP results'
snrdB, N1, N2 = 0, 64, 32
idx = (4,3)
L_set = [1,2,3,4,5]
bf_gain_omp = []
bf_gain_DNN_trainable_sensing = []
bf_gain_RNN = []
bf_gain_opt = []
bf_gain_codebook = []
bf_gain_DNN_random_sensing = []
for L in L_set:
    results_path = 'OMP_L_SNR_tau1_tau2_N1_N2_' + str((L, snrdB, idx[0], idx[1], N1, N2)) + '.mat'
    data = sio.loadmat(results_path)
    bf_gain_omp.append(data['bf_gain_omp_dB'][0][0])

    results_path = 'DNN_trainable_sensing_generalize_tau' + str((idx[0],
                                                                 idx[1])) \
                   +str(L)+ '.mat'
    data = sio.loadmat(results_path)
    bf_gain_DNN_trainable_sensing.append(data['bf_gain_dB'][0][0])

    results_path ='DNN_random_tau_generalize' + str((idx[0], idx[1])) + str(L)+ '.mat'
    data = sio.loadmat(results_path)
    bf_gain_DNN_random_sensing.append(data['bf_gain_dB'][0][0])

    data = sio.loadmat('RNN_generalize_tau_'+str(6)+'_'+str(L)+'.mat')
    bf_gain_RNN.append(data['bf_gain_dB'][0][0])
    bf_gain_opt.append(data['bf_gain_opt_dB'][0][0])

    results_path = 'pingpong_L_SNR_tau_N1_N2_' + str(
        (L, snrdB, 6, N1, N2)) + '.mat'
    data = sio.loadmat(results_path)
    bf_gain_codebook.append(data['bf_gain_pingpong_dB'][0][-1])


'Plot results'
plt.figure(1)
plt.plot(L_set, bf_gain_opt, '-', label='SVD w/ Perfect CSI')
plt.plot(L_set, bf_gain_RNN, 's-', label='Proposed active sensing method')
plt.plot(L_set, bf_gain_DNN_trainable_sensing, '<-', label='DNN-based design (learned sensing vectors)')
plt.plot(L_set, bf_gain_DNN_random_sensing, 'v-', label='DNN-based design (random sensing vectors)')
plt.plot(L_set, bf_gain_omp, 'o-', label='SVD w/ OMP channel estimation')
plt.plot(L_set,bf_gain_codebook,'*-',
         label='Codebook bisection search')
plt.xlabel('Number of paths')
plt.ylabel('Average beamforming gain (dB)')
plt.grid()
plt.legend()
plt.ylim([6,34])
plt.xticks(L_set)
plt.savefig('bf_gain_hybrid_generalize_L.pdf', format='pdf',
            bbox_inches='tight')
plt.show()
