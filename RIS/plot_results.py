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

'results on perfect CSI'
data=sio.loadmat('results/bcd_perfectCSI.mat')
bf_gain_opt_dB = data['bf_gain_opt_dB'][0][0]
bf_gain_random_RIS_optimal_beamform_dB = data['bf_gain_random_RIS_optimal_beamform_dB'][0][0]

'results of deep learning'
tau_set = np.array([2,4,6,8,10])
bf_gain_RNN_both_control_RIS_final_RIS_from_both = []
bf_gain_DNN_trainable_sensing_both_control_RIS = []
bf_gain_RNN_both_control_RIS_final_RIS_from_both_trainable_RIS = []
bf_gain_RNN_random = []

for tau in tau_set:
    data = sio.loadmat('results/RNN_both_control_RIS_final_RIS_from_both' + str(tau) + '.mat')
    bf_gain_RNN_both_control_RIS_final_RIS_from_both.append(data['bf_gain_opt_dB'][0][0])
    data = sio.loadmat('results/DNN_trainable_sensing_both_control_RIS' + str(tau) + '.mat')
    bf_gain_DNN_trainable_sensing_both_control_RIS.append(data['bf_gain_opt_dB'][0][0])
    data = sio.loadmat('results/RNN_both_control_final_RIS_from_both_trainable_RIS_' + str(tau) + '.mat')
    bf_gain_RNN_both_control_RIS_final_RIS_from_both_trainable_RIS.append(data['bf_gain_opt_dB'][0][0])
    data = sio.loadmat('./results/RNN_random_RIS_' + str(tau) + '.mat')
    bf_gain_RNN_random.append(data['bf_gain_opt_dB'][0][0])
'Plot results'
plt.figure(1)
taus = tau_set * 2
plt.plot(taus, bf_gain_opt_dB*np.ones(len(taus)), '-', label='BCD w/ Perfect CSI')
plt.plot(taus, bf_gain_RNN_both_control_RIS_final_RIS_from_both, 's-',
         label='Active sensing beamformers + active sensing RIS coefficients')
plt.plot(taus,bf_gain_RNN_both_control_RIS_final_RIS_from_both_trainable_RIS,'<-',
         label='Active sensing beamformers + learned sensing RIS coefficients')
plt.plot(taus, bf_gain_RNN_random, 'v-',
         label='Active sensing beamformers + random sensing RIS coefficients')
plt.plot(taus, bf_gain_DNN_trainable_sensing_both_control_RIS,'o-', label='DNN-based design (learned sensing vectors)')
plt.plot(taus, bf_gain_random_RIS_optimal_beamform_dB*np.ones(len(taus)), '-',
         label='Optimal beamformers and random RIS w/ Perfect CSI')

plt.xlabel('Pilot transmission overhead')
plt.ylabel('Average beamforming gain (dB)')
plt.grid()
plt.legend(loc=2,bbox_to_anchor=(0.07,0.5))
plt.savefig('bf_gain_RIS.pdf', format='pdf', bbox_inches='tight')
plt.show()
