import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
#
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["New Century Schoolbook"],
    "text.latex.preamble": r"\usepackage{bm}"
})
plot_idx = 1

'load data'
data =sio.loadmat('bcd_perfectCSI1.mat')
v_np = data['v_np']/8
w_tx_bcd = data['w_tx']
w_rx_bcd = data['w_rx']

tau = 8  # Communication rounds
data =sio.loadmat('RNN_interpretation' + str(tau) + '.mat')
v_RNN = np.squeeze(data['ris_final']/8)
w_Tx_final = data['w_Tx_final'][0]
w_Rx_final = data['w_Rx_final'][0]

# data =sio.loadmat('DNN_trainable_interpretation' + str(tau) + '.mat')
# v_DNN_trainable = data['ris_final']/64
'discrete AoA/AoDs'
num_AoAs = 5000
phi_all = np.linspace(start=-90 * (np.pi / 180), stop=90 * (np.pi / 180), num=num_AoAs)
phi_all_degree = phi_all * 180 / np.pi
N = 64  # Number of BS's antennas
A_dic = np.exp(1j * np.pi * np.reshape(np.arange(N), (N, 1)) * np.sin(phi_all))/np.sqrt(N)
bf_gain = 10 * np.log10(np.abs(v_np@A_dic) ** 2)
bf_gain_RNN = 10 * np.log10(np.abs(v_RNN@A_dic) ** 2)
# bf_gain_DNN_trainable = 10 * np.log10(np.abs(v_DNN_trainable@A_dic) ** 2)

N1 = 64  # Number of BS's antennas
N2 = 32
A_dic1 = np.exp(1j * np.pi * np.reshape(np.arange(N1), (N1, 1)) * np.sin(phi_all))/np.sqrt(N1)
A_dic2 = np.exp(1j * np.pi * np.reshape(np.arange(N2), (N2, 1)) * np.sin(phi_all))/np.sqrt(N2)
w_Tx_final_her = np.transpose(w_Tx_final.conj(),[0,2,1])
bf_gain_Tx_final = np.squeeze(10 * np.log10(np.abs(w_Tx_final_her @ A_dic1) ** 2))
w_Rx_final_her = np.transpose(w_Rx_final.conj(),[0,2,1])
bf_gain_Rx_final = np.squeeze(10 * np.log10(np.abs(w_Rx_final_her @ A_dic2) ** 2))

w_Tx_final_her = np.transpose(w_tx_bcd.conj(),[0,2,1])
bf_gain_Tx_final_bcd = np.squeeze(10 * np.log10(np.abs(w_Tx_final_her @ A_dic1) ** 2))
w_Rx_final_her = np.transpose(w_rx_bcd.conj(),[0,2,1])
bf_gain_Rx_final_bcd = np.squeeze(10 * np.log10(np.abs(w_Rx_final_her @ A_dic2) ** 2))

fig1, axs1 = plt.subplots(1,3, figsize=(20, 3))
# fig1.add_subplot(111, frameon=False)
# plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
# plt.xlabel('Angle $\\theta$ in degree', fontsize=20)
# plt.ylabel('Normalized beamforming gain (dB)', labelpad=30, fontsize=20)

axs1[0].plot(phi_all_degree, bf_gain[plot_idx].squeeze(),color='tab:blue', lw=2, label='Proposed active learning')
axs1[0].plot(phi_all_degree, bf_gain_RNN[plot_idx].squeeze(),'--', color='tab:orange', lw=2, label='BCD w/ perfect CSI')
axs1[0].set_ylim(bottom=-80, top=0)
axs1[0].set_title('(a) Final RIS reflection coefficients: $\\bm v$')
axs1[0].set_xlabel('Angle $\\theta$ in degree')
axs1[0].set_ylabel('Normalized array response (dB)')
axs1[0].legend(loc=4)

axs1[1].plot(phi_all_degree, bf_gain_Tx_final_bcd[plot_idx].squeeze(), lw=2,  color='tab:blue',label='Proposed active learning')
axs1[1].plot(phi_all_degree, bf_gain_Tx_final[plot_idx].squeeze(), '--',color='tab:orange', lw=2,  label='BCD w/ perfect CSI')
axs1[1].set_xlabel('Angle $\\theta$ in degree')
axs1[1].set_ylabel('Normalized array response (dB)')
axs1[1].set_ylim(bottom=-80, top=0)
axs1[1].set_title('(b) Final beamformer at agent A: ${\\bm{w}}_{\\rm{t}}$')
axs1[1].legend(loc=4)


axs1[2].plot(phi_all_degree, bf_gain_Rx_final_bcd[plot_idx].squeeze(), lw=2, color='tab:blue', label='Proposed active learning')
axs1[2].plot(phi_all_degree, bf_gain_Rx_final[plot_idx].squeeze(),'--', color='tab:orange', lw=2, label='BCD w/ perfect CSI')
axs1[2].set_xlabel('Angle $\\theta$ in degree')
axs1[2].set_ylabel('Normalized array response (dB)')
axs1[2].set_title('(c) Final beamformer at agent B: $\\bm{w}_{\\rm{r}}$')
axs1[2].set_ylim(bottom=-80, top=0)
axs1[2].legend(loc=4)

fig1.tight_layout()
fig1.savefig('bf_final_pattern_RIS'+str(tau)+'_'+str(plot_idx)+'.pdf', format='pdf', bbox_inches='tight')
fig1.show()
#
# plt.figure()
# plt.title('RIS beam pattern')
# plt.plot(phi_all_degree,bf_gain_RNN[plot_idx].squeeze(),label='Active learning')
# plt.plot(phi_all_degree,bf_gain[plot_idx].squeeze(),'--',label='BCD')
# # plt.plot(phi_all_degree,bf_gain_DNN_trainable[plot_idx].squeeze(),label='DNN trainable')
# plt.legend()
# plt.xlabel('Effective Angle $\\theta$ in degree')
# plt.ylabel('Normalized beamforming gain (dB)')
# plt.show()
#
# plt.figure()
# plt.title('Tx beam pattern')
# plt.plot(phi_all_degree,bf_gain_Tx_final[plot_idx].squeeze(),label='Active learning')
# plt.plot(phi_all_degree,bf_gain_Tx_final_bcd[plot_idx].squeeze(),'--',label='BCD')
# # plt.plot(phi_all_degree,bf_gain_DNN_trainable[plot_idx].squeeze(),label='DNN trainable')
# plt.legend()
# plt.xlabel('Effective Angle $\\theta$ in degree')
# plt.ylabel('Normalized beamforming gain (dB)')
# plt.show()
#
# plt.figure()
# plt.title('Rx beam pattern')
# plt.plot(phi_all_degree,bf_gain_Rx_final[plot_idx].squeeze(),label='Active learning')
# plt.plot(phi_all_degree,bf_gain_Rx_final_bcd[plot_idx].squeeze(),'--',label='BCD')
# # plt.plot(phi_all_degree,bf_gain_DNN_trainable[plot_idx].squeeze(),label='DNN trainable')
# plt.legend()
# plt.xlabel('Effective Angle $\\theta$ in degree')
# plt.ylabel('Normalized beamforming gain (dB)')
# plt.show()