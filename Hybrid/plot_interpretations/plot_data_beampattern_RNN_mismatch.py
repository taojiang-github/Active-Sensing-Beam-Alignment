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

'load data'
tau = 6
data =sio.loadmat('RNN_interpretation_tau_' + str(tau) + '.mat')
bf_gain_opt_dB = data['bf_gain_opt_dB'][0][0]
bf_gain_RNN_dB = data['bf_gain_dB'][0][0]
w_Tx_t = data['w_Tx_t'][:,0:50,:,:]
w_Tx_r = data['w_Tx_r'][:,0:50,:,:]
w_Rx_r = data['w_Rx_r'][:,0:50,:,:]
w_Rx_t = data['w_Rx_t'][:,0:50,:,:]
w_Tx_final = data['w_Tx_final'][:,0:50,:,:]
w_Rx_final = data['w_Rx_final'][:,0:50,:,:]
G = data['G'][0:50,:,:]
phi_1_val = data['phi_1_val'][0:50,:]
phi_2_val = data['phi_2_val'][0:50,:]
phi_2_val = np.sort(phi_2_val,axis=-1) * 180 / np.pi
phi_1_val = np.sort(phi_1_val,axis=-1) * 180 / np.pi

plot_idx = 20

'discrete AoA/AoDs'
num_AoAs = 5000
phi_all = np.linspace(start=-90 * (np.pi / 180), stop=90 * (np.pi / 180), num=num_AoAs)
phi_all_degree = phi_all * 180 / np.pi
N1 = 64  # Number of BS's antennas
N2 = 32
A_dic1 = np.exp(1j * np.pi * np.reshape(np.arange(N1), (N1, 1)) * np.sin(phi_all))/np.sqrt(N1)
A_dic2 = np.exp(1j * np.pi * np.reshape(np.arange(N2), (N2, 1)) * np.sin(phi_all))/np.sqrt(N2)

'Array response of Rx_t'
w_Rx_t_her = np.transpose(w_Rx_t.conj(),[0,1,3,2])
bf_gain_Rx_t = 10 * np.log10(np.abs(w_Rx_t_her @ A_dic2) ** 2)
w_Rx_r_her = np.transpose(w_Rx_r.conj(),[0,1,3,2])
bf_gain_Rx_r = 10 * np.log10(np.abs(w_Rx_r_her @ A_dic2) ** 2)
w_Tx_t_her = np.transpose(w_Tx_t.conj(),[0,1,3,2])
bf_gain_Tx_t = 10 * np.log10(np.abs(w_Tx_t_her @ A_dic1) ** 2)
w_Tx_r_her = np.transpose(w_Tx_r.conj(),[0,1,3,2])
bf_gain_Tx_r = 10 * np.log10(np.abs(w_Tx_r_her @ A_dic1) ** 2)

[u, s, vh] = np.linalg.svd(G)
u_max = np.expand_dims(u[:, :, 0:3], axis=[0])
uh_max = np.transpose(u_max.conj(), [0, 1, 3, 2])
vh_max = np.expand_dims(vh[:, 0:3, :], axis=[0])
v_max = np.transpose(vh_max.conj(), [0, 1, 3, 2])

w_Tx_final_her = np.transpose(w_Tx_final.conj(),[0,1,3,2])
bf_gain_Tx_final = np.squeeze( 10 * np.log10(np.abs(w_Tx_final_her @ A_dic1) ** 2))
w_Rx_final_her = np.transpose(w_Rx_final.conj(),[0,1,3,2])
bf_gain_Rx_final = np.squeeze(10 * np.log10(np.abs(w_Rx_final_her @ A_dic2) ** 2))

w_Tx_opt_her =uh_max[0,:,0:1,:]
bf_gain_Tx_optimal = 10 * np.log10(np.abs(w_Tx_opt_her @ A_dic1) ** 2)
w_Rx_opt_her = vh_max[0,:,0:1,:]
bf_gain_Rx_optimal = 10 * np.log10(np.abs(w_Rx_opt_her @ A_dic2) ** 2)

w_Tx_opt_her =uh_max[0,:,1:2,:]
bf_gain_Tx_2optimal = 10 * np.log10(np.abs(w_Tx_opt_her @ A_dic1) ** 2)
w_Rx_opt_her = vh_max[0,:,1:2,:]
bf_gain_Rx_2optimal = 10 * np.log10(np.abs(w_Rx_opt_her @ A_dic2) ** 2)
w_Tx_opt_her =uh_max[0,:,2:3,:]
bf_gain_Tx_3optimal = 10 * np.log10(np.abs(w_Tx_opt_her @ A_dic1) ** 2)
w_Rx_opt_her = vh_max[0,:,2:3,:]
bf_gain_Rx_3optimal = 10 * np.log10(np.abs(w_Rx_opt_her @ A_dic2) ** 2)
'Array response'
fig1, axs1 = plt.subplots(1, 2, figsize=(10, 3))
fig1.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Angle $\\theta$ in degree',fontsize=14)
plt.ylabel('Normalized array response (dB)',labelpad=10,fontsize=14)

axs1[0].plot(phi_all_degree, bf_gain_Tx_final[plot_idx].squeeze(),'-', color='tab:green', lw=2, label='Final beam via active learning')
axs1[0].plot(phi_all_degree, bf_gain_Tx_optimal[plot_idx].squeeze(),'--', color='tab:orange', lw=2, label='First left-singular vector of $\\bm G$')
axs1[0].plot(phi_all_degree, bf_gain_Tx_2optimal[plot_idx].squeeze(), '-.', color='tab:red', lw=2,
                 label='Second left-singular vector of $\\bm G$')
axs1[0].set_ylim(bottom=-80, top=0)
axs1[0].set_title('Final beamformer at agent A: $\\bm{w}_{\\rm{t}}$',fontsize=14)
axs1[0].legend(loc=4,fontsize=14)

axs1[1].plot(phi_all_degree, bf_gain_Rx_final[plot_idx].squeeze(),'-', color='tab:green', lw=2, label='Final beam via active learning')
axs1[1].plot(phi_all_degree, bf_gain_Rx_optimal[plot_idx].squeeze(),'--', color='tab:orange', lw=2, label='First right-singular vector of $\\bm G$')
axs1[1].plot(phi_all_degree, bf_gain_Rx_2optimal[plot_idx].squeeze(), '-.', color='tab:red', lw=2,
                 label='Second right-singular vector of $\\bm G$')
axs1[1].set_ylim(bottom=-80, top=0)
axs1[1].set_title('Final beamformer at agent B: $\\bm{w}_{\\rm{r}}$',fontsize=14)
axs1[1].legend(loc=4,fontsize=14)

fig1.tight_layout()
plt.show()
fig1.savefig('bf_pattern_data'+str(tau)+'_'+str(plot_idx)+'.pdf', format='pdf', bbox_inches='tight')
