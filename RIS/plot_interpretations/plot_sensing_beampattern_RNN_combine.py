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
tau = 8
plot_idx = 1
data =sio.loadmat('RNN_interpretation' + str(tau) + '.mat')
bf_gain_RNN_dB = data['bf_gain_opt_dB'][0][0]
w_Tx_t = data['w_Tx_t'][:,plot_idx,:,:]
w_Tx_r = data['w_Tx_r'][:,plot_idx,:,:]
w_Rx_r = data['w_Rx_r'][:,plot_idx,:,:]
w_Rx_t = data['w_Rx_t'][:,plot_idx,:,:]
w_Tx_final = data['w_Tx_final'][:,plot_idx,:,:]
w_Rx_final = data['w_Rx_final'][:,plot_idx,:,:]
G_Tx = data['G_Tx'][:,plot_idx,:,:]
G_Rx = data['G_Rx'][:,plot_idx,:,:]
# G_final = data['G_final'][0,plot_idx,:,:]
# [u,s,v] = np.linalg.svd(G_final)

'discrete AoA/AoDs'
num_AoAs = 5000
phi_all = np.linspace(start=-90 * (np.pi / 180), stop=90 * (np.pi / 180), num=num_AoAs)
phi_all_degree = phi_all * 180 / np.pi
N1 = 64  # Number of BS's antennas
N2 = 32
A_dic1 = np.exp(1j * np.pi * np.reshape(np.arange(N1), (N1, 1)) * np.sin(phi_all))/np.sqrt(N1)
A_dic2 = np.exp(1j * np.pi * np.reshape(np.arange(N2), (N2, 1)) * np.sin(phi_all))/np.sqrt(N2)

'Array response of Rx_t'
w_Rx_t_her = np.transpose(w_Rx_t.conj(),[0,2,1])
bf_gain_Rx_t = 10 * np.log10(np.abs(w_Rx_t_her @ A_dic2) ** 2)
w_Rx_r_her = np.transpose(w_Rx_r.conj(),[0,2,1])
bf_gain_Rx_r = 10 * np.log10(np.abs(w_Rx_r_her @ A_dic2) ** 2)
w_Tx_t_her = np.transpose(w_Tx_t.conj(),[0,2,1])
bf_gain_Tx_t = 10 * np.log10(np.abs(w_Tx_t_her @ A_dic1) ** 2)
w_Tx_r_her = np.transpose(w_Tx_r.conj(),[0,2,1])
bf_gain_Tx_r = 10 * np.log10(np.abs(w_Tx_r_her @ A_dic1) ** 2)

[u, s_Tx, vh] = np.linalg.svd(G_Tx)
u_max = u[:, :, 0:1]
uh_Tx_t = np.transpose(u_max.conj(), [0, 2, 1])
v_Rx_r = vh[:,0:1, :]

[u, s_Rx, vh] = np.linalg.svd(G_Rx)
u_max = u[:, :, 0:1]
uh_Tx_r = np.transpose(u_max.conj(), [0, 2, 1])
v_Rx_t = vh[:,0:1, :]

bf_gain_Tx_t_optimal = 10 * np.log10(np.abs(uh_Tx_t @ A_dic1) ** 2)
bf_gain_Rx_r_optimal = 10 * np.log10(np.abs(v_Rx_r @ A_dic2) ** 2)
bf_gain_Tx_r_optimal = 10 * np.log10(np.abs(uh_Tx_r @ A_dic1) ** 2)
bf_gain_Rx_t_optimal = 10 * np.log10(np.abs(v_Rx_t @ A_dic2) ** 2)

w_Tx_final_her = np.transpose(w_Tx_final.conj(),[0,2,1])
bf_gain_Tx_final = np.squeeze(10 * np.log10(np.abs(w_Tx_final_her @ A_dic1) ** 2))
w_Rx_final_her = np.transpose(w_Rx_final.conj(),[0,2,1])
bf_gain_Rx_final = np.squeeze(10 * np.log10(np.abs(w_Rx_final_her @ A_dic2) ** 2))

'RIS'
data =sio.loadmat('RNN_interpretation' + str(tau) + '.mat')
bf_gain_RNN_dB = data['bf_gain_opt_dB'][0][0]
ris_sensing_tx = data['ris_sensing_Tx'][:,plot_idx,:]/8
ris_sensing_rx = data['ris_sensing_rx'][:,plot_idx,:]/8

w_Tx_t = data['w_Tx_t'][:,plot_idx,:,:]
w_Tx_r = data['w_Tx_r'][:,plot_idx,:,:]
w_Rx_r = data['w_Rx_r'][:,plot_idx,:,:]
w_Rx_t = data['w_Rx_t'][:,plot_idx,:,:]
w_Tx_final = data['w_Tx_final'][:,plot_idx,:,:]
w_Rx_final = data['w_Rx_final'][:,plot_idx,:,:]
G_Tx = data['G_Tx'][:,plot_idx,:,:]
G_Rx = data['G_Rx'][:,plot_idx,:,:]
A = data['A'][plot_idx,:,:]
B = data['B'][plot_idx,:,:]
G_final = np.squeeze(data['G_final'][:,plot_idx,:,:])
G_random = np.squeeze(data['G_random_ris'][:,plot_idx,:,:])
[u,s_final,v] = np.linalg.svd(G_final)
[u,s_random,v] = np.linalg.svd(G_random)

'discrete AoA/AoDs'
num_AoAs = 5000
phi_all = np.linspace(start=-90 * (np.pi / 180), stop=90 * (np.pi / 180), num=num_AoAs)
phi_all_degree = phi_all * 180 / np.pi
N1 = 64  # Number of BS's antennas
A_dic1 = np.exp(1j * np.pi * np.reshape(np.arange(N1), (N1, 1)) * np.sin(phi_all))/np.sqrt(N1)

bf_gain_tx = np.squeeze(10*np.log10(np.abs(np.expand_dims(ris_sensing_tx.conj(),axis=1)@A_dic1)**2))
bf_gain_rx = np.squeeze(10*np.log10(np.abs(np.expand_dims(ris_sensing_rx.conj(),axis=1)@A_dic1)**2))

tmp1 = np.reshape(np.transpose(np.conj(w_Tx_t), [0, 2, 1]) @ A, [-1, N1])
tmp2 = np.reshape(B @ w_Rx_r, [-1, N1])
opt_angle = -np.angle(tmp1 * tmp2)
v_tx_opt = (np.cos(opt_angle)+1j* np.sin(opt_angle))/np.sqrt(N1)
bf_gain_tx_opt = np.squeeze(10*np.log10(np.abs(np.expand_dims(v_tx_opt.conj(),axis=1)@A_dic1)**2))

tmp1 = np.reshape(np.transpose(np.conj(w_Tx_r), [0, 2, 1]) @ A, [-1, N1])
tmp2 = np.reshape(B @ w_Rx_t, [-1, N1])
opt_angle = -np.angle(tmp1 * tmp2)
v_rx_opt = (np.cos(opt_angle)+1j* np.sin(opt_angle))/np.sqrt(N1)
bf_gain_rx_opt = np.squeeze(10*np.log10(np.abs(np.expand_dims(v_rx_opt.conj(),axis=1)@A_dic1)**2))


'Array response'
fig1, axs1 = plt.subplots(tau,6, figsize=(20, 15))
fig1.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Angle $\\theta$ in degree', fontsize=20)
plt.ylabel('Normalized array response (dB)', labelpad=30, fontsize=20)
for ii in range(tau):
    axs1[ii,0].plot(phi_all_degree, bf_gain_Rx_t_optimal[ii,0].squeeze(),'--', color='tab:orange', lw=2, label='Channel matching')
    axs1[ii,0].plot(phi_all_degree, bf_gain_Rx_t[ii, 0].squeeze(),color='tab:blue', lw=2, label='Sensing via active learning')
    axs1[ii,0].set_ylim(bottom=-80, top=0)
    axs1[ii,0].set_title('Transmit beamformer at agent B: $\\bm{w}^{\\rm{B}}_{\\rm{t}, %d}$' % (ii))
    axs1[ii,0].legend(loc=4)

    axs1[ii,1].plot(phi_all_degree, bf_gain_Rx_r_optimal[ii,0].squeeze(), '--',color='tab:orange', lw=2,  label='Channel matching')
    axs1[ii,1].plot(phi_all_degree, bf_gain_Rx_r[ii, 0].squeeze(), lw=2,  color='tab:blue',label='Sensing via active learning')
    axs1[ii,1].set_ylim(bottom=-80, top=0)
    axs1[ii,1].set_title('Receive beamformer at agent B: ${\\bm{w}}^{\\rm{B}}_{\\rm{r}, %d}$' % (ii))
    axs1[ii,1].legend(loc=4)


    axs1[ii,2].plot(phi_all_degree, bf_gain_Tx_t_optimal[ii,0].squeeze(),'--', color='tab:orange', lw=2, label='Channel matching')
    axs1[ii,2].plot(phi_all_degree, bf_gain_Tx_t[ii, 0].squeeze(), lw=2, color='tab:blue', label='Sensing via active learning')
    axs1[ii,2].set_title('Transmit beamformer at agent A: $\\bm{w}^{\\rm A}_{\\rm{t}, %d}$' % (ii))
    axs1[ii, 2].set_ylim(bottom=-80, top=0)
    axs1[ii, 2].legend(loc=4)


    axs1[ii,3].plot(phi_all_degree, bf_gain_Tx_r_optimal[ii,0].squeeze(),'--', color='tab:orange', lw=2, label='Channel matching')
    axs1[ii,3].plot(phi_all_degree, bf_gain_Tx_r[ii, 0].squeeze(), lw=2, color='tab:blue', label='Sensing via active learning')
    axs1[ii, 3].set_ylim(bottom=-80, top=0)
    axs1[ii,3].set_title('Receive beamformer at agent A: ${\\bm{w}}^{\\rm A}_{\\rm{r}, %d}$' % (ii))
    axs1[ii, 3].legend(loc=4)

    axs1[ii,4].plot(phi_all_degree, bf_gain_tx_opt[ii].squeeze(),'--', color='tab:orange', lw=2, label='Phase matching')
    axs1[ii,4].plot(phi_all_degree, bf_gain_tx[ii].squeeze(),color='tab:blue', lw=2, label='Sensing via active learning')
    axs1[ii,4].set_ylim(bottom=-80, top=0)
    axs1[ii,4].set_title('RIS reflection coefficients: $\\bm{v}_{%d}$ ' % (ii))
    axs1[ii,4].legend(loc=4)

    axs1[ii,5].plot(phi_all_degree, bf_gain_rx_opt[ii].squeeze(), '--',color='tab:orange', lw=2,  label='Phase matching')
    axs1[ii,5].plot(phi_all_degree, bf_gain_rx[ii].squeeze(), lw=2,  color='tab:blue',label='Active sensing')
    axs1[ii,5].set_ylim(bottom=-80, top=0)
    axs1[ii,5].set_title('RIS reflection coefficients: $\\tilde{\\bm{v}}_{%d}$ ' % (ii))
    axs1[ii,5].legend(loc=4)


fig1.tight_layout()
fig1.savefig('bf_sensing_pattern_RIS'+str(tau)+'_'+str(plot_idx)+'.pdf', format='pdf', bbox_inches='tight')
fig1.show()