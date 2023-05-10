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

[u, s_Tx, vh] = np.linalg.svd(G_Tx)
[u, s_Rx, vh] = np.linalg.svd(G_Rx)
largest_eig = []
largest_eig.append(s_random[0])
for ii in range(tau):
    largest_eig.append(s_Tx[ii,0])
    largest_eig.append(s_Rx[ii,0])
largest_eig.append(s_final[0])
fig0 = plt.plot(np.arange(tau*2+2),largest_eig,'o-')
plt.xlabel('Pilot transmission')
plt.ylabel('Largest eigenvalue of the combined channel')
plt.show()
# w_Tx_t_opt_her =uh_max[0,:,1:2,:]
# bf_gain_Tx_2optimal = 10 * np.log10(np.abs(w_Tx_t_opt_her @ A_dic1) ** 2)
# w_Rx_opt_her = vh_max[0,:,1:2,:]
# bf_gain_Rx_2optimal = 10 * np.log10(np.abs(w_Rx_opt_her @ A_dic2) ** 2)
# w_Tx_opt_her =uh_max[0,:,2:3,:]
# bf_gain_Tx_3optimal = 10 * np.log10(np.abs(w_Tx_opt_her @ A_dic1) ** 2)
# w_Rx_opt_her = vh_max[0,:,2:3,:]
# bf_gain_Rx_3optimal = 10 * np.log10(np.abs(w_Rx_opt_her @ A_dic2) ** 2)
'Array response'
fig1, axs1 = plt.subplots(tau,2, figsize=(10, 15))
fig1.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('Effective Angle $\\theta$ in degree', fontsize=20)
plt.ylabel('Normalized beamforming gain (dB)', labelpad=30, fontsize=20)
for ii in range(tau):
    axs1[ii,0].plot(phi_all_degree, bf_gain_tx_opt[ii].squeeze(),'--', color='tab:orange', lw=2, label='Optimal')
    axs1[ii,0].plot(phi_all_degree, bf_gain_tx[ii].squeeze(),color='tab:blue', lw=2, label='Active sensing')
    axs1[ii,0].set_ylim(bottom=-80, top=0)
    axs1[ii,0].set_title('Tx to Rx in %d-th round ' % (ii))
    axs1[ii,0].legend(loc=4)

    axs1[ii,1].plot(phi_all_degree, bf_gain_rx_opt[ii].squeeze(), '--',color='tab:orange', lw=2,  label='Optimal')
    axs1[ii,1].plot(phi_all_degree, bf_gain_rx[ii].squeeze(), lw=2,  color='tab:blue',label='Active sensing')
    axs1[ii,1].set_ylim(bottom=-80, top=0)
    axs1[ii,1].set_title('Rx to Tx in %d-th round' % (ii))
    axs1[ii,1].legend(loc=4)

fig1.tight_layout()
# fig1.savefig('ris_sensing_pattern'+str(tau)+'_'+str(plot_idx)+'.pdf', format='pdf', bbox_inches='tight')
fig1.show()