import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
#
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["New Century Schoolbook"],
# })

'RNN'
tau = 6
data = sio.loadmat('RNN_interpretation_tau_' + str(tau) + '.mat')
bf_gain_opt_dB = data['bf_gain_opt_dB'][0][0]
bf_gain_RNN_dB = data['bf_gain_dB'][0][0]
w_Tx_t = data['w_Tx_t']
w_Tx_r = data['w_Tx_r']
w_Rx_r = data['w_Rx_r']
w_Rx_t = data['w_Rx_t']
w_Tx_final = data['w_Tx_final']
w_Rx_final = data['w_Rx_final']
G = data['G']
# phi_1_val = data['phi_1_val']
# phi_2_val = data['phi_2_val']
# phi_2_val = np.sort(phi_2_val, axis=-1) * 180 / np.pi
# phi_1_val = np.sort(phi_1_val, axis=-1) * 180 / np.pi

# plot the array response to the effective channel
# G is the channel from Rx to Tx
[u, s, vh] = np.linalg.svd(G)
u_effective = np.expand_dims(u[:, :, 0:3], axis=[0])
uh_effective = np.transpose(u_effective.conj(), [0, 1, 3, 2])
vh_effective = np.expand_dims(vh[:, 0:3, :], axis=[0])
v_effective = np.transpose(vh_effective.conj(), [0, 1, 3, 2])

# bf_gain_Tx_final_effective = np.squeeze(10 * np.log10(np.abs(uh_effective @ w_Tx_final) ** 2))
# bf_gain_Rx_final_effective = np.squeeze(10 * np.log10(np.abs(vh_effective @ w_Rx_final) ** 2))
bf_gain_final = np.squeeze(10 * np.log10(np.abs((uh_effective @ w_Tx_final) * (vh_effective @ w_Rx_final)) ** 2))
bf_gain = 10 * np.log10(np.mean(np.abs(np.transpose(w_Tx_final[0].conj(), [0, 2, 1]) @ G @ w_Rx_final[0]) ** 2))
bf_gain_opt = 10 * np.log10(np.mean(np.abs(uh_effective[0,:,0:1,:] @ G @ v_effective[0,:,:,0:1]) ** 2))
print('RNN bf_gain, bf_gain_opt,gap:',bf_gain, bf_gain_opt, bf_gain.squeeze() - bf_gain_opt.squeeze())

idx = np.argmax(bf_gain_final,axis=1)
value = np.max(bf_gain_final,axis=1)

labels = 'First', 'Second', 'Third'
sizes = [np.sum(idx==0),  np.sum(idx==1),  np.sum(idx==2)]
fig2, ax2 = plt.subplots(2,4,figsize=(20, 5))
ax2[0,0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,textprops={'fontsize': 14})
ax2[0,0].axis('equal')
ax2[0,0].set_title('(a) Proposed active learning',fontsize=14)
ax2[0,1].hist(value[idx==0],bins=np.arange(-20,1), density=True)
ax2[0,1].set_xlabel('Normalized beamforming gain (dB)',fontsize=14)
ax2[0,1].set_ylabel('Empirical probability',fontsize=14)
ax2[0,1].set_title('(b) Matching the 1st-singular vector pair (Proposed)',fontsize=14)
ax2[0,2].hist(value[idx==1],bins=np.arange(-20,1),density=True)
ax2[0,2].set_xlabel('Normalized beamforming gain (dB)',fontsize=14)
ax2[0,2].set_ylabel('Empirical probability',fontsize=14)
ax2[0,2].set_title('(c) Matching the 2nd-singular vector pair (Proposed)',fontsize=14)
ax2[0,3].hist(value[idx==2],bins=np.arange(-20,1), density=True)
ax2[0,3].set_xlabel('Normalized beamforming gain (dB)',fontsize=14)
ax2[0,3].set_ylabel('Empirical probability',fontsize=14)
ax2[0,3].set_title('(d) Matching the 3rd-singular vector pair (Proposed)',fontsize=14)



'DNN trainable'
tau1,tau2 = 4,3
data = sio.loadmat('DNN_trainable_interpretation_tau_' + str(tau1)+str(tau2) + '.mat')
bf_gain_opt_dB = data['bf_gain_opt_dB'][0][0]
bf_gain_DNN_dB = data['bf_gain_dB'][0][0]
print('DNN traibale bf_gain,bf_gain_opt,gap:',bf_gain_DNN_dB, bf_gain_opt_dB, bf_gain_DNN_dB.squeeze() - bf_gain_opt_dB.squeeze())

w_Tx_final = data['w_Tx_final']
w_Rx_final = data['w_Rx_final']
G = data['G']
[u, s, vh] = np.linalg.svd(G)
u_effective = np.expand_dims(u[:, :, 0:3], axis=[0])
uh_effective = np.transpose(u_effective.conj(), [0, 1, 3, 2])
vh_effective = np.expand_dims(vh[:, 0:3, :], axis=[0])
v_effective = np.transpose(vh_effective.conj(), [0, 1, 3, 2])
bf_gain_final_DNN = np.squeeze(10 * np.log10(np.abs((uh_effective @ w_Tx_final) * (vh_effective @ w_Rx_final)) ** 2))

idx = np.argmax(bf_gain_final_DNN,axis=1)
value = np.max(bf_gain_final_DNN,axis=1)

labels = 'First', 'Second', 'Third'
sizes = [np.sum(idx==0),  np.sum(idx==1),  np.sum(idx==2)]
ax2[1,0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,textprops={'fontsize': 14})
ax2[1,0].axis('equal')
ax2[1,0].set_title('(e) DNN-based design (learned sensing vectors)',fontsize=14)

ax2[1,1].hist(value[idx==0],bins=np.arange(-20,1), density=True)
ax2[1,1].set_xlabel('Normalized beamforming gain (dB)',fontsize=14)
ax2[1,1].set_ylabel('Empirical probability')
ax2[1,1].set_title('(f) Matching the 1st-singular vector pair (DNN)',fontsize=14)
ax2[1,2].hist(value[idx==1],bins=np.arange(-20,1),density=True)
ax2[1,2].set_xlabel('Normalized beamforming gain (dB)',fontsize=14)
ax2[1,2].set_ylabel('Empirical probability',fontsize=14)
ax2[1,2].set_title('(g) Matching the 2nd-singular vector pair (DNN)',fontsize=14)
ax2[1,3].hist(value[idx==2],bins=np.arange(-20,1), density=True)
ax2[1,3].set_xlabel('Normalized beamforming gain (dB)',fontsize=14)
ax2[1,3].set_ylabel('Empirical probability',fontsize=14)
ax2[1,3].set_title('(h) Matching the 3rd-singular vector pair (DNN)',fontsize=14)

fig2.tight_layout()
fig2.savefig('bf_gain_pie'+str(tau)+'.pdf', format='pdf', bbox_inches='tight')

plt.show()

#
# fig6, axs6 = plt.subplots(2, 2, figsize=(8, 8))
# fig6.add_subplot(111, frameon=False)
# plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
# plt.xlabel('Number of ping-pong rounds', fontsize=15)
# plt.ylabel('Normalized beamforming gain (dB)', labelpad=20, fontsize=15)
#
# bf_gain_Rx_t_effective = np.squeeze(10 * np.log10(np.abs(vh_effective @ w_Rx_t) ** 2))
# bf_gain_Rx_r_effective = np.squeeze(10 * np.log10(np.abs(vh_effective @ w_Rx_r) ** 2))
# bf_gain_Tx_t_effective = np.squeeze(10 * np.log10(np.abs(uh_effective @ w_Tx_t) ** 2))
# bf_gain_Tx_r_effective = np.squeeze(10 * np.log10(np.abs(uh_effective @ w_Tx_r) ** 2))
# bf_gain_Tx_Rx = bf_gain_Tx_t_effective + bf_gain_Rx_r_effective
# bf_gain_Rx_Tx = bf_gain_Tx_r_effective + bf_gain_Rx_t_effective
#
# 'bf_gain_Tx_r'
# axs6[0, 0].plot(np.mean(bf_gain_Tx_r_effective[:, :, 0],axis=1), 'o-', label='First')
# axs6[0, 0].plot(np.mean(bf_gain_Tx_r_effective[:, :, 1],axis=1), 's-', label='Second')
# axs6[0, 0].plot(np.mean(bf_gain_Tx_r_effective[:, :, 2],axis=1), 'v-', label='Third')
# axs6[0, 0].legend()
# axs6[0, 0].set_ylim(bottom=-25, top=-5)
# axs6[0, 0].set_title('Tx r')
#
# 'bf_gain_Tx_t'
# axs6[0, 1].plot(np.mean(bf_gain_Tx_t_effective[:, :, 0],axis=1), 'o-', label='First')
# axs6[0, 1].plot(np.mean(bf_gain_Tx_t_effective[:, :, 1],axis=1), 's-', label='Second')
# axs6[0, 1].plot(np.mean(bf_gain_Tx_t_effective[:, :, 2],axis=1), 'v-', label='Third')
# axs6[0, 1].legend()
# axs6[0, 1].set_ylim(bottom=-25, top=-5)
# axs6[0, 1].set_title('Tx t')
#
# 'bf_gain_Rx_r'
# axs6[1, 0].plot(np.mean(bf_gain_Rx_r_effective[:, :, 0],axis=1), 'o-', label='First')
# axs6[1, 0].plot(np.mean(bf_gain_Rx_r_effective[:, :, 1],axis=1), 's-', label='Second')
# axs6[1, 0].plot(np.mean(bf_gain_Rx_r_effective[:, :, 2],axis=1), 'v-', label='Third')
# axs6[1, 0].legend()
# axs6[1, 0].set_ylim(bottom=-25, top=-5)
# axs6[1, 0].set_title('Rx r')
#
# 'bf_gain_Rx_t'
# axs6[1, 1].plot(np.mean(bf_gain_Rx_t_effective[:, :, 0],axis=1), 'o-', label='First')
# axs6[1, 1].plot(np.mean(bf_gain_Rx_t_effective[:, :, 1],axis=1), 's-', label='Second')
# axs6[1, 1].plot(np.mean(bf_gain_Rx_t_effective[:, :, 2],axis=1), 'v-', label='Third')
# axs6[1, 1].legend()
# axs6[1, 1].set_ylim(bottom=-25, top=-5)
# axs6[1, 1].set_title('Rx t')
# fig6.tight_layout()
#
# fig6.savefig('bf_gain_interpretation'+str(tau)+'.pdf', format='pdf', bbox_inches='tight')