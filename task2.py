import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
#Load image file
img=mpimg.imread('image.JPG')
[r,g,b] = [img[:,:,i] for i in range(3)]

#Show image
fig = plt.figure(1)
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax1.imshow(img)
ax2.imshow(r, cmap = 'Reds')
ax3.imshow(g, cmap = 'Greens')
ax4.imshow(b, cmap = 'Blues')
plt.show()

#SVD
U_r, sigma_r, V_r = np.linalg.svd(r,full_matrices = True)
U_g, sigma_g, V_g = np.linalg.svd(g,full_matrices = True)
U_b, sigma_b, V_b = np.linalg.svd(b,full_matrices = True)
n_sigma = np.count_nonzero(sigma_r)
print("non zero element for sigma of r, g, b : ", n_sigma)
num_data, dim = r.shape

# Compress-Lower Resolution picture
#red image
sigma_r_30=np.zeros_like(sigma_r)
sigma_r_30[:30] = sigma_r[:30]
r_30 = np.matrix(U_r) * np.diag(sigma_r_30) * np.matrix(V_r[:num_data, :])

#green image
sigma_g_30=np.zeros_like(sigma_g)
sigma_g_30[:30] = sigma_g[:30]
g_30 = np.matrix(U_g) * np.diag(sigma_g_30) * np.matrix(V_g[:num_data, :])

#blue image
sigma_b_30=np.zeros_like(sigma_b)
sigma_b_30[:30] = sigma_b[:30]
b_30 = np.matrix(U_b) * np.diag(sigma_b_30) * np.matrix(V_b[:num_data, :])

#Show red, green, blue image.
fig = plt.figure(1)
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
ax1.imshow(r_30, cmap = 'Reds')
ax2.imshow(g_30, cmap = 'Greens')
ax3.imshow(b_30, cmap = 'Blues')
fig = plt.gcf()
fig.set_size_inches(15,15)
plt.suptitle('Lower Resolution, size=30')
plt.show()

# Compress-Higher Resolution picture
#red image
sigma_r_200=np.zeros_like(sigma_r)
sigma_r_200[:200] = sigma_r[:200]
r_200 = np.matrix(U_r) * np.diag(sigma_r_200) * np.matrix(V_r[:num_data, :])

#green image
sigma_g_200=np.zeros_like(sigma_g)
sigma_g_200[:200] = sigma_g[:200]
g_200 = np.matrix(U_g) * np.diag(sigma_g_200) * np.matrix(V_g[:num_data, :])

#blue image
sigma_b_200=np.zeros_like(sigma_b)
sigma_b_200[:200] = sigma_b[:200]
b_200 = np.matrix(U_b) * np.diag(sigma_b_200) * np.matrix(V_b[:num_data, :])

#Show red, green, blue image.
fig = plt.figure(1)
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
ax1.imshow(r_200, cmap = 'Reds')
ax2.imshow(g_200, cmap = 'Greens')
ax3.imshow(b_200, cmap = 'Blues')
fig = plt.gcf()
fig.set_size_inches(15,15)
plt.suptitle('Higher Resolution, size=200')
plt.show()