# mask for white areas in image
#"""
#mask = cv2.inRange(view['img'], 10, 100)
#masked = cv2.bitwise_and(view['img'],view['img'],mask=mask)
#result = view['img'] - masked
view = views[0]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 10))
ax1.imshow(view['img'], cmap='gray')
ax2.imshow(view['masked'], cmap='gray')
ax3.imshow(view['hsv_masked'], cmap='gray')
plt.show()
#"""
#view['masked']

########################################################################################################################
########################################################################################################################
########################################################################################################################

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
#%%
nemo = cv2.cvtColor(view['frame'], cv2.COLOR_BGR2RGB)

r, g, b = cv2.split(nemo)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

pixel_colors = nemo.reshape((np.shape(nemo)[0]*np.shape(nemo)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()
#%%
hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_nemo)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()
#%%
light_orange = (0, 0, 200)
dark_orange = (145, 60, 255)

mask = cv2.inRange(hsv_nemo, light_orange, dark_orange)
result = cv2.bitwise_and(nemo, nemo, mask=mask)
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()

########################################################################################################################
########################################################################################################################
########################################################################################################################


########################################################################################################################
########################################################################################################################
########################################################################################################################