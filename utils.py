import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# session config
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0", # specify GPU number
        allow_growth=False
    )
)

# calculate total parameters
def cal_parameter():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    return print('Total params: %d ' % total_parameters)

# calculate jaccard
def jaccard(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.double(np.bitwise_and(im1, im2).sum()) / np.double(np.bitwise_or(im1, im2).sum())

# calculate L1
def L1norm(im1, im2):
    im1 = np.asarray(im1)
    im2 = np.asarray(im2)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.double(np.mean(abs(im1 - im2)))

def matplotlib_plt(X, filename):
    fig = plt.figure()
    plt.title('latent distribution')
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel('dim_1')
    ax.set_ylabel('dim_2')
    ax.set_zlabel('dim_3')
    ax.scatter(X[:,0], X[:,1], X[:,2] , marker="x"
               # , c=y/len(set(y))
    )
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.savefig(filename + "3D/{:03d}.jpg".format(angle))
    # plt.savefig(filename)
    # plt.show()
