'''
# Project test_data/training data into latent space
# Author: Zhihui Lu
# Date: 2018/10/17
'''
import tensorflow as tf
import os
import argparse
import numpy as np
from tqdm import tqdm
import dataIO as io
from network import cnn_encoder, cnn_decoder, mnist_decoder, mnist_encoder, encoder, decoder, deepest_encoder, deepest_decoder, encoder_r, decoder_r,\
    shallow_encoder, shallow_decoder, deep_encoder, deep_decoder, deeper_encoder, deeper_decoder
from model import Variational_Autoencoder
import utils
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #for windows

def main():

    # tf flag
    flags = tf.flags
    flags.DEFINE_string("test_data_txt", "./input/CT/patch/test.txt", "i1")
    flags.DEFINE_string("model", './output/CT/patch/deep/z20/model/model_{}'.format(4000), "i2")
    flags.DEFINE_string("outdir", "./output/CT/patch/deep/z20/latent/", "i3")
    flags.DEFINE_float("beta", 1, "hyperparameter beta")
    flags.DEFINE_integer("num_of_test", 607, "number of test data")
    flags.DEFINE_integer("batch_size", 1, "batch size")
    flags.DEFINE_integer("latent_dim", 20, "latent dim")
    flags.DEFINE_list("image_size", [9 * 9 * 9], "image size")
    FLAGS = flags.FLAGS

    # check folder
    if not (os.path.exists(FLAGS.outdir)):
        os.makedirs(FLAGS.outdir)


    # read list
    test_data_list = io.load_list(FLAGS.test_data_txt)

    # test step
    test_step = FLAGS.num_of_test // FLAGS.batch_size
    if FLAGS.num_of_test % FLAGS.batch_size != 0:
        test_step += 1

    # load test data
    test_set = tf.data.TFRecordDataset(test_data_list)
    test_set = test_set.map(lambda x: _parse_function(x, image_size=FLAGS.image_size),
                            num_parallel_calls=os.cpu_count())
    test_set = test_set.batch(FLAGS.batch_size)
    test_iter = test_set.make_one_shot_iterator()
    test_data = test_iter.get_next()


    # initializer
    init_op = tf.group(tf.initializers.global_variables(),
                       tf.initializers.local_variables())

    with tf.Session(config = utils.config) as sess:

        # set network
        kwargs = {
            'sess': sess,
            'outdir': FLAGS.outdir,
            'beta': FLAGS.beta,
            'latent_dim': FLAGS.latent_dim,
            'batch_size': FLAGS.batch_size,
            'image_size': FLAGS.image_size,
            'encoder': deep_encoder,
            'decoder': deep_decoder
        }
        VAE = Variational_Autoencoder(**kwargs)

        sess.run(init_op)

        patch_side = 9
        patch_center = int(patch_side / 2)

        # testing
        VAE.restore_model(FLAGS.model)
        tbar = tqdm(range(test_step), ascii=True)
        preds = []
        ori = []
        latent_space = []
        for k in tbar:
            test_data_batch = sess.run(test_data)
            ori_single = test_data_batch
            z = VAE.plot_latent(ori_single)
            z = z.flatten()
            latent_space.append(z)

        latent_space = np.asarray(latent_space)
        print("latent_space =",latent_space.shape)
        # print(latent_space[0])
        # print(latent_space[1])
        # print(latent_space[2])
        # print(latent_space[3])
        # print(latent_space[4])

        plt.figure(figsize=(8, 6))
        fig = plt.scatter(latent_space[:, 0], latent_space[:, 1])
        plt.xlabel('dim_1')
        plt.ylabel('dim_2')
        plt.title('latent distribution')
        plt.savefig(FLAGS.outdir + "latent_space.png")

        if FLAGS.latent_dim == 3 :
            if not (os.path.exists(FLAGS.outdir + "3D/")):
                os.makedirs(FLAGS.outdir + "3D/")
            utils.matplotlib_plt(latent_space, FLAGS.outdir)
            # check folder

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection="3d")
            # ax.scatter(latent_space[:, 0], latent_space[:, 1], latent_space[:, 2], marker="x")
            # ax.scatter(latent_space[:5, 0], latent_space[:5, 1], latent_space[:5, 2], marker="o", color='orange')


        plt.figure(figsize=(8, 6))
        plt.scatter(latent_space[:, 0], latent_space[:, 1])
        plt.scatter(latent_space[:5, 0],latent_space[:5, 1], color='orange')
        plt.title('latent distribution')
        plt.xlabel('dim_1')
        plt.ylabel('dim_2')
        plt.savefig(FLAGS.outdir + "back_projection.png")
        # plt.show()

        #### display a 2D manifold of digits
        n = 13
        digit_size = patch_side
        figure1 = np.zeros((digit_size * n, digit_size * n))
        figure2 = np.zeros((digit_size * n, digit_size * n))
        figure3 = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-3, 3, n)
        grid_y = np.linspace(-3, 3, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = []

                if FLAGS.latent_dim == 2:
                    z_sample = np.array([[xi, yi]])
                if FLAGS.latent_dim == 3:
                    z_sample = np.array([[xi, yi, 0]])
                if FLAGS.latent_dim == 4:
                    z_sample = np.array([[xi, yi, 0, 0]])
                if FLAGS.latent_dim == 7:
                    z_sample = np.array([[xi, yi, 0, 0,0,0,0]])
                if FLAGS.latent_dim == 20:
                    z_sample = np.array([[xi, yi,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

                x_decoded = VAE.generate_sample(z_sample)
                generate_data = x_decoded[0].reshape(digit_size, digit_size, digit_size)
                digit_axial = generate_data[patch_center, :, :]
                digit_coronal = generate_data[:, patch_center, :]
                digit_sagital = generate_data[:, :, patch_center]
                digit1 = np.reshape(digit_axial, [patch_side, patch_side])
                digit2 = np.reshape(digit_coronal, [patch_side, patch_side])
                digit3 = np.reshape(digit_sagital, [patch_side, patch_side])
                # plt.imshow(digit, cmap='Greys_r')
                # plt.savefig(str(i) + '@' + str(j) + 'fig.png')
                figure1[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit1
                figure2[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit2
                figure3[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit3

        # set graph
        start_range = digit_size // 2
        end_range = n * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)

        # axial
        plt.figure(figsize=(10, 10))
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure1, cmap='Greys_r')
        plt.savefig(FLAGS.outdir + "digit_axial.png")
        # plt.show()

        # coronal
        plt.figure(figsize=(10, 10))
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure2, cmap='Greys_r')
        plt.savefig(FLAGS.outdir + "digit_coronal.png")
        # plt.show()

        # sagital
        plt.figure(figsize=(10, 10))
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure3, cmap='Greys_r')
        plt.savefig(FLAGS.outdir + "digit_sagital.png")
        # plt.show()


# # load tfrecord function
def _parse_function(record, image_size=[9, 9, 9]):
    keys_to_features = {
        'img_raw': tf.FixedLenFeature(np.prod(image_size), tf.float32),
    }
    parsed_features = tf.parse_single_example(record, keys_to_features)
    image = parsed_features['img_raw']
    image = tf.reshape(image, image_size)
    return image


if __name__ == '__main__':
    main()