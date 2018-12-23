'''
# Reconstruct image and evalute the performance by Generalization
# Author: Zhihui Lu
# Date: 2018/10/17
'''

import tensorflow as tf
import os
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import csv
import dataIO as io
from network import cnn_encoder, cnn_decoder, mnist_decoder, mnist_encoder, encoder, decoder, encoder_r, decoder_r, \
    deepest_encoder, deepest_decoder, shallow_encoder, shallow_decoder, encoder5, decoder5,deeper_encoder, deeper_decoder, deep_encoder, deep_decoder
from model import Variational_Autoencoder
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #for windows

def main():

    # tf flag
    flags = tf.flags
    flags.DEFINE_string("test_data_txt", "./input/CT/patch/test.txt", "i1")
    flags.DEFINE_string("model", './output/CT/patch/deep/z20/model/model_{}'.format(4000), "i2")
    flags.DEFINE_string("outdir", "./output/CT/patch/deep/z20/gen/", "i3")
    flags.DEFINE_float("beta", 1, "hyperparameter beta")
    flags.DEFINE_integer("num_of_test", 607, "number of test data")
    flags.DEFINE_integer("batch_size", 1, "batch size")
    flags.DEFINE_integer("latent_dim", 20, "latent dim")
    flags.DEFINE_list("image_size", [9*9*9], "image size")
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

        # testing
        VAE.restore_model(FLAGS.model)
        tbar = tqdm(range(test_step), ascii=True)
        preds = []
        ori = []
        test_max = 667.0
        test_min = 0.0

        for k in tbar:
            test_data_batch = sess.run(test_data)
            ori_single = test_data_batch
            preds_single = VAE.reconstruction_image(ori_single)
            # print(preds_single)
            preds_single = preds_single[0, :]
            ori_single = ori_single[0, :]

            preds.append(preds_single)
            ori.append(ori_single)

        patch_side = 9
        # preds = np.array(preds)
        # ori = np.array(ori)
        preds = np.reshape(preds, [FLAGS.num_of_test, patch_side , patch_side , patch_side])
        ori = np.reshape(ori, [FLAGS.num_of_test, patch_side, patch_side, patch_side])
        # preds = preds.tolist()
        # ori = ori.tolist()
        n_preds = preds
        n_ori = ori

        preds = preds * (test_max - test_min) + test_min
        ori = ori * (test_max - test_min) + test_min

        # label
        generalization_single = []
        # generalization_single = np.zeros([FLAGS.num_of_test, patch_side * patch_side * patch_side])
        for j in range(len(preds)):

            # EUDT
            eudt_image = sitk.GetImageFromArray(preds[j])
            # eudt_image.SetSpacing([1, 1])
            eudt_image.SetOrigin([0, 0, 0])
            # eudt_image.SetSize(0, patch_side)
            # eudt_image.SetSize(1, patch_side)
            # eudt_image.SetSize(2, patch_side)
            eudt_image.SetSpacing([0.885,0.885,1])


            # output image

            io.write_mhd_and_raw(eudt_image, '{}.mhd'.format(os.path.join(FLAGS.outdir, 'EUDT', 'recon_{}'.format(j))))
            # io.write_mhd_and_raw(preds[j], '{}.raw'.format(os.path.join(FLAGS.outdir, 'recon_{}'.format(j))))
            # io.write_mhd_and_raw(label_image, '{}.mhd'.format(os.path.join(FLAGS.outdir, 'label', 'recon_{}'.format(j))))

            # print(preds.shape)
            # print(ori.shape)
            # generalization_single.append(utils.L1norm(ori[j], preds[j]))

            # np.append(generalization_single, ori - preds)

        # print(generalization_single.shape)
        generalization = np.average(np.average(abs(ori - preds), axis=0))
        print('generalization = %f' % generalization)

        n_generalization = np.average(np.average(abs(n_ori - n_preds), axis=0))
        print('n_generalization = %f' % n_generalization)


    # output csv file
    with open(os.path.join(FLAGS.outdir, 'generalization.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        # writer.writerows(generalization_single)
        writer.writerow(['generalization= ', generalization])

    # plot reconstruction
    fig, axes = plt.subplots(ncols=10, nrows=2, figsize=(18, 4))
    X = ori[:,4,:]
    Xe = preds[:,4,:]
    print(Xe.shape)

    for i in range(10):
        axes[0, i].imshow(X[i, :].reshape(patch_side, patch_side),cmap=cm.Greys_r)
        axes[0, i].set_title('original %d' % i)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)

        axes[1, i].imshow(Xe[i, :].reshape(patch_side, patch_side),cmap=cm.Greys_r)
        axes[1, i].set_title('reconstruction %d' % i)
        axes[1, i].get_xaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)
    plt.savefig(FLAGS.outdir + "reconstruction.png")
    # plt.show()

# # load tfrecord function
def _parse_function(record, image_size=[5 * 5 * 5]):
    keys_to_features = {
        'img_raw': tf.FixedLenFeature(np.prod(image_size), tf.float32),
    }
    parsed_features = tf.parse_single_example(record, keys_to_features)
    image = parsed_features['img_raw']
    image = tf.reshape(image, image_size)
    return image


if __name__ == '__main__':
    main()