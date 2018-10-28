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
from network import cnn_encoder, cnn_decoder, mnist_decoder, mnist_encoder
from model import Variational_Autoencoder
import utils
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #for windows

def main():
    parser = argparse.ArgumentParser(description='py, test_data_txt, model, outdir')

    parser.add_argument('--test_data_txt', '-i1', default='')

    parser.add_argument('--model', '-i2', default='./model_{}'.format(50000))

    parser.add_argument('--outdir', '-i3', default='')

    args = parser.parse_args()

    # check folder
    if not (os.path.exists(args.outdir)):
        os.makedirs(args.outdir)

    # tf flag
    flags = tf.flags
    flags.DEFINE_float("beta", 0.1, "hyperparameter beta")
    flags.DEFINE_integer("num_of_test", 100, "number of test data")
    flags.DEFINE_integer("batch_size", 1, "batch size")
    flags.DEFINE_integer("latent_dim", 2, "latent dim")
    flags.DEFINE_list("image_size", [512, 512, 1], "image size")
    FLAGS = flags.FLAGS

    # read list
    test_data_list = io.load_list(args.test_data_txt)

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
            'outdir': args.outdir,
            'beta': FLAGS.beta,
            'latent_dim': FLAGS.latent_dim,
            'batch_size': FLAGS.batch_size,
            'image_size': FLAGS.image_size,
            'encoder': cnn_encoder,
            'decoder': cnn_decoder
        }
        VAE = Variational_Autoencoder(**kwargs)

        sess.run(init_op)

        # testing
        VAE.restore_model(args.model)
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
        plt.figure(figsize=(8, 6))
        fig = plt.scatter(latent_space[:, 0], latent_space[:, 1])
        plt.title('latent distribution')
        plt.xlabel('dim_1')
        plt.ylabel('dim_2')
        plt.show()



# # load tfrecord function
def _parse_function(record, image_size=[512, 512, 1]):
    keys_to_features = {
        'img_raw': tf.FixedLenFeature(np.prod(image_size), tf.float32),
    }
    parsed_features = tf.parse_single_example(record, keys_to_features)
    image = parsed_features['img_raw']
    image = tf.reshape(image, image_size)
    return image


if __name__ == '__main__':
    main()