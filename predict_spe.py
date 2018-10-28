'''
# Generate image and evalute the performance by Specificity
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
from network import cnn_encoder, cnn_decoder
from model import Variational_Autoencoder
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #for windows

def main():
    parser = argparse.ArgumentParser(description='py, test_data_txt, ground_truth_txt, outdir')

    parser.add_argument('--ground_truth_txt', '-i1', default='')

    parser.add_argument('--model', '-i2', default='./model_{}'.format(50000))

    parser.add_argument('--outdir', '-i3', default='')

    args = parser.parse_args()

    # check folder
    if not (os.path.exists(args.outdir)):
        os.makedirs(args.outdir)

    # tf flag
    flags = tf.flags
    flags.DEFINE_float("beta", 0.1, "hyperparameter beta")
    flags.DEFINE_integer("num_of_generate", 1000, "number of generate data")
    flags.DEFINE_integer("batch_size", 1, "batch size")
    flags.DEFINE_integer("latent_dim", 2, "latent dim")
    flags.DEFINE_list("image_size", [512, 512, 1], "image size")
    FLAGS = flags.FLAGS

    # load ground truth
    ground_truth = io.load_matrix_data(args.ground_truth_txt, 'int32')
    print(ground_truth.shape)

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
        tbar = tqdm(range(FLAGS.num_of_generate), ascii=True)
        specificity = []
        for i in tbar:
            sample_z = np.random.normal(0, 1.0, (1, FLAGS.latent_dim))
            generate_data = VAE.generate_sample(sample_z)
            generate_data = generate_data[0, :, :, 0]

            # EUDT
            eudt_image = sitk.GetImageFromArray(generate_data)
            eudt_image.SetSpacing([1, 1])
            eudt_image.SetOrigin([0, 0])

            # label
            label = np.where(generate_data > 0, 0, 1)
            label_image = sitk.GetImageFromArray(label)
            label_image.SetSpacing([1, 1])
            label_image.SetOrigin([0, 0])

            # calculate ji
            case_max_ji = 0.
            for image_index in range(ground_truth.shape[0]):
                ji = utils.jaccard(label, ground_truth[image_index])
                if ji > case_max_ji:
                    case_max_ji = ji
            specificity.append([case_max_ji])

            # output image
            io.write_mhd_and_raw(eudt_image, '{}.mhd'.format(os.path.join(args.outdir, 'EUDT', str(i+1))))
            io.write_mhd_and_raw(label_image, '{}.mhd'.format(os.path.join(args.outdir, 'label', str(i + 1))))

    print('specificity = %f' % np.mean(specificity))

    # output csv file
    with open(os.path.join(args.outdir, 'specificity.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(specificity)
        writer.writerow(['specificity:', np.mean(specificity)])



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