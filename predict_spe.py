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
from network import cnn_encoder, cnn_decoder, encoder, decoder
from model import Variational_Autoencoder
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #for windows

def main():
    # parser = argparse.ArgumentParser(description='py, test_data_txt, ground_truth_txt, outdir')
    #
    # parser.add_argument('--ground_truth_txt', '-i1', default='')
    #
    # parser.add_argument('--model', '-i2', default='./model_{}'.format(50000))
    #
    # parser.add_argument('--outdir', '-i3', default='')
    #
    # args = parser.parse_args()
    #
    # # check folder
    # if not (os.path.exists(args.outdir)):
    #     os.makedirs(args.outdir)

    # tf flag
    flags = tf.flags
    flags.DEFINE_string("ground_truth_txt","./input/shift/axis1/noise/test.txt","i1")
    flags.DEFINE_string("model", './output/shift/axis1/noise/z3/model/model_{}'.format(22000), "i2")
    flags.DEFINE_string('outdir', "./output/shift/axis1/noise/z3/spe/", 'i3')
    flags.DEFINE_float("beta", 1, "hyperparameter beta")
    flags.DEFINE_integer("num_of_generate", 100, "number of generate data")
    flags.DEFINE_integer("batch_size", 1, "batch size")
    flags.DEFINE_integer("latent_dim", 3, "latent dim")
    flags.DEFINE_list("image_size", [9 * 9 * 9], "image size")
    FLAGS = flags.FLAGS

    # check folder
    if not (os.path.exists(FLAGS.outdir)):
        os.makedirs(FLAGS.outdir)

    # read list
    test_data_list = io.load_list(FLAGS.ground_truth_txt)

    # test step
    test_step = FLAGS.num_of_generate // FLAGS.batch_size
    if FLAGS.num_of_generate % FLAGS.batch_size != 0:
        test_step += 1

    # load test data
    test_set = tf.data.TFRecordDataset(test_data_list)
    test_set = test_set.map(lambda x: _parse_function(x, image_size=FLAGS.image_size),
                            num_parallel_calls=os.cpu_count())
    test_set = test_set.batch(FLAGS.batch_size)
    test_iter = test_set.make_one_shot_iterator()
    test_data = test_iter.get_next()

    # load ground truth
    # ground_truth = io.load_matrix_data(FLAGS.ground_truth_txt, 'float32')
    # print(ground_truth.shape)

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
            'encoder': encoder,
            'decoder': decoder
        }
        VAE = Variational_Autoencoder(**kwargs)

        sess.run(init_op)

        # testing
        VAE.restore_model(FLAGS.model)
        tbar = tqdm(range(FLAGS.num_of_generate), ascii=True)
        specificity = []
        generate_data = []
        ori = []
        test_max = 667.0
        test_min = 0.0

        for k in tbar:
            test_data_batch = sess.run(test_data)
            ori_single = test_data_batch
            ori_single = ori_single[0, :]
            ori.append(ori_single)

        for i in tbar:
            sample_z = np.random.normal(0, 1.0, (1, FLAGS.latent_dim))
            generate_data_single = VAE.generate_sample(sample_z)
            generate_data_single = generate_data_single[0, :]
            generate_data.append(generate_data_single)

        generate_data = np.array(generate_data)
        ori = np.array(ori)
        generate_data = np.reshape(generate_data, [FLAGS.num_of_generate, 9, 9, 9])
        ori = np.reshape(ori, [FLAGS.num_of_generate, 9 * 9 * 9])
        # preds = preds.tolist()
        # ori = ori.tolist()
        n_generate_data = generate_data
        n_ori = ori
        generate_data = generate_data * (test_max - test_min) + test_min
        ori = ori * (test_max - test_min) + test_min

        for j in range(len(generate_data)):
            # EUDT
            generate_data = np.reshape(generate_data, [FLAGS.num_of_generate, 9, 9, 9])
            eudt_image = sitk.GetImageFromArray(generate_data[j])
            eudt_image.SetSpacing([0.885, 0.885, 1])
            eudt_image.SetOrigin([0, 0, 0])

            # label
            # label = np.where(generate_data > 0, 0, 1)
            # label_image = sitk.GetImageFromArray(label)
            # label_image.SetSpacing([1, 1])
            # label_image.SetOrigin([0, 0])

            # calculate ji
            # case_max_ji = 0.
            # for image_index in range(ground_truth.shape[0]):
            #     ji = utils.jaccard(label, ground_truth[image_index])
            #     if ji > case_max_ji:
            #         case_max_ji = ji
            # specificity.append([case_max_ji])


            print(j)

            generate_data = np.reshape(generate_data, [FLAGS.num_of_generate, 9 * 9 * 9])
            # generate_data = generate_data.tolist()
            # ori = ori.tolist()
            # calculate ji
            case_min_specificity = 0.
            for image_index in range(ori.shape[0]):
                specificity = np.mean(abs(ori[image_index] - generate_data))
                # specificity = sum(specificity)/len(specificity)
                # print(specificity)
                if specificity < case_min_specificity:
                    case_min_specificity = specificity
            # specificity.append([case_min_specificity])
            np.append(specificity,case_min_specificity)

            # output image
            io.write_mhd_and_raw(eudt_image, '{}.mhd'.format(os.path.join(FLAGS.outdir, 'EUDT', str(j+1))))
            # io.write_mhd_and_raw(label_image, '{}.mhd'.format(os.path.join(FLAGS.outdir, 'label', str(i + 1))))

        # specificity = np.min(abs(ori - generate_data), axis=0)
        print('specificity = %f' % np.mean(specificity))

    specificity = specificity.tolist()
    # print(specificity)
    # output csv file
    with open(os.path.join(FLAGS.outdir, 'specificity.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(specificity)
        writer.writerow(['specificity:', np.mean(specificity)])


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