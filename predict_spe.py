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
from network import cnn_encoder, cnn_decoder, encoder, decoder, deepest_encoder, deepest_decoder, encoder_r, decoder_r, encoder2, decoder2,\
    shallow_encoder, shallow_decoder, encoder5, decoder5,deep_encoder, deep_decoder, deeper_encoder, deeper_decoder, encoder1, decoder1
from model import Variational_Autoencoder
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #for windows

def main():

    # tf flag
    flags = tf.flags
    flags.DEFINE_string("train_data_txt", "./input/CT/patch/train.txt", "train data txt")
    # flags.DEFINE_string("train_data_txt", "./input/axis2/noise/train.txt", "train data txt")
    flags.DEFINE_string("ground_truth_txt", "./input/CT/patch/test.txt","i1")
    # flags.DEFINE_string("ground_truth_txt", "./input/axis2/noise/test.txt","i1")
    flags.DEFINE_string("model", './output/CT/patch/model2/z24/alpha_1e-5/beta_1/fine/model/model_{}'.format(617000), "i2")
    flags.DEFINE_string('outdir', "./output/CT/patch/model2/z24/alpha_1e-5/beta_1/fine/spe/", 'i3')
    # flags.DEFINE_string("model", './output/axis2/noise/model2/z24/alpha_1e-5/beta_0.1/model/model_{}'.format(9489000), "i2")
    # flags.DEFINE_string("outdir", "./output/axis2/noise/model2/z24/alpha_1e-5/beta_0.1/spe/", "i3")
    flags.DEFINE_float("beta", 1, "hyperparameter beta")
    flags.DEFINE_integer("num_of_generate", 5000, "number of generate data")
    flags.DEFINE_integer("num_of_test", 607, "number of test data")
    # flags.DEFINE_integer("num_of_test", 3000, "number of test data")
    flags.DEFINE_integer("num_of_train", 1825, "number of train data")
    # flags.DEFINE_integer("num_of_train", 10000, "number of train data")
    flags.DEFINE_integer("batch_size", 1, "batch size")
    flags.DEFINE_integer("latent_dim", 24, "latent dim")
    flags.DEFINE_list("image_size", [9 * 9 * 9], "image size")
    FLAGS = flags.FLAGS

    # check folder
    if not (os.path.exists(FLAGS.outdir)):
        os.makedirs(FLAGS.outdir)

    # read list
    test_data_list = io.load_list(FLAGS.ground_truth_txt)
    train_data_list = io.load_list(FLAGS.train_data_txt)

    # test step
    test_step = FLAGS.num_of_generate // FLAGS.batch_size
    if FLAGS.num_of_generate % FLAGS.batch_size != 0:
        test_step += 1

    # load train data
    train_set = tf.data.TFRecordDataset(train_data_list)
    train_set = train_set.map(lambda x: _parse_function(x, image_size=FLAGS.image_size),
                            num_parallel_calls=os.cpu_count())
    train_set = train_set.batch(FLAGS.batch_size)
    train_iter = train_set.make_one_shot_iterator()
    train_data = train_iter.get_next()

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
            'encoder': encoder2,
            'decoder': decoder2
        }
        VAE = Variational_Autoencoder(**kwargs)

        sess.run(init_op)

        # testing
        VAE.restore_model(FLAGS.model)
        tbar = tqdm(range(FLAGS.num_of_generate), ascii=True)
        specificity = []
        generate_data = []
        ori = []
        latent_space = []
        # test_max = 667.0
        # test_min = 0.0
        patch_side = 9

        for i in range(FLAGS.num_of_train):
            train_data_batch = sess.run(train_data)
            z = VAE.plot_latent(train_data_batch)
            z = z.flatten()
            latent_space.append(z)

        mu = np.mean(latent_space, axis=0)
        var = np.var(latent_space, axis=0)

        for i in range(FLAGS.num_of_test):
            test_data_batch = sess.run(test_data)
            ori_single = test_data_batch
            ori_single = ori_single[0, :]
            ori.append(ori_single)

        for j in tbar:
            sample_z = np.random.normal(mu, var, (1, FLAGS.latent_dim))
            generate_data_single = VAE.generate_sample(sample_z)
            generate_data_single = generate_data_single[0, :]
            generate_data.append(generate_data_single)
            gen = np.reshape(generate_data_single, [patch_side, patch_side, patch_side])

        # generate_data = np.array(generate_data)
        # ori = np.array(ori)
        # generate_data = np.reshape(generate_data, [FLAGS.num_of_generate, patch_side, patch_side, patch_side])
        # ori = np.reshape(ori, [FLAGS.num_of_test, patch_side * patch_side * patch_side])
        # preds = preds.tolist()
        # ori = ori.tolist()
        # n_generate_data = generate_data
        # n_ori = ori
        # generate_data = generate_data * (test_max - test_min) + test_min
        # ori = ori * (test_max - test_min) + test_min

        # for j in range(len(generate_data)):
        # for j in tbar:

            # EUDT
            # generate_data = np.reshape(generate_data, [FLAGS.num_of_generate, patch_side, patch_side, patch_side])
            # eudt_image = sitk.GetImageFromArray(generate_data[j])
            eudt_image = sitk.GetImageFromArray(gen)
            eudt_image.SetSpacing([0.885, 0.885, 1])
            eudt_image.SetOrigin([0, 0, 0])


            # generate_data = np.reshape(generate_data, [FLAGS.num_of_generate, patch_side * patch_side * patch_side])
            # generate_data = generate_data.tolist()
            # ori = ori.tolist()
            # calculate ji
            case_min_specificity = 1.0
            for image_index in range(FLAGS.num_of_test):
                specificity_tmp = utils.L1norm(ori[image_index] ,generate_data_single)
                # print('o=',ori[image_index])
                # print('g=',generate_data[image_index])
                # specificity = sum(specificity)/len(specificity)
                # print('tmp=', specificity_tmp)
                # print(specificity)
                if specificity_tmp < case_min_specificity:
                    case_min_specificity = specificity_tmp
                # print('case_min=',case_min_specificity)
            specificity.append([case_min_specificity])
            # print('specificity=', specificity)
            # np.append(specificity, case_min_specificity)

            # output image
            io.write_mhd_and_raw(eudt_image, '{}.mhd'.format(os.path.join(FLAGS.outdir, 'EUDT', str(j+1))))
            # io.write_mhd_and_raw(label_image, '{}.mhd'.format(os.path.join(FLAGS.outdir, 'label', str(i + 1))))

        # specificity = np.min(abs(ori - generate_data), axis=0)
    print('specificity = %f' % np.mean(specificity))
    np.savetxt(os.path.join(FLAGS.outdir, 'specificity.csv'), specificity, delimiter=",")


    # specificity = specificity.tolist()
    # print(specificity)
    # output csv file
    # with open(os.path.join(FLAGS.outdir, 'specificity.csv'), 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(specificity)
    #     writer.writerow(['specificity:', specificity])


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