'''
# Variational AutoEncoder Trainer
# Author: Zhihui Lu
# Date: 2018/10/17
'''
import tensorflow as tf
import os, random
import argparse
import numpy as np
from tqdm import tqdm
import dataIO as io
from network import cnn_encoder, cnn_decoder, mnist_encoder, mnist_decoder
from model import Variational_Autoencoder
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'          #for windows
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main():
    parser = argparse.ArgumentParser(description='py, train_data_txt, val_data_txt, outdir')

    parser.add_argument('--train_data_txt', '-i1', default='', help='train data txt')

    parser.add_argument('--val_data_txt', '-i2', default='', help='validation data txt')

    parser.add_argument('--outdir', '-i3', default='./beta_0.1', help='outdir')

    args = parser.parse_args()

    # check folder
    if not (os.path.exists(os.path.join(args.outdir, 'tensorboard', 'train'))):
        os.makedirs(os.path.join(args.outdir, 'tensorboard', 'train'))
    if not (os.path.exists(os.path.join(args.outdir, 'tensorboard', 'val'))):
        os.makedirs(os.path.join(args.outdir, 'tensorboard', 'val'))
    if not (os.path.exists(os.path.join(args.outdir, 'tensorboard', 'rec'))):
        os.makedirs(os.path.join(args.outdir, 'tensorboard', 'rec'))
    if not (os.path.exists(os.path.join(args.outdir, 'tensorboard', 'kl'))):
        os.makedirs(os.path.join(args.outdir, 'tensorboard', 'kl'))
    if not (os.path.exists(os.path.join(args.outdir, 'model'))):
        os.makedirs(os.path.join(args.outdir, 'model'))

    # tf flag
    flags = tf.flags
    flags.DEFINE_float("beta", 0.1, "hyperparameter beta")
    flags.DEFINE_integer("num_of_val", 1000, "number of validation data")
    flags.DEFINE_integer("batch_size", 30, "batch size")
    flags.DEFINE_integer("num_iteration", 50001, "number of iteration")
    flags.DEFINE_integer("save_loss_step", 50, "step of save loss")
    flags.DEFINE_integer("save_model_step", 500, "step of save model and validation")
    flags.DEFINE_integer("shuffle_buffer_size", 10000, "buffer size of shuffle")
    flags.DEFINE_integer("latent_dim", 2, "latent dim")
    flags.DEFINE_list("image_size", [512, 512, 1], "image size")
    FLAGS = flags.FLAGS

    # read list
    train_data_list = io.load_list(args.train_data_txt)
    val_data_list = io.load_list(args.val_data_txt)

    # shuffle list
    random.shuffle(train_data_list)
    # val step
    val_step = FLAGS.num_of_val // FLAGS.batch_size
    if FLAGS.num_of_val % FLAGS.batch_size != 0:
        val_step += 1

    # load train data and validation data
    train_set = tf.data.TFRecordDataset(train_data_list)
    train_set = train_set.map(lambda x: _parse_function(x, image_size=FLAGS.image_size),
                              num_parallel_calls=os.cpu_count())
    train_set = train_set.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
    train_set = train_set.repeat()
    train_set = train_set.batch(FLAGS.batch_size)
    train_iter = train_set.make_one_shot_iterator()
    train_data = train_iter.get_next()

    val_set = tf.data.TFRecordDataset(val_data_list)
    val_set = val_set.map(lambda x: _parse_function(x, image_size=FLAGS.image_size),
                          num_parallel_calls=os.cpu_count())
    val_set = val_set.repeat()
    val_set = val_set.batch(FLAGS.batch_size)
    val_iter = val_set.make_one_shot_iterator()
    val_data = val_iter.get_next()

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

        # print parmeters
        utils.cal_parameter()

        # prepare tensorboard
        writer_train = tf.summary.FileWriter(os.path.join(args.outdir, 'tensorboard', 'train'), sess.graph)
        writer_val = tf.summary.FileWriter(os.path.join(args.outdir, 'tensorboard', 'val'))
        writer_rec = tf.summary.FileWriter(os.path.join(args.outdir, 'tensorboard', 'rec'))
        writer_kl = tf.summary.FileWriter(os.path.join(args.outdir, 'tensorboard', 'kl'))

        value_loss = tf.Variable(0.0)
        tf.summary.scalar("loss", value_loss)
        merge_op = tf.summary.merge_all()

        # initialize
        sess.run(init_op)

        # training
        tbar = tqdm(range(FLAGS.num_iteration), ascii=True)
        for i in tbar:
            train_data_batch = sess.run(train_data)
            train_loss, rec_loss, kl_loss = VAE.update(train_data_batch)

            if i % FLAGS.save_loss_step is 0:
                s = "Loss: {:.4f}, rec_loss: {:.4f}, kl_loss: {:.4f}".format(train_loss, rec_loss, kl_loss)
                tbar.set_description(s)
                summary_train_loss = sess.run(merge_op, {value_loss: train_loss})
                writer_train.add_summary(summary_train_loss, i)

                summary_rec_loss = sess.run(merge_op, {value_loss: rec_loss})
                summary_kl_loss = sess.run(merge_op, {value_loss: kl_loss})
                writer_rec.add_summary(summary_rec_loss, i)
                writer_kl.add_summary(summary_kl_loss, i)


            if i % FLAGS.save_model_step is 0:
                # save model
                VAE.save_model(i)

                # validation
                val_loss = 0.
                for j in range(val_step):
                    val_data_batch = sess.run(val_data)
                    val_loss += VAE.validation(val_data_batch)
                val_loss /= val_step

                summary_val = sess.run(merge_op, {value_loss: val_loss})
                writer_val.add_summary(summary_val, i)

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