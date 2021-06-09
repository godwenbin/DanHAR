import tensorflow as tf
import numpy as np
# from sklearn.metrics import confusion_matrix
from utils import shuffle
from utils import extract_batch_size
from utils import cross_val
from visual import plot_att


class AttCNN(object):
    def __init__(self, train_x, train_y, test_x, test_y,
                 seg_len=50, num_channels=3, num_labels=3,
                 num_conv_for_extract=3, filters=64, k_size=5, conv_strides=1, pool_size=2, pool_strides=2,
                 batch_size=200, learning_rate=0.0001, num_epochs=100,
                 print_val_each_epoch=10, print_test_each_epoch=50, print_test=True,
                 cpt_func='dot', norm_func='softmax', padding='same',
                 att_cnn_filters1=192, att_cnn_filters2=192, att_cnn_filters3=192,
                 cnn_type='1d', bool_bn=False, bool_visual_att=False, act_func='relu',
                 no_exp=1):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.seg_len = seg_len
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_batches = train_x.shape[0] // self.batch_size
        self.cnn_type = cnn_type
        if cnn_type == '2d':
            self.X = tf.placeholder(tf.float32, (None, self.seg_len, self.num_channels, 1))
            self.train_x = train_x.reshape([-1, self.seg_len, self.num_channels, 1])
            self.test_x = test_x.reshape([-1, self.seg_len, self.num_channels, 1])
        else:
            self.X = tf.placeholder(tf.float32, (None, self.seg_len, self.num_channels))
        self.Y = tf.placeholder(tf.float32, (None, self.num_labels))
        self.is_training = tf.placeholder(tf.bool)

        self.num_conv_for_extract = num_conv_for_extract
        self.filters = filters
        self.k_size = k_size
        self.conv_strides = conv_strides
        self.pool_size = pool_size
        self.pool_strides = pool_strides

        self.att_cnn_filters1 = att_cnn_filters1
        self.att_cnn_filters2 = att_cnn_filters2
        self.att_cnn_filters3 = att_cnn_filters3

        self.print_val_each_epoch = print_val_each_epoch
        self.print_test_each_epoch = print_test_each_epoch
        self.print_test = print_test

        self.cpt_func = cpt_func
        self.norm_func = norm_func
        self.padding = padding

        self.num_batches_test = test_x.shape[0] // self.batch_size
        self.bool_bn = bool_bn
        self.bool_visual_att = bool_visual_att

        # if act_func == 'relu':
        #     self.act_func = tf.nn.relu
        # else:
        #     self.act_func = None
        self.act_func = act_func
        self.no_exp = no_exp

    def _batch_norm(self, x):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=self.is_training,
                                            updates_collections=None)

    def build_network(self):
        for i in range(self.num_conv_for_extract):
            if i == 0:
                conv_extract = tf.layers.conv1d(
                    inputs=self.X,
                    filters=self.filters,
                    kernel_size=self.k_size,
                    strides=self.conv_strides,
                    padding=self.padding,
                    activation=self.act_func,
                    # kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    # bias_initializer=tf.constant_initializer(0.01)
                )
                print('# conv shape {}'.format(conv_extract.shape))
                if self.bool_bn:
                    conv_extract = self._batch_norm(conv_extract)
                if self.act_func:
                    conv_extract = tf.nn.relu(conv_extract)
            else:
                conv_extract = tf.layers.conv1d(
                    inputs=conv_extract,
                    filters=int(self.filters * (i + 1)),
                    kernel_size=self.k_size,
                    strides=self.conv_strides,
                    padding=self.padding,
                    activation=self.act_func,
                    # kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    # bias_initializer=tf.constant_initializer(0.01)
                )
                print('# conv shape {}'.format(conv_extract.shape))
                if self.bool_bn:
                    conv_extract = self._batch_norm(conv_extract)
                if self.act_func:
                    conv_extract = tf.nn.relu(conv_extract)
        conv1 = tf.layers.conv1d(
            inputs=conv_extract,
            filters=self.att_cnn_filters1,
            kernel_size=self.k_size,
            strides=self.conv_strides,
            padding=self.padding,
            # activation=self.act_func,
            # kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            # bias_initializer=tf.constant_initializer(0.01)
        )
        print('# conv1 shape {}'.format(conv1.shape))
        if self.bool_bn:
            conv1 = self._batch_norm(conv1)
        if self.act_func:
            conv1 = tf.nn.relu(conv1)
        pool1 = tf.layers.max_pooling1d(
            inputs=conv1,
            pool_size=self.pool_size,
            strides=self.pool_strides,
            padding='same'
        )
        print('# pool1 shape {}'.format(pool1.shape))
        conv2 = tf.layers.conv1d(
            inputs=pool1,
            filters=self.att_cnn_filters2,
            kernel_size=self.k_size,
            strides=self.conv_strides,
            padding=self.padding,
            # activation=self.act_func,
            # kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            # bias_initializer=tf.constant_initializer(0.01)
        )
        print('# conv2 shape {}'.format(conv2.shape))
        if self.bool_bn:
            conv2 = self._batch_norm(conv2)
        if self.act_func:
            conv2 = tf.nn.relu(conv2)
        pool2 = tf.layers.max_pooling1d(
            inputs=conv2,
            pool_size=self.pool_size,
            strides=self.pool_strides,
            padding='same'
        )
        print('# pool2 shape {}'.format(pool2.shape))
        conv3 = tf.layers.conv1d(
            inputs=pool2,
            filters=self.att_cnn_filters3,
            kernel_size=self.k_size,
            strides=self.conv_strides,
            padding=self.padding,
            # activation=self.act_func,
            # kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            # bias_initializer=tf.constant_initializer(0.01)
        )
        print('# conv3 shape {}'.format(conv3.shape))
        if self.bool_bn:
            conv3 = self._batch_norm(conv3)
        if self.act_func:
            conv3 = tf.nn.relu(conv3)
        pool3 = tf.layers.max_pooling1d(
            inputs=conv3,
            pool_size=self.pool_size,
            strides=self.pool_strides,
            padding='same'
        )
        print('# pool3 shape {}'.format(pool3.shape))
        l_op = pool3
        shape = l_op.get_shape().as_list()
        flat = tf.reshape(l_op, [-1, shape[1] * shape[2]])
        fc1 = tf.layers.dense(
            inputs=flat,
            units=self.att_cnn_filters3,
            activation=tf.nn.relu,
            # kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            # bias_initializer=tf.constant_initializer(0)
        )
        return conv1, conv2, conv3, fc1

    def build_network_2d(self):
        for i in range(self.num_conv_for_extract):
            if i == 0:
                conv_extract = tf.layers.conv2d(
                    inputs=self.X,
                    filters=self.filters,
                    kernel_size=(self.k_size, 1),
                    strides=(self.conv_strides, 1),
                    padding=self.padding,
                    activation=tf.nn.relu
                )
                print('# conv shape {}'.format(conv_extract.shape))
            else:
                conv_extract = tf.layers.conv2d(
                    inputs=conv_extract,
                    filters=int(self.filters * (i + 1)),
                    kernel_size=(self.k_size, 1),
                    strides=(self.conv_strides, 1),
                    padding=self.padding,
                    activation=tf.nn.relu
                )
                print('# conv shape {}'.format(conv_extract.shape))
        conv1 = tf.layers.conv2d(
            inputs=conv_extract,
            filters=self.att_cnn_filters1,
            kernel_size=(self.k_size, 1),
            strides=(self.conv_strides, 1),
            padding=self.padding,
            activation=tf.nn.relu
        )
        print('# conv1 shape {}'.format(conv1.shape))
        if self.bool_bn:
            conv1 = self._batch_norm(conv1)
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=(self.pool_size, 1),
            strides=(self.pool_strides, 1),
            padding='same'
        )
        print('# pool1 shape {}'.format(pool1.shape))
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=self.att_cnn_filters2,
            kernel_size=(self.k_size, 1),
            strides=(self.conv_strides, 1),
            padding=self.padding,
            activation=tf.nn.relu
        )
        print('# conv2 shape {}'.format(conv2.shape))
        if self.bool_bn:
            conv2 = self._batch_norm(conv2)
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=(self.pool_size, 1),
            strides=(self.pool_strides, 1),
            padding='same'
        )
        print('# pool2 shape {}'.format(pool2.shape))
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=self.att_cnn_filters3,
            kernel_size=(self.k_size, 1),
            strides=(self.conv_strides, 1),
            padding=self.padding,
            activation=tf.nn.relu
        )
        print('# conv3 shape {}'.format(conv3.shape))
        if self.bool_bn:
            conv3 = self._batch_norm(conv3)
        pool3 = tf.layers.max_pooling2d(
            inputs=conv3,
            pool_size=(self.pool_size, 1),
            strides=(self.pool_strides, 1),
            padding='same'
        )
        print('# pool shape {}'.format(pool3.shape))
        l_op = self._batch_norm(pool3)
        shape = l_op.get_shape().as_list()
        flat = tf.reshape(l_op, [-1, shape[1] * shape[2] * shape[3]])
        fc1 = tf.layers.dense(
            inputs=flat,
            units=self.att_cnn_filters3,
            activation=self.act_func,
        )
        shape_conv1 = conv1.get_shape().as_list()
        conv1 = tf.reshape(conv1, [-1, shape_conv1[1] * shape_conv1[2], shape_conv1[3]])
        shape_conv2 = conv2.get_shape().as_list()
        conv2 = tf.reshape(conv2, [-1, shape_conv2[1] * shape_conv2[2], shape_conv2[3]])
        shape_conv3 = conv3.get_shape().as_list()
        conv3 = tf.reshape(conv3, [-1, shape_conv3[1] * shape_conv3[2], shape_conv3[3]])
        return conv1, conv2, conv3, fc1

    def build_attention(self, local_feature, global_feature, name='Attention'):
        cpt_func = self.cpt_func
        norm_func = self.norm_func
        with tf.name_scope(name):
            n_units = local_feature.get_shape().as_list()[2]
            n_units_g = global_feature.get_shape().as_list()[1]
            if n_units_g != n_units:
                global_feature = tf.layers.dense(
                    inputs=global_feature,
                    units=n_units,
                    # activation=tf.nn.relu,
                    name='{}-Gbf_fc'.format(name),
                )
                print("# {} output shape {}".format(global_feature.name, global_feature.shape))
            with tf.name_scope('{}-Get_cpt_score'.format(name)):
                # local_feature_t = tf.transpose(local_feature, (0, 2, 1))
                print("name", name)
                if cpt_func == 'pc':
                    print("l_vector", local_feature.shape)
                    g_vector = tf.expand_dims(global_feature, axis=1)
                    print("g_vector", g_vector.shape)
                    add_ = tf.add(local_feature, g_vector)
                    print("add_", add_.shape)
                    add_t = tf.transpose(add_, (0, 2, 1))
                    print("add_t", add_t.shape)
                    # len_lcf = local_feature.get_shape().as_list()[1]
                    u_para = tf.Variable(tf.random_normal([1, n_units, 1], mean=0.01, stddev=0.01),
                                         name="{}-U_para".format(name))
                    print("u_para", u_para.shape)
                    dot_ = tf.multiply(add_t, u_para)
                    print("dot_", dot_.shape)
                    score_vector = tf.reduce_sum(dot_, 1)  # (batch,56*56)
                    print("score_vector", score_vector.shape)
                    # compatibility_function = tf.reshape(tf.tensordot(add_, u_para, axes=1),
                    #                                     [-1, len_lcf, 1], name='{}-Cpt_func_pc'.format(name))
                    if norm_func == 'softmax':
                        score = tf.nn.softmax(score_vector)  # (batch,56*56)
                    elif norm_func == 'tanh':
                        score = tf.nn.tanh(score_vector)
                    elif norm_func == 'sigmoid':
                        score = tf.nn.sigmoid(score_vector)
                    else:
                        score = score_vector
                    a_score = tf.expand_dims(score, axis=1)  # (batch,1,56*56)
                    print("a_score", a_score.shape)
                    l_vector_t = tf.transpose(local_feature, (0, 2, 1))
                    print("l_vector_t", l_vector_t.shape)
                    gas = tf.multiply(l_vector_t, a_score)  # (batch,256,56*56)
                    print("gas", gas.shape)
                    ga = tf.reduce_sum(gas, [2])
                    print("ga", ga.shape)
                    return score, ga, score_vector
                else:
                    # compatibility_function = tf.matmul(local_feature, g, name='{}-Cpt_func_dot'.format(name))
                    l_vector_t = tf.transpose(local_feature, (0, 2, 1))
                    print("l_vector_t", l_vector_t.shape)
                    g_vector = tf.expand_dims(global_feature, axis=2)
                    print("g_vector", g_vector.shape)
                    dot_ = tf.multiply(l_vector_t, g_vector)
                    print("dot_", dot_.shape)
                    score_vector = tf.reduce_sum(dot_, 1)  # (batch,56*56)
                    print("score_vector", score_vector.shape)
                    if norm_func == 'softmax':
                        score = tf.nn.softmax(score_vector)  # (batch,56*56)
                    elif norm_func == 'tanh':
                        score = tf.nn.tanh(score_vector)
                    else:
                        score = score_vector
                    a_score = tf.expand_dims(score, axis=1)  # (batch,1,56*56)
                    gas = tf.multiply(l_vector_t, a_score)  # (batch,256,56*56)
                    ga = tf.reduce_sum(gas, [2])
                    print(ga.shape)
                    return score, ga, score_vector

    def train(self):
        if self.cnn_type == '2d':
            conv1, conv2, conv3, fc1 = self.build_network_2d()
        else:
            conv1, conv2, conv3, fc1 = self.build_network()
        cpt1, ga1, cpf1 = self.build_attention(conv1, fc1, name='Att_1')
        cpt2, ga2, cpf2 = self.build_attention(conv2, fc1, name='Att_2')
        cpt3, ga3, cpf3 = self.build_attention(conv3, fc1, name='Att_3')
        ga = tf.concat((ga1, ga2, ga3), axis=1)
        # ga = ga3
        print('# ga shape {}'.format(ga.shape))
        y_ = tf.layers.dense(
            inputs=ga,
            units=self.num_labels,
            activation=tf.nn.softmax
        )
        loss = -tf.reduce_mean(self.Y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0)))
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        correct = tf.equal(tf.argmax(y_, 1), tf.argmax(self.Y, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        train_x, train_y = shuffle(self.train_x, self.train_y)
        train_xc, train_yc, val_xc, val_yc = cross_val(train_x, train_y, self.no_exp)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for epoch in range(self.num_epochs):
                train_xc, train_yc = shuffle(train_xc, train_yc)
                for i in range(train_xc.shape[0] // self.batch_size):
                    batch_x = extract_batch_size(train_xc, i, self.batch_size)
                    batch_y = extract_batch_size(train_yc, i, self.batch_size)
                    _, c = sess.run([train_op, loss], feed_dict={self.X: batch_x, self.Y: batch_y,
                                                                 self.is_training: True})
                if (epoch + 1) % self.print_val_each_epoch == 0:
                    val_acc = np.empty(0)
                    val_loss = np.empty(0)
                    for i in range(val_xc.shape[0] // self.batch_size):
                        batch_x_v = extract_batch_size(val_xc, i, self.batch_size)
                        batch_y_v = extract_batch_size(val_yc, i, self.batch_size)
                        val_acc = np.append(val_acc, sess.run(correct,
                                                              feed_dict={self.X: batch_x_v, self.Y: batch_y_v,
                                                                         self.is_training: False}))
                        val_loss = np.append(val_loss, sess.run(loss,
                                                                feed_dict={self.X: batch_x_v, self.Y: batch_y_v,
                                                                           self.is_training: False}))

                    # print(test_acc.shape)
                    _val_acc = np.average(val_acc)
                    _val_loss = np.average(val_loss)
                    print("### Epoch: ", epoch + 1,
                          "|Train loss = ", c,
                          "|Val loss = ", _val_loss,
                          "|Val acc = ", _val_acc, " ###")
                if self.print_test:
                    # if (epoch + 1) % self.print_test_each_epoch == 0:
                    #     print("### 1st After Epoch: ", epoch + 1,
                    #           " |Test acc = ", sess.run(accuracy,
                    #                                     feed_dict={self.X: self.test_x, self.Y: self.test_y,
                    #                                                self.is_training: False}), " ###")
                    if (epoch + 1) % self.print_test_each_epoch == 0:
                        test_acc = np.empty(0)
                        for i in range(self.num_batches_test):
                            batch_x_t = extract_batch_size(self.test_x, i, self.batch_size)
                            batch_y_t = extract_batch_size(self.test_y, i, self.batch_size)
                            test_acc = np.append(test_acc,
                                                 sess.run(correct,
                                                          feed_dict={self.X: batch_x_t, self.Y: batch_y_t,
                                                                     self.is_training: False}))
                        # print(test_acc.shape)
                        _test_acc = np.average(test_acc)
                        print("### After Epoch: ", epoch + 1,
                              " |Test acc = ", _test_acc, " ###")
                        if self.bool_visual_att:
                            n = np.random.randint(0, self.test_x.shape[0] - 100)
                            for i in range(6):
                                n += 8
                                label = np.argmax(self.test_y[n])
                                data = self.test_x[n]
                                data_ = self.test_x[n:n + 1]
                                cpt1_ = sess.run(cpt1, feed_dict={self.X: data_, self.is_training: False})
                                # cpf = sess.run(cpf3, feed_dict={self.X: data_, self.is_training: False})
                                cpt1_ = cpt1_.reshape([cpt1_.shape[1]])
                                # cpf_ = cpf.reshape([cpf.shape[1]])
                                cpt2_ = sess.run(cpt2, feed_dict={self.X: data_, self.is_training: False})
                                cpt2_ = cpt2_.reshape([cpt2_.shape[1]])
                                cpt3_ = sess.run(cpt3, feed_dict={self.X: data_, self.is_training: False})
                                cpt3_ = cpt3_.reshape([cpt3_.shape[1]])
                                plot_att(data,
                                         'Epoch {}-{}-Test no.{} label{}'.format(epoch + 1, i + 1, n, label),
                                         cpt1_, cpt2_, cpt3_, plot_type='bar')
