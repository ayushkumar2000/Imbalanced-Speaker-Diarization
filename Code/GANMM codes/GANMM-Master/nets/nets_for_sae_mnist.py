import tensorflow as tf
import tflib.ops.linear
import tflib as lib

def Generator(n_samples, feature_dim=256, BATCH_SIZE=32, N_CLUSTER=4, noise=None, name='Generator.'):
    if noise is None:
        noise = tf.random_normal([n_samples, 64])

    output = lib.ops.linear.Linear(name + 'Generator.Input', 64, 32, noise)
    output = tf.nn.relu(output)

    output = lib.ops.linear.Linear(name + 'Generator.L1', 32, feature_dim, output)
    return tf.reshape(output, [-1, feature_dim])

def Discriminator(inputs, feature_dim=256, BATCH_SIZE=32, N_CLUSTER=4, name = 'Discriminator.'):

    output = lib.ops.linear.Linear(name + 'Discriminator.Input', feature_dim, 32, inputs)
    output = tf.nn.relu(output)

    output = lib.ops.linear.Linear(name + 'Discriminator.L2', 32,1, output)

    return tf.reshape(output, [-1])

def MNN(inputs, feature_dim=256, BATCH_SIZE=32, N_CLUSTER=4, name="MNN"):
    print('MNN')

    output = lib.ops.linear.Linear(name+'.MNN.Input', feature_dim, 32, inputs)
    output = tf.nn.relu(output)

    output = lib.ops.linear.Linear(name+'.MNN.L2',32,N_CLUSTER, output)
    output = tf.nn.softmax(output)
    return tf.reshape(output, [-1, N_CLUSTER])