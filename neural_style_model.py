import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.pyplot as imshow
from PIL import Image

import numpy as np
import tensorflow as tf

class CONFIG:
    IMAGE_WIDTH=400
    IMAGE_HEIGHT=300
    COLOR_CHANNELS=3
    NOISE_RATIO=0.6
    MEANS=np.array([123.68,116.79,113.939]).reshape((1,1,1,3))
    VGG_MODEL="imagenet-vgg-verydeep-19.mat"
    
def load_vgg_model(path):
    vgg=scipy.io.loadmat(path)
    vgg_layers=vgg['layers']
    def _weights(layer,expected_layer_name):
        wb=vgg_layers[0][layer][0][0][2]
        W=wb[0][0]
        b=wb[0][1]
        layer_name=vgg_layers[0][layer][0][0][0][0]
        assert layer_name==expected_layer_name
        return W,b

    def _relu(conv2d_layer):
        return tf.nn.relu(conv2d_layer)

    def _conv2d(prev_layer,layer,layer_name):
        W,b=_weights(layer,layer_name)
        W=tf.constant(W)
        b=tf.constant(np.reshape(b,(b.size)))
        return tf.nn.conv2d(prev_layer,filter=W,strides=[1,1,1,1],padding='SAME') + b
    
    def _conv2d_relu(prev_layer,layer,layer_name):
        return _relu(_conv2d(prev_layer,layer,layer_name))

    def _avgpool(prev_layer):
        return tf.nn.avg_pool(prev_layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    graph={}
    graph['input']=tf.Variable(np.zeros((1,CONFIG.IMAGE_HEIGHT,CONFIG.IMAGE_WIDTH,CONFIG.COLOR_CHANNELS)),dtype='float32')
    graph['conv1_1']=_conv2d_relu(graph['input'],0,'conv1_1')
    graph['conv1_2']=_conv2d_relu(graph['conv1_1'],2,'conv1_2')
    graph['avgpool1']=_avgpool(graph['conv1_2'])
    graph['conv2_1']=_conv2d_relu(graph['avgpool1'],5,'conv2_1')
    graph['conv2_2']=_conv2d_relu(graph['conv2_1'],7,'conv2_2')
    graph['avgpool2']=_avgpool(graph['conv2_2'])
    graph['conv3_1']=_conv2d_relu(graph['avgpool2'],10,'conv3_1')
    graph['conv3_2']=_conv2d_relu(graph['conv3_1'],12,'conv3_2')
    graph['conv3_3']=_conv2d_relu(graph['conv3_2'],14,'conv3_3')
    graph['conv3_4']=_conv2d_relu(graph['conv3_3'],16,'conv3_4')
    graph['avgpool3']=_avgpool(graph['conv3_4'])
    graph['conv4_1']=_conv2d_relu(graph['avgpool3'],19,'conv4_1')
    graph['conv4_2']=_conv2d_relu(graph['conv4_1'],21,'conv4_2')
    graph['conv4_3']=_conv2d_relu(graph['conv4_2'],23,'conv4_3')
    graph['conv4_4']=_conv2d_relu(graph['conv4_3'],25,'conv4_4')
    graph['avgpool4']=_avgpool(graph['conv4_4'])
    graph['conv5_1']=_conv2d_relu(graph['avgpool4'],28,'conv5_1')
    graph['conv5_2']=_conv2d_relu(graph['conv5_1'],30,'conv5_2')
    graph['conv5_3']=_conv2d_relu(graph['conv5_2'],32,'conv5_3')
    graph['conv5_4']=_conv2d_relu(graph['conv5_3'],34,'conv5_4')
    graph['avgpool5']=_avgpool(graph['conv5_4'])
    return graph

def reshape_and_normalize_image(image):
    image=np.reshape(image,(1,)+image.shape)
    image=image-CONFIG.MEANS
    return image

def save_image(path,image):
    image=image+CONFIG.MEANS
    image=np.clip(image[0],0,255).astype('uint8')
    scipy.misc.imsave(path,image)

def generate_noise_image(content_image,noise_ratio=CONFIG.NOISE_RATIO):
    noise_image=np.random.uniform(-20,20,(1,CONFIG.IMAGE_HEIGHT,CONFIG.IMAGE_WIDTH,CONFIG.COLOR_CHANNELS)).astype('float32')
    image=noise_image*noise_ratio+content_image*(1-noise_ratio)
    return image

def compute_content_cost(a_C,a_G):
    m,n_H,n_W,n_C=a_G.get_shape().as_list()
    a_C=tf.reshape(a_C,(-1,))
    a_G=tf.reshape(a_G,(-1,))
    J_content=tf.multiply(tf.reduce_sum(tf.squared_difference(a_C,a_G)),1/(4*n_H*n_W*n_C))
    return J_content

def gram_matrix(A):
    GA=tf.matmul(A,tf.transpose(A))
    return GA

def compute_layer_style_cost(a_S,a_G):
    m,n_H,n_W,n_C=a_G.get_shape().as_list()
    a_S=tf.transpose(tf.reshape(a_S,(n_H*n_W,n_C)))
    a_G=tf.transpose(tf.reshape(a_G,(n_H*n_W,n_C)))
    GS=gram_matrix(a_S)
    GG=gram_matrix(a_G)
    J_layer_style_cost=tf.multiply(tf.reduce_sum(tf.squared_difference(GS,GG)),1/(4*n_W**2*n_H**2*n_C**2))
    return J_layer_style_cost

def compute_style_cost(model,STYLE_LAYERS):
    J_style=0
    for layer,weight in STYLE_LAYERS:
        out=model[layer]
        a_S=sess.run(out)
        a_G=out
        J_layer_style=compute_layer_style_cost(a_S,a_G)
        J_style+=weight*J_layer_style
    return J_style
    
def total_cost(J_content,J_style,alpha=10,beta=40):
    J=alpha*J_content+beta*J_style
    return J

def total_cost2(J_content,J_style,J_style2,alpha=10,beta=40,gamma=30):
    J=alpha*J_content+beta*J_style+gamma*J_style2
    return J

tf.reset_default_graph()
sess=tf.InteractiveSession()
content_image = scipy.misc.imread('content.jpg')
content_image = reshape_and_normalize_image(content_image)
style_image = scipy.misc.imread('style.jpg')
style_image = reshape_and_normalize_image(style_image)
style_image2 = scipy.misc.imread('style2.jpg')
style_image2 = reshape_and_normalize_image(style_image2)
generated_image = generate_noise_image(content_image)
model = load_vgg_model("imagenet-vgg-verydeep-19.mat")
sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
a_G = out
J_content = compute_content_cost(a_C, a_G)
STYLE_LAYERS = [('conv1_1', 0.2),('conv2_1', 0.2),('conv3_1', 0.2),('conv4_1', 0.2),('conv5_1', 0.2)]
sess.run(model['input'].assign(style_image))
J_style = compute_style_cost(model, STYLE_LAYERS)
sess.run(model['input'].assign(style_image2))
J_style2 = compute_style_cost(model, STYLE_LAYERS)
J = total_cost2(J_content,J_style,J_style2)
optimizer = tf.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(J)

def model_nn(sess, input_image, num_iterations = 1000):
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))    
    for i in range(num_iterations):
        sess.run(train_step)
        generated_image = sess.run(model['input'])
        if i%100 == 0:
            Jt, Jc, Js, Js2 = sess.run([J, J_content, J_style, J_style2])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            print("style cost2 = " + str(Js2))
            save_image("output/" + str(i) + ".png", generated_image)
    save_image('output/generated_image.jpg', generated_image)
    return generated_image

model_nn(sess, generated_image)
