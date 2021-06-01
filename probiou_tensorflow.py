import tensorflow as tf
import numpy as np


def probiou_loss(boxes_pred, target_boxes_, EPS = 1e-3, mode='l2'):

    """
        boxes_pred    -> a matrix [N,5](x,y,w,h,angle - in radians) containing ours predicted box ;in case of HBB angle == 0
        target_boxes_  -> a matrix [N,5](x,y,w,h,angle - in radians) containing ours target    box ;in case of HBB angle == 0
        EPS     -> threshold to avoid infinite values
        mode       -> ('l1' in [0,1] or 'l2' in [0,inf]) metrics according our paper

    """

    x1, y1, w1, h1, theta1 = tf.unstack(boxes_pred, axis=1)
    x2, y2, w2, h2, theta2 = tf.unstack(target_boxes_, axis=1)
    x1 = tf.reshape(x1, [-1, 1])
    y1 = tf.reshape(y1, [-1, 1])
    h1 = tf.reshape(h1, [-1, 1])
    w1 = tf.reshape(w1, [-1, 1])
    theta1 = tf.reshape(theta1, [-1, 1])
    x2 = tf.reshape(x2, [-1, 1])
    y2 = tf.reshape(y2, [-1, 1])
    h2 = tf.reshape(h2, [-1, 1])
    w2 = tf.reshape(w2, [-1, 1])
    theta2 = tf.reshape(theta2, [-1, 1])

    # gbb form
    aa = w1**2/12; bb = h1**2/12; angles = theta1
    # rotated form
    a1 = aa*tf.math.pow(tf.math.cos(angles), 2.) + bb*tf.math.pow(tf.math.sin(angles), 2.)
    b1 = aa*tf.math.pow(tf.math.sin(angles), 2.) + bb*tf.math.pow(tf.math.cos(angles), 2.)
    c1 = 0.5*(aa - bb)*tf.math.sin(2.*angles)

    # gbb form
    aa = w2**2/12; bb = h2**2/12; angles = theta2
    # rotated form
    a2 = aa*tf.math.pow(tf.math.cos(angles), 2.) + bb*tf.math.pow(tf.math.sin(angles), 2.)
    b2 = aa*tf.math.pow(tf.math.sin(angles), 2.) + bb*tf.math.pow(tf.math.cos(angles), 2.)
    c2 = 0.5*(aa - bb)*tf.math.sin(2.*angles)

    B1 = 1/4.*( (a1+a2)*(y1-y2)**2. + (b1+b2)*(x1-x2)**2. ) + 1/2.*( (c1+c2)*(x2-x1)*(y1-y2) )
    B1 = B1 / ( (a1+a2)*(b1+b2) - (c1+c2)**2. + EPS )


    sqrt = (a1*b1-c1**2)*(a2*b2-c2**2)
    sqrt = tf.clip_by_value(sqrt, EPS, tf.reduce_max(sqrt)+EPS)
    B2 = ( (a1+a2)*(b1+b2) - (c1+c2)**2. )/( 4.*tf.math.sqrt(sqrt) + EPS )
    B2 = tf.clip_by_value(B2, EPS, tf.reduce_max(B2)+EPS)
    B2 = 1/2.*tf.math.log(B2)

    Bd = B1 + B2
    Bd = tf.clip_by_value(Bd, EPS, 100.)

    l1 = tf.math.sqrt(1 - tf.math.exp(-Bd) + EPS)

    if mode=='l2':
        l2 = tf.math.pow(l1, 2.)
        probiou = - tf.math.log(1. - l2 + EPS)
    else:
        probiou = l1

    return probiou

def main():

    g1 = tf.random.Generator.from_seed(1)
    P =  g1.normal(shape=[8, 5])
    g2 = tf.random.Generator.from_seed(2)
    T =  g2.normal(shape=[8, 5])

    LOSS        = probiou_loss(P,T,mode='l1')
    REDUCE_LOSS = tf.reduce_mean(LOSS)
    print(REDUCE_LOSS)

if __name__ == '__main__':
    main()
