# pole_bal

import tensorflow as tf
import numpy as np
import gym
import random

#Functions

def rev(a):
    if a==1:
        return 0
    else:
        return 1

def get_action(epsilon, sess, y, s, x):
    action = sess.run(y, feed_dict={x:[s]})
    action = np.argmax(action)
    if np.random.rand() > 1-epsilon:
        return random.randint(0, 1)
    else:
        return action

def build_network(l):
    x = tf.placeholder(tf.float32, [None, l[0]])
    W = [0]
    b = [0]
    y2 = [0]
    for i in range(len(l[1:])):
        k = i + 1
        W.append(tf.Variable(tf.random_uniform([l[k], l[k-1]])))
        b.append(tf.Variable(tf.random_uniform([l[k], 1])))
        for j in range(l[k]):
            for m in range(l[k-1]):
                pass
                #tf.scalar_summary('W'+str(i+1)+str(j)+str(m), W[-1][j, m])
        for j in range(l[k]):
            pass
            #tf.scalar_summary('b'+str(i+1)+str(j), b[-1][j, 1])
        if i>0 and i<len(l[1:])-1:
            y2.append(tf.sigmoid(tf.matmul(W[k], y2[k-1]) + b[k]))
        elif i==0:
            y2.append(tf.sigmoid(tf.matmul(W[k], tf.transpose(x)) + b[k]))
        else:
            y = tf.transpose(tf.matmul(W[k], y2[k-1]) + b[k])
    return x, y, W, b, y2

#TODO: return array containg WEIGHTS too

def execute(episodes):
    #Network setup
    l = [4, 2, 2]
    x, y, W, b, y2 = build_network(l)
    
    #Traning Setup
    y_ = tf.placeholder(tf.float32, [None, l[-1]])
    least_squares = tf.reduce_sum(tf.square(y_ - y))
    alpha = 0.5
    train_step = tf.train.GradientDescentOptimizer(alpha).minimize(least_squares)
    
    #Initialization
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    #Training
    runs = 5
    env = gym.make('CartPole-v0')
    #for each step, append y's, loss, episodes
    for i in range(episodes):
        print 'episode %d/%d in progress...' %(i+1, episodes)
        s = env.reset()
        done = False
        while not done:
            env.render()
            #ACTION MUST BE 0 OR 1
            a = get_action(0.1, sess, y, s, x)
            sPrime, r, done, info = env.step(a)
            goal = np.zeros([1, 2])
            if done:
                goal[0, a] = r
            else:
                goal[0, a] = r + np.argmax(sess.run(y, feed_dict={x:[sPrime]})[0,:])
            goal[0, rev(a)] = sess.run(y, feed_dict={x:[s]})[0, rev(a)]
            ys = sess.run(y, feed_dict={x:[s]})
            if 'res' not in locals():
                res = np.array([[ys[0, 0], ys[0,1], sess.run(least_squares, feed_dict={x:[s], y_:goal}), i+1]])
            else:
                res = np.append(res, np.array([[ys[0, 0], ys[0,1], sess.run(least_squares, feed_dict={x:[s], y_:goal}), i+1]]), axis=0)
            sess.run(train_step, feed_dict={x:[s], y_:goal})
            s = sPrime
    return res
