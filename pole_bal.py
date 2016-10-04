# pole_bal w/ Viz (Tensorboard)

import tensorflow as tf
import numpy as np
import shutil
import os
import gym
import random

#flags
with open('run_increment', 'r') as fr:
    last_run = int(fr.read())+1
with open('run_increment', 'w') as fw:
    fw.write(str(last_run))
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summaries_dir', '/tmp/pole_bal/run_%s' %last_run, 'Summaries directory')

#Functions

def rev(a):
    if a==1:
        return 0
    else:
        return 1

def get_action(epsilon):
    action = sess.run(y, feed_dict={x:[observation]})
    action = np.argmax(action)
    if np.random.rand() > 1-epsilon:
        return random.randint(0, 1)
    else:
        return action

def build_network(l):
    x = tf.placeholder(tf.float32, [None, l[0]])
    xhat = tf.placeholder(tf.float32, [None, l[0]])
    W = [0]
    b = [0]
    y2 = [0]
    What = [0]
    bhat = [0]
    y2hat = [0]
    for i in range(len(l[1:])):
        k = i + 1
        W.append(tf.Variable(tf.random_uniform([l[k], l[k-1]])))
        b.append(tf.Variable(tf.random_uniform([l[k], 1])))
        for j in range(l[k]):
            for m in range(l[k-1]):
                tf.scalar_summary('W'+str(i+1)+str(j)+str(m), W[-1][j, m])
        for j in range(l[k]):
            tf.scalar_summary('b'+str(i+1)+str(j), b[-1][j, 1])
        if i>0 and i<len(l[1:])-1:
            y2.append(tf.sigmoid(tf.matmul(W[k], y2[k-1]) + b[k]))
        elif i==0:
            y2.append(tf.sigmoid(tf.matmul(W[k], tf.transpose(x)) + b[k]))
        else:
            y = tf.transpose(tf.matmul(W[k], y2[k-1]) + b[k])
    for i in range(len(l[1:])):
        k = i + 1
        What.append(tf.Variable(tf.random_uniform([l[k], l[k-1]])))
        bhat.append(tf.Variable(tf.random_uniform([l[k], 1])))
        for j in range(l[k]):
            for m in range(l[k-1]):
                tf.scalar_summary('What'+str(i+1)+str(j)+str(m), What[-1][j, m])
        for j in range(l[k]):
            tf.scalar_summary('bhat'+str(i+1)+str(j), bhat[-1][j, 1])
        if i>0 and i<len(l[1:])-1:
            y2hat.append(tf.sigmoid(tf.matmul(What[k], y2hat[k-1]) + bhat[k]))
        elif i==0:
            y2hat.append(tf.sigmoid(tf.matmul(What[k], tf.transpose(xhat)) + bhat[k]))
        else:
            yhat = tf.transpose(tf.matmul(What[k], y2hat[k-1]) + bhat[k])
    return x, y, xhat, yhat, W, b, y2, What, bhat, y2hat

#Network setup
l = [4, 2, 2]
x, y, xhat, yhat, W, b, y2, What, bhat, y2hat = build_network(l)
tf.scalar_summary('y0', y[0, 0])   #Viz
tf.scalar_summary('y1', y[0, 1])   #Viz

#Traning Setup
y_ = tf.placeholder(tf.float32, [None, l[-1]])
least_squares = tf.reduce_sum(tf.square(y_ - y))
elapsed_eps = tf.Variable(0)
tf.scalar_summary('episodes', elapsed_eps)
tf.scalar_summary('loss', least_squares)    #Viz
merged = tf.merge_all_summaries()    #Viz
alpha = 0.1
train_step = tf.train.GradientDescentOptimizer(alpha).minimize(least_squares)

#Initialization
sess = tf.Session()
writer = tf.train.SummaryWriter(FLAGS.summaries_dir, sess.graph)   #Viz
sess.run(tf.initialize_all_variables())

#Training
#Copy W, b
for m in range(len(W[1:])):
    #Copy W, b
    sess.run(What[m+1].assign(sess.run(W[m+1])))
    sess.run(bhat[m+1].assign(sess.run(b[m+1])))
episodes = 50
runs = 5
env = gym.make('CartPole-v0')
k = 0
counter = 0
delta = 400
for i in range(episodes):
    print 'episode %d/%d in progress...' %(i+1, episodes)
    observation = env.reset()
    done = False
    while not done:
        env.render()
        #ACTION MUST BE 0 OR 1
        action = get_action(0.1)
        observationPrime, reward, done, info = env.step(action)
        if not 'exp' in locals():
            exp = np.empty([0, 5])
            exp = np.append(exp, [[observation, action, reward, observationPrime, done]], axis=0)
        else:
            exp = np.append(exp, [[observation, action, reward, observationPrime, done]], axis=0)
        for j in range(runs):
            rndIdx = random.randint(0,exp.shape[0]-1)
            #We assume here that state = observation
            s = exp[rndIdx, 0]
            a = exp[rndIdx, 1]
            r = exp[rndIdx, 2]
            sPrime = exp[rndIdx, 3]
            term = exp[rndIdx, 4]
            goal = np.zeros([1, 2])
            if term:
                goal[0, a] = r
            else:
                goal[0, a] = r + np.argmax(sess.run(yhat, feed_dict={xhat:[sPrime]})[0,:])
            goal[0, rev(a)] = sess.run(y, feed_dict={x:[s]})[0, rev(a)]
            sess.run(train_step, feed_dict={x:[s], y_:goal})
            summ = sess.run(merged, feed_dict={x:[s], y_:goal})    #Viz
            k = k+1
            writer.add_summary(summ, k)
        counter = counter + 1
        if counter > delta:
            for m in range(len(W[1:])):
                #Copy W, b
                sess.run(What[m+1].assign(sess.run(W[m+1])))
                sess.run(bhat[m+1].assign(sess.run(b[m+1])))
            counter = 0
        observation = observationPrime
    sess.run(elapsed_eps.assign(sess.run(elapsed_eps)+1))
