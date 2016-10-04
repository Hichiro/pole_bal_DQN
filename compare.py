#Get mean performance for each agent
#Set num_of_runs to average over
#Set episodes for session length
#Set agents to test in vers2train
'''
import sys
def trace(frame, event, arg):
    print "%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno)
    return trace
sys.settrace(trace)
'''
import numpy as np
import viz

num_of_runs = 1
episodes = 1
vers2train = [1, 7]
#recs = np.zeros([len(vers2train), num_of_runs])
recs = []
for i in vers2train:
    exec('import pole_'+str(i))
    recs.append([])
    for j in range(num_of_runs):
        print('Running version '+str(i)+' / iteration '+str(j+1)+'...')
        #Every recs cell is a sum training sesh
        recs[vers2train.index(i)].append(eval('pole_'+str(i)+'.execute(episodes)'))
#Recording complete
print('Recording complete. Initiating averaging...')
avg = []
viz.init()
over = False
for i in range(len(vers2train)):
    mini = min([x.shape[0] for x in recs[i]])
    temp = [x[:mini, :] for x in recs[i]]
    avg.append(np.mean(np.array(temp), axis=0))
    if i == len(vers2train)-1:
        over = True
    viz.show(avg[i], vers2train[i], over)
