from GraphWalkTest import *
import time
import numpy as np
from tqdm import trange
from SyncMap import *
simulation = 30
time_delay = 10
sequence_length = 100000
env=GraphWalkTest(time_delay,'fixed_10_20.dot')
output_size=env.getOutputSize()
use_time=[]

for i in trange(simulation):
    adp_rate=output_size*0.001
    map_dim=3
    eps=0.5
    algorithm_class = SyncMap(output_size, map_dim, adp_rate)
    start=time.time()
    neuron_group = algorithm_class
    input_sequence, input_class = env.getSequence(sequence_length)
    neuron_group.input(input_sequence)    
    end=time.time()
    use_time.append(end-start)
print("mean: " +str(np.mean(use_time))+" std:"+str(np.std(use_time)))
