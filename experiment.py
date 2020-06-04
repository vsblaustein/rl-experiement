import gym, time, random, datetime, os
import numpy as np
import tensorflow as tf
from envs.raindrops_gym import RaindropsGym
from keras.losses import MeanSquaredError
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

env = RaindropsGym()

model = Sequential()
model.add(Conv2D(32, kernel_size=(8,8), strides=(4,4), padding='same', input_shape=(100,60,4)))
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(4,4), strides=(2,2), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(11))

try:
    model.load_weights("model.h5")
except:
    print('no weights file found')

folder_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") #creates unique folder name each time file is run

log_directory = os.path.join('logs', folder_name) #creates this folder in logs

writer = tf.summary.create_file_writer(logdir=log_directory)

model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001))

epsilon = 0.1
gamma = 0.99
state = env.no_op()
replay_memory = []

for current_frame in range(0,1000000):
    loss = 0
    if random.random() <= epsilon or current_frame < 3200:
        action = env.action_space.sample() #random action is taken
    else:
        #highest value is the action taken
        q = model.predict(state) #tensor (1,11)
        m = 0
        for x in range(0,11):
            if q[0][x] > m:
                m = q[0][x]
                action = x
    agent_action = env.step(action, state) #4-value tuple: state_next, reward, game_over, info 
    
    if epsilon > 0.0001 and current_frame > 3200:
        epsilon -= 0.0999/300000 #3,200 frames of self-play until annealing at around 0.0001
    
    memory = (state, action, agent_action[1],agent_action[2],agent_action[0])
    replay_memory.append(memory) #state, action, reward, game_over, state_next
    
    if len(replay_memory) > 50000:
        del replay_memory[0] #removes first item
    
    if current_frame > 33: #start training
        #randomly selects 32 observations and adds them to batch
        batch = random.sample(replay_memory, 32)
        
        state_list = []
        action_list = []
        reward_list = []
        game_over_list = []
        state_next_list = []
        
        for value in batch:
            state_list.append(value[0])
            action_list.append(value[1])
            reward_list.append(value[2])
            game_over_list.append(value[3])
            state_next_list.append(value[4])

        state_prepped = np.concatenate(state_list)
        state_next_prepped = np.concatenate(state_next_list)

        targets = model.predict(state_prepped) #state (32) x action (11)
        q_table_next = model.predict(state_next_prepped) #tensor (32,11)

        #updates q function of action
        for i in range(0,32):
            m = 0
            for j in range(0, 11): #finds max value in qtable for next state
                if q_table_next[i][j] > m:
                    m = q_table_next[i][j]
            targets[i][action_list[i]] = reward_list[i] + gamma * (m*np.invert(game_over_list[i]))

        new_loss = model.train_on_batch(state_prepped, targets)
        loss += new_loss

        with writer.as_default():
            if current_frame > 3200:
                tf.summary.scalar ('loss', loss, step=current_frame)
        
        model.save_weights("model.h5", overwrite=True)

    state = agent_action[0] #setting state to state_next for next iteration of loop

    env.render()











    




















