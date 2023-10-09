import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_states = 6 #步数
Actions = ['left', 'right']
epsilon = 0.9
alpha = 0.1
LAMBDA = 0.9
max_epsilon = 30
fresh_time = 0.1

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states,len(actions))),
        columns=actions
    )

    #print(table)
    return table

def choose_action(state,q_table):
    state_actions = q_table.loc[state,:]
    if (np.random.uniform() > epsilon) or (state_actions.all() == 0):
        action_name = np.random.choice(Actions)
    else:
        action_name = state_actions.argmax()
        if action_name ==1:
            action_name='right'
    return action_name


def get_env_feedback(S,A):
    if A== 'right':
        if S == N_states - 2:
            S_='terminal'
            R =1
        else:
            S_=S+1
            R = 0
    else:
        R=0
        if S==0:
            S_=S
        else:
            S_=S-1
    return S_,R

def update_env(S,Episode,step_counter):
    env_list = ['-']*(N_states-1) + ['T']
    if S=='terminal':
        interaction = 'Episode: %s  :  total_step = %s' %(Episode+1,step_counter)
        print('\r{}'.format(interaction),end='')
        time.sleep(1)
        print('\r                   ',end='')
    else:
        env_list[S] ='o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(fresh_time)

def rl():
    q_table = build_q_table(N_states,Actions)
    for epside in range(max_epsilon):
        step_counter = 0
        S=0
        is_terminal = False
        update_env(S,epside,step_counter)

        while not is_terminal:

            A = choose_action(S,q_table)
            S_,R = get_env_feedback(S,A)

            q_predict = q_table.loc[S,A]
            if S_ !='terminal':
                q_target = R+LAMBDA * q_table.iloc[S_,:].max()
            else:
                q_target = R
                is_terminal = True
            q_table.loc[S,A] += alpha * (q_target-q_predict)
            S=S_

            update_env(S,epside,step_counter+1)
            step_counter +=1
    return q_table

if __name__ =='__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)