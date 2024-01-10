#Ji Min Ryu 
# This python script implements q learning in a static environment and 
# shows how it converges to an effective policy

import random


#environment variables
num_states = 16 # for simplicity, every action goes to the same state which is shown as the row number in the array
num_actions = 4
num_episodes = 5000 # how many times we want to run this
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1   # exploration exploitation trade off

#for printing at the end
group_size = 500
curr_index = 0

# Initialize Q table 
Q = [[0,0,0,0],
     [0,0,0,0],
     [0,0,0,0],
     [0,0,0,0],
     [0,0,0,0],
     [0,0,0,0],
     [0,0,0,0],
     [0,0,0,0],
     [0,0,0,0],
     [0,0,0,0],
     [0,0,0,0],
     [0,0,0,0],
     [0,0,0,0],
     [0,0,0,0],
     [0,0,0,0],
     [0,0,0,0]]

# Define the reward table
# this is our environment
R = [[ 3, 9, 20, 6],
     [ 8, 72, 1, 74],
     [ 94, 8, 6, 43],
     [ 14, 8, 63, 3],
     [ 2, 34, 6, 2],
     [ 0, 54, 34, 3],
     [ 1, 38, 5, 8],
     [ 7, 9, 0, 59],
     [ 4, 8, 7, 49],
     [ 5, 1, 0, 88],
     [ 29, 12, 14, 74],
     [ 47, 14, 6, 74],
     [ 47, 1, 43, 4],
     [ 43, 5, 73, 0],
     [ 74, 14, 8, 24],
     [ 7, 94, 4, 64]]

Avg_Reward = []


# Q learning algorithm
for iteration in range(num_episodes):
    state = 0 # Start off at state 0
    cum_reward = 0 # Cumulative reward

    while state < num_states:  # while we are not at the goal state
        # Epsilon greedy stradegy
        if random.random() < epsilon:
            action = random.randint(0, num_actions-1)  # Explore
        else:
            action = Q[state].index(max(Q[state]))  # Exploit, this will give us the action with the highest value

        # Take the chosen action and observe the next state and reward
        next_state = state + 1
        reward = R[state][action]
        cum_reward = cum_reward + reward

        # Q learning update rule
        if(next_state >= num_states): # If there is no next state
            Q[state][action] = Q[state][action] + alpha * (reward - Q[state][action]) # Dont account for the next state 
        else: 
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * max(Q[next_state]) - Q[state][action])

        # Now we take the next state
        state = next_state
    
    Avg_Reward.append(cum_reward/num_states) # Store our avg reward



# Print the learned Q table
for i in range(int(len(Avg_Reward)/group_size)):
    curr_group = Avg_Reward[(i)*group_size :group_size*(i+1) - 1]

    average = sum(curr_group) / len(curr_group)
    print(f"Mean of average reward values from iteration {(i)*group_size + 1} to {group_size*(i+1)}: {average}")

print('''\nThe most optimal policy will result in an E(X) of around 63. This can be achieved if we
reduce the exploration factor epsilon. But it will take more runs to acheive a better policy
rather than a higher epsilon factor in which we achieve a better policy quicker but we 
converge to a good policy, just not the optimal one''')

