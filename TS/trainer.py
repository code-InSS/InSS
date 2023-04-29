#import gym
import numpy as np
import matplotlib.pyplot as plt
from env import Environment, Setting
import time
import os


TRAINING_EVALUATION_RATIO = 4
RUNS = 10
EPISODES_PER_RUN = 2000
STEPS_PER_EPISODE = 200
SAVE_EPI_NUM = 50
ALPHA_INITIAL_Initial = 1.
REPLAY_BUFFER_BATCH_SIZE_Initial = 100
DISCOUNT_RATE_Initial = 0.99
LEARNING_RATE_Initial = 10 ** -4
SOFT_UPDATE_INTERPOLATION_FACTOR_Initial = 0.01
TRAIN_PER_STEP = 5
import torch

layer_num=17
hidden = 512

full_memory=[8000,8000,6000,6000]
def state_norl(env,state):
    for i in range(env.model_num):
        state[3*i]/=100
        state[3 * i+1] /= 100
    for i in range(env.GPU_num):
        state[3*env.model_num+i]/=full_memory[i]
    return state

class Actor(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation, models_num, GPUs_num):
        super(Actor, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=hidden)
        self.layer_2 = torch.nn.Linear(in_features=hidden, out_features=hidden//2)
        self.layer_list =[]
        for i in range(models_num):
            self.layer_list.append(torch.nn.Linear(in_features=hidden//2, out_features=GPUs_num))
            self.layer_list.append(torch.nn.Linear(in_features=hidden//2, out_features=layer_num))
        self.output_layer = torch.nn.Linear(in_features=hidden//2, out_features=output_dimension)
        self.output_activation = output_activation
        self.num = models_num

    def forward(self, inpt):
        layer_1_output = torch.nn.functional.relu(self.layer_1(inpt))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))

        x = []
        for i in range(self.num):
            x.append(self.output_activation(self.layer_list[2*i](layer_2_output)))
            x.append(self.output_activation(self.layer_list[2*i + 1](layer_2_output)))
        
        output = x[0]
        for i in range(1, len(x)):
            output = torch.cat([output, x[i]], dim=1)
        return output

class Critic(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation=torch.nn.Identity()):
        super(Critic, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=hidden)
        self.layer_2 = torch.nn.Linear(in_features=hidden, out_features=hidden//2)
        self.output_layer = torch.nn.Linear(in_features=hidden//2, out_features=output_dimension)
        self.output_activation = output_activation

    def forward(self, inpt):
        layer_1_output = torch.nn.functional.relu(self.layer_1(inpt))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = self.output_layer(layer_2_output)


        output = self.output_activation(layer_3_output)
        return output

class SACAgent:
    ALPHA_INITIAL = ALPHA_INITIAL_Initial
    REPLAY_BUFFER_BATCH_SIZE = REPLAY_BUFFER_BATCH_SIZE_Initial
    DISCOUNT_RATE = DISCOUNT_RATE_Initial
    LEARNING_RATE = LEARNING_RATE_Initial
    SOFT_UPDATE_INTERPOLATION_FACTOR = SOFT_UPDATE_INTERPOLATION_FACTOR_Initial

    def __init__(self, environment):
        self.environment = environment
        self.state_dim = self.environment.state_dim
        self.action_dim = self.environment.action_dim

        self.GPU_num = environment.GPU_num
        self.model_num = environment.model_num
        action_dim = (self.GPU_num+layer_num)*self.model_num

        self.critic_local = Critic(input_dimension=self.state_dim,
                                    output_dimension=action_dim)
        self.critic_local2 = Critic(input_dimension=self.state_dim,
                                     output_dimension=action_dim)
        self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.LEARNING_RATE)
        self.critic_optimiser2 = torch.optim.Adam(self.critic_local2.parameters(), lr=self.LEARNING_RATE)

        self.critic_target = Critic(input_dimension=self.state_dim,
                                     output_dimension=action_dim)
        self.critic_target2 = Critic(input_dimension=self.state_dim,
                                      output_dimension=action_dim)

        self.soft_update_target_networks(tau=1.)

        self.actor_local = Actor(
            input_dimension=self.state_dim,
            output_dimension=self.action_dim,
            output_activation=torch.nn.Softmax(dim=1),
            models_num=self.model_num,
            GPUs_num=self.GPU_num

        )
        self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=self.LEARNING_RATE)

        self.replay_buffer = ReplayBuffer(self.environment)

        self.target_entropy = 0.98 * -np.log(1 / self.action_dim)
        self.log_alpha = torch.tensor(np.log(self.ALPHA_INITIAL), requires_grad=True)
        self.alpha = self.log_alpha
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.LEARNING_RATE)

    def get_next_action(self, state, evaluation_episode=False):
        if evaluation_episode:
            discrete_action = self.get_action_deterministically(state)
        else:
            discrete_action = self.get_action_nondeterministically(state)
        return discrete_action

    def get_action_nondeterministically(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action =[]
        temp = self.GPU_num + layer_num
        for i in range(self.model_num):
            GPU_select = action_probabilities[i * temp: i * temp + self.GPU_num]
            resource = action_probabilities[i * temp + self.GPU_num: i * temp + self.GPU_num +layer_num]
            discrete_action.append(np.random.choice(range(0, self.GPU_num), p=GPU_select))
            discrete_action.append(np.random.choice(range(0, layer_num), p=resource))
        return discrete_action

    def get_action_deterministically(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action = []
        temp = self.GPU_num + layer_num
        for i in range(self.model_num):
            GPU_select = action_probabilities[i * temp: i * temp + self.GPU_num]
            resource = action_probabilities[i * temp + self.GPU_num: i * temp + self.GPU_num + layer_num]
            discrete_action.append(np.argmax(GPU_select))
            discrete_action.append(np.argmax(resource))
        return discrete_action


    def train_on_transition(self, state, discrete_action, next_state, reward, done):
        count = 0
        for i in range(self.model_num):
            discrete_action[2 * i] = discrete_action[2 * i] + count
            count += self.GPU_num
            discrete_action[2 * i + 1] = discrete_action[2 * i + 1] + count
            count += layer_num
        transition = (state, discrete_action, reward, next_state, done)
        self.train_networks(transition)

    def train_networks(self, transition):
        # Set all the gradients stored in the optimisers to zero.
        self.critic_optimiser.zero_grad()
        self.critic_optimiser2.zero_grad()
        self.actor_optimiser.zero_grad()
        self.alpha_optimiser.zero_grad()
        # Calculate the loss for this transition.
        self.replay_buffer.add_transition(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network
        # parameters.
        if self.replay_buffer.get_size() >= self.REPLAY_BUFFER_BATCH_SIZE:
            # get minibatch of 100 transitions from replay buffer
            minibatch = self.replay_buffer.sample_minibatch(self.REPLAY_BUFFER_BATCH_SIZE)
            minibatch_separated = list(map(list, zip(*minibatch)))

            # unravel transitions to get states, actions, rewards and next states
            states_tensor = torch.tensor(np.array(minibatch_separated[0]), dtype=torch.float32)
            actions_tensor = torch.tensor(np.array(minibatch_separated[1]),dtype=torch.float32)
            rewards_tensor = torch.tensor(np.array(minibatch_separated[2])).float()
            next_states_tensor = torch.tensor(np.array(minibatch_separated[3]))
            done_tensor = torch.tensor(np.array(minibatch_separated[4]))
            #actions_tensor_2 = torch.tensor(np.array(minibatch_separated[5]), dtype=torch.float32)

            critic_loss, critic2_loss = \
                self.critic_loss(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor)

            critic_loss.backward()
            critic2_loss.backward()
            self.critic_optimiser.step()
            self.critic_optimiser2.step()

            actor_loss, log_action_probabilities = self.actor_loss(states_tensor)

            actor_loss.backward()
            self.actor_optimiser.step()

            alpha_loss = self.temperature_loss(log_action_probabilities)

            alpha_loss.backward()
            self.alpha_optimiser.step()
            self.alpha = self.log_alpha.exp()

            self.soft_update_target_networks()

    def critic_loss(self, states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor):
        with torch.no_grad():
            action_probabilities, log_action_probabilities = self.get_action_info(next_states_tensor)
            next_q_values_target = self.critic_target.forward(next_states_tensor)
            next_q_values_target2 = self.critic_target2.forward(next_states_tensor)


            temp = action_probabilities * (
                    torch.min(next_q_values_target, next_q_values_target2) - self.alpha * log_action_probabilities
            )

            list = []
            for i in range(self.model_num):
                list.append(self.GPU_num)
                list.append(layer_num)
            soft_state_value = torch.split(temp, list, dim=1)

            next_q_values = []
            for i in range(len(soft_state_value)):
                next_q_values.append(rewards_tensor + ~done_tensor * self.DISCOUNT_RATE*soft_state_value[i].sum(dim=1))

        temp = torch.split(actions_tensor, 1, dim=1)

        soft_q_value = self.critic_local(states_tensor)
        soft_q_value2 = self.critic_local2(states_tensor)
        soft_q_values = []
        soft_q_values2 = []

        critic_square_error = 0
        critic2_square_error = 0
        for i in range(len(temp)):
            temp1 = soft_q_value.gather(1, temp[i].type(torch.int64).squeeze().unsqueeze(-1)).squeeze(-1)
            soft_q_values.append(temp1)
            temp2 = soft_q_value2.gather(1, temp[i].type(torch.int64).squeeze().unsqueeze(-1)).squeeze(-1)
            soft_q_values2.append(temp2)
            critic_square_error += torch.nn.MSELoss(reduction="none")(temp1, next_q_values[i])
            critic2_square_error += torch.nn.MSELoss(reduction="none")(temp2, next_q_values[i])

        weight_update = [min(l1.item(), l2.item()) for l1, l2 in zip(critic_square_error, critic2_square_error)]
        self.replay_buffer.update_weights(weight_update)
        critic_loss = critic_square_error.mean()
        critic2_loss = critic2_square_error.mean()
        return critic_loss, critic2_loss

    def actor_loss(self, states_tensor,):
        action_probabilities, log_action_probabilities = self.get_action_info(states_tensor)
        q_values_local = self.critic_local(states_tensor)
        q_values_local2 = self.critic_local2(states_tensor)
        inside_term = self.alpha * log_action_probabilities - torch.min(q_values_local, q_values_local2)
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        return policy_loss, log_action_probabilities

    def temperature_loss(self, log_action_probabilities):
        alpha_loss = -(self.log_alpha * (log_action_probabilities + self.target_entropy).detach()).mean()
        return alpha_loss

    def get_action_info(self, states_tensor):
        action_probabilities = self.actor_local.forward(states_tensor)
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action_probabilities, log_action_probabilities

    def get_action_probabilities(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probabilities = self.actor_local.forward(state_tensor)
        return action_probabilities.squeeze(0).detach().numpy()

    def soft_update_target_networks(self, tau=SOFT_UPDATE_INTERPOLATION_FACTOR):
        self.soft_update(self.critic_target, self.critic_local, tau)
        self.soft_update(self.critic_target2, self.critic_local2, tau)

    def soft_update(self, target_model, origin_model, tau):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def predict_q_values(self, state):
        q_values = self.critic_local(state)
        q_values2 = self.critic_local2(state)
        return torch.min(q_values, q_values2)


class ReplayBuffer:

    def __init__(self, environment, capacity=10000):
        transition_type_str = self.get_transition_type_str(environment)
        self.buffer = np.zeros(capacity, dtype=transition_type_str)
        self.weights = np.zeros(capacity)
        self.head_idx = 0
        self.count = 0
        self.capacity = capacity
        self.max_weight = 10**-2
        self.delta = 10**-4
        self.indices = None

    def get_transition_type_str(self, environment):
        state_dim = environment.state_dim
        state_dim_str = '' if state_dim == () else str(state_dim)
        state_type_str = "float32"
        action_dim = environment.model_num *2 #environment.action_dim
        action_dim_str = '' if action_dim == () else str(action_dim)
        action_type_str = "float32"#"int"

        # type str for transition = 'state type, action type, reward type, state type'
        transition_type_str = '{0}{1}, {2}{3}, float32, {0}{1}, bool'.format(state_dim_str, state_type_str,
                                                                             action_dim_str, action_type_str)

        return transition_type_str

    def add_transition(self, transition):
        self.buffer[self.head_idx] = transition
        self.weights[self.head_idx] = self.max_weight

        self.head_idx = (self.head_idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample_minibatch(self, size=100):
        set_weights = self.weights[:self.count] + self.delta
        probabilities = set_weights / sum(set_weights)
        self.indices = np.random.choice(range(self.count), size, p=probabilities, replace=False)
        return self.buffer[self.indices]

    def update_weights(self, prediction_errors):
        max_error = max(prediction_errors)
        self.max_weight = max(self.max_weight, max_error)
        self.weights[self.indices] = prediction_errors

    def get_size(self):
        return self.count

if __name__ == "__main__":


    time = time.time()
    print(time)
    path = str(time)
    dir = 'result/new_'+ path
    if not os.path.exists(dir):
        os.makedirs(dir)

    arg = Setting()
    env = Environment(args=arg)
    agent_results = []
    for run in range(RUNS):
        agent = SACAgent(env)
        run_results = []
        run_SLO = []
        run_throughput = []
        run_memory = []
        run_throughput_norm = []
        for episode_number in range(EPISODES_PER_RUN):
            print('\r', f'Run: {run + 1}/{RUNS} | Episode: {episode_number + 1}/{EPISODES_PER_RUN}', end=' ')
            evaluation_episode = episode_number % TRAINING_EVALUATION_RATIO == 0
            episode_reward = 0
            state = env.reset(evaluation_episode)
            state = state_norl(env, state)
            task_num = env.tasks_num
            num = 0
            done = False
            i = 0
            SLO_count = 0
            throughput = 0
            throughput_norm = 0
            memory_v = 0
            while not done and i < STEPS_PER_EPISODE:
                i += 1
                action = agent.get_next_action(state, evaluation_episode=evaluation_episode)
                number = 0

                next_state, reward, done, info = env.step(action)
                next_state = state_norl(env, next_state)
                SLO_count += sum(info[0])
                throughput += info[1]
                throughput_norm += sum(info[3])
                if info[2] == False:
                    memory_v += 1
                if not evaluation_episode:
                    agent.train_on_transition(state, action, next_state, reward, done)
                # else:
                episode_reward += reward
                num += 1
                state = next_state

            if evaluation_episode:
                print("episoed:%d, reward:%.5f, SLO:%d, tasks:%d, throughput:%d,throuput_norm:%.3f, memmory_v:%d" % (
                    episode_number, episode_reward / num, SLO_count, task_num, throughput / num,throughput_norm/num, memory_v))
                run_results.append(episode_reward / num)
                run_SLO.append(SLO_count / task_num)
                run_throughput.append(throughput / num)
                run_throughput_norm.append(throughput_norm/num)
                run_memory.append(memory_v)
                if SLO_count / task_num < 0.01 and memory_v == 0:
                    torch.save(agent.actor_local,
                               "{}/run{}-episode{}-actor.pkl".format(dir, run, episode_number))
                    np.savez("{}/run{}-episode{}-reward".format(dir, run, episode_number),
                             reward=run_results, SLO=run_SLO, throughput=run_throughput, throughput_norm = throughput_norm,memory=run_memory)
        agent_results.append(run_results)


    n_results = EPISODES_PER_RUN // TRAINING_EVALUATION_RATIO
    results_mean = [np.mean([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
    results_std = [np.std([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
    mean_plus_std = [m + s for m, s in zip(results_mean, results_std)]
    mean_minus_std = [m - s for m, s in zip(results_mean, results_std)]

    x_vals = list(range(len(results_mean)))
    x_vals = [x_val * (TRAINING_EVALUATION_RATIO - 1) for x_val in x_vals]


    ax = plt.gca()
    #ax.set_ylim([-30, 0])
    ax.set_ylabel('Episode Score')
    ax.set_xlabel('Training Episode')
    ax.plot(x_vals, results_mean, label='Average Result', color='blue')
    ax.plot(x_vals, mean_plus_std, color='blue', alpha=0.1)
    ax.fill_between(x_vals, y1=mean_minus_std, y2=mean_plus_std, alpha=0.1, color='blue')
    ax.plot(x_vals, mean_minus_std, color='blue', alpha=0.1)
    plt.legend(loc='best')
    plt.savefig("./result/1.png".format(path))
    plt.show()
