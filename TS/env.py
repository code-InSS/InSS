import math
import random
import numpy as np
import copy
#import gym
from gym import spaces

from inf_time_new import getExecutionLatency
from fit_memory import get_memory

class Setting(object):
    def __init__(self):
        self.env = {
            'model_num': 6,  # 0:res, 1:vgg, 2: googlenet, 3:alex, 'densenet', 'inception', 'mobilenet', 'squeezenet'
            'model_name': ["alexnet", "resnet50", "vgg16", "mnasnet","mobilenet_v3","efficientnet"],#["alexnet", "resnet50", "vgg16", "resnet18","vgg19", "mnasnet","mobilenet_v3","efficientnet"],
            'model_max_rate':[500,400,200,400,450,400],#[600,500,200,550,550,600],#[400,30,15,35,30,350],#[35,25,15,25,12,35,25,30],
            'model_min_rate':[100,100,50,100,100,100],#[10,10,5,10,5,10,10,10],
            'GPU_type': [0,0,1,1],  #total:4 GPU  [0,0,1,1]
            'GPU_memory': [8000, 8000, 6000, 6000],   #two type of GPU
            'arr_rate': [[i for i in range(100, 500)],
                         [i for i in range(100, 400)],
                         [i for i in range(50, 200)],
                         [i for i in range(100, 400)],
                         [i for i in range(100, 450)],
                         [i for i in range(100, 400)]
                         ],  # x个任务/s
            'total_time': 200,  # s
            'SLO': [25, 80, 150, 50, 60, 40],#[25, 80, 150, 60, 200, 50, 60, 40],# 延迟上限
            'model_memory': [1000, 1900, 1500, 1500, 1000, 1900, 1500, 1500],
            'SLO_penalty': 40,
            'max_throughput': [788.6414427571721, 215.8675974375845, 146.90274566576682, 443.52946070316005, 306.5129790635446, 217.3192926197879]

        }


class Task(object):
    def __init__(self, model, batch_size, SLO, start_time):
        self.model = model
        self.batch_size = batch_size

        self.SLO = SLO/1000  # 延迟上限
        self.start_t = start_time

        self.data = 0

        self.exec_t = 0
        self.wait_t = 0
        self.t_inf = 0
        self.time = 0

        self.comp = False
        self.drop = False


class Workload(object):
    def __init__(self, time, index, tasks):
        self.time = time
        self.value = index
        self.uncomp = tasks

class Model(object):
    def __init__(self, model_type, model_memory):
        self.type = model_type
        name = ["alexnet", "resnet50", "vgg16", "mnasnet","mobilenet_v3","efficientnet"]
        self.name = name[model_type]
        self.memory = model_memory
        self.tasks = []

        self.generate_rate = 20
        self.arr_rate = 0
        self.workload = Workload(0, 0,[])
        self.uncomp = []

        self.GPU = 0
        self.batch_size = 32
        self.max_batch = 32
        self.allo_res = 0


class GPU(object):
    def __init__(self, type, memory):
        self.type = type   #0:2060s    1:2060
        full_res=[102,105]
        self.full_res=full_res[type]
        sm=[6,7]
        self.sm = sm[type]
        self.memory = memory
        self.models = []
        self.res = {}
        self.batch={}
        self.decision = []
        self.throughput = 0



class Environment(object):
    def __init__(self, args):
        self.GPU_list = args.env['GPU_type']
        self.GPU_num = len(self.GPU_list)
        self.GPU_memory = args.env['GPU_memory']
        self.model_num = args.env['model_num']
        self.arr_rate = args.env['arr_rate']
        self.model_max_rate = args.env['model_max_rate']
        self.total_time = args.env['total_time']
        self.SLO = args.env['SLO']
        self.model_memory = args.env['model_memory']
        self.SLO_penalty = args.env['SLO_penalty']
        self.model_min_rate =args.env['model_min_rate']
        self.max_throughput = args.env['max_throughput']

        self.action_space = (
            spaces.Box(low=-1, high=1,
                       shape=((self.model_num * 2,)), dtype=np.float32)
        )
        self.observation_space = spaces.Box(low=0, high=8000,
                                            shape=((self.model_num * 3 + self.GPU_num,)), dtype=np.float32)

        self.GPUs = []
        for i in self.GPU_list:
            memory = self.GPU_memory[i]
            self.GPUs.append(GPU(i,memory))

        self.models = []
        for i in range(self.model_num):
            self.models.append(Model(i, self.model_memory[i]))

        self.t = 0
        self.tasks = []  # tasks[t]:t-1 ~ t 产生的task
        self.tasks_num = self.generate_tasks(False)  # task生成



        # update model
        tasks = self.tasks[self.t]
        for task in tasks:
            i = task.model
            self.models[i].tasks.append(task)
            # self.models[i].workload += task.batch_size
            self.models[i].arr_rate += 1

        # state: model arr_rate...\workload...
        self.state_dim = self.model_num * 3 + self.GPU_num

        self.state = np.zeros(self.state_dim)
        for i in range(self.GPU_num):
            self.state[self.model_num * 3 + i] = self.GPUs[i].memory

        # action: model-gpu \allo (未归一化）
        self.action_dim = self.model_num * self.GPU_num + self.model_num
        # 归一化后的action维度
        #self.action_dim = self.model_num * 2
        self.action = np.zeros(self.action_dim)


    def generate_tasks(self,evaluate):
        init_rate=[200,150,80,200,250,250]
        flag = [1, 0, 0, 1, 0, 1]
        if not evaluate:
            init_rate = []
            flag=[]
            for i in range(self.model_num):
                init_rate.append(int(self.arr_rate[i][random.randint(0, len(self.arr_rate[i]) - 1)]))
                flag.append(random.randint(0,1))

        tasks_num = 0  # 任务总数
        for t in range(self.total_time):
            self.tasks.append([])
            for i in range(self.model_num):

                if t % 10 == 0:
                    self.models[i].generate_rate = init_rate[i]
                    if init_rate[i] > self.model_max_rate[i] or init_rate[i] < self.model_min_rate[i]:
                        flag[i] += 1
                    if flag[i] % 2==0:
                        init_rate[i] +=20
                    else:
                        init_rate[i] -=20


                rate = self.models[i].generate_rate
                for j in range(rate):
                    start_time = t + j / rate
                    #m = random.randint(0, self.model_num - 1)
                    m = self.models[i].type
                    task = Task(model=m, batch_size=1, SLO=self.SLO[m], start_time=start_time)
                    # tasks[t]: tasks in t ~ t+1
                    self.tasks[t].append(task)
                    tasks_num += 1

        return tasks_num


    def reset(self,evaluate):
        self.t = 0

        self.models = []
        for i in range(self.model_num):
            self.models.append(Model(i, self.model_memory[i]))

        for i in range(self.GPU_num):
            self.GPUs[i].throughput = 0
            self.GPUs[i].models=[]
            self.GPUs[i].res={}
            self.GPUs[i].batch = {}
            self.GPUs[i].decision=[]
            self.GPUs[i].memory = self.GPU_memory[i]

        self.tasks = []  # tasks[t]:t ~ t+1 产生的task
        self.tasks_num = self.generate_tasks(evaluate)  # task生成

        # update model
        tasks = self.tasks[self.t]
        for task in tasks:
            i = task.model
            self.models[i].tasks.append(task)
            # self.models[i].workload += task.batch_size
            self.models[i].arr_rate += 1


        # state: model arr_rate\workload
        self.state = np.zeros(self.state_dim)
        for i in range(self.model_num):
            self.state[3 * i] = self.models[i].arr_rate
            self.state[3 * i + 1] = self.models[i].workload.value
            self.state[3 * i + 2] = self.models[i].GPU
        for i in range(self.GPU_num):
            self.state[self.model_num * 3 + i] = self.GPUs[i].memory

        return self.state

    def get_norl_reward(self, slo, throughput_model,complete):
        throughput_norl = throughput_model/np.array(self.max_throughput)
        throughput_norl = throughput_norl#*complete
        reward = sum(throughput_norl) - 50*sum(slo)/len(slo)
        flag = True
        if False in self.place_flag:
            flag=False
        return reward,throughput_norl, flag

    def step(self, action):
        place, allo = self.del_action_dis(action)
        batch_action, flag = self.batch_policy([place, allo])
        self.place_flag = []
        models=[]
        decisions = []
        res=[]
        batch=[]
        available_memory = []
        total_memory = []#self.GPU_memory
        throughput_model = np.zeros(self.model_num)
        for i in range(self.GPU_num):
            models.append([])
            decisions.append([])
            res.append({})
            batch.append({})
            available_memory.append(self.GPUs[i].memory)
            total_memory.append(self.GPU_memory[i])
        for i in range(self.model_num):
            self.models[i].batch_size = batch_action[i]
            k = place[i]
            self.models[i].GPU = k
            self.models[i].allo_res = allo[i]

            model = self.models[i]
            place_flag = True

            if i in self.GPUs[k].models:
                if self.GPUs[k].res[i] == allo[i]:
                    place_flag = False
            if place_flag:

                model_memory = get_memory(model.name, model.batch_size, k)
            else:
                model_memory = get_memory(model.name, model.batch_size, k) - get_memory(model.name,
                                                                                        self.GPUs[k].batch[i], k)

            if available_memory[k] -model_memory >= 0:
                self.place_flag.append(True)
                available_memory[k] -= model_memory
                total_memory[k] -= model_memory
                models[k].append(i)
                res[k][i] = self.models[i].allo_res
                batch[k][i] = self.models[i].batch_size
                decisions[k].append([self.models[i].name, self.models[i].batch_size, self.models[i].allo_res])

            else:
                self.place_flag.append(False)
                continue

        for i in range(0, self.model_num):
            if allo[i] == 0 or self.place_flag[i] == False:
                throughput_model[i] = 0
                continue


            # model workload
            # task latency
            model = self.models[i]
            gpu = self.GPUs[model.GPU]
            inference, _ = getExecutionLatency(gpu.type,model.name, model.batch_size,model.allo_res,decisions[model.GPU])
            inference *= 10**(-3)


            throughput = math.floor(model.batch_size/inference)
            gpu.throughput += throughput
            throughput_model[i] = throughput


            temp = []
            tasks_i = copy.deepcopy(model.tasks)
            index = 0
            while(len(tasks_i)):
                #上一个batch的完成时间
                t_workload = model.workload.time
                t_startwait = tasks_i[0].start_t
                if t_workload > t_startwait + tasks_i[0].SLO - inference:
                    tasks_i.pop(0)
                    model.tasks[index].drop = True
                    index += 1
                    continue
                task = tasks_i.pop(0)
                temp.append(task)
                for j in range(model.batch_size - 1):
                    if len(tasks_i) == 0: break
                    '''
                    inference_time = get_inf_time(model.name, j+2, model.GPU, model.allo_res) * 10**(-3)
                    if tasks_i[0].start_t - t_startwait > task.SLO - inference_time or t_workload - t_startwait > task.SLO - inference_time:
                        break
                    else:
                        temp.append(tasks_i.pop(0))
                    '''
                    temp.append(tasks_i.pop(0))

                batch_num = len(temp)
                inf,_ = getExecutionLatency(gpu.type,model.name, model.batch_size,model.allo_res,decisions[model.GPU])
                inf *= 10**(-3)
                t_start = max([t_workload,temp[batch_num-1].start_t])
                #t_start = max([t_workload, temp[model.batch_size - 1].start_t])
                t_end = t_start + inf
                model.workload.time = t_end
                for j in temp:
                    model.tasks[index].time = t_end - model.tasks[index].start_t
                    model.tasks[index].exec_t = inf
                    model.tasks[index].wait_t = t_end - model.tasks[index].start_t - inf
                    index += 1
                temp = []

                if t_end > self.t + 1:
                    t_workload = model.workload.time
                    while(len(tasks_i)):
                        t_startwait = tasks_i[0].start_t
                        if t_workload > t_startwait + tasks_i[0].SLO - inference:
                            tasks_i.pop(0)
                            model.tasks[index].drop = True
                            index += 1
                        else: break
                    break

            model.workload.value = index

            model.workload.uncomp = []
            for task in tasks_i:
                model.workload.uncomp.append(task)


        SLO_count = np.zeros(self.model_num)
        cal = np.zeros(self.model_num)
        task_num = np.zeros(self.model_num)
        for i in range(self.model_num):
            task_num[i] = len(self.models[i].tasks)
            if allo[i] == 0 or self.place_flag[i]==False:
                #SLO_count += len(self.models[i].tasks)
                SLO_count[i] = len(self.models[i].tasks)
                self.models[i].workload.value = 0
                self.models[i].workload.time = self.t + 1
                continue
            model = self.models[i]
            for j in range(model.workload.value):
                task = model.tasks[j]
                if task.drop == True:
                    SLO_count[i] += 1
                    continue
                if task.time > task.SLO:
                    SLO_count[i] += 1
                    continue
                cal[i] +=1
            #self.models[i].workload.value = 0
            #self.models[i].workload.time = self.t + 1
        complete = cal/task_num
        SLO = SLO_count/task_num
        reward, throughput_norm, flag = self.get_norl_reward(SLO,throughput_model,complete)
        throughput = sum(throughput_model)


        self.t += 1
        if self.t == self.total_time:
            done = True
            return self.state, reward, done, [SLO_count, throughput,flag, throughput_norm]

        # update model
        tasks = self.tasks[self.t]
        # update arriving rate
        for i in range(self.model_num):
            self.models[i].arr_rate = 0
            self.models[i].tasks = []
            for task in self.models[i].workload.uncomp:
                self.models[i].tasks.append(task)
            #self.models[i].workload.uncomp=[]

        for task in tasks:
            i = task.model
            self.models[i].tasks.append(task)
            # self.models[i].workload += task.batch_size
            self.models[i].arr_rate += 1

        # update GPUs

        for i in range(self.GPU_num):
            #self.GPUs[i].memory = self.GPU_memory[i]
            self.GPUs[i].throughput = 0
            self.GPUs[i].models = models[i]
            self.GPUs[i].res = res[i]
            self.GPUs[i].batch = batch[i]
            self.GPUs[i].decision = decisions[i]
            self.GPUs[i].memory = total_memory[i]

        # update state
        self.state = np.zeros(self.state_dim)
        for i in range(self.model_num):
            self.state[3 * i] = self.models[i].arr_rate
            self.state[3 * i + 1] = len(self.models[i].workload.uncomp)
            self.models[i].workload.uncomp=[]
            self.state[3 * i + 2] = place[i]
        for i in range(self.GPU_num):
            self.state[self.model_num * 3 + i] = self.GPUs[i].memory

        done = False

        return self.state, reward, done, [SLO_count, throughput, flag, throughput_norm]

    def batch_policy(self, action):
        place = action[0]  # 计算干扰时使用
        allo = action[1]
        default = 4
        decisions = []
        for i in range(len(self.GPU_list)):
            decisions.append([])
        for i in range(self.model_num):
            k = place[i]
            decisions[k].append([self.models[i].name, default, allo[i]])
        batch = []
        flag = [True for _ in range(self.model_num)]

        gpu_memory = []
        total_memory = []
        for i in range(self.GPU_num):
            gpu_memory.append(self.GPUs[i].memory)
            total_memory.append(self.GPU_memory[i])

        for i in range(self.model_num):
            batch_temp = 0
            k = place[i]
            place_flag = True
            if i in self.GPUs[k].models:
                if self.GPUs[k].res[i] == allo[i]:
                    place_flag = False

            SLO = self.SLO[self.models[i].type] / 1000
            rate = self.state[3 * i] * 100
            workload = self.state[3 * i + 1] * 100

            task_i = self.models[i].tasks

            start = self.models[i].max_batch
            end = 1

            while start >= end:
                b = (start + end) // 2
                time = max([self.models[i].workload.time, self.t])
                inf, trp_cal = getExecutionLatency(self.GPUs[k].type, self.models[i].name, b, allo[i], decisions[k])
                inf = inf / 1000
                # inf = get_inf_time(self.models[i].name, b, self.models[i].GPU, allo[i]) / 1000
                throughput = math.floor(b * 1000 / trp_cal)
                num = math.ceil((workload + 1) / b)
                index = int(workload)
                slo_flag = True
                for j in range(num):
                    if index + 1 < (j + 1) * b:
                        wait = ((j + 1) * b - index + 1) / rate
                    else:
                        wait = 0
                    end_time = max([task_i[index].start_t + wait, time]) + inf
                    if end_time - task_i[j * b].start_t > SLO:  # 违反延迟上线
                        slo_flag = False
                        break
                    time = end_time
                if place_flag:
                    model_memory = get_memory(self.models[i].name, b, k)
                else:
                    model_memory = get_memory(self.models[i].name, b, k) - get_memory(self.models[i].name,
                                                                                         self.GPUs[k].batch[i], k)
                if slo_flag == False or gpu_memory[k] < model_memory or total_memory[k] < model_memory:
                    start = b - 1
                    if throughput < rate:
                        break
                else:
                    if throughput > rate:
                        batch_temp = b
                    end = b + 1
            if batch_temp != 0:
                batch.append(batch_temp)
                if place_flag:

                    model_memory = get_memory(self.models[i].name, batch_temp, k)
                else:
                    model_memory = get_memory(self.models[i].name, batch_temp, k) - get_memory(self.models[i].name,
                                                                                                  self.GPUs[k].batch[i],
                                                                                                  k)
                gpu_memory[k] -= model_memory  # get_memory(self.models[i].name, 1, k)
                total_memory[k] -= model_memory
            else:
                flag[i] = False
                batch.append(default)
                if place_flag:
                    model_memory = get_memory(self.models[i].name, default, k)
                else:
                    model_memory = get_memory(self.models[i].name, default, k) - get_memory(self.models[i].name,
                                                                                               self.GPUs[k].batch[i], k)
                gpu_memory[k] -=  model_memory # get_memory(self.models[i].name, 1, k)
                total_memory[k] -= model_memory
        return batch, flag


    def del_action_dis(self, action):
        place = []
        allo = []
        for i in range(self.model_num):
            place.append(action[i * 2])
            allo.append(action[2 * i + 1] + 1)
        # place = action[0:self.model_num]
        # allo = action[self.model_num:self.model_num * 2]

        sum = np.zeros(self.GPU_num)
        sum_index = [[] for i in range(self.GPU_num)]
        for i in range(self.model_num):
            sum[place[i]] += allo[i]
            sum_index[place[i]].append(i)

        for i in range(self.GPU_num):
            if sum[i] == 0: continue
            diff = sum[i] - self.GPUs[i].full_res / self.GPUs[i].sm
            while diff != 0:
                for j in sum_index[i]:
                    if diff == 0: break
                    if diff > 0:
                        if allo[j] > 1:
                            allo[j] -= 1
                            diff -= 1
                    if diff < 0:
                        allo[j] += 1
                        diff += 1
            for j in sum_index[i]:
                allo[j] = min(allo[j] * self.GPUs[i].sm, 100)

        return place, allo