import numpy as np
import copy

import os

class Layer:
    def __init__(self, in_size, out_size, activation = "relu",
                 initialization = "default"):
        size = (in_size, out_size)
        if initialization == "default":
            self.W = np.random.normal(scale=1, size=size) * np.sqrt(2 / (in_size + out_size))
        elif initialization == "zeros":
            self.W = np.zeros(size)
            
        self.b = np.zeros(out_size)

        self.change_func(activation)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)


    def forward(self, X):
        X = np.dot(X, self.W) + self.b
        X = self.f(X)
        return X


    def mutate(self, power, momentum = 0.9):
        self.dW *= momentum
        self.dW += (1 - momentum) * np.random.normal(0, power, self.W.shape)
        self.W += self.dW

        self.db *= momentum
        self.db += (1 - momentum) * np.random.normal(0, power, self.b.shape)
        self.b += self.db


    def change_func(self, activation):
        if activation == "sigmoid":
            self.f = self.sigmoid
        elif activation == "tanh":
            self.f = self.tanh
        elif activation == "relu":
            self.f = self.relu
        elif activation == "linear":
            self.f = self.linear

    
    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))


    @staticmethod
    def tanh(X):
        exp_x = np.exp(X)
        exp_mx = np.exp(-X)
        return (exp_x - exp_mx)/(exp_x + exp_mx)

    @staticmethod
    def relu(X):
        return X.clip(0, np.inf)

    @staticmethod
    def linear(X):
        return X


##################################################
class Network:
    def __init__(self, layer_sizes, start_power = 0.2,
                 mutation_power = None, initialization = "default",
                 last_act = "linear"):
        if isinstance(layer_sizes, str):
            with open("saved_models/" + layer_sizes + "/config.txt", "r") as f:
                load_f = layer_sizes
                layer_sizes = list(map(int, f.read().rstrip().split(" ")))
        else:
            load_f = None
            
        self.layer_sizes = list(layer_sizes)
        
        if mutation_power is None:
            self.mutation_pow = abs(np.random.normal(0, start_power))
        else:
            self.mutation_pow = mutation_power

        self.arch_mutation_pow = self.mutation_pow
            
        self.brain = []
        for i in range(1, len(layer_sizes)):
            layer = Layer(layer_sizes[i - 1], layer_sizes[i],
                          initialization = initialization)
            self.brain.append(layer)

        if load_f != None:
            self.load_weights(load_f)
            
        self.brain[-1].change_func(last_act)

    def forward(self, X):
        for layer in self.brain:
            X = layer.forward(X)
        return X


    def mutation(self, extreme_changes_rare = 50):
        self.change_weights()
        self.change_mutation_pow()
        rand1 = np.random.random()
        rand2 = np.random.random()
        
        if rand1 < self.arch_mutation_pow:
            self.add_random_neuron()

        if rand2 < self.arch_mutation_pow:
            self.delete_random_neuron()

        if rand1 * extreme_changes_rare < self.arch_mutation_pow:
            self.add_layer()

        if rand2 * extreme_changes_rare < self.arch_mutation_pow:
            self.delete_layer()
        

    def change_weights(self):
        for layer in self.brain:
            layer.mutate(self.mutation_pow)


    def change_mutation_pow(self, very_low = 0.005, more = 0.1):
            self.mutation_pow *= abs(np.random.normal(1, 0.1))
            self.arch_mutation_pow *= abs(np.random.normal(1, 0.1))

            if self.mutation_pow < very_low:
                self.mutation_pow = more * np.random.random()


    def add_random_neuron(self):
        l_count = len(self.brain)
        if l_count > 1:
            random_index = np.random.randint(0, l_count - 1)

            self.layer_sizes[random_index + 1] += 1

            layer1 = self.brain[random_index]
            layer2 = self.brain[random_index + 1]

            layer1.W = np.append(layer1.W,
                                 np.zeros((layer1.W.shape[0], 1)),
                                 axis = 1)
            layer1.b = np.append(layer1.b, 0)

            layer2.W = np.append(layer2.W,
                                 np.zeros((1, layer2.W.shape[1])),
                                 axis = 0)

            layer1.dW = np.append(layer1.dW,
                                 np.zeros((layer1.dW.shape[0], 1)),
                                 axis = 1)
            layer1.db = np.append(layer1.db, 0)

            layer2.dW = np.append(layer2.dW,
                                 np.zeros((1, layer2.dW.shape[1])),
                                 axis = 0)


    def add_layer(self, index = None):
        if index is None:
            index = np.random.randint(1, len(self.brain) + 1)

        input_neurons = self.layer_sizes[index - 1]
        output_neurons = self.layer_sizes[index]

        num_neurons = np.random.randint(10, 50)
        self.layer_sizes.insert(index, num_neurons)

        self.brain.pop(index - 1)
        self.brain.insert(index - 1, Layer(input_neurons, num_neurons))
        self.brain.insert(index, Layer(num_neurons, output_neurons))


    def delete_random_neuron(self):
        l_count = len(self.brain)
        if l_count > 1:
            random_index = np.random.randint(0, l_count - 1)
            random_neuron = np.random.randint(0,self.layer_sizes[random_index + 1])
            
            self.layer_sizes[random_index + 1] -= 1
            if self.layer_sizes[random_index + 1] == 0:
                self.delete_layer(index = random_index)
                return None
            
            layer1 = self.brain[random_index]
            layer2 = self.brain[random_index + 1]

            layer1.W = np.delete(layer1.W, random_neuron, 1)
            layer1.b = np.delete(layer1.b, random_neuron, 0)

            layer2.W = np.delete(layer2.W, random_neuron, 0)

            layer1.dW = np.delete(layer1.dW, random_neuron, 1)
            layer1.db = np.delete(layer1.db, random_neuron, 0)

            layer2.dW = np.delete(layer2.dW, random_neuron, 0)

            
    def delete_layer(self, index = None):
        if len(self.layer_sizes) == 2:
            return None
        
        if index is None:
            index = np.random.randint(0, len(self.brain) - 1)

        self.layer_sizes.pop(index + 1)
        input_neurons = self.layer_sizes[index]
        output_neurons= self.layer_sizes[index + 1]
        self.brain.pop(index)
        self.brain[index] = Layer(input_neurons, output_neurons)


    def param_size(self):
        summ = 0
        for layer in self.brain:
            summ += layer.W.size
            summ += layer.b.size
        return summ


    def get_weights(self):
        W = np.array([])
        for layer in self.brain:
            W = np.append(W, layer.W)
            W = np.append(W, layer.b)
        return W

    def abs_mean_weights(self):
        return abs(self.get_weights()).mean()

    def summary(self):
        print("total: {} parameters\nlayers: {}".format(self.param_size(),
                                                        self.layer_sizes))
        print("mutation_pow: {}".format(self.mutation_pow))
        print("arch_mutation_pow: {}".format(self.arch_mutation_pow))
        print("abs mean weights: {}".format(self.abs_mean_weights()))

    def save_weights(self, name = "model"):
        if not os.path.exists("saved_models"):
            os.mkdir("saved_models")
        if not os.path.exists("saved_models/" + name):
            os.mkdir("saved_models/" + name)
        with open("saved_models/" + name + "/config.txt", "w") as f:
            print(*self.layer_sizes, file = f)
        for ind, layer in enumerate(self.brain):
            np.save("saved_models/{}/layer_{}_W".format(name, ind + 1), layer.W)
            np.save("saved_models/{}/layer_{}_b".format(name, ind + 1), layer.b)

    def load_weights(self, name = "model"):
        for ind in range(len(self.brain)):
            self.brain[ind].W = np.load("saved_models/{}/layer_{}_W.npy".format(name,ind + 1))
            self.brain[ind].b = np.load("saved_models/{}/layer_{}_b.npy".format(name,ind + 1))

        

class Population:
    def __init__(self, count, hidden, model = None, input_size = 1, output_size = 1):
        sizes = [input_size]
        for l in hidden:
            sizes.append(l)
        sizes.append(output_size)
        
        self.P = []
        if model is not None:
            for _ in range(count):
                self.P.append(copy.deepcopy(model))
            count = 0
            
        for _ in range(count):
            self.P.append(Network(sizes, mutation_power = 0.1))

    def select_best(self, loss_func, X, Y):
        self.P.sort(key = lambda obj: loss_func(obj, X, Y))
        return self.P[0]

    def kill_the_weak(self, share = 0.5):
        self.to_kill = int(len(self.P) * share)
        for _ in range(self.to_kill):
            self.P.pop()

    def reproduction(self):
        for i in range(self.to_kill):
            self.P.append(copy.deepcopy(self.P[i]))

    def mutate(self, p = 0.5):
        for u in self.P:
            if np.random.random() < p:
                u.mutation()

                
def loss_func(obj, X, Y):
    Y_pred = obj.forward(X)
    loss = (abs(Y_pred - Y)).mean()
    loss += sum(regularization(obj))
    return loss

def regularization(obj, L2 = 0.01, param_L = 0.00001):
    L2_loss = L2 * obj.abs_mean_weights() ** 2
    n_param_loss = param_L * obj.param_size()
    return L2_loss, n_param_loss

    
def Learn(X, Y, model = None, epochs = 1000, batch_size = 64,
          hidden = [50], count = 128, kill_share = 0.9,
          mutate_p = 0.8, verboze = 200):
    p = Population(count, hidden, model, input_size = X.shape[-1],
                   output_size = Y.shape[-1])
    if len(X) < batch_size:
        batch_size = len(X)
    for epoch in range(epochs):
        RandBatch = np.random.randint(0, len(X), batch_size)
        best = p.select_best(loss_func, X[RandBatch], Y[RandBatch])
        p.kill_the_weak(share = kill_share)
        p.reproduction()
        if epoch % verboze == verboze - 1:
            print("loss = {:.4f}".format(loss_func(best, X[RandBatch], Y[RandBatch])),
                  end = " ")
            print("(L2_loss:{:.4f}, param_loss: {:.4f})".format(*regularization(best)))
            #best.summary()
            
        p.mutate(p = mutate_p)

    return best
    
