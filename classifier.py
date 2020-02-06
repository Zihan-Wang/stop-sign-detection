import numpy as np
import time

class Trainer():
    def __init__(self):
        self.w = np.zeros((8, 1)) # weights for different feature
        self.b = 0
        self.X_red = np.load("red.npy") # red from red stop sign data
        self.X_fakered = np.load("fakered.npy") # not red from random data
        self.Y_red = np.zeros((1, self.X_red.shape[1]))
        print("Start preparing Y")
        start = time.time()
        for idx in range(self.X_red.shape[1]):
            if (self.X_red[0][idx] > (self.X_red[1][idx] + self.X_red[2][idx]) * 0.65) :
                self.Y_red[0][idx] = 1 # stands for red
        print("Finish Y Preprocess")
        end = time.time()
        print(f"Time cost {end - start}s")

        min_num = min(self.X_red.shape[1], self.X_fakered.shape[1]) # make sure the number matches for both true and false

        self.Y_fakered = np.zeros((1, self.X_fakered.shape[1])) # default to be zero
        self.X_tmp = np.concatenate((self.X_red[:, :min_num], self.X_fakered[:, :min_num]), axis=1)
        self.Y = np.concatenate((self.Y_red[:, :min_num], self.Y_fakered[:, :min_num]), axis=1)
        self.X_tmp = self.X_tmp / 255.
        print("Start preparing 8 dim X")
        start = time.time()
        self.X = np.ones((8,self.X_tmp.shape[1]))
        self.X[0] = self.X_tmp[0] # r
        self.X[1] = self.X_tmp[1] # g
        self.X[2] = self.X_tmp[2]  # b
        self.X[3] = np.multiply(self.X_tmp[0], self.X_tmp[0]) # r^2
        self.X[4] = np.multiply(self.X_tmp[1], self.X_tmp[1]) # g^2
        self.X[5] = np.multiply(self.X_tmp[2], self.X_tmp[2]) # b^2
        self.X[6] = np.multiply(self.X_tmp[0], self.X_tmp[1])  # rg
        self.X[7] = np.multiply(self.X_tmp[0], self.X_tmp[2])  # rb
        print("Finish 8 dim X preprocess")
        end = time.time()
        print(f"Time cost {end - start}s")
        self.params = {}
        print("X_red : {} ; X_fakered : {}".format(self.X_red.shape[1], self.X_fakered.shape[1]))
        print("X's shape " + str(self.X.shape[1]))
        print(f"{sum(self.Y_red[0])} / {self.Y_red.shape[1]}")

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def propagate(self, w, b, X, Y):
        sample_num = X.shape[1] # sample number

        # front propagate
        Y_head = self.sigmoid(np.dot(w.T, X) + b)
        cost = -(np.sum(Y * np.log(Y_head) + (1 - Y) * np.log(1 - Y_head))) / sample_num

        # back propagate
        dZ = Y_head - Y
        dw = (np.dot(X, dZ.T)) / sample_num
        db = (np.sum(dZ)) / sample_num

        grads = {"dw": dw, "db": db}

        return grads, cost

    def optimize(self, w, b, X, Y, num_iterations, learning_rate):

        for i in range(num_iterations):
            start = time.time()
            grads, cost = self.propagate(w, b, X, Y)
            dw = grads["dw"]
            db = grads["db"]

            w -= learning_rate * dw
            b -= learning_rate * db

            end = time.time()

            if i % 10 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
                print(f"Est Finish in {(end - start) * (num_iterations - i)}s")

        params = {"w": w, "b": b}
        return params

    def train(self):
        self.params = self.optimize(self.w, self.b, self.X, self.Y, 5000, 5.0, )
        print(self.params)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
