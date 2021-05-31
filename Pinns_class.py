"""
@author: Yufeng Wang
"""
import tensorflow as tf
import numpy as np
import time
import pickle


###############################################################################
######################## Define Pinns weq ############################
###############################################################################

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x_u, y_u, t_u, x_f, y_f, t_f, u, vp, layers, niter_bfgs, ExistModel=0, modelDir=''):
        self.count = 0
        X = np.concatenate([x_u, y_u, t_u], 1)       
        self.lb = X.min(0)
        self.ub = X.max(0)            
        self.X = X
        
        self.x_u = X[:,0:1]
        self.y_u = X[:,1:2]
        self.t_u = X[:,2:3]
        
        self.x_f = x_f
        self.y_f = y_f
        self.t_f = t_f

        self.u = u      
        self.vp = vp
        
        self.layers = layers

        # Iternum
        self.niter_bfgs = niter_bfgs
        
        # Initialize NN
        if ExistModel== 0 :
            self.weights, self.biases = self.initialize_NN(self.layers)
        else:
            self.weights, self.biases = self.load_NN(modelDir, self.layers)        
        
        # Initialize parameters
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)

        
        # tf placeholders and graph
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.y_u_tf = tf.placeholder(tf.float32, shape=[None, self.y_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])  

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.y_f_tf = tf.placeholder(tf.float32, shape=[None, self.y_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])          
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])


        self.u_pred = self.net_u(self.x_u_tf, self.y_u_tf, self.t_u_tf)
        self.f_u_pred = self.net_weq(self.x_f_tf, self.y_f_tf, self.t_f_tf)
        
        self.loss_data = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        self.loss_phy = tf.reduce_mean(tf.square(self.f_u_pred))
        # self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
        #             tf.reduce_mean(tf.square(self.f_u_pred))
        self.loss = self.loss_data + self.loss_phy

                    
        self.optimizer_phy = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                var_list=self.weights + self.biases,
                                                                options = {'maxiter': self.niter_bfgs,
                                                                           'maxfun': self.niter_bfgs,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'gtol': 1e-07,
                                                                           'ftol' : 0.001 * np.finfo(float).eps}) 

        self.optimizer_data = tf.contrib.opt.ScipyOptimizerInterface(self.loss_data, 
                                                                method = 'L-BFGS-B', 
                                                                var_list=self.weights + self.biases,
                                                                options = {'maxiter': self.niter_bfgs,
                                                                           'maxfun': self.niter_bfgs,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'gtol' : 1e-07,
                                                                           'ftol' : 0.001 * np.finfo(float).eps})         
        


        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss,
                                                            var_list=self.weights + self.biases)
        self.train_op_Adam_data = self.optimizer_Adam.minimize(self.loss_data,
                                                                var_list=self.weights + self.biases)      

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def save_NN(self, fileDir):
        weights = self.sess.run(self.weights)
        biases = self.sess.run(self.biases)
        with open(fileDir, 'wb') as f:
            # pickle.dump([np.array(uv_weights), np.array(uv_biases)], f)
            pickle.dump([weights, biases], f)
            print("Save NN parameters successfully...")

    def load_NN(self, fileDir, layers):
        weights_temp = []
        biases_temp = []
        num_layers = len(layers)
        with open(fileDir, 'rb') as f:
            weights, biases = pickle.load(f)
            # print(len(uv_weights))
            # print(np.shape(uv_weights))
            # print(num_layers)

            # Stored model must has the same # of layers
            assert num_layers == (len(weights)+1)

            for num in range(0, num_layers - 1):
                W = tf.Variable(weights[num])
                b = tf.Variable(biases[num])
                weights_temp.append(W)
                biases_temp.append(b)
                print("Load NN parameters successfully...")
        return weights_temp, biases_temp

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, y, t):
        u = self.neural_net(tf.concat([x,y,t],1), self.weights, self.biases)
        return u


    def net_weq(self, x, y, t):       

        u = self.net_u(x, y, t)     
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_tt = tf.gradients(u_t, t)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]        
        f_u = (u_xx + u_yy) - 1.0/(self.vp*self.vp)*u_tt

        return f_u
    
    def callback(self, loss, loss_data, loss_phy):
        self.count = self.count + 1
        print('Iter:%d L-BFGS-B training Loss: %.3e Loss_data: %.3e Loss_phy: %.3e' % (self.count, loss, loss_data, loss_phy))
        if self.count % 10 == 0:
            with open('history_loss_bfgs.txt', 'ab') as f:
                np.savetxt(f, np.atleast_1d(loss))
   
    # training with Adam optimizer   
    def train(self, iter, learning_rate, batch_num, whichloss): 
        loss_out = []
        for i in range(batch_num):
            col_num = self.x_f.shape[0]
            idx_start = int(i*col_num/batch_num)
            idx_end = int((i+1)*col_num/batch_num)
            tf_dict = {self.x_f_tf: self.x_f[idx_start:idx_end,:], self.y_f_tf: self.y_f[idx_start:idx_end,:], self.t_f_tf: self.t_f[idx_start:idx_end,:],
                        self.x_u_tf: self.x_u, self.y_u_tf: self.y_u, self.t_u_tf: self.t_u,
                        self.u_tf: self.u,
                        self.learning_rate: learning_rate}
        
            start_time = time.time()
            for it in range(iter):
                if whichloss == 0:
                    self.sess.run(self.train_op_Adam_data, tf_dict)
                else:
                    self.sess.run(self.train_op_Adam, tf_dict)
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    loss_data_value = self.sess.run(self.loss_data, tf_dict)
                    loss_phy_value = self.sess.run(self.loss_phy, tf_dict)
                    lambda_1_value = self.sess.run(self.lambda_1)
                    print('Adam training Iter: %d, Loss: %.3e, Loss_data: %.3e, Loss_phy: %.3e, Time: %.2f' % 
                        (i*iter+it, loss_value, loss_data_value, loss_phy_value, elapsed))
                    start_time = time.time()
                if it % 10 == 0:
                    loss_out.append(loss_value)

        return loss_out

    # training with L-BFGS-B optimizer            
    def train_bfgs(self, batch_num, whichloss):
        for i in range(batch_num):
            col_num = self.x_f.shape[0]
            idx_start = int(i*col_num/batch_num)
            idx_end = int((i+1)*col_num/batch_num)
            tf_dict = {self.x_f_tf: self.x_f[idx_start:idx_end,:], self.y_f_tf: self.y_f[idx_start:idx_end,:], self.t_f_tf: self.t_f[idx_start:idx_end,:],
                        self.x_u_tf: self.x_u, self.y_u_tf: self.y_u, self.t_u_tf: self.t_u,
                        self.u_tf: self.u}
            
            if whichloss == 0:
                self.optimizer_data.minimize(self.sess,
                                        feed_dict = tf_dict,
                                        fetches = [self.loss, self.loss_data, self.loss_phy],
                                        loss_callback = self.callback)
            else:
                self.optimizer_phy.minimize(self.sess,
                                        feed_dict = tf_dict,
                                        fetches = [self.loss, self.loss_data, self.loss_phy],
                                        loss_callback = self.callback)

    def predict(self, x_star, y_star, t_star):   
        tf_dict = {self.x_u_tf: x_star, self.y_u_tf: y_star, self.t_u_tf: t_star}
        u_pred = self.sess.run(self.u_pred, tf_dict)       
        return u_pred