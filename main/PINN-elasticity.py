from google.colab import drive
drive.mount('/content/drive')

try:
    import mymodule
except ImportError:
    !pip install pyDOE

%tensorflow_version 1.9
import numpy as np
import time
from pyDOE import lhs
import matplotlib
import matplotlib.pyplot as plt
import pickle
import scipy.io
import random
import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

class PINN_elasticity:
    # Initialize the class
    def __init__(self, lamd, mu, Q, Collo, DirichletBCxy, NeumannBCxy, NeumannBCn, NeumannBCt, bodyf, uv_layers, trained_model = 0, uvDir='./drive/MyDrive/PINN/nnwb.pickle'):

        # Count for callback function
        self.count=0

        # Mat. properties
        self.lamd = lamd
        self.mu = mu
        self.Q = Q

        # Collocation point
        self.x_c = Collo[:, 0:1]
        self.y_c = Collo[:, 1:2]

        self.x_bodyf = bodyf[:, 0:1]
        self.y_bodyf = bodyf[:, 1:2]

        self.x_Dirichlet = DirichletBCxy[:, 0:1]
        self.y_Dirichlet = DirichletBCxy[:, 1:2]

        self.x_Neumann = NeumannBCxy[:, 0:1]
        self.y_Neumann = NeumannBCxy[:, 1:2]

        self.Neumann_n1 = NeumannBCn[:, 0:1]
        self.Neumann_n2 = NeumannBCn[:, 1:2]

        self.Neumann_t1 = NeumannBCt[:, 0:1]
        self.Neumann_t2 = NeumannBCt[:, 1:2]

        # Define layers
        self.uv_layers = uv_layers

        self.loss_rec = []
        self.loss_Dir = []
        self.loss_Neu = []
        self.loss_gov = []

        # Initialize NNs
        if trained_model == 0 :
            self.uv_weights, self.uv_biases = self.initialize_NN(self.uv_layers)
        else:
            print("Loading trained NN model...")
            self.uv_weights, self.uv_biases = self.load_NN(uvDir, self.uv_layers)

        # tf placeholders
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])

        self.x_bodyf_tf = tf.placeholder(tf.float32, shape=[None, self.x_bodyf.shape[1]])
        self.y_bodyf_tf = tf.placeholder(tf.float32, shape=[None, self.x_bodyf.shape[1]])

        self.x_Dirichlet_tf = tf.placeholder(tf.float32, shape=[None, self.x_Dirichlet.shape[1]])
        self.y_Dirichlet_tf = tf.placeholder(tf.float32, shape=[None, self.y_Dirichlet.shape[1]])

        self.x_Neumann_tf = tf.placeholder(tf.float32, shape=[None, self.x_Neumann.shape[1]])
        self.y_Neumann_tf = tf.placeholder(tf.float32, shape=[None, self.y_Neumann.shape[1]])
        self.n1_Neumann_tf = tf.placeholder(tf.float32, shape=[None, self.Neumann_n1.shape[1]])
        self.n2_Neumann_tf = tf.placeholder(tf.float32, shape=[None, self.Neumann_n2.shape[1]])
        self.Neumannt1_tf = tf.placeholder(tf.float32, shape=[None, self.Neumann_t1.shape[1]])
        self.Neumannt2_tf = tf.placeholder(tf.float32, shape=[None, self.Neumann_t2.shape[1]])

        self.x_c_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_c_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])

        # tf graphs
        self.u_pred, self.v_pred, self.s11_pred, self.s22_pred, self.s12_pred, _, _ = self.net_uv(self.x_c_tf, self.y_c_tf)
        # for minimization
        self.f_s11r_pred, self.f_s22r_pred, self.f_s12r_pred, self.f_s1_pred, self.f_s2_pred = self.net_f(self.x_c_tf, self.y_c_tf)
        # for Dirichlet BC minimization
        self.u_Dirichlet_pred, self.v_Dirichlet_pred, _, _, _, _, _ = self.net_uv(self.x_Dirichlet_tf, self.y_Dirichlet_tf)
        # for Neumann BC minimization
        self.x_Neumann_pred, self.y_Neumann_pred, _, _, _, self.Neumannt1_pred, self.Neumannt2_pred = self.net_uv(self.x_Neumann_tf, self.y_Neumann_tf, self.n1_Neumann_tf, self.n2_Neumann_tf)

        self.loss_f = tf.reduce_mean(tf.square(self.f_s11r_pred)) \
                      + tf.reduce_mean(tf.square(self.f_s22r_pred))\
                      + tf.reduce_mean(tf.square(self.f_s12r_pred))\
                      + tf.reduce_mean(tf.square(self.x_bodyf_tf+self.f_s1_pred))\
                      + tf.reduce_mean(tf.square(self.y_bodyf_tf+self.f_s2_pred))

        self.loss_Dirichlet = tf.reduce_mean(tf.square(self.u_Dirichlet_pred)) \
                         + tf.reduce_mean(tf.square(self.v_Dirichlet_pred))

        self.loss_Neumann = tf.reduce_mean(tf.square(self.Neumannt1_tf-self.Neumannt1_pred))\
                         + tf.reduce_mean(tf.square(self.Neumannt2_tf-self.Neumannt2_pred))

        self.loss = self.loss_f + self.loss_Dirichlet + self.loss_Neumann

        # Optimizer train_bfgs
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                var_list=self.uv_weights + self.uv_biases,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 100000,
                                                                         'maxfun': 100000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol ': 2e-8*np.finfo(float).eps})
        
        # Optimizer train_Adam
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss,
                                                          var_list=self.uv_weights + self.uv_biases)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32)

    def save_NN(self, fileDir):

        uv_weights = self.sess.run(self.uv_weights)
        uv_biases = self.sess.run(self.uv_biases)

        with open(fileDir, 'wb') as f:
            pickle.dump([uv_weights, uv_biases], f)
            print(" - Save neural networks parameters on Google Drive successfully...")

    def load_NN(self, fileDir, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        with open(fileDir, 'rb') as f:
            uv_weights, uv_biases = pickle.load(f)

            # Stored model must has the same # of layers
            assert num_layers == (len(uv_weights)+1)

            for num in range(0, num_layers - 1):
                W = tf.Variable(uv_weights[num], dtype=tf.float32)
                b = tf.Variable(uv_biases[num], dtype=tf.float32)
                weights.append(W)
                biases.append(b)
                print(" - Load NN parameters on Google Drive successfully...")
        return weights, biases

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        # H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_uv(self, x, y, n1=0, n2=0):
        temp = self.neural_net(tf.concat([x, y], 1), self.uv_weights, self.uv_biases)
        u = temp[:,0:1]
        v = temp[:,1:2]
        s11 = temp[:, 2:3]
        s22 = temp[:, 3:4]
        s12 = temp[:, 4:5]

        # tractions
        t1 = tf.math.multiply(s11,n1) + tf.math.multiply(s12,n2)
        t2 = tf.math.multiply(s12,n1) + tf.math.multiply(s22,n2)

        return u, v, s11, s22, s12, t1, t2

    def net_f(self, x, y):

        lamd=self.lamd
        mu=self.mu
        Q=self.Q
        Pi=np.pi

        u, v, s11, s22, s12, _, _ = self.net_uv(x, y)

        s11_1 = tf.gradients(s11, x)[0]
        s12_2 = tf.gradients(s12, y)[0]
        s22_2 = tf.gradients(s22, y)[0]
        s12_1 = tf.gradients(s12, x)[0]

        # Plane stress problem
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]

        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]

        e_xx = u_x
        e_yy = v_y
        e_xy = (u_y + v_x)/2.0
        e_kk = e_xx + e_yy

        f_s11r = lamd*e_kk + 2*mu*e_xx - s11
        f_s22r = lamd*e_kk + 2*mu*e_yy - s22
        f_s12r = 2*mu*e_xy - s12

        f_s1r = s11_1 + s12_2
        f_s2r = s12_1 + s22_2

        return f_s11r, f_s22r, f_s12r, f_s1r, f_s2r


    def callback(self, loss_gov, loss_Neu, loss_Dir, loss):
        self.count = self.count+1
        self.loss_Neu.append(loss_Neu)
        self.loss_Dir.append(loss_Dir)
        self.loss_gov.append(loss_gov)
        self.loss_rec.append(loss)

        if self.count % 100 == 0:
            print('{} th iterations, loss_f: {}, loss_Neumann: {}, loss_Dirichlet: {}, loss_total: {}'.format(self.count, loss_gov, loss_Neu, loss_Dir, loss ))

    def train(self, iter, learning_rate):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
                   self.x_bodyf_tf: self.x_bodyf, self.y_bodyf_tf: self.y_bodyf,
                   self.x_Neumann_tf: self.x_Neumann, self.y_Neumann_tf: self.y_Neumann,
                   self.Neumannt1_tf: self.Neumann_t1, self.Neumannt2_tf: self.Neumann_t2,
                   self.n1_Neumann_tf: self.Neumann_n1, self.n2_Neumann_tf: self.Neumann_n2,
                   self.x_Dirichlet_tf: self.x_Dirichlet, self.y_Dirichlet_tf: self.y_Dirichlet,
                   self.learning_rate: learning_rate}

        for it in range(iter):

            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 100 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_valueF = self.sess.run(self.loss_f, tf_dict)
                loss_valueN = self.sess.run(self.loss_Neumann, tf_dict)
                loss_valueD = self.sess.run(self.loss_Dirichlet, tf_dict)

                print('It: %d, Loss: %.3e, LossF: %.3e, LossN: %.3e, LossD: %.3e' %
                      (it, loss_value, loss_valueF, loss_valueN, loss_valueD))
 
            self.loss_Dir.append(self.sess.run(self.loss_Dirichlet, tf_dict))
            self.loss_Neu.append(self.sess.run(self.loss_Neumann, tf_dict))
            self.loss_gov.append(self.sess.run(self.loss_f, tf_dict))
            self.loss_rec.append(self.sess.run(self.loss, tf_dict))

        return self.loss_f, self.loss_Dirichlet, self.loss_Neumann, self.loss

    def train_bfgs(self):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
                   self.x_bodyf_tf: self.x_bodyf, self.y_bodyf_tf: self.y_bodyf,
                   self.x_Neumann_tf: self.x_Neumann, self.y_Neumann_tf: self.y_Neumann,
                   self.Neumannt1_tf: self.Neumann_t1, self.Neumannt2_tf: self.Neumann_t2,
                   self.n1_Neumann_tf: self.Neumann_n1, self.n2_Neumann_tf: self.Neumann_n2,
                   self.x_Dirichlet_tf: self.x_Dirichlet, self.y_Dirichlet_tf: self.y_Dirichlet}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss_f, self.loss_Neumann, self.loss_Dirichlet, self.loss],
                                loss_callback=self.callback)

    def predict(self, x_star, y_star):
        u_star = self.sess.run(self.u_pred, {self.x_c_tf: x_star, self.y_c_tf: y_star})
        v_star = self.sess.run(self.v_pred, {self.x_c_tf: x_star, self.y_c_tf: y_star})

        s11_star = self.sess.run(self.s11_pred, {self.x_c_tf: x_star, self.y_c_tf: y_star})
        s22_star = self.sess.run(self.s22_pred, {self.x_c_tf: x_star, self.y_c_tf: y_star})
        s12_star = self.sess.run(self.s12_pred, {self.x_c_tf: x_star, self.y_c_tf: y_star})
        return u_star, v_star, s11_star, s22_star, s12_star

    def plot_loss(self,savepath="./drive/MyDrive/PINN"):

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        line1, = plt.plot(self.loss_rec, label="Total loss", linestyle='-')
        line2, = plt.plot(self.loss_Dir, label="Dirichlet", linestyle='-.')
        line3, = plt.plot(self.loss_Neu, label="Neumman", linestyle=':')
        line4, = plt.plot(self.loss_gov, label="Governing equation", linestyle='--')
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.legend()
        plt.savefig(savepath+'/loss.png', dpi=300)
        plt.show()  

def bodyf(lamd, mu, Q, x, y):
    # body force
    Pi = np.pi
    b1=lamd*(4*Pi**2*np.multiply(np.cos(2*Pi*x),np.sin(Pi*y))-Q*Pi*np.multiply(np.cos(Pi*x),np.power(y,3)))\
        +mu*(9.*Pi**2*np.multiply(np.cos(2*Pi*x),np.sin(Pi*y))-Q*Pi*np.multiply(np.cos(Pi*x),np.power(y,3)))
    b2=lamd*(-3*Q*np.multiply(np.sin(Pi*x),np.power(y,2))+2*Pi**2.*np.multiply(np.sin(2*Pi*x),np.cos(Pi*y)))\
        +mu*(-6.*Q*np.multiply(np.sin(Pi*x),np.power(y,2))+2*Pi**2.*np.multiply(np.sin(2*Pi*x),np.cos(Pi*y))\
        +Q*Pi**2.*np.multiply(np.sin(Pi*x),np.power(y,4))/4.)
    return b1, b2

def analytical_soln(lamd, mu, Q, xx, yy):
    # analytical solution
    Pi = np.pi
    u = np.multiply(np.cos(2*Pi*xx),np.sin(Pi*yy))
    v = Q*np.multiply(np.sin(Pi*xx),np.power(yy,4))/4
    sxx = lamd*(Q*np.multiply(np.sin(np.pi*xx),np.power(yy,3))-2*Pi*np.multiply(np.sin(2*Pi*xx),np.sin(Pi*yy)))\
        -4*mu*Pi*np.multiply(np.sin(2*Pi*xx),np.sin(Pi*yy))
    syy = lamd*(Q*np.multiply(np.sin(np.pi*xx),np.power(yy,3))-2*Pi*np.multiply(np.sin(2*Pi*xx),np.sin(Pi*yy)))\
        +2*mu*Q*np.multiply(np.sin(Pi*xx),np.power(yy,3))
    sxy = mu*(np.multiply(np.cos(np.pi*xx),np.power(yy,4))*Pi*Q/4+Pi*np.multiply(np.cos(2*Pi*xx),np.cos(Pi*yy)))

    return u, v, sxx, syy, sxy

def postProcess_field( field_anal, field_PINN, marksize = 2, alpha=0.8, marker='o',savepath="./drive/MyDrive/PINN"):

    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 1
    [x_anal, y_anal, u_anal, v_anal, sxx_anal, syy_anal, sxy_anal ] = field_anal
    [x_PINN, y_PINN, u_PINN, v_PINN, sxx_PINN, syy_PINN, sxy_PINN ] = field_PINN
    normuv = matplotlib.colors.Normalize(vmin=-0.8, vmax=0.8)
    normsigma = matplotlib.colors.Normalize(vmin=-10, vmax=10)

    # plot PINN results
    fig11, ax11 = plt.subplots()
    ax11.set_aspect('equal')
    cp = ax11.scatter(x_PINN, y_PINN, c=u_PINN, alpha=alpha-0.1, edgecolors='none', cmap='rainbow', marker=marker, s=int(marksize), vmin=-0.8, vmax=0.8)   
    ax11.set_xticks([])
    ax11.set_yticks([])
    ax11.set_xlim([xmin, xmax])
    ax11.set_ylim([ymin, ymax])
    ax11.set_xlabel("x (m)")
    ax11.set_ylabel("y (m)")
    plt.title('PINN u $(m)$')
    fig11.colorbar(cp)
    plt.savefig(savepath+'/u_PINN.png', dpi=300)
    plt.show()

    fig21, ax21 = plt.subplots()
    ax21.set_aspect('equal')
    cp = ax21.scatter(x_PINN, y_PINN, c=v_PINN, alpha=alpha-0.1, edgecolors='none', cmap='rainbow', marker=marker, s=int(marksize), vmin=0, vmax=0.8)   
    ax21.set_xticks([])
    ax21.set_yticks([])
    ax21.set_xlim([xmin, xmax])
    ax21.set_ylim([ymin, ymax])
    ax21.set_xlabel("x (m)")
    ax21.set_ylabel("y (m)")
    plt.title('PINN v $(m)$')
    fig21.colorbar(cp)
    plt.savefig(savepath+'/v_PINN.png', dpi=300)
    plt.show()

    fig31, ax31 = plt.subplots()
    ax31.set_aspect('equal')
    cp = ax31.scatter(x_PINN, y_PINN, c=sxx_PINN, alpha=alpha-0.1, edgecolors='none', cmap='rainbow', marker=marker, s=int(marksize), vmin=-10, vmax=10)   
    ax31.set_xticks([])
    ax31.set_yticks([])
    ax31.set_xlim([xmin, xmax])
    ax31.set_ylim([ymin, ymax])
    ax31.set_xlabel("x (m)")
    ax31.set_ylabel("y (m)")
    plt.title('PINN $\sigma_{xx}\hspace{0.2}(N/m^2$)')
    fig31.colorbar(cp)
    plt.savefig(savepath+'/sxx_PINN.png', dpi=300)
    plt.show()

    fig41, ax41 = plt.subplots()
    ax41.set_aspect('equal')
    cp = ax41.scatter(x_PINN, y_PINN, c=syy_PINN, alpha=alpha-0.1, edgecolors='none', cmap='rainbow', marker=marker, s=int(marksize), vmin=-6, vmax=8)   
    ax41.set_xticks([])
    ax41.set_yticks([])
    ax41.set_xlim([xmin, xmax])
    ax41.set_ylim([ymin, ymax])
    ax41.set_xlabel("x (m)")
    ax41.set_ylabel("y (m)")
    plt.title('PINN $\sigma_{yy}\hspace{0.2}(N/m^2$)')
    fig41.colorbar(cp)
    plt.savefig(savepath+'/syy_PINN.png', dpi=300)
    plt.show()

    fig51, ax51 = plt.subplots()
    ax51.set_aspect('equal')
    cp = ax51.scatter(x_PINN, y_PINN, c=sxy_PINN, alpha=alpha-0.1, edgecolors='none', cmap='rainbow', marker=marker, s=int(marksize), vmin=-3, vmax=3)   
    ax51.set_xticks([])
    ax51.set_yticks([])
    ax51.set_xlim([xmin, xmax])
    ax51.set_ylim([ymin, ymax])
    ax51.set_xlabel("x (m)")
    ax51.set_ylabel("y (m)")
    plt.title('PINN $\sigma_{xy}\hspace{0.2}(N/m^2$)')
    fig51.colorbar(cp)
    plt.savefig(savepath+'/sxy_PINN.png', dpi=300)
    plt.show()

    # plot analytical results
    fig12, ax12 = plt.subplots()
    ax12.set_aspect('equal')
    cp = ax12.scatter(x_anal, y_anal, c=u_anal, alpha=alpha-0.1, edgecolors='none', cmap='rainbow', marker=marker, s=int(marksize), vmin=-0.8, vmax=0.8)   
    ax12.set_xticks([])
    ax12.set_yticks([])
    ax12.set_xlim([xmin, xmax])
    ax12.set_ylim([ymin, ymax])
    ax12.set_xlabel("x (m)")
    ax12.set_ylabel("y (m)")
    plt.title('Analytical u $(m)$')
    fig12.colorbar(cp)
    plt.savefig(savepath+'/u_anal.png', dpi=300)
    plt.show()

    fig22, ax22 = plt.subplots()
    ax22.set_aspect('equal')
    cp = ax22.scatter(x_anal, y_anal, c=v_anal, alpha=alpha-0.1, edgecolors='none', cmap='rainbow', marker=marker, s=int(marksize), vmin=0, vmax=0.8)   
    ax22.set_xticks([])
    ax22.set_yticks([])
    ax22.set_xlim([xmin, xmax])
    ax22.set_ylim([ymin, ymax])
    ax22.set_xlabel("x (m)")
    ax22.set_ylabel("y (m)")
    plt.title('Analytical v $(m)$')
    fig22.colorbar(cp)
    plt.savefig(savepath+'/v_anal.png', dpi=300)
    plt.show()

    fig32, ax32 = plt.subplots()
    ax32.set_aspect('equal')
    cp = ax32.scatter(x_anal, y_anal, c=sxx_anal, alpha=alpha-0.1, edgecolors='none', cmap='rainbow', marker=marker, s=int(marksize), vmin=-10, vmax=10)   
    ax32.set_xticks([])
    ax32.set_yticks([])
    ax32.set_xlim([xmin, xmax])
    ax32.set_ylim([ymin, ymax])
    ax32.set_xlabel("x (m)")
    ax32.set_ylabel("y (m)")
    plt.title('Analytical $\sigma_{xx}\hspace{0.2}(N/m^2$)')
    fig32.colorbar(cp)
    plt.savefig(savepath+'/sxx_anal.png', dpi=300)
    plt.show()

    fig42, ax42 = plt.subplots()
    ax42.set_aspect('equal')
    cp = ax42.scatter(x_anal, y_anal, c=syy_anal, alpha=alpha-0.1, edgecolors='none', cmap='rainbow', marker=marker, s=int(marksize), vmin=-6, vmax=8)   
    ax42.set_xticks([])
    ax42.set_yticks([])
    ax42.set_xlim([xmin, xmax])
    ax42.set_ylim([ymin, ymax])
    ax42.set_xlabel("x (m)")
    ax42.set_ylabel("y (m)")
    plt.title('Analytical $\sigma_{yy}\hspace{0.2}(N/m^2$)')
    fig42.colorbar(cp)
    plt.savefig(savepath+'/syy_anal.png', dpi=300)
    plt.show()

    fig52, ax52 = plt.subplots()
    ax52.set_aspect('equal')
    cp = ax52.scatter(x_anal, y_anal, c=sxy_anal, alpha=alpha-0.1, edgecolors='none', cmap='rainbow', marker=marker, s=int(marksize), vmin=-3, vmax=3)   
    ax52.set_xticks([])
    ax52.set_yticks([])
    ax52.set_xlim([xmin, xmax])
    ax52.set_ylim([ymin, ymax])
    ax52.set_xlabel("x (m)")
    ax52.set_ylabel("y (m)")
    plt.title('Analytical $\sigma_{xy}\hspace{0.2}(N/m^2$)')
    fig52.colorbar(cp)
    plt.savefig(savepath+'/sxy_anal.png', dpi=300)
    plt.show()

    plt.close('all')

if __name__ == "__main__":

    lamd = 1
    mu = 0.5
    Q = 4

    # Network configuration
    # output u,v,sigxx,sigyy,sigxy
    uv_layers = [2] + 10*[40] + [5]

    # number of domain training samples
    num_dom_train_samples = 200
    # number of boundary training samples
    num_b_train_samples = 300  

    # collocation points in the domain
    XY_c1 = np.linspace(0.0, 1.0, num=num_dom_train_samples)               # x = 0 ~ +1
    XY_c2 = np.linspace(0.0, 1.0, num=num_dom_train_samples)               # y = 0 ~ +1
    XY_c1, XY_c2 = np.meshgrid(XY_c1, XY_c2)
    XY_c = np.vstack((XY_c1.flatten(),XY_c2.flatten())).T

    # Neumman BC
    xy_left = np.zeros((num_b_train_samples, 2))
    xy_left[..., 0] = np.zeros(num_b_train_samples)                               # x = 0
    xy_left[..., 1] = np.linspace(0.0, 1.0, num=num_b_train_samples)              # y = 0 ~ +1
    n_left = np.zeros((num_b_train_samples, 2))
    n_left[..., 0] = -1.*np.ones(num_b_train_samples)
    n_left[..., 1] = np.zeros(num_b_train_samples)
    t_left = np.zeros((num_b_train_samples, 2))
    t_left[..., 0] = np.zeros(num_b_train_samples)
    t_left[..., 1] = -mu*np.pi*(np.cos(np.pi*xy_left[..., 1])+np.power(xy_left[..., 1],4)*Q/4)

    xy_top = np.zeros((num_b_train_samples, 2))
    xy_top[..., 0] = np.linspace(0.0, 1.0, num=num_b_train_samples)                # x = 0 ~ +1
    xy_top[..., 1] = np.ones(num_b_train_samples)                                  # y = 0 
    n_top = np.zeros((num_b_train_samples, 2))
    n_top[..., 0] = np.zeros(num_b_train_samples)
    n_top[..., 1] = np.ones(num_b_train_samples)
    t_top = np.zeros((num_b_train_samples, 2))
    t_top[..., 0] = mu*np.pi*(-np.cos(2*np.pi*xy_top[..., 0])+np.cos(np.pi*xy_top[..., 0])*Q/4)
    t_top[..., 1] = (lamd+2*mu)*Q*np.sin(np.pi*xy_top[..., 0])

    xy_right = np.zeros((num_b_train_samples, 2))
    xy_right[..., 0] = np.ones(num_b_train_samples)                               # x = 1
    xy_right[..., 1] = np.linspace(0.0, 1.0, num=num_b_train_samples)             # y = 0 ~ +1
    n_right = np.zeros((num_b_train_samples, 2))
    n_right[..., 0] = np.ones(num_b_train_samples)
    n_right[..., 1] = np.zeros(num_b_train_samples)
    t_right = np.zeros((num_b_train_samples, 2))
    t_right[..., 0] = np.zeros(num_b_train_samples)
    t_right[..., 1] = mu*np.pi*(np.cos(np.pi*xy_right[..., 1])-np.power(xy_right[..., 1],4)*Q/4)

    xy_NeumannBC = np.concatenate((xy_left, xy_top, xy_right), 0)
    n_NeumannBC = np.concatenate((n_left, n_top, n_right), 0)
    t_NeumannBC = np.concatenate((t_left, t_top, t_right), 0)

    # Dirichelet BC
    xy_bottom = np.zeros((num_b_train_samples, 2))
    xy_bottom[..., 0] = np.linspace(0.0, 1.0, num=num_b_train_samples)          # x = 0 ~ +1
    xy_bottom[..., 1] = np.zeros(num_b_train_samples)                           # y = 1 
    xy_DirichletBC = xy_bottom

    XY_c = np.concatenate((XY_c, xy_NeumannBC, xy_DirichletBC), 0)

    bf = np.zeros(XY_c.shape)
    bf[..., 0], bf[..., 1] = bodyf(lamd, mu, Q, XY_c[..., 0], XY_c[..., 1])

    # Visualize the collocation points
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.scatter(XY_c[:,0:1], XY_c[:,1:2], marker='o', alpha=0.1 ,color='blue')
    plt.scatter(xy_NeumannBC[:,0:1], xy_NeumannBC[:,1:2], marker='o', alpha=0.2 , color='black')
    plt.scatter(xy_DirichletBC[:, 0:1], xy_DirichletBC[:, 1:2], marker='o', alpha=0.2, color='orange')
    plt.show()

    with tf.device('/device:GPU:0'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        # Train nn model or load trained model
        model = PINN_elasticity(lamd, mu, Q, XY_c, xy_DirichletBC, xy_NeumannBC, n_NeumannBC, t_NeumannBC, bf, uv_layers)

        start_time = time.time()
        loss_f, loss_Dirichlet, loss_Neumann, loss = model.train(iter=10000, learning_rate=1e-3)
        # postProcess_field( field_anal, field_PINN, marksize=2, alpha=0.8, marker='o',savepath="./drive/MyDrive/PINN")

        # Save neural network
        model.save_NN('./drive/MyDrive/PINN/NN_Adam.pickle')

        num_PINN = 200
        x_PINN = np.linspace(0, 1, num_PINN)
        y_PINN = np.linspace(0, 1, num_PINN)
        x_PINN, y_PINN = np.meshgrid(x_PINN, y_PINN)
        x_PINN = x_PINN.flatten()[:, None]
        y_PINN = y_PINN.flatten()[:, None]
        u_PINN, v_PINN, s11_PINN, s22_PINN, s12_PINN = model.predict(x_PINN, y_PINN)
        field_PINN = [x_PINN, y_PINN, u_PINN, v_PINN, s11_PINN, s22_PINN, s12_PINN]

        u_anal, v_anal, sigxx_anal, sigyy_anal, sigxy_anal = analytical_soln(lamd, mu, Q, x_PINN, y_PINN)
        field_anal = [x_PINN, y_PINN, u_anal, v_anal, sigxx_anal, sigyy_anal, sigxy_anal]

        model.train_bfgs()
        u_PINN, v_PINN, s11_PINN, s22_PINN, s12_PINN = model.predict(x_PINN, y_PINN)
        field_PINN = [x_PINN, y_PINN, u_PINN, v_PINN, s11_PINN, s22_PINN, s12_PINN]
        # Save neural network after bfgs training
        model.save_NN('./drive/MyDrive/PINN/NN_Adam_bfgs.pickle')
        print("--- %s seconds ---" % (time.time() - start_time))

        model.plot_loss()

        postProcess_field( field_anal, field_PINN, marksize=2, alpha=0.8, marker='o',savepath="./drive/MyDrive/PINN")

        # Save loss history
        with open('./drive/MyDrive/PINN/loss_history.pickle', 'wb') as f:
            pickle.dump([model.loss_rec, model.loss_Dir, model.loss_Neu, model.loss_gov], f)