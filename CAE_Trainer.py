#exec(open("E:/Dropbox/Research/ManifoldLearning/Patch_AE_Trainer.py").read())

#exec(open("C:/Users/sscho/Dropbox/Research/ManifoldLearning/Patch_AE_Trainer.py").read())

computer = 2 #1: Laptop, 2: RPI

import tensorflow as tf
#import tensorflow.python.debug as tf_debug
import numpy as np
import trimesh as tm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import listdir
import scipy.io as spio
from scipy.stats import special_ortho_group
import sklearn
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import sys
import shelve
if computer == 1:
    sys.path.insert(0, "C:/Users/sscho/Dropbox/Research/ManifoldLearning/")
    base_path = "C:/Users/sscho/Dropbox/Research/ManifoldLearning/"
if computer == 2:
    sys.path.insert(0, "E:/Dropbox/Research/ManifoldLearning/")
    base_path = "E:/Dropbox/Research/ManifoldLearning/"
import Chart_Model_4
#import PatchandField_Model_4
#import PatchConfidence_Model_4
#import PatchLearnedConfidence_Model_4
import Chart_Model_10
#import ChartConvolution_Model_10
import Standard_Model 

import datetime

## Save Parameters
sname = 'MNISTConv2'
debug = 0

## Model Parameters
model_type = 2  #1:Auto Encoder, 2:Patch Enocoder_4, 3:Patch_Model_10, 4:Unit_Model_10, 
                 #5:PatchandField, 6:PatchConfidence_Model_4, 7:PatchLearnedConfindence. 8:Convolutional
patch_dim  = 25 #dimension of patches
n_latent   = 2   #pre-patch latent dim size
n_hidden   = 250  #units in hidden layers
reg_lam    = 10000 #regularization parameter

## Data paramters
data_type = 3 #1:infinte sphere, 2:discrete sphere, 3:'8'.obj, 4:kitten, 
              #5:MNIST, 6:Dancer, 7:genus 3, 8:Circles Big, 9:Circles Small
n_pts     = 10000 #number of pts in sample

## Training and Vis Paramters
FPS         = 1 #FPS sampling for initial points
batch_size  = 500  #batch during training
n_train     = 500000 #training iterations (of size batch_size)
alpha       = 3e-4 #learning rate
noise       = 0 #noise level
n_sample    = 5000  #samples in reconstruction test

initial_fit = 1#on/off intitial overfit each patch
n_initial   = 15000 #steps for initilization (per patch, no batches)
report      = 100 #how often to input training


###### Utilities ##############################################################################

def sample_sphere(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return np.transpose(vec)
    
def load_obj_pts(data_type):
    if data_type == 2:
        mesh = tm.load( base_path + "/sphere2k.off")
        return 100*mesh.vertices, 3
    if data_type == 3:
        mat = spio.loadmat(base_path + 'eight.mat', squeeze_me=True)
        return 100*mat['pts'],3 
    if data_type == 4:
        mesh = tm.load(base_path + "kitten.off")
        return mesh.vertices, 3 
    if data_type == 5:
        mnist = tf.keras.datasets.mnist
        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return np.reshape(x_train,(60000,28**2)), 28**2
    if data_type == 6:
        #FIX DANCER DATA
        return 0
    if data_type == 7:
        mat = spio.loadmat(base_path + 'gen3.mat', squeeze_me=True)
        return 100*mat['pts'],3
    if data_type == 8:
        mat = spio.loadmat(base_path + 'CirclesBig.mat', squeeze_me=True)
        return 100*mat['pts'],3
    if data_type == 9:
        mat = spio.loadmat(base_path + 'CirclesSmall.mat', squeeze_me=True)
        return 100*mat['pts'],3
    return  0
    
def sample_pts(pts,n_sample):
    return pts[np.random.randint(1,len(pts),n_sample),:]

def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)

def FPS_sampling(pts,K,input_dim):
    farthest_pts = np.zeros((K, input_dim))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts


def rand_rotation(dim):
    projmat = special_ortho_group.rvs(dim)
    return projmat
    
############  SESSION  #############################################################################################
print("Loading Data")
sname = sname + '_Model' + str(model_type)
pts,input_dim = load_obj_pts(data_type)

print("Initializing Model")
if model_type == 1:
    mld = Standard_Model.Model(input_dim,n_latent,n_hidden)
    train_step = tf.train.AdamOptimizer(alpha).minimize(mdl.TotalLoss)
if model_type == 2 or model_type == 5 or model_type == 6 or model_type == 7:
    if model_type == 2:
        mdl = Patch_Model_4.Model(input_dim,n_latent,patch_dim,n_hidden,batch_size,reg_lam)
        from Patch_Tester_4 import Test
    if model_type == 5:
        mdl = PatchandField_Model_4.Model(input_dim,n_latent,patch_dim,n_hidden,batch_size,reg_lam)
        from PatchandField_Tester_4 import Test
    if model_type == 6:
        mdl = PatchConfidence_Model_4.Model(input_dim,n_latent,patch_dim,n_hidden,batch_size,reg_lam)
        from Patch_Tester_4 import Test 
    if model_type == 7:
        mdl = PatchLearnedConfidence_Model_4.Model(input_dim,n_latent,patch_dim,n_hidden,batch_size,reg_lam)
        from Patch_Tester_4 import Test 
    train_step    = tf.train.AdamOptimizer(alpha).minimize(mdl.TotalLoss)
    initial_step1 = tf.train.AdamOptimizer(alpha).minimize(mdl.e1loss)
    initial_step2 = tf.train.AdamOptimizer(alpha).minimize(mdl.e2loss)
    initial_step3 = tf.train.AdamOptimizer(alpha).minimize(mdl.e3loss)
    initial_step4 = tf.train.AdamOptimizer(alpha).minimize(mdl.e4loss)  
    pre_step      = tf.train.AdamOptimizer(alpha).minimize(mdl.pre_loss)
    n_patches     = 4
    
if model_type == 3 or model_type == 4 or  model_type == 8:
    if model_type == 3:
        mdl = Patch_Model_10.Model(input_dim,n_latent,patch_dim,n_hidden,batch_size,reg_lam)
    if model_type == 4:
        mdl = Unity_Model_10.Model(input_dim,n_latent,patch_dim,n_hidden,batch_size,reg_lam)
    if model_type == 8:
        mdl = PatchConvolution_Model_10.Model(input_dim,n_latent,patch_dim,n_hidden,batch_size,reg_lam)
    train_step     = tf.train.AdamOptimizer(alpha).minimize(mdl.TotalLoss)
    initial_step1  = tf.train.AdamOptimizer(alpha).minimize(mdl.e1loss)
    initial_step2  = tf.train.AdamOptimizer(alpha).minimize(mdl.e2loss)
    initial_step3  = tf.train.AdamOptimizer(alpha).minimize(mdl.e3loss)
    initial_step4  = tf.train.AdamOptimizer(alpha).minimize(mdl.e4loss)  
    initial_step5  = tf.train.AdamOptimizer(alpha).minimize(mdl.e5loss)
    initial_step6  = tf.train.AdamOptimizer(alpha).minimize(mdl.e6loss)
    initial_step7  = tf.train.AdamOptimizer(alpha).minimize(mdl.e7loss)
    initial_step8  = tf.train.AdamOptimizer(alpha).minimize(mdl.e8loss)  
    initial_step9  = tf.train.AdamOptimizer(alpha).minimize(mdl.e9loss)
    initial_step10 = tf.train.AdamOptimizer(alpha).minimize(mdl.e10loss)
    pre_step       = tf.train.AdamOptimizer(alpha).minimize(mdl.pre_loss)
    n_patches      = 10
    if data_type != 5:
        from Patch_Tester_10 import Test
    else:
        from Patch_Tester_MNIST import Test

init = tf.global_variables_initializer()
sess = tf.Session()
if debug == 1:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan) 
sess.run(init)
#####################################################################################################3

#choose some initials
if FPS == 1:
    print("Computing FPS")
    initial_pts = FPS_sampling(pts,n_patches,input_dim)
else:
	initial_pts = sample_pts(pts,n_patches)  

#initialize with over fit
if initial_fit == 1:
    print('Training Initial Patches')
    for i in range(n_initial):
        if data_type == 1:
           input_pts= np.array(sample_point(i%n_patches))
        else:
           input_pts = [initial_pts[i%n_patches,:]]
    
        Ptrue = np.zeros(n_patches)
        Ptrue[i%n_patches] = 1
        
        if i % n_patches == 0:
            _, initLoss,z1val = sess.run([initial_step1, mdl.e1loss, mdl.z1], 
                feed_dict = {mdl.InputPts:input_pts, mdl.Noise:noise})   
        if i % n_patches == 1:
            _, initLoss = sess.run([initial_step2, mdl.e2loss], 
                feed_dict = {mdl.InputPts:input_pts, mdl.Noise:noise})   
        if i % n_patches == 2:
            _, initLoss = sess.run([initial_step3, mdl.e3loss], 
                feed_dict = {mdl.InputPts:input_pts, mdl.Noise:noise})
        if i % n_patches == 3:
            _, initLoss = sess.run([initial_step4, mdl.e4loss], 
                feed_dict = {mdl.InputPts:input_pts, mdl.Noise:noise})
        #only valid if n_patches > 4
        if i % n_patches == 4:
            _, initLoss = sess.run([initial_step5, mdl.e5loss], 
                feed_dict = {mdl.InputPts:input_pts, mdl.Noise:noise})    
        if i % n_patches == 5:
            _, initLoss = sess.run([initial_step6, mdl.e6loss], 
                feed_dict = {mdl.InputPts:input_pts, mdl.Noise:noise})   
        if i % n_patches == 6:
            _, initLoss = sess.run([initial_step7, mdl.e7loss], 
                feed_dict = {mdl.InputPts:input_pts, mdl.Noise:noise})    
        if i % n_patches == 7:
            _, initLoss = sess.run([initial_step8, mdl.e8loss], 
                feed_dict = {mdl.InputPts:input_pts, mdl.Noise:noise})                
        if i % n_patches == 8:
            _, initLoss = sess.run([initial_step9, mdl.e9loss], 
                feed_dict = {mdl.InputPts:input_pts, mdl.Noise:noise})    
        if i % n_patches == 9:
            _, initLoss = sess.run([initial_step10, mdl.e10loss], 
                feed_dict = {mdl.InputPts:input_pts, mdl.Noise:noise})
       #predictor
        _, preLoss,zReg = sess.run([pre_step, mdl.pre_loss, mdl.zreg], 
                feed_dict = {mdl.InputPts:input_pts, mdl.Noise:noise, mdl.InputP:np.expand_dims(Ptrue,0)})
        
        if not i % report:
            print('Iter %03d, Loss: %.3e  %.3e  %.3e' %(i, initLoss,preLoss,zReg))
            print(z1val)

        
#main training loop
print('Begining Full Training') 
Losses = []
lavg = 0
for i in range(n_train):
    #generate samples pts
    if data_type == 1:
        input_pts = sample_sphere(batch_size)
    else:
        input_pts = sample_pts(pts,batch_size)

    _, l1, l2, l3, l4,test = sess.run([train_step, mdl.TotalLoss, mdl.pt_loss, mdl.pred_loss, mdl.part_loss, mdl.test], 
        feed_dict = {mdl.InputPts:input_pts, mdl.Noise:noise})
    lavg = lavg + l1
    
    if not i % report:
        lavg = (1/batch_size)*lavg
        if not i == 0:
            Losses.append(l1)
        print('Iteration %03d: PT/Pred/Part:%.3e,%.3e,%.3e Total: %.3e' % (i, l2,l3,l4,lavg))
        #print(test)
        lavg = 0
            
    

################## save ########################################################################
print('Saving Model')
if computer == 2:
    saver = tf.train.Saver()
    saver.save(sess,base_path + '/SavedModels/' + sname + '.ckpt')

print('Saving Workspace')
workspace = shelve.open(base_path + '/SavedModels/' + sname + 'shelve.out', 'n')
for key in dir():
    try:
        workspace[key] = globals()[key]
    except TypeError:
        print('ERROR shelving: {0}'.format(key))
workspace.close()
print('Done')

############Test##########################################
Test(n_sample,data_type,pts,sess,mdl,n_latent,base_path,sname,Losses, patch_dim)

#Convergences
fig = plt.figure(1)
ax1 =  fig.add_subplot(211)
ax1.plot(range(len(Losses)),Losses)
ax1 =  fig.add_subplot(212)
ax1.plot(range(len(Losses)),np.log(Losses))
plt.show()
