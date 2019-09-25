import tensorflow as tf
    
def sigma2(x, alpha=0.1):
    #return tf.math.sigmoid(x)
    #return tf.nn.relu(x)
    #return tf.maximum(x, tf.multiply(x, alpha))
    return tf.nn.elu(x)

def sigma(x):
    return tf.math.sigmoid(x)
    
def weight_var(shape):
    initial = tf.truncated_normal(shape=shape, stddev=.05)
    return tf.Variable(initial)

class Model:
    def __init__(self,input_dim,n_latent,patch_dim,n_hidden,batch_size,reg_lambda):
        #Model Parameters
        self.n_latent   = n_latent
        self.batch_size = batch_size
        self.input_dim  = input_dim
        self.patch_dim  = patch_dim
        self.reg_lambda = reg_lambda
        self.center = tf.constant([.5], shape =[1,patch_dim])
                             
        #Placeholders
        self.InputPts  = tf.placeholder(dtype=tf.float32, shape=[None,input_dim])
        self.Noise     = tf.placeholder(dtype=tf.float32)
        self.InputP    = tf.placeholder(dtype=tf.float32, shape=[None,4])
        #Encoder Variables
        self.EW1 = weight_var([input_dim, n_hidden])
        self.EW2 = weight_var([n_hidden, n_hidden])
        self.EW3 = weight_var([n_hidden, n_latent])
        self.EB1 = weight_var([n_hidden])
        self.EB2 = weight_var([n_hidden])
        self.EB3 = weight_var([n_latent])
        #Splitter Variables
        self.S1W1 = weight_var([n_latent,n_hidden])
        self.S1W2 = weight_var([n_hidden,patch_dim])
        self.S2W1 = weight_var([n_latent,n_hidden])
        self.S2W2 = weight_var([n_hidden,patch_dim])
        self.S3W1 = weight_var([n_latent,n_hidden])
        self.S3W2 = weight_var([n_hidden,patch_dim])
        self.S4W1 = weight_var([n_latent,n_hidden])
        self.S4W2 = weight_var([n_hidden,patch_dim])
        self.S1B1 = weight_var([n_hidden])
        self.S1B2 = weight_var([patch_dim])
        self.S2B1 = weight_var([n_hidden])
        self.S2B2 = weight_var([patch_dim])
        self.S3B1 = weight_var([n_hidden])
        self.S3B2 = weight_var([patch_dim])
        self.S4B1 = weight_var([n_hidden])
        self.S4B2 = weight_var([patch_dim])
        #Decoder 1
        self.D1W1 = weight_var([patch_dim, n_hidden])
        self.D1W2 = weight_var([n_hidden, n_hidden])
        self.D1W3 = weight_var([n_hidden, n_hidden])
        self.D1W4 = weight_var([n_hidden,input_dim])
        self.D1B1 = weight_var([n_hidden])
        self.D1B2 = weight_var([n_hidden])
        self.D1B3 = weight_var([n_hidden])
        self.D1B4 = weight_var([input_dim])
        #Decoder 2
        self.D2W1 = weight_var([patch_dim, n_hidden])
        self.D2W2 = weight_var([n_hidden, n_hidden])
        self.D2W3 = weight_var([n_hidden, n_hidden])
        self.D2W4 = weight_var([n_hidden,input_dim])
        self.D2B1 = weight_var([n_hidden])
        self.D2B2 = weight_var([n_hidden])
        self.D2B3 = weight_var([n_hidden])
        self.D2B4 = weight_var([input_dim])
        #Decoder 3
        self.D3W1 = weight_var([patch_dim, n_hidden])
        self.D3W2 = weight_var([n_hidden, n_hidden])
        self.D3W3 = weight_var([n_hidden, n_hidden])
        self.D3W4 = weight_var([n_hidden,input_dim])
        self.D3B1 = weight_var([n_hidden])
        self.D3B2 = weight_var([n_hidden])
        self.D3B3 = weight_var([n_hidden])
        self.D3B4 = weight_var([input_dim])
        #Decoder 4
        self.D4W1 = weight_var([patch_dim, n_hidden])
        self.D4W2 = weight_var([n_hidden, n_hidden])
        self.D4W3 = weight_var([n_hidden, n_hidden])
        self.D4W4 = weight_var([n_hidden, input_dim])
        self.D4B1 = weight_var([n_hidden])
        self.D4B2 = weight_var([n_hidden])
        self.D4B3 = weight_var([n_hidden])
        self.D4B4 = weight_var([input_dim]) 
        #Predictor
        self.PW1 = weight_var([n_latent,n_hidden])
        self.PW2 = weight_var([n_hidden,n_hidden])
        self.PW3 = weight_var([n_hidden,4])
        self.PB1 = weight_var([n_hidden])
        self.PB2 = weight_var([n_hidden])
        self.PB3 = weight_var([4])
        
        #Network Connections   
        self.z  = self.encoder(self.InputPts)
        self.z1, self.z2, self.z3, self.z4 = self.splitter(self.z)
        self.x_out1  = self.decoder_1(self.z1)
        self.x_out2  = self.decoder_2(self.z2)
        self.x_out3  = self.decoder_3(self.z3)
        self.x_out4  = self.decoder_4(self.z4)
        self.predict, self.log_predict = self.predict_patch(self.z)
        self.reg     = self.regularizer_fun()
        
        #Loss
        self.reg = self.regularizer_fun()
        self.zreg = self.z_dist()
        self.pre_loss = -tf.reduce_sum(tf.multiply(self.InputP,self.log_predict))+self.reg_lambda*(self.reg+self.zreg)
        self.pt_loss, self.part_loss, self.pred_loss, self.e1loss, self.e2loss, self.e3loss, self.e4loss, self.test = self.predict_loss(self.x_out1,self.x_out2,self.x_out3,self.x_out4)         
        self.TotalLoss = (1/self.batch_size)*(self.pt_loss + self.pred_loss + self.part_loss) + self.reg_lambda+self.reg
    
    def predict_loss(self,x_out1,x_out2,x_out3,x_out4):
        e1 = tf.reduce_sum(tf.squared_difference(x_out1,self.InputPts),axis=1)
        e2 = tf.reduce_sum(tf.squared_difference(x_out2,self.InputPts),axis=1)
        e3 = tf.reduce_sum(tf.squared_difference(x_out3,self.InputPts),axis=1)
        e4 = tf.reduce_sum(tf.squared_difference(x_out4,self.InputPts),axis=1)
        pt_lossVec = tf.stack([e1,e2,e3,e4])
        pt_loss = tf.reduce_sum(tf.reduce_min(pt_lossVec, axis = 0))
        labels = tf.transpose(tf.divide(1.0,pt_lossVec))
        labels = tf.stop_gradient(tf.nn.softmax(labels))
        pred_loss = -tf.reduce_mean(tf.multiply(labels,self.log_predict)) 
        part_loss = -tf.reduce_mean(tf.multiply(tf.transpose(pt_lossVec), self.log_predict))
        test = pt_lossVec
        # print(x_out1)
        # print(e1)
        # print(pt_lossVec)
        # print(pt_loss)
        # print(labels)
        # print(pred_loss)
        # print(part_loss)
        return pt_loss, part_loss, pred_loss, e1, e2, e3, e4, test
        
    def regularizer_fun(self):
        SUMDecoder = (tf.norm(self.D1W1)*tf.norm(self.D1W2)*tf.norm(self.D1W3)
            +tf.norm(self.D2W1)*tf.norm(self.D2W2)*tf.norm(self.D2W3)
            +tf.norm(self.D3W1)*tf.norm(self.D3W2)*tf.norm(self.D3W3)
            +tf.norm(self.D4W1)*tf.norm(self.D4W2)*tf.norm(self.D4W3))
        MAXDecoder = tf.reduce_max(tf.stack([tf.norm(self.D1W1)*tf.norm(self.D1W2)*tf.norm(self.D1W3)
            ,tf.norm(self.D2W1)*tf.norm(self.D2W2)*tf.norm(self.D2W3)
            ,tf.norm(self.D3W1)*tf.norm(self.D3W2)*tf.norm(self.D3W3)
            ,tf.norm(self.D4W1)*tf.norm(self.D4W2)*tf.norm(self.D4W3)],axis = 0))
        return MAXDecoder + .1*SUMDecoder
    
    def z_dist(self):
        d1  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z1,self.center),axis = 1),axis = 0)
        d2  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z2,self.center),axis = 1),axis = 0)
        d3  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z3,self.center),axis = 1),axis = 0)
        d4  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z4,self.center),axis = 1),axis = 0)
        all_dist = tf.stack([d1,d2,d3,d4])
        zreg = tf.reduce_sum(tf.multiply(all_dist,self.InputP))
        return zreg
     
    def encoder(self,InputPts):
        x  = InputPts+self.Noise*tf.random_normal([self.batch_size,3])
        x  = sigma2(tf.linalg.matmul(InputPts,self.EW1)+self.EB1)
        x  = sigma2(tf.linalg.matmul(x,self.EW2)+self.EB2)
        x  = sigma2(tf.linalg.matmul(x,self.EW3)+self.EB3)
        return x

    def splitter(self,z):
        z1 = sigma(tf.linalg.matmul(sigma2(tf.linalg.matmul(z,self.S1W1)+self.S1B1),self.S1W2)+self.S1B2)
        z2 = sigma(tf.linalg.matmul(sigma2(tf.linalg.matmul(z,self.S2W1)+self.S2B1),self.S2W2)+self.S2B2)
        z3 = sigma(tf.linalg.matmul(sigma2(tf.linalg.matmul(z,self.S3W1)+self.S3B1),self.S3W2)+self.S3B2)
        z4 = sigma(tf.linalg.matmul(sigma2(tf.linalg.matmul(z,self.S4W1)+self.S4B1),self.S4W2)+self.S4B2)
        return z1,z2,z3,z4
        
    def decoder_1(self,y):
        x = sigma2(tf.linalg.matmul(y,self.D1W1)+self.D1B1)
        x = sigma2(tf.linalg.matmul(x,self.D1W2)+self.D1B2)
        x = sigma2(tf.linalg.matmul(x,self.D1W3)+self.D1B3)        
        x = tf.linalg.matmul(x,self.D1W4)+self.D1B4
        return x

    def decoder_2(self,y):
        x = sigma2(tf.linalg.matmul(y,self.D2W1)+self.D2B1)
        x = sigma2(tf.linalg.matmul(x,self.D2W2)+self.D2B2)
        x = sigma2(tf.linalg.matmul(x,self.D2W3)+self.D2B3)
        x = tf.linalg.matmul(x,self.D2W4)+self.D2B4
        return x
        
    def decoder_3(self,y):
        x = sigma2(tf.linalg.matmul(y,self.D3W1)+self.D3B1)
        x = sigma2(tf.linalg.matmul(x,self.D3W2)+self.D3B2)
        x = sigma2(tf.linalg.matmul(x,self.D3W3)+self.D3B3)        
        x = tf.linalg.matmul(x,self.D3W4)+self.D3B4
        return x

    def decoder_4(self,y):
        x = sigma2(tf.linalg.matmul(y,self.D4W1)+self.D4B1)
        x = sigma2(tf.linalg.matmul(x,self.D4W2)+self.D4B2)
        x = sigma2(tf.linalg.matmul(x,self.D4W3)+self.D4B3)
        x = tf.linalg.matmul(x,self.D4W4)+self.D4B4
        return x
        
    def predict_patch(self,z):
        x = sigma2(tf.linalg.matmul(z,self.PW1)+self.PB1)
        x = sigma2(tf.linalg.matmul(x,self.PW2)+self.PB2) 
        x = tf.linalg.matmul(x,self.PW3)+self.PB3
        predict = tf.nn.softmax(x)
        log_predict = tf.nn.log_softmax(x)
        return predict, log_predict
               
    