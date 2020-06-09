import tensorflow as tf

def sigma(x, alpha=0.1):
    #return tf.math.sigmoid(x)
    #return tf.nn.relu(x)
    return tf.maximum(x, tf.multiply(x, alpha))
    
def sigma2(x, alpha=0.1):
    return tf.math.sigmoid(x)
    #return tf.nn.relu(x)
    #return tf.maximum(x, tf.multiply(x, alpha))
    
def sigma3(x, alpha=0.1):
    return x
    #return tf.nn.relu(x)
    #return tf.maximum(x, tf.multiply(x, alpha))
    
    
def weight_var(shape):
    initial = tf.truncated_normal(shape=shape, stddev=.05)
    return tf.Variable(initial)

class Model:
    def __init__(self,input_dim,n_latent,patch_dim,n_hidden,batch_size,reg_lam):
        #Model Parameters
        self.n_latent = n_latent
        self.batch_size = batch_size
        self.input_dim  = input_dim
        self.patch_dim  = patch_dim
        self.reg_lambda = reg_lam
        #Placeholders
        self.InputPts  = tf.placeholder(dtype=tf.float32, shape=[None,input_dim])
        self.Noise     = tf.placeholder(dtype=tf.float32)
        self.InputP    = tf.placeholder(dtype=tf.float32, shape=[None,10])
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
        self.S5W1 = weight_var([n_latent,n_hidden])
        self.S5W2 = weight_var([n_hidden,patch_dim])
        self.S6W1 = weight_var([n_latent,n_hidden])
        self.S6W2 = weight_var([n_hidden,patch_dim])
        self.S7W1 = weight_var([n_latent,n_hidden])
        self.S7W2 = weight_var([n_hidden,patch_dim])
        self.S8W1 = weight_var([n_latent,n_hidden])
        self.S8W2 = weight_var([n_hidden,patch_dim])
        self.S9W1 = weight_var([n_latent,n_hidden])
        self.S9W2 = weight_var([n_hidden,patch_dim])
        self.S10W1 = weight_var([n_latent,n_hidden])
        self.S10W2 = weight_var([n_hidden,patch_dim])        
        self.S1B1 = weight_var([n_hidden])
        self.S1B2 = weight_var([patch_dim])
        self.S2B1 = weight_var([n_hidden])
        self.S2B2 = weight_var([patch_dim])
        self.S3B1 = weight_var([n_hidden])
        self.S3B2 = weight_var([patch_dim])
        self.S4B1 = weight_var([n_hidden])
        self.S4B2 = weight_var([patch_dim])
        self.S5B1 = weight_var([n_hidden])
        self.S5B2 = weight_var([patch_dim])
        self.S6B1 = weight_var([n_hidden])
        self.S6B2 = weight_var([patch_dim])
        self.S7B1 = weight_var([n_hidden])
        self.S7B2 = weight_var([patch_dim])
        self.S8B1 = weight_var([n_hidden])
        self.S8B2 = weight_var([patch_dim])
        self.S9B1 = weight_var([n_hidden])
        self.S9B2 = weight_var([patch_dim])
        self.S10B1 = weight_var([n_hidden])
        self.S10B2 = weight_var([patch_dim])
       
        #Decoder 1
        self.D1W1 = weight_var([patch_dim, n_hidden])
        self.D1W2 = weight_var([n_hidden, n_hidden])
        self.D1W3 = weight_var([n_hidden, n_hidden])
        self.D1W4 = weight_var([n_hidden, n_hidden])
        self.D1W5 = weight_var([n_hidden,input_dim])
        self.D1B1 = weight_var([n_hidden])
        self.D1B2 = weight_var([n_hidden])
        self.D1B3 = weight_var([n_hidden])
        self.D1B4 = weight_var([n_hidden])
        self.D1B5 = weight_var([input_dim])
        #Decoder 2
        self.D2W1 = weight_var([patch_dim, n_hidden])
        self.D2W2 = weight_var([n_hidden, n_hidden])
        self.D2W3 = weight_var([n_hidden, n_hidden])
        self.D2W4 = weight_var([n_hidden, n_hidden])
        self.D2W5 = weight_var([n_hidden,input_dim])
        self.D2B1 = weight_var([n_hidden])
        self.D2B2 = weight_var([n_hidden])
        self.D2B3 = weight_var([n_hidden])
        self.D2B4 = weight_var([n_hidden])
        self.D2B5 = weight_var([input_dim])
        #Decoder 3
        self.D3W1 = weight_var([patch_dim, n_hidden])
        self.D3W2 = weight_var([n_hidden, n_hidden])
        self.D3W3 = weight_var([n_hidden, n_hidden])
        self.D3W4 = weight_var([n_hidden, n_hidden])
        self.D3W5 = weight_var([n_hidden,input_dim])
        self.D3B1 = weight_var([n_hidden])
        self.D3B2 = weight_var([n_hidden])
        self.D3B3 = weight_var([n_hidden])
        self.D3B4 = weight_var([n_hidden])
        self.D3B5 = weight_var([input_dim])
        #Decoder 4
        self.D4W1 = weight_var([patch_dim, n_hidden])
        self.D4W2 = weight_var([n_hidden, n_hidden])
        self.D4W3 = weight_var([n_hidden, n_hidden])
        self.D4W4 = weight_var([n_hidden, n_hidden])
        self.D4W5 = weight_var([n_hidden, input_dim])
        self.D4B1 = weight_var([n_hidden])
        self.D4B2 = weight_var([n_hidden])
        self.D4B3 = weight_var([n_hidden])
        self.D4B4 = weight_var([n_hidden])
        self.D4B5 = weight_var([input_dim]) 
        #Decoder 5
        self.D5W1 = weight_var([patch_dim, n_hidden])
        self.D5W2 = weight_var([n_hidden, n_hidden])
        self.D5W3 = weight_var([n_hidden, n_hidden])
        self.D5W4 = weight_var([n_hidden, n_hidden])
        self.D5W5 = weight_var([n_hidden, input_dim])
        self.D5B1 = weight_var([n_hidden])
        self.D5B2 = weight_var([n_hidden])
        self.D5B3 = weight_var([n_hidden])
        self.D5B4 = weight_var([n_hidden])
        self.D5B5 = weight_var([input_dim]) 
        #Decoder 6
        self.D6W1 = weight_var([patch_dim, n_hidden])
        self.D6W2 = weight_var([n_hidden, n_hidden])
        self.D6W3 = weight_var([n_hidden, n_hidden])
        self.D6W4 = weight_var([n_hidden, n_hidden])
        self.D6W5 = weight_var([n_hidden, input_dim])
        self.D6B1 = weight_var([n_hidden])
        self.D6B2 = weight_var([n_hidden])
        self.D6B3 = weight_var([n_hidden])
        self.D6B4 = weight_var([n_hidden])
        self.D6B5 = weight_var([input_dim]) 
        #Decoder 7
        self.D7W1 = weight_var([patch_dim, n_hidden])
        self.D7W2 = weight_var([n_hidden, n_hidden])
        self.D7W3 = weight_var([n_hidden, n_hidden])
        self.D7W4 = weight_var([n_hidden, n_hidden])
        self.D7W5 = weight_var([n_hidden, input_dim])
        self.D7B1 = weight_var([n_hidden])
        self.D7B2 = weight_var([n_hidden])
        self.D7B3 = weight_var([n_hidden])
        self.D7B4 = weight_var([n_hidden])
        self.D7B5 = weight_var([input_dim]) 
        #Decoder 8
        self.D8W1 = weight_var([patch_dim, n_hidden])
        self.D8W2 = weight_var([n_hidden, n_hidden])
        self.D8W3 = weight_var([n_hidden, n_hidden])
        self.D8W4 = weight_var([n_hidden, n_hidden])
        self.D8W5 = weight_var([n_hidden, input_dim])
        self.D8B1 = weight_var([n_hidden])
        self.D8B2 = weight_var([n_hidden])
        self.D8B3 = weight_var([n_hidden])
        self.D8B4 = weight_var([n_hidden])
        self.D8B5 = weight_var([input_dim]) 
        #Decoder 9
        self.D9W1 = weight_var([patch_dim, n_hidden])
        self.D9W2 = weight_var([n_hidden, n_hidden])
        self.D9W3 = weight_var([n_hidden, n_hidden])
        self.D9W4 = weight_var([n_hidden, n_hidden])
        self.D9W5 = weight_var([n_hidden, input_dim])
        self.D9B1 = weight_var([n_hidden])
        self.D9B2 = weight_var([n_hidden])
        self.D9B3 = weight_var([n_hidden])
        self.D9B4 = weight_var([n_hidden])
        self.D9B5 = weight_var([input_dim]) 
        #Decoder 10
        self.D10W1 = weight_var([patch_dim, n_hidden])
        self.D10W2 = weight_var([n_hidden, n_hidden])
        self.D10W3 = weight_var([n_hidden, n_hidden])
        self.D10W4 = weight_var([n_hidden, n_hidden])
        self.D10W5 = weight_var([n_hidden, input_dim])
        self.D10B1 = weight_var([n_hidden])
        self.D10B2 = weight_var([n_hidden])
        self.D10B3 = weight_var([n_hidden])
        self.D10B4 = weight_var([n_hidden])
        self.D10B5 = weight_var([input_dim]) 
        #Predictor
        self.PW1 = weight_var([n_latent,n_hidden])
        self.PW2 = weight_var([n_hidden,n_hidden])
        self.PW3 = weight_var([n_hidden,n_hidden])
        self.PW4 = weight_var([n_hidden,10])
        self.PB1 = weight_var([n_hidden])
        self.PB2 = weight_var([n_hidden])
        self.PB3 = weight_var([n_hidden])
        self.PB4 = weight_var([10])
        
        #Network Connections   
        self.z  = self.encoder(self.InputPts)
        self.z1,self.z2,self.z3,self.z4,self.z5,self.z6,self.z7,self.z8,self.z9,self.z10 = self.splitter(self.z)
        self.x_out1  = self.decoder_1(self.z1)
        self.x_out2  = self.decoder_2(self.z2)
        self.x_out3  = self.decoder_3(self.z3)
        self.x_out4  = self.decoder_4(self.z4)
        self.x_out5  = self.decoder_5(self.z5)
        self.x_out6  = self.decoder_6(self.z6)
        self.x_out7  = self.decoder_7(self.z7)
        self.x_out8  = self.decoder_8(self.z8)
        self.x_out9  = self.decoder_9(self.z9)
        self.x_out10 = self.decoder_10(self.z10)
        self.predict, self.log_predict = self.predict_patch(self.z)
        self.test = self.x_out1
        
        #Losses
        self.reg = self.regularizer_fun() 
        self.zreg = self.z_dist()
        self.pre_loss = -tf.reduce_mean(tf.multiply(self.InputP,self.log_predict))+ self.reg_lambda*(self.zreg + self.reg)
        #self.pre_loss = tf.nn.softmax_cross_entropy_with_logits_v2((self.predict),self.InputP)+ self.reg_lambda*(self.zreg + self.reg)
        self.pt_loss,self.part_loss,self.pred_loss,self.e1loss,self.e2loss,self.e3loss,self.e4loss,self.e5loss,self.e6loss,self.e7loss,self.e8loss,self.e9loss,self.e10loss = self.predict_loss(
            self.x_out1,self.x_out2,self.x_out3,self.x_out4,self.x_out5,
            self.x_out6,self.x_out7,self.x_out8,self.x_out9,self.x_out10)            
        self.TotalLoss = (1/self.batch_size)*(self.pt_loss + self.pred_loss + self.part_loss) + self.reg_lambda*self.reg
        
    def predict_loss(self,x_out1,x_out2,x_out3,x_out4,x_out5,x_out6,x_out7,x_out8,x_out9,x_out10):
        #check each reconstruction loss
        e1 = tf.reduce_sum(tf.squared_difference(x_out1,self.InputPts),axis = 1)
        e2 = tf.reduce_sum(tf.squared_difference(x_out2,self.InputPts),axis = 1)
        e3 = tf.reduce_sum(tf.squared_difference(x_out3,self.InputPts),axis = 1)
        e4 = tf.reduce_sum(tf.squared_difference(x_out4,self.InputPts),axis = 1)
        e5 = tf.reduce_sum(tf.squared_difference(x_out5,self.InputPts),axis = 1)
        e6 = tf.reduce_sum(tf.squared_difference(x_out6,self.InputPts),axis = 1)
        e7 = tf.reduce_sum(tf.squared_difference(x_out7,self.InputPts),axis = 1)
        e8 = tf.reduce_sum(tf.squared_difference(x_out8,self.InputPts),axis = 1)
        e9 = tf.reduce_sum(tf.squared_difference(x_out9,self.InputPts),axis = 1)
        e10 = tf.reduce_sum(tf.squared_difference(x_out10,self.InputPts),axis = 1)
        #compute losses
        pt_lossVec = tf.stack([e1,e2,e3,e4,e5,e6,e7,e8,e9,e10])
        pt_loss = tf.reduce_sum(tf.reduce_min(pt_lossVec,axis = 0))
        labels = tf.transpose(tf.divide(1.0,pt_lossVec))
        labels = tf.nn.softmax(labels)#tf.stop_gradient(tf.nn.softmax(labels))
        pred_loss = -tf.reduce_mean(tf.multiply(labels,self.log_predict)) 
        part_loss = -tf.reduce_mean(tf.multiply(tf.transpose(pt_lossVec), self.log_predict))
        # print(x_out1)
        # print(e1)
        # print(pt_lossVec)
        # print(pt_loss)
        # print(labels)
        # print(pred_loss)
        # print(part_loss)
        return pt_loss, part_loss, pred_loss, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10    
        
    def regularizer_fun(self):
        SUMDecoderVars = (tf.reduce_sum(tf.stack([tf.norm(self.D1W1)*tf.norm(self.D1W2)*tf.norm(self.D1W3),
            tf.norm(self.D2W1)*tf.norm(self.D2W2)*tf.norm(self.D2W3),
            tf.norm(self.D3W1)*tf.norm(self.D3W2)*tf.norm(self.D3W3),
            tf.norm(self.D4W1)*tf.norm(self.D4W2)*tf.norm(self.D4W3),
            tf.norm(self.D5W1)*tf.norm(self.D5W2)*tf.norm(self.D5W3),
            tf.norm(self.D6W1)*tf.norm(self.D6W2)*tf.norm(self.D6W3),
            tf.norm(self.D7W1)*tf.norm(self.D7W2)*tf.norm(self.D7W3),
            tf.norm(self.D8W1)*tf.norm(self.D8W2)*tf.norm(self.D8W3),
            tf.norm(self.D9W1)*tf.norm(self.D9W2)*tf.norm(self.D9W3),
            tf.norm(self.D10W1)*tf.norm(self.D10W2)+tf.norm(self.D10W3)], axis=0)))
        MAXDecoderVars = (tf.reduce_max(tf.stack([tf.norm(self.D1W1)*tf.norm(self.D1W2)*tf.norm(self.D1W3),
            tf.norm(self.D2W1)*tf.norm(self.D2W2)*tf.norm(self.D2W3),
            tf.norm(self.D3W1)*tf.norm(self.D3W2)*tf.norm(self.D3W3),
            tf.norm(self.D4W1)*tf.norm(self.D4W2)*tf.norm(self.D4W3),
            tf.norm(self.D5W1)*tf.norm(self.D5W2)*tf.norm(self.D5W3),
            tf.norm(self.D6W1)*tf.norm(self.D6W2)*tf.norm(self.D6W3),
            tf.norm(self.D7W1)*tf.norm(self.D7W2)*tf.norm(self.D7W3),
            tf.norm(self.D8W1)*tf.norm(self.D8W2)*tf.norm(self.D8W3),
            tf.norm(self.D9W1)*tf.norm(self.D9W2)*tf.norm(self.D9W3),
            tf.norm(self.D10W1)*tf.norm(self.D10W2)+tf.norm(self.D10W3)], axis=0)))
        EncoderVars = (tf.norm(self.EW1)+tf.norm(self.EW2)+tf.norm(self.EW3))
        SplitterVars = (tf.norm(self.S1W1)+tf.norm(self.S1W2)+tf.norm(self.S2W1)+tf.norm(self.S2W2)+
            tf.norm(self.S3W1)+tf.norm(self.S3W2)+tf.norm(self.S4W1)+tf.norm(self.S4W2)+
            tf.norm(self.S5W1)+tf.norm(self.S5W2)+tf.norm(self.S6W1)+tf.norm(self.S6W2)+
            tf.norm(self.S7W1)+tf.norm(self.S7W2)+tf.norm(self.S8W1)+tf.norm(self.S8W2)+
            tf.norm(self.S9W1)+tf.norm(self.S9W2)+tf.norm(self.S10W1)+tf.norm(self.S10W2))
        PredictorVars = (tf.norm(self.PW1)+tf.norm(self.PW2)+tf.norm(self.PW3))
        Total = MAXDecoderVars + (1/10)*SUMDecoderVars+(1/1000)*(EncoderVars+SplitterVars+PredictorVars)
        return Total
    
    def z_dist(self):
        d1  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z1,[[.5],[.5]]),axis = 1),axis = 0)
        d2  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z2,[[.5],[.5]]),axis = 1),axis = 0)
        d3  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z3,[[.5],[.5]]),axis = 1),axis = 0)
        d4  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z4,[[.5],[.5]]),axis = 1),axis = 0)
        d5  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z5,[[.5],[.5]]),axis = 1),axis = 0)
        d6  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z6,[[.5],[.5]]),axis = 1),axis = 0)
        d7  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z7,[[.5],[.5]]),axis = 1),axis = 0)
        d8  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z8,[[.5],[.5]]),axis = 1),axis = 0)
        d9  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z9,[[.5],[.5]]),axis = 1),axis = 0)
        d10 = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z10,[[.5],[.5]]),axis = 1),axis = 0)
        all_dist = tf.stack([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10])
        zreg = tf.reduce_sum(tf.multiply(all_dist,tf.transpose(self.InputP)))
        # print(d1)
        # print(all_dist)
        # print(self.InputP)
        # print(zreg)
        return zreg
    
    def encoder(self,InputPts):
        x  = InputPts+self.Noise*tf.random_normal([self.batch_size,3])
        x  = sigma(tf.linalg.matmul(InputPts,self.EW1)+self.EB1)
        x  = sigma(tf.linalg.matmul(x,self.EW2)+self.EB2)
        x  = sigma(tf.linalg.matmul(x,self.EW3)+self.EB3)
        return x

    def splitter(self,z):
        z1 = sigma2(tf.linalg.matmul( sigma( tf.linalg.matmul(z,self.S1W1)  +self.S1B1),self.S1W2)+self.S1B2)
        z2 = sigma2(tf.linalg.matmul( sigma( tf.linalg.matmul(z,self.S2W1)  +self.S2B1),self.S2W2)+self.S2B2)
        z3 = sigma2(tf.linalg.matmul( sigma( tf.linalg.matmul(z,self.S3W1)  +self.S3B1),self.S3W2)+self.S3B2)
        z4 = sigma2(tf.linalg.matmul( sigma( tf.linalg.matmul(z,self.S4W1)  +self.S4B1),self.S4W2)+self.S4B2)
        z5 = sigma2(tf.linalg.matmul( sigma( tf.linalg.matmul(z,self.S5W1)  +self.S5B1),self.S5W2)+self.S5B2)
        z6 = sigma2(tf.linalg.matmul( sigma( tf.linalg.matmul(z,self.S6W1)  +self.S6B1),self.S6W2)+self.S6B2)
        z7 = sigma2(tf.linalg.matmul( sigma( tf.linalg.matmul(z,self.S7W1)  +self.S7B1),self.S7W2)+self.S7B2)
        z8 = sigma2(tf.linalg.matmul( sigma( tf.linalg.matmul(z,self.S8W1)  +self.S8B1),self.S8W2)+self.S8B2)
        z9 = sigma2(tf.linalg.matmul( sigma( tf.linalg.matmul(z,self.S9W1)  +self.S9B1),self.S9W2)+self.S9B2)
        z10 = sigma2(tf.linalg.matmul( sigma( tf.linalg.matmul(z,self.S10W1)+self.S10B1),self.S10W2)+self.S10B2)
        return z1,z2,z3,z4,z5,z6,z7,z8,z9,z10
        
    def decoder_1(self,y):
        x = sigma(tf.linalg.matmul(y,self.D1W1)+self.D1B1)
        x = sigma(tf.linalg.matmul(x,self.D1W2)+self.D1B2)
        x = sigma(tf.linalg.matmul(x,self.D1W3)+self.D1B3)
        x = sigma(tf.linalg.matmul(x,self.D1W4)+self.D1B4)        
        x = sigma3(tf.linalg.matmul(x,self.D1W5)+self.D1B5)
        return x
    def decoder_2(self,y):
        x = sigma(tf.linalg.matmul(y,self.D2W1)+self.D2B1)
        x = sigma(tf.linalg.matmul(x,self.D2W2)+self.D2B2)
        x = sigma(tf.linalg.matmul(x,self.D2W3)+self.D2B3)
        x = sigma(tf.linalg.matmul(x,self.D2W4)+self.D2B4)
        x = sigma3(tf.linalg.matmul(x,self.D2W5)+self.D2B5)
        return x
    def decoder_3(self,y):
        x = sigma(tf.linalg.matmul(y,self.D3W1)+self.D3B1)
        x = sigma(tf.linalg.matmul(x,self.D3W2)+self.D3B2)
        x = sigma(tf.linalg.matmul(x,self.D3W3)+self.D3B3)
        x = sigma(tf.linalg.matmul(x,self.D3W4)+self.D3B4)          
        x = sigma3(tf.linalg.matmul(x,self.D3W5)+self.D3B5)
        return x
    def decoder_4(self,y):
        x = sigma(tf.linalg.matmul(y,self.D4W1)+self.D4B1)
        x = sigma(tf.linalg.matmul(x,self.D4W2)+self.D4B2)
        x = sigma(tf.linalg.matmul(x,self.D4W3)+self.D4B3)
        x = sigma(tf.linalg.matmul(x,self.D4W4)+self.D4B4)
        x = sigma3(tf.linalg.matmul(x,self.D4W5)+self.D4B5)
        return x
    def decoder_5(self,y):
        x = sigma(tf.linalg.matmul(y,self.D5W1)+self.D5B1)
        x = sigma(tf.linalg.matmul(x,self.D5W2)+self.D5B2)
        x = sigma(tf.linalg.matmul(x,self.D5W3)+self.D5B3)
        x = sigma(tf.linalg.matmul(x,self.D5W4)+self.D5B4)
        x = sigma3(tf.linalg.matmul(x,self.D5W5)+self.D5B5)
        return x        
    def decoder_6(self,y):
        x = sigma(tf.linalg.matmul(y,self.D6W1)+self.D6B1)
        x = sigma(tf.linalg.matmul(x,self.D6W2)+self.D6B2)
        x = sigma(tf.linalg.matmul(x,self.D6W3)+self.D6B3)
        x = sigma(tf.linalg.matmul(x,self.D6W4)+self.D6B4)
        x = sigma3(tf.linalg.matmul(x,self.D6W5)+self.D6B5)
        return x
    def decoder_7(self,y):
        x = sigma(tf.linalg.matmul(y,self.D7W1)+self.D7B1)
        x = sigma(tf.linalg.matmul(x,self.D7W2)+self.D7B2)
        x = sigma(tf.linalg.matmul(x,self.D7W3)+self.D7B3)
        x = sigma(tf.linalg.matmul(x,self.D7W4)+self.D7B4)
        x = sigma2(tf.linalg.matmul(x,self.D7W5)+self.D7B5)
        return x 
    def decoder_8(self,y):
        x = sigma(tf.linalg.matmul(y,self.D8W1)+self.D8B1)
        x = sigma(tf.linalg.matmul(x,self.D8W2)+self.D8B2)
        x = sigma(tf.linalg.matmul(x,self.D8W3)+self.D8B3)
        x = sigma(tf.linalg.matmul(x,self.D8W4)+self.D8B4)
        x = sigma3(tf.linalg.matmul(x,self.D8W5)+self.D8B5)
        return x 
    def decoder_9(self,y):
        x = sigma(tf.linalg.matmul(y,self.D9W1)+self.D9B1)
        x = sigma(tf.linalg.matmul(x,self.D9W2)+self.D9B2)
        x = sigma(tf.linalg.matmul(x,self.D9W3)+self.D9B3)
        x = sigma(tf.linalg.matmul(x,self.D9W4)+self.D9B4)
        x = sigma3(tf.linalg.matmul(x,self.D9W5)+self.D9B5)
        return x
    def decoder_10(self,y):
        x = sigma(tf.linalg.matmul(y,self.D10W1)+self.D10B1)
        x = sigma(tf.linalg.matmul(x,self.D10W2)+self.D10B2)
        x = sigma(tf.linalg.matmul(x,self.D10W3)+self.D10B3)
        x = sigma(tf.linalg.matmul(x,self.D10W4)+self.D10B4)
        x = sigma3(tf.linalg.matmul(x,self.D10W5)+self.D10B5)
        return x
    
    def predict_patch(self,z):
        x = sigma(tf.linalg.matmul(z,self.PW1)+self.PB1)
        x = sigma(tf.linalg.matmul(x,self.PW2)+self.PB2)
        x = sigma(tf.linalg.matmul(x,self.PW3)+self.PB3)        
        x = tf.linalg.matmul(x,self.PW4)+self.PB4
        predict = tf.nn.softmax(x)
        log_predict = tf.nn.log_softmax(x)
        return predict, log_predict
    # def predict_patch(self,z):
        # center = tf.constant(0.5, shape=[self.batch_size, self.patch_dim])
        # d1  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z1,center),axis = 1),axis = 0)
        # d2  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z2,center),axis = 1),axis = 0)
        # d3  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z3,center),axis = 1),axis = 0)
        # d4  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z4,center),axis = 1),axis = 0)
        # d5  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z5,center),axis = 1),axis = 0)
        # d6  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z6,center),axis = 1),axis = 0)
        # d7  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z7,center),axis = 1),axis = 0)
        # d8  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z8,center),axis = 1),axis = 0)
        # d9  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z9,center),axis = 1),axis = 0)
        # d10  = tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.z10,center),axis = 1),axis = 0)
        # all_dist = tf.stack([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10], axis = 0)
        # x = tf.divide(1.0,all_dist)
        # predict = tf.nn.softmax(x)
        # log_predict = tf.nn.log_softmax(x)
        # return predict, log_predict
        

               
    
