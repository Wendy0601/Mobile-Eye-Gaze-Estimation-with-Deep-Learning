import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 
import pickle as plk
from numpy import *
from IPython.display import clear_output, Image, display, HTML

num_train=48000
num_channel=3
num_test=5000 
learning_rate = 0.001
training_iters =16000
batch_size = 128 
display_step = 10  

#author wenting li 
 

def normalize(x):
    shape=x.shape
    x=np.reshape(x,(shape[0],-1))
    x=x.astype('float32')/255
    x=x-np.mean(x)
    return np.reshape(x, shape) 

def load_data():
    npzfile = np.load("train_and_val.npz")
    train_eye_left = npzfile["train_eye_left"]
    train_eye_right = npzfile["train_eye_right"]
    train_face = npzfile["train_face"]
    train_face_mask = npzfile["train_face_mask"]
    train_y = npzfile["train_y"]
    val_eye_left = npzfile["val_eye_left"]
    val_eye_right = npzfile["val_eye_right"]
    val_face = npzfile["val_face"]
    val_face_mask = npzfile["val_face_mask"]
    val_y = npzfile["val_y"] 
    return   train_eye_left ,train_eye_right, train_face, train_face_mask,train_y ,val_eye_left,val_eye_right,val_face, val_face_mask, val_y
# load data
train_eye_left ,train_eye_right, train_face, train_face_mask,train_y ,val_eye_left,val_eye_right,val_face, val_face_mask, val_y=load_data()    


def conv2d(x, W, b,k_h, k_w, c_o, s_h, s_w,  padding,name):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, [1, s_h, s_w, 1], padding=padding) 
    x = tf.nn.bias_add(x, b)  
    return tf.nn.relu(x)  

def conv2d_no_relu(x, W, b,k_h, k_w, c_o, s_h, s_w,  padding,name):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, [1, s_h, s_w, 1], padding=padding) 
    x = tf.nn.bias_add(x, b)  
    return x

def maxpool2d(x, k_h, k_w,s_h, s_w,padding): 
    return tf.nn.max_pool(x, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

def averagepool(x, k_h, k_w,s_h, s_w,padding  ):
    return tf.nn.avg_pool(x, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

def one_path(eye_left):   
    with tf.variable_scope('conv1_eye_left'):
        k_h = 5; k_w = 5; c_o = 64; s_h = 2; s_w = 2
        wc1=tf.get_variable( 'weight1',shape = [5, 5,3, 64],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)) 
        bc1=tf.get_variable( 'bias1',
          shape = [64],
          initializer=tf.constant_initializer(0.0))
        conv1_e1 = conv2d(eye_left, wc1, bc1, k_h, k_w, c_o, s_h, s_w, padding="VALID", name='conv1') 
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool1_e1 =maxpool2d (conv1_e1, k_h, k_w,s_h, s_w,padding)
        variable_summaries(wc1)
        variable_summaries(bc1)
    with tf.variable_scope('conv2_eye_left'):
        k_h = 5; k_w = 5; c_o = 64; s_h = 1; s_w = 1;  
        wc2=tf.get_variable( 'weight2',shape = [5, 5, 64, 64],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)) 
        bc2=tf.get_variable(  'bias2',
          shape = [64],
          initializer=tf.constant_initializer(0.0))
        conv2_e1 = conv2d(maxpool1_e1, wc2, bc2, k_h, k_w, c_o, s_h, s_w, padding="SAME" , name='conv2')  # output is 14 14 64
        k_h = 3; k_w=3;s_w=2; s_h = 2;  padding = 'VALID'
        maxpool2_e1 =maxpool2d (conv2_e1, k_h, k_w,s_h, s_w,padding)
        variable_summaries(wc2)
        variable_summaries(bc2)
    with tf.variable_scope('conv3_eye_left'):
        k_h = 3; k_w = 3; c_o = 128; s_h = 1; s_w = 1 
        wc3= tf.get_variable( 'weight3',shape = [3,3, 64, 128],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)) 
        bc3=tf.get_variable(  'bias3',
          shape = [128],
          initializer=tf.constant_initializer(0.0))
        conv3_e1 = conv2d(maxpool2_e1, wc3,bc3,k_h, k_w, c_o, s_h, s_w, padding="SAME",name='conv3') 
        variable_summaries(wc3)
        variable_summaries(bc3)
    with tf.variable_scope('conv4_eye_left'):
        k_h = 1; k_w = 1; c_o = 64; s_h = 1; s_w = 1 
        wc4= tf.get_variable( 'weight4',shape = [1,1, 128, 64],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)) 
        bc4=tf.get_variable(  'bias4',
          shape = [64],
          initializer=tf.constant_initializer(0.0))
        conv4_e1 = conv2d(conv3_e1, wc4,bc4,k_h, k_w, c_o, s_h, s_w, padding="VALID",name='conv4')# output is 6 6 64
        variable_summaries(wc4)
        variable_summaries(bc4)
    with tf.variable_scope('f_c_left'):
        fc1_e = tf.reshape(conv4_e1, [-1, int(prod(conv4_e1.get_shape()[1:]))])  
        fc_e_w= tf.get_variable( 'weight_out',shape = [6*6*64, 2],
        initializer = tf.contrib.layers.xavier_initializer())  
        fc_e_b=tf.get_variable(  'out',
          shape = [2],
          initializer=tf.constant_initializer(0.0)) 
        fc1 = tf.add(tf.matmul(fc1_e, fc_e_w), fc_e_b )
    return fc1  

def four_path(eye_left, eye_right, face, face_mask):   
    with tf.variable_scope('conv1_eye_left'):
        k_h = 5; k_w = 5; c_o = 64; s_h = 2; s_w = 2
        wc1=tf.get_variable( 'weight1',shape = [5, 5,3, 64],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)) 
        bc1=tf.get_variable( 'bias1',
          shape = [64],
          initializer=tf.constant_initializer(0.0))
        conv1_e1 = conv2d(eye_left, wc1, bc1, k_h, k_w, c_o, s_h, s_w, padding="VALID", name='conv1') 
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool1_e1 =maxpool2d (conv1_e1, k_h, k_w,s_h, s_w,padding)
        variable_summaries(wc1)
        variable_summaries(bc1)
    with tf.variable_scope('conv2_eye_left'):
        k_h = 5; k_w = 5; c_o = 64; s_h = 1; s_w = 1;  
        wc2=tf.get_variable( 'weight2',shape = [5, 5, 64, 64],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)) 
        bc2=tf.get_variable(  'bias2',
          shape = [64],
          initializer=tf.constant_initializer(0.0))
        conv2_e1 = conv2d(maxpool1_e1, wc2, bc2, k_h, k_w, c_o, s_h, s_w, padding="SAME" , name='conv2')  # output is 14 14 64
        k_h = 3; k_w=3;s_w=2; s_h = 2;  padding = 'VALID'
        maxpool2_e1 =maxpool2d (conv2_e1, k_h, k_w,s_h, s_w,padding)
        variable_summaries(wc2)
        variable_summaries(bc2)
    with tf.variable_scope('conv3_eye_left'):
        k_h = 3; k_w = 3; c_o = 128; s_h = 1; s_w = 1 
        wc3= tf.get_variable( 'weight3',shape = [3,3, 64, 128],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)) 
        bc3=tf.get_variable(  'bias3',
          shape = [128],
          initializer=tf.constant_initializer(0.0))
        conv3_e1 = conv2d(maxpool2_e1, wc3,bc3,k_h, k_w, c_o, s_h, s_w, padding="SAME",name='conv3') 
        variable_summaries(wc3)
        variable_summaries(bc3)
    with tf.variable_scope('conv4_eye_left'):
        k_h = 1; k_w = 1; c_o = 64; s_h = 1; s_w = 1 
        wc4= tf.get_variable( 'weight4',shape = [1,1, 128, 64],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)) 
        bc4=tf.get_variable(  'bias4',
          shape = [64],
          initializer=tf.constant_initializer(0.0))
        conv4_e1 = conv2d_no_relu(conv3_e1, wc4,bc4,k_h, k_w, c_o, s_h, s_w, padding="VALID",name='conv4')# output is 6 6 64
        variable_summaries(wc4)
        variable_summaries(bc4)
    #eye_right 
    with tf.variable_scope('conv1_eye_right'):
        k_h = 5; k_w = 5; c_o = 64; s_h = 2; s_w = 2
        wc1=tf.get_variable( 'weight1',shape = [5, 5,3, 64],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)) 
        bc1=tf.get_variable( 'bias1',
          shape = [64],
          initializer=tf.constant_initializer(0.0))
        conv1_e1 = conv2d(eye_right, wc1, bc1, k_h, k_w, c_o, s_h, s_w, padding="VALID", name='conv1') 
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool1_e1 =maxpool2d (conv1_e1, k_h, k_w,s_h, s_w,padding)
        variable_summaries(wc1)
        variable_summaries(bc1)
    with tf.variable_scope('conv2_eye_right'):
        k_h = 5; k_w = 5; c_o = 64; s_h = 1; s_w = 1;  
        wc2=tf.get_variable( 'weight2',shape = [5, 5, 64, 64],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)) 
        bc2=tf.get_variable(  'bias2',
          shape = [64],
          initializer=tf.constant_initializer(0.0))
        conv2_e1 = conv2d(maxpool1_e1, wc2, bc2, k_h, k_w, c_o, s_h, s_w, padding="SAME" , name='conv2')   
        k_h = 3; k_w=3;s_w=2; s_h = 2;  padding = 'VALID'
        maxpool2_e1 =maxpool2d (conv2_e1, k_h, k_w,s_h, s_w,padding)
        variable_summaries(wc2)
        variable_summaries(bc2)
    with tf.variable_scope('conv3_eye_right'):
        k_h = 3; k_w = 3; c_o = 128; s_h = 1; s_w = 1 
        wc3= tf.get_variable( 'weight3',shape = [3,3, 64, 128],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)) 
        bc3=tf.get_variable(  'bias3',
          shape = [128],
          initializer=tf.constant_initializer(0.0))
        conv3_e1 = conv2d(maxpool2_e1, wc3,bc3,k_h, k_w, c_o, s_h, s_w, padding="SAME",name='conv3') 
        variable_summaries(wc3)
        variable_summaries(bc3)
    with tf.variable_scope('conv4_eye_right'):
        k_h = 1; k_w = 1; c_o = 64; s_h = 1; s_w = 1 
        wc4= tf.get_variable( 'weight4',shape = [1,1, 128, 64],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)) 
        bc4=tf.get_variable(  'bias4',
          shape = [64],
          initializer=tf.constant_initializer(0.0))
        conv4_e2 = conv2d_no_relu(conv3_e1, wc4,bc4,k_h, k_w, c_o, s_h, s_w, padding="VALID",name='conv4')# output is 6 6 64
        variable_summaries(wc4)
        variable_summaries(bc4)
    #eye
    with tf.variable_scope('concat_e'):
        conv_e=tf.concat([conv4_e1,conv4_e2],3)
        k_h = 1; k_w = 1; c_o = 128; s_h = 1; s_w = 1 
        fc_e=  tf.get_variable( 'weight9',shape = [6*6*128, 128],
        initializer = tf.contrib.layers.xavier_initializer()) 
        b_fc_e=tf.get_variable(  'bias9',
          shape = [128],                      
          initializer=tf.constant_initializer(0.0)) 
        fc_e1=tf.reshape(conv_e, [-1, int(prod(conv_e.get_shape()[1:]))]) 
        fc_e = tf.nn.relu(tf.add(tf.matmul(fc_e1,fc_e), b_fc_e))   
    # face
    with tf.variable_scope('conv1_face'):
        k_h = 5; k_w = 5; c_o = 64; s_h = 2; s_w = 2
        wc5= tf.get_variable( 'weight5',shape = [5, 5,3, 64],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32))
        bc5=tf.get_variable( 'bias5',
          shape = [64],
          initializer=tf.constant_initializer(0.0))  
        conv1_f = conv2d(face, wc5, bc5, k_h, k_w, c_o, s_h, s_w, padding="VALID", name='conv1') 
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool1_f =maxpool2d (conv1_f, k_h, k_w,s_h, s_w,padding)
    with tf.variable_scope('conv2_face'):
        k_h = 5; k_w = 5; c_o = 64; s_h = 1; s_w = 1;  
        wc6= tf.get_variable( 'weight6',shape = [5, 5, 64, 64],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32))  
        bc6=tf.get_variable(  'bias6',
          shape = [64],
          initializer=tf.constant_initializer(0.0))
        conv2_f = conv2d(maxpool1_f, wc6, bc6, k_h, k_w, c_o, s_h, s_w, padding="SAME" , name='conv2')  # output is 14 14 64
        k_h = 3; k_w=3;s_w=2; s_h = 2;  padding = 'VALID'
        maxpool2_f =maxpool2d (conv2_f, k_h, k_w,s_h, s_w,padding)
    with tf.variable_scope('conv3_face'):
        k_h = 3; k_w = 3; c_o = 128; s_h = 1; s_w = 1 
        wc7=tf.get_variable( 'weight7',shape = [3,3, 64, 128],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)) 
        bc7=tf.get_variable(  'bias7',
          shape = [128],
          initializer=tf.constant_initializer(0.0))
        conv3_f = conv2d(maxpool2_f, wc7,bc7,k_h, k_w, c_o, s_h, s_w, padding="SAME",name='conv3') 
    with tf.variable_scope('conv4_face'):
        k_h = 1; k_w = 1; c_o = 64; s_h = 1; s_w = 1 
        wc8= tf.get_variable( 'weight8',shape = [1,1, 128, 64],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)) 
        bc8=tf.get_variable(  'bias8',
          shape = [64],                      
          initializer=tf.constant_initializer(0.0))
        conv4_f = conv2d_no_relu(conv3_f, wc8,bc8,k_h, k_w, c_o, s_h, s_w, padding="VALID",name='conv4')# output is 6 6 64    
    with tf.variable_scope('fc_f1'):
        fc_f11=tf.reshape(conv4_f, [-1, int(prod(conv4_f.get_shape()[1:]))]) 
        fc_f1=tf.get_variable( 'weight10',shape = [6*6*64, 128],
        initializer = tf.contrib.layers.xavier_initializer())  
        b_fc_f1=tf.get_variable(  'bias10',
          shape = [128],                      
          initializer=tf.constant_initializer(0.0))
        fc_f1 = tf.nn.relu(tf.add(tf.matmul(fc_f11, fc_f1), b_fc_f1))
    with tf.variable_scope('fc_f2'):
        fc_f2=tf.get_variable( 'weight11',shape = [128 ,64],
        initializer = tf.contrib.layers.xavier_initializer())
        b_fc_f2=tf.get_variable(  'bias11',
          shape = [64],                      
          initializer=tf.constant_initializer(0.0))
        fc_f2 = tf.nn.relu(tf.add(tf.matmul(fc_f1, fc_f2),b_fc_f2))
    #face mask
    with tf.variable_scope('fg_f1'):
        fg_f1=tf.get_variable( 'weight12',shape = [25*25 , 256],
        initializer = tf.contrib.layers.xavier_initializer())
        b_fg_f1=tf.get_variable(  'bias12',
          shape = [256],initializer=tf.constant_initializer(0.0))
        fg_f11=tf.reshape(face_mask, [-1, int(prod(face_mask.get_shape()[1:]))]) 
        fg_f1 = tf.nn.relu(tf.add(tf.matmul(fg_f11, fg_f1), b_fg_f1))
    with tf.variable_scope('fg_f2'):
        fg_f2=tf.get_variable( 'weight13',shape = [256 ,128],
        initializer = tf.contrib.layers.xavier_initializer())
        b_fg_f2=tf.get_variable(  'bias13',
          shape = [128],                      
          initializer=tf.constant_initializer(0.0))
        fg_f22=tf.reshape(fg_f1, [-1, int(prod(fg_f1.get_shape()[1:]))]) 
        fg_f2 = tf.nn.relu(tf.add(tf.matmul(fg_f22,  fg_f2 ), b_fg_f2))
    # fc1
    with tf.variable_scope('fc1'):
        fc=tf.concat([fc_e,fc_f2,fg_f2],1)
        fcc=tf.reshape(fc, [-1, int(prod(fc.get_shape()[1:]))]) 
        fc1=tf.get_variable( 'weight14',shape = [320 ,128],
        initializer = tf.contrib.layers.xavier_initializer())
        b_fc1=tf.get_variable(  'bias14',
          shape = [128],                      
          initializer=tf.constant_initializer(0.0))
        fc1 = tf.nn.relu(tf.add(tf.matmul(fcc, fc1), b_fc1)) 
    with tf.variable_scope('fc2'):
        fc2=tf.get_variable( 'weight_out',shape = [128, 2],
        initializer = tf.contrib.layers.xavier_initializer())
        b_fc2=tf.get_variable(  'bias_out',
          shape = [2],                      
          initializer=tf.constant_initializer(0.0))
        fc1_1 =  tf.reshape(fc1, [-1, int(prod(fc1.get_shape()[1:]))]) 
        #fc1_1=tf.nn.dropout(fc1_1,keep_prob=keep_prob) #0.5 drop out
        fc2 = tf.add(tf.matmul(fc1_1, fc2), b_fc2)  
    return fc2 

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

def establish_model(train_eye_left,train_eye_right, train_face,train_face_mask, train_y ,                    val_eye_left,val_eye_right,val_face,val_face_mask, val_y):
    global batch_size, training_iters, learning_rate 
    with tf.name_scope('input'):
        eye_left = tf.placeholder(tf.float32, [None, 64,64,3])
        eye_right = tf.placeholder(tf.float32, [None, 64,64,3])
        face = tf.placeholder(tf.float32, [None, 64,64,3])
        face_mask = tf.placeholder(tf.float32, [None,25,25])
        y = tf.placeholder(tf.float32, [None,2])
    # Construct model
    #predict_op= one_path(eye_left) 
    predict_op=four_path(eye_left, eye_right, face, face_mask)
    step=1
    loss_list=[]
    train_rate=[]
    val_rate=[]
    # Define loss and optimizer 
    with tf.name_scope('loss'):
        cost = tf.losses.mean_squared_error(y, predict_op)/(2)  
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.8, momentum=0.0005,).minimize(cost)
    with tf.name_scope('err'):
        err=tf.reduce_mean(tf.sqrt(tf.reduce_sum((predict_op-y)**2,axis=1)))
    
    tf.summary.scalar('err',err)
    tf.summary.scalar('loss',cost)
    
    #save
    saver = tf.train.Saver()
    # Launch the graph
    sess = tf.InteractiveSession()
    # Merge all the summaries and write them out to C:/Users/Lab/
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('C:/Users/Lab/' + '/train',
                                      sess.graph)
    test_writer = tf.summary.FileWriter('C:/Users/Lab/' + '/test') 
    train_writer.add_graph(sess.graph)
    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)
    while step < training_iters:
        ind = np.arange(train_eye_left.shape[0])
        batch_idx = np.random.choice(ind, batch_size, replace=False)
        batch_e1 = train_eye_left[batch_idx]
        batch_e2 = train_eye_right[batch_idx]
        batch_f = train_face[batch_idx] 
        batch_fg = train_face_mask[batch_idx]                    
        batch_y= train_y[batch_idx]
        batch_e1=normalize(batch_e1)
        batch_e2=normalize(batch_e2)
        batch_f=normalize(batch_f)                    
        indtest = np.arange(val_eye_left.shape[0])
        test_idx = np.random.choice(indtest, batch_size, replace=False)
        batch_e1_val = val_eye_left[test_idx]
        batch_e2_val = val_eye_right[test_idx]
        batch_f_val = val_face[test_idx] 
        batch_fg_val = val_face_mask[test_idx]                    
        batch_y_val= val_y[test_idx]
        batch_e1_val=normalize(batch_e1_val)
        batch_e2_val=normalize(batch_e2_val)
        batch_f_val=normalize(batch_f_val) 
        loss, train_err,summary = sess.run([cost, err,merged], feed_dict={eye_left: batch_e1,eye_right:batch_e2,face:batch_f,face_mask:batch_fg, y: batch_y   }) 
        val_err=sess.run(err, feed_dict={eye_left: batch_e1_val, eye_right:batch_e2_val, face:batch_f_val,face_mask:batch_fg_val, y: batch_y_val   })          
        train_writer.add_summary(summary, step)
        loss_list.append(loss)
        train_rate.append(train_err)
        val_rate.append(val_err)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={eye_left: batch_e1,eye_right:batch_e2,face:batch_f,face_mask:batch_fg, y: batch_y  })    
        if step % display_step == 0:
            # Calculate batch loss and err
            print("Iter " + str(step ) + ", Minibatch Loss= " +  "{:.6f}".format(loss) + ", testing err= " + "{:.5f}".format(val_err)+ ", training err= " + \
                  "{:.5f}".format(train_err))
        step += 1
    val_eye_left=normalize(val_eye_left)
    val_eye_right=normalize(val_eye_right) 
    val_face=normalize(val_face) 
    final_err=sess.run(err, feed_dict={eye_left:val_eye_left,eye_right:val_eye_right,face:val_face, face_mask: val_face_mask,y:val_y })  
    print("Testing err:", final_err)    
    print("Optimization Finished!")
    # Create the collection.
    tf.get_collection("validation_nodes")
    # Add stuff to the collection.
    tf.add_to_collection("validation_nodes",eye_left)
    tf.add_to_collection("validation_nodes",eye_right)
    tf.add_to_collection("validation_nodes",face)
    tf.add_to_collection("validation_nodes",face_mask)
    tf.add_to_collection("validation_nodes", predict_op) 
    save_path = saver.save(sess, "C:/Users/Lab/my_model")  
    return loss_list,step,train_rate,val_rate 

def plot_loss(loss,train_step,from_second,name_save, plot_name,plot_title):
    if from_second :
        plt.plot(range(0,train_step-1,1),loss[1:])
    else:
        plt.plot(range(0,train_step,1),loss[0:])
    plt.xlabel('Iterative times (t)')
    plt.ylabel(plot_name)
    plt.title(plot_title)
    plt.grid(True)
    plt.savefig(name_save)
    plt.show() 

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

def main():
    loss_list,step,train_rate,val_rate =establish_model(train_eye_left,train_eye_right, train_face,train_face_mask, train_y ,                    val_eye_left,val_eye_right,val_face,val_face_mask, val_y)
    plot_loss(loss_list,step-1,False,"Loss_value.png", 'Loss','Loss function value with iterations') 
    plot_loss(train_rate,step-1,False,"Train_err_rate.png",'Training accurate rate','Training classification with iterations')
    plot_loss(val_rate,step-1,False,"Val_err_rate.png",'Validating accurate rate','Validating classification with iterations')
    show_graph(tf.get_default_graph().as_graph_def())



if __name__=='__main__':
    main()




