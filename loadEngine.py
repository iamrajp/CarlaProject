#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:06:20 2018

@author: priyanksharma
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 21:25:57 2017
@author: priyanksharma
"""

#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import socket
import threading
import cv2
import os
import shutil
import pickle
import time

#%%
IMG_SIZE = 50
train_data = np.load('train_data.npy')

X_train = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array([i[1] for i in train_data])

#%%

# loaing learned data
tf.reset_default_graph()
sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph('./savedmodel/model/my-model-final.meta')
saver.restore(sess, tf.train.latest_checkpoint('./savedmodel/model/'))
graph = tf.get_default_graph()

#%%

index= 370

x_input = graph.get_tensor_by_name("x_input:0")
predict = tf.argmax(graph.get_tensor_by_name("output/fc3/Relu:0"),1)
feed_dict={x_input:np.reshape(X_train[index],[-1,50,50,1]),"keep_prob:0":1.0}
print(predict.eval(feed_dict))
#plt.imshow(np.reshape(X_train[index],[50,50]))

#%%

# Calculate accuracy

x_input = graph.get_tensor_by_name("x_input:0")
y_input = graph.get_tensor_by_name("y_input:0")
accuracy = graph.get_tensor_by_name("accuracy/Mean:0")
print("Test Accuracy",accuracy.eval(feed_dict={x_input:X_train,y_input:y_train,"keep_prob:0":1.0}))





#%%

    
imgnum = 1

trainmode = False

def processImage(s):
    
        global imgnum
        
        startime = time.time()
        
        while True:
             
            
            response = ["F","L","R"]
            filedata = b''
            data = s.recv(1024)
            filedata = data
          
            
            while not b'<end>' in filedata:
                
                data = s.recv(1024)
                filedata = filedata+data
                 
        
            
           
            if trainmode:
              
               
               folder = filedata[-1:].decode()
               print("Saving image {0}".format(folder))
               img_data =  pickle.loads(filedata[:-6],fix_imports=True, encoding="bytes", errors="strict") #ignoring <end tag>
               cv2.imwrite("./{0}/track1_{1}.jpg".format(folder,imgnum),img_data)
               s.send(b'F')
               
            else:
               
               img_data =  pickle.loads(filedata[:-5],fix_imports=True, encoding="bytes", errors="strict") #ignoring <end tag>
               x_input = graph.get_tensor_by_name("x_input:0")
               predict = tf.argmax(graph.get_tensor_by_name("output/fc3/Relu:0"),1)
               direction= predict.eval(session=sess,feed_dict={x_input:np.reshape(img_data,[-1,50,50,1]),"keep_prob:0":1.0})
               print(response[direction[0]])
               s.send(response[direction[0]].encode())
               savethread = threading.Thread(target=SaveImage,args=(response[direction[0]],img_data,imgnum))
               savethread.start()
               
               
            
              
            if imgnum % 10 == 0:
               print("Time taken to process {0}  images {1}".format(imgnum,time.time()-startime))
               starttime = time.time()
            
            imgnum = imgnum +1
    
       
                


def SaveImage(folder,img_data,imgnum):
    cv2.imwrite("./{0}/track1_{1}.jpg".format(folder,imgnum),img_data)
    
    
     

def Main():
      
    host = '192.168.1.6'
    port = 9004
    s = socket.socket()
    s.bind((host,port))
    s.listen(5)
    
    try :
         
    
          print("Car Engine Started.....")
    
          while True:
            c,addr = s.accept()
            print("client connected ip:<"+str(addr)+">")
            t = threading.Thread(target=processImage,args=(c,))
            t.start()
            
          s.close()
    
    except:
         print("error")    
    finally:
         s.close()
        
        
        
  

if __name__ == "__main__":
    
    Main()                   
                




