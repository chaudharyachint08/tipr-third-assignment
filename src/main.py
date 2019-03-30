import os,sys,shutil,datetime,pickle,codecs,tempfile,gzip

import numpy as np
import scipy as sp
import pandas as pd
import cv2

import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn, skimage
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import accuracy_score, f1_score

import tensorflow.keras
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D

def unpickle(file,is_bytes=True):
  with open(file, 'rb') as fp:
    dict = pickle.load(fp, encoding='bytes')
  return dict
    
def load_mnist(path, kind='train'):
  'Load MNIST data from `path'
  labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
  images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

  with gzip.open(labels_path, 'rb') as lbpath:
    labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

  with gzip.open(images_path, 'rb') as imgpath:
    images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

  return images, labels

from tensorflow.keras.utils import get_custom_objects

def swish2(x):
  return x*K.sigmoid(x)

def addswish():
  get_custom_objects().update({'swish': Activation(swish2)})

def one_hot(y):
    return 1*(y==y.max(axis=1,keepdims=True))

def get_metrices(y_act,y_pred):
  if len(y_pred.shape)==2:
    y_pred = np.argmax( one_hot(y_pred), axis=1 )
  elif len(y_pred.shape)==1:
    y_pred = np.round(y_pred)
  else:
    raise Exception('Multi dimensional labels')
  y_act = np.argmax( y_act, axis=1 ) if len(y_act.shape)==2 else y_act
  return accuracy_score(y_act,y_pred), f1_score(y_act,y_pred,average='macro'), f1_score(y_act,y_pred,average='micro')
  

def SSplit(X,Y,K=10,seed=True):
    'Stratified Split Function'
    if seed:
        np.random.seed(42)
    Y = pd.DataFrame([tuple(y) for y in Y])
    classes = set(Y)
    c2i = {}
    for index,label in Y.iterrows():
        label = label[0]
        if label in c2i:
            c2i[label].add(index)
        else:
            c2i[label] = {index}
    
    # Each class -> list of indices
    for i in c2i:
        c2i[i] = list(c2i[i])
        np.random.shuffle(c2i[i])
    
    # Each class with its set of train, test split indices
    c2is = {}
    for cls in c2i:
        a = int(np.round(len(c2i[cls])/K))
        c2is[cls] = []
        for fold in range(K):
            test_indices  = c2i[cls][a*fold:a*(fold+1)]
            train_indices = c2i[cls][0:a*fold] + c2i[cls][a*(fold+1):]
            c2is[cls].append((train_indices,test_indices))
        np.random.shuffle(c2is[cls])
        
    index_set = []
    for i in range(K):
        train,test = set(),set()
        for cls in c2is:
            _ = c2is[cls][i]
            train.update(set(_[0]))
            test.update (set(_[1]))
        index_set.append((list(train),list(test)))
    return index_set

class CNN:
  def __init__(self,name='dataset_name', init_method='glorot_normal'):
    self.name = name
    self.model = Sequential()
    self.init_method = init_method
    addswish()
  
  def data_feed( self, M, L, targets):
    self.raw, self.labels, self.target_names = M, L, targets

  def data_validate( self, M=np.array([]), L=np.array([]) ):
    self.vraw, self.vlabels = M, L    

  def data_preprocess(self,mode='scale'):
    sp = np.nan_to_num
    self.preprocess_mode = mode
    if mode=='scale':
      try:
        self.mn, self.mx
      except:
        self.mn, self.mx = self.raw.min(axis=0), self.raw.max(axis=0)
      mx = np.where(self.mx==self.mn,self.mx+1,self.mx)
      self.data  = sp((self.raw - self.mn)/(mx-self.mn))
      try: # If validation data is defined
        self.vdata = sp((self.vraw - self.mn)/(self.mx-self.mn))
      except:
        self.vdata = self.data
    elif mode=='standard':
      try:
        self.mean, self.std
      except:
        self.mean, self.std   = self.raw.mean(axis=0), self.raw.std(axis=0)
      std = np.where(self.std==0,1,self.std)
      self.data = sp((self.raw-self.mean)/std)
      try: # If validation data is defined
        self.vdata  =  sp((self.vraw-self.mean)/std)
      except:
        self.vdata = self.data
    else:
      raise Exception('Code should be unreachable')
    self.data, self.vdata = self.data.astype('float32'), self.vdata.astype('float32')
    try:
      self.vlabels
    except:
      self.vlabels = self.labels

  def add_CONVs(self,FN=[10,], KS =[(3,3),], KSS = [(1,1),], PAD=['same',], ACT=['relu',], PS=[(2,2),], PSS=[(1,1),], BN = [True,], DROP=[0.5,]):
    for i in range(len(ACT)):
      if i==0:
        self.model.add( Conv2D(filters=FN[i], kernel_size=KS[i], strides=KSS[i], padding=PAD[i],kernel_initializer = self.init_method,
                             bias_initializer = self.init_method, input_shape=self.data.shape[1:]) )
      else:
        self.model.add( Conv2D(filters=FN[i], kernel_size=KS[i], strides=KSS[i], padding=PAD[i],kernel_initializer = self.init_method,
                             bias_initializer = self.init_method) )
      self.model.add( Activation(ACT[i]) )
      if BN[i]:
        self.model.add( BatchNormalization() )
      if PS[i]!=(1,1):
        self.model.add( MaxPooling2D(pool_size=PS[i],strides=PSS[i],data_format='channels_last') )
      if DROP[i]:
        self.model.add( Dropout(rate=DROP[i]) )

  def add_FCs(self,UN=[10,], ACT=['relu',], BN = [True,], DROP=[0.5,]):
    self.model.add(Flatten())
    for i in range(len(ACT)):
      if not any(('Conv' in str(x)) for x in self.model.layers):
        if i==0:
          self.model.add( Dense(units=UN[i], kernel_initializer = self.init_method, bias_initializer = self.init_method), input_shape=int(round(np.exp(np.sum(np.log(self.data.shape[1:]))))) )
        else:
          self.model.add( Dense(units=UN[i], kernel_initializer = self.init_method, bias_initializer = self.init_method) )
      if BN[i]:
        self.model.add( BatchNormalization() )
      self.model.add( Activation(ACT[i]) )
      if DROP[i]:
        self.model.add( Dropout(rate=DROP[i]) )
    self.model.add(Dense( len(self.target_names) ))
    self.model.add(Activation('softmax'))

  def optimizer(self,name='Adam',rate=0.001,decay=1e-6):
    if name=='Adam':
      exec( 'self.opt = tensorflow.keras.optimizers.{}(lr={},decay={})'.format(name,rate,decay) )
    elif name=='Nadam':
      exec( 'self.opt = tensorflow.keras.optimizers.{}(lr={},schedule_decay={})'.format(name,rate,decay) )
    

  def compile(self,loss='categorical_crossentropy'):
    self.model.compile(loss=loss, optimizer=self.opt, metrics=['accuracy'])

  def train(self,epochs=10, batch_size = 100):
    self.model.fit( self.data, self.labels, epochs=epochs, batch_size=batch_size, validation_data=(self.vdata, self.vlabels), shuffle=True )

  def evaluate(self,data,labels):
    y_pred = self.model.predict(data,batch_size=300,verbose=True)
    return get_metrices(labels,y_pred)

  def save_model(self):
    global save_dir, model_path
    try:
      save_dir
    except:
      save_dir = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
    model_path = os.path.join(save_dir, self.name)
    self.model.save(model_path)
    print('Saved "{}" trained model at {}'.format(self.name,model_path))
    dct = {}
    with open(model_path+'-scale','wb') as f:
      if self.preprocess_mode == 'standard':
        dct['mean'], dct['std'] = self.mean, self.std
      elif self.preprocess_mode == 'scale':
        dct['mn'], dct['mx'] = self.mn, self.mx
      else:
        raise Exception('Code should be unreachable')
      np.savez(f,**dct)

  def load_model(self):
    self.model = load_model( model_path , custom_objects=get_custom_objects())
    with open(model_path+'-scale','rb') as f:
      npzfile = np.load(f)
      if ('mean' in npzfile) and ('std' in npzfile):
        self.mean, self.std = npzfile['mean'], npzfile['std']
      elif ('mx' in npzfile) and ('mn' in npzfile):
        self.mn, self.mx = npzfile['mn'], npzfile['mx']
      else:
        raise Exception('Code should be unreachable')  

  def set_vectorial(self,batch_size = 100):
    X, X2, flg, _ = None, None, False, list(map(str,self.model.layers))[::-1]
    for lyr in range(len(_)):
      if 'Activation' in _[lyr]:
        if flg:
          break
        flg = True
    lyr = -1*(lyr+1)
    repr_fun = K.function([self.model.layers[0].input], [self.model.layers[lyr].output])
    for i in range(int(np.ceil(len(self.data)/batch_size))):
      tmp = repr_fun([self.data[i*batch_size:(i+1)*batch_size]])[0]
      X = pd.concat( (X,pd.DataFrame(tmp)) ) if X is not None else pd.DataFrame(tmp)
    self.vectorial_repr = np.array(X)
    for i in range(int(np.ceil(len(self.vdata)/batch_size))):
      tmp = repr_fun([self.vdata[i*batch_size:(i+1)*batch_size]])[0]
      X2 = pd.concat( (X2,pd.DataFrame(tmp)) ) if X2 is not None else pd.DataFrame(tmp)
    self.vvectorial_repr = np.array(X2)

  def clustering(self):
    kmeans_obj = KMeans(n_clusters=len(self.target_names)).fit(self.vectorial_repr)
    y_true, y_now = np.argmax(self.labels,axis=1), kmeans_obj.labels_

    lbl2indx = {}
    for i in range(len(self.vectorial_repr)):
      if y_now[i] not in lbl2indx:
        lbl2indx[ y_now[i] ] = {i}
      else:
        lbl2indx[ y_now[i] ].add(i)

    clustermajority = {}
    for lbl in lbl2indx:
      tmp = {}
      for i in lbl2indx[lbl]:
        if y_true[i] not in tmp:
          tmp[y_true[i]] =  1
        else:
          tmp[y_true[i]] += 1
      for i in range(len(self.target_names)):
        if i not in tmp:
          tmp[i] = 0
      clustermajority[lbl] = tmp

    remaining_labels = set(list(range(len(self.target_names))))
    self.lbl2pred = {}
    for _ in self.target_names:
      assgn_lbl = max( set(lbl2indx.keys())-set(self.lbl2pred.keys()),key=lambda lbl:clustermajority[lbl][max(clustermajority[lbl],key=lambda x: clustermajority[lbl][x] if (x in remaining_labels) else 0 )] )
      pred_freq = sorted(clustermajority[assgn_lbl].items(),key=lambda x:x[1],reverse=True)
      for act_lbl, true_freq in pred_freq:
        if act_lbl in remaining_labels:
          self.lbl2pred[assgn_lbl] = act_lbl
          remaining_labels.remove(act_lbl)
          break
      else:
        print(assgn_lbl , remaining_labels, pred_freq)
        raise Exception('No remaining Label, should not happen')

    def accuracy(self,data,labels):
      y_act  = np.argmax(labels, axis=1) if len(labels.shape)>1 else labels
      y_pred = np.array( [ self.lbl2pred[x] for x in kmeans_obj.predict(data) ] )
      matches = np.sum(y_act==y_pred)
      return matches/len(y_act)

    self.cluster_accuracy  = accuracy(self,self.vectorial_repr,  self.labels)  *100
    self.vcluster_accuracy = accuracy(self,self.vvectorial_repr, self.vlabels) *100

  def VisTSNE(self,X=None,Y=None,target_names=None):
    if X is None:
      X, Y, target_names = np.array(list(self.vectorial_repr)+list(self.vvectorial_repr)), np.array(list(self.labels)+list(self.vlabels)), self.target_names
    plt.close()
    Y2 =  np.argmax(Y,axis=1) if len(Y.shape)>1 else Y
    Y2_set = set(Y2)
    X2 = TSNE(n_iter=250).fit(X)
    for y_now in Y2_set:
      x_now = X2[ [i for i in range(X2.shape[0]) if Y2[i]==y_now] ].T
      _ = plt.scatter(x_now[0],x_now[1],label=str(target_names[y_now]),s=2)
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.axis('off')
    title = 't-SNE {0}({1:0.1f}%) - Labeled_ACC={2:0.2f}% Unlabeled_ACC={3:0.2f}%'.format(self.name,round(100*len(self.vectorial_repr)/len(datasets[self.name][0]),2),
    round(self.cluster_accuracy,2),round(self.vcluster_accuracy,2))
    plt.title(title)
    plt.savefig(title+'.png',dpi=300,transparent=True,bbox_inches='tight')
    plt.show()

def wrapper(name='dataset_name',preproc='standard',UN=[4096,4096],activation='relu',init='glorot_normal',epochs=10,batch_size=300,
            optimizer='Adam',lr=0.001,decay=1e-4,data=None,labels=None,vdata=None,vlabels=None,ext=False,cnn=True,fraction=0.8):
  
  cnn = CNN(name,init)
  if data is None:
    _ = tensorflow.keras.utils.to_categorical(datasets[name][1], len(datasets[name][2]))
    index_set = SSplit(datasets[name][0],_,len(datasets[name][2]),seed=42)
    train_index = set(v for tmp in index_set[:round(fraction*10)] for v in tmp[1])
    test_index  = set(range(len(datasets[name][0])))-train_index
    train_index, test_index = list(train_index), list(test_index)
    tmp_x_train, tmp_x_test = datasets[name][0][train_index], datasets[name][0][test_index]
    tmp_y_train, tmp_y_test = datasets[name][1][train_index], datasets[name][1][test_index]

  cnn.data_feed(tmp_x_train,    tmp_y_train,datasets[name][2]) if data is None else cnn.data_feed(data,labels,datasets[name][2])
  cnn.data_validate(tmp_x_test, tmp_y_test,)                   if vdata is None else cnn.data_validate(vdata,vlabels)
  # Convert class vectors to binary class matrices.
  num_classes = len(set(np.ravel(cnn.labels))|set(np.ravel(cnn.vlabels)))
  cnn.labels  = tensorflow.keras.utils.to_categorical(cnn.labels,  num_classes)
  cnn.vlabels = tensorflow.keras.utils.to_categorical(cnn.vlabels, num_classes)
  cnn.data_preprocess(preproc)
  if cnn:
    cnn.add_CONVs( FN=[96,256,384,384,256], KS =[(11,11),(5,5),(3,3),(3,3),(3,3)], KSS = [(4,4),(1,1),(1,1),(1,1),(1,1)], PAD=['same','same','same','same','same'],
               ACT=[activation]*5, PS=[(3,3),(3,3),(1,1),(1,1),(1,1)], PSS = [(2,2),(2,2),(1,1),(1,1),(1,1)], BN = [True,False,False,False,False], DROP=[0.0,0.0,0.2,0.0,0.2] )
  cnn.add_FCs( UN, ACT=[activation]*len(UN), BN = [False]*len(UN), DROP=[0.5,]*len(UN))
  cnn.optimizer(optimizer,lr,decay)
  cnn.compile()
  cnn.train(epochs,batch_size)
  res = cnn.evaluate(cnn.data,cnn.labels)
  res2 = cnn.evaluate(cnn.vdata,cnn.vlabels)
  res = res+(res2[0],)
  if ext:
    print('Embedding > ',end='') ;  cnn.set_vectorial()
    print('Clustering > ',end='') ; cnn.clustering()
    print('VisTSNE') ;    cnn.VisTSNE()
  return cnn, res



try:
  datasets
except:
  datasets = {}
else:
  datasets = {}

def add_dataset(data_path,name='dataset_name',test=0):
  if name=='CIFAR-10':
    X,Y,target_names = None, None, None
    for file_name in os.listdir(data_path):
      if ('test_batch' if test==1 else 'data_batch') in file_name:
        temp = unpickle( os.path.join(data_path,file_name) )
        X = pd.concat( (pd.DataFrame(temp[b'data']),  X) ) if X is not None else pd.DataFrame( temp[b'data']   )
        Y = pd.concat( (pd.DataFrame(temp[b'labels']),Y) ) if Y is not None else pd.DataFrame( temp[b'labels'] )
      elif 'batches' in file_name:
        temp = unpickle( '/'.join((data_path,file_name)) )
        target_names = temp[b'label_names']
      
    X = np.array( [np.array(x).reshape((32,32,3)) for i,x in X.iterrows()] )
    Y = np.array( Y )
    datasets['CIFAR-10'] = (X,Y,[x.decode('ascii') for x in target_names])
    
  elif name=='Fashion-MNIST':
    X, Y = map(pd.DataFrame,load_mnist(data_path, kind='t10k')) if test==1 else map(pd.DataFrame,load_mnist(data_path, kind='train'))
    X = np.array( [np.array([np.array(x).reshape((28,28)),np.array(x).reshape((28,28)),np.array(x).reshape((28,28))]) for i,x in X.iterrows()] )
    Y = np.array( Y )
    datasets['Fashion-MNIST'] = (np.array([np.array(x).reshape((28,28,3)) for x in X]), np.array(Y), ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])




argv = list(sys.argv)

name = argv[argv.index('--dataset')+1]
if '--train-data' in argv:
  save_dir = os.path.join(os.getcwd(),'saved_models') # Remove Later
  model_path = os.path.join(save_dir, name)           # Remove Later
  activation = argv[argv.index('--activation')+1] if '--activation' in argv else 'relu'
  train_path = argv[argv.index('--train-data')+1]
  config   = []
  for st in argv[argv.index('--configuration')+1:]:
    st  = st.strip()
    if st.endswith(']'):
      config.append(int(st.strip('[]')))
      break
    else:
      config.append(int(st.strip('[]')))
  add_dataset(data_path=train_path,name=name,test=0)
  cnn,res = wrapper(name,'standard',config,activation,'glorot_normal',50,300,'Adam',ext=False)
  cnn.save_model()                                    # Remove Later
else:
  save_dir = os.path.join(os.getcwd(),'saved_models')
  model_path = os.path.join(save_dir, name)
  cnn = CNN(name)
  cnn.load_model()

test_path = argv[argv.index('--test-data')+1]
add_dataset(data_path=test_path,name=name,test=1)
cnn.data_feed(*datasets[name])
cnn.data_preprocess('standard')
num_classes = 10 # len(set(np.ravel(datasets[name][1])))
cnn.labels  = tensorflow.keras.utils.to_categorical(cnn.labels, num_classes)
cnn.vlabels = tensorflow.keras.utils.to_categorical(cnn.vlabels, num_classes)

_ = cnn.evaluate(cnn.data,cnn.labels)
print('Test Accuracy :: {0:.2f}'.format(round(_[0]*100,2)))
print('Test Macro-F1 Score :: {0:.2f}'.format(round(_[1]*100,2)))
print('Test Micro-F1 Score :: {0:.2f}'.format(round(_[2]*100,2)))

