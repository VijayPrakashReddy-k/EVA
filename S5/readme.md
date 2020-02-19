# 1.First CODE :
   We started from a network from the previous assignment where the problem is having the large number of parameters
          
            Batch Normalisation,
            DropOut,
            and with Batch Size = 128,
            and also Optimizer as Stochastic gradient descent(SGD)\
            
  with that we got,
  
          Total params: 40688
          Training accuracy:99.55
          Test accuracy:99.54
 
 # 2.Second CODE :
 
   In second code I made Changes,
   
          Making the model lighther by introducing the 1*1 convolution and reduced the number of kernels
  
  with that i got,
  
          Total params: 9,562
          Training accuracy:98.45
          Test accuracy:99.20
 
 # 3.Third CODE :
 
   In Third code I made Changes as,
   
            Tweaking the Dropout
            
  with that i got,
  
          Total params: 9,562
          Training accuracy:99.07
          Test accuracy:99.35
 
 # 4.Forth CODE :
 
   In Forth code I made Changes as,
   
          Introduced the rotation
          Total params: 9,562
          Training accuracy:98.91
          Test accuracy:99.45
          
  # 5.Fifth CODE :
   
   In Fifth code I made Changes as,
   
          Introduced LR Scheduler
          Total params: 9,562
          Training accuracy:98.89
          Test accuracy:99.45

At the end of 5 code iterations achieved:

***Nearer to 99.4% accuracy consistently
Less than 10k Parameters***
