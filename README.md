# Translating-Dates-With-Attention-Model

This repository will implement a model to translate human readable dates **("25th of June, 2009")** into machine readable dates **("2009-06-25")** based on attention model.

## How to start?  
git clone to your local computer and run "**python -m Neural_Machine_Translation.py**"on your command line    
I have commeted some plot function in the Neural_Machine_Translation.py. If you want to see some detailed plots such as **attention map**, you could run with the jupyter notebook **Neural_Machine_Translation.ipynb**

## How it works
For the training set we have many pairs of human readable dates and machine readable dates. To feed them into the LSTM, we have to transform every character in the dates to certain integer. Then we feed them into a bidirectional LSTM  
<img src='https://github.com/mingbocui/Translating-Dates-With-Attention-Model/blob/master/Machine%20Translation.PNG'>  


- There are two separate LSTMs in this model (see diagram above). Because the one at the bottom of the picture is a Bi-directional LSTM and comes *before* the attention mechanism, we will call it *pre-attention* Bi-LSTM. The LSTM at the top of the diagram comes *after* the attention mechanism, so we will call it the *post-attention* LSTM. The pre-attention Bi-LSTM goes through Tx time steps; the post-attention LSTM goes through Ty time steps, here Tx represents the maximum length of human readable dates, and Ty represents the machine readable dates, generally Ty=10 because the length of "MM-DD-YYYY" is equal to 10. 

- The post-attention LSTM passes activatation and hidden states, **s_t and c_t**, from one time step to the next. We are using an LSTM here, the LSTM has both the output activation s_t and the hidden cell state c_t. In this model the post-activation LSTM at time t does will not take the specific generated y_(t-1) as input; it only takes **s_t and c_t** as input. ** Because (unlike language generation where adjacent characters are highly correlated) there isn't as strong a dependency between the previous character and the next character in a YYYY-MM-DD date.**  

- We use vector a(see graph above) to represent the concatenation of the activations of both the forward-direction and backward-directions of the pre-attention Bi-LSTM. 

- The diagram on the right uses a `RepeatVector` node to copy s_(t-1)'s value Tx times, and then `Concatenation` to concatenate s_(t-1) and a_t to compute e_(t,t'), which is then passed through a softmax to compute alpha(t,t').   

## Attention Maps
