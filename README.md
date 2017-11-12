### LSTM and MLC
Playing around with LSTM text generation versus MLC text generation, in an interactive manner.

#### LSTM
Long-short-term-memory model using Keras with Tensorflow


#### MLC
Maximum-likelihood character model, built with simple probability maps

### Run
Run `python text-gen.py`. The MLC models will be trained automatically, while LSTM can be trained epoch-by-epoch by issuing the *train* command.

- train: Further train LSTM
- sample: Sample text generation, sampling is seeded by random character segments from input text
- `Ctrl-c`: exit

#### Dependencies
Main library dependencies
- Tensorflow `pip install tensorflow`
- Keras `pip install keras`
- Numpy `pip install numpy`
