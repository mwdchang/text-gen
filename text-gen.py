from mlc import MLC
from lstm import CLSTM
import random

filename = "input2.txt"



def printText(title, text): 
  print("\033[33m" + title)
  print("=" * 40)
  print("\033[39m" + text)
  print("")


if __name__ == "__main__":
  print("Preparing models")
  # For seeding
  data = open(filename).read().lower() 
  #words = []
  #for line in data.split("\n"):
  #  for w in line.split(" "):
  #    words.append(w)
  
  
  mlc2 = MLC(order=2)
  mlc2.train_fulltext(filename)
  
  mlc6 = MLC(order=6)
  mlc6.train_fulltext(filename)
  
  lstm = CLSTM(maxlen=40, step=3)
  lstm.setInput(filename)

  while True:
    print("\033[39m")
    cmd = input("\033[35mEnter one of the following (train | sample) > ")
    print("\033[39m")
    if cmd == "train":
      print("Training LSTM")
      lstm.train()
    elif cmd == "sample":
      idx = random.randint(0, len(data) - 20)
      mlcSeed = data[idx:idx + 10]

      mlc2Sample = mlc2.generate_text(history=mlcSeed)
      mlc6Sample = mlc6.generate_text(history=mlcSeed)
      lstmSample = lstm.generate_text(1.0)

      printText("MLC Sampling - Order " + str(mlc2.order), mlc2Sample)
      printText("MLC Sampling - Order " + str(mlc6.order), mlc6Sample)
      printText("LSTM Sampling - Epoch " + str(lstm.epoch), lstmSample)
    else:
      print("Unknown command")
