### conclusion of labs

#### model
- cnn-lstm
- wav2vec2 series

#### datasets
- TALSER
- IEMOCAP

#### train
- SER(speech emotion recognition)
  - classes
  - PA regressive  
- SV(speaker verification)

#### SER


**talser**

|        model        | datasets  | attention | Epoch |  acc  |  f1   | FP8e5m2 | FP8e4m3 |
| :-----------------: | :-------: | :-------: | :---: | :---: | :---: | :-----: | :-----: |
|         SVM         |   mfcc    |   adam    |   1   |   1   |   1   |    1    |    1    |
|      CNN+LSTM       |   mfcc    |   adam    |   1   |   1   |   1   |    1    |    1    |
|      CNN+LSTM       |   mfcc    |   adam    |   1   |   1   |   1   |    1    |    1    |
| wav2vec2-large-960h | raw-audio |   adam    |   8   |   5   |   8   |    5    |    4    |
|    wav2vec2-base    | raw-audio |   adam    |  10   |  10   |   7   |    2    |    3    |

> ENV：

#### Speaker Verify
|        model        | datasets | attention |  Acc  | FP16  | BF16  | FP8e5m2 | FP8e4m3 |
| :-----------------: | :------: | :-------: | :---: | :---: | :---: | :-----: | :-----: |
|      CNN+LSTM       |   mfcc   |   adam    |   1   |   1   |   1   |    1    |    1    |
|      CNN+LSTM       |   mfcc   |   adam    |   1   |   1   |   1   |    1    |    1    |
| wav2vec2-large-960h |  talser  |   adam    |   8   |   5   |   8   |    5    |    4    |
|    wav2vec2-base    |  talser  |   adam    |  10   |  10   |   7   |    2    |    3    |

> ENV：

###  部署
#### 脚本部署
##### 分析脚本
##### 数据可视化脚本
#### TensorRT C++？

### 教学行为分析
~~可视化？~~

