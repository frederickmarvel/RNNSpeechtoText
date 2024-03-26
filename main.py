import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from jiwer import wer
from IPython import display
import matplotlib.pyplot as plt

# dataUrl = "skripsi/LJSpeech-1.1.tar.bz2"
dataPath = "LJSpeech-1.1"
recPath = dataPath + "/wavs/"
metadataPath = dataPath + "/metadata.csv"
metadataDF = pd.read_csv(metadataPath, sep="|", header=None, quoting=3)
metadataDF.tail()
metadataDF.columns = ["fileName", "transcription", "normalizedTranscription"]
newMetadataDF = metadataDF[["fileName", "transcription"]]
newMetadataDF = newMetadataDF.sample(frac=1).reset_index(drop=True)
newMetadataDF.head()

split = int(len(newMetadataDF) * 0.9 )
df_train = newMetadataDF[:split]
df_val = metadataDF[split:]

print(len(df_train))
print(len(df_val))

chars = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
charToNum = keras.layers.StringLookup(vocabulary=chars, oov_token="")
numToChar = keras.layers.StringLookup(vocabulary=charToNum.get_vocabulary(), oov_token= "", invert=True)

print(charToNum.get_vocabulary())
print(charToNum.vocabulary_size())

frameLength = 256
frameStep = 160
fftLength = 384

def encode_sample(wav_file, label):
  file = tf.io.read_file(recPath + wav_file + ".wav")
  audio, _ = tf.audio.decode_wav(file)
  audio = tf.squeeze(audio, axis=-1)
  audio = tf.cast(audio, tf.float32)
  spectogram = tf.signal.stft(
      audio, frame_length=frameLength, frame_step=frameStep, fft_length=fftLength
  )
  spectogram = tf.abs(spectogram)
  spectogram = tf.math.pow(spectogram, 0.5)

  means = tf.math.reduce_mean(spectogram, 1, keepdims=True)
  stddevs = tf.math.reduce_mean(spectogram, 1, keepdims=True)
  spectogram = (spectogram - means) / (stddevs + 1e-10)

  label = tf.strings.lower(label)
  label = tf.strings.unicode_split(label, input_encoding="UTF-8")
  label = charToNum(label)
  return spectogram, label

batchSize = 32
train_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_train["fileName"]), list(df_train["transcription"]))
)
train_dataset = (
    train_dataset.map(encode_sample, num_parallel_calls=tf.data.AUTOTUNE).padded_batch(batchSize).prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_val['fileName']), list(df_val['normalizedTranscription']))
)

validation_dataset = (
    validation_dataset.map(encode_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batchSize)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

fig = plt.figure(figsize=(8, 5))
for batch in train_dataset.take(1):
  spectogram = batch[0][0].numpy()
  spectogram = np.array([np.trim_zeros(x) for x in np.transpose(spectogram)])
  label = batch[1][0]

  label=tf.strings.reduce_join(numToChar(label)).numpy().decode("utf-8")
  ax = plt.subplot(2, 1, 1)
  ax.imshow(spectogram, vmax=1)
  ax.set_title(label)
  ax.axis("off")

  file = tf.io.read_file(recPath+list(df_train['fileName'])[0] + ".wav")
  audio, _ = tf.audio.decode_wav(file)
  audio = audio.numpy()
  ax = plt.subplot(2,1,2)
  plt.plot(audio)
  ax.set_title("signal wave")
  ax.set_xlim(0, len(audio))
  display.display(display.Audio(np.transpose(audio), rate = 16000))
plt.show()


def CTCloss(y_true, y_pred):
  batchLen = tf.cast(tf.shape(y_true)[0], dtype='int64')
  inputLen=  tf.cast(tf.shape(y_pred)[1], dtype="int64")
  labelLen=  tf.cast(tf.shape(y_true)[1], dtype="int64")


  inputLen = inputLen * tf.ones(shape=(batchLen, 1), dtype="int64")
  labelLen = labelLen * tf.ones(shape=(batchLen, 1), dtype="int64")

  loss = keras.backend.ctc_batch_cost(y_true, y_pred, inputLen, labelLen)
  return loss

def buildModel(input_dim, output_dim, rnn_layers=5, rnn_units=128):
  inputSpectogram = layers.Input((None, input_dim), name='input')
  x = layers.Reshape((-1 ,input_dim, 1), name="expand_dim")(inputSpectogram)
  x = layers.Conv2D(
      filters=32,
      kernel_size=[11, 41],
      strides=[2,2],
      padding="same",
      use_bias=False,
      name="conv_1"
  )(x)
  x = layers.BatchNormalization(name="conv_1_bn")(x)
  x = layers.ReLU(name="conv_1_relu")(x)

  x= layers.Conv2D(
      filters=32,
      kernel_size=[11, 21],
      strides=[1,2],
      padding="same",
      use_bias=False,
      name="conv_2",
  )(x)
  x = layers.BatchNormalization(name="conv_2_bn")(x)
  x = layers.ReLU(name="conv_2_relu")(x)

  x = layers.Reshape((-1,x.shape[-2] * x.shape[-1]))(x)

  for i in range(1, rnn_layers+1):
    recurrent = layers.GRU(
        units=rnn_units,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        return_sequences=True,
        reset_after=True,
        name=f'gru_{i}',
    )
    x = layers.Bidirectional(
        recurrent, name=f'bidirectional_{i}', merge_mode="concat"
    )(x)
    if i < rnn_layers:
      x = layers.Dropout(rate=0.5)(x)
  x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
  x = layers.ReLU(name="dense_1_relu")(x)
  x = layers.Dropout(rate=0.5)(x)

  output = layers.Dense(units=output_dim + 1, activation="softmax")(x)

  model = keras.Model(inputSpectogram, output, name="Sunny")
  opt = keras.optimizers.Adam(learning_rate=1e-4)

  model.compile(optimizer=opt, loss=CTCloss)

  return model


model = buildModel(
    input_dim = fftLength // 2+1,
    output_dim = charToNum.vocabulary_size(),
    rnn_units=512,
)
model.summary(line_length=110)

def decode_batch_prediction(pred):
    # Assuming pred shape is [batch_size, time_steps, num_classes]
    # Calculate the length of sequences - considering all timesteps
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    # Decoding the batch of predictions
    results, _ = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
    outputText = []

    for result in results[0]:
        # Converting the tensor to strings
        text = tf.strings.reduce_join(numToChar(result)).numpy().decode("utf-8")
        outputText.append(text)
    return outputText

class CallbackeEval(keras.callbacks.Callback):

  def __init__(self, dataset):
    super().__init__()
    self.dataset = dataset

  def on_epoch_end(self, epoch: int, logs=None):
    predictions = []
    targets = []
    for batch in self.dataset:
      X, y = batch
      batch_predictions = model.predict(X)
      batch_predictions = decode_batch_prediction(batch_predictions)
      predictions.extend(batch_predictions)
      for label in y:
        label = (
            tf.strings.reduce_join(numToChar(label)).numpy().decode("utf-8")
        )
        targets.append(label)
    wer_score = wer(targets, predictions)
    print("-"*100)
    print(f'Word Error Rate:{wer_score:.4f}')
    print("-"*100)
    for i in np.random.randint(0, len(predictions), 2):
        print(f"Target: {targets[i]}")
        print(f"Prediction: {predictions[i]}")

epochs = 25

validation_callback = CallbackeEval(validation_dataset)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[validation_callback],
)


predictions = []
targets = []

for batch in validation_dataset:
    X, y = batch
    batch_predictions = model.predict(X)
    batch_predictions = decode_batch_prediction(batch_predictions)
    predictions.extend(batch_predictions)
    for label in y:
        label = tf.strings.reduce_join(numToChar(label)).numpy().decode("utf-8")
        targets.append(label)
wer_score = wer(targets, predictions)
print("-"*100)
print(f'Word Error Rate:{wer_score:.4f}')
print("-"*100)
for i in np.random.randint(0, len(predictions), 2):
    print(f"Target: {targets[i]}")
    print(f"Prediction: {predictions[i]}")


model.save("SpeechToTextV1.2")
