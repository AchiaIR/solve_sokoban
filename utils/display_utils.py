"""
A video display configuration.
"""
import os
import time
import base64
import IPython
import subprocess
import matplotlib.pyplot as plt


def embed_mp4(filename):
  """Embeds an mp4 file"""
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

  return IPython.display.HTML(tag)


def DisplayVideo(video_filename):
  embed_mp4(video_filename)
  try:
    if os.name == 'nt':
      os.startfile(video_filename)  # Windows
    else:
      subprocess.call(['xdg-open', video_filename])  # Linux and macOS
  except Exception as e:
    print("Error opening the video:", e)


def PlotGraphs(losses, rewards):
  plt.figure(figsize=(20, 10))

  plt.subplot(1, 2, 1)
  plt.plot(losses)
  plt.title('Losses over time')
  plt.xlabel('Episodes')
  plt.ylabel('Loss')

  plt.subplot(1, 2, 2)
  plt.plot(rewards)
  plt.title('Rewards over time')
  plt.xlabel('Episodes')
  plt.ylabel('Reward')

  plt.show()

  # sleep for 5 seconds to allow user to view the graphs
  time.sleep(5)
  plt.close()