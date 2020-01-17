import librosa
import pydub
import numpy as np

"""
Summary
------------------
    read
    audioGraph
    findnonSilent
    extractIntervals
    write
"""


"""
MP3 to numpy array
Parameters
file: string pathname to audio file
"""
def read(file):
    print ("Reading File...")
    a = pydub.AudioSegment.from_mp3(file)
    y = np.array(a.get_array_of_samples()).astype(float)
    if a.channels == 2:
        y = y.reshape((-1, 2)) #-1 means infer, 2 means 2-wide
    else:
        return y, a.frame_rate

"""
Visualize array of samples
Parameters
y: ndarray of audio samples
"""
def audioGraph(y):
    print ("Visualizing audio file")
    x = []
    import matplotlib.pyplot as plt
    print ("There are " + str(len(y)) + " samples")
    for i in range(0, len(y)//50):
        x.append(y[i*4])
    plt.plot(x)
    plt.ylabel('samples')
    plt.show()

"""
samples to decibels
"""
# # Convert to mono
# y_mono = librosa.core.to_mono(y)
# # Compute the MSE for the signal
# mse = librosa.feature.rms(y=y_mono,
#                   frame_length=2048,
#                   hop_length=512)**2
#
# dec = librosa.core.power_to_db(mse.squeeze(),ref=np.max,top_db=None)
# for i in range(0, len(dec)):
#     print(dec[i])

"""
Finds nonsilent intervals
Parameters
y: ndarray of audio samples
top_db: integer decibel threshold under which to consider silence
shortest_song: integer length in seconds of shortest nonsilent interval allowed
Returns: nonsilent intervals array shape(n, 2)
"""
def findnonSilent(y, top_db = 60, shortest_song = 10):
    print ("Splitting on Silence")
    indices = []
    intervals = librosa.effects.split(y, top_db = top_db)
    num_intervals = np.shape(intervals)[0]

    # removes false flags (ex. rests in songs, lots of quiet-ish noise from audience at ends of audio files)
    for i in range(num_intervals):
        if (intervals[i, 1] - intervals[i, 0]) < shortest_song*44100:
            indices.append(i)
    intervals = np.delete(intervals, indices, 0)

    return intervals

"""
Extracts interval of a sound file and writes it to a new file
Parameters
startTime: integer sample
endTime: integer sample
location: string pathname to sound file
outName: string name of output file
count: integer number iteration of extraction
"""
def extractIntervals(startTime, endTime, location, outName, count = 1):
    print ("Extracting interval " + str(count))
    from pydub import AudioSegment
    start = startTime/44100*1000
    end = endTime/44100*1000
    song = AudioSegment.from_mp3(location)
    extract = song[start:end]
    extract.export( outName + '.mp3', format="mp3")

"""
Write from numpy array to MP3
"""
# def write(f, sr, x, normalized=False):
#     channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
#     if normalized:  # normalized array - each item should be a float in [-1, 1)
#         y = np.int16(x * 2 ** 15)
#     else:
#         y = np.int16(x)
#     song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
#     song.export(f, format="mp3", bitrate="320k")
# usage:
# write('out2.mp3', sr, x)

"""
Example
"""
# Load some audio. Librosa audio load defaults to audioread, which is much slower
filepath = '/Users/Jonathan/Desktop/Informática/PYTHON/audioChopper/UMMGC_1952Fall_3.mp3'
out_filename = 'UMMGC_1952F'
y, sr = read(filepath)
# librosa load for reference:
# y, sr = librosa.load("/Users/Jonathan/Desktop/UMMGC_1952F_3-extract.mp3")

intervals = findnonSilent(y, top_db = 55, shortest_song = 45)
print (intervals)
print ('\a')
extract = input("Extract intervals? (y/n)")
if extract == "y":
    count = 1
    for i in intervals:
        extractIntervals(i[0], i[1], filepath, out_filename + "_" + str(count), count)
        count += 1
    print('Done.')
    print ('\a\a')
