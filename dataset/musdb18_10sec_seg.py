import numpy as np
import musdb

mus = musdb.DB(root='/home/lego/NAS189/Database/musdb18')
hop_size = 1024
rate = 44100
window = rate * 10

for track in mus:
    print(track.name)
    activation = []
    track_num = len(track)
    ref = np.max(track.stems)
    for source in track.stems:
        # (sample_n, 2)
        num = len(np.where(abs(source[:,0]) <= ref/1000  )[0])
        print(num/len(source))
        
