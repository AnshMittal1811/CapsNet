from keras.utils import get_file
import json
import numpy as np
import random
random.seed(100)
from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 15

x = get_file('filename', 'path to file')

video = []

with open(x, 'r') as fin:
    # Append each line to the video
    video = [json.loads(l) for l in fin]


# Remove non-video posts if exist
video_with_video = [video for video in video if 'posts/captions/video' in video[0]]
video = [video for video in video if 'posts/captions' not in video[0]]
print(f'Found {len(video)} video.')


# Mapping videos to integers & vice versa
video_index = {video[0]: idx for idx, video in enumerate(video)}
index_video = {idx: video for video, idx in video_index.items()}

# Most linked videos
# from collections import Counter, OrderedDict

# def count_items(l):
#     """Return ordered dictionary of counts of objects in `l`"""
    
#     # Create a counter object
#     counts = Counter(l)
    
#     # Sort by highest count first and place in ordered dictionary
#     counts = sorted(counts.items(), key = lambda x: x[1], reverse = True)
#     counts = OrderedDict(counts)
    
#     return counts


unique_vidlinks = list(chain(*[list(set(video[2])) for video in video]))

vidlink_counts = count_items(unique_vidlinks)

# remove capitalization if present to normlize
vidlinks = [link.lower() for link in unique_vidlinks]

# Removing most popular and linked videos (TF-IDF)
to_remove = high_freq_words

for t in to_remove:
    vidlinks.remove(t)
    _ = vidlink_counts.pop(t)
    
# if no. of videolinking increases, limit according to number of videos app has (here 5)
links = [t[0] for t in vidlink_counts.items() if t[1] >= 5]

# Clean more data to make it suitable for later purposes

# Videolinks to index
link_index = {link: idx for idx, link in enumerate(links)}
index_link = {idx: link for link, idx in link_index.items()}

# Supervised ML task
## train set building
pairs = []


# Iterate through each video
for video in video:
    # Iterate through the links in the video
    pairs.extend((video_index[video[0]], link_index[link.lower()]) for link in video[2] if link.lower() in links)
    
# length of the pairs, videos, cascaded video linking

# create the negative examples by randomly sampling from the links
pairs_set = set(pairs)


# Generator for training samples
def generate_batch(pairs, n_positive = 50, negative_ratio = 1.0, classification = False):
    """Generate batches of samples for training"""
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    
    # Adjust label based on task
    if classification:
        neg_label = 0
    else:
        neg_label = -1
    
    # This creates a generator
    while True:
        # randomly choose positive examples
        for idx, (vid_id, link_id) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (vid_id, link_id, 1)

        # Increment idx by 1
        idx += 1
        
        # Add negative examples until reach batch size
        while idx < batch_size:
            
            # random selection
            random_video = random.randrange(len(video))
            random_link = random.randrange(len(links))
            
            # Check to make sure this is not a positive example
            if (random_video, random_link) not in pairs_set:
                
                # Add to batch and increment index
                batch[idx, :] = (random_video, random_link, neg_label)
                idx += 1
                
        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'video': batch[:, 0], 'link': batch[:, 1]}, batch[:, 2]

        
x, y = next(generate_batch(pairs, n_positive = 2, negative_ratio = 2))

# Neural Network Embedding model
def video_embedding_model(embedding_size = 50, classification = False):
    """Model to embed video and vidlinks using the functional API.
       Trained to discern if a link is present in a article"""
    
    # Both inputs are 1-dimensional
    video = Input(name = 'video', shape = [1])
    link = Input(name = 'link', shape = [1])
    
    # Embedding the video (shape will be (None, 1, 50))
    video_embedding = Embedding(name = 'video_embedding',
                               input_dim = len(video_index),
                               output_dim = embedding_size)(video)
    
    # Embedding the link (shape will be (None, 1, 50))
    link_embedding = Embedding(name = 'link_embedding',
                               input_dim = len(link_index),
                               output_dim = embedding_size)(link)
    
    # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
    merged = Dot(name = 'dot_product', normalize = True, axes = 2)([video_embedding, link_embedding])
    
    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape = [1])(merged)
    
    # If classifcation, add extra layer and loss function is binary cross entropy
    if classification:
        merged = Dense(1, activation = 'sigmoid')(merged)
        model = Model(inputs = [video, link], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Otherwise loss function is mean squared error
    else:
        model = Model(inputs = [video, link], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'mse')
    
    return model

# Instantiate model and show parameters
model = video_embedding_model()
model.summary()

n_positive = 1024

gen = generate_batch(pairs, n_positive, negative_ratio = 2)

# Train
h = model.fit_generator(gen, epochs = 15, 
                        steps_per_epoch = len(pairs) // n_positive,
                        verbose = 2)



def find_similar(name, weights, index_name = 'video', n = 10, least = False, return_dist = False, plot = False):
    """Find n most similar items (or least) to name based on embeddings. Option to also plot the results"""
    
    # Select index and reverse index
    if index_name == 'video':
        index = video_index
        rindex = index_video
    elif index_name == 'page':
        index = link_index
        rindex = index_link
    
    # Check to make sure `name` is in index
    try:
        # Calculate dot product between video and all others
        dists = np.dot(weights, weights[index[name]])
    except KeyError:
        print(f'{name} Not Found.')
        return
    
    # Sort distance indexes from smallest to largest
    sorted_dists = np.argsort(dists)
    
    # Plot results if specified
    if plot:
        
        # Find furthest and closest items
        furthest = sorted_dists[:(n // 2)]
        closest = sorted_dists[-n-1: len(dists) - 1]
        items = [rindex[c] for c in furthest]
        items.extend(rindex[c] for c in closest)
        
        # Find furthest and closets distances
        distances = [dists[c] for c in furthest]
        distances.extend(dists[c] for c in closest)
        
        colors = ['r' for _ in range(n //2)]
        colors.extend('g' for _ in range(n))
        
        data = pd.DataFrame({'distance': distances}, index = items)
        
        # Horizontal bar chart
        data['distance'].plot.barh(color = colors, figsize = (10, 8),
                                   edgecolor = 'k', linewidth = 2)
        plt.xlabel('Cosine Similarity');
        plt.axvline(x = 0, color = 'k');
        
        # Formatting for italicized title
        name_str = f'{index_name.capitalize()}s Most and Least Similar to'
        for word in name.split():
            # Title uses latex for italize
            name_str += ' $\it{' + word + '}$'
        plt.title(name_str, x = 0.2, size = 28, y = 1.05)
        
        return None
    
    # If specified, find the least similar
    if least:
        # Take the first n from sorted distances
        closest = sorted_dists[:n]
         
        print(f'{index_name.capitalize()}s furthest from {name}.\n')
        
    # Otherwise find the most similar
    else:
        # Take the last n sorted distances
        closest = sorted_dists[-n:]
        
        # Need distances later on
        if return_dist:
            return dists, closest
        
        
        print(f'{index_name.capitalize()}s closest to {name}.\n')
        
    # Need distances later on
    if return_dist:
        return dists, closest
    
    
    # Print formatting
    max_width = max([len(rindex[c]) for c in closest])
    
    # Print the most similar and distances
    for c in reversed(closest):
        print(f'{index_name.capitalize()}: {rindex[c]:{max_width + 2}} Similarity: {dists[c]:.{2}}')

        
# Call find similar
# THese embeddings can also be sent to 2 phas model discussed earlier