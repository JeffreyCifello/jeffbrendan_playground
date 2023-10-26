Made an enviroment using Open AI gymnasium and then trained an agent on it using Stable Baseline 3 (these are the only new packages I installed besides numpy).

The enviroment is essentially the skeleton we built last time - we have a lil guy who needs to get to a goal. I added a way to generate random barriers.

I am currently working on how to actually give the AI agent information since I realized it can't actually see anything. Im' either going to feed it Euclidean distance
or maybe just feed it the raw tuples which represent the map. Excited to see how it develops.