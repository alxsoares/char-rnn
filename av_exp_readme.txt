H1 vs. GM pos in p10 neg not in p2:
- was getting exploting with lr 0.001, hence dropped to 0.0005 in final iteration
- I shuffled the inputs but thinking about how batching is done, not sure that was necessary
- Was getting substantial overfitting over 50 epochs with 512 LSTMs and 2 layers
- Adding dropout seemed to improve perf for negatives a bit, did not help perf for positives. Negative perf reached best somewhere in middle, then went up, then was going down again before it hit 20 epochs.
- I restarted training the negatives from a checkpoint that seemed to do well in the dropout case, for a few more epochs to see what happened. 
