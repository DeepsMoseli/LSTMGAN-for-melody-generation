## Comparison of encoder-decoder LSTM and WGAN LSTM for Music Generation


### Summary
This work implements and compares adversarial and non-adversarial training of LSTM music composers on MIDI data. The resulting generated music samples are evaluated by voluntee human listeners, and their preferences are recorded to test whether adversarial training produces music samples that are more pleasing to listen to

### DATA
The [Lahk Midi Dataset](https://colinraffel.com/projects/lmd/) is used for both training configurations. Note extraction and preprocessing were done with the help of [Dan Shiebler](https://github.com/dshieble/Musical_Matrices/blob/master/midi_manipulation.py)'s repo. Only Pitch,Velocity and the delta time extracted and used. 
### Encoder-decoder LSTM
A 3 layered bidirectional encoder decoder LSTM is implemented as shown below.

### WGAN LSTM

### Evaluation

### Generated Samples

<audio src="https://github.com/DeepsMoseli/LSTMGAN-for-melody-generation/gan1.mid" controls preload></audio>

### Conclusion

### References

```markdown

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
