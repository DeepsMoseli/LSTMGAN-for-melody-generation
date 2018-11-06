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
sample1:
<audio src="audio.mp3" controls preload></audio>
### Conclusion

### References

```markdown
Syntax highlighted code block
- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/DeepsMoseli/LSTMGAN-for-melody-generation/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
