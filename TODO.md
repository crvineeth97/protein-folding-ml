### Improvements that can be made to the code

- [ ] Fix resnet_2d.py
- [ ] Fix validation part of the train function in main
- [ ] Ensure that the padded values do not affect the network by giving the feature vector of the pads an all-zero vector
- [ ] Store the preprocessed files without the padding so that the size of the file is minimized
- [ ] Write code to read the above files and then pad them according to batches where the maximum length protein determines the padding of the whole batch
- [ ] The above can only be done if a dynamic Resnet (one with changing input size but with fixed channels) can be implemented. See if that is possible. If not, we can still use dynamic LSTMS
