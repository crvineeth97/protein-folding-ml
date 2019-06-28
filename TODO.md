### Improvements that can be made to the code

- [ ] Fix resnet_2d.py
- [ ] Fix validation part of the train function in main
- [ ] Add multiple GPU support
- [ ] Improve the write_out function so that it is efficient
- [ ] Ensure that the padded values do not affect the network by giving the feature vector of the pads an all-zero vector
- [ ] Store the preprocessed files without the padding so that the size of the file is minimized
- [ ] Write code to read the above files and then pad them according to batches where the maximum length protein determines the padding of the whole batch
- [ ] The above can only be done if a dynamic Resnet (one with changing input size but with fixed channels) can be implemented. See if that is possible. If not, we can still use dynamic LSTMS

### Parameters to tweak

- Kernel size of the 1D resnet basically determines how many surrounding amino acids to consider for the current amino acid. So a kernel size of 65 implies 32 AAs before and 32 after the current AA will be considered for the convolutions

### Interesting points to consider

- We have tested minibatches with only a single protein or with several proteins. Both work well. However, it is much easier to implement minibatches with only a single protein.
