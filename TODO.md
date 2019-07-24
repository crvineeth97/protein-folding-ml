### Improvements that can be made to the code

- [ ] Fix resnet_2d.py
- [ ] Fix validation part of the train function in main
- [ ] Fix calculation of RMSD and DRMSD
- [ ] Add multiple GPU support
- [ ] Improve the write_out function so that it is efficient
- [ ] Optimize input, output and target generation in Resnet
- [ ] Use float16 instead of float32?
- [x] Resnet_1D is not multiplying the channels by 2. Check why
- [x] Store the preprocessed files without the padding so that the size of the file is minimized
- [x] Write code to read the above files and then pad them according to batches where the maximum length protein determines the padding of the whole batch

### Parameters to tweak

- Kernel size of the 1D resnet basically determines how many surrounding amino acids to consider for the current amino acid. So a kernel size of 65 implies 32 AAs before and 32 after the current AA will be considered for the convolutions

### Interesting points to consider

- We have tested minibatches with only a single protein or with several proteins. Both work well. However, it is much easier to implement minibatches with only a single protein. -> From the contact map paper
- Because of missing residues in proteins, while we are training a model, an amino acid depends on another one further away from it. What can we do about this situation?
