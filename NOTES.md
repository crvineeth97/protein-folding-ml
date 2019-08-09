# Notes

## TODO

- [ ] Fix resnet_2d.py
- [ ] Check calculation of RMSD and DRMSD
- [ ] Add multiple GPU support and distributed training
- [ ] Optimize input, output and target generation in Resnet
- [ ] Use radians instead of degrees, everywhere
- [ ] Combine omega and psi calculation
- [ ] Use LSTM layers after the resnet instead of fully connected layers
- [ ] Fix minibatch error, drop_last is set to true so find a way to ensure that the last set is taken into consideration

## Done

- [x] Improve the write_out function so that it is efficient
- [x] Fix validation part of the train function in main
- [x] Use float16 instead of float32? Avoid using float16 because compute capability of the 1080TI is 6.1 and float16 performance is much slower than float 32
- [x] Resnet_1D is not multiplying the channels by 2. Check why
- [x] Store the preprocessed files without the padding so that the size of the file is minimized
- [x] Write code to read the above files and then pad them according to batches where the maximum length protein determines the padding of the whole batch

## List of parameters to tweak for various models

### Common

- Learning rate
- Batch size to improve speed

### RNN

- Number of RNN units
- Number of RNN layers
- Bidirectional or Unidirectional

### Resnet

- Number of layers
- Kernel size of the 1D resnet basically determines how many surrounding amino acids to consider for the current amino acid. So a kernel size of 65 implies 32 AAs before and 32 after the current AA will be considered for the convolutions

## Statistics

### Input/Output

- Time taken to read the file and make dictionaries
  - training_30 | 4 mins 12.5 seconds
  - testing | 0.5 seconds
  - validation | 2 seconds

## Interesting points to consider

- We have tested minibatches with only a single protein or with several proteins. Both work well. However, it is much easier to implement minibatches with only a single protein. -> From the contact map paper
- Because of missing residues in proteins, while we are training a model, an amino acid depends on another one further away from it. What can we do about this situation?
