# RandomSamplingGAN

## Installation

1. Run pip install -r requirements.txt

2. Make sure data is contained in root directory under a folder called "data"

## Running the code and analyzing results

1. Open train.ipynb
2. Run first cell for imports
3. Run second cell to train multiple RandomGANs
4. Run third cell to analyze and choose median result

## Census Data set expectations

- The census data set should be condensed to only contain unique individuals given the variable choices
- The first column of the census data set should contain the summed weights over all instances of that unique individual

- E.g. If there are 5 women from the Midwest with weights (1, 5, 10, and 2), they should all be condensed into a single row with weight 18

- The code has been changed to sample census data sets WITH replacement

## Survey data set expectations

- The Survey data set should contain individual entries and should NOT be condensed.
- In the same example above, the 5 women from the Midwest would remain as 5 individual rows
