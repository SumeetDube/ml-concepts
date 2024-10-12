The interquartile range (IQR) is a statistical measure that represents the spread of the middle 50% of a dataset. It is calculated by subtracting the first quartile (Q1) from the third quartile (Q3), using the formula:

$$ \text{IQR} = Q3 - Q1 $$

### Understanding Quartiles

Quartiles divide a dataset into four equal parts:

- **Q1 (First Quartile)**: The value below which 25% of the data falls.
- **Q2 (Second Quartile)**: The median of the dataset, dividing it into two halves.
- **Q3 (Third Quartile)**: The value below which 75% of the data falls.

The IQR provides a measure of variability that is less affected by outliers compared to the overall range of the dataset, making it particularly useful for skewed distributions or datasets with extreme values[1][2][4].

### Calculation Steps

To calculate the IQR, follow these steps:

1. **Order the Data**: Arrange the dataset in ascending order.
2. **Determine Q1 and Q3**:
   - For Q1, find the median of the first half of the data (the values below the overall median).
   - For Q3, find the median of the second half of the data (the values above the overall median).
3. **Apply the IQR Formula**: Subtract Q1 from Q3.

### Example

For a dataset of numbers: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19

1. **Order the Data**: Already ordered.
2. **Find Q1**: The first half is 1, 3, 5, 7, 9. The median (Q1) is 5.
3. **Find Q3**: The second half is 11, 13, 15, 17, 19. The median (Q3) is 15.
4. **Calculate IQR**: 

   $$ \text{IQR} = 15 - 5 = 10 $$

### Applications

The IQR is commonly used in descriptive statistics to summarize the spread of data, particularly in boxplots, where it helps to visualize the distribution and identify potential outliers. Because it focuses on the central portion of the data, it provides a clearer picture of variability than the overall range, which can be skewed by extreme values[2][3][5].

Citations:
[1] https://www.statisticshowto.com/probability-and-statistics/interquartile-range/
[2] https://statisticsbyjim.com/basics/interquartile-range/
[3] https://www.scribbr.com/statistics/interquartile-range/
[4] https://www.geeksforgeeks.org/interquartile-range/
[5] https://byjus.com/maths/interquartile-range/