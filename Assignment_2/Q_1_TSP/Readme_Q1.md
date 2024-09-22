

### Dynamic Programming (DP)

- **Output**:
  - **Value Function**: 
    ```
    [-29.53, -51.93, -33.57, -43.02, -36.25, -38.72, -48.49, -43.60, -30.57, -33.41, 
     -39.63, -41.36, -44.28, -35.90, -37.11]
    ```
  - **Optimal Policy**: 
    ```
    [17, 21, 46, 43, 9, 34, 5, 5, 18, 5, 6, 27, 12, 24, 46, 34, 34, 3, 5, 6, 
     18, 18, 29, 41, 34, 36, 34, 5, 34, 34, 42, 17, 19, 1, 12, 5, 40, 22, 
     3, 16, 43, 41, 13, 5, 35, 35, 43, 9, 35, 44]
    ```
- **Performance**: The DP method demonstrated stable and relatively low cost values, indicating a well-optimized route through the target points.

### Monte Carlo (MC)

#### First Visit MC

- **Output**:
  - **Value Function**: 
    ```
    [-98312.95, -94703.87, -103069.42, ..., -97590.69, -99317.56, -107795.91]
    ```
  - **Optimal Policy**: 
    ```
    [25, 0, 48, 27, 47, 41, 38, 20, 6, 20, 20, 3, 9, 35, 13, 30, 26, 32, 
     45, 45, 48, 5, 25, 4, 45, 41, 34, 19, 33, 46, 1, 13, 5, 41, 32, 24, 
     0, 11, 43, 11, 26, 0, 9, 39, 49, 48, 30, 42, 4, 22]
    ```
- **Performance**: The First Visit MC method produced significantly higher cost values, indicating less efficient routing compared to DP.

#### Every Visit MC

- **Output**:
  - **Value Function**: 
    ```
    [-101886.60, -94280.02, -95195.85, ..., -97842.96, -100307.14, -104094.39]
    ```
  - **Optimal Policy**: 
    ```
    [31, 32, 19, 18, 35, 31, 35, 48, 16, 11, 33, 8, 35, 34, 15, 47, 31, 
     32, 40, 25, 46, 12, 6, 43, 42, 8, 29, 0, 23, 48, 31, 45, 30, 7, 
     49, 7, 32, 31, 32, 38, 27, 29, 22, 34, 2, 5, 1, 42, 27, 1]
    ```
- **Performance**: The Every Visit MC method also yielded high cost values, similar to the First Visit approach, reinforcing the observation of inefficient routing.

## Comparison Summary

| Method                  | Value Function Cost          | Optimal Policy                                                                                   |
|-------------------------|------------------------------|-------------------------------------------------------------------------------------------------|
| **Dynamic Programming** | Low and stable costs         | [17, 21, 46, ..., 35, 44]                                                                      |
| **Monte Carlo (First Visit)** | High costs (-98312.95)      | [25, 0, 48, ..., 39, 49]                                                                        |
| **Monte Carlo (Every Visit)** | Highest costs (-101886.60)  | [31, 32, 19, ..., 1, 42, 27, 1]                                                               |

## Conclusion

The **Dynamic Programming** method outperformed both **Monte Carlo** approaches in solving the Traveling Salesman Problem. DP provided a more efficient and optimal route with significantly lower cost values, while the Monte Carlo methods exhibited higher costs and less stable routing, indicating inefficiencies in exploration and value estimation.

The combination of a complete model, systematic exploration, and deterministic updates makes DP a more effective method for solving TSP compared to Monte Carlo methods, particularly in environments where the state space can be fully evaluated. Improving the exploration strategies or increasing the number of episodes in MC methods could enhance their performance, but they inherently rely on sampling, which can introduce variability and inefficiency.