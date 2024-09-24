## Assignment 2 
### Question-1: Travelling salesman problem
The following is the result obtained while setting number of target equals 15:

| Method                | Average Return       |
| --------------------- | -------------------- |
| DP (Value Iteration)  | -281.1385502020905   |
| MC (First-Visit)      | -53410.37085702923   | 
| MC (Every-Visit)      | -53001.20186449172   | 

Clearly, DP (Value Iteration) is doing better in this case compared to Monte-Carlo Approach.

### Question-2: Sokoban Puzzle

![Screenshot 2024-09-23 230537](https://github.com/user-attachments/assets/8ce446fd-631a-4a6f-bfd3-60ab16744a35)

Optimal Policy after Value Iteration:

State & Optimal Action  
(0, 0)          UP              
(0, 1)          DOWN           
(0, 2)          DOWN           
(0, 3)          UP             
(0, 4)          UP             
(0, 5)          UP             
(1, 0)          RIGHT          
(1, 1)          DOWN           
(1, 2)          DOWN           
(1, 3)          LEFT           
(1, 4)          UP             
(1, 5)          UP             
(2, 0)          RIGHT          
(2, 1)          DOWN           
(2, 2)          DOWN           
(2, 3)          DOWN           
(2, 4)          DOWN           
(2, 5)          UP             
(3, 0)          RIGHT          
(3, 1)          DOWN           
(3, 2)          DOWN           
(3, 3)          DOWN           
(3, 4)          DOWN           
(3, 5)          LEFT           
(4, 0)          RIGHT          
(4, 1)          RIGHT          
(4, 2)          RIGHT          
(4, 3)          RIGHT          
(4, 4)          LEFT           
(4, 5)          LEFT           
(5, 0)          RIGHT          
(5, 1)          RIGHT          
(5, 2)          UP             
(5, 3)          UP             
(5, 4)          UP             
(5, 5)          UP             
(6, 0)          UP             
(6, 1)          UP             
(6, 2)          UP             
(6, 3)          UP             
(6, 4)          UP             
(6, 5)          UP             
A box is stuck! Episode complete.


#### Optimal Policy after Monte Carlo Control:

##### State & Optimal Action 

(0, 0)          LEFT           
(0, 1)          RIGHT          
(0, 2)          DOWN           
(0, 3)          RIGHT          
(0, 4)          RIGHT          
(0, 5)          UP             
(1, 0)          UP             
(1, 1)          DOWN           
(1, 2)          LEFT           
(1, 3)          RIGHT          
(1, 4)          UP             
(1, 5)          RIGHT          
(2, 0)          LEFT           
(2, 1)          UP             
(2, 2)          DOWN           
(2, 3)          RIGHT          
(2, 4)          LEFT           
(2, 5)          RIGHT          
(3, 0)          RIGHT          
(3, 1)          DOWN           
(3, 2)          RIGHT          
(3, 3)          DOWN           
(3, 4)          DOWN           
(3, 5)          LEFT           
(4, 0)          DOWN           
(4, 1)          RIGHT          
(4, 2)          RIGHT          
(4, 3)          UP             
(4, 4)          UP             
(4, 5)          UP             
(5, 0)          LEFT           
(5, 1)          RIGHT          
(5, 2)          UP             
(5, 3)          UP             
(5, 4)          LEFT           
(5, 5)          LEFT           
(6, 0)          DOWN           
(6, 1)          UP             
(6, 2)          RIGHT          
(6, 3)          UP             
(6, 4)          DOWN           
(6, 5)          DOWN    
